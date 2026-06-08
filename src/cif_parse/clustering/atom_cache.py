"""Process-level LRU cache reading AtomArrays from parse ``atoms/*.pkl``."""

from __future__ import annotations

import logging
import json
import pickle
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from cif_parse.pipeline import infer_case_id

LOGGER = logging.getLogger(__name__)

_MAX_CACHE_SIZE = 128


@lru_cache(maxsize=_MAX_CACHE_SIZE)
def _load_pkl_bytes(path: str) -> bytes | None:
    try:
        return zlib.decompress(Path(path).read_bytes())
    except Exception:
        LOGGER.warning("[fallback] Failed to read/decompress pkl: %s", path)
        return None


def resolve_cases_root(prep_dir: str | Path) -> Path:
    """Return the single parsed cases parent directory from prep manifest."""
    import json as _json
    manifest = Path(prep_dir) / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Prep manifest not found at {manifest}. "
            f"Run `cif-parse-cluster prep` first."
        )
    data = _json.loads(manifest.read_text(encoding="utf-8"))
    root_str = data.get("parsed_input") or data.get("parsed_inputs")
    if not root_str:
        raise ValueError(
            f"Prep manifest at {manifest} has no 'parsed_input'. Rebuild prep."
        )
    if isinstance(root_str, list):
        root_str = root_str[0]
    root = Path(root_str)
    if not root.is_dir():
        raise FileNotFoundError(f"Parsed cases root not found: {root}")
    return root


def load_source_case_dir_map(prep_dir: str | Path) -> dict[str, str]:
    """Load source_path -> source_case_dir from prep Parquet tables."""

    map_path = Path(prep_dir) / "source_case_dir_map.json"
    if map_path.exists():
        try:
            data = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    str(source_path): str(source_case_dir)
                    for source_path, source_case_dir in data.items()
                    if source_path and source_case_dir
                }
        except Exception:
            LOGGER.debug("Failed to load source_case_dir map from %s", map_path, exc_info=True)

    def _read_table(table_name: str, mapping: dict[str, str]) -> None:
        import pyarrow.parquet as pq

        path = Path(prep_dir) / f"{table_name}.parquet"
        if not path.exists():
            return
        pf = pq.ParquetFile(path)
        try:
            schema_names = set(pf.schema_arrow.names)
            if "source_path" not in schema_names or "source_case_dir" not in schema_names:
                return
            for rg_idx in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg_idx, columns=["source_path", "source_case_dir"])
                sources = tbl.column("source_path").to_pylist()
                case_dirs = tbl.column("source_case_dir").to_pylist()
                for source_path, source_case_dir in zip(sources, case_dirs):
                    source_path = str(source_path or "")
                    source_case_dir = str(source_case_dir or "")
                    if source_path and source_case_dir:
                        mapping.setdefault(source_path, source_case_dir)
        finally:
            pf.close()

    try:
        mapping: dict[str, str] = {}
        _read_table("entry_quality", mapping)
        if mapping:
            return mapping
        for table_name in ("monomers", "dimers", "multimers", "antibody_complexes", "tcr_complexes"):
            _read_table(table_name, mapping)
        return mapping
    except Exception:
        LOGGER.debug("Failed to load source_case_dir map from prep", exc_info=True)
        return {}


def _candidate_atoms_dirs(root: Path, case_id: str) -> list[Path]:
    case_path = Path(case_id)
    candidates: list[Path] = []
    if case_path.is_absolute():
        candidates.append(case_path / "atoms")
    else:
        candidates.append(root / case_path / "atoms")
        candidates.append(root / "cases" / case_path / "atoms")
    # Keep order stable while avoiding duplicate filesystem checks.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        key = path.resolve(strict=False)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _single_assembly_cache_path(atoms_dir: Path) -> Path | None:
    try:
        candidates = sorted(
            path
            for path in atoms_dir.glob("*.pkl")
            if path.name != "_none.pkl" and path.is_file()
        )
    except OSError:
        return None
    return candidates[0] if len(candidates) == 1 else None


class PklAtomReader:
    """Read AtomArrays from parse ``atoms/{assembly_id}.pkl`` caches."""

    def __init__(
        self,
        cases_root: str | Path,
        source_case_dir_map: dict[str, str] | None = None,
    ) -> None:
        self._root = Path(cases_root)
        self._source_case_dir_map = dict(source_case_dir_map or {})
        self._aa_cache: dict[tuple[str, str | None], Any] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def load_chain(
        self,
        source_path: str,
        label_asym_id: str,
        assembly_id: str | None = None,
        *,
        filter_hetero: bool = False,
    ) -> Any | None:
        full = self.load_assembly(source_path, assembly_id)
        if full is None or len(full) == 0:
            return None
        mask = full.chain_id == label_asym_id
        if filter_hetero and hasattr(full, "hetero"):
            mask &= ~full.hetero
        chain = full[mask]
        return chain if chain.array_length() > 0 else None

    def load_assembly(
        self,
        source_path: str,
        assembly_id: str | None = None,
    ) -> Any | None:
        case_id = self._source_case_dir_map.get(str(source_path)) or infer_case_id(source_path)
        atoms_dirs = _candidate_atoms_dirs(self._root, case_id)
        atoms_dir = next((candidate for candidate in atoms_dirs if candidate.is_dir()), atoms_dirs[0])
        cache_key = (str(atoms_dir.resolve(strict=False)), assembly_id)
        if cache_key not in self._aa_cache:
            pkl_path = atoms_dir / f"{assembly_id or '_none'}.pkl"
            if assembly_id is None and not pkl_path.exists():
                fallback_path = _single_assembly_cache_path(atoms_dir)
                if fallback_path is not None:
                    pkl_path = fallback_path
            raw = _load_pkl_bytes(str(pkl_path))
            if raw is None:
                return None
            try:
                self._aa_cache[cache_key] = pickle.loads(raw)
            except Exception:
                LOGGER.debug("[fallback] Failed to unpickle %s", pkl_path, exc_info=True)
                return None
        return self._aa_cache[cache_key]

    def load_chains(
        self,
        source_path: str,
        chain_specs: list[tuple[str, int | None]],
        assembly_id: str | None = None,
        *,
        filter_hetero: bool = False,
    ) -> Any | None:
        import biotite.structure as _struc
        full = self.load_assembly(source_path, assembly_id)
        if full is None:
            return None
        arrays = []
        for lbl, _sym in chain_specs:
            mask = full.chain_id == lbl
            if filter_hetero and hasattr(full, "hetero"):
                mask &= ~full.hetero
            if _sym is not None and hasattr(full, "sym_id"):
                sym_mask = mask & (full.sym_id == _sym)
                if bool(sym_mask.any()):
                    mask = sym_mask
            chain = full[mask]
            if chain.array_length() == 0:
                return None
            arrays.append(chain)
        return _struc.concatenate(arrays) if len(arrays) > 1 else arrays[0]
