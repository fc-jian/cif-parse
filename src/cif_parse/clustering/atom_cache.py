"""Process-level LRU cache reading AtomArrays from parse ``atoms/*.pkl``.

Zero directory enumeration: each worker computes the pkl path directly from
``{cases_root}/{case_id}/atoms/{assembly_id}.pkl`` using ``infer_case_id``.
"""

from __future__ import annotations

import logging
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


class PklAtomReader:
    """Read AtomArrays from ``{cases_root}/{case_id}/atoms/{asm}.pkl``.

    Zero directory scan — paths are computed directly from *cases_root*
    and the monomer's source_path via ``infer_case_id``.
    """

    def __init__(self, cases_root: str | Path) -> None:
        self._root = Path(cases_root)
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
        filter_hetero: bool = True,
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
        atoms_dir = self._root / infer_case_id(source_path) / "atoms"
        cache_key = (str(atoms_dir), assembly_id)
        if cache_key not in self._aa_cache:
            pkl_path = atoms_dir / f"{assembly_id or '_none'}.pkl"
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
        filter_hetero: bool = True,
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
