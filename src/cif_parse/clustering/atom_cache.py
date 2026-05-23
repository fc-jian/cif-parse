"""Process-level LRU cache reading AtomArrays from parse ``atoms/*.pkl``.

Eliminates prep Phase 2 (cif_coords binary blob generation).  Each worker
process maintains its own path→AtomArray LRU, avoiding lock contention.
Full AtomArrays are loaded once per ``(atoms_dir, assembly_id)`` and chains
are extracted by mask.
"""

from __future__ import annotations

import logging
import pickle
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Per-process cache: path → decompressed pickle object
_MAX_CACHE_SIZE = 128


@lru_cache(maxsize=_MAX_CACHE_SIZE)
def _load_pkl_bytes(path: str) -> bytes | None:
    """Read and decompress a pkl file.  Cached per unique path."""
    try:
        return zlib.decompress(Path(path).read_bytes())
    except Exception:
        return None


class PklAtomReader:
    """Read AtomArrays from parse ``atoms/*.pkl`` with per-process LRU.

    Parameters
    ----------
    source_to_atoms:
        Mapping from source_path → atoms directory path.
        Built once at startup by scanning parsed case directories.
    """

    def __init__(self, source_to_atoms: dict[str, str | Path]) -> None:
        self._map = {str(k): Path(v) for k, v in source_to_atoms.items()}
        # Per-process atom array cache: (atoms_dir, asm_id) → AtomArray
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
        """Return a single-chain AtomArray, or None."""
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
        """Return the full assembly AtomArray, or None."""
        atoms_dir = self._map.get(source_path)
        if atoms_dir is None:
            return None
        cache_key = (str(atoms_dir), assembly_id)
        if cache_key not in self._aa_cache:
            pkl_path = atoms_dir / f"{assembly_id or '_none'}.pkl"
            raw = _load_pkl_bytes(str(pkl_path))
            if raw is None:
                return None
            try:
                self._aa_cache[cache_key] = pickle.loads(raw)
            except Exception:
                LOGGER.debug("Failed to unpickle %s", pkl_path, exc_info=True)
                return None
        return self._aa_cache[cache_key]

    def load_chains(
        self,
        source_path: str,
        chain_specs: list[tuple[str, int | None]],  # [(label_asym_id, sym_id), ...]
        assembly_id: str | None = None,
        *,
        filter_hetero: bool = True,
    ) -> Any | None:
        """Load and concatenate multiple chain AtomArrays. Returns None on failure."""
        import biotite.structure as _struc
        full = self.load_assembly(source_path, assembly_id)
        if full is None:
            return None
        arrays = []
        for lbl, _sym in chain_specs:
            mask = full.chain_id == lbl
            if filter_hetero and hasattr(full, "hetero"):
                mask &= ~full.hetero
            chain = full[mask]
            if chain.array_length() == 0:
                return None
            arrays.append(chain)
        return _struc.concatenate(arrays) if len(arrays) > 1 else arrays[0]

    def preload(self, source_path: str, assembly_id: str | None = None) -> None:
        """Pre-warm the cache for later use."""
        self.load_assembly(source_path, assembly_id)


def build_source_to_atoms_map(
    case_dirs: list[str | Path],
) -> dict[str, str]:
    """Scan parse output case directories and build source_path → atoms_dir map."""
    mapping: dict[str, str] = {}
    for case_dir in case_dirs:
        atoms = Path(case_dir) / "atoms"
        if not atoms.is_dir():
            continue
        # Find result bundles to extract source_path
        for bundle_path in sorted(Path(case_dir).glob("result*.json.gz")):
            try:
                import gzip as _gz, json as _json
                raw = _gz.decompress(bundle_path.read_bytes())
                summary = _json.loads(raw).get("structure_summary", {})
                sp = summary.get("source_path", "")
                if sp and sp not in mapping:
                    mapping[sp] = str(atoms)
            except Exception:
                pass
    return mapping
