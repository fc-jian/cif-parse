"""Pre-processing for clustering at any scale.

Produces a self-contained directory that replaces thousands of individual
``result.json.gz`` reads with columnar Parquet scans and replaces re-parsing
the same mmCIF files with a single binary index.  The output directory always
contains exactly 8 files regardless of case count.

Layout
------
::

    clustering_prep/
    ├── monomers.parquet              # pre-parsed polymer chain rows
    ├── dimers.parquet                # pre-parsed dimer interface rows
    ├── multimers.parquet             # pre-parsed tight multimer rows
    ├── antibody_complexes.parquet    # pre-parsed antibody-antigen complex rows
    ├── tcr_complexes.parquet         # pre-parsed TCR-pMHC complex rows
    ├── cif_coords.bin               # concatenated pickled AtomArray blobs
    ├── cif_coords.idx               # 76-byte fixed-width index
    └── prep_meta.json               # content hashes and statistics

Builder
-------
``cif-parse-cluster prep`` builds (or refreshes) the directory.  Hash-based
incremental updates skip unchanged case directories.  Workers can be
configured with ``--prep-jobs``.

Consumers
---------
The ``collect_*`` functions in each clustering module use PyArrow to scan the
Parquet files in a streaming fashion.  The ``extract_*`` functions use
memory-mapped I/O to fetch cached AtomArray objects via the binary index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import struct
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

_IDX_ENTRY_STRUCT = struct.Struct("64s Q I")  # hash(64 B hex ASCII), offset(8 B), len(4 B)
_IDX_ENTRY_SIZE = _IDX_ENTRY_STRUCT.size  # 76 bytes

# Parquet output file names
PARQUET_TABLES = [
    "monomers",
    "dimers",
    "multimers",
    "antibody_complexes",
    "tcr_complexes",
]

# ── hash helpers ─────────────────────────────────────────────────────────────


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_case_dir(case_dir: Path) -> str:
    hasher = hashlib.sha256()
    for child_path in sorted(case_dir.iterdir()):
        if child_path.is_file() and child_path.name.endswith((".json", ".json.gz", ".gz")):
            hasher.update(child_path.name.encode())
            hasher.update(_hash_file(child_path).encode())
    return hasher.hexdigest()


def _hash_source_mtime(source_path: str) -> str:
    p = Path(source_path)
    if not p.exists():
        return "missing"
    stat = p.stat()
    return hashlib.sha256(f"{p.resolve()}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()


# ── json extraction helpers (move heavy parsing into prep once) ──────────────


def _normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def _is_polymer_chain(chain_payload: dict[str, Any]) -> bool:
    from cif_parse.constants import POLYMER_CHAIN_TYPES
    return str(chain_payload.get("chain_type", "")) in POLYMER_CHAIN_TYPES


def _classify_polymer_class(chain_payload: dict[str, Any]) -> str | None:
    from cif_parse.constants import PROTEIN_CHAIN_TYPES
    chain_type = str(chain_payload.get("chain_type", ""))
    polymer_type = str(chain_payload.get("polymer_type", "") or "").lower()
    if chain_type in PROTEIN_CHAIN_TYPES:
        return "protein"
    if chain_type == "DNA chain":
        return "dna"
    if chain_type == "RNA chain":
        return "rna"
    if chain_type == "other nucleic acid chain":
        if "deoxyribo" in polymer_type:
            return "dna"
        if "ribo" in polymer_type:
            return "rna"
        return "other_nucleic_acid"
    if chain_type == "other polymer chain":
        if "deoxyribo" in polymer_type:
            return "dna"
        if "ribo" in polymer_type:
            return "rna"
        if "peptide" in polymer_type:
            return "protein"
    return None


def _extract_bundle_rows(bundle: dict[str, Any], case_id: str) -> dict[str, list[dict[str, Any]]]:
    """Parse one case bundle dict and return rows for each Parquet table.

    Returns ``{"monomers": [...], "dimers": [...], ...}``.
    """
    summary = bundle.get("structure_summary", {})
    source_path = str(summary.get("source_path", "") or "")
    pdb_id = str(summary.get("pdb_id", "") or "")
    assembly_ids = [str(a) for a in summary.get("assembly_ids", []) if str(a)]
    if not assembly_ids:
        assembly_ids = [""]
    assembly_id = assembly_ids[0]
    assembly_mode = str(summary.get("assembly_mode", "") or "")

    rows: dict[str, list[dict[str, Any]]] = {t: [] for t in PARQUET_TABLES}

    # ── monomers ─────────────────────────────────────────────────────────
    chain_inventory = bundle.get("chain_inventory", [])
    if isinstance(chain_inventory, list):
        for chain in chain_inventory:
            if not isinstance(chain, dict) or not _is_polymer_chain(chain):
                continue
            polymer_class = _classify_polymer_class(chain)
            if polymer_class is None:
                continue
            seq = _normalize_sequence(chain.get("sequence"))
            if not seq:
                continue
            lbl = str(chain.get("label_asym_id", "") or "")
            pid = str(chain.get("pdb_id", "") or pdb_id)
            if not pid or not lbl:
                continue
            rows["monomers"].append({
                "pdb_id": pid,
                "label_asym_id": lbl,
                "auth_asym_id": chain.get("auth_asym_id"),
                "entity_id": str(chain.get("entity_id", "") or ""),
                "entity_type": str(chain.get("entity_type", "") or ""),
                "entity_description": chain.get("entity_description"),
                "polymer_type": chain.get("polymer_type"),
                "polymer_class": polymer_class,
                "chain_type": str(chain.get("chain_type", "") or ""),
                "subtype": chain.get("subtype"),
                "sequence": seq,
                "length": int(chain.get("length", len(seq)) or len(seq)),
                "residue_count": int(chain.get("residue_count", chain.get("length", len(seq))) or len(seq)),
                "atom_count": int(chain.get("atom_count", 0) or 0),
                "source_path": source_path,
                "source_case_dir": case_id,
                "parsed_coordinate_segments": json.dumps(chain.get("parsed_coordinate_segments", []) or [], ensure_ascii=False),
                "unresolved_sequence_segments": json.dumps(chain.get("unresolved_sequence_segments", []) or [], ensure_ascii=False),
                "special_residue_details": json.dumps(chain.get("special_residue_details", []) or [], ensure_ascii=False),
                "special_component_details": json.dumps(chain.get("special_component_details", []) or [], ensure_ascii=False),
                "assembly_ids": json.dumps(assembly_ids, ensure_ascii=False),
            })

    # ── dimers ───────────────────────────────────────────────────────────
    dimers = bundle.get("dimer_interfaces", [])
    if isinstance(dimers, list):
        for index, dimer in enumerate(dimers, start=1):
            if not isinstance(dimer, dict):
                continue
            rows["dimers"].append({
                "pdb_id": pdb_id,
                "source_path": source_path,
                "assembly_id": str(dimer.get("assembly_id", assembly_id) or assembly_id),
                "assembly_mode": str(dimer.get("assembly_mode", assembly_mode) or assembly_mode),
                "label_asym_id_1": str(dimer.get("label_asym_id_1", "") or ""),
                "auth_asym_id_1": dimer.get("auth_asym_id_1"),
                "chain_type_1": str(dimer.get("chain_type_1", "") or ""),
                "sym_id_1": _optional_int(dimer.get("sym_id_1")),
                "label_asym_id_2": str(dimer.get("label_asym_id_2", "") or ""),
                "auth_asym_id_2": dimer.get("auth_asym_id_2"),
                "chain_type_2": str(dimer.get("chain_type_2", "") or ""),
                "sym_id_2": _optional_int(dimer.get("sym_id_2")),
                "interface_label": str(dimer.get("interface_label", "") or ""),
                "is_same_entity": bool(dimer.get("is_same_entity", False)),
                "contains_antibody_unit": bool(dimer.get("contains_antibody_unit", False)),
                "contains_tcr_pmhc_unit": bool(dimer.get("contains_tcr_pmhc_unit", False)),
                "buried_area": float(dimer.get("buried_area", 0.0) or 0.0),
                "num_residue_contacts": int(dimer.get("num_residue_contacts", 0) or 0),
                "num_atom_contacts": int(dimer.get("num_atom_contacts", 0) or 0),
                "dimer_index": index,
            })

    # ── multimers ────────────────────────────────────────────────────────
    multimer_list = bundle.get("tight_multimers", [])
    if isinstance(multimer_list, list):
        for mindex, multimer in enumerate(multimer_list, start=1):
            if not isinstance(multimer, dict):
                continue
            rows["multimers"].append({
                "pdb_id": pdb_id,
                "source_path": source_path,
                "assembly_id": str(multimer.get("assembly_id", assembly_id) or assembly_id),
                "assembly_mode": str(multimer.get("assembly_mode", assembly_mode) or assembly_mode),
                "multimer_id": str(multimer.get("multimer_id", "") or ""),
                "multimer_index": mindex,
                "num_component_copies": int(multimer.get("num_component_copies", 0) or 0),
                "num_members": int(multimer.get("num_members", 0) or 0),
                "num_member_instances": int(multimer.get("num_member_instances", 0) or 0),
                "num_internal_edges": int(multimer.get("num_internal_edges", 0) or 0),
                "multimer_type": str(multimer.get("multimer_type", "") or ""),
                "support_score": float(multimer.get("support_score", 0.0) or 0.0),
                "contains_antibody_unit": bool(multimer.get("contains_antibody_unit", False)),
                "contains_tcr_pmhc_unit": bool(multimer.get("contains_tcr_pmhc_unit", False)),
                "member_chain_ids": json.dumps(multimer.get("member_chain_ids", []) or [], ensure_ascii=False),
                "member_auth_asym_ids": json.dumps(multimer.get("member_auth_asym_ids", []) or [], ensure_ascii=False),
                "member_entity_ids": json.dumps(multimer.get("member_entity_ids", []) or [], ensure_ascii=False),
                "member_chain_types": json.dumps(multimer.get("member_chain_types", []) or [], ensure_ascii=False),
                "member_copy_numbers": json.dumps(multimer.get("member_copy_numbers", []) or [], ensure_ascii=False),
                "member_instances": json.dumps(multimer.get("member_instances", []) or [], ensure_ascii=False),
            })

    # ── antibody complexes ───────────────────────────────────────────────
    ab_complexes = bundle.get("antibody_antigen_complexes", [])
    if isinstance(ab_complexes, list):
        for cindex, complex_data in enumerate(ab_complexes, start=1):
            if not isinstance(complex_data, dict):
                continue
            rows["antibody_complexes"].append({
                "pdb_id": pdb_id,
                "source_path": source_path,
                "assembly_id": str(complex_data.get("assembly_id", assembly_id) or assembly_id),
                "assembly_mode": str(complex_data.get("assembly_mode", assembly_mode) or assembly_mode),
                "complex_id": str(complex_data.get("complex_id", "") or ""),
                "complex_index": cindex,
                "antibody_unit_type": str(complex_data.get("antibody_unit_type", "") or ""),
                "antibody_heavy_chain": str(complex_data.get("antibody_heavy_chain", "") or ""),
                "antibody_heavy_auth_asym_id": str(complex_data.get("antibody_heavy_auth_asym_id", "") or ""),
                "antibody_light_chain": str(complex_data.get("antibody_light_chain", "") or ""),
                "antibody_light_auth_asym_id": str(complex_data.get("antibody_light_auth_asym_id", "") or ""),
                "antibody_chain_ids": json.dumps(complex_data.get("antibody_chain_ids", []) or [], ensure_ascii=False),
                "antibody_auth_asym_ids": json.dumps(complex_data.get("antibody_auth_asym_ids", []) or [], ensure_ascii=False),
                "antibody_entity_ids": json.dumps(complex_data.get("antibody_entity_ids", []) or [], ensure_ascii=False),
                "antigen_chain_ids": json.dumps(complex_data.get("antigen_chain_ids", []) or [], ensure_ascii=False),
                "antigen_auth_asym_ids": json.dumps(complex_data.get("antigen_auth_asym_ids", []) or [], ensure_ascii=False),
                "antigen_entity_ids": json.dumps(complex_data.get("antigen_entity_ids", []) or [], ensure_ascii=False),
                "antigen_chain_types": json.dumps(complex_data.get("antigen_chain_types", []) or [], ensure_ascii=False),
                "num_antigen_chains": int(complex_data.get("num_antigen_chains", 0) or 0),
                "num_antibody_antigen_interfaces": int(complex_data.get("num_antibody_antigen_interfaces", 0) or 0),
                "contact_score": float(complex_data.get("contact_score", 0.0) or 0.0),
            })

    # ── TCR complexes ────────────────────────────────────────────────────
    tcr_complexes = bundle.get("tcr_pmhc_complexes", [])
    if isinstance(tcr_complexes, list):
        for cindex, complex_data in enumerate(tcr_complexes, start=1):
            if not isinstance(complex_data, dict):
                continue
            rows["tcr_complexes"].append({
                "pdb_id": pdb_id,
                "source_path": source_path,
                "assembly_id": str(complex_data.get("assembly_id", assembly_id) or assembly_id),
                "assembly_mode": str(complex_data.get("assembly_mode", assembly_mode) or assembly_mode),
                "complex_id": str(complex_data.get("complex_id", "") or ""),
                "complex_index": cindex,
                "tcr_chain_ids": json.dumps(complex_data.get("tcr_chain_ids", []) or [], ensure_ascii=False),
                "tcr_auth_asym_ids": json.dumps(complex_data.get("tcr_auth_asym_ids", []) or [], ensure_ascii=False),
                "tcr_entity_ids": json.dumps(complex_data.get("tcr_entity_ids", []) or [], ensure_ascii=False),
                "tcr_type": str(complex_data.get("tcr_type", "") or ""),
                "mhc_chain_ids": json.dumps(complex_data.get("mhc_chain_ids", []) or [], ensure_ascii=False),
                "mhc_auth_asym_ids": json.dumps(complex_data.get("mhc_auth_asym_ids", []) or [], ensure_ascii=False),
                "mhc_entity_ids": json.dumps(complex_data.get("mhc_entity_ids", []) or [], ensure_ascii=False),
                "mhc_chain_roles": json.dumps(complex_data.get("mhc_chain_roles", []) or [], ensure_ascii=False),
                "mhc_class": str(complex_data.get("mhc_class", "") or ""),
                "peptide_chain_ids": json.dumps(complex_data.get("peptide_chain_ids", []) or [], ensure_ascii=False),
                "peptide_auth_asym_ids": json.dumps(complex_data.get("peptide_auth_asym_ids", []) or [], ensure_ascii=False),
                "peptide_entity_ids": json.dumps(complex_data.get("peptide_entity_ids", []) or [], ensure_ascii=False),
                "auxiliary_chain_ids": json.dumps(complex_data.get("auxiliary_chain_ids", []) or [], ensure_ascii=False),
                "auxiliary_auth_asym_ids": json.dumps(complex_data.get("auxiliary_auth_asym_ids", []) or [], ensure_ascii=False),
                "auxiliary_entity_ids": json.dumps(complex_data.get("auxiliary_entity_ids", []) or [], ensure_ascii=False),
                "num_tcr_chains": int(complex_data.get("num_tcr_chains", 0) or 0),
                "num_peptide_chains": int(complex_data.get("num_peptide_chains", 0) or 0),
                "num_tcr_pmhc_interfaces": int(complex_data.get("num_tcr_pmhc_interfaces", 0) or 0),
                "contact_score": float(complex_data.get("contact_score", 0.0) or 0.0),
            })

    return rows


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# ── Phase 1: worker ─────────────────────────────────────────────────────────


def _ingest_cases_to_parquet(
    args: tuple[list[Path], Path, str | None],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Process a batch of case directories and return extracted rows + stats.

    Module-level so ProcessPoolExecutor can pickle it.
    """
    case_dirs, prep_dir, cif_files_directory = args
    from cif_parse.export import load_case_output_bundles

    all_rows: dict[str, list[dict[str, Any]]] = {t: [] for t in PARQUET_TABLES}
    stats = {"ingested": 0, "skipped_unchanged": 0, "skipped_no_bundles": 0, "errors": 0}

    for case_dir in case_dirs:
        case_id = case_dir.name
        try:
            bundles = load_case_output_bundles(case_dir)
            if not bundles:
                stats["skipped_no_bundles"] += 1
                continue
            for bundle in bundles:
                if cif_files_directory:
                    summary = bundle.get("structure_summary", {})
                    sp = str(summary.get("source_path", "") or "")
                    if sp:
                        summary["source_path"] = str(Path(cif_files_directory) / Path(sp).name)
                rows = _extract_bundle_rows(bundle, case_id)
                for table_name in PARQUET_TABLES:
                    all_rows[table_name].extend(rows[table_name])
            stats["ingested"] += 1
        except Exception:
            LOGGER.warning("Failed to ingest case %s", case_dir, exc_info=True)
            stats["errors"] += 1

    return all_rows, stats


# ── Phase 2: cif_cache worker (module-level for ProcessPoolExecutor) ───────


def _cache_one_source(args: tuple[str, list[str | None], str]) -> dict[str, Any]:
    """Parse one mmCIF file and return pickled atom arrays for assemblies.

    Returns a dict with write instructions that the main process collects
    and writes to cif_coords.bin / cif_coords.idx.
    """
    source_path, assembly_ids, cif_files_directory = args
    resolved_path = source_path
    if cif_files_directory:
        resolved_path = str(Path(cif_files_directory) / Path(source_path).name)
    source_hash = _hash_source_mtime(resolved_path)
    entries: list[dict[str, Any]] = []

    try:
        from biotite.structure.io.pdbx import get_assembly, get_structure
        from cif_parse.io import read_cif_file

        cif_file = read_cif_file(resolved_path)
        quality = _read_quality(resolved_path, cif_file)

        for assembly_id in assembly_ids:
            cache_key = f"{source_path}__{assembly_id or ''}"
            try:
                if assembly_id:
                    atom_array = get_assembly(cif_file, assembly_id=assembly_id, model=1, use_author_fields=False)
                else:
                    atom_array = get_structure(cif_file, model=1, use_author_fields=False)
                if atom_array is not None and len(atom_array) > 0:
                    chain_ops_json = _read_chain_ops_json(source_path, assembly_id)
                    blob = pickle.dumps(
                        {"atom_array": atom_array, "quality": quality, "chain_ops": chain_ops_json},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    entries.append({
                        "cache_key": cache_key,
                        "source_hash": source_hash,
                        "blob": blob,
                        "status": "cached",
                    })
                else:
                    entries.append({"cache_key": cache_key, "source_hash": source_hash, "status": "empty"})
            except Exception as exc:
                LOGGER.warning("Failed to cache assembly %s for %s: %s", assembly_id, source_path, exc)
                entries.append({"cache_key": cache_key, "source_hash": source_hash, "status": "error", "error": str(exc)})
    except Exception as exc:
        LOGGER.warning("Failed to read mmCIF %s: %s", source_path, exc)
        for assembly_id in assembly_ids:
            cache_key = f"{source_path}__{assembly_id or ''}"
            entries.append({"cache_key": cache_key, "source_hash": "", "status": "error", "error": str(exc)})

    return {"source_path": source_path, "entries": entries}


# ── quality / chain-ops helpers ──────────────────────────────────────────────


def _read_quality(source_path: str, cif_file=None) -> dict[str, Any] | None:
    try:
        from cif_parse.io import read_cif_file as _read
        from biotite.structure.io.pdbx import CIFBlock
        cf = cif_file or _read(source_path)
        block = cf.block if isinstance(cf, CIFBlock) else cf
        if hasattr(block, "get"):
            exptl = block.get("_exptl.method") or block.get("_exptl_crystal.method") or ""
            if hasattr(exptl, "as_item"):
                exptl = str(exptl.as_item()) if exptl.as_item() else ""
            elif isinstance(exptl, (list, tuple)):
                exptl = str(exptl[0]) if exptl else ""
            else:
                exptl = str(exptl) if exptl else ""
            resolution = block.get("_refine.ls_d_res_high") or block.get("_reflns.d_resolution_high") or ""
            if hasattr(resolution, "as_item"):
                resolution = resolution.as_item()
            elif isinstance(resolution, (list, tuple)):
                resolution = resolution[0] if resolution else None
            try:
                resolution = float(resolution) if resolution else None
            except (ValueError, TypeError):
                resolution = None
            return {"method": exptl, "resolution": resolution}
    except Exception:
        pass
    return None


def _read_chain_ops_json(source_path: str, assembly_id: str | None) -> str | None:
    try:
        from cif_parse.io import read_assembly_chain_operations
        _, chain_ops = read_assembly_chain_operations(source_path, assembly_id=assembly_id)
        if chain_ops:
            return json.dumps(chain_ops, ensure_ascii=False)
    except Exception:
        pass
    return None


# ── Parquet merging helpers ──────────────────────────────────────────────────


def _write_parquet_table(rows: list[dict[str, Any]], output_path: Path) -> int:
    """Write a list of dict rows to a Parquet file. Returns row count."""
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"__empty__": []}), output_path)
        return 0
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({col: [row.get(col) for row in rows] for col in rows[0]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd", compression_level=3)
    return len(rows)


# ── main builder ─────────────────────────────────────────────────────────────


def build_prep_database(
    inputs: Iterable[str | Path],
    prep_dir: str | Path,
    *,
    cif_files_directory: str | None = None,
    prep_jobs: int = 4,
    load_cif_cache: bool = True,
) -> dict[str, Any]:
    """Build (or refresh) the clustering prep directory.

    Parameters
    ----------
    inputs: case-output directories or parent directories.
    prep_dir: output directory for the prep files.
    cif_files_directory: optional override for mmCIF file locations.
    prep_jobs: number of parallel workers.
    load_cif_cache: if True, also pre-load mmCIF atom arrays.

    Returns a manifest dict.
    """
    from cif_parse.clustering.common import discover_case_output_dirs

    prep_dir = Path(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    case_dirs = discover_case_output_dirs(inputs)
    LOGGER.info("Prepping %d case directories into %s", len(case_dirs), prep_dir)

    # Load existing hash metadata for incremental updates
    meta_path = prep_dir / "prep_meta.json"
    existing_meta: dict[str, str] = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8")).get("hashes", {})
        except Exception:
            existing_meta = {}

    # ── Phase 1: Parse JSON bundles → Parquet ────────────────────────────
    t1 = time.monotonic()
    prep_jobs = max(1, int(prep_jobs))
    actual_jobs = max(1, min(prep_jobs, len(case_dirs)))
    LOGGER.info("Phase 1: parsing %d case bundles → Parquet (%d workers)", len(case_dirs), actual_jobs)

    # Hash and filter: only process new/changed case dirs
    new_cases: list[Path] = []
    skipped_unchanged = 0
    case_hashes: dict[str, str] = {}
    for case_dir in case_dirs:
        ch = _hash_case_dir(case_dir)
        case_hashes[str(case_dir.resolve())] = ch
        if existing_meta.get(str(case_dir.resolve())) == ch:
            skipped_unchanged += 1
        else:
            new_cases.append(case_dir)

    if skipped_unchanged:
        LOGGER.info("Skipping %d unchanged case(s)", skipped_unchanged)

    all_rows: dict[str, list[dict[str, Any]]] = {t: [] for t in PARQUET_TABLES}
    all_source_paths: set[str] = set()
    stats = {"total_cases": len(case_dirs), "ingested": 0, "skipped_unchanged": skipped_unchanged,
             "skipped_no_bundles": 0, "errors": 0}

    if new_cases:
        if actual_jobs <= 1:
            result_rows, result_stats = _ingest_cases_to_parquet((new_cases, prep_dir, cif_files_directory))
            for table_name in PARQUET_TABLES:
                all_rows[table_name] = result_rows[table_name]
            stats["ingested"] = result_stats["ingested"]
            stats["skipped_no_bundles"] = result_stats["skipped_no_bundles"]
            stats["errors"] = result_stats["errors"]
        else:
            # Split case dirs into batches for workers
            batch_size = max(1, len(new_cases) // actual_jobs)
            batches = [new_cases[i:i + batch_size] for i in range(0, len(new_cases), batch_size)]
            task_args = [(batch, prep_dir, cif_files_directory) for batch in batches]

            with ProcessPoolExecutor(max_workers=actual_jobs) as executor:
                futures = [executor.submit(_ingest_cases_to_parquet, arg) for arg in task_args]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Parsing case bundles", unit="batch"):
                    try:
                        batch_rows, batch_stats = future.result()
                        for table_name in PARQUET_TABLES:
                            all_rows[table_name].extend(batch_rows[table_name])
                        stats["ingested"] += batch_stats["ingested"]
                        stats["skipped_no_bundles"] += batch_stats["skipped_no_bundles"]
                        stats["errors"] += batch_stats["errors"]
                    except Exception as exc:
                        LOGGER.warning("Phase 1 worker failed: %s", exc)

    # Collect unique source paths from new rows
    for row in tqdm(all_rows["monomers"], desc="Collecting source paths", unit="row"):
        sp = row.get("source_path", "")
        if sp:
            all_source_paths.add(sp)

    # If only incremental update: also collect source paths from existing parquet
    if skipped_unchanged > 0 and all_source_paths:
        try:
            import pyarrow.parquet as pq
            for table_name in ["dimers", "multimers", "antibody_complexes", "tcr_complexes"]:
                existing_path = prep_dir / f"{table_name}.parquet"
                if existing_path.exists():
                    pf = pq.ParquetFile(existing_path)
                    for rg_idx in range(pf.metadata.num_row_groups):
                        tbl = pf.read_row_group(rg_idx, columns=["source_path"])
                        for sp in tbl.column("source_path").to_pylist():
                            if sp:
                                all_source_paths.add(sp)
        except ImportError:
            pass

    # Write Parquet files
    for table_name in tqdm(PARQUET_TABLES, desc="Writing Parquet files", unit="table"):
        output_path = prep_dir / f"{table_name}.parquet"
        if all_rows[table_name]:
            n = _write_parquet_table(all_rows[table_name], output_path)
            LOGGER.info("Wrote %s: %d rows", output_path.name, n)
        elif not output_path.exists():
            _write_parquet_table([], output_path)

    # Save metadata
    json.dumps({"hashes": case_hashes}, ensure_ascii=False)  # pre-validate
    meta_path.write_text(json.dumps({"hashes": case_hashes}, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Phase 1 complete (%.1fs): %d ingested, %d skipped, %d source paths",
                time.monotonic() - t1, stats["ingested"], skipped_unchanged, len(all_source_paths))

    # ── Phase 2: mmCIF → AtomArray binary cache ──────────────────────────
    cif_stats = {"cached": 0, "skipped": 0}
    if load_cif_cache and all_source_paths:
        t2 = time.monotonic()
        LOGGER.info("Phase 2: caching atom arrays for %d source files", len(all_source_paths))

        # Collect (source_path, assembly_id) pairs from all tables
        cache_pairs: dict[str, set[str | None]] = {}
        source_paths_ordered = sorted(all_source_paths)
        for sp in source_paths_ordered:
            cache_pairs.setdefault(sp, set()).add(None)

        # Add assembly_ids from parquet tables (if we have the files)
        try:
            import pyarrow.parquet as pq
            for table_name in ["dimers", "multimers"]:
                p = prep_dir / f"{table_name}.parquet"
                if p.exists():
                    pf = pq.ParquetFile(p)
                    for rg_idx in range(pf.metadata.num_row_groups):
                        tbl = pf.read_row_group(rg_idx, columns=["source_path", "assembly_id"])
                        for sp, aid in zip(tbl.column("source_path").to_pylist(),
                                           tbl.column("assembly_id").to_pylist()):
                            if sp and sp in cache_pairs:
                                cache_pairs[sp].add(aid if aid else None)
        except ImportError:
            pass

        tasks = [(sp, sorted(aids, key=lambda x: x or ""), cif_files_directory)
                 for sp, aids in sorted(cache_pairs.items())]
        LOGGER.info("Dispatching %d source files to %d worker processes", len(tasks), min(actual_jobs, len(tasks)))

        # Workers write to temp files; master merges
        tmpdir = Path(tempfile.mkdtemp(prefix="cif_cache_", dir=prep_dir))
        try:
            bin_path = prep_dir / "cif_coords.bin"
            idx_path = prep_dir / "cif_coords.idx"

            if len(tasks) <= 1:
                all_entries: list[dict[str, Any]] = []
                for task in tqdm(tasks, desc="Caching mmCIF structures", unit="source"):
                    result = _cache_one_source(task)
                    all_entries.extend(result["entries"])
            else:
                # Each worker returns entries with blob data; master writes sequentially
                all_entries = []
                with ProcessPoolExecutor(max_workers=min(actual_jobs, len(tasks))) as executor:
                    futures = [executor.submit(_cache_one_source, task) for task in tasks]
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc="Caching mmCIF structures", unit="source"):
                        try:
                            result = future.result()
                            all_entries.extend(result["entries"])
                        except Exception as exc:
                            LOGGER.warning("Phase 2 worker failed: %s", exc)

            # Write bin + idx sequentially from collected entries
            with bin_path.open("wb") as bin_fh, idx_path.open("wb") as idx_fh:
                offset = 0
                for entry in tqdm(all_entries, desc="Writing cif_coords", unit="blob"):
                    if entry["status"] != "cached":
                        cif_stats["skipped"] += 1
                        continue
                    blob = entry["blob"]
                    blob_len = len(blob)
                    bin_fh.write(blob)
                    source_hash = entry.get("source_hash", "").ljust(64, "\0")[:64]
                    idx_fh.write(_IDX_ENTRY_STRUCT.pack(
                        source_hash.encode("ascii", errors="replace").ljust(64, b"\0")[:64],
                        offset,
                        blob_len,
                    ))
                    offset += blob_len
                    cif_stats["cached"] += 1

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        LOGGER.info("Phase 2 complete (%.1fs): %d cached, %d skipped",
                    time.monotonic() - t2, cif_stats["cached"], cif_stats["skipped"])

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    LOGGER.info("Prep built (%.1fs): %d source paths, %d errors",
                elapsed, len(all_source_paths), stats["errors"])

    return {
        "prep_dir": str(prep_dir.resolve()),
        "total_cases": stats["total_cases"],
        "ingested": stats["ingested"],
        "skipped_unchanged": stats["skipped_unchanged"],
        "skipped_no_bundles": stats["skipped_no_bundles"],
        "errors": stats["errors"],
        "total_source_paths": len(all_source_paths),
        "cif_cached": cif_stats["cached"],
        "cif_skipped": cif_stats["skipped"],
        "elapsed_seconds": round(elapsed, 1),
    }


# ── consumer API (used by collect_* and extract_* functions) ─────────────────


def open_prep_parquet(prep_dir: str | Path, table_name: str) -> "pyarrow.parquet.ParquetFile | None":
    """Open a prep Parquet file for reading. Returns None if not available."""
    try:
        import pyarrow.parquet as pq
        path = Path(prep_dir) / f"{table_name}.parquet"
        if path.exists():
            return pq.ParquetFile(path)
    except ImportError:
        pass
    return None


def iter_parquet_rows(prep_dir: str | Path, table_name: str, columns: list[str] | None = None) -> Iterable[dict[str, Any]]:
    """Iterate over all rows in a prep Parquet file as dicts."""
    pf = open_prep_parquet(prep_dir, table_name)
    if pf is None:
        return
    for rg_idx in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=columns)
        cols = tbl.column_names
        for i in range(tbl.num_rows):
            yield {c: tbl.column(c)[i].as_py() for c in cols}


def load_cif_coords_index(prep_dir: str | Path) -> dict[str, tuple[int, int]] | None:
    """Load the cif_coords.idx into a dict {source_hash: (offset, length)}.

    Returns None if the index file does not exist.
    """
    idx_path = Path(prep_dir) / "cif_coords.idx"
    if not idx_path.exists():
        return None
    index: dict[str, tuple[int, int]] = {}
    with idx_path.open("rb") as fh:
        while True:
            entry = fh.read(_IDX_ENTRY_SIZE)
            if len(entry) < _IDX_ENTRY_SIZE:
                break
            source_hash, offset, length = _IDX_ENTRY_STRUCT.unpack(entry)
            key = source_hash.decode("ascii").rstrip("\0")
            index[key] = (offset, length)
    return index


def load_cif_from_prep(
    prep_dir: str | Path,
    source_path: str,
    assembly_id: str | None = None,
    *,
    index: dict[str, tuple[int, int]] | None = None,
    mmap: Any = None,
) -> dict[str, Any] | None:
    """Fetch a cached AtomArray + metadata from the binary index.

    Returns a dict with ``atom_array``, ``quality``, ``chain_ops`` keys,
    or None if not cached.
    """
    idx = index or load_cif_coords_index(prep_dir)
    if idx is None:
        return None
    cache_key = f"{source_path}__{assembly_id or ''}"
    source_hash = _hash_source_mtime(source_path)
    entry = idx.get(source_hash)
    if entry is None:
        return None

    offset, length = entry
    bin_path = Path(prep_dir) / "cif_coords.bin"
    if mmap is not None:
        data = mmap[offset:offset + length]
    else:
        with bin_path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(length)
    try:
        return pickle.loads(data)
    except Exception:
        return None


# ── backward-compatible wrappers (for higher-order collect functions) ───────


def load_bundles_for_collect(
    case_dirs: Iterable[str | Path],
    *,
    prep_db_path: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Adapter: returns None to signal file-based fallback for collect functions.

    The higher-order collect functions (dimer, multimer, antibody, TCR) still
    use file-by-file reading.  This stub always returns None so they fall back.
    """
    return None


def load_case_bundles(
    case_dir: Path,
    *,
    prep_bundles: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Adapter: falls back to file-based bundle loading."""
    from cif_parse.export import load_case_output_bundles
    return load_case_output_bundles(case_dir)
