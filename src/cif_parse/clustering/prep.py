"""Pre-processing for clustering at any scale.

Produces a self-contained directory that replaces thousands of individual
``result.json.gz`` reads with columnar Parquet scans.  Coordinates are read
from parse-stage ``atoms/*.pkl`` caches via the source map written by prep;
clustering consumers do not read original mmCIF files.

Layout
------
::

    clustering_prep/
    ├── monomers.parquet              # pre-parsed polymer chain rows
    ├── dimers.parquet                # pre-parsed dimer interface rows
    ├── multimers.parquet             # pre-parsed tight multimer rows
    ├── antibody_complexes.parquet    # pre-parsed antibody-antigen complex rows
    ├── tcr_complexes.parquet         # pre-parsed TCR-pMHC complex rows
    ├── entry_quality.parquet         # source-level quality metadata
    └── source_case_dir_map.json      # source_path -> parse case directory

Builder
-------
``cif-parse-cluster prep`` builds (or refreshes) the directory.  Hash-based
incremental updates skip unchanged case directories.  Workers can be
configured with ``--prep-jobs``.

Consumers
---------
The ``collect_*`` functions in each clustering module use PyArrow to scan the
Parquet files in a streaming fashion.  The ``extract_*`` functions fetch
cached AtomArray objects from parse-stage atom pickle files.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

# Parquet output file names
PARQUET_TABLES = [
    "entry_quality",
    "monomers",
    "dimers",
    "multimers",
    "antibody_complexes",
    "tcr_complexes",
]

# ── json extraction helpers (move heavy parsing into prep once) ──────────────


def _normalize_sequence(value: Any) -> str:
    from cif_parse.clustering.polymer import normalize_sequence

    return normalize_sequence(value)


def _is_polymer_chain(chain_payload: dict[str, Any]) -> bool:
    from cif_parse.clustering.polymer import is_polymer_chain

    return is_polymer_chain(chain_payload)


def _classify_polymer_class(chain_payload: dict[str, Any]) -> str | None:
    from cif_parse.clustering.polymer import classify_polymer_class

    return classify_polymer_class(chain_payload)


def _has_sequence_identity(polymer_class: str, sequence: str, chain_payload: dict[str, Any]) -> bool:
    from cif_parse.clustering.polymer import OTHER_POLYMER_CLASS, has_component_sequence_details

    if sequence:
        return True
    return polymer_class == OTHER_POLYMER_CLASS and has_component_sequence_details(chain_payload)


_METHOD_PRIORITY = {
    "x-ray diffraction": 0,
    "electron microscopy": 1,
    "solution nmr": 2,
    "solid-state nmr": 2,
}


def _safe_float(value: Any) -> float | None:
    if value in (None, "", ".", "?"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_quality_row(summary: dict[str, Any], case_id: str) -> dict[str, Any]:
    metadata = summary.get("entry_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    source_path = str(summary.get("source_path", "") or "")
    pdb_id = str(summary.get("pdb_id", "") or "")
    method = str(metadata.get("experimental_method", "") or "").strip()
    methods = [method] if method else []
    primary_method = method or None
    resolution = _safe_float(metadata.get("resolution"))
    return {
        "pdb_id": pdb_id,
        "source_path": source_path,
        "source_case_dir": case_id,
        "experimental_methods": json.dumps(sorted(set(methods)), ensure_ascii=False),
        "primary_method": primary_method,
        "method_priority": _METHOD_PRIORITY.get(method.lower(), 99) if method else 99,
        "resolution": resolution,
        "release_date": str(metadata.get("release_date", "") or ""),
        "metadata_source": str(metadata.get("metadata_source", "") or ""),
        "metadata_warning": str(metadata.get("metadata_warning", "") or ""),
    }


def _bundle_assembly_id(bundle: dict[str, Any]) -> str:
    """Infer the concrete assembly represented by one parse output bundle."""

    summary = bundle.get("structure_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    summary_assembly_id = str(summary.get("assembly_id", "") or "")
    if summary_assembly_id:
        return summary_assembly_id

    observed: set[str] = set()
    for table_name in (
        "dimer_interfaces",
        "tight_multimers",
        "antibody_antigen_complexes",
        "tcr_pmhc_complexes",
    ):
        records = bundle.get(table_name, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            assembly_id = str(record.get("assembly_id", "") or "")
            if assembly_id:
                observed.add(assembly_id)
    if len(observed) == 1:
        return next(iter(observed))

    assembly_ids = [str(a) for a in summary.get("assembly_ids", []) if str(a)]
    if len(assembly_ids) == 1:
        return assembly_ids[0]
    return ""


def _extract_bundle_rows(bundle: dict[str, Any], case_id: str) -> dict[str, list[dict[str, Any]]]:
    """Parse one case bundle dict and return rows for each Parquet table.

    Returns ``{"monomers": [...], "dimers": [...], ...}``.
    """
    summary = bundle.get("structure_summary", {})
    source_path = str(summary.get("source_path", "") or "")
    pdb_id = str(summary.get("pdb_id", "") or "")
    assembly_id = _bundle_assembly_id(bundle)
    assembly_mode = str(summary.get("assembly_mode", "") or "")

    rows: dict[str, list[dict[str, Any]]] = {t: [] for t in PARQUET_TABLES}
    rows["entry_quality"].append(_entry_quality_row(summary, case_id))

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
            if not _has_sequence_identity(polymer_class, seq, chain):
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
                "assembly_id": assembly_id,
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
                "source_case_dir": case_id,
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
                "source_case_dir": case_id,
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
                "source_case_dir": case_id,
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
                "antigen_chain_ids": json.dumps(complex_data.get("antigen_chain_ids", []) or [], ensure_ascii=False),
                "antigen_auth_asym_ids": json.dumps(complex_data.get("antigen_auth_asym_ids", []) or [], ensure_ascii=False),
                "antigen_chain_types": json.dumps(complex_data.get("antigen_chain_types", []) or [], ensure_ascii=False),
                "auxiliary_component_ids": json.dumps(complex_data.get("auxiliary_component_ids", []) or [], ensure_ascii=False),
                "auxiliary_component_auth_asym_ids": json.dumps(complex_data.get("auxiliary_component_auth_asym_ids", []) or [], ensure_ascii=False),
                "auxiliary_branched_ids": json.dumps(complex_data.get("auxiliary_branched_ids", []) or [], ensure_ascii=False),
                "auxiliary_branched_auth_asym_ids": json.dumps(complex_data.get("auxiliary_branched_auth_asym_ids", []) or [], ensure_ascii=False),
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
                "source_case_dir": case_id,
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
    args: tuple[list[Path], Path, int, str | None],
) -> dict[str, Any]:
    """Process a batch of cases and write temp Parquet files directly.

    Each worker writes ``temp_{batch_id}_{table}.parquet``,
    ``temp_{batch_id}_sources.txt``, and ``temp_{batch_id}_atoms.txt``
    (mapping source_path → atoms_dir).
    Returns only file paths and stats — no row data through IPC.
    """
    case_dirs, prep_dir, batch_id, cif_files_directory = args
    from cif_parse.export import load_case_output_bundles

    all_rows: dict[str, list[dict[str, Any]]] = {t: [] for t in PARQUET_TABLES}
    source_paths: set[str] = set()
    atom_cache_map: dict[str, str] = {}  # source_path → atoms_dir
    source_case_dir_map: dict[str, str] = {}  # source_path → source_case_dir
    stats = {"ingested": 0, "skipped_no_bundles": 0, "errors": 0}

    for case_dir in case_dirs:
        case_id = case_dir.name
        atoms_dir = case_dir / "atoms"
        has_atoms = atoms_dir.is_dir()
        try:
            bundles = load_case_output_bundles(case_dir)
            if not bundles:
                stats["skipped_no_bundles"] += 1
                continue
            for bundle in bundles:
                rows = _extract_bundle_rows(bundle, case_id)
                for table_name in PARQUET_TABLES:
                    all_rows[table_name].extend(rows[table_name])
                for row in rows["monomers"]:
                    sp = row.get("source_path", "")
                    if sp:
                        source_paths.add(sp)
                        source_case_dir_map.setdefault(sp, case_id)
                        if has_atoms and sp not in atom_cache_map:
                            atom_cache_map[sp] = str(atoms_dir)
                for table_name in (
                    "dimers",
                    "multimers",
                    "antibody_complexes",
                    "tcr_complexes",
                    "entry_quality",
                ):
                    for row in rows[table_name]:
                        sp = row.get("source_path", "")
                        if sp:
                            source_paths.add(sp)
                            source_case_dir_map.setdefault(sp, case_id)
                            if has_atoms and sp not in atom_cache_map:
                                atom_cache_map[sp] = str(atoms_dir)
            stats["ingested"] += 1
        except Exception:
            LOGGER.warning("Failed to ingest case %s", case_dir, exc_info=True)
            stats["errors"] += 1

    # Write temp files into the shared temp directory
    tmp_dir = Path(prep_dir)  # prep_dir is actually tmp_phase1 passed from main
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for table_name in PARQUET_TABLES:
        if all_rows[table_name]:
            p = tmp_dir / f"temp_{batch_id}_{table_name}.parquet"
            _write_parquet_table(all_rows[table_name], p)
            paths[table_name] = str(p)
    sources_path = tmp_dir / f"temp_{batch_id}_sources.txt"
    sources_path.write_text("\n".join(sorted(source_paths)), encoding="utf-8")
    paths["sources"] = str(sources_path)
    # Serialize atom cache map as JSON lines
    if atom_cache_map:
        atoms_path = tmp_dir / f"temp_{batch_id}_atoms.jsonl"
        with atoms_path.open("w", encoding="utf-8") as fh:
            for sp, ad in sorted(atom_cache_map.items()):
                fh.write(json.dumps({"source_path": sp, "atoms_dir": ad}, ensure_ascii=False) + "\n")
        paths["atoms"] = str(atoms_path)
    if source_case_dir_map:
        source_cases_path = tmp_dir / f"temp_{batch_id}_source_cases.jsonl"
        with source_cases_path.open("w", encoding="utf-8") as fh:
            for sp, case_id in sorted(source_case_dir_map.items()):
                fh.write(json.dumps({"source_path": sp, "source_case_dir": case_id}, ensure_ascii=False) + "\n")
        paths["source_cases"] = str(source_cases_path)
    return {"batch_id": batch_id, "paths": paths, "stats": stats}


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


def _merge_parquet_files(temp_files: list[Path], output_path: Path) -> None:
    """Merge multiple Parquet files into one via row-group concatenation."""
    import pyarrow.parquet as pq
    if len(temp_files) == 1:
        shutil.move(str(temp_files[0]), str(output_path))
        return
    schema = pq.read_schema(temp_files[0])
    with pq.ParquetWriter(output_path, schema, compression="zstd", compression_level=3) as writer:
        for tf in temp_files:
            pf = pq.ParquetFile(tf)
            for rg_idx in range(pf.metadata.num_row_groups):
                writer.write_table(pf.read_row_group(rg_idx))


def _merge_batch_stats(result: dict[str, Any], stats: dict[str, Any]) -> None:
    """Accumulate worker stats into the global stats dict."""
    for key in ("ingested", "skipped_no_bundles", "errors"):
        stats[key] += result.get("stats", {}).get(key, 0)


def _collect_source_paths_from_parquet(prep_dir: Path, sink: set[str]) -> None:
    """Read source_path columns from existing Parquet files into *sink*."""
    try:
        import pyarrow.parquet as pq
        for table_name in PARQUET_TABLES:
            p = prep_dir / f"{table_name}.parquet"
            if not p.exists():
                continue
            pf = pq.ParquetFile(p)
            schema_names = set(pf.schema_arrow.names)
            if "source_path" not in schema_names:
                continue
            for rg_idx in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg_idx, columns=["source_path"])
                for sp in tbl.column("source_path").to_pylist():
                    if sp:
                        sink.add(sp)
    except ImportError:
        pass


# ── main builder ─────────────────────────────────────────────────────────────


def build_prep_database(
    inputs: Iterable[str | Path],
    prep_dir: str | Path,
    *,
    cif_files_directory: str | None = None,
    prep_jobs: int = 4,
) -> dict[str, Any]:
    """Build (or refresh) the clustering prep directory.

    Parameters
    ----------
    inputs: case-output directories or parent directories.
    prep_dir: output directory for the prep files.
    cif_files_directory: deprecated; ignored because prep consumes parse JSON
        and parse-stage atom pkl files only.
    prep_jobs: number of parallel workers.

    Returns a manifest dict.
    """
    from cif_parse.clustering.common import discover_case_output_dirs

    prep_dir = Path(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    case_dirs = discover_case_output_dirs(inputs)
    LOGGER.info("Prepping %d case directories into %s", len(case_dirs), prep_dir)

    # ── Phase 1: Parse JSON bundles → Parquet ────────────────────────────
    t1 = time.monotonic()
    prep_jobs = max(1, int(prep_jobs))
    actual_jobs = max(1, min(prep_jobs, len(case_dirs)))
    LOGGER.info("Phase 1: parsing %d case bundles → Parquet (%d workers)", len(case_dirs), actual_jobs)

    all_source_paths: set[str] = set()
    stats = {"total_cases": len(case_dirs), "ingested": 0, "skipped_no_bundles": 0, "errors": 0}

    from cif_parse.settings import get_fast_temp_dir
    tmp_phase1 = get_fast_temp_dir("phase1")

    # Small batches for load balancing: aim for at least 4× num_workers
    # batches, but keep a modest minimum so 64+ workers are not underfed by
    # a hard 500-case floor.
    cases_per_batch = max(50, len(case_dirs) // (actual_jobs * 4)) if case_dirs else 50
    batches = [case_dirs[i:i + cases_per_batch] for i in range(0, len(case_dirs), cases_per_batch)]
    task_args = [(batch, str(tmp_phase1), bid, cif_files_directory) for bid, batch in enumerate(batches)]

    LOGGER.info("Dispatching %d batches (batch_size=%d) to %d workers",
                len(batches), cases_per_batch, actual_jobs)

    if not batches:
        pass
    elif len(batches) <= 1:
        result = _ingest_cases_to_parquet(task_args[0])
        _merge_batch_stats(result, stats)
    else:
        with ProcessPoolExecutor(max_workers=actual_jobs) as executor:
            futures = [executor.submit(_ingest_cases_to_parquet, arg) for arg in task_args]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Parsing case bundles", unit="batch"):
                try:
                    _merge_batch_stats(future.result(), stats)
                except Exception as exc:
                    LOGGER.warning("Phase 1 worker failed: %s", exc)

    # Collect source_paths and atom cache map from temp files
    atom_cache_map: dict[str, str] = {}
    source_case_dir_map: dict[str, str] = {}
    for sources_file in sorted(tmp_phase1.glob("temp_*_sources.txt")):
        with sources_file.open(encoding="utf-8") as handle:
            for line in handle:
                sp = line.strip()
                if sp:
                    all_source_paths.add(sp)
        sources_file.unlink()
    for atoms_file in sorted(tmp_phase1.glob("temp_*_atoms.jsonl")):
        with atoms_file.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                entry = json.loads(text)
                sp = entry.get("source_path", "")
                ad = entry.get("atoms_dir", "")
                if sp and ad and sp not in atom_cache_map:
                    atom_cache_map[sp] = ad
        atoms_file.unlink()
    for source_cases_file in sorted(tmp_phase1.glob("temp_*_source_cases.jsonl")):
        with source_cases_file.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                entry = json.loads(text)
                sp = entry.get("source_path", "")
                source_case_dir = entry.get("source_case_dir", "")
                if sp and source_case_dir and sp not in source_case_dir_map:
                    source_case_dir_map[sp] = source_case_dir
        source_cases_file.unlink()

    # Merge temp Parquet files into final Parquet files (metadata-only concatenation)
    for table_name in tqdm(PARQUET_TABLES, desc="Merging Parquet", unit="table"):
        temp_files = sorted(tmp_phase1.glob(f"temp_*_{table_name}.parquet"))
        output_path = prep_dir / f"{table_name}.parquet"
        if temp_files:
            _merge_parquet_files(temp_files, output_path)
            for tf in temp_files:
                if tf.exists():
                    tf.unlink()
            LOGGER.info("Merged %s from %d temp files", output_path.name, len(temp_files))
        elif not output_path.exists():
            _write_parquet_table([], output_path)

    # Cleanup temp dir
    try:
        remaining = list(tmp_phase1.iterdir())
        for f in remaining:
            f.unlink()
        tmp_phase1.rmdir()
    except OSError:
        pass

    # Collect source_paths from the Parquet files just written
    _collect_source_paths_from_parquet(prep_dir, all_source_paths)

    LOGGER.info("Phase 1 complete (%.1fs): %d ingested, %d source paths",
                time.monotonic() - t1, stats["ingested"], len(all_source_paths))

    source_map_path = prep_dir / "source_case_dir_map.json"
    source_map_path.write_text(
        json.dumps(source_case_dir_map, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %d source_path -> case_dir mappings", len(source_case_dir_map))

    cif_stats = {
        "cached": 0,
        "cached_alias": 0,
        "skipped": 0,
        "empty": 0,
        "missing_atom_cache": 0,
        "errors": 0,
    }
    coord_chunk_sizes: list[int] = []

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.monotonic() - t0
    LOGGER.info("Prep built (%.1fs): %d source paths, %d errors",
                elapsed, len(all_source_paths), stats["errors"])

    manifest = {
        "prep_dir": str(prep_dir.resolve()),
        "parsed_input": str(Path(next(iter(inputs))).resolve()) if inputs else "",
        "total_cases": stats["total_cases"],
        "ingested": stats["ingested"],
        "skipped_no_bundles": stats["skipped_no_bundles"],
        "errors": stats["errors"],
        "total_source_paths": len(all_source_paths),
        "coordinate_backend": "pkl_atoms",
        "source_case_dir_map": str(source_map_path),
        "source_case_dir_map_entries": len(source_case_dir_map),
        "cif_cached": cif_stats["cached"],
        "cif_skipped": cif_stats["skipped"],
        "coord_cached": cif_stats["cached"],
        "coord_cached_alias": cif_stats["cached_alias"],
        "coord_skipped": cif_stats["skipped"],
        "coord_empty": cif_stats["empty"],
        "coord_missing_atom_cache": cif_stats["missing_atom_cache"],
        "coord_errors": cif_stats["errors"],
        "coord_chunks": len(coord_chunk_sizes),
        "coord_chunk_min_bytes": min(coord_chunk_sizes) if coord_chunk_sizes else 0,
        "coord_chunk_max_bytes": max(coord_chunk_sizes) if coord_chunk_sizes else 0,
        "coord_chunk_mean_bytes": (
            round(sum(coord_chunk_sizes) / len(coord_chunk_sizes), 1) if coord_chunk_sizes else 0.0
        ),
        "elapsed_seconds": round(elapsed, 1),
    }
    dump_path = prep_dir / "manifest.json"
    dump_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ── consumer API (used by collect_* and extract_* functions) ─────────────────


def open_prep_parquet(
    prep_dir: str | Path,
    table_name: str,
    *,
    required: bool = False,
) -> "pyarrow.parquet.ParquetFile | None":
    """Open a prep Parquet file for reading. Returns None if not available."""
    path = Path(prep_dir) / f"{table_name}.parquet"
    try:
        import pyarrow.parquet as pq
        if path.exists():
            return pq.ParquetFile(path)
    except ImportError as exc:
        if required:
            raise RuntimeError("pyarrow is required when --prep-dir is provided") from exc
        return None
    if required:
        raise FileNotFoundError(f"Prep Parquet table not found: {path}")
    return None


def iter_parquet_rows(
    prep_dir: str | Path,
    table_name: str,
    columns: list[str] | None = None,
    *,
    required: bool = False,
) -> Iterable[dict[str, Any]]:
    """Iterate over all rows in a prep Parquet file as dicts."""
    pf = open_prep_parquet(prep_dir, table_name, required=required)
    if pf is None:
        return
    try:
        for rg_idx in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg_idx, columns=columns)
            column_data = tbl.to_pydict()
            for i in range(tbl.num_rows):
                yield {c: values[i] for c, values in column_data.items()}
    finally:
        pf.close()


# Shared pkl reader reused by dimer/multimer/antibody/TCR builders.
_shared_pkl_reader: Any = None
_shared_pkl_reader_prep_dir: Path | None = None


def _get_pkl_reader(prep_dir: str | Path) -> Any | None:
    global _shared_pkl_reader, _shared_pkl_reader_prep_dir
    resolved = Path(prep_dir).resolve()
    if _shared_pkl_reader is not None and _shared_pkl_reader_prep_dir == resolved:
        return _shared_pkl_reader
    try:
        from cif_parse.clustering.atom_cache import (
            PklAtomReader,
            load_source_case_dir_map,
            resolve_cases_root,
        )

        _shared_pkl_reader = PklAtomReader(
            resolve_cases_root(resolved),
            load_source_case_dir_map(resolved),
        )
        _shared_pkl_reader_prep_dir = resolved
        return _shared_pkl_reader
    except Exception:
        LOGGER.debug("Failed to initialize direct parse atom reader", exc_info=True)
        _shared_pkl_reader = None
        _shared_pkl_reader_prep_dir = None
        return None


def close_blob_handles() -> None:
    """Clear process-local prep readers."""
    global _shared_pkl_reader, _shared_pkl_reader_prep_dir
    _shared_pkl_reader = None
    _shared_pkl_reader_prep_dir = None


def assemble_atom_array_from_chains(
    prep_dir: str | Path,
    source_path: str,
    chain_specs: list[tuple[str, int | None]],  # [(label_asym_id, sym_id), ...]
    *,
    assembly_id: str | None = None,
    index: dict[str, tuple[Path, int, int]] | None = None,
) -> "AtomArray | None":
    """Load multiple chains from parse-stage atom pkl caches.

    Returns None if any chain is not found.  Used by dimer/multimer/complex
    extraction.
    """
    del index
    reader = _get_pkl_reader(prep_dir)
    if reader is not None:
        return reader.load_chains(
            source_path,
            chain_specs,
            assembly_id=assembly_id,
            filter_hetero=False,
        )
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
