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
import pickle
import shutil
import struct
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

_IDX_ENTRY_STRUCT = struct.Struct("64s Q I")  # hash(64 B hex ASCII), offset(8 B), len(4 B)
_IDX_ENTRY_SIZE = _IDX_ENTRY_STRUCT.size  # 76 bytes

# Compressed blob format: 4-byte magic + 4-byte raw length + zlib(data)
_BLOB_MAGIC = b"\x01\xC1\xF0\x01"
_BLOB_HEADER = struct.Struct("4s I")
_ZLIB_LEVEL = 3  # fast, ~7x compression on AtomArray pickles


def _compress_blob(data: bytes) -> bytes:
    return _BLOB_HEADER.pack(_BLOB_MAGIC, len(data)) + zlib.compress(data, _ZLIB_LEVEL)


def _decompress_blob(data: bytes) -> bytes:
    magic, raw_len = _BLOB_HEADER.unpack(data[: _BLOB_HEADER.size])
    if magic != _BLOB_MAGIC:
        raise ValueError("cif_coords blob has unexpected magic; rebuild prep")
    return zlib.decompress(data[_BLOB_HEADER.size:])

# Parquet output file names
PARQUET_TABLES = [
    "monomers",
    "dimers",
    "multimers",
    "antibody_complexes",
    "tcr_complexes",
]

# ── hash helpers ─────────────────────────────────────────────────────────────


def _hash_source_mtime(source_path: str) -> str:
    p = Path(source_path)
    if not p.exists():
        return "missing"
    stat = p.stat()
    return hashlib.sha256(f"{p.resolve()}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()


def _blob_index_key(source_path: str, assembly_id: str | None) -> str:
    """Return a stable 64-char hex key that uniquely identifies a cached blob.

    Different assemblies from the same mmCIF file get different keys, so the
    index can store them independently.
    """
    cache_key = f"{source_path}__{assembly_id or ''}"
    return hashlib.sha256(cache_key.encode()).hexdigest()


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
                if cif_files_directory:
                    summary = bundle.get("structure_summary", {})
                    sp = str(summary.get("source_path", "") or "")
                    if sp:
                        summary["source_path"] = str(Path(cif_files_directory) / Path(sp).name)
                rows = _extract_bundle_rows(bundle, case_id)
                for table_name in PARQUET_TABLES:
                    all_rows[table_name].extend(rows[table_name])
                for row in rows["monomers"]:
                    sp = row.get("source_path", "")
                    if sp:
                        source_paths.add(sp)
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
    return {"batch_id": batch_id, "paths": paths, "stats": stats}


# ── Phase 2: cif_cache worker ───────────────────────────────────────────────


def _cache_sources_to_temp(
    args: tuple[list[tuple[str, list[str | None], str]], Path, int, dict[str, str] | None],
) -> dict[str, Any]:
    """Process a batch of mmCIF files and write cached atom arrays to one chunk.

    All source_paths in the batch write to a single ``temp_{worker_id}.bin`` +
    ``temp_{worker_id}.idx`` pair — no inter-process coordination needed because
    each batch runs in exactly one worker process.

    Returns metadata only — no blob data through IPC.
    """
    chunk_tasks, tmp_dir, worker_id, atom_cache_map = args
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    bin_path = tmp_dir / f"temp_{worker_id}.bin"
    idx_path = tmp_dir / f"temp_{worker_id}.idx"
    all_entries: list[dict[str, Any]] = []

    for source_path, assembly_ids, cif_files_directory in chunk_tasks:
        resolved_path = source_path
        if cif_files_directory:
            resolved_path = str(Path(cif_files_directory) / Path(source_path).name)

        # Fast path: use atom cache from parsing stage
        atoms_dir = atom_cache_map.get(source_path) if atom_cache_map else None
        if atoms_dir:
            atoms_dir = Path(atoms_dir)
            try:
                with bin_path.open("ab") as bin_fh, idx_path.open("ab") as idx_fh:
                    for assembly_id in assembly_ids:
                        cache_key = f"{source_path}__{assembly_id or ''}"
                        pkl_name = f"{assembly_id or '_none'}.pkl"
                        pkl_path = atoms_dir / pkl_name
                        if not pkl_path.exists():
                            if assembly_id is None:
                                all_entries.append({"cache_key": cache_key, "status": "empty"})
                                continue
                            pkl_none = atoms_dir / "_none.pkl"
                            if pkl_none.exists():
                                pkl_path = pkl_none
                            else:
                                all_entries.append({"cache_key": cache_key, "status": "empty"})
                                continue
                        raw = pickle.loads(zlib.decompress(pkl_path.read_bytes()))
                        wrapped = _compress_blob(pickle.dumps(
                            {"atom_array": raw, "quality": None, "chain_ops": None},
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ))
                        offset = bin_fh.tell()
                        bin_fh.write(wrapped)
                        blob_key = _blob_index_key(source_path, assembly_id)
                        idx_fh.write(_IDX_ENTRY_STRUCT.pack(
                            blob_key.encode("ascii", errors="replace").ljust(64, b"\0")[:64],
                            offset, len(wrapped),
                        ))
                        all_entries.append({"cache_key": cache_key, "status": "cached"})
                continue  # skip slow path for this source_path
            except Exception as exc:
                LOGGER.debug("Atom cache read failed for %s, falling back to mmCIF: %s", source_path, exc)

        # Slow path: read and parse original mmCIF
        try:
            from biotite.structure.io.pdbx import get_assembly, get_structure
            from cif_parse.io import read_cif_file

            cif_file = read_cif_file(resolved_path)
            quality = _read_quality(resolved_path, cif_file)

            with bin_path.open("ab") as bin_fh, idx_path.open("ab") as idx_fh:
                for assembly_id in assembly_ids:
                    cache_key = f"{source_path}__{assembly_id or ''}"
                    try:
                        if assembly_id:
                            atom_array = get_assembly(cif_file, assembly_id=assembly_id, model=1, use_author_fields=False)
                        else:
                            atom_array = get_structure(cif_file, model=1, use_author_fields=False)
                        if atom_array is not None and len(atom_array) > 0:
                            chain_ops_json = _read_chain_ops_json(source_path, assembly_id)
                            blob = _compress_blob(pickle.dumps(
                                {"atom_array": atom_array, "quality": quality, "chain_ops": chain_ops_json},
                                protocol=pickle.HIGHEST_PROTOCOL,
                            ))
                            offset = bin_fh.tell()
                            bin_fh.write(blob)
                            blob_key = _blob_index_key(source_path, assembly_id)
                            idx_fh.write(_IDX_ENTRY_STRUCT.pack(
                                blob_key.encode("ascii", errors="replace").ljust(64, b"\0")[:64],
                                offset, len(blob),
                            ))
                            all_entries.append({"cache_key": cache_key, "status": "cached"})
                        else:
                            all_entries.append({"cache_key": cache_key, "status": "empty"})
                    except Exception as exc:
                        LOGGER.warning("Failed to cache assembly %s for %s: %s", assembly_id, source_path, exc)
                        all_entries.append({"cache_key": cache_key, "status": "error", "error": str(exc)})
        except Exception as exc:
            LOGGER.warning("Failed to read mmCIF %s: %s", source_path, exc)
            for assembly_id in assembly_ids:
                cache_key = f"{source_path}__{assembly_id or ''}"
                all_entries.append({"cache_key": cache_key, "status": "error", "error": str(exc)})

    return {"entries": all_entries,
            "bin_path": str(bin_path) if bin_path.exists() else None,
            "idx_path": str(idx_path) if idx_path.exists() else None}


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


def _count_cif_entries(result: dict[str, Any], stats: dict[str, int]) -> None:
    """Count cached/skipped/error entries from a Phase 2 worker result."""
    for entry in result.get("entries", []):
        status = entry.get("status", "")
        if status == "cached":
            stats["cached"] += 1
        elif status in ("skipped", "empty", "error"):
            stats["skipped"] += 1


def _read_idx_entries(idx_path: Path) -> list[bytes]:
    """Read all 76-byte index entries from a temp idx file."""
    data = idx_path.read_bytes()
    return [data[i:i + _IDX_ENTRY_SIZE] for i in range(0, len(data), _IDX_ENTRY_SIZE)]


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

    # ── Phase 1: Parse JSON bundles → Parquet ────────────────────────────
    t1 = time.monotonic()
    prep_jobs = max(1, int(prep_jobs))
    actual_jobs = max(1, min(prep_jobs, len(case_dirs)))
    LOGGER.info("Phase 1: parsing %d case bundles → Parquet (%d workers)", len(case_dirs), actual_jobs)

    all_source_paths: set[str] = set()
    stats = {"total_cases": len(case_dirs), "ingested": 0, "skipped_no_bundles": 0, "errors": 0}

    from cif_parse.settings import get_fast_temp_dir
    tmp_phase1 = get_fast_temp_dir("phase1")

    # Small batches for load balancing: at least 4× num_workers batches,
    # but each batch has at least 500 cases.  ThreadPoolExecutor is sufficient
    # because gzip decompression and JSON parsing release the GIL.
    cases_per_batch = max(500, len(case_dirs) // (actual_jobs * 4)) if case_dirs else 500
    batches = [case_dirs[i:i + cases_per_batch] for i in range(0, len(case_dirs), cases_per_batch)]
    task_args = [(batch, str(tmp_phase1), bid, cif_files_directory) for bid, batch in enumerate(batches)]

    LOGGER.info("Dispatching %d batches (batch_size=%d) to %d workers",
                len(batches), cases_per_batch, actual_jobs)

    if len(batches) <= 1:
        result = _ingest_cases_to_parquet(task_args[0])
        _merge_batch_stats(result, stats)
    else:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=actual_jobs) as executor:
            futures = [executor.submit(_ingest_cases_to_parquet, arg) for arg in task_args]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Parsing case bundles", unit="batch"):
                try:
                    _merge_batch_stats(future.result(), stats)
                except Exception as exc:
                    LOGGER.warning("Phase 1 worker failed: %s", exc)

    # Collect source_paths and atom cache map from temp files
    atom_cache_map: dict[str, str] = {}
    for sources_file in sorted(tmp_phase1.glob("temp_*_sources.txt")):
        for line in sources_file.read_text(encoding="utf-8").splitlines():
            sp = line.strip()
            if sp:
                all_source_paths.add(sp)
        sources_file.unlink()
    for atoms_file in sorted(tmp_phase1.glob("temp_*_atoms.jsonl")):
        for line in atoms_file.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line.strip())
            sp = entry.get("source_path", "")
            ad = entry.get("atoms_dir", "")
            if sp and ad and sp not in atom_cache_map:
                atom_cache_map[sp] = ad
        atoms_file.unlink()

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
                if not p.exists():
                    continue
                pf = pq.ParquetFile(p)
                schema_names = set(pf.schema_arrow.names)
                if "source_path" not in schema_names:
                    continue
                for rg_idx in range(pf.metadata.num_row_groups):
                    cols = ["source_path"]
                    if "assembly_id" in schema_names:
                        cols.append("assembly_id")
                    tbl = pf.read_row_group(rg_idx, columns=cols)
                    sp_list = tbl.column("source_path").to_pylist()
                    aid_list = tbl.column("assembly_id").to_pylist() if "assembly_id" in schema_names else [None] * len(sp_list)
                    for sp, aid in zip(sp_list, aid_list):
                        if sp and sp in cache_pairs:
                            cache_pairs[sp].add(aid if aid else None)
        except ImportError:
            pass

        tasks = [(sp, sorted(aids, key=lambda x: x or ""), cif_files_directory)
                 for sp, aids in sorted(cache_pairs.items())]

        num_workers = max(1, min(actual_jobs, len(tasks)))
        LOGGER.info("Dispatching %d source files to %d worker processes", len(tasks), num_workers)

        tmp_phase2 = get_fast_temp_dir("phase2")
        try:
            if len(tasks) <= 1:
                chunk_tasks = [(sp, aids, cfd) for sp, aids, cfd in tasks]
                result = _cache_sources_to_temp((chunk_tasks, tmp_phase2, 0, atom_cache_map))
                _count_cif_entries(result, cif_stats)
            else:
                # Split tasks into num_workers groups; each group → one chunk file
                chunk_size = max(1, len(tasks) // num_workers)
                grouped: list[list[tuple]] = [[] for _ in range(num_workers)]
                for i, (sp, aids, cfd) in enumerate(tasks):
                    grouped[min(i // chunk_size, num_workers - 1)].append((sp, aids, cfd))
                grouped = [g for g in grouped if g]  # drop empty groups
                task_args = [(chunk_tasks, tmp_phase2, wid, atom_cache_map)
                             for wid, chunk_tasks in enumerate(grouped)]
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(_cache_sources_to_temp, arg) for arg in task_args]
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc="Caching mmCIF structures", unit="source"):
                        try:
                            _count_cif_entries(future.result(), cif_stats)
                        except Exception as exc:
                            LOGGER.warning("Phase 2 worker failed: %s", exc)

            # Move temp files to chunked storage — no concatenation needed.
            # Each worker's output becomes an independent chunk file.
            temp_bins = sorted(tmp_phase2.glob("temp_*.bin"))
            temp_idxs = sorted(tmp_phase2.glob("temp_*.idx"))
            if temp_bins:
                coords_dir = prep_dir / "cif_coords"
                coords_dir.mkdir(parents=True, exist_ok=True)
                for chunk_id, (tbin, tidx) in enumerate(zip(temp_bins, temp_idxs)):
                    shutil.move(str(tbin), str(coords_dir / f"chunk_{chunk_id}.bin"))
                    shutil.move(str(tidx), str(coords_dir / f"chunk_{chunk_id}.idx"))

        finally:
            shutil.rmtree(tmp_phase2, ignore_errors=True)

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


def load_cif_coords_index(prep_dir: str | Path) -> dict[str, tuple[Path, int, int]] | None:
    """Load all chunk idx files from ``cif_coords/``.

    Returns ``{blob_key: (chunk_bin_path, offset, length)}``, or None if the
    chunk directory does not exist.  Also accepts legacy single-file
    ``cif_coords.idx`` for backward compatibility.
    """
    prep_dir = Path(prep_dir)
    coords_dir = prep_dir / "cif_coords"
    index: dict[str, tuple[Path, int, int]] = {}

    # New chunked format
    if coords_dir.is_dir():
        for idx_path in sorted(coords_dir.glob("chunk_*.idx")):
            chunk_id = idx_path.stem.replace("chunk_", "")
            bin_path = coords_dir / f"chunk_{chunk_id}.bin"
            if not bin_path.exists():
                continue
            with idx_path.open("rb") as fh:
                while True:
                    entry = fh.read(_IDX_ENTRY_SIZE)
                    if len(entry) < _IDX_ENTRY_SIZE:
                        break
                    source_hash, offset, length = _IDX_ENTRY_STRUCT.unpack(entry)
                    key = source_hash.decode("ascii").rstrip("\0")
                    index[key] = (bin_path, offset, length)
        if index:
            return index

    # Legacy single-file format
    idx_path = prep_dir / "cif_coords.idx"
    bin_path = prep_dir / "cif_coords.bin"
    if idx_path.exists() and bin_path.exists():
        with idx_path.open("rb") as fh:
            while True:
                entry = fh.read(_IDX_ENTRY_SIZE)
                if len(entry) < _IDX_ENTRY_SIZE:
                    break
                source_hash, offset, length = _IDX_ENTRY_STRUCT.unpack(entry)
                key = source_hash.decode("ascii").rstrip("\0")
                index[key] = (bin_path, offset, length)
        if index:
            return index

    return None


def load_cif_from_prep(
    prep_dir: str | Path,
    source_path: str,
    assembly_id: str | None = None,
    *,
    index: dict[str, tuple[Path, int, int]] | None = None,
    mmap: Any = None,
) -> dict[str, Any] | None:
    """Fetch a cached AtomArray + metadata from the binary index.

    Returns a dict with ``atom_array``, ``quality``, ``chain_ops`` keys,
    or None if not cached.
    """
    idx = index or load_cif_coords_index(prep_dir)
    if idx is None:
        return None
    blob_key = _blob_index_key(source_path, assembly_id)
    entry = idx.get(blob_key)
    if entry is None:
        return None

    bin_path, offset, length = entry
    if mmap is not None and hasattr(mmap, "seek"):
        # Legacy: single mmap over old cif_coords.bin
        data = mmap[offset:offset + length]
    else:
        with bin_path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(length)
    try:
        return pickle.loads(_decompress_blob(data))
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
