from __future__ import annotations

import gzip
import itertools
import logging
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import re
from typing import Any, Iterator, TextIO

import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_assembly, get_structure, list_assemblies

from cif_parse.annotate import analyze_antibody_sequence, analyze_immune_sequence
from cif_parse.constants import (
    STANDARD_AMINO_ACIDS,
    STANDARD_DNA_NUCLEOTIDES,
    STANDARD_RNA_NUCLEOTIDES,
)
from cif_parse.models import ChainRecord, StructureSummary
from cif_parse.utils import filter_atom_array_for_analysis

LOGGER = logging.getLogger(__name__)


def _is_empty_atom_array_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and str(exc) == "Array must contain at least one element"


def get_structure_with_altloc_fallback(
    cif_file: CIFFile,
    *,
    model: int | None = 1,
    use_author_fields: bool = False,
):
    """Read coordinates, retrying with altloc='all' for nonstandard alt IDs."""

    try:
        return get_structure(
            cif_file,
            model=model,
            use_author_fields=use_author_fields,
        )
    except ValueError as exc:
        if not _is_empty_atom_array_error(exc):
            raise
        LOGGER.warning(
            "Biotite returned an empty atom array with default altloc handling; "
            "retrying with altloc='all'"
        )
        try:
            return get_structure(
                cif_file,
                model=model,
                use_author_fields=use_author_fields,
                altloc="all",
            )
        except Exception as fallback_exc:
            raise exc from fallback_exc


def get_assembly_with_altloc_fallback(
    cif_file: CIFFile,
    *,
    assembly_id: str,
    model: int | None = 1,
    use_author_fields: bool = False,
):
    """Read assembly coordinates, retrying with altloc='all' if needed."""

    try:
        return get_assembly(
            cif_file,
            assembly_id=assembly_id,
            model=model,
            use_author_fields=use_author_fields,
        )
    except ValueError as exc:
        if not _is_empty_atom_array_error(exc):
            raise
        LOGGER.warning(
            "Biotite returned an empty assembly atom array with default altloc "
            "handling for assembly %s; retrying with altloc='all'",
            assembly_id,
        )
        try:
            return get_assembly(
                cif_file,
                assembly_id=assembly_id,
                model=model,
                use_author_fields=use_author_fields,
                altloc="all",
            )
        except Exception as fallback_exc:
            raise exc from fallback_exc


def _append_warning_detail(record: ChainRecord, warning_code: str, detail: dict[str, Any]) -> None:
    warning_details = record.features.setdefault("coverage_warning_details", {})
    if not isinstance(warning_details, dict):
        warning_details = {}
        record.features["coverage_warning_details"] = warning_details
    warning_details[warning_code] = detail


def read_structure_preflight(path: str | Path, *, cif_file: CIFFile | None = None) -> dict[str, Any]:
    """Read lightweight per-structure chain counts without building atom arrays."""

    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)
    pdb_id = _infer_pdb_id(cif_path, cif_file)
    entity_map = _build_entity_map(cif_file)

    polymer_chain_count = 0
    max_polymer_chain_length = 0
    polymer_chain_lengths: dict[str, int] = {}
    for row in _category_rows(cif_file, "struct_asym"):
        label_asym_id = row.get("id")
        entity_id = row.get("entity_id")
        if not label_asym_id or not entity_id:
            continue
        entity = entity_map.get(entity_id, {})
        if str(entity.get("entity_type") or "").lower() != "polymer":
            continue
        sequence = entity.get("sequence")
        monomer_ids = list(entity.get("monomer_ids") or [])
        length = len(sequence or "") or len(monomer_ids)
        polymer_chain_count += 1
        polymer_chain_lengths[label_asym_id] = length
        if length > max_polymer_chain_length:
            max_polymer_chain_length = length

    return {
        "pdb_id": pdb_id,
        "polymer_chain_count": polymer_chain_count,
        "max_polymer_chain_length": max_polymer_chain_length,
        "polymer_chain_lengths": polymer_chain_lengths,
    }


def preflight_assembly_atom_counts(path: str | Path, *, cif_file: CIFFile | None = None) -> dict[str, int]:
    """Estimate atom count per assembly from chain lengths (fast, no atom_site read).

    Uses ``struct_asym`` + ``entity`` + ``entity_poly`` for per-chain atom estimates
    (residue count × atoms-per-residue heuristic) and ``pdbx_struct_assembly_gen`` for
    assembly composition.  Returns ``{assembly_id: atom_count}``.
    """
    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)

    # Per-chain atom estimate from entity tables (avoids reading atom_site).
    _ATOMS_PER_RES = {"protein": 8, "rna": 20, "dna": 20, "unknown": 5}
    chain_est: dict[str, int] = {}
    entity_info: dict[str, tuple[str, int]] = {}
    poly_seq_lengths: dict[str, int] = defaultdict(int)
    for row in _category_rows(cif_file, "entity_poly_seq"):
        eid = row.get("entity_id")
        monomer_id = row.get("mon_id")
        if eid and monomer_id:
            poly_seq_lengths[eid] += 1
    for row in _category_rows(cif_file, "entity"):
        eid = row.get("id")
        etype = (row.get("type") or "").lower()
        if eid:
            entity_info[eid] = (etype, 0)
    for row in _category_rows(cif_file, "entity_poly"):
        eid = row.get("entity_id")
        if not eid:
            continue
        seq = _sanitize_sequence(row.get("pdbx_seq_one_letter_code_can"))
        length = len(seq) if seq else int(poly_seq_lengths.get(eid, 0))
        ptype = (row.get("type") or "").lower()
        etype = entity_info.get(eid, ("polymer", 0))[0]
        if "deoxyribo" in ptype:
            apx = _ATOMS_PER_RES["dna"]
        elif "ribo" in ptype:
            apx = _ATOMS_PER_RES["rna"]
        elif etype == "polymer":
            apx = _ATOMS_PER_RES["protein"]
        else:
            apx = _ATOMS_PER_RES["unknown"]
        entity_info[eid] = (etype, length * apx)
    for row in _category_rows(cif_file, "struct_asym"):
        aid = row.get("id")
        eid = row.get("entity_id")
        if aid and eid and eid in entity_info:
            chain_est[aid] = entity_info[eid][1]

    assembly_atoms: dict[str, int] = {}
    for row in _category_rows(cif_file, "pdbx_struct_assembly_gen"):
        assembly_id = row.get("assembly_id")
        asym_id_list = row.get("asym_id_list")
        oper_expr = row.get("oper_expression")
        if not assembly_id or not asym_id_list:
            continue
        copies = count_oper_expression_copies(oper_expr)
        total = 0
        for asym_id in (a.strip() for a in asym_id_list.split(",") if a.strip()):
            total += chain_est.get(asym_id, 0) * copies
        if total > 0:
            assembly_atoms[assembly_id] = total

    return assembly_atoms


@contextmanager
def _open_cif_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("rt", encoding="utf-8") as handle:
            yield handle


def _default_block_name(cif_file: CIFFile) -> str:
    block_names = list(cif_file.keys())
    if not block_names:
        raise ValueError("CIF file does not contain any data blocks")
    return str(block_names[0])


def _normalize_cif_value(value: object) -> str | None:
    normalized = str(value).strip()
    if normalized in ("", ".", "?"):
        return None
    return normalized


def _ordered_append(values: list[str], value: str | None) -> None:
    if value and value not in values:
        values.append(value)


def _append_ordered_value(values: list[str], value: str | None) -> None:
    if value:
        values.append(value)


def _get_first_value(cif_file: CIFFile, category_name: str, column_name: str) -> str | None:
    block = cif_file[_default_block_name(cif_file)]
    if category_name not in block:
        return None
    category = block[category_name]
    if column_name not in category:
        return None
    values = category[column_name].as_array().tolist()
    if not values:
        return None
    return _normalize_cif_value(values[0])


def _infer_pdb_id(path: Path, cif_file: CIFFile) -> str:
    entry_id = _get_first_value(cif_file, "entry", "id")
    filename_pdb_id = _infer_pdb_id_from_filename(path)
    if entry_id and entry_id.lower() not in {"xxxx", "unknown", "none"}:
        return entry_id.lower()
    if filename_pdb_id is not None:
        return filename_pdb_id
    if entry_id:
        return entry_id.lower()

    name = path.name
    for suffix in (".cif.gz", ".bcif.gz", ".cif", ".bcif"):
        if name.endswith(suffix):
            return name[: -len(suffix)].lower()
    return path.stem.lower()


def _infer_pdb_id_from_filename(path: Path) -> str | None:
    name = path.name.lower()
    for suffix in (".cif.gz", ".bcif.gz", ".cif", ".bcif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    match = re.match(r"^([0-9][a-z0-9]{3})(?:[-_].*)?$", name)
    if match:
        return match.group(1)
    return None


def _count_category_rows(cif_file: CIFFile, category_name: str, column_name: str) -> int:
    block = cif_file[_default_block_name(cif_file)]
    if category_name not in block:
        return 0
    category = block[category_name]
    if column_name not in category:
        return 0
    return len(category[column_name].as_array())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", ".", "?", "None", "none", "nan", "NaN"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _category_column_values(cif_file: CIFFile, category_name: str, column_name: str) -> list[str]:
    block = cif_file[_default_block_name(cif_file)]
    if category_name not in block:
        return []
    category = block[category_name]
    if column_name not in category:
        return []
    return [str(value) for value in category[column_name].as_array().tolist()]


def _pick_first_metadata_value(values: list[str]) -> str:
    for value in values:
        text = str(value).strip()
        if text and text not in {".", "?"}:
            return text
    return ""


def read_case_metadata(path: str | Path, *, cif_file: CIFFile | None = None) -> dict[str, Any]:
    """Read cheap per-entry metadata from an already parsed CIF when available."""

    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)
    entity_map = _build_entity_map(cif_file)

    asym_ids: list[str] = []
    polymer_count = 0
    for row in _category_rows(cif_file, "struct_asym"):
        asym_id = row.get("id")
        if asym_id:
            asym_ids.append(asym_id)
        entity_id = row.get("entity_id")
        if entity_id:
            entity = entity_map.get(entity_id, {})
            if str(entity.get("entity_type", "")).lower() == "polymer":
                polymer_count += 1

    num_assemblies = len(_category_rows(cif_file, "pdbx_struct_assembly_gen"))
    method = _pick_first_metadata_value(
        [
            *_category_column_values(cif_file, "exptl", "method"),
            *_category_column_values(cif_file, "exptl_crystal", "method"),
        ]
    )
    resolution_values = [
        *_category_column_values(cif_file, "refine", "ls_d_res_high"),
        *_category_column_values(cif_file, "reflns", "d_resolution_high"),
        *_category_column_values(cif_file, "em_3d_reconstruction", "resolution"),
    ]
    resolutions = sorted(value for value in (_safe_float(item) for item in resolution_values) if value is not None)
    release_dates = [
        date
        for date in _category_column_values(cif_file, "pdbx_audit_revision_history", "revision_date")
        if str(date).strip() not in {"", ".", "?"}
    ]

    return {
        "experimental_method": method,
        "resolution": round(resolutions[0], 2) if resolutions else "",
        "release_date": min(release_dates) if release_dates else "",
        "num_polymer_chains": polymer_count,
        "num_asym_ids": len(asym_ids),
        "num_assemblies": num_assemblies,
    }


def _category_rows(cif_file: CIFFile, category_name: str) -> list[dict[str, str | None]]:
    block = cif_file[_default_block_name(cif_file)]
    if category_name not in block:
        return []

    category = block[category_name]
    column_names = list(category.keys())
    if not column_names:
        return []

    arrays = {name: category[name].as_array().tolist() for name in column_names}
    num_rows = len(arrays[column_names[0]])
    rows: list[dict[str, str | None]] = []
    for row_index in range(num_rows):
        row = {
            column_name: _normalize_cif_value(arrays[column_name][row_index])
            for column_name in column_names
        }
        rows.append(row)
    return rows


def _expand_operation_group(expression: str) -> list[str]:
    values: list[str] = []
    for chunk in expression.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            if start.isdigit() and end.isdigit():
                low = int(start)
                high = int(end)
                step = 1 if high >= low else -1
                values.extend(str(value) for value in range(low, high + step, step))
                continue
        values.append(token)
    return values


def expand_oper_expression(oper_expression: str | None) -> list[str]:
    normalized = _normalize_cif_value(oper_expression)
    if normalized is None:
        return ["1"]

    compact = normalized.replace(" ", "")
    groups = re.findall(r"\(([^()]*)\)", compact)
    if groups:
        combinations = [
            "x".join(parts)
            for parts in itertools.product(*[_expand_operation_group(group) for group in groups])
        ]
        return list(dict.fromkeys(combinations))
    return list(dict.fromkeys(_expand_operation_group(compact)))


def count_oper_expression_copies(oper_expression: str | None) -> int:
    return len(expand_oper_expression(oper_expression))


def _assembly_sort_key(assembly_id: str) -> tuple[int, int | str]:
    if assembly_id.isdigit():
        return (0, int(assembly_id))
    return (1, assembly_id)


def read_available_assembly_ids(path: str | Path, *, cif_file: CIFFile | None = None) -> list[str]:
    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)
    block = cif_file[_default_block_name(cif_file)]
    if "pdbx_struct_assembly" not in block:
        return []
    try:
        return sorted(
            [str(assembly_id) for assembly_id in list_assemblies(cif_file)],
            key=_assembly_sort_key,
        )
    except Exception:
        return []


def select_largest_polymer_assembly_id(cif_file: CIFFile) -> str | None:
    rows = _category_rows(cif_file, "pdbx_struct_assembly_gen")
    available_assembly_ids = sorted(
        {
            row["assembly_id"]
            for row in rows
            if row.get("assembly_id") is not None
        },
        key=_assembly_sort_key,
    )
    if not available_assembly_ids:
        return None

    entity_map = _build_entity_map(cif_file)
    polymer_asym_ids = {
        row["id"]
        for row in _category_rows(cif_file, "struct_asym")
        if row.get("id")
        and row.get("entity_id")
        and str(entity_map.get(str(row["entity_id"]), {}).get("entity_type") or "").lower() == "polymer"
    }
    if not polymer_asym_ids:
        return available_assembly_ids[0]

    polymer_counts = {assembly_id: 0 for assembly_id in available_assembly_ids}
    for row in rows:
        assembly_id = row.get("assembly_id")
        asym_id_list = row.get("asym_id_list")
        if assembly_id is None or asym_id_list is None:
            continue
        polymer_member_count = sum(
            1
            for asym_id in (part.strip() for part in asym_id_list.split(","))
            if asym_id and asym_id in polymer_asym_ids
        )
        polymer_counts[assembly_id] += polymer_member_count * count_oper_expression_copies(row.get("oper_expression"))

    return sorted(
        available_assembly_ids,
        key=lambda assembly_id: (-polymer_counts.get(assembly_id, 0), _assembly_sort_key(assembly_id)),
    )[0]


def read_assembly_chain_operations(
    path: str | Path,
    assembly_id: str | None = None,
    *,
    cif_file: CIFFile | None = None,
) -> tuple[str | None, dict[str, list[str]]]:
    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)
    block = cif_file[_default_block_name(cif_file)]
    if "pdbx_struct_assembly_gen" not in block:
        return assembly_id, {}

    rows = _category_rows(cif_file, "pdbx_struct_assembly_gen")
    available_assembly_ids = [
        row["assembly_id"]
        for row in rows
        if row.get("assembly_id") is not None
    ]
    selected_assembly_id = assembly_id or select_largest_polymer_assembly_id(cif_file)
    if selected_assembly_id is None:
        return None, {}

    chain_operations: dict[str, list[str]] = {}
    for row in rows:
        if row.get("assembly_id") != selected_assembly_id:
            continue
        operation_ids = expand_oper_expression(row.get("oper_expression"))
        asym_id_list = row.get("asym_id_list")
        if asym_id_list is None:
            continue
        for asym_id in [part.strip() for part in asym_id_list.split(",") if part.strip()]:
            existing = chain_operations.setdefault(asym_id, [])
            for operation_id in operation_ids:
                if operation_id not in existing:
                    existing.append(operation_id)
    return selected_assembly_id, chain_operations


def read_assembly_copy_numbers(
    path: str | Path,
    assembly_id: str | None = None,
    *,
    cif_file: CIFFile | None = None,
) -> tuple[str | None, dict[str, int]]:
    selected_assembly_id, chain_operations = read_assembly_chain_operations(
        path,
        assembly_id=assembly_id,
        cif_file=cif_file,
    )
    copy_numbers = {
        asym_id: len(operation_ids)
        for asym_id, operation_ids in chain_operations.items()
    }
    return selected_assembly_id, copy_numbers


def _sanitize_sequence(sequence: str | None) -> str | None:
    if sequence is None:
        return None
    return "".join(sequence.split())


def _build_chem_comp_map(cif_file: CIFFile) -> dict[str, dict[str, str | None]]:
    chem_comp_map: dict[str, dict[str, str | None]] = {}
    for row in _category_rows(cif_file, "chem_comp"):
        comp_id = row.get("id")
        if not comp_id:
            continue
        chem_comp_map[comp_id] = {
            "id": comp_id,
            "type": row.get("type"),
            "name": row.get("name"),
            "formula": row.get("formula"),
        }
    return chem_comp_map


def _build_entity_map(cif_file: CIFFile) -> dict[str, dict[str, Any]]:
    entity_map: dict[str, dict[str, Any]] = {}
    for row in _category_rows(cif_file, "entity"):
        entity_id = row.get("id")
        if not entity_id:
            continue
        entity_map[entity_id] = {
            "entity_id": entity_id,
            "entity_type": row.get("type") or "unknown",
            "entity_description": row.get("pdbx_description"),
            "polymer_type": None,
            "sequence": None,
            "monomer_ids": [],
        }

    for row in _category_rows(cif_file, "entity_poly"):
        entity_id = row.get("entity_id")
        if not entity_id:
            continue
        entity = entity_map.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": "polymer",
                "entity_description": None,
                "polymer_type": None,
                "sequence": None,
                "monomer_ids": [],
            },
        )
        entity["polymer_type"] = row.get("type")
        entity["sequence"] = _sanitize_sequence(row.get("pdbx_seq_one_letter_code_can"))

    for row in _category_rows(cif_file, "entity_poly_seq"):
        entity_id = row.get("entity_id")
        monomer_id = row.get("mon_id")
        if not entity_id:
            continue
        entity = entity_map.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": "polymer",
                "entity_description": None,
                "polymer_type": None,
                "sequence": None,
                "monomer_ids": [],
            },
        )
        _append_ordered_value(entity["monomer_ids"], monomer_id)

    for row in _category_rows(cif_file, "pdbx_entity_nonpoly"):
        entity_id = row.get("entity_id")
        monomer_id = row.get("comp_id")
        if not entity_id:
            continue
        entity = entity_map.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": "non-polymer",
                "entity_description": None,
                "polymer_type": None,
                "sequence": None,
                "monomer_ids": [],
            },
        )
        _ordered_append(entity["monomer_ids"], monomer_id)

    return entity_map


def _build_poly_seq_rows(cif_file: CIFFile) -> dict[str, list[dict[str, Any]]]:
    poly_seq_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _category_rows(cif_file, "entity_poly_seq"):
        entity_id = row.get("entity_id")
        num = row.get("num")
        monomer_id = row.get("mon_id")
        if not entity_id or not num:
            continue
        try:
            label_seq_id = int(num)
        except ValueError:
            continue
        poly_seq_rows[entity_id].append(
            {
                "label_seq_id": label_seq_id,
                "monomer_id": monomer_id,
            }
        )
    for entity_id in poly_seq_rows:
        poly_seq_rows[entity_id].sort(key=lambda row: row["label_seq_id"])
    return dict(poly_seq_rows)


def _build_poly_scheme_maps(
    cif_file: CIFFile,
) -> dict[str, dict[int, dict[str, Any]]]:
    scheme_maps: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in _category_rows(cif_file, "pdbx_poly_seq_scheme"):
        label_asym_id = row.get("asym_id")
        seq_id = row.get("seq_id")
        if not label_asym_id or not seq_id:
            continue
        try:
            label_seq_id = int(seq_id)
        except ValueError:
            continue
        scheme_maps[label_asym_id][label_seq_id] = {
            "auth_seq_id": row.get("auth_seq_num") or row.get("auth_seq_id") or row.get("pdb_seq_num"),
            "pdb_strand_id": row.get("pdb_strand_id"),
            "monomer_id": row.get("mon_id") or row.get("pdb_mon_id") or row.get("auth_mon_id"),
        }
    return dict(scheme_maps)


def _build_atom_site_stats(cif_file: CIFFile) -> dict[str, dict[str, Any]]:
    block = cif_file[_default_block_name(cif_file)]
    if "atom_site" not in block:
        return {}

    category = block["atom_site"]
    arrays = {
        column_name: category[column_name].as_array().tolist()
        for column_name in (
            "label_asym_id",
            "auth_asym_id",
            "label_entity_id",
            "label_comp_id",
            "label_seq_id",
            "auth_seq_id",
            "pdbx_PDB_ins_code",
            "group_PDB",
        )
        if column_name in category
    }
    if "label_asym_id" not in arrays:
        return {}

    num_rows = len(arrays["label_asym_id"])
    default_values = [None] * num_rows
    auth_asym_ids = arrays.get("auth_asym_id", default_values)
    entity_ids = arrays.get("label_entity_id", default_values)
    component_ids = arrays.get("label_comp_id", default_values)
    label_seq_ids = arrays.get("label_seq_id", default_values)
    auth_seq_ids = arrays.get("auth_seq_id", default_values)
    ins_codes = arrays.get("pdbx_PDB_ins_code", default_values)
    group_pdb_values = arrays.get("group_PDB", default_values)

    stats: dict[str, dict[str, Any]] = {}
    residue_seen: dict[str, set[tuple[str | None, ...]]] = defaultdict(set)
    for row_index in range(num_rows):
        label_asym_id = _normalize_cif_value(arrays["label_asym_id"][row_index])
        if not label_asym_id:
            continue

        auth_asym_id = _normalize_cif_value(auth_asym_ids[row_index])
        entity_id = _normalize_cif_value(entity_ids[row_index])
        component_id = _normalize_cif_value(component_ids[row_index])
        label_seq_id = _normalize_cif_value(label_seq_ids[row_index])
        auth_seq_id = _normalize_cif_value(auth_seq_ids[row_index])
        ins_code = _normalize_cif_value(ins_codes[row_index])
        group_pdb = _normalize_cif_value(group_pdb_values[row_index])

        chain_stats = stats.setdefault(
            label_asym_id,
            {
                "entity_id": entity_id,
                "atom_count": 0,
                "residue_count": 0,
                "component_ids": [],
                "residue_monomer_ids": [],
                "resolved_label_seq_ids": [],
                "resolved_auth_seq_ids": {},
                "residue_details": [],
                "auth_asym_ids": set(),
            },
        )
        chain_stats["atom_count"] += 1
        if auth_asym_id:
            chain_stats["auth_asym_ids"].add(auth_asym_id)
        _ordered_append(chain_stats["component_ids"], component_id)

        resolved_label_seq_id: int | None = None
        residue_key = (label_seq_id, auth_seq_id, ins_code, component_id, group_pdb)
        if residue_key not in residue_seen[label_asym_id]:
            residue_seen[label_asym_id].add(residue_key)
            chain_stats["residue_count"] += 1
            _append_ordered_value(chain_stats["residue_monomer_ids"], component_id)
            if label_seq_id is not None:
                try:
                    resolved_label_seq_id = int(label_seq_id)
                except ValueError:
                    resolved_label_seq_id = None
                if resolved_label_seq_id is not None:
                    chain_stats["resolved_label_seq_ids"].append(resolved_label_seq_id)
                    chain_stats["resolved_auth_seq_ids"][resolved_label_seq_id] = auth_seq_id
            chain_stats["residue_details"].append(
                {
                    "label_seq_id": resolved_label_seq_id,
                    "auth_seq_id": auth_seq_id,
                    "monomer_id": component_id,
                    "group_pdb": group_pdb,
                    "ins_code": ins_code,
                }
            )

    return stats


def _scheme_auth_id_map(
    cif_file: CIFFile,
    atom_site_stats: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    scheme_categories = [
        "pdbx_poly_seq_scheme",
        "pdbx_nonpoly_scheme",
        "pdbx_branch_scheme",
    ]
    auth_candidates: dict[str, set[str]] = {}
    for category_name in scheme_categories:
        for row in _category_rows(cif_file, category_name):
            label_asym_id = row.get("asym_id")
            auth_asym_id = row.get("pdb_strand_id")
            if not label_asym_id or not auth_asym_id:
                continue
            auth_candidates.setdefault(label_asym_id, set()).add(auth_asym_id)

    for label_asym_id, chain_stats in atom_site_stats.items():
        auth_candidates.setdefault(label_asym_id, set()).update(chain_stats["auth_asym_ids"])

    auth_map: dict[str, str] = {}
    warnings: dict[str, list[str]] = {}
    for label_asym_id, auth_ids in auth_candidates.items():
        if len(auth_ids) == 1:
            auth_map[label_asym_id] = next(iter(auth_ids))
            continue

        ordered_auth_ids = sorted(auth_ids)
        auth_map[label_asym_id] = ordered_auth_ids[0]
        warnings[label_asym_id] = [
            "multiple auth_asym_id values found; using the first sorted value"
        ]
    return auth_map, warnings


def _is_metal_component(comp_id: str | None, chem_comp_map: dict[str, dict[str, str | None]]) -> bool:
    if not comp_id:
        return False
    component = chem_comp_map.get(comp_id, {})
    name = (component.get("name") or "").upper()
    formula = (component.get("formula") or "").strip()
    if "ION" in name and comp_id.upper() != "HOH":
        return True

    formula_tokens = [token for token in formula.split() if token and token[0].isalpha()]
    if len(formula_tokens) == 1 and "C" not in formula.upper():
        return True
    return False


def _description_based_chain_type(
    description: str | None,
    length: int,
) -> tuple[str | None, str | None, list[str], dict[str, Any]]:
    description_lower = (description or "").lower()
    sources: list[str] = []
    features: dict[str, Any] = {}

    if not description_lower:
        return None, None, sources, features

    if "beta-2-microglobulin" in description_lower or "beta 2 microglobulin" in description_lower:
        sources.append("description_heuristic")
        return "beta2m or auxiliary immune chain", "beta2m", sources, features

    mhc_markers = ("major histocompatibility", "histocompatibility antigen", "mhc", "hla-")
    if any(marker in description_lower for marker in mhc_markers):
        sources.append("description_heuristic")
        mhc_class = _infer_mhc_class_from_description(description_lower)
        features["mhc_class"] = mhc_class
        features["mhc_role"] = _infer_mhc_role_from_description(description_lower, mhc_class)
        return "MHC heavy chain", f"class_{mhc_class.lower()}", sources, features

    if "t cell receptor" in description_lower or "tcr" in description_lower:
        sources.append("description_heuristic")
        subtype = "unknown"
        if "alpha" in description_lower:
            subtype = "alpha"
        elif "beta" in description_lower:
            subtype = "beta"
        elif "gamma" in description_lower:
            subtype = "gamma"
        elif "delta" in description_lower:
            subtype = "delta"
        return "TCR chain", subtype, sources, features

    return None, None, sources, features


def _infer_mhc_class_from_description(description_lower: str) -> str:
    if re.search(r"\bclass[\s-]*ii\b", description_lower):
        return "II"
    if re.search(r"\bclass[\s-]*i\b", description_lower):
        return "I"

    class_ii_markers = (
        "hla-dq",
        "hla-dr",
        "hla-dp",
        "h2-ia",
        "h2-ie",
    )
    if any(marker in description_lower for marker in class_ii_markers):
        return "II"

    class_i_markers = (
        "hla-a",
        "hla-b",
        "hla-c",
        "hla-e",
        "hla-f",
        "hla-g",
        "beta-2-microglobulin-free",
    )
    if any(marker in description_lower for marker in class_i_markers):
        return "I"

    return "unknown"


def _infer_mhc_role_from_description(description_lower: str, mhc_class: str) -> str:
    if mhc_class == "I":
        return "class_i_heavy"
    if mhc_class != "II":
        return "unknown"

    if re.search(r"\balpha\b", description_lower):
        return "class_ii_alpha"
    if re.search(r"\bbeta\b", description_lower):
        return "class_ii_beta"

    class_ii_alpha_markers = (
        "hla-dra",
        "hla-dqa1",
        "hla-dpa1",
        "dq alpha",
        "dr alpha",
        "dp alpha",
        "dq-alpha",
        "dr-alpha",
        "dp-alpha",
        "alpha 1 chain",
        "a-b alpha chain",
    )
    if any(marker in description_lower for marker in class_ii_alpha_markers):
        return "class_ii_alpha"

    class_ii_beta_markers = (
        "hla-drb",
        "hla-dqb1",
        "hla-dpb1",
        "dq beta",
        "dr beta",
        "dp beta",
        "dq-beta",
        "dr-beta",
        "dp-beta",
        "beta 1 chain",
        "a beta chain",
    )
    if any(marker in description_lower for marker in class_ii_beta_markers):
        return "class_ii_beta"

    return "unknown"


def _is_standard_polymer_monomer(chain_type: str, monomer_id: str | None) -> bool:
    if not monomer_id:
        return False
    monomer_id = monomer_id.upper()
    if chain_type in {
        "antibody heavy chain",
        "antibody light chain",
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
        "other protein chain",
    }:
        return monomer_id in STANDARD_AMINO_ACIDS
    if chain_type == "DNA chain":
        return monomer_id in STANDARD_DNA_NUCLEOTIDES
    if chain_type == "RNA chain":
        return monomer_id in STANDARD_RNA_NUCLEOTIDES
    if chain_type == "other nucleic acid chain":
        return monomer_id in STANDARD_DNA_NUCLEOTIDES | STANDARD_RNA_NUCLEOTIDES
    return False


def _build_segments(
    label_seq_ids: list[int],
    auth_seq_lookup: dict[int, dict[str, Any]] | dict[int, str | None],
) -> list[dict[str, Any]]:
    if not label_seq_ids:
        return []
    sorted_ids = sorted(set(label_seq_ids))
    segments: list[dict[str, Any]] = []
    start = prev = sorted_ids[0]
    for current in sorted_ids[1:]:
        if current == prev + 1:
            prev = current
            continue
        segments.append(
            _segment_record(start, prev, auth_seq_lookup)
        )
        start = prev = current
    segments.append(_segment_record(start, prev, auth_seq_lookup))
    return segments


def _segment_record(
    start: int,
    end: int,
    auth_seq_lookup: dict[int, dict[str, Any]] | dict[int, str | None],
) -> dict[str, Any]:
    start_info = auth_seq_lookup.get(start)
    end_info = auth_seq_lookup.get(end)
    if isinstance(start_info, dict):
        start_auth_seq_id = start_info.get("auth_seq_id")
    else:
        start_auth_seq_id = start_info
    if isinstance(end_info, dict):
        end_auth_seq_id = end_info.get("auth_seq_id")
    else:
        end_auth_seq_id = end_info
    return {
        "label_seq_start": start,
        "label_seq_end": end,
        "auth_seq_start": start_auth_seq_id,
        "auth_seq_end": end_auth_seq_id,
        "length": end - start + 1,
    }


def _build_special_residue_details(
    chain_type: str,
    entity_poly_rows: list[dict[str, Any]],
    scheme_map: dict[int, dict[str, Any]],
    resolved_label_seq_ids: set[int],
    chem_comp_map: dict[str, dict[str, str | None]],
) -> list[dict[str, Any]]:
    special_rows: list[dict[str, Any]] = []
    for row in entity_poly_rows:
        label_seq_id = row["label_seq_id"]
        monomer_id = row.get("monomer_id")
        if _is_standard_polymer_monomer(chain_type, monomer_id):
            continue
        chem_comp = chem_comp_map.get(monomer_id or "", {})
        scheme_row = scheme_map.get(label_seq_id, {})
        special_rows.append(
            {
                "label_seq_id": label_seq_id,
                "auth_seq_id": scheme_row.get("auth_seq_id"),
                "monomer_id": monomer_id,
                "chem_comp_type": chem_comp.get("type"),
                "chem_comp_name": chem_comp.get("name"),
                "resolved": label_seq_id in resolved_label_seq_ids,
            }
        )
    return special_rows


def _build_special_component_details(
    component_ids: list[str],
    chem_comp_map: dict[str, dict[str, str | None]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for component_id in component_ids:
        chem_comp = chem_comp_map.get(component_id, {})
        details.append(
            {
                "component_id": component_id,
                "chem_comp_type": chem_comp.get("type"),
                "chem_comp_name": chem_comp.get("name"),
                "formula": chem_comp.get("formula"),
            }
        )
    return details


def _confidence_label_to_score(confidence: object) -> float:
    if isinstance(confidence, (int, float)):
        return float(confidence)
    if isinstance(confidence, str):
        return {
            "high": 0.95,
            "medium": 0.85,
            "low": 0.70,
        }.get(confidence.lower(), 0.0)
    return 0.0


def _classify_chain(
    entity: dict[str, Any],
    sequence: str | None,
    component_ids: list[str],
    chem_comp_map: dict[str, dict[str, str | None]],
    *,
    length: int,
    sadie_domain_bitscore_threshold: float = 80.0,
    sadie_domain_limit: int = 4,
) -> tuple[str, str | None, list[str], float, dict[str, Any], list[str]]:
    entity_type_lower = (entity.get("entity_type") or "").lower()
    polymer_type_lower = (entity.get("polymer_type") or "").lower()
    component_types = [
        chem_comp_map.get(component_id, {}).get("type")
        for component_id in component_ids
        if component_id
    ]
    features: dict[str, Any] = {}
    warnings: list[str] = []
    annotation_sources: list[str] = []
    chain_type = "other entity"
    subtype: str | None = None
    annotation_confidence = 0.4

    if entity_type_lower == "polymer":
        annotation_sources.append("mmcif_entity")
        if polymer_type_lower:
            annotation_sources.append("mmcif_entity_poly")

        if "polypeptide" in polymer_type_lower:
            chain_type = "other protein chain"
            subtype = "protein"
            annotation_confidence = 0.95
            immune_annotation = analyze_immune_sequence(
                entity.get("entity_description"),
                sequence,
                domain_bitscore_threshold=sadie_domain_bitscore_threshold,
                domain_limit=sadie_domain_limit,
            )
            antibody_annotation = None
            heuristic_type, heuristic_subtype, heuristic_sources, heuristic_features = (
                _description_based_chain_type(entity.get("entity_description"), length)
            )
            if immune_annotation.chain_type:
                chain_type = immune_annotation.chain_type
                subtype = immune_annotation.subtype
                annotation_sources.append("sadie_sequence_analysis")
                annotation_confidence = max(
                    annotation_confidence,
                    _confidence_label_to_score(immune_annotation.annotation_confidence),
                )
                features["variable_domains"] = [
                    domain.to_dict() for domain in immune_annotation.variable_domains
                ]
                warnings.extend(
                    warning for warning in immune_annotation.warnings if warning not in warnings
                )
                if immune_annotation.warning_details:
                    features["immune_annotation_warning_details"] = dict(
                        immune_annotation.warning_details
                    )
                if immune_annotation.chain_type.startswith("antibody"):
                    antibody_annotation = analyze_antibody_sequence(
                        entity.get("entity_description"),
                        sequence,
                        immune_annotation=immune_annotation,
                        domain_bitscore_threshold=sadie_domain_bitscore_threshold,
                        domain_limit=sadie_domain_limit,
                    )
                    features["antibody_analysis"] = antibody_annotation.to_feature_dict()
                    if antibody_annotation.antibody_domains:
                        features["antibody_domains"] = [
                            dict(domain) for domain in antibody_annotation.antibody_domains
                        ]
                    if antibody_annotation.antibody_units:
                        features["antibody_units"] = [
                            dict(unit) for unit in antibody_annotation.antibody_units
                        ]
                    if antibody_annotation.primary_antibody_unit_id:
                        features["primary_antibody_unit_id"] = antibody_annotation.primary_antibody_unit_id
                    if antibody_annotation.unit_type:
                        features["antibody_unit_type"] = antibody_annotation.unit_type
                    if antibody_annotation.contains_fused_heavy_fv:
                        features["contains_fused_heavy_fv"] = True
                    if antibody_annotation.contains_fused_light_fv:
                        features["contains_fused_light_fv"] = True
                if immune_annotation.chain_type == "TCR chain":
                    features["tcr_analysis"] = immune_annotation.to_feature_dict()

            if heuristic_type and not immune_annotation.chain_type:
                chain_type = heuristic_type
                subtype = heuristic_subtype
                annotation_sources.extend(heuristic_sources)
                features.update(heuristic_features)
                annotation_confidence = 0.8
            elif heuristic_type and heuristic_type == chain_type:
                annotation_sources.extend(source for source in heuristic_sources if source not in annotation_sources)
                if subtype in {None, "protein", "heavy"} and heuristic_subtype == "VHH":
                    subtype = heuristic_subtype
                if heuristic_subtype and chain_type == "antibody light chain" and subtype == "light":
                    subtype = heuristic_subtype
                for key, value in heuristic_features.items():
                    features.setdefault(key, value)
            elif heuristic_type and immune_annotation.chain_type and (
                "antibody" in heuristic_type or heuristic_type == "TCR chain"
            ):
                warnings.append("description_and_sequence_immune_annotation_disagree")

            return chain_type, subtype, annotation_sources, annotation_confidence, features, warnings

        if (
            "polydeoxyribonucleotide" in polymer_type_lower
            and "polyribonucleotide" in polymer_type_lower
        ):
            return (
                "other nucleic acid chain",
                "dna/rna hybrid",
                annotation_sources,
                0.95,
                features,
                warnings,
            )
        if "polydeoxyribonucleotide" in polymer_type_lower:
            return "DNA chain", "DNA", annotation_sources, 0.95, features, warnings
        if "polyribonucleotide" in polymer_type_lower:
            return "RNA chain", "RNA", annotation_sources, 0.95, features, warnings
        return "other polymer chain", None, annotation_sources, 0.8, features, warnings

    if entity_type_lower == "branched":
        annotation_sources.extend(["mmcif_entity", "branched_entity"])
        return "glycan / branched component", "glycan", annotation_sources, 0.95, features, warnings

    if entity_type_lower == "non-polymer":
        annotation_sources.extend(["mmcif_entity", "chem_comp"])
        if component_ids and all(_is_metal_component(component_id, chem_comp_map) for component_id in component_ids):
            return "metal ion", "ion", annotation_sources, 0.95, features, warnings
        return "small molecule compound", "ligand", annotation_sources, 0.9, features, warnings

    if entity_type_lower == "water":
        annotation_sources.append("mmcif_entity")
        return "other entity", "water", annotation_sources, 0.9, features, warnings

    annotation_sources.append("mmcif_entity")
    return chain_type, subtype, annotation_sources, annotation_confidence, features, warnings


def _append_unique(items: list[str], value: str | None) -> None:
    if value and value not in items:
        items.append(value)


def _is_hoh_water_chain(
    entity: dict[str, Any],
    monomer_ids: list[str],
    component_ids: list[str],
    sequence: str | None,
) -> bool:
    entity_type = str(entity.get("entity_type") or "").strip().lower()
    if entity_type == "water":
        return True

    normalized_monomers = {str(monomer_id).strip().upper() for monomer_id in monomer_ids if monomer_id}
    if normalized_monomers == {"HOH"}:
        return True

    normalized_components = {str(component_id).strip().upper() for component_id in component_ids if component_id}
    if normalized_components == {"HOH"}:
        return True

    normalized_sequence = str(sequence).strip().upper() if sequence is not None else None
    return normalized_sequence == "HOH"


def _apply_single_chain_coverage(
    cif_file: CIFFile,
    chain_records: list[ChainRecord],
    *,
    model: int,
    coverage_mode: str,
    drop_hydrogens_for_analysis: bool,
    distance_threshold: float = 4.5,
) -> None:
    if coverage_mode != "nearest":
        for record in chain_records:
            record.features["coverage_mode"] = coverage_mode
            record.features["coverage_status"] = "unsupported_mode"
            record.warnings.append(f"coverage mode {coverage_mode!r} is not implemented; coverage skipped")
            _append_warning_detail(
                record,
                "coverage mode not implemented",
                {
                    "coverage_mode": coverage_mode,
                    "supported_coverage_modes": ["nearest"],
                },
            )
        return

    subordinate_chain_types = {
        "glycan / branched component",
        "small molecule compound",
        "metal ion",
    }

    for record in chain_records:
        record.features["coverage_mode"] = coverage_mode

    try:
        atom_array = get_structure_with_altloc_fallback(
            cif_file,
            model=model,
            use_author_fields=False,
        )
    except ValueError as exc:
        if str(exc) != "Array must contain at least one element":
            raise
        for record in chain_records:
            if record.chain_type not in subordinate_chain_types:
                continue
            record.features["coverage_status"] = "structure_extraction_failed"
            record.warnings.append("coverage skipped because atom array extraction failed")
            _append_warning_detail(
                record,
                "coverage skipped because atom array extraction failed",
                {
                    "coverage_mode": coverage_mode,
                    "model": model,
                    "drop_hydrogens_for_analysis": drop_hydrogens_for_analysis,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        return

    atom_array, _ = filter_atom_array_for_analysis(
        atom_array,
        drop_hydrogens=drop_hydrogens_for_analysis,
        drop_nonfinite=True,
    )
    chain_coords: dict[str, np.ndarray] = {}
    for label_asym_id in sorted({str(chain_id) for chain_id in atom_array.chain_id.tolist()}):
        chain_coords[label_asym_id] = atom_array.coord[atom_array.chain_id == label_asym_id]

    primary_chain_types = {
        "antibody heavy chain",
        "antibody light chain",
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
        "other protein chain",
        "DNA chain",
        "RNA chain",
        "other nucleic acid chain",
    }
    primary_records = {
        record.label_asym_id: record
        for record in chain_records
        if record.chain_type in primary_chain_types and record.label_asym_id in chain_coords
    }
    primary_labels = sorted(primary_records)
    primary_tree = None
    primary_tree_labels: list[str] = []
    if primary_labels:
        try:
            from scipy.spatial import cKDTree

            coords: list[np.ndarray] = []
            labels: list[str] = []
            for owner_label in primary_labels:
                owner_coords = chain_coords[owner_label]
                if owner_coords.size == 0:
                    continue
                coords.append(owner_coords)
                labels.extend([owner_label] * owner_coords.shape[0])
            if coords:
                primary_tree = cKDTree(np.vstack(coords))
                primary_tree_labels = labels
        except Exception:
            LOGGER.debug("Falling back to brute-force coverage assignment", exc_info=True)

    for record in chain_records:
        if record.chain_type not in subordinate_chain_types:
            continue

        sub_coords = chain_coords.get(record.label_asym_id)
        if sub_coords is None or sub_coords.size == 0:
            record.features["coverage_status"] = "no_coordinates"
            record.warnings.append("coverage skipped because the chain has no coordinates")
            _append_warning_detail(
                record,
                "coverage skipped because the chain has no coordinates",
                {
                    "coverage_mode": coverage_mode,
                    "distance_threshold": distance_threshold,
                    "label_asym_id": record.label_asym_id,
                    "auth_asym_id": record.auth_asym_id,
                },
            )
            continue

        owner_labels: list[str] = []
        if primary_tree is not None and primary_tree_labels:
            distances, indices = primary_tree.query(sub_coords, k=1, distance_upper_bound=distance_threshold)
            for distance, index in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False):
                if np.isfinite(distance) and int(index) < len(primary_tree_labels):
                    owner_labels.append(primary_tree_labels[int(index)])
        else:
            for atom_coord in sub_coords:
                best_owner_label: str | None = None
                best_distance_sq: float | None = None
                for owner_label in primary_records:
                    owner_coords = chain_coords[owner_label]
                    if owner_coords.size == 0:
                        continue
                    deltas = owner_coords - atom_coord
                    distance_sq = float(np.sum(deltas * deltas, axis=1).min())
                    if best_distance_sq is None or distance_sq < best_distance_sq:
                        best_distance_sq = distance_sq
                        best_owner_label = owner_label

                if best_owner_label and best_distance_sq is not None:
                    if best_distance_sq ** 0.5 <= distance_threshold:
                        owner_labels.append(best_owner_label)

        unique_owner_labels = sorted(set(owner_labels))
        owner_auth_ids = [
            primary_records[owner_label].auth_asym_id
            for owner_label in unique_owner_labels
            if owner_label in primary_records
        ]
        record.features["coverage_owner_label_asym_ids"] = unique_owner_labels
        record.features["coverage_owner_auth_asym_ids"] = owner_auth_ids

        if len(unique_owner_labels) == 1:
            owner_record = primary_records[unique_owner_labels[0]]
            record.features["coverage_status"] = "single_owner"
            if record.chain_type == "glycan / branched component":
                _append_unique(owner_record.covered_branched_ids, record.label_asym_id)
                _append_unique(owner_record.covered_branched_auth_asym_ids, record.auth_asym_id)
            else:
                _append_unique(owner_record.covered_nonpolymer_ids, record.label_asym_id)
                _append_unique(owner_record.covered_nonpolymer_auth_asym_ids, record.auth_asym_id)
        elif len(unique_owner_labels) > 1:
            record.features["coverage_status"] = "multiple_owners"
            record.warnings.append("coverage assigned to multiple main chains")
            _append_warning_detail(
                record,
                "coverage assigned to multiple main chains",
                {
                    "coverage_mode": coverage_mode,
                    "distance_threshold": distance_threshold,
                    "owner_label_asym_ids": unique_owner_labels,
                    "owner_auth_asym_ids": owner_auth_ids,
                },
            )
        else:
            record.features["coverage_status"] = "unassigned"
            record.warnings.append("coverage owner not found within the nearest-distance threshold")
            _append_warning_detail(
                record,
                "coverage owner not found within the nearest-distance threshold",
                {
                    "coverage_mode": coverage_mode,
                    "distance_threshold": distance_threshold,
                    "owner_label_asym_ids": unique_owner_labels,
                    "owner_auth_asym_ids": owner_auth_ids,
                },
            )


def read_cif_file(path: str | Path) -> CIFFile:
    """Read a text or gzip-compressed mmCIF file into a Biotite CIF object."""

    cif_path = Path(path)
    with _open_cif_text(cif_path) as handle:
        return CIFFile.read(handle)


def read_chain_inventory(
    path: str | Path,
    *,
    model: int = 1,
    coverage_mode: str = "nearest",
    drop_hydrogens_for_analysis: bool = True,
    sadie_domain_bitscore_threshold: float = 80.0,
    sadie_domain_limit: int = 4,
    cif_file: CIFFile | None = None,
) -> list[ChainRecord]:
    """Build the annotated chain inventory for one mmCIF structure."""

    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)
    pdb_id = _infer_pdb_id(cif_path, cif_file)
    LOGGER.debug("Reading chain inventory for %s from %s", pdb_id, cif_path)
    entity_map = _build_entity_map(cif_file)
    poly_seq_rows = _build_poly_seq_rows(cif_file)
    poly_scheme_maps = _build_poly_scheme_maps(cif_file)
    chem_comp_map = _build_chem_comp_map(cif_file)
    atom_site_stats = _build_atom_site_stats(cif_file)
    auth_id_map, auth_warnings = _scheme_auth_id_map(cif_file, atom_site_stats)

    chain_records: list[ChainRecord] = []
    for row in _category_rows(cif_file, "struct_asym"):
        label_asym_id = row.get("id")
        entity_id = row.get("entity_id")
        if not label_asym_id or not entity_id:
            continue

        entity = entity_map.get(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": "unknown",
                "entity_description": None,
                "polymer_type": None,
                "sequence": None,
                "monomer_ids": [],
            },
        )
        chain_stats = atom_site_stats.get(
            label_asym_id,
            {
                "atom_count": 0,
                "residue_count": 0,
                "component_ids": [],
                "residue_monomer_ids": [],
            },
        )
        warnings = list(auth_warnings.get(label_asym_id, []))
        auth_asym_id = auth_id_map.get(label_asym_id)
        if auth_asym_id is None:
            warnings.append("auth_asym_id not found in scheme categories or atom_site")

        monomer_ids = list(chain_stats.get("residue_monomer_ids") or entity.get("monomer_ids") or [])
        sequence = entity.get("sequence")
        fallback_sequence_from_monomers = False
        if sequence is None and monomer_ids:
            sequence = "-".join(monomer_ids)
            fallback_sequence_from_monomers = True
        component_ids = list(chain_stats.get("component_ids") or [])

        if _is_hoh_water_chain(entity, monomer_ids, component_ids, sequence):
            continue

        length = len(monomer_ids) if fallback_sequence_from_monomers else len(sequence or "")
        residue_count = int(chain_stats.get("residue_count") or len(monomer_ids))
        if entity.get("entity_type") != "polymer":
            length = residue_count

        chain_type, subtype, annotation_sources, annotation_confidence, features, rule_warnings = _classify_chain(
            entity,
            sequence,
            monomer_ids,
            chem_comp_map,
            length=length,
            sadie_domain_bitscore_threshold=sadie_domain_bitscore_threshold,
            sadie_domain_limit=sadie_domain_limit,
        )
        warnings.extend(rule_warnings)
        features["chain_id_mapping"] = {
            "label_asym_id": label_asym_id,
            "auth_asym_id": auth_asym_id,
            "entity_id": entity_id,
        }

        parsed_coordinate_segments: list[dict[str, Any]] = []
        unresolved_sequence_segments: list[dict[str, Any]] = []
        special_residue_details: list[dict[str, Any]] = []
        special_component_details: list[dict[str, Any]] = []

        if entity.get("entity_type") == "polymer":
            entity_poly_rows = poly_seq_rows.get(entity_id, [])
            scheme_map = poly_scheme_maps.get(label_asym_id, {})
            expected_label_seq_ids = [row["label_seq_id"] for row in entity_poly_rows]
            resolved_label_seq_ids = {
                seq_id for seq_id in chain_stats.get("resolved_label_seq_ids", []) if seq_id is not None
            }
            parsed_coordinate_segments = _build_segments(
                sorted(resolved_label_seq_ids),
                scheme_map,
            )
            unresolved_sequence_segments = _build_segments(
                sorted(set(expected_label_seq_ids) - resolved_label_seq_ids),
                scheme_map,
            )
            special_residue_details = _build_special_residue_details(
                chain_type,
                entity_poly_rows,
                scheme_map,
                resolved_label_seq_ids,
                chem_comp_map,
            )
            features["num_parsed_coordinate_segments"] = len(parsed_coordinate_segments)
            features["num_unresolved_sequence_segments"] = len(unresolved_sequence_segments)
        else:
            special_component_details = _build_special_component_details(
                monomer_ids,
                chem_comp_map,
            )

        chain_records.append(
            ChainRecord(
                pdb_id=pdb_id,
                entity_id=entity_id,
                entity_type=entity.get("entity_type") or "unknown",
                entity_description=entity.get("entity_description"),
                label_asym_id=label_asym_id,
                auth_asym_id=auth_asym_id,
                polymer_type=entity.get("polymer_type"),
                chain_type=chain_type,
                subtype=subtype,
                sequence=sequence,
                length=length,
                residue_count=residue_count,
                atom_count=int(chain_stats.get("atom_count", 0)),
                special_residue_details=special_residue_details,
                special_component_details=special_component_details,
                parsed_coordinate_segments=parsed_coordinate_segments,
                unresolved_sequence_segments=unresolved_sequence_segments,
                annotation_sources=annotation_sources,
                annotation_confidence=annotation_confidence,
                features=features,
                warnings=warnings,
            )
        )

    _apply_single_chain_coverage(
        cif_file,
        chain_records,
        model=model,
        coverage_mode=coverage_mode,
        drop_hydrogens_for_analysis=drop_hydrogens_for_analysis,
    )
    sorted_records = sorted(chain_records, key=lambda record: (record.label_asym_id, record.entity_id))
    LOGGER.debug(
        "Built %d chain records for %s using coverage mode %s",
        len(sorted_records),
        pdb_id,
        coverage_mode,
    )
    return sorted_records


def read_structure_summary(
    path: str | Path,
    *,
    model: int = 1,
    use_author_fields: bool = False,
    coverage_mode: str = "nearest",
    drop_hydrogens_for_analysis: bool = True,
    chain_inventory: list[ChainRecord] | None = None,
    sadie_domain_bitscore_threshold: float = 80.0,
    sadie_domain_limit: int = 4,
    cif_file: CIFFile | None = None,
) -> StructureSummary:
    """Read a structure-level summary and filtered visible chain ids."""

    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)
    LOGGER.debug("Reading structure summary from %s", cif_path)
    try:
        assembly_map = {
            str(assembly_id): str(description)
            for assembly_id, description in list_assemblies(cif_file).items()
        }
    except Exception:
        assembly_map = {}
    if chain_inventory is None:
        chain_inventory = read_chain_inventory(
            cif_path,
            model=model,
            coverage_mode=coverage_mode,
            drop_hydrogens_for_analysis=drop_hydrogens_for_analysis,
            sadie_domain_bitscore_threshold=sadie_domain_bitscore_threshold,
            sadie_domain_limit=sadie_domain_limit,
            cif_file=cif_file,
        )
    try:
        atom_array = get_structure_with_altloc_fallback(
            cif_file,
            model=model,
            use_author_fields=use_author_fields,
        )
        atom_count = int(atom_array.array_length())
        chain_ids = sorted({str(chain_id) for chain_id in atom_array.chain_id.tolist()})
    except ValueError as exc:
        if str(exc) != "Array must contain at least one element":
            raise
        LOGGER.warning(
            "Structure summary atom array extraction failed for %s; falling back to chain inventory only",
            cif_path,
        )
        atom_count = 0
        chain_ids = []
    visible_chain_ids = {
        record.auth_asym_id if use_author_fields else record.label_asym_id
        for record in chain_inventory
        if (record.auth_asym_id if use_author_fields else record.label_asym_id) is not None
    }
    if chain_ids:
        chain_ids = [chain_id for chain_id in chain_ids if chain_id in visible_chain_ids]
    else:
        chain_ids = sorted(str(chain_id) for chain_id in visible_chain_ids)
    chain_type_counts: dict[str, int] = {}
    for record in chain_inventory:
        chain_type_counts[record.chain_type] = chain_type_counts.get(record.chain_type, 0) + 1

    block_name = _default_block_name(cif_file)
    summary = StructureSummary(
        pdb_id=_infer_pdb_id(cif_path, cif_file),
        source_path=str(cif_path.resolve()),
        data_block=block_name,
        chain_id_source="auth_asym_id" if use_author_fields else "label_asym_id",
        model=model,
        atom_count=atom_count,
        entity_count=_count_category_rows(cif_file, "entity", "id"),
        chain_ids=chain_ids,
        chain_id_pairs=[
            {
                "label_asym_id": record.label_asym_id,
                "auth_asym_id": record.auth_asym_id,
                "entity_id": record.entity_id,
                "chain_type": record.chain_type,
            }
            for record in chain_inventory
        ],
        chain_type_counts=chain_type_counts,
        assembly_ids=sorted(assembly_map),
        assembly_descriptions=assembly_map,
        title=_get_first_value(cif_file, "struct", "title"),
    )
    LOGGER.debug(
        "Read structure summary for %s with %d atoms and %d visible chains",
        summary.pdb_id,
        summary.atom_count,
        len(summary.chain_ids),
    )
    return summary
