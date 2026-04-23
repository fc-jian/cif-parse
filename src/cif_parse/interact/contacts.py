from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from biotite.structure import AtomArray, sasa

from cif_parse.constants import (
    ATOM_CHUNK_SIZE,
    GENERIC_VDW_RADII,
)
from cif_parse.utils import filter_atom_array_for_analysis, normalize_element_symbol


@dataclass(slots=True)
class ChainGeometry:
    instance_id: str
    label_asym_id: str
    sym_id: int
    assembly_id: str | None
    atom_array: AtomArray
    atom_coords: np.ndarray
    residue_rep_coords: np.ndarray
    residue_ids: list[tuple[int, str, str]]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    centroid: np.ndarray
    residue_atom_ranges: list[tuple[int, int]] | None = None


def _prepare_atom_array_for_area(atom_array: AtomArray) -> tuple[AtomArray, dict[str, int]]:
    return filter_atom_array_for_analysis(atom_array, drop_hydrogens=True, drop_nonfinite=True)


def _prepare_fallback_radii(atom_array: AtomArray) -> tuple[AtomArray, np.ndarray, int]:
    if atom_array.array_length() == 0:
        return atom_array, np.asarray([], dtype=np.float32), 0

    keep_mask = np.zeros(atom_array.array_length(), dtype=bool)
    radii = np.zeros(atom_array.array_length(), dtype=np.float32)
    unsupported_atoms = 0
    for atom_index, (element, atom_name) in enumerate(
        zip(atom_array.element, atom_array.atom_name, strict=False)
    ):
        normalized = normalize_element_symbol(str(element), str(atom_name))
        radius = GENERIC_VDW_RADII.get(normalized)
        if radius is None:
            unsupported_atoms += 1
            continue
        keep_mask[atom_index] = True
        radii[atom_index] = radius

    return atom_array[keep_mask], radii[keep_mask], unsupported_atoms


def _run_sasa_triplet(
    atom_array_1: AtomArray,
    atom_array_2: AtomArray,
    complex_array: AtomArray,
    *,
    vdw_radii: str | tuple[np.ndarray, np.ndarray, np.ndarray] = "ProtOr",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(vdw_radii, tuple):
        radii_1, radii_2, radii_complex = vdw_radii
        sasa_1 = np.asarray(sasa(atom_array_1, ignore_ions=False, vdw_radii=radii_1), dtype=np.float32)
        sasa_2 = np.asarray(sasa(atom_array_2, ignore_ions=False, vdw_radii=radii_2), dtype=np.float32)
        complex_sasa = np.asarray(
            sasa(complex_array, ignore_ions=False, vdw_radii=radii_complex),
            dtype=np.float32,
        )
        return sasa_1, sasa_2, complex_sasa

    sasa_1 = np.asarray(sasa(atom_array_1), dtype=np.float32)
    sasa_2 = np.asarray(sasa(atom_array_2), dtype=np.float32)
    complex_sasa = np.asarray(sasa(complex_array), dtype=np.float32)
    return sasa_1, sasa_2, complex_sasa


def _bbox_distance(bbox_min_1: np.ndarray, bbox_max_1: np.ndarray, bbox_min_2: np.ndarray, bbox_max_2: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(bbox_min_1 - bbox_max_2, bbox_min_2 - bbox_max_1))
    return float(np.sqrt(np.sum(delta * delta)))


def _select_representative_atom(chain_type: str, atom_names: list[str]) -> int:
    preferred_names: tuple[str, ...]
    if chain_type in {"DNA chain", "RNA chain", "other nucleic acid chain"}:
        preferred_names = ("P", "C4'", "C3'")
    else:
        preferred_names = ("CA", "C1'", "P")

    for preferred_name in preferred_names:
        if preferred_name in atom_names:
            return atom_names.index(preferred_name)
    return 0


def build_chain_geometries(
    atom_array: AtomArray,
    chain_records: list[Any],
    *,
    drop_hydrogens_for_analysis: bool = True,
) -> dict[str, ChainGeometry]:
    chain_type_map = {record.label_asym_id: record.chain_type for record in chain_records}
    geometries: dict[str, ChainGeometry] = {}
    for label_asym_id in sorted({str(chain_id) for chain_id in atom_array.chain_id.tolist()}):
        mask = atom_array.chain_id == label_asym_id
        chain_atoms, _ = filter_atom_array_for_analysis(
            atom_array[mask],
            drop_hydrogens=drop_hydrogens_for_analysis,
            drop_nonfinite=True,
        )
        atom_coords = np.asarray(chain_atoms.coord, dtype=np.float32)
        if atom_coords.size == 0:
            continue

        residue_rep_coords: list[np.ndarray] = []
        residue_ids: list[tuple[int, str, str]] = []
        residue_atom_ranges: list[tuple[int, int]] = []
        residue_start = 0
        for atom_index in range(1, chain_atoms.array_length() + 1):
            is_boundary = atom_index == chain_atoms.array_length()
            if not is_boundary:
                same_residue = (
                    chain_atoms.res_id[atom_index] == chain_atoms.res_id[residue_start]
                    and chain_atoms.ins_code[atom_index] == chain_atoms.ins_code[residue_start]
                    and chain_atoms.res_name[atom_index] == chain_atoms.res_name[residue_start]
                )
                if same_residue:
                    continue

            residue_slice = chain_atoms[residue_start:atom_index]
            atom_names = residue_slice.atom_name.tolist()
            representative_index = _select_representative_atom(
                chain_type_map.get(label_asym_id, "other protein chain"),
                atom_names,
            )
            residue_rep_coords.append(np.asarray(residue_slice.coord[representative_index], dtype=np.float32))
            residue_ids.append(
                (
                    int(residue_slice.res_id[0]),
                    str(residue_slice.ins_code[0]),
                    str(residue_slice.res_name[0]),
                )
            )
            residue_atom_ranges.append((residue_start, atom_index))
            residue_start = atom_index

        residue_coords = np.asarray(residue_rep_coords, dtype=np.float32)
        geometries[label_asym_id] = ChainGeometry(
            instance_id=label_asym_id,
            label_asym_id=label_asym_id,
            sym_id=0,
            assembly_id=None,
            atom_array=chain_atoms,
            atom_coords=atom_coords,
            residue_rep_coords=residue_coords,
            residue_ids=residue_ids,
            residue_atom_ranges=residue_atom_ranges,
            bbox_min=atom_coords.min(axis=0),
            bbox_max=atom_coords.max(axis=0),
            centroid=atom_coords.mean(axis=0),
        )
    return geometries


def build_instance_geometries(
    atom_array: AtomArray,
    chain_records: list[Any],
    *,
    drop_hydrogens_for_analysis: bool = True,
) -> dict[str, ChainGeometry]:
    chain_type_map = {record.label_asym_id: record.chain_type for record in chain_records}
    geometries: dict[str, ChainGeometry] = {}
    sym_ids = atom_array.sym_id.tolist() if hasattr(atom_array, "sym_id") else [0] * atom_array.array_length()
    instance_keys = sorted(
        {
            (str(chain_id), int(sym_id))
            for chain_id, sym_id in zip(atom_array.chain_id.tolist(), sym_ids, strict=False)
        }
    )
    for label_asym_id, sym_id in instance_keys:
        mask = (atom_array.chain_id == label_asym_id) & (atom_array.sym_id == sym_id)
        chain_atoms, _ = filter_atom_array_for_analysis(
            atom_array[mask],
            drop_hydrogens=drop_hydrogens_for_analysis,
            drop_nonfinite=True,
        )
        atom_coords = np.asarray(chain_atoms.coord, dtype=np.float32)
        if atom_coords.size == 0:
            continue

        residue_rep_coords: list[np.ndarray] = []
        residue_ids: list[tuple[int, str, str]] = []
        residue_atom_ranges: list[tuple[int, int]] = []
        residue_start = 0
        for atom_index in range(1, chain_atoms.array_length() + 1):
            is_boundary = atom_index == chain_atoms.array_length()
            if not is_boundary:
                same_residue = (
                    chain_atoms.res_id[atom_index] == chain_atoms.res_id[residue_start]
                    and chain_atoms.ins_code[atom_index] == chain_atoms.ins_code[residue_start]
                    and chain_atoms.res_name[atom_index] == chain_atoms.res_name[residue_start]
                )
                if same_residue:
                    continue

            residue_slice = chain_atoms[residue_start:atom_index]
            atom_names = residue_slice.atom_name.tolist()
            representative_index = _select_representative_atom(
                chain_type_map.get(label_asym_id, "other protein chain"),
                atom_names,
            )
            residue_rep_coords.append(np.asarray(residue_slice.coord[representative_index], dtype=np.float32))
            residue_ids.append(
                (
                    int(residue_slice.res_id[0]),
                    str(residue_slice.ins_code[0]),
                    str(residue_slice.res_name[0]),
                )
            )
            residue_atom_ranges.append((residue_start, atom_index))
            residue_start = atom_index

        residue_coords = np.asarray(residue_rep_coords, dtype=np.float32)
        instance_id = f"{label_asym_id}@{sym_id + 1}"
        geometries[instance_id] = ChainGeometry(
            instance_id=instance_id,
            label_asym_id=label_asym_id,
            sym_id=sym_id,
            assembly_id=None,
            atom_array=chain_atoms,
            atom_coords=atom_coords,
            residue_rep_coords=residue_coords,
            residue_ids=residue_ids,
            residue_atom_ranges=residue_atom_ranges,
            bbox_min=atom_coords.min(axis=0),
            bbox_max=atom_coords.max(axis=0),
            centroid=atom_coords.mean(axis=0),
        )
    return geometries


def _count_distance_contacts(coords_1: np.ndarray, coords_2: np.ndarray, cutoff: float) -> tuple[int, float]:
    if coords_1.size == 0 or coords_2.size == 0:
        return 0, float("inf")

    cutoff_sq = cutoff * cutoff
    contact_count = 0
    min_distance_sq: float | None = None
    for start in range(0, coords_1.shape[0], ATOM_CHUNK_SIZE):
        chunk = coords_1[start : start + ATOM_CHUNK_SIZE]
        deltas = chunk[:, None, :] - coords_2[None, :, :]
        distance_sq = np.sum(deltas * deltas, axis=2)
        contact_count += int(np.count_nonzero(distance_sq <= cutoff_sq))
        chunk_min = float(distance_sq.min())
        if min_distance_sq is None or chunk_min < min_distance_sq:
            min_distance_sq = chunk_min

    return contact_count, float(np.sqrt(min_distance_sq)) if min_distance_sq is not None else float("inf")


def _residue_contact_metrics(
    geometry_1: ChainGeometry,
    geometry_2: ChainGeometry,
    cutoff: float,
) -> dict[str, float | int]:
    if geometry_1.residue_rep_coords.size == 0 or geometry_2.residue_rep_coords.size == 0:
        return {
            "num_residue_contacts": 0,
            "interface_residue_count_1": 0,
            "interface_residue_count_2": 0,
            "residue_min_distance": float("inf"),
        }

    cutoff_sq = cutoff * cutoff
    deltas = geometry_1.residue_rep_coords[:, None, :] - geometry_2.residue_rep_coords[None, :, :]
    distance_sq = np.sum(deltas * deltas, axis=2)
    contact_mask = distance_sq <= cutoff_sq
    return {
        "num_residue_contacts": int(np.count_nonzero(contact_mask)),
        "interface_residue_count_1": int(np.count_nonzero(contact_mask.any(axis=1))),
        "interface_residue_count_2": int(np.count_nonzero(contact_mask.any(axis=0))),
        "residue_min_distance": float(np.sqrt(distance_sq.min())),
        "contact_mask": contact_mask,
    }


def _format_residue_id(residue_id: tuple[int, str, str]) -> str:
    res_id, ins_code, res_name = residue_id
    suffix = str(ins_code).strip()
    return f"{res_name}:{res_id}{suffix}"


def _closest_residue_atom_pairs(
    geometry_1: ChainGeometry,
    geometry_2: ChainGeometry,
    contact_mask: np.ndarray,
) -> list[list[Any]]:
    if contact_mask.size == 0:
        return []
    if geometry_1.residue_atom_ranges is None or geometry_2.residue_atom_ranges is None:
        return []
    atom_pairs: list[list[Any]] = []
    residue_pairs = np.argwhere(contact_mask)
    for residue_index_1, residue_index_2 in residue_pairs.tolist():
        start_1, end_1 = geometry_1.residue_atom_ranges[residue_index_1]
        start_2, end_2 = geometry_2.residue_atom_ranges[residue_index_2]
        residue_atoms_1 = geometry_1.atom_array[start_1:end_1]
        residue_atoms_2 = geometry_2.atom_array[start_2:end_2]
        if residue_atoms_1.array_length() == 0 or residue_atoms_2.array_length() == 0:
            continue
        deltas = np.asarray(residue_atoms_1.coord, dtype=np.float32)[:, None, :] - np.asarray(
            residue_atoms_2.coord,
            dtype=np.float32,
        )[None, :, :]
        distance_sq = np.sum(deltas * deltas, axis=2)
        flat_index = int(distance_sq.argmin())
        atom_index_1, atom_index_2 = np.unravel_index(flat_index, distance_sq.shape)
        atom_pairs.append(
            [
                geometry_1.instance_id,
                _format_residue_id(geometry_1.residue_ids[residue_index_1]),
                str(residue_atoms_1.atom_name[atom_index_1]),
                geometry_2.instance_id,
                _format_residue_id(geometry_2.residue_ids[residue_index_2]),
                str(residue_atoms_2.atom_name[atom_index_2]),
                round(float(np.sqrt(distance_sq[atom_index_1, atom_index_2])), 4),
            ]
        )
    return atom_pairs


def _interface_area_metrics(geometry_1: ChainGeometry, geometry_2: ChainGeometry) -> dict[str, Any]:
    filtered_1, filter_counts_1 = _prepare_atom_array_for_area(geometry_1.atom_array)
    filtered_2, filter_counts_2 = _prepare_atom_array_for_area(geometry_2.atom_array)
    warnings: list[str] = []
    warning_details: dict[str, Any] = {}
    evidence: dict[str, Any] = {
        "interface_area_atom_filtering": {
            "chain_1": filter_counts_1,
            "chain_2": filter_counts_2,
        }
    }

    if filtered_1.array_length() == 0 or filtered_2.array_length() == 0:
        warnings.append("interface_area_not_computed_after_filtering")
        warning_details["interface_area_not_computed_after_filtering"] = {
            "reason": "no_atoms_remain_after_filtering",
            "chain_1_remaining_atoms": int(filtered_1.array_length()),
            "chain_2_remaining_atoms": int(filtered_2.array_length()),
        }
        evidence["interface_area_method"] = "not_computed"
        return {
            "delta_sasa_1": 0.0,
            "delta_sasa_2": 0.0,
            "buried_area": 0.0,
            "warnings": warnings,
            "warning_details": warning_details,
            "evidence": evidence,
        }

    complex_array = filtered_1 + filtered_2
    split_index = filtered_1.array_length()
    try:
        sasa_1, sasa_2, complex_sasa = _run_sasa_triplet(filtered_1, filtered_2, complex_array)
        evidence["interface_area_method"] = "ProtOr"
    except Exception as exc:  # noqa: BLE001
        warnings.append("interface_area_used_element_vdw_fallback")
        warning_details["interface_area_used_element_vdw_fallback"] = {
            "error": f"{type(exc).__name__}: {exc}",
        }
        evidence["interface_area_fallback_error"] = f"{type(exc).__name__}: {exc}"
        fallback_1, radii_1, unsupported_1 = _prepare_fallback_radii(filtered_1)
        fallback_2, radii_2, unsupported_2 = _prepare_fallback_radii(filtered_2)
        evidence["interface_area_fallback_filtering"] = {
            "chain_1": {"unsupported_atoms_removed": unsupported_1},
            "chain_2": {"unsupported_atoms_removed": unsupported_2},
        }
        if unsupported_1 > 0 or unsupported_2 > 0:
            warnings.append("interface_area_dropped_unsupported_atoms_for_fallback")
            warning_details["interface_area_dropped_unsupported_atoms_for_fallback"] = {
                "chain_1_unsupported_atoms_removed": int(unsupported_1),
                "chain_2_unsupported_atoms_removed": int(unsupported_2),
            }
        if fallback_1.array_length() == 0 or fallback_2.array_length() == 0:
            warnings.append("interface_area_not_computed_after_filtering")
            warning_details["interface_area_not_computed_after_filtering"] = {
                "reason": "no_atoms_remain_after_fallback_filtering",
                "chain_1_remaining_atoms": int(fallback_1.array_length()),
                "chain_2_remaining_atoms": int(fallback_2.array_length()),
            }
            evidence["interface_area_method"] = "not_computed"
            return {
                "delta_sasa_1": 0.0,
                "delta_sasa_2": 0.0,
                "buried_area": 0.0,
                "warnings": warnings,
                "warning_details": warning_details,
                "evidence": evidence,
            }

        complex_array = fallback_1 + fallback_2
        split_index = fallback_1.array_length()
        radii_complex = np.concatenate([radii_1, radii_2]).astype(np.float32, copy=False)
        try:
            sasa_1, sasa_2, complex_sasa = _run_sasa_triplet(
                fallback_1,
                fallback_2,
                complex_array,
                vdw_radii=(radii_1, radii_2, radii_complex),
            )
            evidence["interface_area_method"] = "element_vdw_fallback"
        except Exception as fallback_exc:  # noqa: BLE001
            warnings.append("interface_area_fallback_failed")
            warning_details["interface_area_fallback_failed"] = {
                "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
            }
            evidence["interface_area_method"] = "not_computed"
            evidence["interface_area_fallback_failure"] = (
                f"{type(fallback_exc).__name__}: {fallback_exc}"
            )
            return {
                "delta_sasa_1": 0.0,
                "delta_sasa_2": 0.0,
                "buried_area": 0.0,
                "warnings": warnings,
                "warning_details": warning_details,
                "evidence": evidence,
            }

    complex_sasa_1 = complex_sasa[:split_index]
    complex_sasa_2 = complex_sasa[split_index:]

    delta_sasa_1 = float(np.nansum(sasa_1 - complex_sasa_1))
    delta_sasa_2 = float(np.nansum(sasa_2 - complex_sasa_2))
    buried_area = max((delta_sasa_1 + delta_sasa_2) / 2.0, 0.0)
    return {
        "delta_sasa_1": max(delta_sasa_1, 0.0),
        "delta_sasa_2": max(delta_sasa_2, 0.0),
        "buried_area": buried_area,
        "warnings": warnings,
        "warning_details": warning_details,
        "evidence": evidence,
    }


def compute_interface_metrics(
    geometry_1: ChainGeometry,
    geometry_2: ChainGeometry,
    *,
    residue_contact_cutoff: float = 8.0,
    atom_contact_cutoff: float = 5.0,
    min_residue_contacts: int = 3,
    min_atom_contacts: int = 20,
) -> dict[str, Any] | None:
    bbox_distance = _bbox_distance(
        geometry_1.bbox_min,
        geometry_1.bbox_max,
        geometry_2.bbox_min,
        geometry_2.bbox_max,
    )
    if bbox_distance > residue_contact_cutoff:
        return None

    residue_metrics = _residue_contact_metrics(
        geometry_1,
        geometry_2,
        residue_contact_cutoff,
    )
    residue_contacts = int(residue_metrics["num_residue_contacts"])
    if residue_contacts < min_residue_contacts:
        return None

    atom_contacts, atom_min_distance = _count_distance_contacts(
        geometry_1.atom_coords,
        geometry_2.atom_coords,
        atom_contact_cutoff,
    )
    min_distance = min(float(residue_metrics["residue_min_distance"]), atom_min_distance)
    if atom_contacts < min_atom_contacts:
        return None

    centroid_distance = float(np.linalg.norm(geometry_1.centroid - geometry_2.centroid))
    area_metrics = _interface_area_metrics(geometry_1, geometry_2)
    interface_residue_count_1 = int(residue_metrics["interface_residue_count_1"])
    interface_residue_count_2 = int(residue_metrics["interface_residue_count_2"])
    mean_interface_residue_count = (interface_residue_count_1 + interface_residue_count_2) / 2.0
    buried_area = float(area_metrics["buried_area"])
    contacting_atom_pairs = _closest_residue_atom_pairs(
        geometry_1,
        geometry_2,
        np.asarray(residue_metrics["contact_mask"], dtype=bool),
    )
    return {
        "num_residue_contacts": residue_contacts,
        "num_atom_contacts": atom_contacts,
        "min_distance": min_distance,
        "bbox_distance": bbox_distance,
        "centroid_distance": centroid_distance,
        "delta_sasa_1": area_metrics["delta_sasa_1"],
        "delta_sasa_2": area_metrics["delta_sasa_2"],
        "buried_area": buried_area,
        "area_warnings": list(area_metrics.get("warnings", [])),
        "area_warning_details": dict(area_metrics.get("warning_details", {})),
        "area_evidence": dict(area_metrics.get("evidence", {})),
        "interface_residue_count_1": interface_residue_count_1,
        "interface_residue_count_2": interface_residue_count_2,
        "mean_interface_residue_count": mean_interface_residue_count,
        "buried_area_per_interface_residue": (
            buried_area / mean_interface_residue_count if mean_interface_residue_count > 0.0 else 0.0
        ),
        "atom_contacts_per_interface_residue": (
            atom_contacts / mean_interface_residue_count if mean_interface_residue_count > 0.0 else 0.0
        ),
        "residue_contacts_per_interface_residue": (
            residue_contacts / mean_interface_residue_count if mean_interface_residue_count > 0.0 else 0.0
        ),
        "contacting_atom_pairs": contacting_atom_pairs,
    }
