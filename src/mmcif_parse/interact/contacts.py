from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from biotite.structure import AtomArray, sasa


RESIDUE_CONTACT_CUTOFF = 8.0
ATOM_CONTACT_CUTOFF = 5.0
MIN_RESIDUE_CONTACTS = 3
MIN_ATOM_CONTACTS = 20
ATOM_CHUNK_SIZE = 256


@dataclass(slots=True)
class ChainGeometry:
    instance_id: str
    label_asym_id: str
    sym_id: int
    atom_array: AtomArray
    atom_coords: np.ndarray
    residue_rep_coords: np.ndarray
    residue_ids: list[tuple[int, str, str]]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    centroid: np.ndarray


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


def build_chain_geometries(atom_array: AtomArray, chain_records: list[Any]) -> dict[str, ChainGeometry]:
    chain_type_map = {record.label_asym_id: record.chain_type for record in chain_records}
    geometries: dict[str, ChainGeometry] = {}
    for label_asym_id in sorted({str(chain_id) for chain_id in atom_array.chain_id.tolist()}):
        mask = atom_array.chain_id == label_asym_id
        chain_atoms = atom_array[mask]
        atom_coords = np.asarray(chain_atoms.coord, dtype=np.float32)
        if atom_coords.size == 0:
            continue

        residue_rep_coords: list[np.ndarray] = []
        residue_ids: list[tuple[int, str, str]] = []
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
            residue_start = atom_index

        residue_coords = np.asarray(residue_rep_coords, dtype=np.float32)
        geometries[label_asym_id] = ChainGeometry(
            instance_id=label_asym_id,
            label_asym_id=label_asym_id,
            sym_id=0,
            atom_array=chain_atoms,
            atom_coords=atom_coords,
            residue_rep_coords=residue_coords,
            residue_ids=residue_ids,
            bbox_min=atom_coords.min(axis=0),
            bbox_max=atom_coords.max(axis=0),
            centroid=atom_coords.mean(axis=0),
        )
    return geometries


def build_instance_geometries(atom_array: AtomArray, chain_records: list[Any]) -> dict[str, ChainGeometry]:
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
        chain_atoms = atom_array[mask]
        atom_coords = np.asarray(chain_atoms.coord, dtype=np.float32)
        if atom_coords.size == 0:
            continue

        residue_rep_coords: list[np.ndarray] = []
        residue_ids: list[tuple[int, str, str]] = []
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
            residue_start = atom_index

        residue_coords = np.asarray(residue_rep_coords, dtype=np.float32)
        instance_id = f"{label_asym_id}@{sym_id + 1}"
        geometries[instance_id] = ChainGeometry(
            instance_id=instance_id,
            label_asym_id=label_asym_id,
            sym_id=sym_id,
            atom_array=chain_atoms,
            atom_coords=atom_coords,
            residue_rep_coords=residue_coords,
            residue_ids=residue_ids,
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
    }


def _interface_area_metrics(geometry_1: ChainGeometry, geometry_2: ChainGeometry) -> dict[str, float]:
    sasa_1 = np.asarray(sasa(geometry_1.atom_array), dtype=np.float32)
    sasa_2 = np.asarray(sasa(geometry_2.atom_array), dtype=np.float32)
    complex_array = geometry_1.atom_array + geometry_2.atom_array
    complex_sasa = np.asarray(sasa(complex_array), dtype=np.float32)
    split_index = geometry_1.atom_array.array_length()
    complex_sasa_1 = complex_sasa[:split_index]
    complex_sasa_2 = complex_sasa[split_index:]

    delta_sasa_1 = float(np.nansum(sasa_1 - complex_sasa_1))
    delta_sasa_2 = float(np.nansum(sasa_2 - complex_sasa_2))
    buried_area = max((delta_sasa_1 + delta_sasa_2) / 2.0, 0.0)
    return {
        "delta_sasa_1": max(delta_sasa_1, 0.0),
        "delta_sasa_2": max(delta_sasa_2, 0.0),
        "buried_area": buried_area,
    }


def compute_interface_metrics(geometry_1: ChainGeometry, geometry_2: ChainGeometry) -> dict[str, float | int] | None:
    bbox_distance = _bbox_distance(
        geometry_1.bbox_min,
        geometry_1.bbox_max,
        geometry_2.bbox_min,
        geometry_2.bbox_max,
    )
    if bbox_distance > RESIDUE_CONTACT_CUTOFF:
        return None

    residue_metrics = _residue_contact_metrics(
        geometry_1,
        geometry_2,
        RESIDUE_CONTACT_CUTOFF,
    )
    residue_contacts = int(residue_metrics["num_residue_contacts"])
    if residue_contacts < MIN_RESIDUE_CONTACTS:
        return None

    atom_contacts, atom_min_distance = _count_distance_contacts(
        geometry_1.atom_coords,
        geometry_2.atom_coords,
        ATOM_CONTACT_CUTOFF,
    )
    min_distance = min(float(residue_metrics["residue_min_distance"]), atom_min_distance)
    if atom_contacts < MIN_ATOM_CONTACTS:
        return None

    centroid_distance = float(np.linalg.norm(geometry_1.centroid - geometry_2.centroid))
    area_metrics = _interface_area_metrics(geometry_1, geometry_2)
    return {
        "num_residue_contacts": residue_contacts,
        "num_atom_contacts": atom_contacts,
        "min_distance": min_distance,
        "bbox_distance": bbox_distance,
        "centroid_distance": centroid_distance,
        "delta_sasa_1": area_metrics["delta_sasa_1"],
        "delta_sasa_2": area_metrics["delta_sasa_2"],
        "buried_area": area_metrics["buried_area"],
        "interface_residue_count_1": int(residue_metrics["interface_residue_count_1"]),
        "interface_residue_count_2": int(residue_metrics["interface_residue_count_2"]),
    }
