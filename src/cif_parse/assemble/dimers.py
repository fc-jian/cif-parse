from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from biotite.structure.io.pdbx import CIFFile, list_assemblies

from cif_parse.constants import POLYMER_CHAIN_TYPES
from cif_parse.interact.contacts import (
    build_chain_geometries,
    build_instance_geometries,
    compute_interface_metrics,
)
from cif_parse.io import read_cif_file, select_largest_polymer_assembly_id
from cif_parse.io.cif_reader import (
    get_assembly_with_altloc_fallback,
    get_structure_with_altloc_fallback,
)
from cif_parse.models import DimerInterfaceRecord


def _interface_label(chain_type_1: str, chain_type_2: str, is_same_entity: bool) -> str:
    sorted_types = sorted([chain_type_1, chain_type_2])
    if is_same_entity:
        return f"same-entity:{sorted_types[0]}"
    return f"{sorted_types[0]}__{sorted_types[1]}"


def _contains_antibody_unit(chain_type_1: str, chain_type_2: str) -> bool:
    return "antibody" in chain_type_1 or "antibody" in chain_type_2


def _contains_tcr_pmhc_unit(chain_type_1: str, chain_type_2: str) -> bool:
    relevant_types = {
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
    }
    return chain_type_1 in relevant_types or chain_type_2 in relevant_types


def _dimer_dedup_key(dimer: DimerInterfaceRecord) -> tuple[str, str, str | None, str, str, str, str, bool]:
    left = (dimer.label_asym_id_1, dimer.entity_id_1, dimer.chain_type_1)
    right = (dimer.label_asym_id_2, dimer.entity_id_2, dimer.chain_type_2)
    ordered = sorted([left, right])
    return (
        dimer.assembly_mode,
        dimer.assembly_id,
        ordered[0][0],
        ordered[0][1],
        ordered[1][0],
        ordered[1][1],
        dimer.interface_label,
        dimer.is_same_entity,
    )


def _dimer_rank_key(dimer: DimerInterfaceRecord) -> tuple[float, int, int, float]:
    return (
        float(dimer.buried_area),
        int(dimer.num_atom_contacts),
        int(dimer.num_residue_contacts),
        -float(dimer.min_distance),
    )


def _bbox_distance_from_geometries(geometry_1: object, geometry_2: object) -> float:
    import numpy as np

    delta = np.maximum(
        0.0,
        np.maximum(geometry_1.bbox_min - geometry_2.bbox_max, geometry_2.bbox_min - geometry_1.bbox_max),
    )
    return float(np.sqrt(np.sum(delta * delta)))


def _deduplicate_dimer_interfaces(dimers: list[DimerInterfaceRecord]) -> list[DimerInterfaceRecord]:
    grouped: dict[tuple[str, str, str | None, str, str, str, str, bool], list[DimerInterfaceRecord]] = {}
    for dimer in dimers:
        grouped.setdefault(_dimer_dedup_key(dimer), []).append(dimer)

    deduped: list[DimerInterfaceRecord] = []
    for group in grouped.values():
        representative = max(group, key=_dimer_rank_key)
        supporting_pairs = sorted(
            {
                (
                    dimer.instance_id_1 or dimer.label_asym_id_1,
                    dimer.instance_id_2 or dimer.label_asym_id_2,
                )
                for dimer in group
            }
        )
        representative.num_supporting_instance_pairs = len(supporting_pairs)
        representative.evidence = {
            **representative.evidence,
            "dedup_stage": "dimer_dedup_v1",
            "supporting_instance_pairs": [list(pair) for pair in supporting_pairs],
        }
        if len(group) > 1 and "deduplicated_over_instance_pairs" not in representative.warnings:
            representative.warnings.append("deduplicated_over_instance_pairs")
        deduped.append(representative)

    return sorted(
        deduped,
        key=lambda dimer: (
            dimer.pdb_id,
            dimer.label_asym_id_1,
            dimer.label_asym_id_2,
            dimer.instance_id_1 or "",
            dimer.instance_id_2 or "",
        ),
    )


def identify_dimer_interfaces(
    path: str | Path,
    chain_inventory: list,
    *,
    model: int = 1,
    assembly_mode: str = "largest_assembly",
    assembly_id: str | None = None,
    drop_hydrogens_for_analysis: bool = True,
    residue_contact_cutoff: float = 8.0,
    atom_contact_cutoff: float = 5.0,
    min_residue_contacts: int = 3,
    min_atom_contacts: int = 20,
    cif_file: CIFFile | None = None,
) -> list[DimerInterfaceRecord]:
    cif_path = Path(path)
    cif_file = cif_file or read_cif_file(cif_path)
    try:
        assembly_ids = sorted(str(assembly_id) for assembly_id in list_assemblies(cif_file))
    except Exception:
        assembly_ids = []
    polymer_chains = [
        chain for chain in chain_inventory if chain.chain_type in POLYMER_CHAIN_TYPES
    ]
    chain_map = {chain.label_asym_id: chain for chain in polymer_chains}
    dimers: list[DimerInterfaceRecord] = []
    if assembly_mode in {"largest_assembly", "first_assembly", "all"} and assembly_ids:
        selected_assembly_id = assembly_id
        if selected_assembly_id is None and assembly_mode == "largest_assembly":
            selected_assembly_id = select_largest_polymer_assembly_id(cif_file)
        elif selected_assembly_id is None and assembly_mode == "first_assembly":
            selected_assembly_id = assembly_ids[0]
        if selected_assembly_id is None:
            return []
        try:
            atom_array_inputs = [
                (
                    selected_assembly_id,
                    get_assembly_with_altloc_fallback(
                        cif_file,
                        assembly_id=selected_assembly_id,
                        model=model,
                        use_author_fields=False,
                    ),
                )
            ]
        except ValueError as exc:
            if str(exc) != "Array must contain at least one element":
                raise
            return []
    elif assembly_mode == "all":
        return []
    else:
        try:
            atom_array_inputs = [
                (
                    None,
                    get_structure_with_altloc_fallback(
                        cif_file,
                        model=model,
                        use_author_fields=False,
                    ),
                )
            ]
        except ValueError as exc:
            if str(exc) != "Array must contain at least one element":
                raise
            return []
    geometries: dict[str, object] = {}
    if assembly_mode in {"largest_assembly", "first_assembly", "all"}:
        selected_assembly_id, atom_array = atom_array_inputs[0]
        built_geometries = build_instance_geometries(
            atom_array,
            polymer_chains,
            drop_hydrogens_for_analysis=drop_hydrogens_for_analysis,
        )
        geometries = {
            instance_id: replace(geometry, assembly_id=selected_assembly_id)
            for instance_id, geometry in built_geometries.items()
        }
    else:
        _, atom_array = atom_array_inputs[0]
        geometries = build_chain_geometries(
            atom_array,
            polymer_chains,
            drop_hydrogens_for_analysis=drop_hydrogens_for_analysis,
        )
        if assembly_mode == "input_assembly" and assembly_id is not None:
            geometries = {
                instance_id: replace(geometry, assembly_id=assembly_id)
                for instance_id, geometry in geometries.items()
            }

    ordered_instance_ids = sorted(geometries)
    for index, instance_id_1 in enumerate(ordered_instance_ids):
        geometry_1 = geometries[instance_id_1]
        label_asym_id_1 = geometry_1.label_asym_id
        if label_asym_id_1 not in chain_map:
            continue
        for instance_id_2 in ordered_instance_ids[index + 1 :]:
            geometry_2 = geometries[instance_id_2]
            if _bbox_distance_from_geometries(geometry_1, geometry_2) > residue_contact_cutoff:
                continue
            label_asym_id_2 = geometry_2.label_asym_id
            if label_asym_id_2 not in chain_map:
                continue

            chain_1 = chain_map[label_asym_id_1]
            chain_2 = chain_map[label_asym_id_2]
            metrics = compute_interface_metrics(
                geometry_1,
                geometry_2,
                residue_contact_cutoff=residue_contact_cutoff,
                atom_contact_cutoff=atom_contact_cutoff,
                min_residue_contacts=min_residue_contacts,
                min_atom_contacts=min_atom_contacts,
            )
            if metrics is None:
                continue

            output_instance_id_1 = geometry_1.instance_id
            output_instance_id_2 = geometry_2.instance_id
            contacting_atom_pairs = []
            for atom_pair in metrics.get("contacting_atom_pairs", []):
                normalized_pair = list(atom_pair)
                if len(normalized_pair) == 7:
                    normalized_pair[0] = output_instance_id_1
                    normalized_pair[3] = output_instance_id_2
                contacting_atom_pairs.append(normalized_pair)

            is_same_entity = chain_1.entity_id == chain_2.entity_id
            interface_residue_count_1 = int(metrics["interface_residue_count_1"])
            interface_residue_count_2 = int(metrics["interface_residue_count_2"])
            dimer_warnings = list(metrics.get("area_warnings", []))
            dimer_warning_details = metrics.get("area_warning_details", {})
            dimer_evidence = {
                "stage": "dimer_interface_v1",
                "metric": "representative_residue_contacts_plus_atom_contacts",
                "residue_contact_cutoff": residue_contact_cutoff,
                "atom_contact_cutoff": atom_contact_cutoff,
                "min_residue_contacts": min_residue_contacts,
                "min_atom_contacts": min_atom_contacts,
                "interface_area_metric": "buried_area_from_delta_sasa",
                "interface_area_method": str(
                    metrics.get("area_evidence", {}).get("interface_area_method", "ProtOr")
                ),
                "instance_granularity": (
                    "chain_id_plus_sym_id"
                    if assembly_mode in {"largest_assembly", "first_assembly", "all"}
                    else "pre_split_assembly_label_asym_id"
                    if assembly_mode == "input_assembly"
                    else "label_asym_id"
                ),
                "bbox_distance": round(float(metrics["bbox_distance"]), 4),
            }
            area_evidence = metrics.get("area_evidence", {})
            if isinstance(area_evidence, dict):
                dimer_evidence.update(area_evidence)
            if isinstance(dimer_warning_details, dict) and dimer_warning_details:
                dimer_evidence["warning_details"] = dimer_warning_details
            dimers.append(
                DimerInterfaceRecord(
                    pdb_id=chain_1.pdb_id,
                    assembly_mode=assembly_mode,
                    assembly_id=geometry_1.assembly_id
                    if assembly_mode in {"largest_assembly", "first_assembly", "all", "input_assembly"}
                    else None,
                    num_supporting_instance_pairs=1,
                    instance_id_1=output_instance_id_1,
                    sym_id_1=geometry_1.sym_id,
                    label_asym_id_1=chain_1.label_asym_id,
                    auth_asym_id_1=chain_1.auth_asym_id,
                    entity_id_1=chain_1.entity_id,
                    chain_type_1=chain_1.chain_type,
                    instance_id_2=output_instance_id_2,
                    sym_id_2=geometry_2.sym_id,
                    label_asym_id_2=chain_2.label_asym_id,
                    auth_asym_id_2=chain_2.auth_asym_id,
                    entity_id_2=chain_2.entity_id,
                    chain_type_2=chain_2.chain_type,
                    interface_residue_count_1=interface_residue_count_1,
                    interface_residue_count_2=interface_residue_count_2,
                    interface_residue_ratio_1=round(interface_residue_count_1 / max(chain_1.residue_count, 1), 6),
                    interface_residue_ratio_2=round(interface_residue_count_2 / max(chain_2.residue_count, 1), 6),
                    num_residue_contacts=int(metrics["num_residue_contacts"]),
                    num_atom_contacts=int(metrics["num_atom_contacts"]),
                    min_distance=round(float(metrics["min_distance"]), 4),
                    centroid_distance=round(float(metrics["centroid_distance"]), 4),
                    delta_sasa_1=round(float(metrics["delta_sasa_1"]), 4),
                    delta_sasa_2=round(float(metrics["delta_sasa_2"]), 4),
                    buried_area=round(float(metrics["buried_area"]), 4),
                    mean_interface_residue_count=round(float(metrics["mean_interface_residue_count"]), 4),
                    buried_area_per_interface_residue=round(
                        float(metrics["buried_area_per_interface_residue"]),
                        4,
                    ),
                    atom_contacts_per_interface_residue=round(
                        float(metrics["atom_contacts_per_interface_residue"]),
                        4,
                    ),
                    residue_contacts_per_interface_residue=round(
                        float(metrics["residue_contacts_per_interface_residue"]),
                        4,
                    ),
                    contacting_atom_pairs=contacting_atom_pairs,
                    is_same_entity=is_same_entity,
                    interface_label=_interface_label(
                        chain_1.chain_type,
                        chain_2.chain_type,
                        is_same_entity,
                    ),
                    contains_antibody_unit=_contains_antibody_unit(
                        chain_1.chain_type,
                        chain_2.chain_type,
                    ),
                    contains_tcr_pmhc_unit=_contains_tcr_pmhc_unit(
                        chain_1.chain_type,
                        chain_2.chain_type,
                    ),
                    evidence=dimer_evidence,
                    warnings=dimer_warnings,
                )
            )

            if chain_1.label_asym_id != chain_2.label_asym_id and chain_2.label_asym_id not in chain_1.bound_chain_ids:
                chain_1.bound_chain_ids.append(chain_2.label_asym_id)
            if (
                chain_1.auth_asym_id != chain_2.auth_asym_id
                and chain_2.auth_asym_id
                and chain_2.auth_asym_id not in chain_1.bound_auth_asym_ids
            ):
                chain_1.bound_auth_asym_ids.append(chain_2.auth_asym_id)
            if chain_1.label_asym_id != chain_2.label_asym_id and chain_1.label_asym_id not in chain_2.bound_chain_ids:
                chain_2.bound_chain_ids.append(chain_1.label_asym_id)
            if (
                chain_1.auth_asym_id != chain_2.auth_asym_id
                and chain_1.auth_asym_id
                and chain_1.auth_asym_id not in chain_2.bound_auth_asym_ids
            ):
                chain_2.bound_auth_asym_ids.append(chain_1.auth_asym_id)
    return _deduplicate_dimer_interfaces(dimers)
