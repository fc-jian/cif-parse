from __future__ import annotations

from pathlib import Path

from biotite.structure.io.pdbx import get_assembly, get_structure, list_assemblies

from mmcif_parse.interact.contacts import (
    build_chain_geometries,
    build_instance_geometries,
    compute_interface_metrics,
)
from mmcif_parse.io import read_cif_file
from mmcif_parse.models import DimerInterfaceRecord


POLYMER_CHAIN_TYPES = frozenset(
    {
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
        "other polymer chain",
    }
)


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
    assembly_mode: str = "biological_assembly",
) -> list[DimerInterfaceRecord]:
    cif_path = Path(path)
    cif_file = read_cif_file(cif_path)
    assembly_ids = sorted(str(assembly_id) for assembly_id in list_assemblies(cif_file))
    polymer_chains = [
        chain for chain in chain_inventory if chain.chain_type in POLYMER_CHAIN_TYPES
    ]
    chain_map = {chain.label_asym_id: chain for chain in polymer_chains}
    dimers: list[DimerInterfaceRecord] = []
    if assembly_mode == "biological_assembly" and assembly_ids:
        atom_array_inputs = [
            (
                assembly_id,
                get_assembly(
                    cif_file,
                    assembly_id=assembly_id,
                    model=model,
                    use_author_fields=False,
                ),
            )
            for assembly_id in assembly_ids
        ]
    else:
        atom_array_inputs = [
            (
                None,
                get_structure(
                    cif_file,
                    model=model,
                    use_author_fields=False,
                ),
            )
        ]

    use_assembly_prefix = assembly_mode == "biological_assembly" and len(atom_array_inputs) > 1
    for assembly_id, atom_array in atom_array_inputs:
        if assembly_mode == "biological_assembly" and hasattr(atom_array, "sym_id"):
            geometries = build_instance_geometries(atom_array, polymer_chains)
        else:
            geometries = build_chain_geometries(atom_array, polymer_chains)

        ordered_instance_ids = sorted(geometries)
        for index, instance_id_1 in enumerate(ordered_instance_ids):
            geometry_1 = geometries[instance_id_1]
            label_asym_id_1 = geometry_1.label_asym_id
            if label_asym_id_1 not in chain_map:
                continue
            for instance_id_2 in ordered_instance_ids[index + 1 :]:
                geometry_2 = geometries[instance_id_2]
                label_asym_id_2 = geometry_2.label_asym_id
                if label_asym_id_2 not in chain_map:
                    continue

                chain_1 = chain_map[label_asym_id_1]
                chain_2 = chain_map[label_asym_id_2]
                metrics = compute_interface_metrics(
                    geometry_1,
                    geometry_2,
                )
                if metrics is None:
                    continue

                output_instance_id_1 = geometry_1.instance_id
                output_instance_id_2 = geometry_2.instance_id
                if use_assembly_prefix and assembly_id is not None:
                    output_instance_id_1 = f"{assembly_id}:{output_instance_id_1}"
                    output_instance_id_2 = f"{assembly_id}:{output_instance_id_2}"

                is_same_entity = chain_1.entity_id == chain_2.entity_id
                interface_residue_count_1 = int(metrics["interface_residue_count_1"])
                interface_residue_count_2 = int(metrics["interface_residue_count_2"])
                dimers.append(
                    DimerInterfaceRecord(
                        pdb_id=chain_1.pdb_id,
                        assembly_mode=assembly_mode,
                        assembly_id=assembly_id,
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
                        evidence={
                            "stage": "dimer_interface_v1",
                            "metric": "representative_residue_contacts_plus_atom_contacts",
                            "residue_contact_cutoff": 8.0,
                            "atom_contact_cutoff": 5.0,
                            "min_residue_contacts": 3,
                            "min_atom_contacts": 20,
                            "interface_area_metric": "buried_area_from_delta_sasa",
                            "instance_granularity": "chain_id_plus_sym_id"
                            if assembly_mode == "biological_assembly"
                            else "label_asym_id",
                            "bbox_distance": round(float(metrics["bbox_distance"]), 4),
                        },
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
