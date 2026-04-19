from __future__ import annotations

from collections import defaultdict
from typing import Any

from cif_parse.models import AntibodyAntigenComplexRecord


ANTIBODY_CHAIN_TYPES = frozenset({"antibody heavy chain", "antibody light chain"})


def _antibody_unit_type(chain: Any) -> str:
    features = chain.features if isinstance(getattr(chain, "features", None), dict) else {}
    analysis = features.get("antibody_analysis", {})
    if isinstance(analysis, dict):
        unit_type = analysis.get("unit_type")
        if isinstance(unit_type, str) and unit_type:
            return unit_type
    feature_unit_type = features.get("antibody_unit_type")
    if isinstance(feature_unit_type, str) and feature_unit_type:
        return feature_unit_type
    if chain.subtype in {"VHH", "scFv"}:
        return str(chain.subtype)
    if chain.paired_label_asym_id:
        return "paired_heavy_light"
    return "heavy_only"


def _supporting_multimer_ids(
    antibody_chain_ids: set[str],
    antigen_chain_ids: set[str],
    tight_multimers: list[Any],
) -> list[str]:
    multimer_ids: list[str] = []
    for multimer in tight_multimers:
        member_ids = set(multimer.member_chain_ids)
        if antibody_chain_ids.issubset(member_ids) and antigen_chain_ids & member_ids:
            multimer_ids.append(multimer.multimer_id)
    return sorted(multimer_ids)


def _collapse_antigen_groups(
    antigen_groups: dict[str, list[dict[str, Any]]],
    chain_map: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]], bool]:
    entity_groups: dict[str, dict[str, Any]] = {}
    for antigen_label, items in antigen_groups.items():
        antigen_chain = chain_map.get(antigen_label)
        if antigen_chain is None:
            continue
        entity_group = entity_groups.setdefault(
            antigen_chain.entity_id,
            {
                "entity_id": antigen_chain.entity_id,
                "chain_type": antigen_chain.chain_type,
                "chains": [],
            },
        )
        entity_group["chains"].append(
            {
                "chain": antigen_chain,
                "items": items,
                "total_buried_area": round(
                    sum(float(entry["dimer"].buried_area) for entry in items),
                    4,
                ),
                "total_atom_contacts": sum(
                    int(entry["dimer"].num_atom_contacts) for entry in items
                ),
            }
        )

    collapsed = False
    representative_records: list[Any] = []
    summaries: list[dict[str, Any]] = []
    sorted_groups = sorted(
        entity_groups.values(),
        key=lambda group: (
            -sum(float(chain_entry["total_buried_area"]) for chain_entry in group["chains"]),
            min(chain_entry["chain"].label_asym_id for chain_entry in group["chains"]),
        ),
    )
    for group in sorted_groups:
        chain_entries = sorted(
            group["chains"],
            key=lambda entry: (
                -float(entry["total_buried_area"]),
                -int(entry["total_atom_contacts"]),
                entry["chain"].label_asym_id,
            ),
        )
        representative = chain_entries[0]
        representative_records.append(representative["chain"])
        if len(chain_entries) > 1:
            collapsed = True

        all_items = [item for entry in chain_entries for item in entry["items"]]
        summaries.append(
            {
                "antigen_label_asym_id": representative["chain"].label_asym_id,
                "antigen_auth_asym_id": representative["chain"].auth_asym_id,
                "antigen_entity_id": representative["chain"].entity_id,
                "antigen_chain_type": representative["chain"].chain_type,
                "supporting_antibody_chain_ids": sorted(
                    {entry["antibody_label_asym_id"] for entry in all_items}
                ),
                "num_supporting_dimers": len(all_items),
                "total_buried_area": round(
                    sum(float(entry["dimer"].buried_area) for entry in all_items),
                    4,
                ),
                "total_atom_contacts": sum(
                    int(entry["dimer"].num_atom_contacts) for entry in all_items
                ),
                "interface_pairs": sorted(
                    [
                        [
                            entry["antibody_label_asym_id"],
                            entry["antigen_label_asym_id"],
                        ]
                        for entry in all_items
                    ]
                ),
                "all_antigen_chain_ids": [
                    entry["chain"].label_asym_id for entry in chain_entries
                ],
                "all_antigen_auth_asym_ids": [
                    entry["chain"].auth_asym_id for entry in chain_entries
                ],
                "supporting_same_entity_chain_ids": [
                    entry["chain"].label_asym_id for entry in chain_entries[1:]
                ],
                "supporting_same_entity_auth_asym_ids": [
                    entry["chain"].auth_asym_id for entry in chain_entries[1:]
                ],
            }
        )

    return representative_records, summaries, collapsed


def identify_antibody_antigen_complexes(
    chain_inventory: list[Any],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
) -> list[AntibodyAntigenComplexRecord]:
    chain_map = {chain.label_asym_id: chain for chain in chain_inventory}
    antibody_units: list[dict[str, Any]] = []
    seen_units: set[tuple[str, ...]] = set()

    for chain in sorted(chain_inventory, key=lambda item: (item.label_asym_id, item.auth_asym_id or "")):
        if chain.chain_type != "antibody heavy chain":
            continue
        unit_type = _antibody_unit_type(chain)
        light_chain = None
        if unit_type not in {"VHH", "scFv"} and chain.paired_label_asym_id:
            partner = chain_map.get(chain.paired_label_asym_id)
            if partner is not None and partner.chain_type == "antibody light chain":
                light_chain = partner
                unit_type = "paired_heavy_light"
        antibody_chains = [chain] + ([light_chain] if light_chain is not None else [])
        unit_key = tuple(sorted(member.label_asym_id for member in antibody_chains))
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)
        antibody_units.append(
            {
                "unit_type": unit_type,
                "heavy_chain": chain,
                "light_chain": light_chain,
                "antibody_chains": antibody_chains,
            }
        )

    complexes: list[AntibodyAntigenComplexRecord] = []
    sorted_units = sorted(
        antibody_units,
        key=lambda unit: (
            tuple(member.label_asym_id for member in unit["antibody_chains"]),
            unit["unit_type"],
        ),
    )

    for index, unit in enumerate(sorted_units, start=1):
        heavy_chain = unit["heavy_chain"]
        light_chain = unit["light_chain"]
        antibody_chains = unit["antibody_chains"]
        antibody_chain_ids = {chain.label_asym_id for chain in antibody_chains}
        relevant_dimers = []
        for dimer in dimer_interfaces:
            left_is_antibody = dimer.label_asym_id_1 in antibody_chain_ids
            right_is_antibody = dimer.label_asym_id_2 in antibody_chain_ids
            if left_is_antibody == right_is_antibody:
                continue

            if left_is_antibody:
                antigen_label = dimer.label_asym_id_2
                antibody_label = dimer.label_asym_id_1
                antigen_auth = dimer.auth_asym_id_2
                antigen_entity = dimer.entity_id_2
                antigen_type = dimer.chain_type_2
            else:
                antigen_label = dimer.label_asym_id_1
                antibody_label = dimer.label_asym_id_2
                antigen_auth = dimer.auth_asym_id_1
                antigen_entity = dimer.entity_id_1
                antigen_type = dimer.chain_type_1

            if antigen_type in ANTIBODY_CHAIN_TYPES:
                continue
            antigen_chain = chain_map.get(antigen_label)
            if antigen_chain is None:
                continue

            relevant_dimers.append(
                {
                    "dimer": dimer,
                    "antibody_label_asym_id": antibody_label,
                    "antigen_label_asym_id": antigen_label,
                    "antigen_auth_asym_id": antigen_auth,
                    "antigen_entity_id": antigen_entity,
                    "antigen_chain_type": antigen_type,
                }
            )

        if not relevant_dimers:
            continue

        antigen_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in relevant_dimers:
            antigen_groups[item["antigen_label_asym_id"]].append(item)

        antigen_chain_records, antigen_interface_summaries, collapsed_same_entity_antigens = (
            _collapse_antigen_groups(antigen_groups, chain_map)
        )
        auxiliary_component_ids = sorted(
            {
                component_id
                for chain in antibody_chains + antigen_chain_records
                for component_id in chain.covered_nonpolymer_ids
            }
        )
        auxiliary_component_auth_asym_ids = sorted(
            {
                component_id
                for chain in antibody_chains + antigen_chain_records
                for component_id in chain.covered_nonpolymer_auth_asym_ids
            }
        )
        auxiliary_branched_ids = sorted(
            {
                component_id
                for chain in antibody_chains + antigen_chain_records
                for component_id in chain.covered_branched_ids
            }
        )
        auxiliary_branched_auth_asym_ids = sorted(
            {
                component_id
                for chain in antibody_chains + antigen_chain_records
                for component_id in chain.covered_branched_auth_asym_ids
            }
        )

        warnings: list[str] = []
        if unit["unit_type"] == "heavy_only":
            warnings.append("heavy_chain_without_paired_light_chain")
        if collapsed_same_entity_antigens:
            warnings.append("same_entity_antigen_copies_collapsed")

        if light_chain is not None:
            heavy_has_direct_antigen = any(
                entry["antibody_label_asym_id"] == heavy_chain.label_asym_id
                for entry in relevant_dimers
            )
            if not heavy_has_direct_antigen:
                warnings.append("paired_unit_antigen_contacts_not_on_heavy_chain")

        supporting_multimers = _supporting_multimer_ids(
            antibody_chain_ids,
            {chain.label_asym_id for chain in antigen_chain_records},
            tight_multimers,
        )
        complexes.append(
            AntibodyAntigenComplexRecord(
                pdb_id=heavy_chain.pdb_id,
                assembly_mode=relevant_dimers[0]["dimer"].assembly_mode,
                assembly_id=relevant_dimers[0]["dimer"].assembly_id,
                complex_id=f"ab_ag_{index:03d}",
                antibody_unit_type=unit["unit_type"],
                antibody_heavy_chain=heavy_chain.label_asym_id,
                antibody_heavy_auth_asym_id=heavy_chain.auth_asym_id,
                antibody_light_chain=light_chain.label_asym_id if light_chain is not None else None,
                antibody_light_auth_asym_id=light_chain.auth_asym_id if light_chain is not None else None,
                antibody_chain_ids=[chain.label_asym_id for chain in antibody_chains],
                antibody_auth_asym_ids=[chain.auth_asym_id for chain in antibody_chains],
                antibody_entity_ids=[chain.entity_id for chain in antibody_chains],
                antigen_chain_ids=[chain.label_asym_id for chain in antigen_chain_records],
                antigen_auth_asym_ids=[chain.auth_asym_id for chain in antigen_chain_records],
                antigen_entity_ids=[chain.entity_id for chain in antigen_chain_records],
                antigen_chain_types=[chain.chain_type for chain in antigen_chain_records],
                auxiliary_component_ids=auxiliary_component_ids,
                auxiliary_component_auth_asym_ids=auxiliary_component_auth_asym_ids,
                auxiliary_branched_ids=auxiliary_branched_ids,
                auxiliary_branched_auth_asym_ids=auxiliary_branched_auth_asym_ids,
                num_antigen_chains=len(antigen_chain_records),
                num_antibody_antigen_interfaces=len(relevant_dimers),
                contact_score=round(
                    sum(float(entry["dimer"].buried_area) for entry in relevant_dimers),
                    4,
                ),
                evidence={
                    "stage": "antibody_antigen_complex_v1",
                    "unit_rule": (
                        "paired_heavy_light_or_single_chain_vhh_scfv_or_heavy_only"
                    ),
                    "antigen_selection_rule": (
                        "non_antibody_polymer_chains_with_direct_dimer_interfaces_to_antibody_unit"
                    ),
                    "contact_score_metric": "sum_buried_area_over_antibody_antigen_dimers",
                    "source_interface_stage": "dimer_interface_v2_assembly_instances",
                    "source_chain_annotation_files": [
                        "final/protein_chains.json",
                        "final/chain_inventory.json",
                    ],
                    "antigen_entity_grouping_rule": (
                        "collapse_same_entity_antigen_copies_to_representative_chain_for_top_level_fields"
                    ),
                    "supporting_multimer_ids": supporting_multimers,
                    "antigen_interface_summaries": antigen_interface_summaries,
                },
                warnings=warnings,
            )
        )

    return sorted(
        complexes,
        key=lambda complex_record: (
            complex_record.pdb_id,
            complex_record.antibody_heavy_chain or "",
            complex_record.antibody_light_chain or "",
            complex_record.complex_id,
        ),
    )
