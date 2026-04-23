from __future__ import annotations

from collections import defaultdict

import networkx as nx

from cif_parse.models import TightMultimerRecord


ANTIBODY_CHAIN_TYPES = frozenset({"antibody heavy chain", "antibody light chain"})
TCR_PMHC_CHAIN_TYPES = frozenset(
    {
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
    }
)


def _multimer_type(member_entity_ids: list[str], num_member_instances: int) -> str:
    if num_member_instances == 2 and len(set(member_entity_ids)) == 1:
        return "homodimer"
    if num_member_instances == 2:
        return "tight_dimer"
    if len(set(member_entity_ids)) == 1:
        return "homomultimer"
    return "heteromultimer"


def _instance_assembly_id(instance_id: str, fallback: str | None) -> str | None:
    if ":" in instance_id:
        prefix, _ = instance_id.split(":", 1)
        if prefix:
            return prefix
    return fallback


def _contains_antibody_unit(member_chain_types: list[str]) -> bool:
    return any(chain_type in ANTIBODY_CHAIN_TYPES for chain_type in member_chain_types)


def _contains_tcr_pmhc_unit(member_chain_types: list[str]) -> bool:
    return any(chain_type in TCR_PMHC_CHAIN_TYPES for chain_type in member_chain_types)


def _multimer_dedup_key(multimer: TightMultimerRecord) -> tuple:
    member_signature = tuple(
        sorted(
            (instance["label_asym_id"], instance["entity_id"], instance["chain_type"])
            for instance in multimer.member_instances
        )
    )
    base_edge_signature = tuple(
        sorted(
            tuple(
                sorted(
                    (
                        edge[0].split("@", 1)[0],
                        edge[1].split("@", 1)[0],
                    )
                )
            )
            for edge in multimer.evidence.get("internal_edges", [])
        )
    )
    return (
        multimer.assembly_mode,
        multimer.assembly_id,
        multimer.multimer_type,
        member_signature,
        base_edge_signature,
    )


def _deduplicate_tight_multimers(multimers: list[TightMultimerRecord]) -> list[TightMultimerRecord]:
    grouped: dict[tuple, list[TightMultimerRecord]] = {}
    for multimer in multimers:
        grouped.setdefault(_multimer_dedup_key(multimer), []).append(multimer)

    deduped: list[TightMultimerRecord] = []
    for group in grouped.values():
        representative = max(
            group,
            key=lambda multimer: (
                float(multimer.support_score),
                int(multimer.num_member_instances),
                int(multimer.num_internal_edges),
            ),
        )
        representative.num_component_copies = len(group)
        representative.evidence = {
            **representative.evidence,
            "dedup_stage": "tight_multimer_dedup_v1",
            "supporting_component_instance_groups": [
                [instance["instance_id"] for instance in multimer.member_instances]
                for multimer in group
            ],
        }
        if len(group) > 1 and "deduplicated_over_component_copies" not in representative.warnings:
            representative.warnings.append("deduplicated_over_component_copies")
        deduped.append(representative)

    return sorted(
        deduped,
        key=lambda multimer: (
            multimer.pdb_id,
            multimer.multimer_type,
            multimer.multimer_id,
        ),
    )


def _detect_communities(
    graph: nx.Graph,
    *,
    resolution: float,
    min_member_instances: int,
) -> tuple[list[list[str]], str]:
    if graph.number_of_edges() == 0:
        return [], "none"

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=0,
    )

    normalized = [
        sorted(community)
        for community in communities
        if len(community) >= min_member_instances
    ]
    normalized.sort(key=tuple)
    return normalized, "louvain_communities"


def identify_tight_multimers(
    chain_inventory: list,
    dimer_interfaces: list,
    *,
    assembly_mode: str,
    assembly_copy_numbers: dict[str, int] | None = None,
    assembly_chain_operations: dict[str, list[str]] | None = None,
    min_buried_area: float = 500.0,
    louvain_resolution: float = 1.0,
    min_member_instances: int = 2,
    large_component_warning_size: int = 8,
) -> list[TightMultimerRecord]:
    chain_map = {chain.label_asym_id: chain for chain in chain_inventory}
    assembly_copy_numbers = assembly_copy_numbers or {}
    assembly_chain_operations = assembly_chain_operations or {}
    graph = nx.Graph()
    edge_map: dict[tuple[str, str], object] = {}
    instance_metadata: dict[str, dict[str, object]] = {}

    for dimer in dimer_interfaces:
        instance_id_1 = dimer.instance_id_1 or dimer.label_asym_id_1
        instance_id_2 = dimer.instance_id_2 or dimer.label_asym_id_2
        instance_metadata[instance_id_1] = {
            "instance_id": instance_id_1,
            "label_asym_id": dimer.label_asym_id_1,
            "auth_asym_id": dimer.auth_asym_id_1,
            "entity_id": dimer.entity_id_1,
            "chain_type": dimer.chain_type_1,
            "sym_id": dimer.sym_id_1,
            "assembly_id": _instance_assembly_id(instance_id_1, dimer.assembly_id),
        }
        instance_metadata[instance_id_2] = {
            "instance_id": instance_id_2,
            "label_asym_id": dimer.label_asym_id_2,
            "auth_asym_id": dimer.auth_asym_id_2,
            "entity_id": dimer.entity_id_2,
            "chain_type": dimer.chain_type_2,
            "sym_id": dimer.sym_id_2,
            "assembly_id": _instance_assembly_id(instance_id_2, dimer.assembly_id),
        }
        buried_area = float(dimer.buried_area or 0.0)
        if buried_area < min_buried_area:
            continue
        edge = tuple(sorted((instance_id_1, instance_id_2)))
        edge_map[edge] = dimer
        graph.add_edge(
            edge[0],
            edge[1],
            weight=buried_area / 1000.0,
            buried_area=buried_area,
        )

    components, clustering_method = _detect_communities(
        graph,
        resolution=louvain_resolution,
        min_member_instances=min_member_instances,
    )

    multimers: list[TightMultimerRecord] = []
    for index, component in enumerate(components, start=1):
        internal_edges = [
            edge
            for edge in edge_map
            if edge[0] in component and edge[1] in component
        ]
        component_instances = [instance_metadata[instance_id] for instance_id in component if instance_id in instance_metadata]
        component_assembly_ids = sorted(
            {
                str(instance["assembly_id"])
                for instance in component_instances
                if instance.get("assembly_id") is not None
            }
        )
        assembly_id = component_assembly_ids[0] if len(component_assembly_ids) == 1 else None
        component_by_chain: dict[str, list[dict[str, object]]] = defaultdict(list)
        for instance in component_instances:
            component_by_chain[str(instance["label_asym_id"])].append(instance)

        member_records = [
            chain_map[label_asym_id]
            for label_asym_id in sorted(component_by_chain)
            if label_asym_id in chain_map
        ]
        member_chain_types = [chain.chain_type for chain in member_records]
        member_entity_ids = [chain.entity_id for chain in member_records]
        member_copy_numbers = [
            int(assembly_copy_numbers.get(chain.label_asym_id, len(component_by_chain.get(chain.label_asym_id, [])) or 1))
            for chain in member_records
        ]
        member_instances: list[dict[str, object]] = []
        for chain in member_records:
            operation_ids = assembly_chain_operations.get(chain.label_asym_id, ["1"])
            known_operations = {
                str(operation_id): copy_ordinal
                for copy_ordinal, operation_id in enumerate(
                    operation_ids,
                    start=1,
                )
            }
            for instance in sorted(component_by_chain.get(chain.label_asym_id, []), key=lambda item: str(item["instance_id"])):
                sym_id = instance.get("sym_id")
                if isinstance(sym_id, int) and 0 <= sym_id < len(operation_ids):
                    operation_id = str(operation_ids[sym_id])
                elif sym_id is None:
                    operation_id = "1"
                else:
                    operation_id = str(sym_id)
                copy_ordinal = known_operations.get(operation_id)
                if copy_ordinal is None:
                    copy_ordinal = len(component_by_chain.get(chain.label_asym_id, [])) or 1
                member_instances.append(
                    {
                        "instance_id": instance["instance_id"],
                        "label_asym_id": chain.label_asym_id,
                        "auth_asym_id": chain.auth_asym_id,
                        "entity_id": chain.entity_id,
                        "chain_type": chain.chain_type,
                        "assembly_id": instance.get("assembly_id"),
                        "operation_id": operation_id,
                        "copy_ordinal": copy_ordinal,
                    }
                )
        warnings: list[str] = []
        if len(component) >= large_component_warning_size:
            warnings.append("large_component_without_bridge_pruning")

        support_score = (
            round(
                sum(float(edge_map[edge].num_atom_contacts) for edge in internal_edges) / len(internal_edges),
                4,
            )
            if internal_edges
            else 0.0
        )
        multimers.append(
            TightMultimerRecord(
                pdb_id=member_records[0].pdb_id if member_records else "",
                assembly_mode=assembly_mode,
                assembly_id=assembly_id,
                multimer_id=f"tm_{index:03d}",
                num_component_copies=1,
                member_chain_ids=[chain.label_asym_id for chain in member_records],
                member_auth_asym_ids=[chain.auth_asym_id for chain in member_records],
                member_entity_ids=member_entity_ids,
                member_chain_types=member_chain_types,
                member_copy_numbers=member_copy_numbers,
                member_instances=member_instances,
                num_members=len(member_records),
                num_member_instances=len(component_instances),
                num_internal_edges=len(internal_edges),
                multimer_type=_multimer_type(member_entity_ids, len(component_instances)),
                support_score=support_score,
                contains_antibody_unit=_contains_antibody_unit(member_chain_types),
                contains_tcr_pmhc_unit=_contains_tcr_pmhc_unit(member_chain_types),
                evidence={
                    "stage": "tight_multimer_v2",
                    "graph_rule": "weighted_instance_level_dimer_graph_with_louvain_communities",
                    "source_interface_stage": "dimer_interface_v2_assembly_instances",
                    "support_score_metric": "average_atom_contacts_per_internal_edge",
                    "edge_min_buried_area": min_buried_area,
                    "edge_weight_metric": "buried_area / 1000.0",
                    "community_detection_method": clustering_method,
                    "community_detection_resolution": louvain_resolution,
                    "bridge_pruning_applied": False,
                    "min_member_instances": min_member_instances,
                    "large_component_warning_size": large_component_warning_size,
                    "member_instance_source": "dimer_interface_instances",
                    "member_copy_number_source": (
                        "pdbx_struct_assembly_gen" if assembly_copy_numbers else "default_one"
                    ),
                    "component_assembly_ids": component_assembly_ids,
                    "graph_edge_count_after_buried_area_filter": graph.number_of_edges(),
                    "internal_edges": [list(edge) for edge in sorted(internal_edges)],
                },
                warnings=warnings,
            )
        )
    return _deduplicate_tight_multimers(multimers)
