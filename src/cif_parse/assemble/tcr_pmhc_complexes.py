from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from cif_parse.constants import (
    AUX_CHAIN_TYPE,
    MHC_CHAIN_TYPE,
    PEPTIDE_CHAIN_TYPE,
    PEPTIDE_MAX_LENGTH,
    TCR_CHAIN_TYPE,
    TCR_PAIR_TYPES,
)
from cif_parse.models import TcrPmhcComplexRecord


def _tcr_pair_type(chain_a: Any, chain_b: Any) -> str | None:
    subtype_set = frozenset({chain_a.subtype or "unknown", chain_b.subtype or "unknown"})
    return TCR_PAIR_TYPES.get(subtype_set)


def _build_tcr_units(chain_inventory: list[Any], dimer_interfaces: list[Any]) -> list[dict[str, Any]]:
    chain_map = {chain.label_asym_id: chain for chain in chain_inventory if chain.chain_type == TCR_CHAIN_TYPE}
    candidates: list[tuple[float, int, str, str, str]] = []
    for dimer in dimer_interfaces:
        if {dimer.chain_type_1, dimer.chain_type_2} != {TCR_CHAIN_TYPE}:
            continue
        chain_1 = chain_map.get(dimer.label_asym_id_1)
        chain_2 = chain_map.get(dimer.label_asym_id_2)
        if chain_1 is None or chain_2 is None:
            continue
        pair_type = _tcr_pair_type(chain_1, chain_2)
        if pair_type is None:
            continue
        left, right = sorted((chain_1.label_asym_id, chain_2.label_asym_id))
        candidates.append(
            (
                float(dimer.buried_area),
                int(dimer.num_atom_contacts),
                left,
                right,
                pair_type,
            )
        )

    used: set[str] = set()
    units: list[dict[str, Any]] = []
    for _, _, left, right, pair_type in sorted(candidates, reverse=True):
        if left in used or right in used:
            continue
        used.update({left, right})
        units.append(
            {
                "tcr_chains": [chain_map[left], chain_map[right]],
                "tcr_type": pair_type,
                "warnings": [],
            }
        )

    for chain in sorted(chain_map.values(), key=lambda item: item.label_asym_id):
        if chain.label_asym_id in used:
            continue
        units.append(
            {
                "tcr_chains": [chain],
                "tcr_type": chain.subtype or "single_tcr",
                "warnings": ["unpaired_tcr_chain"],
            }
        )

    return units


def _mhc_class(chain: Any) -> str:
    features = chain.features if isinstance(getattr(chain, "features", None), dict) else {}
    mhc_class = features.get("mhc_class")
    if isinstance(mhc_class, str) and mhc_class:
        return mhc_class
    if chain.subtype and chain.subtype.startswith("class_"):
        return chain.subtype.removeprefix("class_").upper()
    return "unknown"


def _mhc_role_from_description(description_lower: str, mhc_class: str) -> str:
    if mhc_class == "I":
        return "class_i_heavy"
    if mhc_class != "II":
        return "unknown"

    if re.search(r"\balpha\b", description_lower):
        return "class_ii_alpha"
    if re.search(r"\bbeta\b", description_lower):
        return "class_ii_beta"

    if any(
        marker in description_lower
        for marker in (
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
    ):
        return "class_ii_alpha"
    if any(
        marker in description_lower
        for marker in (
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
    ):
        return "class_ii_beta"
    return "unknown"


def _mhc_role(chain: Any) -> str:
    features = chain.features if isinstance(getattr(chain, "features", None), dict) else {}
    mhc_role = features.get("mhc_role")
    if isinstance(mhc_role, str) and mhc_role:
        return mhc_role

    mhc_class = _mhc_class(chain)
    description = getattr(chain, "entity_description", None)
    if isinstance(description, str) and description:
        return _mhc_role_from_description(description.lower(), mhc_class)
    if mhc_class == "I":
        return "class_i_heavy"
    return "unknown"


def _mhc_role_rank(role: str) -> int:
    if role == "class_ii_alpha":
        return 0
    if role == "class_ii_beta":
        return 1
    if role == "class_i_heavy":
        return 0
    return 2


def _is_length_limited_peptide_candidate(chain: Any, peptide_max_length: int = 30) -> bool:
    if chain.chain_type not in {PEPTIDE_CHAIN_TYPE, "other protein chain"}:
        return False
    return int(chain.length) <= peptide_max_length


def _multimer_member_map(tight_multimers: list[Any]) -> dict[str, set[str]]:
    member_map: dict[str, set[str]] = defaultdict(set)
    for multimer in tight_multimers:
        members = set(multimer.member_chain_ids)
        for member in members:
            member_map[member].update(members)
    return member_map


def _build_pmhc_units(
    chain_inventory: list[Any],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
) -> list[dict[str, Any]]:
    chain_map = {chain.label_asym_id: chain for chain in chain_inventory}
    multimer_members = _multimer_member_map(tight_multimers)
    mhc_chains = [chain for chain in chain_inventory if chain.chain_type == MHC_CHAIN_TYPE]
    mhc_chain_map = {chain.label_asym_id: chain for chain in mhc_chains}
    class_ii_pair_candidates: list[tuple[float, int, str, str]] = []
    for dimer in dimer_interfaces:
        if {dimer.chain_type_1, dimer.chain_type_2} != {MHC_CHAIN_TYPE}:
            continue
        left = mhc_chain_map.get(dimer.label_asym_id_1)
        right = mhc_chain_map.get(dimer.label_asym_id_2)
        if left is None or right is None:
            continue
        if _mhc_class(left) != "II" or _mhc_class(right) != "II":
            continue
        left_role = _mhc_role(left)
        right_role = _mhc_role(right)
        if left_role != "unknown" and right_role != "unknown" and left_role == right_role:
            continue
        low, high = sorted((left.label_asym_id, right.label_asym_id))
        class_ii_pair_candidates.append(
            (
                float(dimer.buried_area),
                int(dimer.num_atom_contacts),
                low,
                high,
            )
        )

    class_ii_pairs: dict[str, str] = {}
    used_class_ii: set[str] = set()
    for _, _, left, right in sorted(class_ii_pair_candidates, reverse=True):
        if left in used_class_ii or right in used_class_ii:
            continue
        used_class_ii.update({left, right})
        class_ii_pairs[left] = right
        class_ii_pairs[right] = left

    pmhc_units: list[dict[str, Any]] = []
    processed_mhc_labels: set[str] = set()
    for mhc_chain in sorted(mhc_chains, key=lambda item: item.label_asym_id):
        if mhc_chain.label_asym_id in processed_mhc_labels:
            continue

        mhc_members = [mhc_chain]
        class_partner_label = class_ii_pairs.get(mhc_chain.label_asym_id)
        if class_partner_label:
            class_partner = mhc_chain_map.get(class_partner_label)
            if class_partner is not None:
                mhc_members.append(class_partner)
        processed_mhc_labels.update(chain.label_asym_id for chain in mhc_members)
        mhc_members = sorted(
            mhc_members,
            key=lambda chain: (_mhc_role_rank(_mhc_role(chain)), chain.label_asym_id),
        )

        auxiliaries: dict[str, Any] = {}
        for mhc_member in mhc_members:
            for dimer in dimer_interfaces:
                if mhc_member.label_asym_id not in {dimer.label_asym_id_1, dimer.label_asym_id_2}:
                    continue

                if dimer.label_asym_id_1 == mhc_member.label_asym_id:
                    partner_label = dimer.label_asym_id_2
                    partner_type = dimer.chain_type_2
                else:
                    partner_label = dimer.label_asym_id_1
                    partner_type = dimer.chain_type_1
                partner = chain_map.get(partner_label)
                if partner is None:
                    continue
                if partner_type == AUX_CHAIN_TYPE:
                    auxiliaries[partner_label] = partner

            for member_label in sorted(multimer_members.get(mhc_member.label_asym_id, set())):
                if member_label == mhc_member.label_asym_id:
                    continue
                partner = chain_map.get(member_label)
                if partner is None:
                    continue
                if partner.chain_type == AUX_CHAIN_TYPE:
                    auxiliaries.setdefault(member_label, partner)

        warnings: list[str] = []
        mhc_class = _mhc_class(mhc_chain)
        if mhc_class == "I" and not auxiliaries:
            warnings.append("class_i_mhc_without_beta2m_or_auxiliary_chain")

        pmhc_units.append(
            {
                "mhc_chains": mhc_members,
                "mhc_class": mhc_class,
                "auxiliaries": [auxiliaries[key] for key in sorted(auxiliaries)],
                "warnings": warnings,
            }
        )
    return pmhc_units


def _select_peptide_chains_for_complex(
    chain_inventory: list[Any],
    dimer_interfaces: list[Any],
    tcr_chain_ids: set[str],
    mhc_chain_ids: set[str],
    peptide_max_length: int = 30,
) -> tuple[list[Any], list[str]]:
    chain_map = {chain.label_asym_id: chain for chain in chain_inventory}
    mhc_contact_labels: set[str] = set()
    tcr_contact_labels: set[str] = set()

    for dimer in dimer_interfaces:
        partner_label: str | None = None
        if dimer.label_asym_id_1 in mhc_chain_ids:
            partner_label = dimer.label_asym_id_2
        elif dimer.label_asym_id_2 in mhc_chain_ids:
            partner_label = dimer.label_asym_id_1
        if partner_label is not None:
            partner = chain_map.get(partner_label)
            if partner is not None and _is_length_limited_peptide_candidate(partner, peptide_max_length=peptide_max_length):
                mhc_contact_labels.add(partner_label)

        partner_label: str | None = None
        if dimer.label_asym_id_1 in tcr_chain_ids:
            partner_label = dimer.label_asym_id_2
        elif dimer.label_asym_id_2 in tcr_chain_ids:
            partner_label = dimer.label_asym_id_1
        if partner_label is None:
            continue
        partner = chain_map.get(partner_label)
        if partner is not None and _is_length_limited_peptide_candidate(partner, peptide_max_length=peptide_max_length):
            tcr_contact_labels.add(partner_label)

    peptide_labels = sorted(mhc_contact_labels & tcr_contact_labels)
    peptides = [chain_map[label] for label in peptide_labels if label in chain_map]
    contextual_ids = [chain.label_asym_id for chain in peptides if chain.chain_type != PEPTIDE_CHAIN_TYPE]
    return peptides, contextual_ids


def _supporting_multimer_ids(
    tcr_chain_ids: set[str],
    pmhc_chain_ids: set[str],
    tight_multimers: list[Any],
) -> list[str]:
    multimer_ids: list[str] = []
    for multimer in tight_multimers:
        member_ids = set(multimer.member_chain_ids)
        if tcr_chain_ids.issubset(member_ids) and pmhc_chain_ids.issubset(member_ids):
            multimer_ids.append(multimer.multimer_id)
    return sorted(multimer_ids)


def identify_tcr_pmhc_complexes(
    chain_inventory: list[Any],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
    peptide_max_length: int = 30,
) -> list[TcrPmhcComplexRecord]:
    chain_map = {chain.label_asym_id: chain for chain in chain_inventory}
    tcr_units = _build_tcr_units(chain_inventory, dimer_interfaces)
    pmhc_units = _build_pmhc_units(chain_inventory, dimer_interfaces, tight_multimers)

    complexes: list[TcrPmhcComplexRecord] = []
    index = 1
    for tcr_unit in tcr_units:
        tcr_chains = tcr_unit["tcr_chains"]
        tcr_chain_ids = {chain.label_asym_id for chain in tcr_chains}
        for pmhc_unit in pmhc_units:
            mhc_chains = pmhc_unit["mhc_chains"]
            auxiliaries = pmhc_unit["auxiliaries"]
            mhc_chain_ids = {chain.label_asym_id for chain in mhc_chains}
            peptides, contextual_peptide_chain_ids = _select_peptide_chains_for_complex(
                chain_inventory,
                dimer_interfaces,
                tcr_chain_ids,
                mhc_chain_ids,
                peptide_max_length=peptide_max_length,
            )
            pmhc_member_ids = {
                *mhc_chain_ids,
                *[chain.label_asym_id for chain in peptides],
                *[chain.label_asym_id for chain in auxiliaries],
            }

            relevant_dimers = []
            for dimer in dimer_interfaces:
                left_is_tcr = dimer.label_asym_id_1 in tcr_chain_ids
                right_is_tcr = dimer.label_asym_id_2 in tcr_chain_ids
                left_is_pmhc = dimer.label_asym_id_1 in pmhc_member_ids
                right_is_pmhc = dimer.label_asym_id_2 in pmhc_member_ids
                if not ((left_is_tcr and right_is_pmhc) or (right_is_tcr and left_is_pmhc)):
                    continue

                if left_is_tcr:
                    tcr_label = dimer.label_asym_id_1
                    pmhc_label = dimer.label_asym_id_2
                    pmhc_type = dimer.chain_type_2
                else:
                    tcr_label = dimer.label_asym_id_2
                    pmhc_label = dimer.label_asym_id_1
                    pmhc_type = dimer.chain_type_1
                relevant_dimers.append(
                    {
                        "dimer": dimer,
                        "tcr_label_asym_id": tcr_label,
                        "pmhc_label_asym_id": pmhc_label,
                        "pmhc_chain_type": pmhc_type,
                    }
                )

            if not relevant_dimers:
                continue

            direct_core_contact = any(
                entry["pmhc_label_asym_id"] in mhc_chain_ids
                or entry["pmhc_label_asym_id"] in {chain.label_asym_id for chain in peptides}
                for entry in relevant_dimers
            )
            if not direct_core_contact:
                continue

            member_summaries = []
            member_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entry in relevant_dimers:
                member_groups[entry["pmhc_label_asym_id"]].append(entry)
            for member_label in sorted(
                member_groups,
                key=lambda label: (
                    -sum(float(entry["dimer"].buried_area) for entry in member_groups[label]),
                    label,
                ),
            ):
                items = member_groups[member_label]
                member_chain = chain_map.get(member_label)
                if member_chain is None:
                    continue
                member_summaries.append(
                    {
                        "pmhc_member_label_asym_id": member_chain.label_asym_id,
                        "pmhc_member_auth_asym_id": member_chain.auth_asym_id,
                        "pmhc_member_entity_id": member_chain.entity_id,
                        "pmhc_member_chain_type": member_chain.chain_type,
                        "pmhc_member_role": _mhc_role(member_chain)
                        if member_chain.chain_type == MHC_CHAIN_TYPE
                        else "",
                        "supporting_tcr_chain_ids": sorted(
                            {entry["tcr_label_asym_id"] for entry in items}
                        ),
                        "num_supporting_dimers": len(items),
                        "total_buried_area": round(
                            sum(float(entry["dimer"].buried_area) for entry in items),
                            4,
                        ),
                        "total_atom_contacts": sum(
                            int(entry["dimer"].num_atom_contacts) for entry in items
                        ),
                    }
                )

            warnings = list(tcr_unit["warnings"]) + list(pmhc_unit["warnings"])
            if not peptides:
                warnings.append("mhc_without_bound_peptide")
            complexes.append(
                TcrPmhcComplexRecord(
                    pdb_id=tcr_chains[0].pdb_id,
                    assembly_mode=relevant_dimers[0]["dimer"].assembly_mode,
                    assembly_id=relevant_dimers[0]["dimer"].assembly_id,
                    complex_id=f"tcr_pmhc_{index:03d}",
                    tcr_chain_ids=[chain.label_asym_id for chain in tcr_chains],
                    tcr_auth_asym_ids=[chain.auth_asym_id for chain in tcr_chains],
                    tcr_entity_ids=[chain.entity_id for chain in tcr_chains],
                    tcr_type=tcr_unit["tcr_type"],
                    mhc_chain_ids=[chain.label_asym_id for chain in mhc_chains],
                    mhc_auth_asym_ids=[chain.auth_asym_id for chain in mhc_chains],
                    mhc_entity_ids=[chain.entity_id for chain in mhc_chains],
                    mhc_chain_roles=[_mhc_role(chain) for chain in mhc_chains],
                    mhc_class=pmhc_unit["mhc_class"],
                    peptide_chain_ids=[chain.label_asym_id for chain in peptides],
                    peptide_auth_asym_ids=[chain.auth_asym_id for chain in peptides],
                    peptide_entity_ids=[chain.entity_id for chain in peptides],
                    auxiliary_chain_ids=[chain.label_asym_id for chain in auxiliaries],
                    auxiliary_auth_asym_ids=[chain.auth_asym_id for chain in auxiliaries],
                    auxiliary_entity_ids=[chain.entity_id for chain in auxiliaries],
                    num_tcr_chains=len(tcr_chains),
                    num_peptide_chains=len(peptides),
                    num_tcr_pmhc_interfaces=len(relevant_dimers),
                    contact_score=round(
                        sum(float(entry["dimer"].buried_area) for entry in relevant_dimers),
                        4,
                    ),
                    evidence={
                        "stage": "tcr_pmhc_complex_v1",
                        "tcr_unit_rule": "paired_alpha_beta_or_gamma_delta_else_single_tcr",
                        "pmhc_unit_rule": "mhc_heavy_plus_auxiliary_partners_with_peptide_resolved_per_tcr_mhc_context",
                        "peptide_rule": "length_limited_chain_with_direct_contacts_to_identified_tcr_and_mhc",
                        "contact_score_metric": "sum_buried_area_over_direct_tcr_pmhc_dimers",
                        "source_interface_stage": "dimer_interface_v2_assembly_instances",
                        "source_chain_annotation_files": [
                            "protein_chains",
                            "chain_inventory",
                        ],
                        "contextual_peptide_chain_ids": contextual_peptide_chain_ids,
                        "supporting_multimer_ids": _supporting_multimer_ids(
                            tcr_chain_ids,
                            pmhc_member_ids,
                            tight_multimers,
                        ),
                        "pmhc_member_interface_summaries": member_summaries,
                    },
                    warnings=sorted(set(warnings)),
                )
            )
            index += 1

    return sorted(
        complexes,
        key=lambda complex_record: (
            complex_record.pdb_id,
            tuple(complex_record.tcr_chain_ids),
            tuple(complex_record.mhc_chain_ids),
            complex_record.complex_id,
        ),
    )
