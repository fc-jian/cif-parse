from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .immune import ImmuneSequenceAnnotation, analyze_immune_sequence


@dataclass(slots=True)
class AntibodyAnnotation:
    chain_type: str | None = None
    subtype: str | None = None
    annotation_confidence: float = 0.0
    description_hits: list[str] = field(default_factory=list)
    sequence_hits: list[str] = field(default_factory=list)
    heavy_score: float = 0.0
    light_score: float = 0.0
    unit_type: str | None = None
    contains_fused_heavy_fv: bool = False
    contains_fused_light_fv: bool = False
    linker_motif: str | None = None
    variable_domain_end_motif: str | None = None
    variable_domains: list[dict[str, Any]] = field(default_factory=list)
    vhh_evidence: dict[str, Any] = field(default_factory=dict)
    heavy_only_evidence: dict[str, Any] = field(default_factory=dict)
    tool: str = ""
    numbering_scheme: str = ""
    region_definition: str = ""
    warnings: list[str] = field(default_factory=list)
    warning_details: dict[str, Any] = field(default_factory=dict)

    def to_feature_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.chain_type is None:
            payload["chain_type"] = ""
        if self.subtype is None:
            payload["subtype"] = ""
        if self.unit_type is None:
            payload["unit_type"] = ""
        if self.linker_motif is None:
            payload["linker_motif"] = ""
        if self.variable_domain_end_motif is None:
            payload["variable_domain_end_motif"] = ""
        if not self.tool:
            payload["tool"] = ""
        if not self.numbering_scheme:
            payload["numbering_scheme"] = ""
        if not self.region_definition:
            payload["region_definition"] = ""
        return payload


def analyze_antibody_sequence(
    description: str | None,
    sequence: str | None,
    *,
    immune_annotation: ImmuneSequenceAnnotation | None = None,
    domain_bitscore_threshold: float = 80.0,
    domain_limit: int = 4,
) -> AntibodyAnnotation:
    if immune_annotation is None:
        immune_annotation = analyze_immune_sequence(
            description,
            sequence,
            domain_bitscore_threshold=domain_bitscore_threshold,
            domain_limit=domain_limit,
        )
    if immune_annotation.chain_type not in {"antibody heavy chain", "antibody light chain"}:
        return AntibodyAnnotation()

    return AntibodyAnnotation(
        chain_type=immune_annotation.chain_type,
        subtype=immune_annotation.subtype,
        annotation_confidence=immune_annotation.annotation_confidence,
        description_hits=list(immune_annotation.description_hits),
        sequence_hits=list(immune_annotation.sequence_hits),
        heavy_score=round(
            immune_annotation.top_bitscore
            if immune_annotation.chain_type == "antibody heavy chain"
            or immune_annotation.contains_fused_heavy_fv
            else 0.0,
            3,
        ),
        light_score=round(
            immune_annotation.top_bitscore
            if immune_annotation.chain_type == "antibody light chain"
            or immune_annotation.contains_fused_light_fv
            else 0.0,
            3,
        ),
        unit_type=immune_annotation.unit_type,
        contains_fused_heavy_fv=immune_annotation.contains_fused_heavy_fv,
        contains_fused_light_fv=immune_annotation.contains_fused_light_fv,
        linker_motif=immune_annotation.linker_motif,
        variable_domains=[domain.to_dict() for domain in immune_annotation.variable_domains],
        vhh_evidence=dict(immune_annotation.vhh_evidence),
        heavy_only_evidence=dict(immune_annotation.heavy_only_evidence),
        tool=immune_annotation.tool,
        numbering_scheme=immune_annotation.numbering_scheme,
        region_definition=immune_annotation.region_definition,
        warnings=list(immune_annotation.warnings),
        warning_details=dict(immune_annotation.warning_details),
    )


def apply_antibody_pairing(chain_records: list[Any], dimer_interfaces: list[Any]) -> None:
    chain_map = {record.label_asym_id: record for record in chain_records}
    pair_candidates: list[tuple[float, Any, Any, Any]] = []
    for dimer in dimer_interfaces:
        chain_1 = chain_map.get(dimer.label_asym_id_1)
        chain_2 = chain_map.get(dimer.label_asym_id_2)
        if chain_1 is None or chain_2 is None:
            continue

        if chain_1.chain_type == "antibody heavy chain" and chain_2.chain_type == "antibody light chain":
            pair_candidates.append((float(dimer.buried_area), chain_1, chain_2, dimer))
        elif chain_2.chain_type == "antibody heavy chain" and chain_1.chain_type == "antibody light chain":
            pair_candidates.append((float(dimer.buried_area), chain_2, chain_1, dimer))

    used_heavy: set[str] = set()
    used_light: set[str] = set()
    for _, heavy_chain, light_chain, dimer in sorted(
        pair_candidates,
        key=lambda item: (
            item[0],
            item[3].num_atom_contacts,
            item[3].num_residue_contacts,
        ),
        reverse=True,
    ):
        if heavy_chain.label_asym_id in used_heavy or light_chain.label_asym_id in used_light:
            continue

        used_heavy.add(heavy_chain.label_asym_id)
        used_light.add(light_chain.label_asym_id)
        heavy_chain.paired_label_asym_id = light_chain.label_asym_id
        heavy_chain.paired_auth_asym_id = light_chain.auth_asym_id
        light_chain.paired_label_asym_id = heavy_chain.label_asym_id
        light_chain.paired_auth_asym_id = heavy_chain.auth_asym_id

        heavy_chain.features["antibody_pairing"] = {
            "partner_label_asym_id": light_chain.label_asym_id,
            "partner_auth_asym_id": light_chain.auth_asym_id,
            "partner_chain_type": light_chain.chain_type,
            "buried_area": dimer.buried_area,
            "num_atom_contacts": dimer.num_atom_contacts,
            "num_residue_contacts": dimer.num_residue_contacts,
        }
        light_chain.features["antibody_pairing"] = {
            "partner_label_asym_id": heavy_chain.label_asym_id,
            "partner_auth_asym_id": heavy_chain.auth_asym_id,
            "partner_chain_type": heavy_chain.chain_type,
            "buried_area": dimer.buried_area,
            "num_atom_contacts": dimer.num_atom_contacts,
            "num_residue_contacts": dimer.num_residue_contacts,
        }

    for chain in chain_records:
        if chain.chain_type != "antibody heavy chain":
            continue
        features = chain.features if isinstance(getattr(chain, "features", None), dict) else {}
        analysis = features.get("antibody_analysis")
        if not isinstance(analysis, dict):
            continue
        paired_light_found = bool(chain.paired_label_asym_id)
        vhh_evidence = analysis.get("vhh_evidence")
        if isinstance(vhh_evidence, dict):
            analysis["vhh_evidence"] = {
                **vhh_evidence,
                "paired_light_found": paired_light_found,
            }
        unit_type = str(analysis.get("unit_type") or features.get("antibody_unit_type") or "")
        analysis["heavy_only_evidence"] = {
            "paired_light_found": paired_light_found,
            "unit_type": unit_type,
            "is_true_heavy_only": (
                chain.chain_type == "antibody heavy chain"
                and unit_type not in {"VHH", "scFv"}
                and not paired_light_found
            ),
        }
