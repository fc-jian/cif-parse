from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cif_parse.constants import (
    ANTIBODY_DESCRIPTION_MARKERS,
    HEAVY_J_MOTIFS,
    HEAVY_PREFIX_MOTIFS,
    LIGHT_J_MOTIFS,
    LIGHT_PREFIX_MOTIFS,
    SCFV_LINKER_MOTIFS,
)

from .immune import analyze_immune_sequence


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


def _protein_letters(sequence: str | None) -> str:
    if not sequence:
        return ""
    return "".join(char for char in sequence.upper() if "A" <= char <= "Z")


def _find_first(sequence: str, motifs: tuple[str, ...]) -> str | None:
    for motif in motifs:
        if motif in sequence:
            return motif
    return None


def analyze_antibody_sequence(
    description: str | None,
    sequence: str | None,
) -> AntibodyAnnotation:
    immune_annotation = analyze_immune_sequence(description, sequence)
    if immune_annotation.chain_type in {"antibody heavy chain", "antibody light chain"}:
        annotation = AntibodyAnnotation(
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
        )
        return annotation

    description_lower = (description or "").lower()
    sequence_clean = _protein_letters(sequence)
    annotation = AntibodyAnnotation()

    if not description_lower and not sequence_clean:
        return annotation

    heavy_score = 0.0
    light_score = 0.0

    if any(marker in description_lower for marker in ANTIBODY_DESCRIPTION_MARKERS):
        annotation.description_hits.append("antibody_context")
        heavy_score += 0.5
        light_score += 0.5

    if "heavy chain" in description_lower:
        annotation.description_hits.append("heavy_chain")
        heavy_score += 2.0
    if "light chain" in description_lower:
        annotation.description_hits.append("light_chain")
        light_score += 2.0
    if "kappa" in description_lower:
        annotation.description_hits.append("kappa")
        light_score += 1.5
    if "lambda" in description_lower:
        annotation.description_hits.append("lambda")
        light_score += 1.5
    if "nanobody" in description_lower or "vhh" in description_lower:
        annotation.description_hits.append("vhh")
        heavy_score += 2.5
        annotation.unit_type = "VHH"
    if "single-chain fv" in description_lower or "scfv" in description_lower:
        annotation.description_hits.append("scfv")
        heavy_score += 2.5
        light_score += 2.5
        annotation.unit_type = "scFv"
        annotation.contains_fused_heavy_fv = True
        annotation.contains_fused_light_fv = True

    if sequence_clean:
        prefix4 = sequence_clean[:4]
        prefix3 = sequence_clean[:3]
        if any(sequence_clean.startswith(motif) for motif in HEAVY_PREFIX_MOTIFS):
            annotation.sequence_hits.append(f"heavy_prefix:{prefix4 or prefix3}")
            heavy_score += 2.0
        if any(sequence_clean.startswith(motif) for motif in LIGHT_PREFIX_MOTIFS):
            annotation.sequence_hits.append(f"light_prefix:{prefix4 or prefix3}")
            light_score += 2.0

        yyc_position = sequence_clean.find("YYC")
        if 70 <= yyc_position <= 120:
            annotation.sequence_hits.append("cdr3_anchor:YYC")
            heavy_score += 1.0
            light_score += 0.5

        heavy_j_motif = _find_first(sequence_clean[90:150], HEAVY_J_MOTIFS)
        if heavy_j_motif:
            annotation.sequence_hits.append(f"heavy_j:{heavy_j_motif}")
            annotation.variable_domain_end_motif = heavy_j_motif
            heavy_score += 1.5

        light_j_motif = _find_first(sequence_clean[80:140], LIGHT_J_MOTIFS)
        if light_j_motif:
            annotation.sequence_hits.append(f"light_j:{light_j_motif}")
            annotation.variable_domain_end_motif = light_j_motif
            light_score += 1.5

        if 105 <= len(sequence_clean) <= 145:
            annotation.sequence_hits.append("single_domain_length")
            heavy_score += 0.5
            light_score += 0.5
        if len(sequence_clean) >= 210:
            annotation.sequence_hits.append("long_chain_length")
            heavy_score += 0.5

        linker_motif = _find_first(sequence_clean, SCFV_LINKER_MOTIFS)
        if linker_motif:
            annotation.sequence_hits.append(f"scfv_linker:{linker_motif}")
            annotation.linker_motif = linker_motif
            annotation.unit_type = "scFv"
            annotation.contains_fused_heavy_fv = True
            annotation.contains_fused_light_fv = True
            heavy_score += 2.0
            light_score += 2.0

    annotation.heavy_score = round(heavy_score, 3)
    annotation.light_score = round(light_score, 3)

    score_gap = abs(heavy_score - light_score)
    dominant_score = max(heavy_score, light_score)
    if annotation.unit_type == "scFv":
        annotation.chain_type = "antibody heavy chain"
        annotation.subtype = "scFv"
        annotation.annotation_confidence = 0.9 if dominant_score >= 4.0 else 0.75
        return annotation

    if annotation.unit_type == "VHH":
        annotation.chain_type = "antibody heavy chain"
        annotation.subtype = "VHH"
        annotation.annotation_confidence = 0.9 if heavy_score >= 3.0 else 0.75
        return annotation

    if dominant_score < 2.5:
        return annotation

    if heavy_score > light_score and (score_gap >= 0.75 or heavy_score >= 4.0):
        annotation.chain_type = "antibody heavy chain"
        annotation.subtype = "heavy"
        annotation.annotation_confidence = 0.85 if heavy_score >= 4.0 else 0.7
        return annotation

    if light_score > heavy_score and (score_gap >= 0.75 or light_score >= 4.0):
        annotation.chain_type = "antibody light chain"
        annotation.subtype = "light"
        if "kappa" in annotation.description_hits:
            annotation.subtype = "light:kappa"
        elif "lambda" in annotation.description_hits:
            annotation.subtype = "light:lambda"
        annotation.annotation_confidence = 0.85 if light_score >= 4.0 else 0.7
        return annotation

    return annotation


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
