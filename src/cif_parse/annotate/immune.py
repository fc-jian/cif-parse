from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from operator import itemgetter
import re
from typing import Any

import pyhmmer
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sadie.numbering import Numbering
from sadie.renumbering.aligners.hmmer import HMMER
from sadie.renumbering.numbering_translator import NumberingTranslator

from cif_parse.constants import (
    ANTIBODY_DESCRIPTION_MARKERS,
    CAMELID_SPECIES,
    CHAIN_CODE_DESCRIPTION_HINTS,
    IMGT_CDR_RANGES,
    SADIE_CHAIN_CODES,
    SADIE_REGION_DEFINITION,
    SADIE_SCHEME,
    SCFV_LINKER_MOTIFS,
    TCR_DESCRIPTION_MARKERS,
)


@dataclass(slots=True)
class VariableDomainAnnotation:
    domain_no: int
    chain_code: str
    chain_type: str
    subtype: str | None
    seq_start: int
    seq_end: int
    length: int
    bitscore: float
    evalue: float
    species: str | None = None
    cdr_regions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.subtype is None:
            payload["subtype"] = ""
        if self.species is None:
            payload["species"] = ""
        return payload


@dataclass(slots=True)
class ImmuneSequenceAnnotation:
    chain_type: str | None = None
    subtype: str | None = None
    annotation_confidence: float = 0.0
    description_hits: list[str] = field(default_factory=list)
    sequence_hits: list[str] = field(default_factory=list)
    unit_type: str | None = None
    contains_fused_heavy_fv: bool = False
    contains_fused_light_fv: bool = False
    linker_motif: str | None = None
    tool: str = "sadie-antibody"
    numbering_scheme: str = SADIE_SCHEME
    region_definition: str = SADIE_REGION_DEFINITION
    selected_chain_codes: list[str] = field(default_factory=list)
    top_bitscore: float = 0.0
    variable_domains: list[VariableDomainAnnotation] = field(default_factory=list)
    vhh_evidence: dict[str, Any] = field(default_factory=dict)
    heavy_only_evidence: dict[str, Any] = field(default_factory=dict)

    def to_feature_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variable_domains"] = [domain.to_dict() for domain in self.variable_domains]
        if self.chain_type is None:
            payload["chain_type"] = ""
        if self.subtype is None:
            payload["subtype"] = ""
        if self.unit_type is None:
            payload["unit_type"] = ""
        if self.linker_motif is None:
            payload["linker_motif"] = ""
        return payload


def _protein_letters(sequence: str | None) -> str:
    if not sequence:
        return ""
    return "".join(char for char in sequence.upper() if "A" <= char <= "Z")


def _decode_name(name: object) -> str:
    if isinstance(name, (bytes, bytearray)):
        return name.decode()
    return str(name)


def _find_first(sequence: str, motifs: tuple[str, ...]) -> str | None:
    for motif in motifs:
        if motif in sequence:
            return motif
    return None


def _description_suggests_vhh(description_lower: str) -> bool:
    if not description_lower:
        return False
    if "nanobody" in description_lower or "nanobdoy" in description_lower or "vhh" in description_lower:
        return True
    if re.search(r"\bnb[_-]?[a-z0-9]+\b", description_lower):
        return True
    if re.search(r"\bla[a-z]{1,3}\d+\b", description_lower):
        return True
    return False


def _domain_overlap(left: dict[str, Any], right: dict[str, Any]) -> int:
    return max(
        0,
        min(int(left["query_end"]), int(right["query_end"]))
        - max(int(left["query_start"]), int(right["query_start"])),
    )


def _preferred_chain_codes(description_lower: str) -> list[str]:
    preferred: list[str] = []
    for chain_code, markers in CHAIN_CODE_DESCRIPTION_HINTS.items():
        if any(marker in description_lower for marker in markers):
            preferred.append(chain_code)
    return preferred


def _sadie_chain_type(chain_code: str) -> tuple[str | None, str | None]:
    if chain_code == "H":
        return "antibody heavy chain", "heavy"
    if chain_code == "K":
        return "antibody light chain", "light:kappa"
    if chain_code == "L":
        return "antibody light chain", "light:lambda"
    if chain_code == "A":
        return "TCR chain", "alpha"
    if chain_code == "B":
        return "TCR chain", "beta"
    if chain_code == "G":
        return "TCR chain", "gamma"
    if chain_code == "D":
        return "TCR chain", "delta"
    return None, None


def _annotation_confidence(bitscore: float, domain_bitscore_threshold: float = 80.0) -> float:
    if bitscore >= 170.0:
        return 0.98
    if bitscore >= 140.0:
        return 0.95
    if bitscore >= 110.0:
        return 0.9
    if bitscore >= domain_bitscore_threshold:
        return 0.82
    return 0.7


@lru_cache(maxsize=1)
def _offline_sadie_runtime() -> tuple[HMMER, Numbering]:
    translator = NumberingTranslator()
    HMMER.g3.chains = set()
    hmmer = HMMER(
        species=sorted(translator.species),
        chains=sorted(translator.chains),
        use_numbering_hmms=True,
    )
    numbering = Numbering(scheme=SADIE_SCHEME, region=SADIE_REGION_DEFINITION)
    return hmmer, numbering


def _collect_candidate_hits(
    sequence: str,
    *,
    domain_bitscore_threshold: float = 80.0,
) -> list[dict[str, Any]]:
    hmmer, _ = _offline_sadie_runtime()
    seqrecord = SeqRecord(Seq(sequence), id="query")
    sequences = hmmer.transform_seqs([seqrecord])
    seq_name_to_sequence = {
        _decode_name(seq.name): seq.textize().sequence
        for seq in sequences
    }
    candidate_hits: list[dict[str, Any]] = []
    for top_hits in pyhmmer.hmmsearch(hmmer.hmms, sequences, cpus=1):
        for hit in top_hits:
            domain = hit.best_domain
            if domain.score < domain_bitscore_threshold:
                continue
            alignment = domain.alignment
            hmm_name = _decode_name(alignment.hmm_name)
            query_name = _decode_name(hit.name)
            candidate_hits.append(
                {
                    "order": 0,
                    "n": len(hit.domains),
                    "query": query_name,
                    "query_length": len(seq_name_to_sequence[query_name]),
                    "hmm_seq": alignment.hmm_sequence,
                    "hmm_start": alignment.hmm_from - 1,
                    "hmm_end": alignment.hmm_to,
                    "id": hmm_name,
                    "description": hit.description or "",
                    "evalue": float(f"{domain.c_evalue:.2e}"),
                    "bitscore": round(float(domain.score), 1),
                    "bias": round(float(hit.bias), 1),
                    "query_seq": alignment.target_sequence.upper(),
                    "query_start": alignment.target_from - 1,
                    "query_end": alignment.target_to,
                    "species": hmm_name.split("_")[0],
                    "chain_type": hmm_name.split("_")[1],
                }
            )
    return sorted(
        candidate_hits,
        key=lambda item: (-float(item["bitscore"]), int(item["query_start"]), str(item["id"])),
    )


def _select_variable_domains(
    candidate_hits: list[dict[str, Any]],
    description_lower: str,
    *,
    domain_limit: int = 4,
) -> list[dict[str, Any]]:
    preferred_codes = _preferred_chain_codes(description_lower)
    clusters: list[list[dict[str, Any]]] = []
    for hit in candidate_hits:
        placed = False
        for cluster in clusters:
            if any(
                min(
                    int(hit["query_end"]) - int(hit["query_start"]),
                    int(existing["query_end"]) - int(existing["query_start"]),
                )
                > 0
                and _domain_overlap(hit, existing)
                / min(
                    int(hit["query_end"]) - int(hit["query_start"]),
                    int(existing["query_end"]) - int(existing["query_start"]),
                )
                >= 0.4
                for existing in cluster
            ):
                cluster.append(hit)
                placed = True
                break
        if not placed:
            clusters.append([hit])

    selected: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster = sorted(cluster, key=lambda item: (-float(item["bitscore"]), str(item["chain_type"])))
        choice = cluster[0]
        for candidate in cluster:
            if (
                str(candidate["chain_type"]) in preferred_codes
                and float(cluster[0]["bitscore"]) - float(candidate["bitscore"]) <= 25.0
            ):
                choice = candidate
                break
        selected.append(choice)
        if len(selected) >= domain_limit:
            break
    return sorted(selected, key=lambda item: (int(item["query_start"]), -float(item["bitscore"])))


def _number_selected_domains(
    sequence: str,
    selected_hits: list[dict[str, Any]],
) -> list[VariableDomainAnnotation]:
    if not selected_hits:
        return []
    hmmer, numbering = _offline_sadie_runtime()
    numbering_keys = [
        "id",
        "description",
        "evalue",
        "bitscore",
        "bias",
        "query_start",
        "query_end",
        "species",
        "chain_type",
    ]
    alignment_entry = (
        [numbering_keys, *[itemgetter(*numbering_keys)(hit) for hit in selected_hits]],
        [hmmer.get_vector_state(**hit) for hit in selected_hits],
        selected_hits,
    )
    numbered, details, _ = numbering.number_sequences_from_alignment(
        [("query", sequence)],
        [alignment_entry],
        scheme=SADIE_SCHEME,
        allow=list(SADIE_CHAIN_CODES),
        assign_germline=False,
        allowed_species=sorted(NumberingTranslator().species),
    )
    parsed = numbering.parsed_output([("query", sequence)], numbered, details)
    remaining_hits = selected_hits.copy()
    domains: list[VariableDomainAnnotation] = []
    for domain_no, item in enumerate(parsed):
        chain_code = str(item["chain_type"])
        chain_type, subtype = _sadie_chain_type(chain_code)
        if chain_type is None:
            continue
        matched_hit_index = next(
            (
                index
                for index, hit in enumerate(remaining_hits)
                if str(hit["chain_type"]) == chain_code
            ),
            None,
        )
        matched_hit = remaining_hits.pop(matched_hit_index) if matched_hit_index is not None else None
        domains.append(
            VariableDomainAnnotation(
                domain_no=domain_no,
                chain_code=chain_code,
                chain_type=chain_type,
                subtype=subtype,
                seq_start=int(item["seqstart_index"]) + 1,
                seq_end=int(item["seqend_index"]) + 1,
                length=int(item["seqend_index"]) - int(item["seqstart_index"]) + 1,
                bitscore=float(matched_hit["bitscore"]) if matched_hit is not None else 0.0,
                evalue=float(matched_hit["evalue"]) if matched_hit is not None else 0.0,
                species=str(item["hmm_species"]) if item.get("hmm_species") else None,
                cdr_regions=_cdr_regions_from_numbering(item, int(item["seqstart_index"])),
            )
        )
    return domains


# Known J-region C-terminal 4-mer motifs for extending Fv annotations.
# When SADIE's CDR3 reaches IMGT position ≥ 115 (full CDR3) but the
# annotation lacks the terminal FR4 region, we extend the boundary
# to include the conserved J-region end signature.
_HEAVY_END_MOTIFS: tuple[str, ...] = (
    "VTVS", "VTVT", "VTVW", "VTVA", "VTVM", "VTVP", "VTVL",
    "VSSP", "VSSS", "VSSA", "VSSM",
    "MVTW", "MVTV",
)
_KAPPA_END_MOTIFS: tuple[str, ...] = (
    "VEIK", "LEIK", "VDIK", "LDIK", "TEIK", "VEIN", "LEIN",
)
_LAMBDA_END_MOTIFS: tuple[str, ...] = (
    "VTVL", "LTVL", "VTVF", "LTVF", "VTVS", "LTVS",
)
_FV_EXTEND_LIMIT = 40
_CDR3_IMGT_FULL_END = 115  # CDR3 reaching this IMGT position implies FR4 should follow


def _cdr3_numbering_end(domain: VariableDomainAnnotation) -> int:
    """Return the highest IMGT numbering_end among CDR3 regions, or 0."""
    for cdr in domain.cdr_regions:
        if cdr.get("name") == "cdr3":
            try:
                return int(cdr["numbering_end"])
            except (ValueError, KeyError):
                pass
    return 0


def _extend_fv_by_end_motif(
    domains: list[VariableDomainAnnotation],
    sequence: str,
) -> tuple[list[VariableDomainAnnotation], list[dict[str, Any]]]:
    """Extend Fv domains where CDR3 is complete but FR4 is missing.

    Returns ``(extended_domains, warnings)``.  A warning is emitted for each
    domain that was extended.
    """
    extended_domains: list[VariableDomainAnnotation] = []
    warnings: list[dict[str, Any]] = []
    for domain in domains:
        cdr3_end = _cdr3_numbering_end(domain)
        missing_fr4 = (
            cdr3_end >= _CDR3_IMGT_FULL_END
            and domain.seq_end + 5 < len(sequence)
        )
        if not missing_fr4:
            extended_domains.append(domain)
            continue

        if domain.chain_code == "H":
            motifs = _HEAVY_END_MOTIFS
        elif domain.chain_code == "K":
            motifs = _KAPPA_END_MOTIFS
        elif domain.chain_code == "L":
            motifs = _LAMBDA_END_MOTIFS
        else:
            extended_domains.append(domain)
            continue

        current_end = domain.seq_end
        search_start = max(current_end - 5, 0)
        search_end = min(current_end + _FV_EXTEND_LIMIT, len(sequence) - 3)

        best_offset: int | None = None
        best_motif: str | None = None
        for i in range(search_start, search_end):
            candidate = sequence[i:i + 4]
            if candidate in motifs:
                best_offset = i + 4 - current_end
                best_motif = candidate
                break

        if best_offset is None or best_offset <= 0:
            extended_domains.append(domain)
            continue

        new_end = current_end + best_offset
        new_length = domain.length + best_offset

        new_cdrs: list[dict[str, Any]] = []
        for cdr in domain.cdr_regions:
            cdr_copy = dict(cdr)
            if cdr_copy.get("seq_end", 0) >= current_end - 5:
                cdr_copy["seq_end"] = min(cdr_copy["seq_end"] + best_offset, new_end)
                cdr_copy["length"] = cdr_copy["seq_end"] - cdr_copy["seq_start"] + 1
            new_cdrs.append(cdr_copy)

        extended_domains.append(
            VariableDomainAnnotation(
                domain_no=domain.domain_no,
                chain_code=domain.chain_code,
                chain_type=domain.chain_type,
                subtype=domain.subtype,
                seq_start=domain.seq_start,
                seq_end=new_end,
                length=new_length,
                bitscore=domain.bitscore,
                evalue=domain.evalue,
                species=domain.species,
                cdr_regions=new_cdrs,
            )
        )
        warnings.append({
            "warning_code": "fv_extended_by_end_motif",
            "chain_code": domain.chain_code,
            "chain_type": domain.chain_type,
            "original_seq_end": current_end,
            "extended_seq_end": new_end,
            "extension": best_offset,
            "end_motif": best_motif,
            "cdr3_imgt_end": cdr3_end,
        })
    return extended_domains, warnings


def _cdr_regions_from_numbering(item: dict[str, Any], domain_seq_start: int) -> list[dict[str, Any]]:
    numbering = item.get("Numbering")
    numbered_sequence = item.get("Numbered_Sequence")
    insertions = item.get("Insertion")
    if not isinstance(numbering, list) or not isinstance(numbered_sequence, list):
        return []
    if not isinstance(insertions, list):
        insertions = [""] * len(numbering)

    query_position = int(domain_seq_start)
    cdr_positions: dict[str, list[int]] = {name: [] for name in IMGT_CDR_RANGES}
    cdr_numbering: dict[str, list[str]] = {name: [] for name in IMGT_CDR_RANGES}
    for number, insertion, amino_acid in zip(numbering, insertions, numbered_sequence, strict=False):
        if str(amino_acid) == "-":
            continue
        query_position += 1
        try:
            numeric_number = int(number)
        except (TypeError, ValueError):
            continue
        numbering_label = f"{numeric_number}{str(insertion or '')}"
        for cdr_name, (start, end) in IMGT_CDR_RANGES.items():
            if start <= numeric_number <= end:
                cdr_positions[cdr_name].append(query_position)
                cdr_numbering[cdr_name].append(numbering_label)

    regions: list[dict[str, Any]] = []
    for cdr_name in ("cdr1", "cdr2", "cdr3"):
        positions = cdr_positions[cdr_name]
        if not positions:
            continue
        labels = cdr_numbering[cdr_name]
        regions.append(
            {
                "name": cdr_name,
                "seq_start": min(positions),
                "seq_end": max(positions),
                "length": len(positions),
                "numbering_start": labels[0],
                "numbering_end": labels[-1],
                "numbering_scheme": SADIE_SCHEME,
                "region_definition": SADIE_REGION_DEFINITION,
            }
        )
    return regions


def analyze_immune_sequence(
    description: str | None,
    sequence: str | None,
    *,
    domain_bitscore_threshold: float = 80.0,
    domain_limit: int = 4,
) -> ImmuneSequenceAnnotation:
    annotation = ImmuneSequenceAnnotation()
    sequence_clean = _protein_letters(sequence)
    description_lower = (description or "").lower()
    if not sequence_clean:
        return annotation

    if any(marker in description_lower for marker in ANTIBODY_DESCRIPTION_MARKERS):
        annotation.description_hits.append("antibody_context")
    if any(marker in description_lower for marker in TCR_DESCRIPTION_MARKERS):
        annotation.description_hits.append("tcr_context")
    if "nanobody" in description_lower or "vhh" in description_lower:
        annotation.description_hits.append("vhh")
    if "scfv" in description_lower or "single-chain fv" in description_lower:
        annotation.description_hits.append("scfv")

    try:
        candidate_hits = _collect_candidate_hits(
            sequence_clean,
            domain_bitscore_threshold=domain_bitscore_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        annotation.sequence_hits.append(f"sadie_error:{exc.__class__.__name__}")
        return annotation

    selected_hits = _select_variable_domains(
        candidate_hits,
        description_lower,
        domain_limit=domain_limit,
    )
    annotation.sequence_hits.extend(
        [
            f"sadie_domain:{hit['chain_type']}:{int(hit['query_start']) + 1}-{int(hit['query_end'])}"
            for hit in selected_hits
        ]
    )
    numbered_domains = _number_selected_domains(sequence_clean, selected_hits)
    annotation.variable_domains, fv_warnings = _extend_fv_by_end_motif(
        numbered_domains, sequence_clean,
    )
    if fv_warnings:
        annotation.sequence_hits.extend(
            f"fv_extended:{w['chain_code']}:{w['original_seq_end']}->{w['extended_seq_end']}:{w['end_motif']}"
            for w in fv_warnings
        )
    if not annotation.variable_domains:
        return annotation

    chain_codes = [domain.chain_code for domain in annotation.variable_domains]
    annotation.selected_chain_codes = chain_codes
    annotation.top_bitscore = max(domain.bitscore for domain in annotation.variable_domains)

    linker_motif = _find_first(sequence_clean, SCFV_LINKER_MOTIFS)
    if linker_motif:
        annotation.linker_motif = linker_motif

    has_heavy = any(code == "H" for code in chain_codes)
    has_light = any(code in {"K", "L"} for code in chain_codes)
    camelid_species = {
        domain.species.lower()
        for domain in annotation.variable_domains
        if isinstance(domain.species, str) and domain.species
    }
    has_single_heavy_domain = (
        len(annotation.variable_domains) == 1
        and annotation.variable_domains[0].chain_code == "H"
    )
    description_suggests_vhh = _description_suggests_vhh(description_lower)
    has_conventional_antibody_description = (
        "heavy chain" in description_lower
        or "light chain" in description_lower
        or "fab" in description_lower
    )
    length_in_vhh_range = 95 <= len(sequence_clean) <= 160
    camelid_single_domain = (
        has_single_heavy_domain
        and length_in_vhh_range
        and bool(camelid_species & CAMELID_SPECIES)
    )
    annotation.vhh_evidence = {
        "description_suggests_vhh": description_suggests_vhh,
        "camelid_species_hit": bool(camelid_species & CAMELID_SPECIES),
        "camelid_species_labels": sorted(camelid_species),
        "single_heavy_domain": has_single_heavy_domain,
        "length_in_vhh_range": length_in_vhh_range,
        "inferred_by_sequence_context": False,
        "paired_light_found": False,
    }
    if has_heavy and has_light:
        annotation.chain_type = "antibody heavy chain"
        annotation.subtype = "scFv"
        annotation.unit_type = "scFv"
        annotation.contains_fused_heavy_fv = True
        annotation.contains_fused_light_fv = True
    elif has_heavy:
        looks_like_vhh = description_suggests_vhh or (
            bool(description_lower)
            and camelid_single_domain
            and not has_conventional_antibody_description
        )
        annotation.vhh_evidence["inferred_by_sequence_context"] = (
            not description_suggests_vhh
            and looks_like_vhh
        )
        annotation.chain_type = "antibody heavy chain"
        annotation.subtype = "VHH" if looks_like_vhh else "heavy"
        if looks_like_vhh:
            annotation.unit_type = "VHH"
    elif any(code in {"K", "L"} for code in chain_codes):
        light_domain = next(domain for domain in annotation.variable_domains if domain.chain_code in {"K", "L"})
        annotation.chain_type = "antibody light chain"
        annotation.subtype = light_domain.subtype
    else:
        first_domain = annotation.variable_domains[0]
        if first_domain.chain_type == "TCR chain":
            annotation.chain_type = "TCR chain"
            annotation.subtype = first_domain.subtype

    if annotation.chain_type:
        annotation.annotation_confidence = _annotation_confidence(
            annotation.top_bitscore,
            domain_bitscore_threshold=domain_bitscore_threshold,
        )
    annotation.heavy_only_evidence = {
        "paired_light_found": False,
        "unit_type": annotation.unit_type or "",
        "is_true_heavy_only": (
            annotation.chain_type == "antibody heavy chain"
            and annotation.subtype != "VHH"
            and annotation.unit_type not in {"VHH", "scFv"}
        ),
    }
    return annotation
