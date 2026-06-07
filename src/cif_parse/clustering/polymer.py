from __future__ import annotations

import json
import re
from typing import Any

from cif_parse.constants import POLYMER_CHAIN_TYPES, PROTEIN_CHAIN_TYPES


OTHER_POLYMER_CLASS = "other_polymer"


def normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def is_polymer_chain(chain_payload: dict[str, Any]) -> bool:
    return str(chain_payload.get("chain_type", "")) in POLYMER_CHAIN_TYPES


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _detail_component_id(detail: dict[str, Any]) -> str:
    for key in ("monomer_id", "comp_id", "component_id", "label_comp_id"):
        value = detail.get(key)
        if value not in (None, "", ".", "?"):
            return str(value).strip().upper()
    return ""


def _detail_chem_comp_type(detail: dict[str, Any]) -> str:
    value = detail.get("chem_comp_type", detail.get("type", ""))
    return re.sub(r"\s+", " ", str(value or "").strip())


def _safe_int(value: Any) -> int | None:
    if value in (None, "", ".", "?"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def has_component_sequence_details(monomer_or_payload: Any) -> bool:
    details = (
        monomer_or_payload.get("special_residue_details", [])
        if isinstance(monomer_or_payload, dict)
        else getattr(monomer_or_payload, "special_residue_details", [])
    )
    for detail in _json_list(details):
        if isinstance(detail, dict) and _detail_component_id(detail):
            return True
    return False


def classify_polymer_class(chain_payload: dict[str, Any]) -> str | None:
    chain_type = str(chain_payload.get("chain_type", ""))
    polymer_type = str(chain_payload.get("polymer_type", "") or "").lower()
    if chain_type in PROTEIN_CHAIN_TYPES:
        return "protein"
    if chain_type == "DNA chain":
        return "dna"
    if chain_type == "RNA chain":
        return "rna"
    if chain_type == "other nucleic acid chain":
        if "deoxyribo" in polymer_type:
            return "dna"
        if "ribo" in polymer_type:
            return "rna"
        return "other_nucleic_acid"
    if chain_type == "other polymer chain":
        if "deoxyribo" in polymer_type:
            return "dna"
        if "ribo" in polymer_type:
            return "rna"
        if "peptide" in polymer_type:
            return "protein"
        return OTHER_POLYMER_CLASS
    return None


def component_sequence_key(monomer_or_payload: Any) -> str:
    """Return a stable exact-clustering key for non-standard/mixed polymers."""

    def get_value(name: str, default: Any = "") -> Any:
        if isinstance(monomer_or_payload, dict):
            return monomer_or_payload.get(name, default)
        return getattr(monomer_or_payload, name, default)

    ordered_tokens: list[tuple[int, int, str]] = []
    for index, detail in enumerate(_json_list(get_value("special_residue_details", []))):
        if not isinstance(detail, dict):
            continue
        component_id = _detail_component_id(detail)
        if not component_id:
            continue
        chem_type = _detail_chem_comp_type(detail)
        token = f"{component_id}:{chem_type}" if chem_type else component_id
        label_seq_id = _safe_int(detail.get("label_seq_id"))
        sort_seq = label_seq_id if label_seq_id is not None else index
        ordered_tokens.append((sort_seq, index, token))
    if ordered_tokens:
        return "|".join(token for _, _, token in sorted(ordered_tokens))

    sequence = normalize_sequence(get_value("sequence", ""))
    polymer_type = re.sub(r"\s+", " ", str(get_value("polymer_type", "") or "").strip())
    description = re.sub(
        r"\s+",
        " ",
        str(get_value("entity_description", "") or "").strip(),
    )
    return f"fallback|polymer_type={polymer_type}|description={description}|sequence={sequence}"
