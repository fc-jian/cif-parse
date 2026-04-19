from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(slots=True)
class TightMultimerRecord:
    pdb_id: str
    assembly_mode: str
    assembly_id: str | None
    multimer_id: str
    num_component_copies: int = 1
    member_chain_ids: list[str] = field(default_factory=list)
    member_auth_asym_ids: list[str | None] = field(default_factory=list)
    member_entity_ids: list[str] = field(default_factory=list)
    member_chain_types: list[str] = field(default_factory=list)
    member_copy_numbers: list[int] = field(default_factory=list)
    member_instances: list[dict[str, Any]] = field(default_factory=list)
    num_members: int = 0
    num_member_instances: int = 0
    num_internal_edges: int = 0
    multimer_type: str = ""
    support_score: float = 0.0
    contains_antibody_unit: bool = False
    contains_tcr_pmhc_unit: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "assembly_mode": self.assembly_mode,
            "assembly_id": self.assembly_id or "",
            "multimer_id": self.multimer_id,
            "num_component_copies": self.num_component_copies,
            "member_chain_ids": _json_cell(self.member_chain_ids),
            "member_auth_asym_ids": _json_cell(self.member_auth_asym_ids),
            "member_entity_ids": _json_cell(self.member_entity_ids),
            "member_chain_types": _json_cell(self.member_chain_types),
            "member_copy_numbers": _json_cell(self.member_copy_numbers),
            "member_instances": _json_cell(self.member_instances),
            "num_members": self.num_members,
            "num_member_instances": self.num_member_instances,
            "num_internal_edges": self.num_internal_edges,
            "multimer_type": self.multimer_type,
            "support_score": self.support_score,
            "contains_antibody_unit": self.contains_antibody_unit,
            "contains_tcr_pmhc_unit": self.contains_tcr_pmhc_unit,
            "evidence": _json_cell(self.evidence),
            "warnings": _json_cell(self.warnings),
        }
