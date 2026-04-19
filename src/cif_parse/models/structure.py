from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(slots=True)
class ChainRecord:
    pdb_id: str
    entity_id: str
    entity_type: str
    entity_description: str | None
    label_asym_id: str
    auth_asym_id: str | None
    polymer_type: str | None
    chain_type: str
    subtype: str | None
    sequence: str | None
    length: int
    residue_count: int
    atom_count: int
    special_residue_details: list[dict[str, Any]] = field(default_factory=list)
    special_component_details: list[dict[str, Any]] = field(default_factory=list)
    parsed_coordinate_segments: list[dict[str, Any]] = field(default_factory=list)
    unresolved_sequence_segments: list[dict[str, Any]] = field(default_factory=list)
    paired_label_asym_id: str | None = None
    paired_auth_asym_id: str | None = None
    covered_nonpolymer_ids: list[str] = field(default_factory=list)
    covered_nonpolymer_auth_asym_ids: list[str] = field(default_factory=list)
    covered_branched_ids: list[str] = field(default_factory=list)
    covered_branched_auth_asym_ids: list[str] = field(default_factory=list)
    bound_chain_ids: list[str] = field(default_factory=list)
    bound_auth_asym_ids: list[str] = field(default_factory=list)
    annotation_sources: list[str] = field(default_factory=list)
    annotation_confidence: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "entity_description": self.entity_description or "",
            "label_asym_id": self.label_asym_id,
            "auth_asym_id": self.auth_asym_id or "",
            "polymer_type": self.polymer_type or "",
            "chain_type": self.chain_type,
            "subtype": self.subtype or "",
            "sequence": self.sequence or "",
            "length": self.length,
            "residue_count": self.residue_count,
            "atom_count": self.atom_count,
            "special_residue_details": _json_cell(self.special_residue_details),
            "special_component_details": _json_cell(self.special_component_details),
            "parsed_coordinate_segments": _json_cell(self.parsed_coordinate_segments),
            "unresolved_sequence_segments": _json_cell(self.unresolved_sequence_segments),
            "paired_label_asym_id": self.paired_label_asym_id or "",
            "paired_auth_asym_id": self.paired_auth_asym_id or "",
            "covered_nonpolymer_ids": _json_cell(self.covered_nonpolymer_ids),
            "covered_nonpolymer_auth_asym_ids": _json_cell(
                self.covered_nonpolymer_auth_asym_ids
            ),
            "covered_branched_ids": _json_cell(self.covered_branched_ids),
            "covered_branched_auth_asym_ids": _json_cell(
                self.covered_branched_auth_asym_ids
            ),
            "bound_chain_ids": _json_cell(self.bound_chain_ids),
            "bound_auth_asym_ids": _json_cell(self.bound_auth_asym_ids),
            "annotation_sources": _json_cell(self.annotation_sources),
            "annotation_confidence": self.annotation_confidence,
            "features": _json_cell(self.features),
            "warnings": _json_cell(self.warnings),
        }


@dataclass(slots=True)
class StructureSummary:
    pdb_id: str
    source_path: str
    data_block: str
    chain_id_source: str
    model: int
    atom_count: int
    entity_count: int
    chain_ids: list[str] = field(default_factory=list)
    chain_id_pairs: list[dict[str, str | None]] = field(default_factory=list)
    chain_type_counts: dict[str, int] = field(default_factory=dict)
    assembly_ids: list[str] = field(default_factory=list)
    assembly_descriptions: dict[str, str] = field(default_factory=dict)
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "source_path": self.source_path,
            "data_block": self.data_block,
            "chain_id_source": self.chain_id_source,
            "model": self.model,
            "atom_count": self.atom_count,
            "entity_count": self.entity_count,
            "chain_ids": ";".join(self.chain_ids),
            "chain_id_pairs": _json_cell(self.chain_id_pairs),
            "chain_type_counts": _json_cell(self.chain_type_counts),
            "assembly_ids": ";".join(self.assembly_ids),
            "assembly_descriptions": _json_cell(self.assembly_descriptions),
            "title": self.title or "",
        }
