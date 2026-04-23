from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(slots=True)
class DimerInterfaceRecord:
    pdb_id: str
    assembly_mode: str
    assembly_id: str | None
    num_supporting_instance_pairs: int
    instance_id_1: str | None
    sym_id_1: int | None
    label_asym_id_1: str
    auth_asym_id_1: str | None
    entity_id_1: str
    chain_type_1: str
    instance_id_2: str | None
    sym_id_2: int | None
    label_asym_id_2: str
    auth_asym_id_2: str | None
    entity_id_2: str
    chain_type_2: str
    interface_residue_count_1: int
    interface_residue_count_2: int
    interface_residue_ratio_1: float
    interface_residue_ratio_2: float
    num_residue_contacts: int
    num_atom_contacts: int
    min_distance: float
    centroid_distance: float
    delta_sasa_1: float
    delta_sasa_2: float
    buried_area: float
    is_same_entity: bool
    interface_label: str
    contains_antibody_unit: bool
    contains_tcr_pmhc_unit: bool
    mean_interface_residue_count: float = 0.0
    buried_area_per_interface_residue: float = 0.0
    atom_contacts_per_interface_residue: float = 0.0
    residue_contacts_per_interface_residue: float = 0.0
    contacting_atom_pairs: list[list[Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "assembly_mode": self.assembly_mode,
            "assembly_id": self.assembly_id or "",
            "num_supporting_instance_pairs": self.num_supporting_instance_pairs,
            "instance_id_1": self.instance_id_1 or "",
            "sym_id_1": "" if self.sym_id_1 is None else self.sym_id_1,
            "label_asym_id_1": self.label_asym_id_1,
            "auth_asym_id_1": self.auth_asym_id_1 or "",
            "entity_id_1": self.entity_id_1,
            "chain_type_1": self.chain_type_1,
            "instance_id_2": self.instance_id_2 or "",
            "sym_id_2": "" if self.sym_id_2 is None else self.sym_id_2,
            "label_asym_id_2": self.label_asym_id_2,
            "auth_asym_id_2": self.auth_asym_id_2 or "",
            "entity_id_2": self.entity_id_2,
            "chain_type_2": self.chain_type_2,
            "interface_residue_count_1": self.interface_residue_count_1,
            "interface_residue_count_2": self.interface_residue_count_2,
            "interface_residue_ratio_1": self.interface_residue_ratio_1,
            "interface_residue_ratio_2": self.interface_residue_ratio_2,
            "num_residue_contacts": self.num_residue_contacts,
            "num_atom_contacts": self.num_atom_contacts,
            "min_distance": self.min_distance,
            "centroid_distance": self.centroid_distance,
            "delta_sasa_1": self.delta_sasa_1,
            "delta_sasa_2": self.delta_sasa_2,
            "buried_area": self.buried_area,
            "mean_interface_residue_count": self.mean_interface_residue_count,
            "buried_area_per_interface_residue": self.buried_area_per_interface_residue,
            "atom_contacts_per_interface_residue": self.atom_contacts_per_interface_residue,
            "residue_contacts_per_interface_residue": self.residue_contacts_per_interface_residue,
            "contacting_atom_pairs": _json_cell(self.contacting_atom_pairs),
            "is_same_entity": self.is_same_entity,
            "interface_label": self.interface_label,
            "contains_antibody_unit": self.contains_antibody_unit,
            "contains_tcr_pmhc_unit": self.contains_tcr_pmhc_unit,
            "evidence": _json_cell(self.evidence),
            "warnings": _json_cell(self.warnings),
        }
