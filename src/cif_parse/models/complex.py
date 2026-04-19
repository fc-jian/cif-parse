from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(slots=True)
class AntibodyAntigenComplexRecord:
    pdb_id: str
    assembly_mode: str
    assembly_id: str | None
    complex_id: str
    antibody_unit_type: str
    antibody_heavy_chain: str | None
    antibody_heavy_auth_asym_id: str | None
    antibody_light_chain: str | None
    antibody_light_auth_asym_id: str | None
    antibody_chain_ids: list[str] = field(default_factory=list)
    antibody_auth_asym_ids: list[str | None] = field(default_factory=list)
    antibody_entity_ids: list[str] = field(default_factory=list)
    antigen_chain_ids: list[str] = field(default_factory=list)
    antigen_auth_asym_ids: list[str | None] = field(default_factory=list)
    antigen_entity_ids: list[str] = field(default_factory=list)
    antigen_chain_types: list[str] = field(default_factory=list)
    auxiliary_component_ids: list[str] = field(default_factory=list)
    auxiliary_component_auth_asym_ids: list[str] = field(default_factory=list)
    auxiliary_branched_ids: list[str] = field(default_factory=list)
    auxiliary_branched_auth_asym_ids: list[str] = field(default_factory=list)
    num_antigen_chains: int = 0
    num_antibody_antigen_interfaces: int = 0
    contact_score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "assembly_mode": self.assembly_mode,
            "assembly_id": self.assembly_id or "",
            "complex_id": self.complex_id,
            "antibody_unit_type": self.antibody_unit_type,
            "antibody_heavy_chain": self.antibody_heavy_chain or "",
            "antibody_heavy_auth_asym_id": self.antibody_heavy_auth_asym_id or "",
            "antibody_light_chain": self.antibody_light_chain or "",
            "antibody_light_auth_asym_id": self.antibody_light_auth_asym_id or "",
            "antibody_chain_ids": _json_cell(self.antibody_chain_ids),
            "antibody_auth_asym_ids": _json_cell(self.antibody_auth_asym_ids),
            "antibody_entity_ids": _json_cell(self.antibody_entity_ids),
            "antigen_chain_ids": _json_cell(self.antigen_chain_ids),
            "antigen_auth_asym_ids": _json_cell(self.antigen_auth_asym_ids),
            "antigen_entity_ids": _json_cell(self.antigen_entity_ids),
            "antigen_chain_types": _json_cell(self.antigen_chain_types),
            "auxiliary_component_ids": _json_cell(self.auxiliary_component_ids),
            "auxiliary_component_auth_asym_ids": _json_cell(self.auxiliary_component_auth_asym_ids),
            "auxiliary_branched_ids": _json_cell(self.auxiliary_branched_ids),
            "auxiliary_branched_auth_asym_ids": _json_cell(self.auxiliary_branched_auth_asym_ids),
            "num_antigen_chains": self.num_antigen_chains,
            "num_antibody_antigen_interfaces": self.num_antibody_antigen_interfaces,
            "contact_score": self.contact_score,
            "evidence": _json_cell(self.evidence),
            "warnings": _json_cell(self.warnings),
        }


@dataclass(slots=True)
class TcrPmhcComplexRecord:
    pdb_id: str
    assembly_mode: str
    assembly_id: str | None
    complex_id: str
    tcr_chain_ids: list[str] = field(default_factory=list)
    tcr_auth_asym_ids: list[str | None] = field(default_factory=list)
    tcr_entity_ids: list[str] = field(default_factory=list)
    tcr_type: str = ""
    mhc_chain_ids: list[str] = field(default_factory=list)
    mhc_auth_asym_ids: list[str | None] = field(default_factory=list)
    mhc_entity_ids: list[str] = field(default_factory=list)
    mhc_chain_roles: list[str] = field(default_factory=list)
    mhc_class: str = ""
    peptide_chain_ids: list[str] = field(default_factory=list)
    peptide_auth_asym_ids: list[str | None] = field(default_factory=list)
    peptide_entity_ids: list[str] = field(default_factory=list)
    auxiliary_chain_ids: list[str] = field(default_factory=list)
    auxiliary_auth_asym_ids: list[str | None] = field(default_factory=list)
    auxiliary_entity_ids: list[str] = field(default_factory=list)
    num_tcr_chains: int = 0
    num_peptide_chains: int = 0
    num_tcr_pmhc_interfaces: int = 0
    contact_score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "assembly_mode": self.assembly_mode,
            "assembly_id": self.assembly_id or "",
            "complex_id": self.complex_id,
            "tcr_chain_ids": _json_cell(self.tcr_chain_ids),
            "tcr_auth_asym_ids": _json_cell(self.tcr_auth_asym_ids),
            "tcr_entity_ids": _json_cell(self.tcr_entity_ids),
            "tcr_type": self.tcr_type,
            "mhc_chain_ids": _json_cell(self.mhc_chain_ids),
            "mhc_auth_asym_ids": _json_cell(self.mhc_auth_asym_ids),
            "mhc_entity_ids": _json_cell(self.mhc_entity_ids),
            "mhc_chain_roles": _json_cell(self.mhc_chain_roles),
            "mhc_class": self.mhc_class,
            "peptide_chain_ids": _json_cell(self.peptide_chain_ids),
            "peptide_auth_asym_ids": _json_cell(self.peptide_auth_asym_ids),
            "peptide_entity_ids": _json_cell(self.peptide_entity_ids),
            "auxiliary_chain_ids": _json_cell(self.auxiliary_chain_ids),
            "auxiliary_auth_asym_ids": _json_cell(self.auxiliary_auth_asym_ids),
            "auxiliary_entity_ids": _json_cell(self.auxiliary_entity_ids),
            "num_tcr_chains": self.num_tcr_chains,
            "num_peptide_chains": self.num_peptide_chains,
            "num_tcr_pmhc_interfaces": self.num_tcr_pmhc_interfaces,
            "contact_score": self.contact_score,
            "evidence": _json_cell(self.evidence),
            "warnings": _json_cell(self.warnings),
        }
