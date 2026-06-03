from __future__ import annotations

import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from biotite.structure import AtomArray

from cif_parse.clustering.common import (
    canonical_monomer_id,
    load_monomer_cluster_assignments,
    load_monomer_inventory,
    resolve_monomer_cluster,
)
from cif_parse.clustering.dimers import DimerObservation, extract_dimer_structure
from cif_parse.clustering.high_order_refinement import refine_signature_groups_three_phase
from cif_parse.clustering.protein_structures import (
    USalignAlignmentResult,
    parse_usalign_output,
)
from cif_parse.clustering.parallel import normalize_worker_count
from cif_parse.clustering.signature_outputs import write_signature_cluster_membership_csv
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.settings import resolve_source_path


LOGGER = logging.getLogger(__name__)


def _signature_index(signature_cluster_id: str) -> str:
    return signature_cluster_id.rsplit("_", 1)[-1]


def _antibody_complex_cluster_id(signature_cluster_id: str, local_index: int = 1) -> str:
    return f"abag_{_signature_index(signature_cluster_id)}_{local_index}"


def _antibody_signature_cluster_id(index: int) -> str:
    return f"abagsig_{index}"


def _chain_role(
    *,
    chain_id: str,
    heavy_chain_id: str | None,
    light_chain_id: str | None,
) -> str:
    if heavy_chain_id and chain_id == heavy_chain_id:
        return "heavy"
    if light_chain_id and chain_id == light_chain_id:
        return "light"
    return "antibody_member"


@dataclass(slots=True)
class AntibodyComplexObservation:
    complex_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    assembly_mode: str
    complex_id: str
    antibody_unit_type: str
    antibody_heavy_chain: str | None
    antibody_heavy_auth_asym_id: str | None
    antibody_light_chain: str | None
    antibody_light_auth_asym_id: str | None
    antibody_chain_ids: list[str]
    antibody_auth_asym_ids: list[str]
    antigen_chain_ids: list[str]
    antigen_auth_asym_ids: list[str]
    antigen_chain_types: list[str]
    auxiliary_component_ids: list[str]
    auxiliary_component_auth_asym_ids: list[str]
    auxiliary_branched_ids: list[str]
    auxiliary_branched_auth_asym_ids: list[str]
    structural_auxiliary_chain_ids: list[str]
    num_antigen_chains: int
    num_antibody_antigen_interfaces: int
    contact_score: float
    antibody_member_descriptors: list[dict[str, str]]
    antigen_member_descriptors: list[dict[str, str]]
    auxiliary_member_descriptors: list[dict[str, str]]
    signature_key: str
    num_unclustered_monomer_members: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "complex_observation_id": self.complex_observation_id,
            "pdb_id": self.pdb_id,
            "source_path": self.source_path,
            "assembly_id": self.assembly_id or "",
            "assembly_mode": self.assembly_mode,
            "complex_id": self.complex_id,
            "antibody_unit_type": self.antibody_unit_type,
            "antibody_heavy_chain": self.antibody_heavy_chain or "",
            "antibody_light_chain": self.antibody_light_chain or "",
            "antibody_chain_ids": json.dumps(self.antibody_chain_ids, ensure_ascii=False),
            "antibody_auth_asym_ids": json.dumps(self.antibody_auth_asym_ids, ensure_ascii=False),
            "antigen_chain_ids": json.dumps(self.antigen_chain_ids, ensure_ascii=False),
            "antigen_auth_asym_ids": json.dumps(self.antigen_auth_asym_ids, ensure_ascii=False),
            "antigen_chain_types": json.dumps(self.antigen_chain_types, ensure_ascii=False),
            "auxiliary_component_ids": json.dumps(
                self.auxiliary_component_ids,
                ensure_ascii=False,
            ),
            "auxiliary_branched_ids": json.dumps(self.auxiliary_branched_ids, ensure_ascii=False),
            "structural_auxiliary_chain_ids": json.dumps(
                self.structural_auxiliary_chain_ids,
                ensure_ascii=False,
            ),
            "num_antigen_chains": self.num_antigen_chains,
            "num_antibody_antigen_interfaces": self.num_antibody_antigen_interfaces,
            "contact_score": self.contact_score,
            "antibody_member_descriptors": json.dumps(
                self.antibody_member_descriptors,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "antigen_member_descriptors": json.dumps(
                self.antigen_member_descriptors,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "auxiliary_member_descriptors": json.dumps(
                self.auxiliary_member_descriptors,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "signature_key": self.signature_key,
            "num_unclustered_monomer_members": self.num_unclustered_monomer_members,
        }

    def structural_sort_key(self) -> tuple[float, float, float, str]:
        return (
            -float(self.contact_score),
            -float(self.num_antibody_antigen_interfaces),
            -float(self.num_antigen_chains),
            self.complex_observation_id,
        )


@dataclass(slots=True)
class ExtractedAntibodyComplexStructure:
    complex_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    complex_id: str
    extracted_pdb_path: str
    residue_count: int
    atom_count: int
    filter_counts: dict[str, int]
    interface_residue_cutoff: float | None = None
    interface_residue_count_1: int | None = None
    interface_residue_count_2: int | None = None
    interface_atom_contact_count: int | None = None
    antibody_pdb_chain_ids: list[str] | None = None
    antigen_pdb_chain_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ordered_unique_chain_ids(chain_ids: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for chain_id in chain_ids:
        if chain_id and chain_id not in seen:
            seen.add(chain_id)
            ordered.append(chain_id)
    return ordered


def _antibody_antigen_chain_parts(
    observation: AntibodyComplexObservation,
) -> tuple[list[str], list[str]]:
    antibody_chain_ids = _ordered_unique_chain_ids(observation.antibody_chain_ids)
    antigen_chain_ids = _ordered_unique_chain_ids(observation.antigen_chain_ids)
    if not antibody_chain_ids:
        raise ValueError(
            f"Antibody complex {observation.complex_observation_id} has no antibody chains"
        )
    if not antigen_chain_ids:
        raise ValueError(
            f"Antibody complex {observation.complex_observation_id} has no antigen chains"
        )
    return antibody_chain_ids, antigen_chain_ids


def _dimer_observation_for_antibody_complex(
    observation: AntibodyComplexObservation,
    antibody_chain_ids: list[str],
    antigen_chain_ids: list[str],
) -> DimerObservation:
    return DimerObservation(
        dimer_observation_id=observation.complex_observation_id,
        pdb_id=observation.pdb_id,
        source_path=observation.source_path,
        assembly_id=observation.assembly_id,
        assembly_mode=observation.assembly_mode,
        sym_id_1=None,
        label_asym_id_1="+".join(antibody_chain_ids),
        auth_asym_id_1="+".join(observation.antibody_auth_asym_ids) or None,
        chain_type_1="antibody chain set",
        monomer_id_1=f"{observation.complex_observation_id}:antibody",
        monomer_sequence_cluster_id_1=None,
        monomer_structure_cluster_id_1=None,
        sym_id_2=None,
        label_asym_id_2="+".join(antigen_chain_ids),
        auth_asym_id_2="+".join(observation.antigen_auth_asym_ids) or None,
        chain_type_2="antigen chain set",
        monomer_id_2=f"{observation.complex_observation_id}:antigen",
        monomer_sequence_cluster_id_2=None,
        monomer_structure_cluster_id_2=None,
        interface_label="antibody_chain_set__antigen_chain_set",
        is_same_entity=False,
        contains_antibody_unit=True,
        contains_tcr_pmhc_unit=False,
        buried_area=float(observation.contact_score),
        num_residue_contacts=int(observation.num_antibody_antigen_interfaces),
        num_atom_contacts=0,
        signature_key=observation.signature_key,
        signature_members=[
            *observation.antibody_member_descriptors,
            *observation.antigen_member_descriptors,
        ],
        cluster_source_1="antibody_complex",
        cluster_source_2="antibody_complex",
    )


def _build_antibody_observation(
    *,
    pdb_id: str,
    source_path: str,
    assembly_id: str | None,
    assembly_mode: str,
    complex_id: str,
    complex_index: int,
    antibody_unit_type: str,
    antibody_heavy_chain: str | None,
    antibody_heavy_auth_asym_id: str | None,
    antibody_light_chain: str | None,
    antibody_light_auth_asym_id: str | None,
    antibody_chain_ids: list[str],
    antibody_auth_asym_ids: list[str],
    antigen_chain_ids: list[str],
    antigen_auth_asym_ids: list[str],
    antigen_chain_types: list[str],
    auxiliary_component_ids: list[str],
    auxiliary_component_auth_asym_ids: list[str],
    auxiliary_branched_ids: list[str],
    auxiliary_branched_auth_asym_ids: list[str],
    num_antigen_chains: int,
    num_antibody_antigen_interfaces: int,
    contact_score: float,
    monomer_cluster_assignments: dict[str, dict[str, str]],
    monomer_inventory: dict[str, dict[str, Any]],
) -> AntibodyComplexObservation:
    antibody_member_descriptors: list[dict[str, str]] = []
    antigen_member_descriptors: list[dict[str, str]] = []
    auxiliary_component_descriptors: list[dict[str, str]] = []
    structural_auxiliary_chain_ids: list[str] = []
    num_unclustered = 0
    heavy_cluster_id: str | None = None
    light_cluster_id: str | None = None

    for chain_id in antibody_chain_ids:
        monomer_id = canonical_monomer_id(pdb_id, chain_id, assembly_id or "")
        cluster_source, cluster_id, _ = resolve_monomer_cluster(monomer_id, monomer_cluster_assignments)
        if cluster_source == "unclustered":
            num_unclustered += 1
        role = _chain_role(chain_id=chain_id, heavy_chain_id=antibody_heavy_chain,
                           light_chain_id=antibody_light_chain)
        if role == "heavy":
            heavy_cluster_id = cluster_id
        elif role == "light":
            light_cluster_id = cluster_id
        antibody_member_descriptors.append({"role": role, "monomer_cluster_id": cluster_id})

    for chain_id, chain_type in zip(antigen_chain_ids, antigen_chain_types):
        monomer_id = canonical_monomer_id(pdb_id, chain_id, assembly_id or "")
        cluster_source, cluster_id, _ = resolve_monomer_cluster(monomer_id, monomer_cluster_assignments)
        if cluster_source == "unclustered":
            num_unclustered += 1
        antigen_member_descriptors.append({"chain_type": chain_type, "monomer_cluster_id": cluster_id})

    for chain_id in auxiliary_component_ids:
        monomer_id = canonical_monomer_id(pdb_id, chain_id, assembly_id or "")
        chain_payload = monomer_inventory.get(monomer_id, {})
        if monomer_id in monomer_inventory or monomer_id in monomer_cluster_assignments:
            cluster_source, cluster_id, _ = resolve_monomer_cluster(monomer_id, monomer_cluster_assignments)
            if cluster_source == "unclustered":
                num_unclustered += 1
            auxiliary_component_descriptors.append(
                {"chain_type": str(chain_payload.get("chain_type", "") or ""), "monomer_cluster_id": cluster_id})
            structural_auxiliary_chain_ids.append(chain_id)
        else:
            auxiliary_component_descriptors.append(
                {"chain_type": "auxiliary_nonpolymer", "monomer_cluster_id": "auxiliary_nonpolymer"})

    sig_key = json.dumps({
        "antibody_unit_type": antibody_unit_type,
        "heavy_cluster_id": heavy_cluster_id or "",
        "light_cluster_id": light_cluster_id or "",
        "antibody_members": sorted(antibody_member_descriptors,
                                   key=lambda x: (x["role"], x["monomer_cluster_id"])),
        "antigen_members": sorted(antigen_member_descriptors,
                                  key=lambda x: (x["chain_type"], x["monomer_cluster_id"])),
        "auxiliary_components": sorted(auxiliary_component_descriptors,
                                       key=lambda x: (x["chain_type"], x["monomer_cluster_id"])),
        "num_auxiliary_branched": len(auxiliary_branched_ids),
    }, ensure_ascii=False, sort_keys=True)

    return AntibodyComplexObservation(
        complex_observation_id=f"{pdb_id}:{assembly_id or 'na'}:{complex_index}",
        pdb_id=pdb_id, source_path=source_path, assembly_id=assembly_id,
        assembly_mode=assembly_mode, complex_id=complex_id,
        antibody_unit_type=antibody_unit_type,
        antibody_heavy_chain=antibody_heavy_chain,
        antibody_heavy_auth_asym_id=antibody_heavy_auth_asym_id,
        antibody_light_chain=antibody_light_chain,
        antibody_light_auth_asym_id=antibody_light_auth_asym_id,
        antibody_chain_ids=antibody_chain_ids, antibody_auth_asym_ids=antibody_auth_asym_ids,
        antigen_chain_ids=antigen_chain_ids, antigen_auth_asym_ids=antigen_auth_asym_ids,
        antigen_chain_types=antigen_chain_types,
        auxiliary_component_ids=auxiliary_component_ids,
        auxiliary_component_auth_asym_ids=auxiliary_component_auth_asym_ids,
        auxiliary_branched_ids=auxiliary_branched_ids,
        auxiliary_branched_auth_asym_ids=auxiliary_branched_auth_asym_ids,
        structural_auxiliary_chain_ids=structural_auxiliary_chain_ids,
        num_antigen_chains=num_antigen_chains,
        num_antibody_antigen_interfaces=num_antibody_antigen_interfaces,
        contact_score=contact_score,
        antibody_member_descriptors=antibody_member_descriptors,
        antigen_member_descriptors=antigen_member_descriptors,
        auxiliary_member_descriptors=auxiliary_component_descriptors,
        signature_key=sig_key,
        num_unclustered_monomer_members=num_unclustered,
    )


def collect_antibody_complex_observations(
    case_dirs: Iterable[str | Path],
    monomer_cluster_assignments: dict[str, dict[str, str]],
    monomer_inventory: dict[str, dict[str, Any]],
    cif_files_directory: str | None = None,
    prep_db_path: str | Path | None = None,
    prep_dir: str | Path | None = None,
) -> list[AntibodyComplexObservation]:
    observations: list[AntibodyComplexObservation] = []

    # Fast path: read pre-parsed Parquet
    from cif_parse.clustering.prep import open_prep_parquet, iter_parquet_rows
    pf = open_prep_parquet(prep_dir, "antibody_complexes", required=True) if prep_dir else None
    if pf is not None:
        for row in tqdm(iter_parquet_rows(prep_dir, "antibody_complexes", required=True),
                        desc="Collecting antibody complexes", unit="complex"):
            pdb_id = row.get("pdb_id", "")
            if not pdb_id:
                continue
            sp = resolve_source_path(row.get("source_path", ""), cif_files_directory)
            observations.append(_build_antibody_observation(
                pdb_id=pdb_id, source_path=sp,
                assembly_id=row.get("assembly_id"),
                assembly_mode=row.get("assembly_mode", ""),
                complex_id=row.get("complex_id", ""),
                complex_index=row.get("complex_index", 0),
                antibody_unit_type=row.get("antibody_unit_type", ""),
                antibody_heavy_chain=row.get("antibody_heavy_chain") or None,
                antibody_heavy_auth_asym_id=row.get("antibody_heavy_auth_asym_id") or None,
                antibody_light_chain=row.get("antibody_light_chain") or None,
                antibody_light_auth_asym_id=row.get("antibody_light_auth_asym_id") or None,
                antibody_chain_ids=json.loads(row.get("antibody_chain_ids", "[]")),
                antibody_auth_asym_ids=json.loads(row.get("antibody_auth_asym_ids", "[]")),
                antigen_chain_ids=json.loads(row.get("antigen_chain_ids", "[]")),
                antigen_auth_asym_ids=json.loads(row.get("antigen_auth_asym_ids", "[]")),
                antigen_chain_types=json.loads(row.get("antigen_chain_types", "[]")),
                auxiliary_component_ids=json.loads(row.get("auxiliary_component_ids", "[]")),
                auxiliary_component_auth_asym_ids=json.loads(row.get("auxiliary_component_auth_asym_ids", "[]")),
                auxiliary_branched_ids=json.loads(row.get("auxiliary_branched_ids", "[]")),
                auxiliary_branched_auth_asym_ids=json.loads(row.get("auxiliary_branched_auth_asym_ids", "[]")),
                num_antigen_chains=row.get("num_antigen_chains", 0),
                num_antibody_antigen_interfaces=row.get("num_antibody_antigen_interfaces", 0),
                contact_score=row.get("contact_score", 0.0),
                monomer_cluster_assignments=monomer_cluster_assignments,
                monomer_inventory=monomer_inventory,
            ))
        LOGGER.info("Collected %d antibody complex observations from prep Parquet", len(observations))
        return observations

    from cif_parse.clustering.prep import load_bundles_for_collect, load_case_bundles

    sorted_dirs = sorted(Path(path).resolve() for path in case_dirs)
    prep_bundles = load_bundles_for_collect(sorted_dirs, prep_db_path=prep_db_path)
    for case_dir in tqdm(sorted_dirs, desc="Collecting antibody complexes", unit="case"):
        payloads = load_case_bundles(case_dir, prep_bundles=prep_bundles)
        for payload in payloads:
            summary = payload.get("structure_summary", {})
            pdb_id = str(summary.get("pdb_id", "") or "")
            source_path = resolve_source_path(
                str(summary.get("source_path", "") or ""),
                cif_files_directory,
            )
            assembly_ids = [str(item) for item in summary.get("assembly_ids", []) if str(item)]
            default_assembly_id = assembly_ids[0] if len(assembly_ids) == 1 else None
            complexes = payload.get("antibody_antigen_complexes", [])
            if not isinstance(complexes, list):
                continue
            for index, complex_payload in enumerate(complexes, start=1):
                if not isinstance(complex_payload, dict):
                    continue
                antibody_chain_ids = [
                    str(item) for item in complex_payload.get("antibody_chain_ids", []) if str(item)
                ]
                antigen_chain_ids = [
                    str(item) for item in complex_payload.get("antigen_chain_ids", []) if str(item)
                ]
                if not antibody_chain_ids:
                    continue
                heavy_chain_id = (
                    str(complex_payload.get("antibody_heavy_chain"))
                    if complex_payload.get("antibody_heavy_chain") is not None
                    else None
                )
                light_chain_id = (
                    str(complex_payload.get("antibody_light_chain"))
                    if complex_payload.get("antibody_light_chain") is not None
                    else None
                )
                antibody_member_descriptors: list[dict[str, str]] = []
                antigen_member_descriptors: list[dict[str, str]] = []
                num_unclustered = 0
                heavy_cluster_id: str | None = None
                light_cluster_id: str | None = None
                slow_ab_asm = str(complex_payload.get("assembly_id") or default_assembly_id or "")
                for chain_id in antibody_chain_ids:
                    monomer_id = canonical_monomer_id(pdb_id, chain_id, slow_ab_asm)
                    cluster_source, cluster_id, _ = resolve_monomer_cluster(
                        monomer_id,
                        monomer_cluster_assignments,
                    )
                    if cluster_source == "unclustered":
                        num_unclustered += 1
                    role = _chain_role(
                        chain_id=chain_id,
                        heavy_chain_id=heavy_chain_id,
                        light_chain_id=light_chain_id,
                    )
                    if role == "heavy":
                        heavy_cluster_id = cluster_id
                    elif role == "light":
                        light_cluster_id = cluster_id
                    antibody_member_descriptors.append(
                        {
                            "role": role,
                            "monomer_cluster_id": cluster_id,
                        }
                    )
                antigen_chain_types = [
                    str(item) for item in complex_payload.get("antigen_chain_types", [])
                ]
                for chain_id, chain_type in zip(
                    antigen_chain_ids,
                    antigen_chain_types,
                    strict=False,
                ):
                    monomer_id = canonical_monomer_id(pdb_id, chain_id, slow_ab_asm)
                    cluster_source, cluster_id, _ = resolve_monomer_cluster(
                        monomer_id,
                        monomer_cluster_assignments,
                    )
                    if cluster_source == "unclustered":
                        num_unclustered += 1
                    antigen_member_descriptors.append(
                        {
                            "chain_type": chain_type,
                            "monomer_cluster_id": cluster_id,
                        }
                    )

                auxiliary_component_ids = [
                    str(item)
                    for item in complex_payload.get("auxiliary_component_ids", [])
                    if str(item)
                ]
                auxiliary_component_descriptors: list[dict[str, str]] = []
                structural_auxiliary_chain_ids: list[str] = []
                for chain_id in auxiliary_component_ids:
                    monomer_id = canonical_monomer_id(pdb_id, chain_id, slow_ab_asm)
                    chain_payload = monomer_inventory.get(monomer_id, {})
                    if monomer_id in monomer_inventory or monomer_id in monomer_cluster_assignments:
                        cluster_source, cluster_id, _ = resolve_monomer_cluster(
                            monomer_id,
                            monomer_cluster_assignments,
                        )
                        if cluster_source == "unclustered":
                            num_unclustered += 1
                        auxiliary_component_descriptors.append(
                            {
                                "chain_type": str(chain_payload.get("chain_type", "") or ""),
                                "monomer_cluster_id": cluster_id,
                            }
                        )
                        structural_auxiliary_chain_ids.append(chain_id)
                    else:
                        auxiliary_component_descriptors.append(
                            {
                                "chain_type": "auxiliary_nonpolymer",
                                "monomer_cluster_id": "auxiliary_nonpolymer",
                            }
                        )

                signature_key = json.dumps(
                    {
                        "antibody_unit_type": str(
                            complex_payload.get("antibody_unit_type", "") or ""
                        ),
                        "heavy_cluster_id": heavy_cluster_id or "",
                        "light_cluster_id": light_cluster_id or "",
                        "antibody_members": sorted(
                            antibody_member_descriptors,
                            key=lambda item: (item["role"], item["monomer_cluster_id"]),
                        ),
                        "antigen_members": sorted(
                            antigen_member_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                        "auxiliary_components": sorted(
                            auxiliary_component_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                        "num_auxiliary_branched": len(
                            complex_payload.get("auxiliary_branched_ids", []) or []
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                assembly_id = complex_payload.get("assembly_id")
                observations.append(
                    AntibodyComplexObservation(
                        complex_observation_id=(
                            f"{pdb_id}:{assembly_id or default_assembly_id or 'na'}:{index}"
                        ),
                        pdb_id=pdb_id,
                        source_path=source_path,
                        assembly_id=str(assembly_id) if assembly_id is not None else default_assembly_id,
                        assembly_mode=str(complex_payload.get("assembly_mode", "") or ""),
                        complex_id=str(complex_payload.get("complex_id", "") or ""),
                        antibody_unit_type=str(
                            complex_payload.get("antibody_unit_type", "") or ""
                        ),
                        antibody_heavy_chain=heavy_chain_id,
                        antibody_heavy_auth_asym_id=(
                            str(complex_payload.get("antibody_heavy_auth_asym_id"))
                            if complex_payload.get("antibody_heavy_auth_asym_id") is not None
                            else None
                        ),
                        antibody_light_chain=light_chain_id,
                        antibody_light_auth_asym_id=(
                            str(complex_payload.get("antibody_light_auth_asym_id"))
                            if complex_payload.get("antibody_light_auth_asym_id") is not None
                            else None
                        ),
                        antibody_chain_ids=antibody_chain_ids,
                        antibody_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("antibody_auth_asym_ids", [])
                            if str(item)
                        ],
                        antigen_chain_ids=antigen_chain_ids,
                        antigen_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("antigen_auth_asym_ids", [])
                            if str(item)
                        ],
                        antigen_chain_types=antigen_chain_types,
                        auxiliary_component_ids=auxiliary_component_ids,
                        auxiliary_component_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("auxiliary_component_auth_asym_ids", [])
                            if str(item)
                        ],
                        auxiliary_branched_ids=[
                            str(item)
                            for item in complex_payload.get("auxiliary_branched_ids", [])
                            if str(item)
                        ],
                        auxiliary_branched_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("auxiliary_branched_auth_asym_ids", [])
                            if str(item)
                        ],
                        structural_auxiliary_chain_ids=structural_auxiliary_chain_ids,
                        num_antigen_chains=int(
                            complex_payload.get("num_antigen_chains", len(antigen_chain_ids))
                            or len(antigen_chain_ids)
                        ),
                        num_antibody_antigen_interfaces=int(
                            complex_payload.get("num_antibody_antigen_interfaces", 0) or 0
                        ),
                        contact_score=float(complex_payload.get("contact_score", 0.0) or 0.0),
                        antibody_member_descriptors=sorted(
                            antibody_member_descriptors,
                            key=lambda item: (item["role"], item["monomer_cluster_id"]),
                        ),
                        antigen_member_descriptors=sorted(
                            antigen_member_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                        auxiliary_member_descriptors=sorted(
                            auxiliary_component_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                        signature_key=signature_key,
                        num_unclustered_monomer_members=num_unclustered,
                    )
                )
    LOGGER.info("Collected %d antibody complex observations from %d case dir(s)", len(observations), len(sorted_dirs))
    return observations


def extract_antibody_complex_structure(
    observation: AntibodyComplexObservation,
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    atom_array: AtomArray | None = None,
    interface_residue_cutoff: float | None = 20.0,
) -> ExtractedAntibodyComplexStructure:
    if not observation.source_path:
        raise ValueError(
            f"Missing source_path for antibody complex {observation.complex_observation_id}"
        )
    if atom_array is None:
        raise ValueError(
            f"Cached coordinates are required for antibody complex {observation.complex_observation_id}"
        )

    antibody_chain_ids, antigen_chain_ids = _antibody_antigen_chain_parts(observation)
    extracted = extract_dimer_structure(
        _dimer_observation_for_antibody_complex(
            observation,
            antibody_chain_ids,
            antigen_chain_ids,
        ),
        outdir=outdir,
        model=model,
        drop_hydrogens=drop_hydrogens,
        atom_array=atom_array,
        interface_residue_cutoff=interface_residue_cutoff,
        chain_parts=[
            [(chain_id, None) for chain_id in antibody_chain_ids],
            [(chain_id, None) for chain_id in antigen_chain_ids],
        ],
    )
    return ExtractedAntibodyComplexStructure(
        complex_observation_id=observation.complex_observation_id,
        pdb_id=observation.pdb_id,
        source_path=observation.source_path,
        assembly_id=observation.assembly_id,
        complex_id=observation.complex_id,
        extracted_pdb_path=extracted.extracted_pdb_path,
        residue_count=extracted.residue_count,
        atom_count=extracted.atom_count,
        filter_counts=extracted.filter_counts,
        interface_residue_cutoff=extracted.interface_residue_cutoff,
        interface_residue_count_1=extracted.interface_residue_count_1,
        interface_residue_count_2=extracted.interface_residue_count_2,
        interface_atom_contact_count=extracted.interface_atom_contact_count,
        antibody_pdb_chain_ids=extracted.part_1_pdb_chain_ids,
        antigen_pdb_chain_ids=extracted.part_2_pdb_chain_ids,
    )


def extract_antibody_complex_structures(
    observations: Iterable[AntibodyComplexObservation],
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    interface_residue_cutoff: float | None = 20.0,
    extraction_jobs: int = 1,
    prep_dir: str | Path | None = None,
    show_progress: bool = True,
    log_summary: bool = True,
) -> tuple[dict[str, ExtractedAntibodyComplexStructure], dict[str, Any]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    structures: dict[str, ExtractedAntibodyComplexStructure] = {}
    failures: list[dict[str, str]] = []

    sorted_observations = sorted(observations, key=lambda item: item.complex_observation_id)
    extraction_jobs = normalize_worker_count(extraction_jobs)

    _pkl_reader = None
    if prep_dir:
        from cif_parse.clustering.atom_cache import resolve_cases_root, PklAtomReader
        try:
            _pkl_reader = PklAtomReader(resolve_cases_root(prep_dir))
        except Exception:
            pass

    def _try_assemble_ab(obs: AntibodyComplexObservation) -> AtomArray | None:
        if _pkl_reader is None:
            return None
        antibody_chain_ids, antigen_chain_ids = _antibody_antigen_chain_parts(obs)
        chain_ids = _ordered_unique_chain_ids([*antibody_chain_ids, *antigen_chain_ids])
        if not chain_ids:
            return None
        return _pkl_reader.load_chains(
            obs.source_path,
            [(cid, None) for cid in chain_ids],
            assembly_id=obs.assembly_id,
        )

    def _process_one(observation: AntibodyComplexObservation) -> ExtractedAntibodyComplexStructure | None:
        atom_array = _try_assemble_ab(observation)
        if atom_array is None:
            raise ValueError(
                f"Prep coordinates missing for antibody complex {observation.complex_observation_id}"
            )
        return extract_antibody_complex_structure(
            observation,
            outdir=outdir,
            model=model,
            drop_hydrogens=drop_hydrogens,
            atom_array=atom_array,
            interface_residue_cutoff=interface_residue_cutoff,
        )

    if extraction_jobs <= 1 or len(sorted_observations) <= 1:
        observation_iter = (
            tqdm(sorted_observations, desc="Extracting antibody complex structures", unit="complex")
            if show_progress
            else sorted_observations
        )
        for observation in observation_iter:
            try:
                structures[observation.complex_observation_id] = _process_one(observation)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to extract antibody complex %s: %s",
                    observation.complex_observation_id,
                    exc,
                )
                failures.append(
                    {"complex_observation_id": observation.complex_observation_id, "error": str(exc)}
                )
    else:
        with ThreadPoolExecutor(max_workers=min(extraction_jobs, len(sorted_observations))) as executor:
            future_to_obs = {
                executor.submit(_process_one, observation): observation
                for observation in sorted_observations
            }
            future_iter = as_completed(future_to_obs)
            if show_progress:
                future_iter = tqdm(
                    future_iter,
                    total=len(future_to_obs),
                    desc="Extracting antibody complex structures",
                    unit="complex",
                )
            for future in future_iter:
                observation = future_to_obs[future]
                try:
                    extracted = future.result()
                    if extracted is not None:
                        structures[observation.complex_observation_id] = extracted
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to extract antibody complex %s: %s",
                        observation.complex_observation_id,
                        exc,
                    )
                    failures.append(
                        {"complex_observation_id": observation.complex_observation_id, "error": str(exc)}
                    )

    dump_jsonl(
        outdir / "antibody_complex_structure_extraction_failures.jsonl",
        failures,
    )
    dump_jsonl(
        outdir / "antibody_complex_structures.jsonl",
        [item.to_dict() for item in structures.values()],
    )
    manifest = {
        "num_antibody_complex_observations": len(sorted_observations),
        "num_extracted_antibody_complex_structures": len(structures),
        "num_failed_antibody_complex_structure_extractions": len(failures),
        "extraction_jobs": extraction_jobs,
        "antibody_complex_interface_residue_cutoff": (
            float(interface_residue_cutoff) if interface_residue_cutoff is not None else None
        ),
    }
    dump_json(outdir / "antibody_complex_structure_manifest.json", manifest, indent=2)
    if log_summary:
        LOGGER.info("Extracted %d antibody complex structures (%d failures)", len(structures), len(failures))
    return structures, manifest


def run_antibody_complex_usalign_alignment(
    query: ExtractedAntibodyComplexStructure,
    target: ExtractedAntibodyComplexStructure,
    *,
    usalign_executable: str = "USalign",
    tm_score_threshold: float = 0.50,
) -> USalignAlignmentResult:
    if shutil.which(usalign_executable) is None:
        raise FileNotFoundError(f"{usalign_executable} executable not found in PATH")
    command = [
        usalign_executable,
        query.extracted_pdb_path,
        target.extracted_pdb_path,
        "-mol",
        "auto",
        "-mm",
        "1",
        "-ter",
        "1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_usalign_output(
        completed.stdout,
        query_monomer_id=query.complex_observation_id,
        target_monomer_id=target.complex_observation_id,
        query_length=query.residue_count,
        target_length=target.residue_count,
        tm_score_threshold=tm_score_threshold,
        min_alignment_coverage_ratio=0.0,
    )


def refine_antibody_complex_signature_clusters(
    signature_groups: list[tuple[str, list[AntibodyComplexObservation]]],
    extracted_structures: dict[str, ExtractedAntibodyComplexStructure],
    *,
    tm_score_threshold: float = 0.50,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
    show_progress: bool = True,
    log_summary: bool = True,
) -> dict[str, Any]:
    runner = alignment_runner or run_antibody_complex_usalign_alignment
    alignment_jobs = normalize_worker_count(alignment_jobs)
    total_observations = sum(len(members) for _, members in signature_groups)
    multi_member = sum(1 for _, m in signature_groups if len(m) > 1)
    if log_summary:
        LOGGER.info(
            "Refining %d antibody complex signature clusters (%d observations, %d multi-member, %d alignment workers)",
            len(signature_groups),
            total_observations,
            multi_member,
            alignment_jobs,
        )
    signature_iter = list(
        tqdm(signature_groups, desc="Refining antibody complex clusters", unit="sig-group")
        if show_progress
        else signature_groups
    )
    refined = refine_signature_groups_three_phase(
        signature_iter,
        extracted_structures,
        member_id=lambda item: item.complex_observation_id,
        structural_sort_key=lambda item: item.structural_sort_key(),
        alignment_row=lambda signature_cluster_id, result: {
            "signature_cluster_id": signature_cluster_id,
            "query_complex_observation_id": result.query_monomer_id,
            "target_complex_observation_id": result.target_monomer_id,
            "aligned_length": result.aligned_length,
            "rmsd": result.rmsd,
            "tm_score_query": result.tm_score_query,
            "tm_score_target": result.tm_score_target,
            "tm_score_min": result.min_tm_score,
            "tm_score_max": result.max_tm_score,
            "tm_score_for_clustering": result.max_tm_score,
        },
        alignment_failure_warning=lambda signature_cluster_id, representative, candidate, exc: {
            "warning_code": "antibody_complex_usalign_failed",
            "signature_cluster_id": signature_cluster_id,
            "representative_complex_observation_id": representative.complex_observation_id,
            "candidate_complex_observation_id": candidate.complex_observation_id,
            "error": str(exc),
        },
        unavailable_warning=lambda signature_cluster_id, member: {
            "warning_code": "antibody_complex_structure_unavailable_singleton_cluster",
            "signature_cluster_id": signature_cluster_id,
            "complex_observation_id": member.complex_observation_id,
        },
        runner=runner,
        alignment_jobs=alignment_jobs,
        usalign_executable=usalign_executable,
        tm_score_threshold=tm_score_threshold,
        show_progress=show_progress,
        can_skip_alignment=lambda a, b: (
            a.source_path == b.source_path
            and a.assembly_id == b.assembly_id
            and sorted(a.antibody_chain_ids) == sorted(b.antibody_chain_ids)
            and sorted(a.antigen_chain_ids) == sorted(b.antigen_chain_ids)
        ),
    )
    alignment_cache = refined.alignment_cache
    alignment_rows = refined.alignment_rows
    warning_rows = refined.warning_rows
    cluster_members = refined.cluster_members
    num_alignment_runs = refined.num_alignment_runs
    num_alignment_failures = refined.num_alignment_failures
    num_signature_clusters_split = refined.num_signature_clusters_split

    grouped_signature_sizes = {signature_cluster_id: 0 for signature_cluster_id, _ in signature_groups}
    for signature_cluster_id, _, _, _ in cluster_members:
        grouped_signature_sizes[signature_cluster_id] = grouped_signature_sizes.get(signature_cluster_id, 0) + 1

    membership_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    local_cluster_counts: dict[str, int] = {}
    for signature_cluster_id, representative_id, members, representative in cluster_members:
        local_cluster_counts[signature_cluster_id] = local_cluster_counts.get(signature_cluster_id, 0) + 1
        cluster_id = _antibody_complex_cluster_id(
            signature_cluster_id,
            local_cluster_counts[signature_cluster_id],
        )
        representative_rows.append(
            {
                "antibody_complex_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "representative_complex_observation_id": representative_id,
                "num_members": len(members),
                "pdb_id": representative.pdb_id,
                "assembly_id": representative.assembly_id or "",
                "complex_id": representative.complex_id,
                "antibody_unit_type": representative.antibody_unit_type,
                "contact_score": representative.contact_score,
                "signature_key": representative.signature_key,
            }
        )
        signature_rows.append(
            {
                "antibody_complex_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "signature_key": representative.signature_key,
                "antibody_unit_type": representative.antibody_unit_type,
                "antibody_member_descriptors": json.dumps(
                    representative.antibody_member_descriptors,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "antigen_member_descriptors": json.dumps(
                    representative.antigen_member_descriptors,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "auxiliary_member_descriptors": json.dumps(
                    representative.auxiliary_member_descriptors,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "num_refined_clusters_in_signature_group": grouped_signature_sizes.get(
                    signature_cluster_id,
                    1,
                ),
            }
        )
        for member in members:
            tm_score_for_clustering: float | str = ""
            if member.complex_observation_id != representative_id:
                pair_key = tuple(sorted((representative_id, member.complex_observation_id)))
                if pair_key in alignment_cache:
                    tm_score_for_clustering = alignment_cache[pair_key].max_tm_score
            membership_rows.append(
                {
                    "antibody_complex_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "complex_observation_id": member.complex_observation_id,
                    "pdb_id": member.pdb_id,
                    "assembly_id": member.assembly_id or "",
                    "complex_id": member.complex_id,
                    "antibody_unit_type": member.antibody_unit_type,
                    "antibody_chain_ids": json.dumps(member.antibody_chain_ids, ensure_ascii=False),
                    "antigen_chain_ids": json.dumps(member.antigen_chain_ids, ensure_ascii=False),
                    "antigen_chain_types": json.dumps(
                        member.antigen_chain_types,
                        ensure_ascii=False,
                    ),
                    "structural_auxiliary_chain_ids": json.dumps(
                        member.structural_auxiliary_chain_ids,
                        ensure_ascii=False,
                    ),
                    "num_antigen_chains": member.num_antigen_chains,
                    "num_antibody_antigen_interfaces": member.num_antibody_antigen_interfaces,
                    "contact_score": member.contact_score,
                    "num_unclustered_monomer_members": member.num_unclustered_monomer_members,
                    "representative_complex_observation_id": representative_id,
                    "tm_score_to_representative": tm_score_for_clustering,
                }
            )

    manifest = {
        "num_signature_clusters": len(signature_groups),
        "num_antibody_complex_clusters": len(cluster_members),
        "num_alignment_runs": num_alignment_runs,
        "num_alignment_failures": num_alignment_failures,
        "num_signature_clusters_split": num_signature_clusters_split,
        "antibody_complex_tm_score_threshold": tm_score_threshold,
        "alignment_jobs": alignment_jobs,
    }
    if log_summary:
        LOGGER.info(
            "Antibody complex refinement: %d signature clusters -> %d refined clusters (%d alignments, %d failures, %d splits)",
            len(signature_groups),
            len(cluster_members),
            num_alignment_runs,
            num_alignment_failures,
            num_signature_clusters_split,
        )
    return {
        "manifest": manifest,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "signature_rows": signature_rows,
        "alignment_rows": alignment_rows,
        "warning_rows": warning_rows,
    }


def build_antibody_complex_signature_clusters(
    *,
    case_dirs: Iterable[str | Path],
    clustering_outdir: str | Path,
    outdir: str | Path,
    structure_refinement_mode: str = "greedy",
    antibody_complex_tm_score_threshold: float = 0.50,
    antibody_complex_interface_residue_cutoff: float | None = 20.0,
    model: int = 1,
    drop_hydrogens: bool = True,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
    cif_files_directory: str | None = None,
    prep_dir: str | Path | None = None,
    include_structure_assignments: bool = True,
) -> dict[str, Any]:
    monomer_assignments = load_monomer_cluster_assignments(
        clustering_outdir,
        include_structure=include_structure_assignments,
    )
    monomer_inventory = load_monomer_inventory(clustering_outdir)
    observations = collect_antibody_complex_observations(
        case_dirs,
        monomer_assignments,
        monomer_inventory,
        cif_files_directory=cif_files_directory,
        prep_dir=prep_dir,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dump_jsonl(
        outdir / "antibody_complex_inventory.jsonl",
        [item.to_dict() for item in observations],
    )
    dump_csv_rows(
        outdir / "antibody_complex_inventory.csv",
        [item.to_record() for item in observations],
    )

    grouped: dict[str, list[AntibodyComplexObservation]] = {}
    for observation in tqdm(observations, desc="Grouping antibody signatures", unit="complex"):
        grouped.setdefault(observation.signature_key, []).append(observation)
    signature_groups = [
        (_antibody_signature_cluster_id(index), members)
        for index, (_, members) in enumerate(
            sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
            start=1,
        )
    ]
    write_signature_cluster_membership_csv(
        outdir / "antibody_complex_signature_cluster_membership.csv",
        signature_groups,
        observation_id_field="complex_observation_id",
        observation_id=lambda item: item.complex_observation_id,
        extra_fields=lambda item: {
            "complex_id": item.complex_id,
            "antibody_unit_type": item.antibody_unit_type,
            "antibody_chain_ids": json.dumps(item.antibody_chain_ids, ensure_ascii=False),
            "antigen_chain_ids": json.dumps(item.antigen_chain_ids, ensure_ascii=False),
            "num_antigen_chains": item.num_antigen_chains,
            "num_antibody_antigen_interfaces": item.num_antibody_antigen_interfaces,
            "contact_score": item.contact_score,
            "num_unclustered_monomer_members": item.num_unclustered_monomer_members,
        },
    )

    extraction_manifest = {
        "num_extracted_antibody_complex_structures": 0,
        "num_failed_antibody_complex_structure_extractions": 0,
        "num_antibody_complex_structure_extraction_candidates": 0,
        "num_singleton_antibody_complex_observations_skipped_structure_extraction": 0,
    }
    if structure_refinement_mode == "greedy":
        refinement_observations = [
            observation
            for _, members in signature_groups
            if len(members) > 1
            for observation in members
        ]
        extraction_manifest["num_antibody_complex_structure_extraction_candidates"] = len(refinement_observations)
        extraction_manifest["num_singleton_antibody_complex_observations_skipped_structure_extraction"] = (
            len(observations) - len(refinement_observations)
        )
        LOGGER.info(
            "Antibody complex structure refinement will extract %d/%d observations; %d singleton observations need no USalign",
            len(refinement_observations),
            len(observations),
            extraction_manifest["num_singleton_antibody_complex_observations_skipped_structure_extraction"],
        )

    if structure_refinement_mode == "greedy":
        extracted_structures: dict[str, ExtractedAntibodyComplexStructure] = {}
        structure_rows: list[dict[str, Any]] = []
        structure_failure_rows: list[dict[str, Any]] = []

        def _record_extraction_result(
            structures: dict[str, ExtractedAntibodyComplexStructure],
            extract_manifest: dict[str, Any],
            group_outdir: Path,
        ) -> None:
            extraction_manifest["num_extracted_antibody_complex_structures"] += extract_manifest.get(
                "num_extracted_antibody_complex_structures",
                0,
            )
            extraction_manifest["num_failed_antibody_complex_structure_extractions"] += extract_manifest.get(
                "num_failed_antibody_complex_structure_extractions",
                0,
            )
            structure_rows.extend(item.to_dict() for item in structures.values())
            failures_path = group_outdir / "antibody_complex_structure_extraction_failures.jsonl"
            if failures_path.exists():
                for line in failures_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        structure_failure_rows.append(json.loads(line))

        multi_member_groups = [
            (signature_cluster_id, members)
            for signature_cluster_id, members in signature_groups
            if len(members) > 1
        ]
        if multi_member_groups:
            max_extract_workers = min(normalize_worker_count(alignment_jobs), len(multi_member_groups))
            with ThreadPoolExecutor(max_workers=max_extract_workers) as executor:
                future_to_group = {}
                for signature_cluster_id, members in multi_member_groups:
                    group_outdir = outdir / "structures" / "groups" / signature_cluster_id
                    future = executor.submit(
                        extract_antibody_complex_structures,
                        members,
                        outdir=group_outdir,
                        model=model,
                        drop_hydrogens=drop_hydrogens,
                        interface_residue_cutoff=antibody_complex_interface_residue_cutoff,
                        extraction_jobs=1,
                        prep_dir=prep_dir,
                        show_progress=False,
                        log_summary=False,
                    )
                    future_to_group[future] = (signature_cluster_id, members, group_outdir)
                future_iter = tqdm(
                    as_completed(future_to_group),
                    total=len(future_to_group),
                    desc="Extracting antibody signature groups",
                    unit="sig-group",
                )
                for future in future_iter:
                    signature_cluster_id, members, group_outdir = future_to_group[future]
                    structures, extract_manifest = future.result()
                    extracted_structures.update(structures)
                    _record_extraction_result(structures, extract_manifest, group_outdir)

        refined = refine_antibody_complex_signature_clusters(
            signature_groups,
            extracted_structures,
            tm_score_threshold=antibody_complex_tm_score_threshold,
            usalign_executable=usalign_executable,
            alignment_runner=alignment_runner,
            alignment_jobs=alignment_jobs,
            show_progress=True,
            log_summary=True,
        )
        membership_rows = refined["membership_rows"]
        representative_rows = refined["representative_rows"]
        signature_rows = refined["signature_rows"]
        alignment_rows = refined["alignment_rows"]
        warning_rows = refined["warning_rows"]
        refined_manifest = refined["manifest"]

        def _antibody_cluster_sort_key(cluster_id: str) -> tuple[int, int]:
            _, sig_idx, local_idx = cluster_id.split("_")
            return (int(sig_idx), int(local_idx))

        membership_rows.sort(
            key=lambda row: (
                _antibody_cluster_sort_key(row["antibody_complex_cluster_id"]),
                row["complex_observation_id"],
            )
        )
        representative_rows.sort(
            key=lambda row: _antibody_cluster_sort_key(row["antibody_complex_cluster_id"])
        )
        signature_rows.sort(key=lambda row: _antibody_cluster_sort_key(row["antibody_complex_cluster_id"]))

        structures_dir = outdir / "structures"
        dump_jsonl(
            structures_dir / "antibody_complex_structure_extraction_failures.jsonl",
            structure_failure_rows,
        )
        dump_jsonl(structures_dir / "antibody_complex_structures.jsonl", structure_rows)
        dump_json(
            structures_dir / "antibody_complex_structure_manifest.json",
            {
                "num_antibody_complex_observations": extraction_manifest[
                    "num_antibody_complex_structure_extraction_candidates"
                ],
                "num_extracted_antibody_complex_structures": extraction_manifest[
                    "num_extracted_antibody_complex_structures"
                ],
                "num_failed_antibody_complex_structure_extractions": extraction_manifest[
                    "num_failed_antibody_complex_structure_extractions"
                ],
                "extraction_jobs": normalize_worker_count(alignment_jobs),
                "num_signature_groups_pipelined": len(multi_member_groups),
                "antibody_complex_interface_residue_cutoff": (
                    float(antibody_complex_interface_residue_cutoff)
                    if antibody_complex_interface_residue_cutoff is not None
                    else None
                ),
            },
            indent=2,
        )
        manifest = {
            "num_antibody_complex_observations": len(observations),
            "num_antibody_complex_clusters": refined_manifest["num_antibody_complex_clusters"],
            "num_signature_clusters": refined_manifest["num_signature_clusters"],
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1 for observation in observations if observation.num_unclustered_monomer_members > 0
            ),
            "num_alignment_runs": refined_manifest["num_alignment_runs"],
            "num_alignment_failures": refined_manifest["num_alignment_failures"],
            "num_signature_clusters_split": refined_manifest["num_signature_clusters_split"],
            "antibody_complex_tm_score_threshold": antibody_complex_tm_score_threshold,
            "antibody_complex_interface_residue_cutoff": (
                float(antibody_complex_interface_residue_cutoff)
                if antibody_complex_interface_residue_cutoff is not None
                else None
            ),
            "structure_refinement_mode": structure_refinement_mode,
            "alignment_jobs": refined_manifest["alignment_jobs"],
            **extraction_manifest,
        }
    else:
        membership_rows = []
        representative_rows = []
        signature_rows = []
        alignment_rows = []
        warning_rows = []
        for signature_cluster_id, members in signature_groups:
            representative = max(
                members,
                key=lambda item: (
                    item.contact_score,
                    item.num_antibody_antigen_interfaces,
                    item.num_antigen_chains,
                    item.complex_observation_id,
                ),
            )
            cluster_id = _antibody_complex_cluster_id(signature_cluster_id)
            representative_rows.append(
                {
                    "antibody_complex_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "representative_complex_observation_id": representative.complex_observation_id,
                    "num_members": len(members),
                    "pdb_id": representative.pdb_id,
                    "assembly_id": representative.assembly_id or "",
                    "complex_id": representative.complex_id,
                    "antibody_unit_type": representative.antibody_unit_type,
                    "contact_score": representative.contact_score,
                    "signature_key": representative.signature_key,
                }
            )
            signature_rows.append(
                {
                    "antibody_complex_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "signature_key": representative.signature_key,
                    "antibody_unit_type": representative.antibody_unit_type,
                    "antibody_member_descriptors": json.dumps(
                        representative.antibody_member_descriptors,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "antigen_member_descriptors": json.dumps(
                        representative.antigen_member_descriptors,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "auxiliary_member_descriptors": json.dumps(
                        representative.auxiliary_member_descriptors,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "num_refined_clusters_in_signature_group": 1,
                }
            )
            for member in sorted(members, key=lambda item: item.complex_observation_id):
                membership_rows.append(
                    {
                        "antibody_complex_cluster_id": cluster_id,
                        "signature_cluster_id": signature_cluster_id,
                        "complex_observation_id": member.complex_observation_id,
                        "pdb_id": member.pdb_id,
                        "assembly_id": member.assembly_id or "",
                        "complex_id": member.complex_id,
                        "antibody_unit_type": member.antibody_unit_type,
                        "antibody_chain_ids": json.dumps(member.antibody_chain_ids, ensure_ascii=False),
                        "antigen_chain_ids": json.dumps(member.antigen_chain_ids, ensure_ascii=False),
                        "antigen_chain_types": json.dumps(
                            member.antigen_chain_types,
                            ensure_ascii=False,
                        ),
                        "structural_auxiliary_chain_ids": json.dumps(
                            member.structural_auxiliary_chain_ids,
                            ensure_ascii=False,
                        ),
                        "num_antigen_chains": member.num_antigen_chains,
                        "num_antibody_antigen_interfaces": member.num_antibody_antigen_interfaces,
                        "contact_score": member.contact_score,
                        "num_unclustered_monomer_members": member.num_unclustered_monomer_members,
                        "representative_complex_observation_id": representative.complex_observation_id,
                        "tm_score_to_representative": "",
                    }
                )
        manifest = {
            "num_antibody_complex_observations": len(observations),
            "num_antibody_complex_clusters": len(grouped),
            "num_signature_clusters": len(grouped),
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1 for observation in observations if observation.num_unclustered_monomer_members > 0
            ),
            "num_alignment_runs": 0,
            "num_alignment_failures": 0,
            "num_signature_clusters_split": 0,
            "antibody_complex_tm_score_threshold": antibody_complex_tm_score_threshold,
            "antibody_complex_interface_residue_cutoff": (
                float(antibody_complex_interface_residue_cutoff)
                if antibody_complex_interface_residue_cutoff is not None
                else None
            ),
            "structure_refinement_mode": structure_refinement_mode,
            **extraction_manifest,
        }

    dump_csv_rows(outdir / "antibody_complex_cluster_membership.csv", membership_rows)
    dump_csv_rows(outdir / "antibody_complex_cluster_representatives.csv", representative_rows)
    dump_csv_rows(outdir / "antibody_complex_cluster_signatures.csv", signature_rows)
    dump_jsonl(outdir / "antibody_complex_pairwise_alignments.jsonl", alignment_rows)
    dump_jsonl(outdir / "antibody_complex_cluster_warnings.jsonl", warning_rows)
    dump_json(outdir / "antibody_complex_cluster_manifest.json", manifest, indent=2)
    return {
        "manifest": manifest,
        "observations": observations,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "signature_rows": signature_rows,
        "alignment_rows": alignment_rows,
    }
