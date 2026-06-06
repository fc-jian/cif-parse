from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

import biotite.structure as struc
from biotite.structure import AtomArray, get_residues
from biotite.structure.io.pdb import PDBFile

from cif_parse.clustering.common import (
    canonical_monomer_id,
    load_monomer_cluster_assignments,
    load_monomer_inventory,
    monomer_inventory_source_paths,
    resolve_monomer_cluster,
)
from cif_parse.clustering.dimers import _resolve_usalign_executable
from cif_parse.clustering.high_order_refinement import refine_signature_groups_greedy
from cif_parse.clustering.protein_structures import (
    USalignAlignmentResult,
    parse_usalign_output,
)
from cif_parse.clustering.parallel import iter_threaded_results, normalize_worker_count
from cif_parse.clustering.signature_outputs import write_signature_cluster_membership_csv
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.settings import resolve_source_path
from cif_parse.utils.atom_filters import atom_array_filter_counts, filter_atom_array_for_analysis


LOGGER = logging.getLogger(__name__)

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _signature_index(signature_cluster_id: str) -> str:
    return signature_cluster_id.rsplit("_", 1)[-1]


def _tcr_complex_cluster_id(signature_cluster_id: str, local_index: int = 1) -> str:
    return f"tcrcx_{_signature_index(signature_cluster_id)}_{local_index}"


def _tcr_signature_cluster_id(index: int) -> str:
    return f"tcrsig_{index}"


def _tcr_roles(tcr_type: str, count: int) -> list[str]:
    if tcr_type == "alpha_beta":
        base = ["alpha", "beta"]
    elif tcr_type == "gamma_delta":
        base = ["gamma", "delta"]
    else:
        base = []
    if count <= len(base):
        return base[:count]
    return [*base, *[f"tcr_{index}" for index in range(len(base) + 1, count + 1)]]


@dataclass(slots=True)
class TcrComplexObservation:
    complex_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    assembly_mode: str
    complex_id: str
    tcr_type: str
    mhc_class: str
    tcr_chain_ids: list[str]
    tcr_auth_asym_ids: list[str]
    mhc_chain_ids: list[str]
    mhc_auth_asym_ids: list[str]
    mhc_chain_roles: list[str]
    peptide_chain_ids: list[str]
    peptide_auth_asym_ids: list[str]
    auxiliary_chain_ids: list[str]
    auxiliary_auth_asym_ids: list[str]
    structural_auxiliary_chain_ids: list[str]
    num_tcr_chains: int
    num_peptide_chains: int
    num_tcr_pmhc_interfaces: int
    contact_score: float
    tcr_member_descriptors: list[dict[str, str]]
    mhc_member_descriptors: list[dict[str, str]]
    peptide_member_descriptors: list[dict[str, str]]
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
            "tcr_type": self.tcr_type,
            "mhc_class": self.mhc_class,
            "tcr_chain_ids": json.dumps(self.tcr_chain_ids, ensure_ascii=False),
            "mhc_chain_ids": json.dumps(self.mhc_chain_ids, ensure_ascii=False),
            "mhc_chain_roles": json.dumps(self.mhc_chain_roles, ensure_ascii=False),
            "peptide_chain_ids": json.dumps(self.peptide_chain_ids, ensure_ascii=False),
            "auxiliary_chain_ids": json.dumps(self.auxiliary_chain_ids, ensure_ascii=False),
            "structural_auxiliary_chain_ids": json.dumps(
                self.structural_auxiliary_chain_ids,
                ensure_ascii=False,
            ),
            "num_tcr_chains": self.num_tcr_chains,
            "num_peptide_chains": self.num_peptide_chains,
            "num_tcr_pmhc_interfaces": self.num_tcr_pmhc_interfaces,
            "contact_score": self.contact_score,
            "tcr_member_descriptors": json.dumps(
                self.tcr_member_descriptors,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "mhc_member_descriptors": json.dumps(
                self.mhc_member_descriptors,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "peptide_member_descriptors": json.dumps(
                self.peptide_member_descriptors,
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
            -float(self.num_tcr_pmhc_interfaces),
            -float(self.num_peptide_chains),
            self.complex_observation_id,
        )


@dataclass(slots=True)
class ExtractedTcrComplexStructure:
    complex_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    complex_id: str
    extracted_pdb_path: str
    residue_count: int
    atom_count: int
    filter_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pdb_chain_id(index: int) -> str:
    if index >= len(PDB_CHAIN_IDS):
        raise ValueError(f"Too many TCR-complex chains for PDB export: {index + 1}")
    return PDB_CHAIN_IDS[index]


def _select_chain_atoms(atom_array: AtomArray, chain_id: str, sym_id: int | None = None) -> AtomArray:
    mask = atom_array.chain_id == chain_id
    if hasattr(atom_array, "hetero"):
        mask &= ~atom_array.hetero
    if sym_id is not None and hasattr(atom_array, "sym_id"):
        mask &= atom_array.sym_id == sym_id
    return atom_array[mask]


def _coerce_chain_id(atom_array: AtomArray, chain_id: str) -> AtomArray:
    copied = atom_array.copy()
    copied.chain_id[:] = chain_id
    return copied


def _structure_chain_ids(
    observation: TcrComplexObservation,
) -> list[str]:
    chain_ids: list[str] = []
    seen: set[str] = set()
    for chain_id in [
        *observation.tcr_chain_ids,
        *observation.mhc_chain_ids,
        *observation.peptide_chain_ids,
        *observation.structural_auxiliary_chain_ids,
    ]:
        if chain_id and chain_id not in seen:
            seen.add(chain_id)
            chain_ids.append(chain_id)
    return chain_ids


def _build_tcr_observation(
    *,
    pdb_id: str, source_path: str, assembly_id: str | None, assembly_mode: str,
    complex_id: str, complex_index: int,
    tcr_type: str, mhc_class: str,
    tcr_chain_ids: list[str], tcr_auth_asym_ids: list[str],
    mhc_chain_ids: list[str], mhc_auth_asym_ids: list[str],
    mhc_chain_roles: list[str],
    peptide_chain_ids: list[str], peptide_auth_asym_ids: list[str],
    auxiliary_chain_ids: list[str], auxiliary_auth_asym_ids: list[str],
    num_tcr_chains: int, num_peptide_chains: int,
    num_tcr_pmhc_interfaces: int, contact_score: float,
    monomer_cluster_assignments: dict[str, dict[str, str]],
    monomer_inventory: dict[str, dict[str, Any]],
) -> TcrComplexObservation:
    num_unclustered = 0
    tcr_asm = assembly_id or ""
    tcr_desc: list[dict[str, str]] = []
    for role, cid in zip(_tcr_roles(tcr_type, len(tcr_chain_ids)), tcr_chain_ids):
        cs, cluster_id, _ = resolve_monomer_cluster(canonical_monomer_id(pdb_id, cid, tcr_asm), monomer_cluster_assignments)
        if cs == "unclustered": num_unclustered += 1
        tcr_desc.append({"role": role, "monomer_cluster_id": cluster_id})

    mhc_desc: list[dict[str, str]] = []
    for cid, role in zip(mhc_chain_ids, mhc_chain_roles):
        cs, cluster_id, _ = resolve_monomer_cluster(canonical_monomer_id(pdb_id, cid, tcr_asm), monomer_cluster_assignments)
        if cs == "unclustered": num_unclustered += 1
        mhc_desc.append({"role": role, "monomer_cluster_id": cluster_id})

    pep_desc: list[dict[str, str]] = []
    for cid in peptide_chain_ids:
        mid = canonical_monomer_id(pdb_id, cid, tcr_asm)
        cp = monomer_inventory.get(mid, {})
        cs, cluster_id, _ = resolve_monomer_cluster(mid, monomer_cluster_assignments)
        if cs == "unclustered": num_unclustered += 1
        pep_desc.append({"chain_type": str(cp.get("chain_type", "") or ""), "monomer_cluster_id": cluster_id})

    aux_desc: list[dict[str, str]] = []
    structural_aux: list[str] = []
    for cid in auxiliary_chain_ids:
        mid = canonical_monomer_id(pdb_id, cid, tcr_asm)
        cp = monomer_inventory.get(mid, {})
        if mid in monomer_inventory or mid in monomer_cluster_assignments:
            cs, cluster_id, _ = resolve_monomer_cluster(mid, monomer_cluster_assignments)
            if cs == "unclustered": num_unclustered += 1
            aux_desc.append({"chain_type": str(cp.get("chain_type", "") or ""), "monomer_cluster_id": cluster_id})
            structural_aux.append(cid)
        else:
            aux_desc.append({"chain_type": "auxiliary_nonpolymer", "monomer_cluster_id": "auxiliary_nonpolymer"})

    sig_key = json.dumps({
        "tcr_type": tcr_type, "mhc_class": mhc_class,
        "tcr_members": sorted(tcr_desc, key=lambda x: (x["role"], x["monomer_cluster_id"])),
        "mhc_members": sorted(mhc_desc, key=lambda x: (x["role"], x["monomer_cluster_id"])),
        "peptide_members": sorted(pep_desc, key=lambda x: (x["chain_type"], x["monomer_cluster_id"])),
        "auxiliary_members": sorted(aux_desc, key=lambda x: (x["chain_type"], x["monomer_cluster_id"])),
    }, ensure_ascii=False, sort_keys=True)

    return TcrComplexObservation(
        complex_observation_id=f"{pdb_id}:{assembly_id or 'na'}:{complex_index}",
        pdb_id=pdb_id, source_path=source_path, assembly_id=assembly_id,
        assembly_mode=assembly_mode, complex_id=complex_id,
        tcr_type=tcr_type, mhc_class=mhc_class,
        tcr_chain_ids=tcr_chain_ids, tcr_auth_asym_ids=tcr_auth_asym_ids,
        mhc_chain_ids=mhc_chain_ids, mhc_auth_asym_ids=mhc_auth_asym_ids,
        mhc_chain_roles=mhc_chain_roles,
        peptide_chain_ids=peptide_chain_ids, peptide_auth_asym_ids=peptide_auth_asym_ids,
        auxiliary_chain_ids=auxiliary_chain_ids, auxiliary_auth_asym_ids=auxiliary_auth_asym_ids,
        structural_auxiliary_chain_ids=structural_aux,
        num_tcr_chains=num_tcr_chains, num_peptide_chains=num_peptide_chains,
        num_tcr_pmhc_interfaces=num_tcr_pmhc_interfaces, contact_score=contact_score,
        tcr_member_descriptors=tcr_desc, mhc_member_descriptors=mhc_desc,
        peptide_member_descriptors=pep_desc, auxiliary_member_descriptors=aux_desc,
        signature_key=sig_key, num_unclustered_monomer_members=num_unclustered,
    )


def collect_tcr_complex_observations(
    case_dirs: Iterable[str | Path],
    monomer_cluster_assignments: dict[str, dict[str, str]],
    monomer_inventory: dict[str, dict[str, Any]],
    cif_files_directory: str | None = None,
    prep_db_path: str | Path | None = None,
    prep_dir: str | Path | None = None,
    allowed_source_paths: set[str] | None = None,
) -> list[TcrComplexObservation]:
    observations: list[TcrComplexObservation] = []

    # Fast path: read pre-parsed Parquet
    from cif_parse.clustering.prep import iter_parquet_rows
    if prep_dir:
        for row in tqdm(iter_parquet_rows(prep_dir, "tcr_complexes", required=True),
                        desc="Collecting TCR complexes", unit="complex"):
            pdb_id = row.get("pdb_id", "")
            if not pdb_id:
                continue
            sp = resolve_source_path(row.get("source_path", ""), cif_files_directory)
            if allowed_source_paths is not None and sp not in allowed_source_paths:
                continue
            observations.append(_build_tcr_observation(
                pdb_id=pdb_id, source_path=sp,
                assembly_id=row.get("assembly_id"),
                assembly_mode=row.get("assembly_mode", ""),
                complex_id=row.get("complex_id", ""),
                complex_index=row.get("complex_index", 0),
                tcr_type=row.get("tcr_type", ""),
                mhc_class=row.get("mhc_class", ""),
                tcr_chain_ids=json.loads(row.get("tcr_chain_ids", "[]")),
                tcr_auth_asym_ids=json.loads(row.get("tcr_auth_asym_ids", "[]")),
                mhc_chain_ids=json.loads(row.get("mhc_chain_ids", "[]")),
                mhc_auth_asym_ids=json.loads(row.get("mhc_auth_asym_ids", "[]")),
                mhc_chain_roles=json.loads(row.get("mhc_chain_roles", "[]")),
                peptide_chain_ids=json.loads(row.get("peptide_chain_ids", "[]")),
                peptide_auth_asym_ids=json.loads(row.get("peptide_auth_asym_ids", "[]")),
                auxiliary_chain_ids=json.loads(row.get("auxiliary_chain_ids", "[]")),
                auxiliary_auth_asym_ids=json.loads(row.get("auxiliary_auth_asym_ids", "[]")),
                num_tcr_chains=row.get("num_tcr_chains", 0),
                num_peptide_chains=row.get("num_peptide_chains", 0),
                num_tcr_pmhc_interfaces=row.get("num_tcr_pmhc_interfaces", 0),
                contact_score=row.get("contact_score", 0.0),
                monomer_cluster_assignments=monomer_cluster_assignments,
                monomer_inventory=monomer_inventory,
            ))
        LOGGER.info("Collected %d TCR complex observations from prep Parquet", len(observations))
        return observations

    from cif_parse.clustering.prep import load_bundles_for_collect, load_case_bundles

    sorted_dirs = sorted(Path(path).resolve() for path in case_dirs)
    prep_bundles = load_bundles_for_collect(sorted_dirs, prep_db_path=prep_db_path)
    for case_dir in tqdm(sorted_dirs, desc="Collecting TCR complexes", unit="case"):
        payloads = load_case_bundles(case_dir, prep_bundles=prep_bundles)
        for payload in payloads:
            summary = payload.get("structure_summary", {})
            pdb_id = str(summary.get("pdb_id", "") or "")
            source_path = resolve_source_path(
                str(summary.get("source_path", "") or ""),
                cif_files_directory,
            )
            if allowed_source_paths is not None and source_path not in allowed_source_paths:
                continue
            assembly_ids = [str(item) for item in summary.get("assembly_ids", []) if str(item)]
            default_assembly_id = assembly_ids[0] if len(assembly_ids) == 1 else None
            complexes = payload.get("tcr_pmhc_complexes", [])
            if not isinstance(complexes, list):
                continue
            for index, complex_payload in enumerate(complexes, start=1):
                if not isinstance(complex_payload, dict):
                    continue
                tcr_chain_ids = [str(item) for item in complex_payload.get("tcr_chain_ids", []) if str(item)]
                mhc_chain_ids = [str(item) for item in complex_payload.get("mhc_chain_ids", []) if str(item)]
                if not tcr_chain_ids or not mhc_chain_ids:
                    continue

                num_unclustered = 0
                slow_tcr_asm = str(complex_payload.get("assembly_id") or default_assembly_id or "")
                tcr_member_descriptors: list[dict[str, str]] = []
                for role, chain_id in zip(
                    _tcr_roles(str(complex_payload.get("tcr_type", "") or ""), len(tcr_chain_ids)),
                    tcr_chain_ids,
                    strict=False,
                ):
                    monomer_id = canonical_monomer_id(pdb_id, chain_id, slow_tcr_asm)
                    cluster_source, cluster_id, _ = resolve_monomer_cluster(
                        monomer_id,
                        monomer_cluster_assignments,
                    )
                    if cluster_source == "unclustered":
                        num_unclustered += 1
                    tcr_member_descriptors.append(
                        {
                            "role": role,
                            "monomer_cluster_id": cluster_id,
                        }
                    )

                mhc_member_descriptors: list[dict[str, str]] = []
                for chain_id, role in zip(
                    mhc_chain_ids,
                    [str(item) for item in complex_payload.get("mhc_chain_roles", [])],
                    strict=False,
                ):
                    monomer_id = canonical_monomer_id(pdb_id, chain_id, slow_tcr_asm)
                    cluster_source, cluster_id, _ = resolve_monomer_cluster(
                        monomer_id,
                        monomer_cluster_assignments,
                    )
                    if cluster_source == "unclustered":
                        num_unclustered += 1
                    mhc_member_descriptors.append(
                        {
                            "role": role,
                            "monomer_cluster_id": cluster_id,
                        }
                    )

                peptide_member_descriptors: list[dict[str, str]] = []
                for chain_id in complex_payload.get("peptide_chain_ids", []) or []:
                    chain_label = str(chain_id)
                    monomer_id = canonical_monomer_id(pdb_id, chain_label, slow_tcr_asm)
                    chain_payload = monomer_inventory.get(monomer_id, {})
                    cluster_source, cluster_id, _ = resolve_monomer_cluster(
                        monomer_id,
                        monomer_cluster_assignments,
                    )
                    if cluster_source == "unclustered":
                        num_unclustered += 1
                    peptide_member_descriptors.append(
                        {
                            "chain_type": str(chain_payload.get("chain_type", "") or ""),
                            "monomer_cluster_id": cluster_id,
                        }
                    )

                auxiliary_member_descriptors: list[dict[str, str]] = []
                structural_auxiliary_chain_ids: list[str] = []
                for chain_id in complex_payload.get("auxiliary_chain_ids", []) or []:
                    chain_label = str(chain_id)
                    monomer_id = canonical_monomer_id(pdb_id, chain_label, slow_tcr_asm)
                    chain_payload = monomer_inventory.get(monomer_id, {})
                    if monomer_id in monomer_inventory or monomer_id in monomer_cluster_assignments:
                        cluster_source, cluster_id, _ = resolve_monomer_cluster(
                            monomer_id,
                            monomer_cluster_assignments,
                        )
                        if cluster_source == "unclustered":
                            num_unclustered += 1
                        auxiliary_member_descriptors.append(
                            {
                                "chain_type": str(chain_payload.get("chain_type", "") or ""),
                                "monomer_cluster_id": cluster_id,
                            }
                        )
                        structural_auxiliary_chain_ids.append(chain_label)
                    else:
                        auxiliary_member_descriptors.append(
                            {
                                "chain_type": "auxiliary_nonpolymer",
                                "monomer_cluster_id": "auxiliary_nonpolymer",
                            }
                        )

                signature_key = json.dumps(
                    {
                        "tcr_type": str(complex_payload.get("tcr_type", "") or ""),
                        "mhc_class": str(complex_payload.get("mhc_class", "") or ""),
                        "tcr_members": sorted(
                            tcr_member_descriptors,
                            key=lambda item: (item["role"], item["monomer_cluster_id"]),
                        ),
                        "mhc_members": sorted(
                            mhc_member_descriptors,
                            key=lambda item: (item["role"], item["monomer_cluster_id"]),
                        ),
                        "peptide_members": sorted(
                            peptide_member_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                        "auxiliary_members": sorted(
                            auxiliary_member_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                assembly_id = complex_payload.get("assembly_id")
                observations.append(
                    TcrComplexObservation(
                        complex_observation_id=(
                            f"{pdb_id}:{assembly_id or default_assembly_id or 'na'}:{index}"
                        ),
                        pdb_id=pdb_id,
                        source_path=source_path,
                        assembly_id=str(assembly_id) if assembly_id is not None else default_assembly_id,
                        assembly_mode=str(complex_payload.get("assembly_mode", "") or ""),
                        complex_id=str(complex_payload.get("complex_id", "") or ""),
                        tcr_type=str(complex_payload.get("tcr_type", "") or ""),
                        mhc_class=str(complex_payload.get("mhc_class", "") or ""),
                        tcr_chain_ids=tcr_chain_ids,
                        tcr_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("tcr_auth_asym_ids", [])
                            if str(item)
                        ],
                        mhc_chain_ids=mhc_chain_ids,
                        mhc_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("mhc_auth_asym_ids", [])
                            if str(item)
                        ],
                        mhc_chain_roles=[
                            str(item) for item in complex_payload.get("mhc_chain_roles", [])
                        ],
                        peptide_chain_ids=[
                            str(item)
                            for item in complex_payload.get("peptide_chain_ids", [])
                            if str(item)
                        ],
                        peptide_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("peptide_auth_asym_ids", [])
                            if str(item)
                        ],
                        auxiliary_chain_ids=[
                            str(item)
                            for item in complex_payload.get("auxiliary_chain_ids", [])
                            if str(item)
                        ],
                        auxiliary_auth_asym_ids=[
                            str(item)
                            for item in complex_payload.get("auxiliary_auth_asym_ids", [])
                            if str(item)
                        ],
                        structural_auxiliary_chain_ids=structural_auxiliary_chain_ids,
                        num_tcr_chains=int(complex_payload.get("num_tcr_chains", 0) or 0),
                        num_peptide_chains=int(complex_payload.get("num_peptide_chains", 0) or 0),
                        num_tcr_pmhc_interfaces=int(
                            complex_payload.get("num_tcr_pmhc_interfaces", 0) or 0
                        ),
                        contact_score=float(complex_payload.get("contact_score", 0.0) or 0.0),
                        tcr_member_descriptors=sorted(
                            tcr_member_descriptors,
                            key=lambda item: (item["role"], item["monomer_cluster_id"]),
                        ),
                        mhc_member_descriptors=sorted(
                            mhc_member_descriptors,
                            key=lambda item: (item["role"], item["monomer_cluster_id"]),
                        ),
                        peptide_member_descriptors=sorted(
                            peptide_member_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                        auxiliary_member_descriptors=sorted(
                            auxiliary_member_descriptors,
                            key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                        ),
                        signature_key=signature_key,
                        num_unclustered_monomer_members=num_unclustered,
                    )
                )
    LOGGER.info("Collected %d TCR complex observations from %d case dir(s)", len(observations), len(sorted_dirs))
    return observations


def extract_tcr_complex_structure(
    observation: TcrComplexObservation,
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    atom_array: AtomArray | None = None,
) -> ExtractedTcrComplexStructure:
    if not observation.source_path:
        raise ValueError(f"Missing source_path for TCR complex {observation.complex_observation_id}")
    if atom_array is None:
        raise ValueError(f"Cached coordinates are required for TCR complex {observation.complex_observation_id}")

    chain_ids = _structure_chain_ids(observation)
    # In assembly mode, sym copies of the same chain exist. Pick one
    # representative sym_id per chain to avoid extracting 60x copies.
    chain_sym: dict[str, int | None] = {}
    _has_sym = hasattr(atom_array, "sym_id")
    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        if not chain_mask.any():
            continue
        if _has_sym:
            chain_syms = frozenset(atom_array.sym_id[chain_mask])
            chain_sym[chain_id] = min(chain_syms) if chain_syms else None
        else:
            chain_sym[chain_id] = None

    sym_sets = [frozenset(atom_array.sym_id[atom_array.chain_id == cid])
                for cid in chain_ids
                if _has_sym and (atom_array.chain_id == cid).any()
                and frozenset(atom_array.sym_id[atom_array.chain_id == cid])]
    common_sym: int | None = None
    if sym_sets:
        common = sym_sets[0]
        for s in sym_sets[1:]:
            common = common & s
        if common:
            common_sym = min(common)

    chain_arrays: list[AtomArray] = []
    for chain_index, chain_id in enumerate(chain_ids):
        sym = common_sym if common_sym is not None else chain_sym.get(chain_id)
        selected = _select_chain_atoms(atom_array, chain_id, sym_id=sym)
        if selected.array_length() == 0:
            raise ValueError(
                f"Requested TCR complex chain {chain_id!r} is unavailable for "
                f"{observation.complex_observation_id}"
            )
        chain_arrays.append(_coerce_chain_id(selected, _pdb_chain_id(chain_index)))
    if not chain_arrays:
        raise ValueError(f"No polymer atoms found for TCR complex {observation.complex_observation_id}")

    complex_atoms = struc.concatenate(chain_arrays)
    complex_atoms, filter_counts = filter_atom_array_for_analysis(
        complex_atoms,
        drop_hydrogens=drop_hydrogens,
        drop_nonfinite=True,
    )
    if complex_atoms.array_length() == 0:
        raise ValueError(
            f"No analyzable atoms left for TCR complex {observation.complex_observation_id}"
        )
    _, residue_names = get_residues(complex_atoms)
    residue_count = int(len(residue_names))
    if residue_count <= 2:
        raise ValueError(
            f"Resolved residue count {residue_count} is too short for TCR complex USalign: "
            f"{observation.complex_observation_id}"
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdb_path = outdir / (
        f"{observation.pdb_id}_{observation.assembly_id or 'asu'}_{observation.complex_id}.pdb"
    )
    pdb_file = PDBFile()
    pdb_file.set_structure(complex_atoms)
    pdb_file.write(pdb_path)
    return ExtractedTcrComplexStructure(
        complex_observation_id=observation.complex_observation_id,
        pdb_id=observation.pdb_id,
        source_path=observation.source_path,
        assembly_id=observation.assembly_id,
        complex_id=observation.complex_id,
        extracted_pdb_path=str(pdb_path),
        residue_count=residue_count,
        atom_count=int(complex_atoms.array_length()),
        filter_counts=atom_array_filter_counts(filter_counts),
    )


def extract_tcr_complex_structures(
    observations: Iterable[TcrComplexObservation],
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    extraction_jobs: int = 1,
    prep_dir: str | Path | None = None,
    show_progress: bool = True,
    log_summary: bool = True,
    raise_on_failure: bool = False,
) -> tuple[dict[str, ExtractedTcrComplexStructure], dict[str, Any]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    structures: dict[str, ExtractedTcrComplexStructure] = {}
    failures: list[dict[str, str]] = []

    sorted_observations = sorted(
        observations,
        key=lambda item: (
            item.source_path,
            item.assembly_id or "",
            item.complex_observation_id,
        ),
    )
    extraction_jobs = normalize_worker_count(extraction_jobs)

    def _try_assemble_tcr(obs: TcrComplexObservation) -> AtomArray | None:
        if prep_dir is None:
            return None
        from cif_parse.clustering.prep import assemble_atom_array_from_chains

        chain_ids = _structure_chain_ids(obs)
        if not chain_ids:
            return None
        return assemble_atom_array_from_chains(
            prep_dir,
            obs.source_path,
            [(cid, None) for cid in chain_ids],
            assembly_id=obs.assembly_id,
        )

    def _process_one(observation: TcrComplexObservation) -> ExtractedTcrComplexStructure | None:
        atom_array = _try_assemble_tcr(observation)
        if atom_array is None:
            raise ValueError(
                f"Prep coordinates missing for TCR complex {observation.complex_observation_id}"
            )
        return extract_tcr_complex_structure(
            observation,
            outdir=outdir,
            model=model,
            drop_hydrogens=drop_hydrogens,
            atom_array=atom_array,
        )

    for observation, extracted, error in iter_threaded_results(
        sorted_observations,
        _process_one,
        max_workers=min(extraction_jobs, max(1, len(sorted_observations))),
        total=len(sorted_observations),
        show_progress=show_progress,
        progress_desc="Extracting TCR complex structures",
        progress_unit="complex",
    ):
        if error is not None:
            if raise_on_failure:
                raise RuntimeError(
                    f"Failed to extract TCR complex {observation.complex_observation_id}: {error}"
                ) from error
            LOGGER.debug(
                "Failed to extract TCR complex %s: %s",
                observation.complex_observation_id,
                error,
            )
            failures.append(
                {"complex_observation_id": observation.complex_observation_id, "error": str(error)}
            )
            continue
        if extracted is not None:
            structures[observation.complex_observation_id] = extracted

    failure_path = outdir / "tcr_complex_structure_extraction_failures.jsonl"
    dump_jsonl(failure_path, failures)
    dump_jsonl(
        outdir / "tcr_complex_structures.jsonl",
        (item.to_dict() for item in structures.values()),
    )
    manifest = {
        "num_tcr_complex_observations": len(sorted_observations),
        "num_extracted_tcr_complex_structures": len(structures),
        "num_failed_tcr_complex_structure_extractions": len(failures),
        "extraction_jobs": extraction_jobs,
    }
    dump_json(outdir / "tcr_complex_structure_manifest.json", manifest, indent=2)
    if failures:
        LOGGER.warning(
            "Skipped %d failed TCR complex structure extractions; details written to %s",
            len(failures),
            failure_path,
        )
    if log_summary:
        LOGGER.info("Extracted %d TCR complex structures (%d failures)", len(structures), len(failures))
    return structures, manifest


def run_tcr_complex_usalign_alignment(
    query: ExtractedTcrComplexStructure,
    target: ExtractedTcrComplexStructure,
    *,
    usalign_executable: str = "USalign",
    tm_score_threshold: float = 0.50,
    min_alignment_coverage_ratio: float = 0.50,
) -> USalignAlignmentResult:
    resolved_executable = _resolve_usalign_executable(usalign_executable)
    if resolved_executable is None:
        raise FileNotFoundError(f"{usalign_executable} executable not found in PATH")
    command = [
        resolved_executable,
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
        min_alignment_coverage_ratio=min_alignment_coverage_ratio,
    )


def refine_tcr_complex_signature_clusters(
    signature_groups: list[tuple[str, list[TcrComplexObservation]]],
    extracted_structures: dict[str, ExtractedTcrComplexStructure],
    *,
    tm_score_threshold: float = 0.50,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
    min_alignment_coverage_ratio: float = 0.50,
    show_progress: bool = True,
    log_summary: bool = True,
) -> dict[str, Any]:
    runner = alignment_runner or run_tcr_complex_usalign_alignment
    alignment_jobs = normalize_worker_count(alignment_jobs)
    total_observations = sum(len(members) for _, members in signature_groups)
    multi_member = sum(1 for _, m in signature_groups if len(m) > 1)
    if log_summary:
        LOGGER.info(
            "Refining %d TCR complex signature clusters (%d observations, %d multi-member, %d alignment workers)",
            len(signature_groups),
            total_observations,
            multi_member,
            alignment_jobs,
        )
    signature_iter = list(
        tqdm(signature_groups, desc="Refining TCR complex clusters", unit="sig-group")
        if show_progress
        else signature_groups
    )
    refined = refine_signature_groups_greedy(
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
            "alignment_coverage_shorter": result.shorter_length_coverage,
            "alignment_coverage_resolved": result.resolved_length_coverage,
        },
        alignment_failure_warning=lambda signature_cluster_id, representative, candidate, exc: {
            "warning_code": "tcr_complex_usalign_failed",
            "signature_cluster_id": signature_cluster_id,
            "representative_complex_observation_id": representative.complex_observation_id,
            "candidate_complex_observation_id": candidate.complex_observation_id,
            "error": str(exc),
        },
        unavailable_warning=lambda signature_cluster_id, member: {
            "warning_code": "tcr_complex_structure_unavailable_skipped",
            "signature_cluster_id": signature_cluster_id,
            "complex_observation_id": member.complex_observation_id,
        },
        runner=runner,
        alignment_jobs=alignment_jobs,
        usalign_executable=usalign_executable,
        tm_score_threshold=tm_score_threshold,
        min_alignment_coverage_ratio=min_alignment_coverage_ratio,
        show_progress=show_progress,
        can_skip_alignment=lambda a, b: (
            a.source_path == b.source_path
            and a.assembly_id == b.assembly_id
            and sorted(a.tcr_chain_ids) == sorted(b.tcr_chain_ids)
            and sorted(a.mhc_chain_ids) == sorted(b.mhc_chain_ids)
            and sorted(a.peptide_chain_ids) == sorted(b.peptide_chain_ids)
        ),
    )
    alignment_cache = refined.alignment_cache
    alignment_rows = refined.alignment_rows
    warning_rows = refined.warning_rows
    cluster_members = refined.cluster_members
    num_alignment_runs = refined.num_alignment_runs
    num_alignment_failures = refined.num_alignment_failures
    num_signature_clusters_split = refined.num_signature_clusters_split
    num_skipped_missing_structure = refined.num_members_skipped_missing_structure

    grouped_signature_sizes = {signature_cluster_id: 0 for signature_cluster_id, _ in signature_groups}
    for signature_cluster_id, _, _, _ in cluster_members:
        grouped_signature_sizes[signature_cluster_id] = grouped_signature_sizes.get(signature_cluster_id, 0) + 1

    membership_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    local_cluster_counts: dict[str, int] = {}
    for signature_cluster_id, representative_id, members, representative in cluster_members:
        local_cluster_counts[signature_cluster_id] = local_cluster_counts.get(signature_cluster_id, 0) + 1
        cluster_id = _tcr_complex_cluster_id(signature_cluster_id, local_cluster_counts[signature_cluster_id])
        representative_rows.append(
            {
                "tcr_complex_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "representative_complex_observation_id": representative_id,
                "num_members": len(members),
                "pdb_id": representative.pdb_id,
                "assembly_id": representative.assembly_id or "",
                "complex_id": representative.complex_id,
                "tcr_type": representative.tcr_type,
                "mhc_class": representative.mhc_class,
                "contact_score": representative.contact_score,
                "signature_key": representative.signature_key,
            }
        )
        signature_rows.append(
            {
                "tcr_complex_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "signature_key": representative.signature_key,
                "tcr_type": representative.tcr_type,
                "mhc_class": representative.mhc_class,
                "tcr_member_descriptors": json.dumps(
                    representative.tcr_member_descriptors,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "mhc_member_descriptors": json.dumps(
                    representative.mhc_member_descriptors,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "peptide_member_descriptors": json.dumps(
                    representative.peptide_member_descriptors,
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
                    "tcr_complex_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "complex_observation_id": member.complex_observation_id,
                    "pdb_id": member.pdb_id,
                    "assembly_id": member.assembly_id or "",
                    "complex_id": member.complex_id,
                    "tcr_type": member.tcr_type,
                    "mhc_class": member.mhc_class,
                    "tcr_chain_ids": json.dumps(member.tcr_chain_ids, ensure_ascii=False),
                    "mhc_chain_ids": json.dumps(member.mhc_chain_ids, ensure_ascii=False),
                    "mhc_chain_roles": json.dumps(member.mhc_chain_roles, ensure_ascii=False),
                    "peptide_chain_ids": json.dumps(member.peptide_chain_ids, ensure_ascii=False),
                    "auxiliary_chain_ids": json.dumps(member.auxiliary_chain_ids, ensure_ascii=False),
                    "structural_auxiliary_chain_ids": json.dumps(
                        member.structural_auxiliary_chain_ids,
                        ensure_ascii=False,
                    ),
                    "num_tcr_chains": member.num_tcr_chains,
                    "num_peptide_chains": member.num_peptide_chains,
                    "num_tcr_pmhc_interfaces": member.num_tcr_pmhc_interfaces,
                    "contact_score": member.contact_score,
                    "num_unclustered_monomer_members": member.num_unclustered_monomer_members,
                    "representative_complex_observation_id": representative_id,
                    "tm_score_to_representative": tm_score_for_clustering,
                }
            )

    manifest = {
        "num_signature_clusters": len(signature_groups),
        "num_tcr_complex_clusters": len(cluster_members),
        "num_alignment_runs": num_alignment_runs,
        "num_alignment_failures": num_alignment_failures,
        "num_signature_clusters_split": num_signature_clusters_split,
        "num_tcr_complex_observations_skipped_missing_structure": (
            num_skipped_missing_structure
        ),
        "tcr_complex_tm_score_threshold": tm_score_threshold,
        "min_alignment_coverage_ratio": min_alignment_coverage_ratio,
        "alignment_jobs": alignment_jobs,
    }
    if log_summary:
        LOGGER.info(
            "TCR complex refinement: %d signature clusters -> %d refined clusters (%d alignments, %d failures, %d splits, %d skipped missing structure)",
            len(signature_groups),
            len(cluster_members),
            num_alignment_runs,
            num_alignment_failures,
            num_signature_clusters_split,
            num_skipped_missing_structure,
        )
    return {
        "manifest": manifest,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "signature_rows": signature_rows,
        "alignment_rows": alignment_rows,
        "warning_rows": warning_rows,
    }


def build_tcr_complex_signature_clusters(
    *,
    case_dirs: Iterable[str | Path],
    clustering_outdir: str | Path,
    outdir: str | Path,
    structure_refinement_mode: str = "greedy",
    tcr_complex_tm_score_threshold: float = 0.50,
    model: int = 1,
    drop_hydrogens: bool = True,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
    min_alignment_coverage_ratio: float = 0.50,
    cif_files_directory: str | None = None,
    prep_dir: str | Path | None = None,
    include_structure_assignments: bool = True,
) -> dict[str, Any]:
    monomer_assignments = load_monomer_cluster_assignments(
        clustering_outdir,
        include_structure=include_structure_assignments,
    )
    monomer_inventory = load_monomer_inventory(clustering_outdir)
    allowed_source_paths = monomer_inventory_source_paths(monomer_inventory)
    observations = collect_tcr_complex_observations(
        case_dirs,
        monomer_assignments,
        monomer_inventory,
        cif_files_directory=cif_files_directory,
        prep_dir=prep_dir,
        allowed_source_paths=allowed_source_paths,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dump_jsonl(
        outdir / "tcr_complex_inventory.jsonl",
        (item.to_dict() for item in observations),
    )
    dump_csv_rows(outdir / "tcr_complex_inventory.csv", [item.to_record() for item in observations])

    grouped: dict[str, list[TcrComplexObservation]] = {}
    for observation in tqdm(observations, desc="Grouping TCR signatures", unit="complex"):
        grouped.setdefault(observation.signature_key, []).append(observation)
    signature_groups = [
        (_tcr_signature_cluster_id(index), members)
        for index, (_, members) in enumerate(
            sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
            start=1,
        )
    ]
    write_signature_cluster_membership_csv(
        outdir / "tcr_complex_signature_cluster_membership.csv",
        signature_groups,
        observation_id_field="complex_observation_id",
        observation_id=lambda item: item.complex_observation_id,
        extra_fields=lambda item: {
            "complex_id": item.complex_id,
            "tcr_type": item.tcr_type,
            "mhc_class": item.mhc_class,
            "tcr_chain_ids": json.dumps(item.tcr_chain_ids, ensure_ascii=False),
            "mhc_chain_ids": json.dumps(item.mhc_chain_ids, ensure_ascii=False),
            "peptide_chain_ids": json.dumps(item.peptide_chain_ids, ensure_ascii=False),
            "num_tcr_chains": item.num_tcr_chains,
            "num_peptide_chains": item.num_peptide_chains,
            "num_tcr_pmhc_interfaces": item.num_tcr_pmhc_interfaces,
            "contact_score": item.contact_score,
            "num_unclustered_monomer_members": item.num_unclustered_monomer_members,
        },
    )

    extraction_manifest = {
        "num_extracted_tcr_complex_structures": 0,
        "num_failed_tcr_complex_structure_extractions": 0,
        "num_tcr_complex_structure_extraction_candidates": 0,
        "num_singleton_tcr_complex_observations_skipped_structure_extraction": 0,
    }
    if structure_refinement_mode == "greedy":
        refinement_observations = [
            observation
            for _, members in signature_groups
            if len(members) > 1
            for observation in members
        ]
        extraction_manifest["num_tcr_complex_structure_extraction_candidates"] = len(refinement_observations)
        extraction_manifest["num_singleton_tcr_complex_observations_skipped_structure_extraction"] = (
            len(observations) - len(refinement_observations)
        )
        LOGGER.info(
            "TCR complex structure refinement will extract %d/%d observations; %d singleton observations need no USalign",
            len(refinement_observations),
            len(observations),
            extraction_manifest["num_singleton_tcr_complex_observations_skipped_structure_extraction"],
        )

    if structure_refinement_mode == "greedy":
        extracted_structures, extract_manifest = extract_tcr_complex_structures(
            refinement_observations,
            outdir=outdir / "structures",
            model=model,
            drop_hydrogens=drop_hydrogens,
            extraction_jobs=alignment_jobs,
            prep_dir=prep_dir,
            show_progress=True,
            log_summary=True,
            raise_on_failure=False,
        )
        extraction_manifest.update(extract_manifest)

        refined = refine_tcr_complex_signature_clusters(
            signature_groups,
            extracted_structures,
            tm_score_threshold=tcr_complex_tm_score_threshold,
            usalign_executable=usalign_executable,
            alignment_runner=alignment_runner,
            alignment_jobs=alignment_jobs,
            min_alignment_coverage_ratio=min_alignment_coverage_ratio,
            show_progress=True,
            log_summary=True,
        )
        membership_rows = refined["membership_rows"]
        representative_rows = refined["representative_rows"]
        signature_rows = refined["signature_rows"]
        alignment_rows = refined["alignment_rows"]
        warning_rows = refined["warning_rows"]
        refined_manifest = refined["manifest"]

        def _tcr_cluster_sort_key(cluster_id: str) -> tuple[int, int]:
            _, sig_idx, local_idx = cluster_id.split("_")
            return (int(sig_idx), int(local_idx))

        membership_rows.sort(
            key=lambda row: (_tcr_cluster_sort_key(row["tcr_complex_cluster_id"]), row["complex_observation_id"])
        )
        representative_rows.sort(key=lambda row: _tcr_cluster_sort_key(row["tcr_complex_cluster_id"]))
        signature_rows.sort(key=lambda row: _tcr_cluster_sort_key(row["tcr_complex_cluster_id"]))

        manifest = {
            "num_tcr_complex_observations": len(observations),
            "num_tcr_complex_clusters": refined_manifest["num_tcr_complex_clusters"],
            "num_signature_clusters": refined_manifest["num_signature_clusters"],
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1 for observation in observations if observation.num_unclustered_monomer_members > 0
            ),
            "num_alignment_runs": refined_manifest["num_alignment_runs"],
            "num_alignment_failures": refined_manifest["num_alignment_failures"],
            "num_signature_clusters_split": refined_manifest["num_signature_clusters_split"],
            "num_tcr_complex_observations_skipped_missing_structure": refined_manifest[
                "num_tcr_complex_observations_skipped_missing_structure"
            ],
            "tcr_complex_tm_score_threshold": tcr_complex_tm_score_threshold,
            "min_alignment_coverage_ratio": min_alignment_coverage_ratio,
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
                    item.num_tcr_pmhc_interfaces,
                    item.num_peptide_chains,
                    item.complex_observation_id,
                ),
            )
            cluster_id = _tcr_complex_cluster_id(signature_cluster_id)
            representative_rows.append(
                {
                    "tcr_complex_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "representative_complex_observation_id": representative.complex_observation_id,
                    "num_members": len(members),
                    "pdb_id": representative.pdb_id,
                    "assembly_id": representative.assembly_id or "",
                    "complex_id": representative.complex_id,
                    "tcr_type": representative.tcr_type,
                    "mhc_class": representative.mhc_class,
                    "contact_score": representative.contact_score,
                    "signature_key": representative.signature_key,
                }
            )
            signature_rows.append(
                {
                    "tcr_complex_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "signature_key": representative.signature_key,
                    "tcr_type": representative.tcr_type,
                    "mhc_class": representative.mhc_class,
                    "tcr_member_descriptors": json.dumps(
                        representative.tcr_member_descriptors,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "mhc_member_descriptors": json.dumps(
                        representative.mhc_member_descriptors,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "peptide_member_descriptors": json.dumps(
                        representative.peptide_member_descriptors,
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
                        "tcr_complex_cluster_id": cluster_id,
                        "signature_cluster_id": signature_cluster_id,
                        "complex_observation_id": member.complex_observation_id,
                        "pdb_id": member.pdb_id,
                        "assembly_id": member.assembly_id or "",
                        "complex_id": member.complex_id,
                        "tcr_type": member.tcr_type,
                        "mhc_class": member.mhc_class,
                        "tcr_chain_ids": json.dumps(member.tcr_chain_ids, ensure_ascii=False),
                        "mhc_chain_ids": json.dumps(member.mhc_chain_ids, ensure_ascii=False),
                        "mhc_chain_roles": json.dumps(member.mhc_chain_roles, ensure_ascii=False),
                        "peptide_chain_ids": json.dumps(member.peptide_chain_ids, ensure_ascii=False),
                        "auxiliary_chain_ids": json.dumps(member.auxiliary_chain_ids, ensure_ascii=False),
                        "structural_auxiliary_chain_ids": json.dumps(
                            member.structural_auxiliary_chain_ids,
                            ensure_ascii=False,
                        ),
                        "num_tcr_chains": member.num_tcr_chains,
                        "num_peptide_chains": member.num_peptide_chains,
                        "num_tcr_pmhc_interfaces": member.num_tcr_pmhc_interfaces,
                        "contact_score": member.contact_score,
                        "num_unclustered_monomer_members": member.num_unclustered_monomer_members,
                        "representative_complex_observation_id": representative.complex_observation_id,
                        "tm_score_to_representative": "",
                    }
                )
        manifest = {
            "num_tcr_complex_observations": len(observations),
            "num_tcr_complex_clusters": len(grouped),
            "num_signature_clusters": len(grouped),
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1 for observation in observations if observation.num_unclustered_monomer_members > 0
            ),
            "num_alignment_runs": 0,
            "num_alignment_failures": 0,
            "num_signature_clusters_split": 0,
            "num_tcr_complex_observations_skipped_missing_structure": 0,
            "tcr_complex_tm_score_threshold": tcr_complex_tm_score_threshold,
            "min_alignment_coverage_ratio": min_alignment_coverage_ratio,
            "structure_refinement_mode": structure_refinement_mode,
            **extraction_manifest,
        }

    dump_csv_rows(outdir / "tcr_complex_cluster_membership.csv", membership_rows)
    dump_csv_rows(outdir / "tcr_complex_cluster_representatives.csv", representative_rows)
    dump_csv_rows(outdir / "tcr_complex_cluster_signatures.csv", signature_rows)
    dump_jsonl(outdir / "tcr_complex_pairwise_alignments.jsonl", alignment_rows)
    dump_jsonl(outdir / "tcr_complex_cluster_warnings.jsonl", warning_rows)
    dump_json(outdir / "tcr_complex_cluster_manifest.json", manifest, indent=2)
    LOGGER.info(
        "TCR complex clustering finished: %d observations, %d clusters, %d extraction failures, %d skipped missing structure, %d warnings",
        manifest["num_tcr_complex_observations"],
        manifest["num_tcr_complex_clusters"],
        manifest["num_failed_tcr_complex_structure_extractions"],
        manifest["num_tcr_complex_observations_skipped_missing_structure"],
        len(warning_rows),
    )
    return {
        "manifest": manifest,
        "observations": observations,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "signature_rows": signature_rows,
        "alignment_rows": alignment_rows,
    }
