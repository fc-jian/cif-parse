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

import biotite.structure as struc
from biotite.structure import AtomArray, get_residues
from biotite.structure.io.pdb import PDBFile

from cif_parse.clustering.common import (
    canonical_monomer_id,
    load_monomer_cluster_assignments,
    resolve_monomer_cluster,
)
from cif_parse.clustering.high_order_refinement import refine_signature_groups_three_phase
from cif_parse.clustering.protein_structures import (
    USalignAlignmentResult,
    parse_usalign_output,
)
from cif_parse.clustering.parallel import normalize_worker_count
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.settings import resolve_source_path
from cif_parse.utils.atom_filters import atom_array_filter_counts, filter_atom_array_for_analysis


LOGGER = logging.getLogger(__name__)

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _signature_index(signature_cluster_id: str) -> str:
    return signature_cluster_id.rsplit("_", 1)[-1]


def _multimer_cluster_id(signature_cluster_id: str, local_index: int = 1) -> str:
    return f"mul_{_signature_index(signature_cluster_id)}_{local_index}"


def _signature_cluster_id(index: int) -> str:
    return f"mulsig_{index}"


def _member_instances_from_payload(multimer: dict[str, Any]) -> list[dict[str, Any]]:
    raw_instances = multimer.get("member_instances", [])
    if not isinstance(raw_instances, list):
        return []
    return [instance for instance in raw_instances if isinstance(instance, dict)]


@dataclass(slots=True)
class MultimerObservation:
    multimer_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    assembly_mode: str
    multimer_id: str
    multimer_type: str
    contains_antibody_unit: bool
    contains_tcr_pmhc_unit: bool
    num_component_copies: int
    num_members: int
    num_member_instances: int
    num_internal_edges: int
    support_score: float
    member_chain_ids: list[str]
    member_auth_asym_ids: list[str]
    member_entity_ids: list[str]
    member_chain_types: list[str]
    member_copy_numbers: list[int]
    member_instances: list[dict[str, Any]]
    member_monomer_ids: list[str]
    member_sequence_cluster_ids: list[str | None]
    member_structure_cluster_ids: list[str | None]
    member_cluster_sources: list[str]
    signature_key: str
    signature_members: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "multimer_observation_id": self.multimer_observation_id,
            "pdb_id": self.pdb_id,
            "source_path": self.source_path,
            "assembly_id": self.assembly_id or "",
            "assembly_mode": self.assembly_mode,
            "multimer_id": self.multimer_id,
            "multimer_type": self.multimer_type,
            "contains_antibody_unit": self.contains_antibody_unit,
            "contains_tcr_pmhc_unit": self.contains_tcr_pmhc_unit,
            "num_component_copies": self.num_component_copies,
            "num_members": self.num_members,
            "num_member_instances": self.num_member_instances,
            "num_internal_edges": self.num_internal_edges,
            "support_score": self.support_score,
            "member_chain_ids": json.dumps(self.member_chain_ids, ensure_ascii=False),
            "member_auth_asym_ids": json.dumps(self.member_auth_asym_ids, ensure_ascii=False),
            "member_entity_ids": json.dumps(self.member_entity_ids, ensure_ascii=False),
            "member_chain_types": json.dumps(self.member_chain_types, ensure_ascii=False),
            "member_copy_numbers": json.dumps(self.member_copy_numbers, ensure_ascii=False),
            "member_instances": json.dumps(self.member_instances, ensure_ascii=False),
            "member_monomer_ids": json.dumps(self.member_monomer_ids, ensure_ascii=False),
            "member_sequence_cluster_ids": json.dumps(
                self.member_sequence_cluster_ids,
                ensure_ascii=False,
            ),
            "member_structure_cluster_ids": json.dumps(
                self.member_structure_cluster_ids,
                ensure_ascii=False,
            ),
            "member_cluster_sources": json.dumps(self.member_cluster_sources, ensure_ascii=False),
            "signature_key": self.signature_key,
            "signature_members": json.dumps(self.signature_members, ensure_ascii=False, sort_keys=True),
        }

    def structural_sort_key(self) -> tuple[float, float, float, str]:
        return (
            -float(self.support_score),
            -float(self.num_internal_edges),
            -float(self.num_member_instances),
            self.multimer_observation_id,
        )


@dataclass(slots=True)
class ExtractedMultimerStructure:
    multimer_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    multimer_id: str
    extracted_pdb_path: str
    residue_count: int
    atom_count: int
    filter_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_multimer_observations(
    case_dirs: Iterable[str | Path],
    monomer_cluster_assignments: dict[str, dict[str, str]],
    cif_files_directory: str | None = None,
    prep_db_path: str | Path | None = None,
    prep_dir: str | Path | None = None,
) -> list[MultimerObservation]:
    observations: list[MultimerObservation] = []

    # Fast path: read pre-parsed Parquet
    from cif_parse.clustering.prep import open_prep_parquet, iter_parquet_rows
    pf = open_prep_parquet(prep_dir, "multimers", required=True) if prep_dir else None
    if pf is not None:
        for row in tqdm(iter_parquet_rows(prep_dir, "multimers", required=True),
                        desc="Collecting multimers", unit="multimer"):
            pdb_id = row.get("pdb_id", "")
            source_path = resolve_source_path(row.get("source_path", ""), cif_files_directory)
            if not pdb_id:
                continue
            m_cids = json.loads(row.get("member_chain_ids", "[]"))
            m_ctypes = json.loads(row.get("member_chain_types", "[]"))
            m_cnums = [int(c or 0) for c in json.loads(row.get("member_copy_numbers", "[]"))]
            descriptors = []
            m_mids, m_scids, m_stcids, m_csrcs = [], [], [], []
            mul_asm_id = str(row.get("assembly_id", "") or "")
            for cid, ctype, cnum in zip(m_cids, m_ctypes, m_cnums):
                mid = canonical_monomer_id(pdb_id, cid, mul_asm_id)
                cs, cid_cluster, scid = resolve_monomer_cluster(mid, monomer_cluster_assignments)
                m_mids.append(mid); m_csrcs.append(cs)
                m_scids.append(scid)
                m_stcids.append(cid_cluster if cs == "structure" else None)
                descriptors.append({"chain_type": ctype, "monomer_cluster_id": cid_cluster, "copy_number": cnum})
            sig_members = sorted(descriptors, key=lambda x: (x["chain_type"], x["monomer_cluster_id"], int(x["copy_number"])))
            sig_key = json.dumps({"members": sig_members, "multimer_type": row.get("multimer_type", ""),
                                  "contains_antibody_unit": row.get("contains_antibody_unit", False),
                                  "contains_tcr_pmhc_unit": row.get("contains_tcr_pmhc_unit", False)},
                                 ensure_ascii=False, sort_keys=True)
            observations.append(MultimerObservation(
                multimer_observation_id=f"{pdb_id}:{row.get('assembly_id') or 'na'}:{row.get('multimer_index', 0)}",
                pdb_id=pdb_id, source_path=source_path,
                assembly_id=row.get("assembly_id"),
                assembly_mode=row.get("assembly_mode", ""),
                multimer_id=row.get("multimer_id", ""),
                member_chain_ids=m_cids,
                member_auth_asym_ids=json.loads(row.get("member_auth_asym_ids", "[]")),
                member_entity_ids=json.loads(row.get("member_entity_ids", "[]")),
                member_chain_types=m_ctypes,
                member_copy_numbers=m_cnums,
                member_instances=json.loads(row.get("member_instances", "[]")),
                num_component_copies=row.get("num_component_copies", 0),
                num_members=row.get("num_members", 0),
                num_member_instances=row.get("num_member_instances", 0),
                num_internal_edges=row.get("num_internal_edges", 0),
                multimer_type=row.get("multimer_type", ""),
                support_score=row.get("support_score", 0.0),
                contains_antibody_unit=row.get("contains_antibody_unit", False),
                contains_tcr_pmhc_unit=row.get("contains_tcr_pmhc_unit", False),
                member_monomer_ids=m_mids,
                member_sequence_cluster_ids=m_scids,
                member_structure_cluster_ids=m_stcids,
                member_cluster_sources=m_csrcs,
                signature_key=sig_key,
                signature_members=sig_members,
            ))
        LOGGER.info("Collected %d multimer observations from prep Parquet", len(observations))
        return observations

    from cif_parse.clustering.prep import load_bundles_for_collect, load_case_bundles

    sorted_dirs = sorted(Path(path).resolve() for path in case_dirs)
    prep_bundles = load_bundles_for_collect(sorted_dirs, prep_db_path=prep_db_path)
    for case_dir in tqdm(sorted_dirs, desc="Collecting multimers", unit="case"):
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
            multimers = payload.get("tight_multimers", [])
            if not isinstance(multimers, list):
                continue
            for index, multimer in enumerate(multimers, start=1):
                if not isinstance(multimer, dict):
                    continue
                member_chain_ids = [str(item) for item in multimer.get("member_chain_ids", [])]
                member_auth_asym_ids = [
                    str(item) if item is not None else ""
                    for item in multimer.get("member_auth_asym_ids", [])
                ]
                member_entity_ids = [str(item) for item in multimer.get("member_entity_ids", [])]
                member_chain_types = [str(item) for item in multimer.get("member_chain_types", [])]
                raw_copy_numbers = multimer.get("member_copy_numbers", []) or []
                member_copy_numbers = [int(item or 0) for item in raw_copy_numbers]
                if not member_chain_ids:
                    continue

                descriptors: list[dict[str, Any]] = []
                member_monomer_ids: list[str] = []
                member_sequence_cluster_ids: list[str | None] = []
                member_structure_cluster_ids: list[str | None] = []
                member_cluster_sources: list[str] = []
                for chain_id, chain_type, copy_number in zip(
                    member_chain_ids,
                    member_chain_types,
                    member_copy_numbers,
                    strict=False,
                ):
                    slow_mul_asm = str(multimer.get("assembly_id") or summary.get("assembly_id") or "")
                    monomer_id = canonical_monomer_id(pdb_id, chain_id, slow_mul_asm)
                    member_monomer_ids.append(monomer_id)
                    cluster_source, cluster_id, sequence_cluster_id = resolve_monomer_cluster(
                        monomer_id,
                        monomer_cluster_assignments,
                    )
                    member_cluster_sources.append(cluster_source)
                    member_sequence_cluster_ids.append(sequence_cluster_id)
                    member_structure_cluster_ids.append(
                        cluster_id if cluster_source == "structure" else None
                    )
                    descriptors.append(
                        {
                            "chain_type": chain_type,
                            "monomer_cluster_id": cluster_id,
                            "copy_number": copy_number,
                        }
                    )
                signature_members = sorted(
                    descriptors,
                    key=lambda item: (
                        item["chain_type"],
                        item["monomer_cluster_id"],
                        int(item["copy_number"]),
                    ),
                )
                signature_key = json.dumps(
                    {
                        "members": signature_members,
                        "multimer_type": str(multimer.get("multimer_type", "") or ""),
                        "contains_antibody_unit": bool(
                            multimer.get("contains_antibody_unit", False)
                        ),
                        "contains_tcr_pmhc_unit": bool(
                            multimer.get("contains_tcr_pmhc_unit", False)
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                assembly_id = multimer.get("assembly_id")
                observations.append(
                    MultimerObservation(
                        multimer_observation_id=(
                            f"{pdb_id}:{assembly_id or default_assembly_id or 'na'}:{index}"
                        ),
                        pdb_id=pdb_id,
                        source_path=source_path,
                        assembly_id=str(assembly_id) if assembly_id is not None else default_assembly_id,
                        assembly_mode=str(multimer.get("assembly_mode", "") or ""),
                        multimer_id=str(multimer.get("multimer_id", "") or ""),
                        multimer_type=str(multimer.get("multimer_type", "") or ""),
                        contains_antibody_unit=bool(
                            multimer.get("contains_antibody_unit", False)
                        ),
                        contains_tcr_pmhc_unit=bool(
                            multimer.get("contains_tcr_pmhc_unit", False)
                        ),
                        num_component_copies=int(multimer.get("num_component_copies", 0) or 0),
                        num_members=int(multimer.get("num_members", 0) or 0),
                        num_member_instances=int(multimer.get("num_member_instances", 0) or 0),
                        num_internal_edges=int(multimer.get("num_internal_edges", 0) or 0),
                        support_score=float(multimer.get("support_score", 0.0) or 0.0),
                        member_chain_ids=member_chain_ids,
                        member_auth_asym_ids=member_auth_asym_ids,
                        member_entity_ids=member_entity_ids,
                        member_chain_types=member_chain_types,
                        member_copy_numbers=member_copy_numbers,
                        member_instances=_member_instances_from_payload(multimer),
                        member_monomer_ids=member_monomer_ids,
                        member_sequence_cluster_ids=member_sequence_cluster_ids,
                        member_structure_cluster_ids=member_structure_cluster_ids,
                        member_cluster_sources=member_cluster_sources,
                        signature_key=signature_key,
                        signature_members=signature_members,
                    )
                )
    LOGGER.info("Collected %d multimer observations from %d case dir(s)", len(observations), len(list(case_dirs)))
    return observations


def _pdb_chain_id(index: int) -> str:
    if index >= len(PDB_CHAIN_IDS):
        raise ValueError(f"Too many multimer member instances for PDB export: {index + 1}")
    return PDB_CHAIN_IDS[index]


def _select_instance_atoms(
    atom_array: AtomArray,
    *,
    label_asym_id: str,
    sym_id: int | None = None,
) -> AtomArray:
    mask = atom_array.chain_id == label_asym_id
    if hasattr(atom_array, "hetero"):
        mask &= ~atom_array.hetero
    if sym_id is not None and hasattr(atom_array, "sym_id"):
        mask &= atom_array.sym_id == sym_id
    return atom_array[mask]


def _coerce_multimer_chain_id(atom_array: AtomArray, chain_id: str) -> AtomArray:
    copied = atom_array.copy()
    copied.chain_id[:] = chain_id
    return copied


def extract_multimer_structure(
    observation: MultimerObservation,
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    atom_array: AtomArray | None = None,
    assembly_chain_operations: dict[str, list[str]] | None = None,
) -> ExtractedMultimerStructure:
    if not observation.source_path:
        raise ValueError(f"Missing source_path for multimer {observation.multimer_observation_id}")

    if atom_array is None:
        raise ValueError(f"Cached coordinates are required for multimer {observation.multimer_observation_id}")

    chain_operations = assembly_chain_operations or {}
    instance_atom_arrays: list[AtomArray] = []
    selection_count = 0
    for instance_index, instance in enumerate(observation.member_instances):
        label_asym_id = str(instance.get("label_asym_id", "") or "")
        if not label_asym_id:
            continue
        sym_id: int | None = None
        operation_id = str(instance.get("operation_id", "") or "")
        if observation.assembly_id and label_asym_id in chain_operations and operation_id:
            operation_ids = chain_operations.get(label_asym_id, [])
            if operation_id in operation_ids:
                sym_id = operation_ids.index(operation_id)
        selected = _select_instance_atoms(atom_array, label_asym_id=label_asym_id, sym_id=sym_id)
        if selected.array_length() == 0 and sym_id is not None:
            selected = _select_instance_atoms(atom_array, label_asym_id=label_asym_id, sym_id=None)
        if selected.array_length() == 0:
            continue
        selection_count += 1
        instance_atom_arrays.append(_coerce_multimer_chain_id(selected, _pdb_chain_id(instance_index)))

    if not instance_atom_arrays:
        for chain_index, label_asym_id in enumerate(observation.member_chain_ids):
            selected = _select_instance_atoms(atom_array, label_asym_id=label_asym_id, sym_id=None)
            if selected.array_length() == 0:
                continue
            selection_count += 1
            instance_atom_arrays.append(_coerce_multimer_chain_id(selected, _pdb_chain_id(chain_index)))

    if not instance_atom_arrays:
        raise ValueError(f"No polymer atoms found for multimer {observation.multimer_observation_id}")

    multimer_atoms = struc.concatenate(instance_atom_arrays)
    multimer_atoms, filter_counts = filter_atom_array_for_analysis(
        multimer_atoms,
        drop_hydrogens=drop_hydrogens,
        drop_nonfinite=True,
    )
    if multimer_atoms.array_length() == 0:
        raise ValueError(
            f"No analyzable atoms left for multimer {observation.multimer_observation_id}"
        )

    _, residue_names = get_residues(multimer_atoms)
    residue_count = int(len(residue_names))
    if residue_count <= 2:
        raise ValueError(
            f"Resolved residue count {residue_count} is too short for multimer USalign: "
            f"{observation.multimer_observation_id}"
        )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdb_path = outdir / (
        f"{observation.pdb_id}_{observation.assembly_id or 'asu'}_{observation.multimer_id}.pdb"
    )
    pdb_file = PDBFile()
    pdb_file.set_structure(multimer_atoms)
    pdb_file.write(pdb_path)
    return ExtractedMultimerStructure(
        multimer_observation_id=observation.multimer_observation_id,
        pdb_id=observation.pdb_id,
        source_path=observation.source_path,
        assembly_id=observation.assembly_id,
        multimer_id=observation.multimer_id,
        extracted_pdb_path=str(pdb_path),
        residue_count=residue_count,
        atom_count=int(multimer_atoms.array_length()),
        filter_counts=atom_array_filter_counts(filter_counts),
    )


def extract_multimer_structures(
    observations: Iterable[MultimerObservation],
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    extraction_jobs: int = 1,
    prep_dir: str | Path | None = None,
    show_progress: bool = True,
    log_summary: bool = True,
) -> tuple[dict[str, ExtractedMultimerStructure], dict[str, Any]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    structures: dict[str, ExtractedMultimerStructure] = {}
    failures: list[dict[str, str]] = []

    sorted_observations = sorted(observations, key=lambda item: item.multimer_observation_id)
    extraction_jobs = normalize_worker_count(extraction_jobs)

    _pkl_reader = None
    if prep_dir:
        from cif_parse.clustering.atom_cache import resolve_cases_root, PklAtomReader
        try:
            _pkl_reader = PklAtomReader(resolve_cases_root(prep_dir))
        except Exception:
            pass

    def _try_assemble_multimer(obs: MultimerObservation) -> AtomArray | None:
        if _pkl_reader is None:
            return None
        chain_specs = []
        for inst in obs.member_instances:
            lbl = str(inst.get("label_asym_id", "") or "")
            if not lbl:
                continue
            sym = inst.get("sym_id")
            if sym is None:
                instance_id = str(inst.get("instance_id", "") or "")
                if "@" in instance_id:
                    try:
                        sym = int(instance_id.rsplit("@", 1)[1]) - 1
                    except ValueError:
                        sym = None
            try:
                sym_id = int(sym) if sym is not None and str(sym) != "" else None
            except (TypeError, ValueError):
                sym_id = None
            chain_specs.append((lbl, sym_id))
        if not chain_specs:
            return None
        return _pkl_reader.load_chains(
            obs.source_path, chain_specs,
            assembly_id=obs.assembly_id,
        )

    def _load_chain_ops_only(observation: MultimerObservation) -> dict[str, list[str]]:
        return {}

    def _process_one(observation: MultimerObservation) -> ExtractedMultimerStructure | None:
        atom_array = _try_assemble_multimer(observation)
        if atom_array is not None:
            chain_ops = _load_chain_ops_only(observation)
        else:
            raise ValueError(
                f"Prep coordinates missing for multimer {observation.multimer_observation_id}"
            )
        return extract_multimer_structure(
            observation,
            outdir=outdir,
            model=model,
            drop_hydrogens=drop_hydrogens,
            atom_array=atom_array,
            assembly_chain_operations=chain_ops,
        )

    if extraction_jobs <= 1 or len(sorted_observations) <= 1:
        observation_iter = (
            tqdm(sorted_observations, desc="Extracting multimer structures", unit="multimer")
            if show_progress
            else sorted_observations
        )
        for observation in observation_iter:
            try:
                structures[observation.multimer_observation_id] = _process_one(observation)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to extract multimer %s: %s",
                    observation.multimer_observation_id,
                    exc,
                )
                failures.append(
                    {"multimer_observation_id": observation.multimer_observation_id, "error": str(exc)}
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
                    desc="Extracting multimer structures",
                    unit="multimer",
                )
            for future in future_iter:
                observation = future_to_obs[future]
                try:
                    extracted = future.result()
                    if extracted is not None:
                        structures[observation.multimer_observation_id] = extracted
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to extract multimer %s: %s",
                        observation.multimer_observation_id,
                        exc,
                    )
                    failures.append(
                        {"multimer_observation_id": observation.multimer_observation_id, "error": str(exc)}
                    )

    dump_jsonl(outdir / "multimer_structure_extraction_failures.jsonl", failures)
    dump_jsonl(outdir / "multimer_structures.jsonl", [item.to_dict() for item in structures.values()])
    manifest = {
        "num_multimer_observations": len(sorted_observations),
        "num_extracted_multimer_structures": len(structures),
        "num_failed_multimer_structure_extractions": len(failures),
        "extraction_jobs": extraction_jobs,
    }
    dump_json(outdir / "multimer_structure_manifest.json", manifest, indent=2)
    if log_summary:
        LOGGER.info("Extracted %d multimer structures (%d failures)", len(structures), len(failures))
    return structures, manifest


def run_multimer_usalign_alignment(
    query: ExtractedMultimerStructure,
    target: ExtractedMultimerStructure,
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
        query_monomer_id=query.multimer_observation_id,
        target_monomer_id=target.multimer_observation_id,
        query_length=query.residue_count,
        target_length=target.residue_count,
        tm_score_threshold=tm_score_threshold,
        min_alignment_coverage_ratio=0.0,
    )


def refine_multimer_signature_clusters(
    signature_groups: list[tuple[str, list[MultimerObservation]]],
    extracted_structures: dict[str, ExtractedMultimerStructure],
    *,
    tm_score_threshold: float = 0.50,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
    show_progress: bool = True,
    log_summary: bool = True,
    max_atoms_for_refinement: int | None = 10_000,
) -> dict[str, Any]:
    runner = alignment_runner or run_multimer_usalign_alignment
    alignment_jobs = normalize_worker_count(alignment_jobs)

    # Split signature groups: skip USalign refinement for groups where any
    # member is too large (e.g. viral capsids with 10k+ atoms).
    small_groups: list[tuple[str, list[MultimerObservation]]] = []
    large_groups: list[tuple[str, list[MultimerObservation]]] = []
    if max_atoms_for_refinement is not None:
        for sig_id, members in signature_groups:
            if any(
                (s := extracted_structures.get(m.multimer_observation_id)) is not None
                and s.atom_count > max_atoms_for_refinement
                for m in members
            ):
                large_groups.append((sig_id, members))
            else:
                small_groups.append((sig_id, members))
    else:
        small_groups = list(signature_groups)

    total_observations = sum(len(members) for _, members in signature_groups)
    multi_member = sum(1 for _, m in signature_groups if len(m) > 1)
    if log_summary:
        parts = [f"Refining {len(small_groups)} multimer signature clusters"]
        if large_groups:
            parts.append(f"{len(large_groups)} skipped (too large, kept as single clusters)")
        parts.append(f"({total_observations} observations, {multi_member} multi-member, {alignment_jobs} alignment workers)")
        LOGGER.info(", ".join(parts))

    # Pre-populate cluster_members for large groups (no USalign, one cluster each).
    large_cluster_members: list[tuple[str, str, list[Any], Any]] = []
    large_warning_rows: list[dict[str, Any]] = []
    for sig_id, members in large_groups:
        extracted = [m for m in members if m.multimer_observation_id in extracted_structures]
        representative = max(extracted, key=lambda m: (extracted_structures[m.multimer_observation_id].atom_count), default=extracted[0] if extracted else members[0])
        large_cluster_members.append((sig_id, representative.multimer_observation_id, sorted(members, key=lambda m: m.multimer_observation_id), representative))
        for m in members:
            if m.multimer_observation_id not in extracted_structures:
                large_warning_rows.append({
                    "warning_code": "multimer_structure_unavailable_single_cluster",
                    "signature_cluster_id": sig_id,
                    "multimer_observation_id": m.multimer_observation_id,
                })

    if not small_groups:
        manifest = {
            "num_signature_clusters": len(signature_groups),
            "num_multimer_clusters": len(large_cluster_members),
            "num_alignment_runs": 0,
            "num_alignment_failures": 0,
            "num_signature_clusters_split": 0,
            "multimer_tm_score_threshold": tm_score_threshold,
            "alignment_jobs": alignment_jobs,
        }
        if log_summary:
            LOGGER.info(
                "Multimer refinement: %d signature clusters -> %d refined clusters (0 alignments, 0 failures, 0 splits — all skipped: too large)",
                len(signature_groups),
                len(large_cluster_members),
            )
        return {
            "manifest": manifest,
            "alignment_cache": {},
            "alignment_rows": [],
            "warning_rows": large_warning_rows,
            "cluster_members": large_cluster_members,
            "num_alignment_runs": 0,
            "num_alignment_failures": 0,
            "num_signature_clusters_split": 0,
            "membership_rows": [],
            "representative_rows": [],
            "signature_rows": [],
            "grouped_signature_sizes": {sig_id: len(members) for sig_id, members in large_groups},
        }

    signature_iter = list(
        tqdm(small_groups, desc="Refining multimer clusters", unit="sig-group")
        if show_progress
        else small_groups
    )
    refined = refine_signature_groups_three_phase(
        signature_iter,
        extracted_structures,
        member_id=lambda item: item.multimer_observation_id,
        structural_sort_key=lambda item: item.structural_sort_key(),
        alignment_row=lambda signature_cluster_id, result: {
            "signature_cluster_id": signature_cluster_id,
            "query_multimer_observation_id": result.query_monomer_id,
            "target_multimer_observation_id": result.target_monomer_id,
            "aligned_length": result.aligned_length,
            "rmsd": result.rmsd,
            "tm_score_query": result.tm_score_query,
            "tm_score_target": result.tm_score_target,
            "tm_score_min": result.min_tm_score,
            "tm_score_max": result.max_tm_score,
            "tm_score_for_clustering": result.max_tm_score,
        },
        alignment_failure_warning=lambda signature_cluster_id, representative, candidate, exc: {
            "warning_code": "multimer_usalign_failed",
            "signature_cluster_id": signature_cluster_id,
            "representative_multimer_observation_id": representative.multimer_observation_id,
            "candidate_multimer_observation_id": candidate.multimer_observation_id,
            "error": str(exc),
        },
        unavailable_warning=lambda signature_cluster_id, member: {
            "warning_code": "multimer_structure_unavailable_singleton_cluster",
            "signature_cluster_id": signature_cluster_id,
            "multimer_observation_id": member.multimer_observation_id,
        },
        runner=runner,
        alignment_jobs=alignment_jobs,
        usalign_executable=usalign_executable,
        tm_score_threshold=tm_score_threshold,
        can_skip_alignment=lambda a, b: (
            a.source_path == b.source_path
            and a.assembly_id == b.assembly_id
            and sorted(a.member_chain_ids) == sorted(b.member_chain_ids)
        ),
    )
    alignment_cache = refined.alignment_cache
    alignment_rows = refined.alignment_rows
    warning_rows = refined.warning_rows + large_warning_rows
    cluster_members = refined.cluster_members + large_cluster_members
    num_alignment_runs = refined.num_alignment_runs
    num_alignment_failures = refined.num_alignment_failures
    num_signature_clusters_split = refined.num_signature_clusters_split

    membership_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    grouped_signature_sizes = {signature_cluster_id: 0 for signature_cluster_id, _ in signature_groups}
    for signature_cluster_id, _, _, _ in cluster_members:
        grouped_signature_sizes[signature_cluster_id] = grouped_signature_sizes.get(signature_cluster_id, 0) + 1

    local_cluster_counts: dict[str, int] = {}
    for signature_cluster_id, representative_id, members, representative in cluster_members:
        local_cluster_counts[signature_cluster_id] = local_cluster_counts.get(signature_cluster_id, 0) + 1
        cluster_id = _multimer_cluster_id(signature_cluster_id, local_cluster_counts[signature_cluster_id])
        representative_rows.append(
            {
                "multimer_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "representative_multimer_observation_id": representative_id,
                "num_members": len(members),
                "pdb_id": representative.pdb_id,
                "assembly_id": representative.assembly_id or "",
                "multimer_type": representative.multimer_type,
                "support_score": representative.support_score,
                "num_internal_edges": representative.num_internal_edges,
                "signature_key": representative.signature_key,
            }
        )
        signature_rows.append(
            {
                "multimer_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "signature_key": representative.signature_key,
                "signature_members": json.dumps(
                    representative.signature_members,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "multimer_type": representative.multimer_type,
                "contains_antibody_unit": representative.contains_antibody_unit,
                "contains_tcr_pmhc_unit": representative.contains_tcr_pmhc_unit,
                "num_refined_clusters_in_signature_group": grouped_signature_sizes.get(
                    signature_cluster_id,
                    1,
                ),
            }
        )
        for member in members:
            tm_score_for_clustering: float | str = ""
            if member.multimer_observation_id != representative_id:
                pair_key = tuple(sorted((representative_id, member.multimer_observation_id)))
                if pair_key in alignment_cache:
                    tm_score_for_clustering = alignment_cache[pair_key].max_tm_score
            membership_rows.append(
                {
                    "multimer_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "multimer_observation_id": member.multimer_observation_id,
                    "pdb_id": member.pdb_id,
                    "assembly_id": member.assembly_id or "",
                    "multimer_id": member.multimer_id,
                    "multimer_type": member.multimer_type,
                    "num_component_copies": member.num_component_copies,
                    "num_members": member.num_members,
                    "num_member_instances": member.num_member_instances,
                    "num_internal_edges": member.num_internal_edges,
                    "support_score": member.support_score,
                    "member_chain_ids": json.dumps(member.member_chain_ids, ensure_ascii=False),
                    "member_auth_asym_ids": json.dumps(
                        member.member_auth_asym_ids,
                        ensure_ascii=False,
                    ),
                    "member_chain_types": json.dumps(member.member_chain_types, ensure_ascii=False),
                    "member_monomer_ids": json.dumps(member.member_monomer_ids, ensure_ascii=False),
                    "member_sequence_cluster_ids": json.dumps(
                        member.member_sequence_cluster_ids,
                        ensure_ascii=False,
                    ),
                    "member_structure_cluster_ids": json.dumps(
                        member.member_structure_cluster_ids,
                        ensure_ascii=False,
                    ),
                    "member_cluster_sources": json.dumps(
                        member.member_cluster_sources,
                        ensure_ascii=False,
                    ),
                    "representative_multimer_observation_id": representative_id,
                    "tm_score_to_representative": tm_score_for_clustering,
                }
            )

    manifest = {
        "num_signature_clusters": len(signature_groups),
        "num_multimer_clusters": len(cluster_members),
        "num_alignment_runs": num_alignment_runs,
        "num_alignment_failures": num_alignment_failures,
        "num_signature_clusters_split": num_signature_clusters_split,
        "multimer_tm_score_threshold": tm_score_threshold,
        "alignment_jobs": alignment_jobs,
    }
    if log_summary:
        LOGGER.info(
            "Multimer refinement: %d signature clusters -> %d refined clusters (%d alignments, %d failures, %d splits)",
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


def build_multimer_signature_clusters(
    *,
    case_dirs: Iterable[str | Path],
    clustering_outdir: str | Path,
    outdir: str | Path,
    structure_refinement_mode: str = "greedy",
    multimer_tm_score_threshold: float = 0.50,
    model: int = 1,
    drop_hydrogens: bool = True,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
    cif_files_directory: str | None = None,
    prep_dir: str | Path | None = None,
    include_structure_assignments: bool = True,
    max_atoms_for_refinement: int | None = 10_000,
) -> dict[str, Any]:
    monomer_assignments = load_monomer_cluster_assignments(
        clustering_outdir,
        include_structure=include_structure_assignments,
    )
    observations = collect_multimer_observations(
        case_dirs, monomer_assignments,
        cif_files_directory=cif_files_directory,
        prep_dir=prep_dir,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dump_jsonl(outdir / "multimer_inventory.jsonl", [item.to_dict() for item in observations])
    dump_csv_rows(outdir / "multimer_inventory.csv", [item.to_record() for item in observations])

    grouped: dict[str, list[MultimerObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.signature_key, []).append(observation)
    signature_groups = [
        (_signature_cluster_id(index), members)
        for index, (_, members) in enumerate(
            sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
            start=1,
        )
    ]

    extraction_manifest = {
        "num_extracted_multimer_structures": 0,
        "num_failed_multimer_structure_extractions": 0,
        "num_multimer_structure_extraction_candidates": 0,
        "num_singleton_multimer_observations_skipped_structure_extraction": 0,
    }
    if structure_refinement_mode == "greedy":
        refinement_observations = [
            observation
            for _, members in signature_groups
            if len(members) > 1
            for observation in members
        ]
        extraction_manifest["num_multimer_structure_extraction_candidates"] = len(refinement_observations)
        extraction_manifest["num_singleton_multimer_observations_skipped_structure_extraction"] = (
            len(observations) - len(refinement_observations)
        )
        LOGGER.info(
            "Multimer structure refinement will extract %d/%d observations; %d singleton observations need no USalign",
            len(refinement_observations),
            len(observations),
            extraction_manifest["num_singleton_multimer_observations_skipped_structure_extraction"],
        )

    if structure_refinement_mode == "greedy":
        extracted_structures: dict[str, ExtractedMultimerStructure] = {}
        structure_rows: list[dict[str, Any]] = []
        structure_failure_rows: list[dict[str, Any]] = []

        def _record_extraction_result(
            structures: dict[str, ExtractedMultimerStructure],
            extract_manifest: dict[str, Any],
            group_outdir: Path,
        ) -> None:
            extraction_manifest["num_extracted_multimer_structures"] += extract_manifest.get(
                "num_extracted_multimer_structures",
                0,
            )
            extraction_manifest["num_failed_multimer_structure_extractions"] += extract_manifest.get(
                "num_failed_multimer_structure_extractions",
                0,
            )
            structure_rows.extend(item.to_dict() for item in structures.values())
            failures_path = group_outdir / "multimer_structure_extraction_failures.jsonl"
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
                        extract_multimer_structures,
                        members,
                        outdir=group_outdir,
                        model=model,
                        drop_hydrogens=drop_hydrogens,
                        extraction_jobs=1,
                        prep_dir=prep_dir,
                        show_progress=False,
                        log_summary=False,
                    )
                    future_to_group[future] = (signature_cluster_id, members, group_outdir)
                for future in as_completed(future_to_group):
                    signature_cluster_id, members, group_outdir = future_to_group[future]
                    structures, extract_manifest = future.result()
                    extracted_structures.update(structures)
                    _record_extraction_result(structures, extract_manifest, group_outdir)

        refined = refine_multimer_signature_clusters(
            signature_groups,
            extracted_structures,
            tm_score_threshold=multimer_tm_score_threshold,
            usalign_executable=usalign_executable,
            alignment_runner=alignment_runner,
            alignment_jobs=alignment_jobs,
            show_progress=True,
            log_summary=True,
            max_atoms_for_refinement=max_atoms_for_refinement,
        )
        membership_rows = refined["membership_rows"]
        representative_rows = refined["representative_rows"]
        signature_rows = refined["signature_rows"]
        alignment_rows = refined["alignment_rows"]
        warning_rows = refined["warning_rows"]
        refined_manifest = refined["manifest"]

        def _multimer_cluster_sort_key(cluster_id: str) -> tuple[int, int]:
            _, sig_idx, local_idx = cluster_id.split("_")
            return (int(sig_idx), int(local_idx))

        membership_rows.sort(
            key=lambda row: (_multimer_cluster_sort_key(row["multimer_cluster_id"]), row["multimer_observation_id"])
        )
        representative_rows.sort(key=lambda row: _multimer_cluster_sort_key(row["multimer_cluster_id"]))
        signature_rows.sort(key=lambda row: _multimer_cluster_sort_key(row["multimer_cluster_id"]))

        structures_dir = outdir / "structures"
        dump_jsonl(structures_dir / "multimer_structure_extraction_failures.jsonl", structure_failure_rows)
        dump_jsonl(structures_dir / "multimer_structures.jsonl", structure_rows)
        dump_json(
            structures_dir / "multimer_structure_manifest.json",
            {
                "num_multimer_observations": extraction_manifest["num_multimer_structure_extraction_candidates"],
                "num_extracted_multimer_structures": extraction_manifest["num_extracted_multimer_structures"],
                "num_failed_multimer_structure_extractions": extraction_manifest[
                    "num_failed_multimer_structure_extractions"
                ],
                "extraction_jobs": normalize_worker_count(alignment_jobs),
                "num_signature_groups_pipelined": len(multi_member_groups),
            },
            indent=2,
        )
        manifest = {
            "num_multimer_observations": len(observations),
            "num_multimer_clusters": refined_manifest["num_multimer_clusters"],
            "num_signature_clusters": refined_manifest["num_signature_clusters"],
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1
                for observation in observations
                if "unclustered" in set(observation.member_cluster_sources)
            ),
            "num_alignment_runs": refined_manifest["num_alignment_runs"],
            "num_alignment_failures": refined_manifest["num_alignment_failures"],
            "num_signature_clusters_split": refined_manifest["num_signature_clusters_split"],
            "multimer_tm_score_threshold": multimer_tm_score_threshold,
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
                    item.support_score,
                    item.num_internal_edges,
                    item.num_member_instances,
                    item.multimer_observation_id,
                ),
            )
            cluster_id = _multimer_cluster_id(signature_cluster_id)
            representative_rows.append(
                {
                    "multimer_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "representative_multimer_observation_id": representative.multimer_observation_id,
                    "num_members": len(members),
                    "pdb_id": representative.pdb_id,
                    "assembly_id": representative.assembly_id or "",
                    "multimer_type": representative.multimer_type,
                    "support_score": representative.support_score,
                    "num_internal_edges": representative.num_internal_edges,
                    "signature_key": representative.signature_key,
                }
            )
            signature_rows.append(
                {
                    "multimer_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "signature_key": representative.signature_key,
                    "signature_members": json.dumps(
                        representative.signature_members,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "multimer_type": representative.multimer_type,
                    "contains_antibody_unit": representative.contains_antibody_unit,
                    "contains_tcr_pmhc_unit": representative.contains_tcr_pmhc_unit,
                    "num_refined_clusters_in_signature_group": 1,
                }
            )
            for member in sorted(members, key=lambda item: item.multimer_observation_id):
                membership_rows.append(
                    {
                        "multimer_cluster_id": cluster_id,
                        "signature_cluster_id": signature_cluster_id,
                        "multimer_observation_id": member.multimer_observation_id,
                        "pdb_id": member.pdb_id,
                        "assembly_id": member.assembly_id or "",
                        "multimer_id": member.multimer_id,
                        "multimer_type": member.multimer_type,
                        "num_component_copies": member.num_component_copies,
                        "num_members": member.num_members,
                        "num_member_instances": member.num_member_instances,
                        "num_internal_edges": member.num_internal_edges,
                        "support_score": member.support_score,
                        "member_chain_ids": json.dumps(member.member_chain_ids, ensure_ascii=False),
                        "member_auth_asym_ids": json.dumps(
                            member.member_auth_asym_ids,
                            ensure_ascii=False,
                        ),
                        "member_chain_types": json.dumps(member.member_chain_types, ensure_ascii=False),
                        "member_monomer_ids": json.dumps(member.member_monomer_ids, ensure_ascii=False),
                        "member_sequence_cluster_ids": json.dumps(
                            member.member_sequence_cluster_ids,
                            ensure_ascii=False,
                        ),
                        "member_structure_cluster_ids": json.dumps(
                            member.member_structure_cluster_ids,
                            ensure_ascii=False,
                        ),
                        "member_cluster_sources": json.dumps(
                            member.member_cluster_sources,
                            ensure_ascii=False,
                        ),
                        "representative_multimer_observation_id": representative.multimer_observation_id,
                        "tm_score_to_representative": "",
                    }
                )
        manifest = {
            "num_multimer_observations": len(observations),
            "num_multimer_clusters": len(grouped),
            "num_signature_clusters": len(grouped),
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1
                for observation in observations
                if "unclustered" in set(observation.member_cluster_sources)
            ),
            "num_alignment_runs": 0,
            "num_alignment_failures": 0,
            "num_signature_clusters_split": 0,
            "multimer_tm_score_threshold": multimer_tm_score_threshold,
            "structure_refinement_mode": structure_refinement_mode,
            **extraction_manifest,
        }

    dump_csv_rows(outdir / "multimer_cluster_membership.csv", membership_rows)
    dump_csv_rows(outdir / "multimer_cluster_representatives.csv", representative_rows)
    dump_csv_rows(outdir / "multimer_cluster_signatures.csv", signature_rows)
    dump_jsonl(outdir / "multimer_pairwise_alignments.jsonl", alignment_rows)
    dump_jsonl(outdir / "multimer_cluster_warnings.jsonl", warning_rows)
    dump_json(outdir / "multimer_cluster_manifest.json", manifest, indent=2)
    return {
        "manifest": manifest,
        "observations": observations,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "signature_rows": signature_rows,
        "alignment_rows": alignment_rows,
    }
