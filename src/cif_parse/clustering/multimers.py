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
from biotite.structure.io.pdbx import get_assembly, get_structure

from cif_parse.clustering.common import (
    canonical_monomer_id,
    load_monomer_cluster_assignments,
    resolve_monomer_cluster,
)
from cif_parse.clustering.protein_structures import (
    USalignAlignmentResult,
    parse_usalign_output,
)
from cif_parse.clustering.parallel import AlignmentTask, normalize_worker_count, run_alignment_tasks
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.io import read_assembly_chain_operations, read_cif_file
from cif_parse.settings import resolve_source_path
from cif_parse.utils.atom_filters import atom_array_filter_counts, filter_atom_array_for_analysis


LOGGER = logging.getLogger(__name__)

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _multimer_cluster_id(index: int) -> str:
    return f"mul_{index}"


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
            for cid, ctype, cnum in zip(m_cids, m_ctypes, m_cnums):
                mid = canonical_monomer_id(pdb_id, cid)
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
                    monomer_id = canonical_monomer_id(pdb_id, chain_id)
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
        cif_file = read_cif_file(observation.source_path)
        if observation.assembly_id:
            atom_array = get_assembly(
                cif_file,
                assembly_id=observation.assembly_id,
                model=model,
                use_author_fields=False,
            )
        else:
            atom_array = get_structure(
                cif_file,
                model=model,
                use_author_fields=False,
            )

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
) -> tuple[dict[str, ExtractedMultimerStructure], dict[str, Any]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    structures: dict[str, ExtractedMultimerStructure] = {}
    failures: list[dict[str, str]] = []

    sorted_observations = sorted(observations, key=lambda item: item.multimer_observation_id)
    extraction_jobs = normalize_worker_count(extraction_jobs)

    import threading as _threading

    _prep_m_idx: dict | None = None
    if prep_dir:
        from cif_parse.clustering.prep import load_cif_coords_index, assemble_atom_array_from_chains
        _prep_m_idx = load_cif_coords_index(prep_dir)

    def _try_assemble_multimer(obs: MultimerObservation) -> AtomArray | None:
        if _prep_m_idx is None:
            return None
        chain_specs = []
        for inst in obs.member_instances:
            lbl = str(inst.get("label_asym_id", "") or "")
            if not lbl:
                continue
            chain_specs.append((lbl, None))
        if not chain_specs:
            return None
        return assemble_atom_array_from_chains(
            prep_dir, obs.source_path, chain_specs,
            assembly_id=obs.assembly_id, index=_prep_m_idx,
        )

    atom_array_cache: dict[tuple[str, str | None], AtomArray] = {}
    chain_ops_cache: dict[tuple[str, str | None], dict[str, list[str]]] = {}
    _lock = _threading.Lock()

    def _load_chain_ops_only(observation: MultimerObservation) -> dict[str, list[str]]:
        cache_key = (observation.source_path, observation.assembly_id)
        with _lock:
            if cache_key not in chain_ops_cache:
                _, chain_ops = read_assembly_chain_operations(
                    observation.source_path,
                    assembly_id=observation.assembly_id,
                )
                chain_ops_cache[cache_key] = chain_ops
        return chain_ops_cache[cache_key]

    def _load_caches(observation: MultimerObservation) -> tuple[AtomArray, dict[str, list[str]]]:
        cache_key = (observation.source_path, observation.assembly_id)
        with _lock:
            if cache_key not in atom_array_cache:
                cif_file = read_cif_file(observation.source_path)
                if observation.assembly_id:
                    atom_array_cache[cache_key] = get_assembly(
                        cif_file,
                        assembly_id=observation.assembly_id,
                        model=model,
                        use_author_fields=False,
                    )
                else:
                    atom_array_cache[cache_key] = get_structure(
                        cif_file,
                        model=model,
                        use_author_fields=False,
                    )
            if cache_key not in chain_ops_cache:
                _, chain_ops = read_assembly_chain_operations(
                    observation.source_path,
                    assembly_id=observation.assembly_id,
                )
                chain_ops_cache[cache_key] = chain_ops
        return atom_array_cache[cache_key], chain_ops_cache[cache_key]

    def _process_one(observation: MultimerObservation) -> ExtractedMultimerStructure | None:
        atom_array = _try_assemble_multimer(observation)
        if atom_array is not None:
            chain_ops = _load_chain_ops_only(observation)
        else:
            atom_array, chain_ops = _load_caches(observation)
        return extract_multimer_structure(
            observation,
            outdir=outdir,
            model=model,
            drop_hydrogens=drop_hydrogens,
            atom_array=atom_array,
            assembly_chain_operations=chain_ops,
        )

    if extraction_jobs <= 1 or len(sorted_observations) <= 1:
        for observation in tqdm(sorted_observations, desc="Extracting multimer structures", unit="multimer"):
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
            for future in tqdm(as_completed(future_to_obs), total=len(future_to_obs),
                               desc="Extracting multimer structures", unit="multimer"):
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
        "prot",
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
) -> dict[str, Any]:
    runner = alignment_runner or run_multimer_usalign_alignment
    alignment_jobs = normalize_worker_count(alignment_jobs)
    total_observations = sum(len(members) for _, members in signature_groups)
    multi_member = sum(1 for _, m in signature_groups if len(m) > 1)
    LOGGER.info(
        "Refining %d multimer signature clusters (%d observations, %d multi-member, %d alignment workers)",
        len(signature_groups),
        total_observations,
        multi_member,
        alignment_jobs,
    )
    alignment_cache: dict[tuple[str, str], USalignAlignmentResult] = {}
    alignment_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    cluster_members: list[tuple[str, str, list[MultimerObservation], MultimerObservation]] = []
    num_alignment_runs = 0
    num_alignment_failures = 0
    num_signature_clusters_split = 0

    for signature_cluster_id, members in tqdm(signature_groups, desc="Refining multimer clusters", unit="sig-group"):
        extracted_members = [
            member for member in members if member.multimer_observation_id in extracted_structures
        ]
        unresolved_members = [
            member for member in members if member.multimer_observation_id not in extracted_structures
        ]

        local_clusters: list[tuple[list[MultimerObservation], MultimerObservation]] = []
        if extracted_members:
            pending = sorted(extracted_members, key=lambda item: item.structural_sort_key())
            while pending:
                representative = pending[0]
                assigned = [representative]
                remaining: list[MultimerObservation] = []
                alignment_tasks: list[AlignmentTask] = []
                for candidate in pending[1:]:
                    pair_key = tuple(
                        sorted(
                            (
                                representative.multimer_observation_id,
                                candidate.multimer_observation_id,
                            )
                        )
                    )
                    if pair_key not in alignment_cache:
                        alignment_tasks.append(
                            AlignmentTask(
                                key=pair_key,
                                query=extracted_structures[representative.multimer_observation_id],
                                target=extracted_structures[candidate.multimer_observation_id],
                                context={"candidate": candidate},
                            )
                        )
                successes, failures = run_alignment_tasks(
                    alignment_tasks,
                    runner,
                    max_workers=alignment_jobs,
                    usalign_executable=usalign_executable,
                    tm_score_threshold=tm_score_threshold,
                )
                failure_by_candidate_id = {
                    task.context["candidate"].multimer_observation_id: exc for task, exc in failures
                }
                for task, result in successes:
                    alignment_cache[task.key] = result
                success_keys = {task.key for task, _ in successes}
                for candidate in pending[1:]:
                    pair_key = tuple(
                        sorted((representative.multimer_observation_id, candidate.multimer_observation_id))
                    )
                    if candidate.multimer_observation_id in failure_by_candidate_id:
                        exc = failure_by_candidate_id[candidate.multimer_observation_id]
                        num_alignment_failures += 1
                        warning_rows.append(
                            {
                                "warning_code": "multimer_usalign_failed",
                                "signature_cluster_id": signature_cluster_id,
                                "representative_multimer_observation_id": representative.multimer_observation_id,
                                "candidate_multimer_observation_id": candidate.multimer_observation_id,
                                "error": str(exc),
                            }
                        )
                        remaining.append(candidate)
                        continue
                    result = alignment_cache[pair_key]
                    if pair_key in success_keys:
                        alignment_rows.append(
                            {
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
                            }
                        )
                        num_alignment_runs += 1
                    if result.max_tm_score >= tm_score_threshold:
                        assigned.append(candidate)
                    else:
                        remaining.append(candidate)
                local_clusters.append((assigned, representative))
                pending = remaining
        elif members:
            representative = min(members, key=lambda item: item.structural_sort_key())
            local_clusters.append((list(members), representative))

        if unresolved_members and extracted_members:
            for unresolved in sorted(unresolved_members, key=lambda item: item.structural_sort_key()):
                local_clusters.append(([unresolved], unresolved))
                warning_rows.append(
                    {
                        "warning_code": "multimer_structure_unavailable_singleton_cluster",
                        "signature_cluster_id": signature_cluster_id,
                        "multimer_observation_id": unresolved.multimer_observation_id,
                    }
                )

        if len(local_clusters) > 1:
            num_signature_clusters_split += 1

        for members_in_cluster, representative in local_clusters:
            cluster_members.append(
                (
                    signature_cluster_id,
                    representative.multimer_observation_id,
                    sorted(members_in_cluster, key=lambda item: item.multimer_observation_id),
                    representative,
                )
            )

    membership_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    grouped_signature_sizes = {signature_cluster_id: 0 for signature_cluster_id, _ in signature_groups}
    for signature_cluster_id, _, _, _ in cluster_members:
        grouped_signature_sizes[signature_cluster_id] = grouped_signature_sizes.get(signature_cluster_id, 0) + 1

    for cluster_index, (signature_cluster_id, representative_id, members, representative) in enumerate(
        cluster_members,
        start=1,
    ):
        cluster_id = _multimer_cluster_id(cluster_index)
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
) -> dict[str, Any]:
    monomer_assignments = load_monomer_cluster_assignments(clustering_outdir)
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
    }
    extracted_structures: dict[str, ExtractedMultimerStructure] = {}
    if structure_refinement_mode == "greedy":
        extracted_structures, extraction_manifest = extract_multimer_structures(
            observations,
            outdir=outdir / "structures",
            model=model,
            drop_hydrogens=drop_hydrogens,
            extraction_jobs=alignment_jobs,
            prep_dir=prep_dir,
        )

    if structure_refinement_mode == "greedy":
        refined = refine_multimer_signature_clusters(
            signature_groups,
            extracted_structures,
            tm_score_threshold=multimer_tm_score_threshold,
            usalign_executable=usalign_executable,
            alignment_runner=alignment_runner,
            alignment_jobs=alignment_jobs,
        )
        membership_rows = refined["membership_rows"]
        representative_rows = refined["representative_rows"]
        signature_rows = refined["signature_rows"]
        alignment_rows = refined["alignment_rows"]
        warning_rows = refined["warning_rows"]
        manifest = {
            "num_multimer_observations": len(observations),
            "num_multimer_clusters": refined["manifest"]["num_multimer_clusters"],
            "num_signature_clusters": refined["manifest"]["num_signature_clusters"],
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1
                for observation in observations
                if "unclustered" in set(observation.member_cluster_sources)
            ),
            "num_alignment_runs": refined["manifest"]["num_alignment_runs"],
            "num_alignment_failures": refined["manifest"]["num_alignment_failures"],
            "num_signature_clusters_split": refined["manifest"]["num_signature_clusters_split"],
            "multimer_tm_score_threshold": multimer_tm_score_threshold,
            "structure_refinement_mode": structure_refinement_mode,
            "alignment_jobs": refined["manifest"]["alignment_jobs"],
            **extraction_manifest,
        }
    else:
        membership_rows = []
        representative_rows = []
        signature_rows = []
        alignment_rows = []
        warning_rows = []
        for cluster_index, (signature_cluster_id, members) in enumerate(signature_groups, start=1):
            representative = max(
                members,
                key=lambda item: (
                    item.support_score,
                    item.num_internal_edges,
                    item.num_member_instances,
                    item.multimer_observation_id,
                ),
            )
            cluster_id = _multimer_cluster_id(cluster_index)
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
