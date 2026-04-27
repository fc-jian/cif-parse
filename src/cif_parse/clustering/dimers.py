from __future__ import annotations

import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import biotite.structure as struc
from biotite.structure import AtomArray, get_residues
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import get_assembly, get_structure

from cif_parse.clustering.common import (
    canonical_monomer_id,
    load_monomer_cluster_assignments as _load_monomer_cluster_assignments,
    resolve_monomer_cluster as _common_resolve_monomer_cluster,
)
from cif_parse.clustering.protein_structures import (
    USalignAlignmentResult,
    parse_usalign_output,
)
from cif_parse.clustering.parallel import AlignmentTask, normalize_worker_count, run_alignment_tasks
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.io import read_cif_file
from cif_parse.settings import resolve_source_path
from cif_parse.utils.atom_filters import atom_array_filter_counts, filter_atom_array_for_analysis


LOGGER = logging.getLogger(__name__)
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _dimer_member_descriptor(
    *,
    chain_type: str,
    monomer_cluster_id: str,
) -> dict[str, str]:
    return {
        "chain_type": chain_type,
        "monomer_cluster_id": monomer_cluster_id,
    }


def _dimer_cluster_id(index: int) -> str:
    return f"dim_{index}"


def _dimer_signature_cluster_id(index: int) -> str:
    return f"dimsig_{index}"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _pdb_chain_id(index: int) -> str:
    if index >= len(PDB_CHAIN_IDS):
        raise ValueError(f"Too many dimer chains for PDB export: {index + 1}")
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


def _coerce_dimer_chain_id(atom_array: AtomArray, chain_id: str) -> AtomArray:
    copied = atom_array.copy()
    copied.chain_id[:] = chain_id
    return copied


@dataclass(slots=True)
class DimerObservation:
    dimer_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    assembly_mode: str
    sym_id_1: int | None
    label_asym_id_1: str
    auth_asym_id_1: str | None
    chain_type_1: str
    monomer_id_1: str
    monomer_sequence_cluster_id_1: str | None
    monomer_structure_cluster_id_1: str | None
    sym_id_2: int | None
    label_asym_id_2: str
    auth_asym_id_2: str | None
    chain_type_2: str
    monomer_id_2: str
    monomer_sequence_cluster_id_2: str | None
    monomer_structure_cluster_id_2: str | None
    interface_label: str
    is_same_entity: bool
    contains_antibody_unit: bool
    contains_tcr_pmhc_unit: bool
    buried_area: float
    num_residue_contacts: int
    num_atom_contacts: int
    signature_key: str
    signature_members: list[dict[str, str]]
    cluster_source_1: str
    cluster_source_2: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "dimer_observation_id": self.dimer_observation_id,
            "pdb_id": self.pdb_id,
            "source_path": self.source_path,
            "assembly_id": self.assembly_id or "",
            "assembly_mode": self.assembly_mode,
            "sym_id_1": self.sym_id_1 if self.sym_id_1 is not None else "",
            "label_asym_id_1": self.label_asym_id_1,
            "auth_asym_id_1": self.auth_asym_id_1 or "",
            "chain_type_1": self.chain_type_1,
            "monomer_id_1": self.monomer_id_1,
            "monomer_sequence_cluster_id_1": self.monomer_sequence_cluster_id_1 or "",
            "monomer_structure_cluster_id_1": self.monomer_structure_cluster_id_1 or "",
            "sym_id_2": self.sym_id_2 if self.sym_id_2 is not None else "",
            "label_asym_id_2": self.label_asym_id_2,
            "auth_asym_id_2": self.auth_asym_id_2 or "",
            "chain_type_2": self.chain_type_2,
            "monomer_id_2": self.monomer_id_2,
            "monomer_sequence_cluster_id_2": self.monomer_sequence_cluster_id_2 or "",
            "monomer_structure_cluster_id_2": self.monomer_structure_cluster_id_2 or "",
            "interface_label": self.interface_label,
            "is_same_entity": self.is_same_entity,
            "contains_antibody_unit": self.contains_antibody_unit,
            "contains_tcr_pmhc_unit": self.contains_tcr_pmhc_unit,
            "buried_area": self.buried_area,
            "num_residue_contacts": self.num_residue_contacts,
            "num_atom_contacts": self.num_atom_contacts,
            "signature_key": self.signature_key,
            "signature_members": json.dumps(self.signature_members, ensure_ascii=False),
            "cluster_source_1": self.cluster_source_1,
            "cluster_source_2": self.cluster_source_2,
        }

    def structural_sort_key(self) -> tuple[float, float, float, str]:
        return (
            -float(self.buried_area),
            -float(self.num_atom_contacts),
            -float(self.num_residue_contacts),
            self.dimer_observation_id,
        )


@dataclass(slots=True)
class ExtractedDimerStructure:
    dimer_observation_id: str
    pdb_id: str
    source_path: str
    assembly_id: str | None
    extracted_pdb_path: str
    residue_count: int
    atom_count: int
    filter_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_monomer_cluster_assignments(clustering_outdir: str | Path) -> dict[str, dict[str, str]]:
    """Load monomer sequence / structure cluster assignments from clustering artifacts."""

    return _load_monomer_cluster_assignments(clustering_outdir)


def _resolve_monomer_cluster(
    monomer_id: str,
    assignments: dict[str, dict[str, str]],
) -> tuple[str, str | None, str | None]:
    return _common_resolve_monomer_cluster(monomer_id, assignments)


def collect_dimer_observations(
    case_dirs: Iterable[str | Path],
    monomer_cluster_assignments: dict[str, dict[str, str]],
    cif_files_directory: str | None = None,
    prep_db_path: str | Path | None = None,
) -> list[DimerObservation]:
    """Collect dimer observations from case-output bundles."""

    observations: list[DimerObservation] = []
    from cif_parse.clustering.prep import load_bundles_for_collect, load_case_bundles

    sorted_dirs = sorted(Path(path).resolve() for path in case_dirs)
    prep_bundles = load_bundles_for_collect(sorted_dirs, prep_db_path=prep_db_path)
    for case_dir in sorted_dirs:
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
            dimers = payload.get("dimer_interfaces", [])
            if not isinstance(dimers, list):
                continue
            for index, dimer in enumerate(dimers, start=1):
                if not isinstance(dimer, dict):
                    continue
                label_asym_id_1 = str(dimer.get("label_asym_id_1", "") or "")
                label_asym_id_2 = str(dimer.get("label_asym_id_2", "") or "")
                if not label_asym_id_1 or not label_asym_id_2:
                    continue
                monomer_id_1 = canonical_monomer_id(pdb_id, label_asym_id_1)
                monomer_id_2 = canonical_monomer_id(pdb_id, label_asym_id_2)
                cluster_source_1, cluster_id_1, sequence_cluster_id_1 = _resolve_monomer_cluster(
                    monomer_id_1,
                    monomer_cluster_assignments,
                )
                cluster_source_2, cluster_id_2, sequence_cluster_id_2 = _resolve_monomer_cluster(
                    monomer_id_2,
                    monomer_cluster_assignments,
                )
                signature_members = sorted(
                    [
                        _dimer_member_descriptor(
                            chain_type=str(dimer.get("chain_type_1", "") or ""),
                            monomer_cluster_id=cluster_id_1,
                        ),
                        _dimer_member_descriptor(
                            chain_type=str(dimer.get("chain_type_2", "") or ""),
                            monomer_cluster_id=cluster_id_2,
                        ),
                    ],
                    key=lambda item: (item["chain_type"], item["monomer_cluster_id"]),
                )
                signature_key = json.dumps(
                    {
                        "members": signature_members,
                        "interface_label": str(dimer.get("interface_label", "") or ""),
                        "is_same_entity": bool(dimer.get("is_same_entity", False)),
                        "contains_antibody_unit": bool(dimer.get("contains_antibody_unit", False)),
                        "contains_tcr_pmhc_unit": bool(
                            dimer.get("contains_tcr_pmhc_unit", False)
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                assembly_id = dimer.get("assembly_id")
                observations.append(
                    DimerObservation(
                        dimer_observation_id=(
                            f"{pdb_id}:{assembly_id or default_assembly_id or 'na'}:{index}"
                        ),
                        pdb_id=pdb_id,
                        source_path=source_path,
                        assembly_id=str(assembly_id) if assembly_id is not None else default_assembly_id,
                        assembly_mode=str(dimer.get("assembly_mode", "") or ""),
                        sym_id_1=_optional_int(dimer.get("sym_id_1")),
                        label_asym_id_1=label_asym_id_1,
                        auth_asym_id_1=(
                            str(dimer.get("auth_asym_id_1"))
                            if dimer.get("auth_asym_id_1") is not None
                            else None
                        ),
                        chain_type_1=str(dimer.get("chain_type_1", "") or ""),
                        monomer_id_1=monomer_id_1,
                        monomer_sequence_cluster_id_1=sequence_cluster_id_1,
                        monomer_structure_cluster_id_1=(
                            cluster_id_1 if cluster_source_1 == "structure" else None
                        ),
                        sym_id_2=_optional_int(dimer.get("sym_id_2")),
                        label_asym_id_2=label_asym_id_2,
                        auth_asym_id_2=(
                            str(dimer.get("auth_asym_id_2"))
                            if dimer.get("auth_asym_id_2") is not None
                            else None
                        ),
                        chain_type_2=str(dimer.get("chain_type_2", "") or ""),
                        monomer_id_2=monomer_id_2,
                        monomer_sequence_cluster_id_2=sequence_cluster_id_2,
                        monomer_structure_cluster_id_2=(
                            cluster_id_2 if cluster_source_2 == "structure" else None
                        ),
                        interface_label=str(dimer.get("interface_label", "") or ""),
                        is_same_entity=bool(dimer.get("is_same_entity", False)),
                        contains_antibody_unit=bool(dimer.get("contains_antibody_unit", False)),
                        contains_tcr_pmhc_unit=bool(dimer.get("contains_tcr_pmhc_unit", False)),
                        buried_area=float(dimer.get("buried_area", 0.0) or 0.0),
                        num_residue_contacts=int(dimer.get("num_residue_contacts", 0) or 0),
                        num_atom_contacts=int(dimer.get("num_atom_contacts", 0) or 0),
                        signature_key=signature_key,
                        signature_members=signature_members,
                        cluster_source_1=cluster_source_1,
                        cluster_source_2=cluster_source_2,
                    )
                )
    LOGGER.info("Collected %d dimer observations from %d case dir(s)", len(observations), len(list(case_dirs)))
    return observations


def extract_dimer_structure(
    observation: DimerObservation,
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    atom_array: AtomArray | None = None,
) -> ExtractedDimerStructure:
    if not observation.source_path:
        raise ValueError(f"Missing source_path for dimer {observation.dimer_observation_id}")

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

    chain_arrays: list[AtomArray] = []
    for chain_index, (label_asym_id, sym_id) in enumerate(
        (
            (observation.label_asym_id_1, observation.sym_id_1),
            (observation.label_asym_id_2, observation.sym_id_2),
        )
    ):
        selected = _select_instance_atoms(atom_array, label_asym_id=label_asym_id, sym_id=sym_id)
        if selected.array_length() == 0 and sym_id is not None:
            selected = _select_instance_atoms(atom_array, label_asym_id=label_asym_id, sym_id=None)
        if selected.array_length() == 0:
            continue
        chain_arrays.append(_coerce_dimer_chain_id(selected, _pdb_chain_id(chain_index)))

    if not chain_arrays:
        raise ValueError(f"No polymer atoms found for dimer {observation.dimer_observation_id}")

    dimer_atoms = struc.concatenate(chain_arrays)
    dimer_atoms, filter_counts = filter_atom_array_for_analysis(
        dimer_atoms,
        drop_hydrogens=drop_hydrogens,
        drop_nonfinite=True,
    )
    if dimer_atoms.array_length() == 0:
        raise ValueError(f"No analyzable atoms left for dimer {observation.dimer_observation_id}")

    _, residue_names = get_residues(dimer_atoms)
    residue_count = int(len(residue_names))
    if residue_count <= 2:
        raise ValueError(
            f"Resolved residue count {residue_count} is too short for dimer USalign: "
            f"{observation.dimer_observation_id}"
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    safe_id = observation.dimer_observation_id.replace(":", "_").replace("/", "_")
    pdb_path = outdir / f"{safe_id}.pdb"
    pdb_file = PDBFile()
    pdb_file.set_structure(dimer_atoms)
    pdb_file.write(pdb_path)
    return ExtractedDimerStructure(
        dimer_observation_id=observation.dimer_observation_id,
        pdb_id=observation.pdb_id,
        source_path=observation.source_path,
        assembly_id=observation.assembly_id,
        extracted_pdb_path=str(pdb_path),
        residue_count=residue_count,
        atom_count=int(dimer_atoms.array_length()),
        filter_counts=atom_array_filter_counts(filter_counts),
    )


def extract_dimer_structures(
    observations: Iterable[DimerObservation],
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    extraction_jobs: int = 1,
) -> tuple[dict[str, ExtractedDimerStructure], dict[str, Any]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    structures: dict[str, ExtractedDimerStructure] = {}
    failures: list[dict[str, str]] = []

    sorted_observations = sorted(observations, key=lambda item: item.dimer_observation_id)
    extraction_jobs = normalize_worker_count(extraction_jobs)

    def _load_atom_array(observation: DimerObservation, cache: dict, lock) -> AtomArray:
        cache_key = (observation.source_path, observation.assembly_id)
        with lock:
            if cache_key not in cache:
                cif_file = read_cif_file(observation.source_path)
                if observation.assembly_id:
                    cache[cache_key] = get_assembly(
                        cif_file,
                        assembly_id=observation.assembly_id,
                        model=model,
                        use_author_fields=False,
                    )
                else:
                    cache[cache_key] = get_structure(
                        cif_file,
                        model=model,
                        use_author_fields=False,
                    )
        return cache[cache_key]

    if extraction_jobs <= 1 or len(sorted_observations) <= 1:
        atom_array_cache: dict[tuple[str, str | None], AtomArray] = {}
        import threading as _threading
        _lock = _threading.Lock()
        for observation in sorted_observations:
            try:
                atom_array = _load_atom_array(observation, atom_array_cache, _lock)
                structures[observation.dimer_observation_id] = extract_dimer_structure(
                    observation,
                    outdir=outdir,
                    model=model,
                    drop_hydrogens=drop_hydrogens,
                    atom_array=atom_array,
                )
            except Exception as exc:
                LOGGER.warning("Failed to extract dimer %s: %s", observation.dimer_observation_id, exc)
                failures.append(
                    {"dimer_observation_id": observation.dimer_observation_id, "error": str(exc)}
                )
    else:
        import threading as _threading

        atom_array_cache: dict[tuple[str, str | None], AtomArray] = {}
        cache_lock = _threading.Lock()

        def _extract_one(observation: DimerObservation) -> ExtractedDimerStructure | None:
            atom_array = _load_atom_array(observation, atom_array_cache, cache_lock)
            return extract_dimer_structure(
                observation,
                outdir=outdir,
                model=model,
                drop_hydrogens=drop_hydrogens,
                atom_array=atom_array,
            )

        with ThreadPoolExecutor(max_workers=min(extraction_jobs, len(sorted_observations))) as executor:
            future_to_obs = {
                executor.submit(_extract_one, observation): observation
                for observation in sorted_observations
            }
            for future in as_completed(future_to_obs):
                observation = future_to_obs[future]
                try:
                    extracted = future.result()
                    if extracted is not None:
                        structures[observation.dimer_observation_id] = extracted
                except Exception as exc:
                    LOGGER.warning("Failed to extract dimer %s: %s", observation.dimer_observation_id, exc)
                    failures.append(
                        {"dimer_observation_id": observation.dimer_observation_id, "error": str(exc)}
                    )

    dump_jsonl(outdir / "dimer_structure_extraction_failures.jsonl", failures)
    dump_jsonl(outdir / "dimer_structures.jsonl", [item.to_dict() for item in structures.values()])
    manifest = {
        "num_dimer_observations": len(sorted_observations),
        "num_extracted_dimer_structures": len(structures),
        "num_failed_dimer_structure_extractions": len(failures),
        "extraction_jobs": extraction_jobs,
    }
    dump_json(outdir / "dimer_structure_manifest.json", manifest, indent=2)
    LOGGER.info("Extracted %d dimer structures (%d failures)", len(structures), len(failures))
    return structures, manifest


def run_dimer_usalign_alignment(
    query: ExtractedDimerStructure,
    target: ExtractedDimerStructure,
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
        query_monomer_id=query.dimer_observation_id,
        target_monomer_id=target.dimer_observation_id,
        query_length=query.residue_count,
        target_length=target.residue_count,
        tm_score_threshold=tm_score_threshold,
        min_alignment_coverage_ratio=0.0,
    )


def refine_dimer_signature_clusters(
    signature_groups: list[tuple[str, list[DimerObservation]]],
    extracted_structures: dict[str, ExtractedDimerStructure],
    *,
    tm_score_threshold: float = 0.50,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
) -> dict[str, Any]:
    runner = alignment_runner or run_dimer_usalign_alignment
    alignment_jobs = normalize_worker_count(alignment_jobs)
    alignment_cache: dict[tuple[str, str], USalignAlignmentResult] = {}
    alignment_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    cluster_members: list[tuple[str, str, list[DimerObservation], DimerObservation]] = []
    num_alignment_runs = 0
    num_alignment_failures = 0
    num_signature_clusters_split = 0

    total_observations = sum(len(members) for _, members in signature_groups)
    LOGGER.info(
        "Refining %d dimer signature clusters (%d observations, %d alignment workers)",
        len(signature_groups),
        total_observations,
        alignment_jobs,
    )
    for signature_cluster_id, members in signature_groups:
        extracted_members = [
            member for member in members if member.dimer_observation_id in extracted_structures
        ]
        unresolved_members = [
            member for member in members if member.dimer_observation_id not in extracted_structures
        ]

        local_clusters: list[tuple[list[DimerObservation], DimerObservation]] = []
        if extracted_members:
            pending = sorted(extracted_members, key=lambda item: item.structural_sort_key())
            while pending:
                representative = pending[0]
                assigned = [representative]
                remaining: list[DimerObservation] = []
                alignment_tasks: list[AlignmentTask] = []
                for candidate in pending[1:]:
                    pair_key = tuple(
                        sorted((representative.dimer_observation_id, candidate.dimer_observation_id))
                    )
                    if pair_key not in alignment_cache:
                        alignment_tasks.append(
                            AlignmentTask(
                                key=pair_key,
                                query=extracted_structures[representative.dimer_observation_id],
                                target=extracted_structures[candidate.dimer_observation_id],
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
                    task.context["candidate"].dimer_observation_id: exc for task, exc in failures
                }
                for task, result in successes:
                    alignment_cache[task.key] = result
                success_keys = {task.key for task, _ in successes}
                for candidate in pending[1:]:
                    pair_key = tuple(
                        sorted((representative.dimer_observation_id, candidate.dimer_observation_id))
                    )
                    if candidate.dimer_observation_id in failure_by_candidate_id:
                        exc = failure_by_candidate_id[candidate.dimer_observation_id]
                        num_alignment_failures += 1
                        warning_rows.append(
                            {
                                "warning_code": "dimer_usalign_failed",
                                "signature_cluster_id": signature_cluster_id,
                                "representative_dimer_observation_id": representative.dimer_observation_id,
                                "candidate_dimer_observation_id": candidate.dimer_observation_id,
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
                                "query_dimer_observation_id": result.query_monomer_id,
                                "target_dimer_observation_id": result.target_monomer_id,
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
                        "warning_code": "dimer_structure_unavailable_singleton_cluster",
                        "signature_cluster_id": signature_cluster_id,
                        "dimer_observation_id": unresolved.dimer_observation_id,
                    }
                )

        if len(local_clusters) > 1:
            num_signature_clusters_split += 1

        for members_in_cluster, representative in local_clusters:
            cluster_members.append(
                (
                    signature_cluster_id,
                    representative.dimer_observation_id,
                    sorted(members_in_cluster, key=lambda item: item.dimer_observation_id),
                    representative,
                )
            )

    grouped_signature_sizes = {signature_cluster_id: 0 for signature_cluster_id, _ in signature_groups}
    for signature_cluster_id, _, _, _ in cluster_members:
        grouped_signature_sizes[signature_cluster_id] = grouped_signature_sizes.get(signature_cluster_id, 0) + 1

    membership_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    for cluster_index, (signature_cluster_id, representative_id, members, representative) in enumerate(
        cluster_members,
        start=1,
    ):
        cluster_id = _dimer_cluster_id(cluster_index)
        representative_rows.append(
            {
                "dimer_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "representative_dimer_observation_id": representative_id,
                "num_members": len(members),
                "pdb_id": representative.pdb_id,
                "assembly_id": representative.assembly_id or "",
                "interface_label": representative.interface_label,
                "buried_area": representative.buried_area,
                "num_atom_contacts": representative.num_atom_contacts,
                "signature_key": representative.signature_key,
            }
        )
        signature_rows.append(
            {
                "dimer_cluster_id": cluster_id,
                "signature_cluster_id": signature_cluster_id,
                "signature_key": representative.signature_key,
                "signature_members": json.dumps(
                    representative.signature_members,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "interface_label": representative.interface_label,
                "contains_antibody_unit": representative.contains_antibody_unit,
                "contains_tcr_pmhc_unit": representative.contains_tcr_pmhc_unit,
                "is_same_entity": representative.is_same_entity,
                "num_refined_clusters_in_signature_group": grouped_signature_sizes.get(
                    signature_cluster_id,
                    1,
                ),
            }
        )
        for member in members:
            tm_score_for_clustering: float | str = ""
            if member.dimer_observation_id != representative_id:
                pair_key = tuple(sorted((representative_id, member.dimer_observation_id)))
                if pair_key in alignment_cache:
                    tm_score_for_clustering = alignment_cache[pair_key].max_tm_score
            membership_rows.append(
                _dimer_membership_record(
                    member,
                    dimer_cluster_id=cluster_id,
                    signature_cluster_id=signature_cluster_id,
                    representative_dimer_observation_id=representative_id,
                    tm_score_to_representative=tm_score_for_clustering,
                )
            )

    manifest = {
        "num_signature_clusters": len(signature_groups),
        "num_dimer_clusters": len(cluster_members),
        "num_alignment_runs": num_alignment_runs,
        "num_alignment_failures": num_alignment_failures,
        "num_signature_clusters_split": num_signature_clusters_split,
        "dimer_tm_score_threshold": tm_score_threshold,
        "alignment_jobs": alignment_jobs,
    }
    LOGGER.info(
        "Dimer refinement: %d signature clusters -> %d refined clusters (%d alignments, %d failures, %d splits)",
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


def _dimer_membership_record(
    member: DimerObservation,
    *,
    dimer_cluster_id: str,
    signature_cluster_id: str,
    representative_dimer_observation_id: str,
    tm_score_to_representative: float | str,
) -> dict[str, Any]:
    return {
        "dimer_cluster_id": dimer_cluster_id,
        "signature_cluster_id": signature_cluster_id,
        "dimer_observation_id": member.dimer_observation_id,
        "pdb_id": member.pdb_id,
        "assembly_id": member.assembly_id or "",
        "label_asym_id_1": member.label_asym_id_1,
        "auth_asym_id_1": member.auth_asym_id_1 or "",
        "chain_type_1": member.chain_type_1,
        "monomer_id_1": member.monomer_id_1,
        "monomer_structure_cluster_id_1": member.monomer_structure_cluster_id_1 or "",
        "monomer_sequence_cluster_id_1": member.monomer_sequence_cluster_id_1 or "",
        "cluster_source_1": member.cluster_source_1,
        "label_asym_id_2": member.label_asym_id_2,
        "auth_asym_id_2": member.auth_asym_id_2 or "",
        "chain_type_2": member.chain_type_2,
        "monomer_id_2": member.monomer_id_2,
        "monomer_structure_cluster_id_2": member.monomer_structure_cluster_id_2 or "",
        "monomer_sequence_cluster_id_2": member.monomer_sequence_cluster_id_2 or "",
        "cluster_source_2": member.cluster_source_2,
        "interface_label": member.interface_label,
        "buried_area": member.buried_area,
        "num_atom_contacts": member.num_atom_contacts,
        "num_residue_contacts": member.num_residue_contacts,
        "representative_dimer_observation_id": representative_dimer_observation_id,
        "tm_score_to_representative": tm_score_to_representative,
    }


def build_dimer_signature_clusters(
    *,
    case_dirs: Iterable[str | Path],
    clustering_outdir: str | Path,
    outdir: str | Path,
    structure_refinement_mode: str = "greedy",
    dimer_tm_score_threshold: float = 0.50,
    model: int = 1,
    drop_hydrogens: bool = True,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    alignment_jobs: int = 1,
    cif_files_directory: str | None = None,
    prep_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build dimer clusters from monomer assignments and optional structure refinement."""

    monomer_assignments = load_monomer_cluster_assignments(clustering_outdir)
    observations = collect_dimer_observations(
        case_dirs, monomer_assignments,
        cif_files_directory=cif_files_directory,
        prep_db_path=prep_dir,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dump_jsonl(outdir / "dimer_inventory.jsonl", [item.to_dict() for item in observations])
    dump_csv_rows(outdir / "dimer_inventory.csv", [item.to_record() for item in observations])

    grouped: dict[str, list[DimerObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.signature_key, []).append(observation)
    signature_groups = [
        (_dimer_signature_cluster_id(index), members)
        for index, (_, members) in enumerate(
            sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])),
            start=1,
        )
    ]

    extraction_manifest = {
        "num_extracted_dimer_structures": 0,
        "num_failed_dimer_structure_extractions": 0,
    }
    extracted_structures: dict[str, ExtractedDimerStructure] = {}
    if structure_refinement_mode == "greedy":
        extracted_structures, extraction_manifest = extract_dimer_structures(
            observations,
            outdir=outdir / "structures",
            model=model,
            drop_hydrogens=drop_hydrogens,
            extraction_jobs=alignment_jobs,
        )

    if structure_refinement_mode == "greedy":
        refined = refine_dimer_signature_clusters(
            signature_groups,
            extracted_structures,
            tm_score_threshold=dimer_tm_score_threshold,
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
            "num_dimer_observations": len(observations),
            "num_dimer_clusters": refined["manifest"]["num_dimer_clusters"],
            "num_signature_clusters": refined["manifest"]["num_signature_clusters"],
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1
                for observation in observations
                if "unclustered" in {observation.cluster_source_1, observation.cluster_source_2}
            ),
            "num_alignment_runs": refined["manifest"]["num_alignment_runs"],
            "num_alignment_failures": refined["manifest"]["num_alignment_failures"],
            "num_signature_clusters_split": refined["manifest"]["num_signature_clusters_split"],
            "dimer_tm_score_threshold": dimer_tm_score_threshold,
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
        for cluster_index, (signature_cluster_id, members) in enumerate(
            signature_groups,
            start=1,
        ):
            cluster_id = _dimer_cluster_id(cluster_index)
            representative = min(members, key=lambda item: item.structural_sort_key())
            representative_rows.append(
                {
                    "dimer_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "representative_dimer_observation_id": representative.dimer_observation_id,
                    "num_members": len(members),
                    "pdb_id": representative.pdb_id,
                    "assembly_id": representative.assembly_id or "",
                    "interface_label": representative.interface_label,
                    "buried_area": representative.buried_area,
                    "num_atom_contacts": representative.num_atom_contacts,
                    "signature_key": representative.signature_key,
                }
            )
            signature_rows.append(
                {
                    "dimer_cluster_id": cluster_id,
                    "signature_cluster_id": signature_cluster_id,
                    "signature_key": representative.signature_key,
                    "signature_members": json.dumps(
                        representative.signature_members,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "interface_label": representative.interface_label,
                    "contains_antibody_unit": representative.contains_antibody_unit,
                    "contains_tcr_pmhc_unit": representative.contains_tcr_pmhc_unit,
                    "is_same_entity": representative.is_same_entity,
                    "num_refined_clusters_in_signature_group": 1,
                }
            )
            for member in sorted(members, key=lambda item: item.dimer_observation_id):
                membership_rows.append(
                    _dimer_membership_record(
                        member,
                        dimer_cluster_id=cluster_id,
                        signature_cluster_id=signature_cluster_id,
                        representative_dimer_observation_id=representative.dimer_observation_id,
                        tm_score_to_representative="",
                    )
                )
        manifest = {
            "num_dimer_observations": len(observations),
            "num_dimer_clusters": len(grouped),
            "num_signature_clusters": len(grouped),
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1
                for observation in observations
                if "unclustered" in {observation.cluster_source_1, observation.cluster_source_2}
            ),
            "num_alignment_runs": 0,
            "num_alignment_failures": 0,
            "num_signature_clusters_split": 0,
            "dimer_tm_score_threshold": dimer_tm_score_threshold,
            "structure_refinement_mode": structure_refinement_mode,
            **extraction_manifest,
        }

    dump_csv_rows(outdir / "dimer_cluster_membership.csv", membership_rows)
    dump_csv_rows(outdir / "dimer_cluster_representatives.csv", representative_rows)
    dump_csv_rows(outdir / "dimer_cluster_signatures.csv", signature_rows)
    dump_jsonl(outdir / "dimer_pairwise_alignments.jsonl", alignment_rows)
    dump_jsonl(outdir / "dimer_cluster_warnings.jsonl", warning_rows)
    dump_json(outdir / "dimer_cluster_manifest.json", manifest, indent=2)
    return {
        "manifest": manifest,
        "observations": observations,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "signature_rows": signature_rows,
        "alignment_rows": alignment_rows,
    }
