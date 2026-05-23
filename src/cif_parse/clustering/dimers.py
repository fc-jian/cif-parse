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
    load_monomer_cluster_assignments as _load_monomer_cluster_assignments,
    resolve_monomer_cluster as _common_resolve_monomer_cluster,
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


def _dimer_member_descriptor(
    *,
    chain_type: str,
    monomer_cluster_id: str,
) -> dict[str, str]:
    return {
        "chain_type": chain_type,
        "monomer_cluster_id": monomer_cluster_id,
    }


def _signature_index(signature_cluster_id: str) -> str:
    return signature_cluster_id.rsplit("_", 1)[-1]


def _dimer_cluster_id(signature_cluster_id: str, local_index: int = 1) -> str:
    return f"dim_{_signature_index(signature_cluster_id)}_{local_index}"


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


def load_monomer_cluster_assignments(
    clustering_outdir: str | Path,
    *,
    include_structure: bool = True,
) -> dict[str, dict[str, str]]:
    """Load monomer sequence / structure cluster assignments from clustering artifacts."""

    return _load_monomer_cluster_assignments(
        clustering_outdir,
        include_structure=include_structure,
    )


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
    prep_dir: str | Path | None = None,
) -> list[DimerObservation]:
    """Collect dimer observations from case-output bundles or prep Parquet."""

    observations: list[DimerObservation] = []

    # Fast path: read pre-parsed Parquet
    from cif_parse.clustering.prep import open_prep_parquet, iter_parquet_rows
    pf = open_prep_parquet(prep_dir, "dimers", required=True) if prep_dir else None
    if pf is not None:
        for row in tqdm(iter_parquet_rows(prep_dir, "dimers", required=True),
                        desc="Collecting dimers", unit="dimer"):
            pdb_id = row.get("pdb_id", "")
            lbl1 = row.get("label_asym_id_1", "")
            lbl2 = row.get("label_asym_id_2", "")
            if not pdb_id or not lbl1 or not lbl2:
                continue
            source_path = resolve_source_path(row.get("source_path", ""), cif_files_directory)
            dim_assembly_id = str(row.get("assembly_id", "") or "")
            monomer_id_1 = canonical_monomer_id(pdb_id, lbl1, dim_assembly_id)
            monomer_id_2 = canonical_monomer_id(pdb_id, lbl2, dim_assembly_id)
            cs1, cid1, scid1 = _resolve_monomer_cluster(monomer_id_1, monomer_cluster_assignments)
            cs2, cid2, scid2 = _resolve_monomer_cluster(monomer_id_2, monomer_cluster_assignments)
            sm = sorted([
                _dimer_member_descriptor(chain_type=row.get("chain_type_1", ""), monomer_cluster_id=cid1),
                _dimer_member_descriptor(chain_type=row.get("chain_type_2", ""), monomer_cluster_id=cid2),
            ], key=lambda x: (x["chain_type"], x["monomer_cluster_id"]))
            sig_key = json.dumps({
                "members": sm,
                "interface_label": row.get("interface_label", ""),
                "is_same_entity": row.get("is_same_entity", False),
                "contains_antibody_unit": row.get("contains_antibody_unit", False),
                "contains_tcr_pmhc_unit": row.get("contains_tcr_pmhc_unit", False),
            }, ensure_ascii=False, sort_keys=True)
            asm_id = row.get("assembly_id") or None
            observations.append(DimerObservation(
                dimer_observation_id=f"{pdb_id}:{asm_id or 'na'}:{row.get('dimer_index', 0)}",
                pdb_id=pdb_id, source_path=source_path,
                assembly_id=str(asm_id) if asm_id else None,
                assembly_mode=row.get("assembly_mode", ""),
                sym_id_1=_optional_int(row.get("sym_id_1")),
                label_asym_id_1=lbl1, auth_asym_id_1=row.get("auth_asym_id_1"),
                chain_type_1=row.get("chain_type_1", ""), monomer_id_1=monomer_id_1,
                monomer_sequence_cluster_id_1=scid1,
                monomer_structure_cluster_id_1=cid1 if cs1 == "structure" else None,
                sym_id_2=_optional_int(row.get("sym_id_2")),
                label_asym_id_2=lbl2, auth_asym_id_2=row.get("auth_asym_id_2"),
                chain_type_2=row.get("chain_type_2", ""), monomer_id_2=monomer_id_2,
                monomer_sequence_cluster_id_2=scid2,
                monomer_structure_cluster_id_2=cid2 if cs2 == "structure" else None,
                interface_label=row.get("interface_label", ""),
                is_same_entity=row.get("is_same_entity", False),
                contains_antibody_unit=row.get("contains_antibody_unit", False),
                contains_tcr_pmhc_unit=row.get("contains_tcr_pmhc_unit", False),
                buried_area=float(row.get("buried_area", 0) or 0),
                num_residue_contacts=int(row.get("num_residue_contacts", 0) or 0),
                num_atom_contacts=int(row.get("num_atom_contacts", 0) or 0),
                signature_key=sig_key, signature_members=sm,
                cluster_source_1=cs1, cluster_source_2=cs2,
            ))
        LOGGER.info("Collected %d dimer observations from prep Parquet", len(observations))
        return observations

    # Slow path: read individual JSON bundles
    from cif_parse.clustering.prep import load_bundles_for_collect, load_case_bundles
    sorted_dirs = sorted(Path(path).resolve() for path in case_dirs)
    prep_bundles = load_bundles_for_collect(sorted_dirs, prep_db_path=prep_dir)
    for case_dir in sorted_dirs:
        payloads = load_case_bundles(case_dir, prep_bundles=prep_bundles)
        for payload in payloads:
            summary = payload.get("structure_summary", {})
            pdb_id = str(summary.get("pdb_id", "") or "")
            source_path = resolve_source_path(str(summary.get("source_path", "") or ""), cif_files_directory)
            assembly_ids = [str(item) for item in summary.get("assembly_ids", []) if str(item)]
            default_assembly_id = assembly_ids[0] if len(assembly_ids) == 1 else None
            dimers = payload.get("dimer_interfaces", [])
            if not isinstance(dimers, list):
                continue
            for index, dimer in enumerate(dimers, start=1):
                if not isinstance(dimer, dict):
                    continue
                lbl1 = str(dimer.get("label_asym_id_1", "") or "")
                lbl2 = str(dimer.get("label_asym_id_2", "") or "")
                if not lbl1 or not lbl2:
                    continue
                slow_asm_id = str(dimer.get("assembly_id", "") or default_assembly_id or "")
                mid1 = canonical_monomer_id(pdb_id, lbl1, slow_asm_id)
                mid2 = canonical_monomer_id(pdb_id, lbl2, slow_asm_id)
                cs1, cid1, scid1 = _resolve_monomer_cluster(mid1, monomer_cluster_assignments)
                cs2, cid2, scid2 = _resolve_monomer_cluster(mid2, monomer_cluster_assignments)
                sm = sorted([
                    _dimer_member_descriptor(chain_type=str(dimer.get("chain_type_1", "") or ""), monomer_cluster_id=cid1),
                    _dimer_member_descriptor(chain_type=str(dimer.get("chain_type_2", "") or ""), monomer_cluster_id=cid2),
                ], key=lambda x: (x["chain_type"], x["monomer_cluster_id"]))
                sig_key = json.dumps({
                    "members": sm,
                    "interface_label": str(dimer.get("interface_label", "") or ""),
                    "is_same_entity": bool(dimer.get("is_same_entity", False)),
                    "contains_antibody_unit": bool(dimer.get("contains_antibody_unit", False)),
                    "contains_tcr_pmhc_unit": bool(dimer.get("contains_tcr_pmhc_unit", False)),
                }, ensure_ascii=False, sort_keys=True)
                asm_id = dimer.get("assembly_id")
                observations.append(DimerObservation(
                    dimer_observation_id=f"{pdb_id}:{asm_id or default_assembly_id or 'na'}:{index}",
                    pdb_id=pdb_id, source_path=source_path,
                    assembly_id=str(asm_id) if asm_id is not None else default_assembly_id,
                    assembly_mode=str(dimer.get("assembly_mode", "") or ""),
                    sym_id_1=_optional_int(dimer.get("sym_id_1")),
                    label_asym_id_1=lbl1, auth_asym_id_1=str(dimer.get("auth_asym_id_1")) if dimer.get("auth_asym_id_1") is not None else None,
                    chain_type_1=str(dimer.get("chain_type_1", "") or ""), monomer_id_1=mid1,
                    monomer_sequence_cluster_id_1=scid1,
                    monomer_structure_cluster_id_1=cid1 if cs1 == "structure" else None,
                    sym_id_2=_optional_int(dimer.get("sym_id_2")),
                    label_asym_id_2=lbl2, auth_asym_id_2=str(dimer.get("auth_asym_id_2")) if dimer.get("auth_asym_id_2") is not None else None,
                    chain_type_2=str(dimer.get("chain_type_2", "") or ""), monomer_id_2=mid2,
                    monomer_sequence_cluster_id_2=scid2,
                    monomer_structure_cluster_id_2=cid2 if cs2 == "structure" else None,
                    interface_label=str(dimer.get("interface_label", "") or ""),
                    is_same_entity=bool(dimer.get("is_same_entity", False)),
                    contains_antibody_unit=bool(dimer.get("contains_antibody_unit", False)),
                    contains_tcr_pmhc_unit=bool(dimer.get("contains_tcr_pmhc_unit", False)),
                    buried_area=float(dimer.get("buried_area", 0.0) or 0.0),
                    num_residue_contacts=int(dimer.get("num_residue_contacts", 0) or 0),
                    num_atom_contacts=int(dimer.get("num_atom_contacts", 0) or 0),
                    signature_key=sig_key, signature_members=sm,
                    cluster_source_1=cs1, cluster_source_2=cs2,
                ))
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
        raise ValueError(f"Cached coordinates are required for dimer {observation.dimer_observation_id}")

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
    prep_dir: str | Path | None = None,
    show_progress: bool = True,
    log_summary: bool = True,
) -> tuple[dict[str, ExtractedDimerStructure], dict[str, Any]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    structures: dict[str, ExtractedDimerStructure] = {}
    failures: list[dict[str, str]] = []

    sorted_observations = sorted(observations, key=lambda item: item.dimer_observation_id)
    extraction_jobs = normalize_worker_count(extraction_jobs)

    # Fast path: assemble dimer from per-chain blobs
    _prep_cif_idx: dict | None = None
    if prep_dir:
        from cif_parse.clustering.prep import load_cif_coords_index, assemble_atom_array_from_chains
        _prep_cif_idx = load_cif_coords_index(prep_dir)

    def _try_assemble_from_chains(obs: DimerObservation) -> AtomArray | None:
        if _prep_cif_idx is None:
            return None
        return assemble_atom_array_from_chains(
            prep_dir, obs.source_path,
            [(obs.label_asym_id_1, obs.sym_id_1), (obs.label_asym_id_2, obs.sym_id_2)],
            assembly_id=obs.assembly_id, index=_prep_cif_idx,
        )

    if extraction_jobs <= 1 or len(sorted_observations) <= 1:
        observation_iter = (
            tqdm(sorted_observations, desc="Extracting dimer structures", unit="dimer")
            if show_progress
            else sorted_observations
        )
        for observation in observation_iter:
            try:
                atom_array = _try_assemble_from_chains(observation)
                if atom_array is None:
                    raise ValueError(
                        f"Prep coordinates missing for dimer {observation.dimer_observation_id}"
                    )
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
        def _extract_one(observation: DimerObservation) -> ExtractedDimerStructure | None:
            atom_array = _try_assemble_from_chains(observation)
            if atom_array is None:
                raise ValueError(
                    f"Prep coordinates missing for dimer {observation.dimer_observation_id}"
                )
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
            future_iter = as_completed(future_to_obs)
            if show_progress:
                future_iter = tqdm(
                    future_iter,
                    total=len(future_to_obs),
                    desc="Extracting dimer structures",
                    unit="dimer",
                )
            for future in future_iter:
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
    if log_summary:
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
        "auto",
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
    show_progress: bool = True,
    log_summary: bool = True,
) -> dict[str, Any]:
    runner = alignment_runner or run_dimer_usalign_alignment
    alignment_jobs = normalize_worker_count(alignment_jobs)

    total_observations = sum(len(members) for _, members in signature_groups)
    multi_member = sum(1 for _, m in signature_groups if len(m) > 1)
    if log_summary:
        LOGGER.info(
            "Refining %d dimer signature clusters (%d observations, %d multi-member, %d alignment workers)",
            len(signature_groups),
            total_observations,
            multi_member,
            alignment_jobs,
        )
    signature_iter = list(
        tqdm(signature_groups, desc="Refining dimer clusters", unit="sig-group")
        if show_progress
        else signature_groups
    )
    refined = refine_signature_groups_three_phase(
        signature_iter,
        extracted_structures,
        member_id=lambda item: item.dimer_observation_id,
        structural_sort_key=lambda item: item.structural_sort_key(),
        alignment_row=lambda signature_cluster_id, result: {
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
        },
        alignment_failure_warning=lambda signature_cluster_id, representative, candidate, exc: {
            "warning_code": "dimer_usalign_failed",
            "signature_cluster_id": signature_cluster_id,
            "representative_dimer_observation_id": representative.dimer_observation_id,
            "candidate_dimer_observation_id": candidate.dimer_observation_id,
            "error": str(exc),
        },
        unavailable_warning=lambda signature_cluster_id, member: {
            "warning_code": "dimer_structure_unavailable_singleton_cluster",
            "signature_cluster_id": signature_cluster_id,
            "dimer_observation_id": member.dimer_observation_id,
        },
        runner=runner,
        alignment_jobs=alignment_jobs,
        usalign_executable=usalign_executable,
        tm_score_threshold=tm_score_threshold,
        can_skip_alignment=lambda a, b: (
            a.source_path == b.source_path
            and {a.label_asym_id_1, a.label_asym_id_2} == {b.label_asym_id_1, b.label_asym_id_2}
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
        cluster_id = _dimer_cluster_id(signature_cluster_id, local_cluster_counts[signature_cluster_id])
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
    if log_summary:
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
    include_structure_assignments: bool = True,
) -> dict[str, Any]:
    """Build dimer clusters from monomer assignments and optional structure refinement."""

    monomer_assignments = load_monomer_cluster_assignments(
        clustering_outdir,
        include_structure=include_structure_assignments,
    )
    observations = collect_dimer_observations(
        case_dirs, monomer_assignments,
        cif_files_directory=cif_files_directory,
        prep_dir=prep_dir,
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
        "num_dimer_structure_extraction_candidates": 0,
        "num_singleton_dimer_observations_skipped_structure_extraction": 0,
    }
    extracted_structures: dict[str, ExtractedDimerStructure] = {}
    if structure_refinement_mode == "greedy":
        refinement_observations = [
            observation
            for _, members in signature_groups
            if len(members) > 1
            for observation in members
        ]
        extraction_manifest["num_dimer_structure_extraction_candidates"] = len(refinement_observations)
        extraction_manifest["num_singleton_dimer_observations_skipped_structure_extraction"] = (
            len(observations) - len(refinement_observations)
        )
        LOGGER.info(
            "Dimer structure refinement will extract %d/%d observations; %d singleton observations need no USalign",
            len(refinement_observations),
            len(observations),
            extraction_manifest["num_singleton_dimer_observations_skipped_structure_extraction"],
        )

    if structure_refinement_mode == "greedy":
        structure_rows: list[dict[str, Any]] = []
        structure_failure_rows: list[dict[str, Any]] = []

        def _record_extraction_result(
            structures: dict[str, ExtractedDimerStructure],
            extract_manifest: dict[str, Any],
            group_outdir: Path,
        ) -> None:
            extraction_manifest["num_extracted_dimer_structures"] += extract_manifest.get(
                "num_extracted_dimer_structures",
                0,
            )
            extraction_manifest["num_failed_dimer_structure_extractions"] += extract_manifest.get(
                "num_failed_dimer_structure_extractions",
                0,
            )
            structure_rows.extend(item.to_dict() for item in structures.values())
            failures_path = group_outdir / "dimer_structure_extraction_failures.jsonl"
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
                        extract_dimer_structures,
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

        refined = refine_dimer_signature_clusters(
            signature_groups,
            extracted_structures,
            tm_score_threshold=dimer_tm_score_threshold,
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

        def _dimer_cluster_sort_key(cluster_id: str) -> tuple[int, int]:
            _, sig_idx, local_idx = cluster_id.split("_")
            return (int(sig_idx), int(local_idx))

        membership_rows.sort(key=lambda row: (_dimer_cluster_sort_key(row["dimer_cluster_id"]), row["dimer_observation_id"]))
        representative_rows.sort(key=lambda row: _dimer_cluster_sort_key(row["dimer_cluster_id"]))
        signature_rows.sort(key=lambda row: _dimer_cluster_sort_key(row["dimer_cluster_id"]))

        structures_dir = outdir / "structures"
        dump_jsonl(structures_dir / "dimer_structure_extraction_failures.jsonl", structure_failure_rows)
        dump_jsonl(structures_dir / "dimer_structures.jsonl", structure_rows)
        dump_json(
            structures_dir / "dimer_structure_manifest.json",
            {
                "num_dimer_observations": extraction_manifest["num_dimer_structure_extraction_candidates"],
                "num_extracted_dimer_structures": extraction_manifest["num_extracted_dimer_structures"],
                "num_failed_dimer_structure_extractions": extraction_manifest[
                    "num_failed_dimer_structure_extractions"
                ],
                "extraction_jobs": normalize_worker_count(alignment_jobs),
                "num_signature_groups_pipelined": len(multi_member_groups),
            },
            indent=2,
        )
        manifest = {
            "num_dimer_observations": len(observations),
            "num_dimer_clusters": refined_manifest["num_dimer_clusters"],
            "num_signature_clusters": refined_manifest["num_signature_clusters"],
            "num_unique_signatures": len(grouped),
            "num_monomer_assignments_loaded": len(monomer_assignments),
            "num_observations_with_unclustered_member": sum(
                1
                for observation in observations
                if "unclustered" in {observation.cluster_source_1, observation.cluster_source_2}
            ),
            "num_alignment_runs": refined_manifest["num_alignment_runs"],
            "num_alignment_failures": refined_manifest["num_alignment_failures"],
            "num_signature_clusters_split": refined_manifest["num_signature_clusters_split"],
            "dimer_tm_score_threshold": dimer_tm_score_threshold,
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
            cluster_id = _dimer_cluster_id(signature_cluster_id)
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
