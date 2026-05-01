from __future__ import annotations

from difflib import SequenceMatcher
import json
import logging
import math
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from biotite.structure import AtomArray, get_residues
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFBlock, CIFCategory, get_structure

from cif_parse.clustering.monomers import MonomerSample
from cif_parse.clustering.parallel import AlignmentTask, normalize_worker_count, run_alignment_tasks
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.io import read_cif_file
from cif_parse.utils.atom_filters import atom_array_filter_counts, filter_atom_array_for_analysis


LOGGER = logging.getLogger(__name__)

ALIGNED_LENGTH_RE = re.compile(
    r"Aligned length=\s*(?P<aligned_length>\d+),\s*RMSD=\s*(?P<rmsd>[0-9.]+),\s*Seq_ID=.*"
)
TM_SCORE_RE = re.compile(r"TM-score=\s*(?P<tm_score>[0-9.]+)\s+\(normalized by length of Structure_(?P<index>[12])")
METHOD_PRIORITY = {
    "x-ray diffraction": 0,
    "electron microscopy": 1,
    "solution nmr": 2,
    "solid-state nmr": 2,
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", ".", "?", "None", "none", "nan", "NaN"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _category_column_values(block: CIFBlock, category_name: str, column_name: str) -> list[str]:
    if category_name not in block:
        return []
    category = block[category_name]
    if not isinstance(category, CIFCategory) or column_name not in category:
        return []
    return [str(value) for value in category[column_name].as_array().tolist()]


def _pick_primary_method(methods: list[str]) -> str | None:
    normalized = [method.strip() for method in methods if method and method.strip() not in {".", "?"}]
    if not normalized:
        return None
    sorted_methods = sorted(
        normalized,
        key=lambda method: (METHOD_PRIORITY.get(method.lower(), 99), method.lower()),
    )
    return sorted_methods[0]


def _method_priority(method: str | None) -> int:
    if method is None:
        return 99
    return METHOD_PRIORITY.get(method.lower(), 99)


def _resolved_residue_count_from_segments(segments: Iterable[dict[str, Any]]) -> int:
    total = 0
    for segment in segments:
        start = _safe_int(segment.get("label_seq_start"))
        end = _safe_int(segment.get("label_seq_end"))
        if start is None or end is None or end < start:
            continue
        total += end - start + 1
    return total


def _coerce_chain_id_for_pdb(atom_array: AtomArray) -> AtomArray:
    if atom_array.array_length() == 0:
        return atom_array
    copied = atom_array.copy()
    copied.chain_id[:] = "A"
    return copied


@dataclass(slots=True)
class EntryQualityMetadata:
    pdb_id: str
    source_path: str
    experimental_methods: list[str]
    primary_method: str | None
    method_priority: int
    resolution: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExtractedMonomerStructure:
    monomer_id: str
    pdb_id: str
    label_asym_id: str
    auth_asym_id: str | None
    source_path: str
    extracted_pdb_path: str
    sequence_length: int
    residue_count: int
    resolved_residue_count: int
    resolved_fraction: float
    atom_count: int
    filter_counts: dict[str, int]
    quality: EntryQualityMetadata

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def quality_sort_key(self) -> tuple[float, float, float, str]:
        if self.quality is None:
            return (4.0, math.inf, -float(self.resolved_fraction), self.monomer_id)
        resolution = self.quality.resolution if self.quality.resolution is not None else math.inf
        return (
            float(self.quality.method_priority),
            float(resolution),
            -float(self.resolved_fraction),
            self.monomer_id,
        )


@dataclass(slots=True)
class USalignAlignmentResult:
    query_monomer_id: str
    target_monomer_id: str
    aligned_length: int
    rmsd: float
    tm_score_query: float
    tm_score_target: float
    min_tm_score: float
    max_tm_score: float
    shorter_length_coverage: float
    meets_tm_threshold: bool
    meets_coverage_threshold: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_entry_quality_metadata(source_path: str | Path, *, pdb_id: str | None = None) -> EntryQualityMetadata:
    """Extract a minimal, clustering-local quality summary from mmCIF metadata."""

    cif_file = read_cif_file(source_path)
    block = cif_file.block
    methods = _category_column_values(block, "exptl", "method")
    primary_method = _pick_primary_method(methods)
    resolution_candidates = [
        *(_safe_float(value) for value in _category_column_values(block, "refine", "ls_d_res_high")),
        *(_safe_float(value) for value in _category_column_values(block, "em_3d_reconstruction", "resolution")),
        *(_safe_float(value) for value in _category_column_values(block, "reflns", "d_resolution_high")),
    ]
    resolutions = sorted(value for value in resolution_candidates if value is not None)
    return EntryQualityMetadata(
        pdb_id=str(pdb_id or Path(source_path).stem),
        source_path=str(source_path),
        experimental_methods=sorted({method for method in methods if method not in {"", ".", "?"}}),
        primary_method=primary_method,
        method_priority=_method_priority(primary_method),
        resolution=resolutions[0] if resolutions else None,
    )


def extract_protein_monomer_structure(
    monomer: MonomerSample,
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    quality_metadata: EntryQualityMetadata | None = None,
    atom_array: AtomArray | None = None,
) -> ExtractedMonomerStructure:
    """Extract one canonical monomer chain from source mmCIF and write a single-chain PDB."""

    full_atom_array = atom_array
    if full_atom_array is None:
        cif_file = read_cif_file(monomer.source_path)
        full_atom_array = get_structure(
            cif_file,
            model=model,
            use_author_fields=False,
        )
    chain_mask = full_atom_array.chain_id == monomer.label_asym_id
    if hasattr(full_atom_array, "hetero"):
        chain_mask &= ~full_atom_array.hetero
    chain_atoms = full_atom_array[chain_mask]
    if chain_atoms.array_length() == 0:
        raise ValueError(f"No polymer atoms found for monomer {monomer.monomer_id}")

    chain_atoms, filter_counts = filter_atom_array_for_analysis(
        chain_atoms,
        drop_hydrogens=drop_hydrogens,
        drop_nonfinite=True,
    )
    if chain_atoms.array_length() == 0:
        raise ValueError(f"No analyzable atoms left for monomer {monomer.monomer_id}")

    chain_atoms = _coerce_chain_id_for_pdb(chain_atoms)
    _, residue_names = get_residues(chain_atoms)
    resolved_residue_count = int(len(residue_names))
    sequence_length = max(int(monomer.length or len(monomer.sequence)), len(monomer.sequence))
    if resolved_residue_count <= 2:
        raise ValueError(
            f"Resolved residue count {resolved_residue_count} is too short for USalign: {monomer.monomer_id}"
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdb_path = outdir / f"{monomer.pdb_id}_{monomer.label_asym_id}.pdb"
    pdb_file = PDBFile()
    pdb_file.set_structure(chain_atoms)
    pdb_file.write(pdb_path)

    quality = quality_metadata or read_entry_quality_metadata(monomer.source_path, pdb_id=monomer.pdb_id)
    resolved_from_segments = _resolved_residue_count_from_segments(monomer.parsed_coordinate_segments)
    denominator = sequence_length if sequence_length > 0 else max(1, resolved_residue_count)
    resolved_fraction = min(
        1.0,
        float(max(resolved_residue_count, resolved_from_segments)) / float(denominator),
    )
    return ExtractedMonomerStructure(
        monomer_id=monomer.monomer_id,
        pdb_id=monomer.pdb_id,
        label_asym_id=monomer.label_asym_id,
        auth_asym_id=monomer.auth_asym_id,
        source_path=monomer.source_path,
        extracted_pdb_path=str(pdb_path),
        sequence_length=sequence_length,
        residue_count=int(monomer.residue_count),
        resolved_residue_count=resolved_residue_count,
        resolved_fraction=resolved_fraction,
        atom_count=int(chain_atoms.array_length()),
        filter_counts=atom_array_filter_counts(filter_counts),
        quality=quality,
    )


def parse_usalign_output(
    stdout: str,
    *,
    query_monomer_id: str,
    target_monomer_id: str,
    query_length: int,
    target_length: int,
    tm_score_threshold: float,
    min_alignment_coverage_ratio: float,
) -> USalignAlignmentResult:
    """Parse `USalign` stdout into a normalized alignment result."""

    aligned_match = ALIGNED_LENGTH_RE.search(stdout)
    if aligned_match is None:
        raise ValueError("USalign output does not contain aligned length / RMSD summary")

    tm_matches = TM_SCORE_RE.findall(stdout)
    if len(tm_matches) < 2:
        raise ValueError("USalign output does not contain both TM-score lines")

    tm_by_index = {index: float(tm_score) for tm_score, index in tm_matches}
    aligned_length = int(aligned_match.group("aligned_length"))
    rmsd = float(aligned_match.group("rmsd"))
    tm_score_query = float(tm_by_index["1"])
    tm_score_target = float(tm_by_index["2"])
    min_tm_score = min(tm_score_query, tm_score_target)
    max_tm_score = max(tm_score_query, tm_score_target)
    shorter_length = max(1, min(int(query_length), int(target_length)))
    shorter_length_coverage = float(aligned_length) / float(shorter_length)
    return USalignAlignmentResult(
        query_monomer_id=query_monomer_id,
        target_monomer_id=target_monomer_id,
        aligned_length=aligned_length,
        rmsd=rmsd,
        tm_score_query=tm_score_query,
        tm_score_target=tm_score_target,
        min_tm_score=min_tm_score,
        max_tm_score=max_tm_score,
        shorter_length_coverage=shorter_length_coverage,
        meets_tm_threshold=max_tm_score >= tm_score_threshold,
        meets_coverage_threshold=shorter_length_coverage >= min_alignment_coverage_ratio,
    )


def run_usalign_alignment(
    query: ExtractedMonomerStructure,
    target: ExtractedMonomerStructure,
    *,
    usalign_executable: str = "USalign",
    mol: str = "prot",
    tm_score_threshold: float = 0.50,
    min_alignment_coverage_ratio: float = 0.80,
) -> USalignAlignmentResult:
    """Run USalign on one monomer pair and parse the result."""

    if shutil.which(usalign_executable) is None:
        raise FileNotFoundError(f"{usalign_executable} executable not found in PATH")

    command = [
        usalign_executable,
        query.extracted_pdb_path,
        target.extracted_pdb_path,
        "-mol",
        mol,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_usalign_output(
        completed.stdout,
        query_monomer_id=query.monomer_id,
        target_monomer_id=target.monomer_id,
        query_length=query.sequence_length,
        target_length=target.sequence_length,
        tm_score_threshold=tm_score_threshold,
        min_alignment_coverage_ratio=min_alignment_coverage_ratio,
    )


def _group_protein_sequence_membership(
    membership_rows: Iterable[dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in membership_rows:
        if row.get("polymer_class") != "protein":
            continue
        cluster_id = str(row.get("cluster_id", ""))
        member_id = str(row.get("member_monomer_id", ""))
        if not cluster_id or not member_id:
            continue
        grouped.setdefault(cluster_id, []).append(member_id)
    return {
        cluster_id: sorted(set(member_ids))
        for cluster_id, member_ids in sorted(grouped.items())
    }


def _sequence_similarity_ratio(sequence_a: str, sequence_b: str) -> float:
    if sequence_a == sequence_b:
        return 1.0
    if not sequence_a or not sequence_b:
        return 0.0
    return float(SequenceMatcher(a=sequence_a, b=sequence_b).ratio())


def _protein_structure_cluster_id(sequence_cluster_id: str, local_cluster_index: int) -> str:
    if not sequence_cluster_id.startswith("prot_"):
        return f"{sequence_cluster_id}_{local_cluster_index}"
    sequence_index = sequence_cluster_id.removeprefix("prot_")
    return f"prot_{sequence_index}_{local_cluster_index}"


def extract_protein_monomer_structures(
    monomers: Iterable[MonomerSample],
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    extraction_jobs: int = 1,
    prep_dir: str | Path | None = None,
) -> tuple[dict[str, ExtractedMonomerStructure], dict[str, Any]]:
    """Extract analyzable PDB files for all protein monomers.

    *extraction_jobs* controls how many monomers are extracted concurrently.
    When *prep_dir* is provided, pre-parsed AtomArrays are fetched from the
    binary index instead of re-reading the original mmCIF files.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Load cif_cache index when prep_dir is available
    cif_idx: dict | None = None
    if prep_dir:
        from cif_parse.clustering.prep import load_cif_coords_index, load_cif_from_prep
        cif_idx = load_cif_coords_index(prep_dir)

    structures: dict[str, ExtractedMonomerStructure] = {}
    failures: list[dict[str, str]] = []

    def _load_cif_cache(_prep_dir, _source_path, _assembly_id, _idx):
        from cif_parse.clustering.prep import load_cif_from_prep as _lcfp
        return _lcfp(_prep_dir, _source_path, _assembly_id, index=_idx)

    def _load_atom_array_for_monomer(_source_path, _observed_assembly_ids):
        """Try asymmetric unit first, then each observed assembly."""
        if cif_idx is None:
            return None
        cached = _load_cif_cache(prep_dir, _source_path, None, cif_idx)
        if cached is not None and cached.get("atom_array") is not None:
            return cached["atom_array"]
        for aid in (_observed_assembly_ids or []):
            cached = _load_cif_cache(prep_dir, _source_path, str(aid), cif_idx)
            if cached is not None and cached.get("atom_array") is not None:
                return cached["atom_array"]
        return None

    protein_monomers = sorted(
        (monomer for monomer in monomers if monomer.polymer_class == "protein"),
        key=lambda item: item.monomer_id,
    )

    extraction_jobs = normalize_worker_count(extraction_jobs)
    if extraction_jobs <= 1 or len(protein_monomers) <= 1:
        quality_cache: dict[str, EntryQualityMetadata] = {}
        atom_array_cache: dict[str, AtomArray] = {}
        for monomer in tqdm(protein_monomers, desc="Extracting monomer structures", unit="monomer"):
            if monomer.source_path not in atom_array_cache:
                _prep_atoms = _load_atom_array_for_monomer(monomer.source_path, monomer.observed_assembly_ids)
                if _prep_atoms is not None:
                    atom_array_cache[monomer.source_path] = _prep_atoms
            if monomer.source_path not in quality_cache:
                quality_cache[monomer.source_path] = read_entry_quality_metadata(
                    monomer.source_path,
                    pdb_id=monomer.pdb_id,
                )
            if monomer.source_path not in atom_array_cache:
                cif_file = read_cif_file(monomer.source_path)
                atom_array_cache[monomer.source_path] = get_structure(
                    cif_file,
                    model=model,
                    use_author_fields=False,
                )
            try:
                structures[monomer.monomer_id] = extract_protein_monomer_structure(
                    monomer,
                    outdir=outdir,
                    model=model,
                    drop_hydrogens=drop_hydrogens,
                    quality_metadata=quality_cache[monomer.source_path],
                    atom_array=atom_array_cache[monomer.source_path],
                )
            except Exception as exc:
                LOGGER.warning("Failed to extract protein monomer %s: %s", monomer.monomer_id, exc)
                failures.append({"monomer_id": monomer.monomer_id, "error": str(exc)})
    else:
        import threading

        quality_cache: dict[str, EntryQualityMetadata] = {}
        atom_array_cache: dict[str, AtomArray] = {}
        cache_lock = threading.Lock()

        def _extract_one(monomer: MonomerSample) -> ExtractedMonomerStructure | None:
            with cache_lock:
                if monomer.source_path not in atom_array_cache:
                    _prep_atoms = _load_atom_array_for_monomer(monomer.source_path, monomer.observed_assembly_ids)
                    if _prep_atoms is not None:
                        atom_array_cache[monomer.source_path] = _prep_atoms
                if monomer.source_path not in quality_cache:
                    quality_cache[monomer.source_path] = read_entry_quality_metadata(
                        monomer.source_path,
                        pdb_id=monomer.pdb_id,
                    )
                if monomer.source_path not in atom_array_cache:
                    cif_file = read_cif_file(monomer.source_path)
                    atom_array_cache[monomer.source_path] = get_structure(
                        cif_file,
                        model=model,
                        use_author_fields=False,
                    )
            return extract_protein_monomer_structure(
                monomer,
                outdir=outdir,
                model=model,
                drop_hydrogens=drop_hydrogens,
                quality_metadata=quality_cache[monomer.source_path],
                atom_array=atom_array_cache[monomer.source_path],
            )

        with ThreadPoolExecutor(max_workers=min(extraction_jobs, len(protein_monomers))) as executor:
            future_to_monomer = {
                executor.submit(_extract_one, monomer): monomer
                for monomer in protein_monomers
            }
            for future in as_completed(future_to_monomer):
                monomer = future_to_monomer[future]
                try:
                    extracted = future.result()
                    if extracted is not None:
                        structures[monomer.monomer_id] = extracted
                except Exception as exc:
                    LOGGER.warning("Failed to extract protein monomer %s: %s", monomer.monomer_id, exc)
                    failures.append({"monomer_id": monomer.monomer_id, "error": str(exc)})

    dump_jsonl(outdir / "protein_structure_extraction_failures.jsonl", failures)
    dump_jsonl(outdir / "protein_structures.jsonl", [item.to_dict() for item in structures.values()])
    manifest = {
        "num_protein_monomers": len(protein_monomers),
        "num_extracted_protein_structures": len(structures),
        "num_failed_protein_structure_extractions": len(failures),
        "extraction_jobs": extraction_jobs,
    }
    dump_json(outdir / "protein_structure_manifest.json", manifest, indent=2)
    return structures, manifest


def greedy_cluster_protein_structures(
    monomers: Iterable[MonomerSample],
    membership_rows: Iterable[dict[str, Any]],
    extracted_structures: dict[str, ExtractedMonomerStructure] | None = None,
    *,
    outdir: str | Path,
    tm_score_threshold: float = 0.50,
    min_alignment_coverage_ratio: float = 0.80,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    sequence_cluster_jobs: int = 1,
    pairwise_alignment_jobs: int = 1,
    extract_fn: Callable[[MonomerSample], ExtractedMonomerStructure | None] | None = None,
) -> dict[str, Any]:
    """Perform quality-directed greedy structural clustering inside protein sequence buckets.

    When *extract_fn* is provided, monomer structures are extracted on-the-fly
    per sequence cluster, allowing extraction and USalign to overlap across
    different clusters (producer-consumer pipeline).  Otherwise *extracted_structures*
    must contain all pre-extracted structures.
    """

    monomer_index = {monomer.monomer_id: monomer for monomer in monomers}
    sequence_groups = _group_protein_sequence_membership(membership_rows)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runner = alignment_runner or run_usalign_alignment

    if extracted_structures is None:
        extracted_structures = {}

    available_structures = len(extracted_structures)
    pipeline_mode = extract_fn is not None and available_structures == 0
    LOGGER.info(
        "Clustering protein monomers: %d sequence groups, %d pre-extracted structures, "
        "%d seq-cluster workers, %d pairwise workers%s",
        len(sequence_groups),
        available_structures,
        sequence_cluster_jobs,
        pairwise_alignment_jobs,
        " (pipelined extraction+clustering)" if pipeline_mode else "",
    )

    alignment_rows: list[dict[str, Any]] = []
    membership_out_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    total_struct_clusters = 0
    total_alignments = 0
    total_alignment_failures = 0
    total_sequence_fallback_assignments = 0
    total_all_failure_cluster_collapses = 0

    total_monomers_to_extract = 0
    total_extraction_failures = 0

    pairwise_alignment_jobs = normalize_worker_count(pairwise_alignment_jobs)

    def process_sequence_group(sequence_group: tuple[str, list[str]]) -> dict[str, Any]:
        sequence_cluster_id, member_ids = sequence_group
        local_alignment_cache: dict[tuple[str, str], USalignAlignmentResult] = {}
        local_alignment_rows: list[dict[str, Any]] = []
        local_membership_rows: list[dict[str, Any]] = []
        local_representative_rows: list[dict[str, Any]] = []
        local_warning_rows: list[dict[str, Any]] = []
        local_struct_clusters = 0
        local_alignments = 0
        local_alignment_failures = 0
        local_sequence_fallback_assignments = 0
        local_all_failure_cluster_collapses = 0
        local_extracted = 0
        local_extraction_failures = 0

        # On-the-fly extraction when no pre-extracted structures available
        if extract_fn is not None:
            for member_id in member_ids:
                if member_id in extracted_structures:
                    continue
                monomer = monomer_index.get(member_id)
                if monomer is None:
                    continue
                try:
                    local_extracted += 1
                    ext = extract_fn(monomer)
                    if ext is not None:
                        extracted_structures[member_id] = ext
                    else:
                        local_extraction_failures += 1
                except Exception:
                    local_extraction_failures += 1

        candidates = [
            extracted_structures[member_id]
            for member_id in member_ids
            if member_id in extracted_structures
        ]
        if not candidates:
            structure_cluster_id = _protein_structure_cluster_id(sequence_cluster_id, 1)
            representative_monomer_id = member_ids[0]
            local_struct_clusters += 1
            local_sequence_fallback_assignments += len(member_ids)
            local_warning_rows.append(
                {
                    "warning_code": "no_extracted_structures_in_sequence_cluster",
                    "sequence_cluster_id": sequence_cluster_id,
                    "structure_cluster_id": structure_cluster_id,
                    "representative_monomer_id": representative_monomer_id,
                    "member_monomer_ids": member_ids,
                }
            )
            for member_id in member_ids:
                local_membership_rows.append(
                    {
                        "polymer_class": "protein",
                        "sequence_cluster_id": sequence_cluster_id,
                        "structure_cluster_id": structure_cluster_id,
                        "representative_monomer_id": representative_monomer_id,
                        "member_monomer_id": member_id,
                        "assignment_reason": "structure_extraction_failure_sequence_cluster_collapse",
                        "tm_score_min": "",
                        "tm_score_max": "",
                        "tm_score_for_clustering": "",
                        "alignment_coverage_shorter": "",
                    }
                )
            local_representative_rows.append(
                {
                    "sequence_cluster_id": sequence_cluster_id,
                    "structure_cluster_id": structure_cluster_id,
                    "representative_monomer_id": representative_monomer_id,
                    "num_members": len(member_ids),
                    "pdb_id": monomer_index[representative_monomer_id].pdb_id,
                    "label_asym_id": monomer_index[representative_monomer_id].label_asym_id,
                    "auth_asym_id": monomer_index[representative_monomer_id].auth_asym_id or "",
                    "resolved_fraction": "",
                    "primary_method": "",
                    "resolution": "",
                }
            )
            return {
                "alignment_rows": local_alignment_rows,
                "membership_rows": local_membership_rows,
                "representative_rows": local_representative_rows,
                "warning_rows": local_warning_rows,
                "num_structure_clusters": local_struct_clusters,
                "num_alignment_runs": local_alignments,
                "num_alignment_failures": local_alignment_failures,
                "num_sequence_fallback_assignments": local_sequence_fallback_assignments,
                "num_all_failure_sequence_cluster_collapses": local_all_failure_cluster_collapses,
            }

        pending = sorted(candidates, key=lambda item: item.quality_sort_key())
        cluster_states: list[dict[str, Any]] = []
        failed_candidates: list[dict[str, Any]] = []
        alignment_success_count = 0
        sequence_member_ids = {candidate.monomer_id for candidate in candidates}
        missing_structure_member_ids = sorted(set(member_ids) - sequence_member_ids)
        local_cluster_index = 0
        while pending:
            representative = pending[0]
            local_cluster_index += 1
            local_struct_clusters += 1
            structure_cluster_id = _protein_structure_cluster_id(
                sequence_cluster_id,
                local_cluster_index,
            )
            assigned = [representative]
            remaining: list[ExtractedMonomerStructure] = []

            alignment_tasks: list[AlignmentTask] = []
            for candidate in pending[1:]:
                pair_key = tuple(sorted((representative.monomer_id, candidate.monomer_id)))
                if pair_key not in local_alignment_cache:
                    alignment_tasks.append(
                        AlignmentTask(
                            key=pair_key,
                            query=representative,
                            target=candidate,
                            context={"candidate": candidate},
                        )
                    )

            successes, failures = run_alignment_tasks(
                alignment_tasks,
                runner,
                max_workers=pairwise_alignment_jobs,
                usalign_executable=usalign_executable,
                tm_score_threshold=tm_score_threshold,
                min_alignment_coverage_ratio=min_alignment_coverage_ratio,
            )
            failure_by_candidate_id: dict[str, Exception] = {
                task.context["candidate"].monomer_id: exc for task, exc in failures
            }
            for task, result in successes:
                local_alignment_cache[task.key] = result
            success_keys = {task.key for task, _ in successes}

            for candidate in pending[1:]:
                pair_key = tuple(sorted((representative.monomer_id, candidate.monomer_id)))
                if candidate.monomer_id in failure_by_candidate_id:
                    exc = failure_by_candidate_id[candidate.monomer_id]
                    local_alignment_failures += 1
                    failed_candidates.append(
                        {
                            "candidate": candidate,
                            "failed_against_representative": representative.monomer_id,
                            "error": str(exc),
                        }
                    )
                    local_warning_rows.append(
                        {
                            "warning_code": "usalign_failed_sequence_fallback",
                            "sequence_cluster_id": sequence_cluster_id,
                            "candidate_monomer_id": candidate.monomer_id,
                            "representative_monomer_id": representative.monomer_id,
                            "error": str(exc),
                        }
                    )
                    continue
                if pair_key in local_alignment_cache:
                    result = local_alignment_cache[pair_key]
                    if pair_key in success_keys:
                        local_alignment_rows.append(
                            {
                                "sequence_cluster_id": sequence_cluster_id,
                                "query_monomer_id": result.query_monomer_id,
                                "target_monomer_id": result.target_monomer_id,
                                "aligned_length": result.aligned_length,
                                "rmsd": result.rmsd,
                                "tm_score_query": result.tm_score_query,
                                "tm_score_target": result.tm_score_target,
                                "tm_score_min": result.min_tm_score,
                                "tm_score_max": result.max_tm_score,
                                "tm_score_for_clustering": result.max_tm_score,
                                "alignment_coverage_shorter": result.shorter_length_coverage,
                                "meets_tm_threshold": result.meets_tm_threshold,
                                "meets_coverage_threshold": result.meets_coverage_threshold,
                            }
                        )
                        local_alignments += 1
                        alignment_success_count += 1
                    if result.meets_tm_threshold and result.meets_coverage_threshold:
                        assigned.append(candidate)
                    else:
                        remaining.append(candidate)
            cluster_states.append(
                {
                    "structure_cluster_id": structure_cluster_id,
                    "representative": representative,
                    "members": assigned,
                    "fallback_member_ids": [],
                }
            )
            pending = remaining

        if failed_candidates and alignment_success_count == 0:
            local_all_failure_cluster_collapses += 1
            local_struct_clusters -= len(cluster_states)
            local_struct_clusters += 1
            representative = min(candidates, key=lambda item: item.quality_sort_key())
            structure_cluster_id = _protein_structure_cluster_id(sequence_cluster_id, 1)
            cluster_states = [
                {
                    "structure_cluster_id": structure_cluster_id,
                    "representative": representative,
                    "members": sorted(candidates, key=lambda item: item.quality_sort_key()),
                    "fallback_member_ids": [],
                }
            ]
            local_warning_rows.append(
                {
                    "warning_code": "all_usalign_failed_sequence_cluster_collapsed",
                    "sequence_cluster_id": sequence_cluster_id,
                    "structure_cluster_id": structure_cluster_id,
                    "member_monomer_ids": sorted(sequence_member_ids),
                }
            )
        elif failed_candidates:
            for failed_item in failed_candidates:
                candidate = failed_item["candidate"]
                candidate_sequence = monomer_index[candidate.monomer_id].sequence
                target_cluster = max(
                    cluster_states,
                    key=lambda state: (
                        _sequence_similarity_ratio(
                            candidate_sequence,
                            monomer_index[state["representative"].monomer_id].sequence,
                        ),
                        -state["representative"].quality_sort_key()[0],
                        -state["representative"].resolved_fraction,
                        state["representative"].monomer_id,
                    ),
                )
                target_cluster["members"].append(candidate)
                local_sequence_fallback_assignments += 1
                local_warning_rows.append(
                    {
                        "warning_code": "usalign_failed_assigned_by_sequence_similarity",
                        "sequence_cluster_id": sequence_cluster_id,
                        "candidate_monomer_id": candidate.monomer_id,
                        "assigned_structure_cluster_id": target_cluster["structure_cluster_id"],
                        "assigned_representative_monomer_id": target_cluster["representative"].monomer_id,
                        "failed_against_representative": failed_item["failed_against_representative"],
                        "error": failed_item["error"],
                        "sequence_similarity_to_assigned_representative": _sequence_similarity_ratio(
                            candidate_sequence,
                            monomer_index[target_cluster["representative"].monomer_id].sequence,
                        ),
                    }
                )

        if missing_structure_member_ids:
            for member_id in missing_structure_member_ids:
                member_sequence = monomer_index[member_id].sequence
                target_cluster = max(
                    cluster_states,
                    key=lambda state: (
                        _sequence_similarity_ratio(
                            member_sequence,
                            monomer_index[state["representative"].monomer_id].sequence,
                        ),
                        -state["representative"].quality_sort_key()[0],
                        -state["representative"].resolved_fraction,
                        state["representative"].monomer_id,
                    ),
                )
                target_cluster["fallback_member_ids"].append(member_id)
                local_sequence_fallback_assignments += 1
                local_warning_rows.append(
                    {
                        "warning_code": "structure_extraction_failed_assigned_by_sequence_similarity",
                        "sequence_cluster_id": sequence_cluster_id,
                        "member_monomer_id": member_id,
                        "assigned_structure_cluster_id": target_cluster["structure_cluster_id"],
                        "assigned_representative_monomer_id": target_cluster["representative"].monomer_id,
                        "sequence_similarity_to_assigned_representative": _sequence_similarity_ratio(
                            member_sequence,
                            monomer_index[target_cluster["representative"].monomer_id].sequence,
                        ),
                    }
                )

        for cluster_state in cluster_states:
            representative = cluster_state["representative"]
            structure_cluster_id = cluster_state["structure_cluster_id"]
            ordered_members = sorted(
                cluster_state["members"],
                key=lambda item: (item.monomer_id != representative.monomer_id, item.monomer_id),
            )
            for member in ordered_members:
                assignment_reason = "representative"
                tm_score_min: float | str = ""
                tm_score_max: float | str = ""
                tm_score_for_clustering: float | str = ""
                alignment_coverage_shorter: float | str = ""
                if member.monomer_id != representative.monomer_id:
                    pair_key = tuple(sorted((representative.monomer_id, member.monomer_id)))
                    if pair_key in local_alignment_cache:
                        result = local_alignment_cache[pair_key]
                        assignment_reason = "representative_alignment"
                        tm_score_min = result.min_tm_score
                        tm_score_max = result.max_tm_score
                        tm_score_for_clustering = result.max_tm_score
                        alignment_coverage_shorter = result.shorter_length_coverage
                    elif failed_candidates:
                        assignment_reason = "tm_failure_sequence_fallback"
                local_membership_rows.append(
                    {
                        "polymer_class": "protein",
                        "sequence_cluster_id": sequence_cluster_id,
                        "structure_cluster_id": structure_cluster_id,
                        "representative_monomer_id": representative.monomer_id,
                        "member_monomer_id": member.monomer_id,
                        "assignment_reason": assignment_reason,
                        "tm_score_min": tm_score_min,
                        "tm_score_max": tm_score_max,
                        "tm_score_for_clustering": tm_score_for_clustering,
                        "alignment_coverage_shorter": alignment_coverage_shorter,
                    }
                )

            for member_id in sorted(cluster_state["fallback_member_ids"]):
                local_membership_rows.append(
                    {
                        "polymer_class": "protein",
                        "sequence_cluster_id": sequence_cluster_id,
                        "structure_cluster_id": structure_cluster_id,
                        "representative_monomer_id": representative.monomer_id,
                        "member_monomer_id": member_id,
                        "assignment_reason": "structure_extraction_failure_sequence_fallback",
                        "tm_score_min": "",
                        "tm_score_max": "",
                        "tm_score_for_clustering": "",
                        "alignment_coverage_shorter": "",
                    }
                )

            local_representative_rows.append(
                {
                    "sequence_cluster_id": sequence_cluster_id,
                    "structure_cluster_id": structure_cluster_id,
                    "representative_monomer_id": representative.monomer_id,
                    "num_members": len(cluster_state["members"]) + len(cluster_state["fallback_member_ids"]),
                    "pdb_id": representative.pdb_id,
                    "label_asym_id": representative.label_asym_id,
                    "auth_asym_id": representative.auth_asym_id or "",
                    "resolved_fraction": representative.resolved_fraction,
                    "primary_method": representative.quality.primary_method or "",
                    "resolution": (
                        representative.quality.resolution
                        if representative.quality.resolution is not None
                        else ""
                    ),
                }
            )

        return {
            "alignment_rows": local_alignment_rows,
            "membership_rows": local_membership_rows,
            "representative_rows": local_representative_rows,
            "warning_rows": local_warning_rows,
            "num_structure_clusters": local_struct_clusters,
            "num_alignment_runs": local_alignments,
            "num_alignment_failures": local_alignment_failures,
            "num_sequence_fallback_assignments": local_sequence_fallback_assignments,
            "num_all_failure_sequence_cluster_collapses": local_all_failure_cluster_collapses,
            "num_extracted": local_extracted,
            "num_extraction_failures": local_extraction_failures,
        }

    sequence_group_items = list(sequence_groups.items())
    sequence_cluster_jobs = min(normalize_worker_count(sequence_cluster_jobs), max(1, len(sequence_group_items)))
    if sequence_cluster_jobs <= 1:
        group_results = [
            process_sequence_group(item)
            for item in tqdm(sequence_group_items, desc="Clustering protein structures", unit="seq-group")
        ]
    else:
        from tqdm.contrib.concurrent import thread_map
        group_results = thread_map(
            process_sequence_group,
            sequence_group_items,
            max_workers=sequence_cluster_jobs,
            desc="Clustering protein structures",
            unit="seq-group",
        )

    for group_result in group_results:
        alignment_rows.extend(group_result["alignment_rows"])
        membership_out_rows.extend(group_result["membership_rows"])
        representative_rows.extend(group_result["representative_rows"])
        warning_rows.extend(group_result["warning_rows"])
        total_struct_clusters += group_result["num_structure_clusters"]
        total_alignments += group_result["num_alignment_runs"]
        total_alignment_failures += group_result["num_alignment_failures"]
        total_sequence_fallback_assignments += group_result["num_sequence_fallback_assignments"]
        total_all_failure_cluster_collapses += group_result["num_all_failure_sequence_cluster_collapses"]
        total_monomers_to_extract += group_result.get("num_extracted", 0)
        total_extraction_failures += group_result.get("num_extraction_failures", 0)

    dump_csv_rows(outdir / "protein_structure_cluster_membership.csv", membership_out_rows)
    dump_csv_rows(outdir / "protein_structure_cluster_representatives.csv", representative_rows)
    dump_jsonl(outdir / "protein_structure_pairwise_alignments.jsonl", alignment_rows)
    dump_jsonl(outdir / "protein_structure_cluster_warnings.jsonl", warning_rows)
    manifest = {
        "num_sequence_clusters": len(sequence_groups),
        "num_structure_clusters": total_struct_clusters,
        "num_alignment_runs": total_alignments,
        "num_alignment_failures": total_alignment_failures,
        "num_sequence_fallback_assignments": total_sequence_fallback_assignments,
        "num_all_failure_sequence_cluster_collapses": total_all_failure_cluster_collapses,
        "num_membership_rows": len(membership_out_rows),
        "num_extracted": total_monomers_to_extract,
        "num_extraction_failures": total_extraction_failures,
        "tm_score_threshold": tm_score_threshold,
        "min_alignment_coverage_ratio": min_alignment_coverage_ratio,
        "sequence_cluster_jobs": sequence_cluster_jobs,
        "pairwise_alignment_jobs": pairwise_alignment_jobs,
    }
    dump_json(outdir / "protein_structure_cluster_manifest.json", manifest, indent=2)
    LOGGER.info(
        "Protein monomer clustering: %d sequence clusters -> %d structure clusters (%d alignments, %d failures)",
        len(sequence_groups),
        total_struct_clusters,
        total_alignments,
        total_alignment_failures,
    )
    return {
        "manifest": manifest,
        "membership_rows": membership_out_rows,
        "representative_rows": representative_rows,
        "alignment_rows": alignment_rows,
        "sequence_groups": sequence_groups,
        "monomer_index": monomer_index,
    }
