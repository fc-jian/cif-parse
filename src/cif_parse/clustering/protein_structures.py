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

from cif_parse.clustering.monomers import MonomerSample
from cif_parse.clustering.parallel import AlignmentTask, normalize_worker_count, run_alignment_tasks
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.utils.atom_filters import atom_array_filter_counts, filter_atom_array_for_analysis


LOGGER = logging.getLogger(__name__)

SKIP_QUALITY_METADATA = object()

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


def _default_entry_quality(*, source_path: str, pdb_id: str) -> EntryQualityMetadata:
    return EntryQualityMetadata(
        pdb_id=str(pdb_id),
        source_path=str(source_path),
        experimental_methods=[],
        primary_method=None,
        method_priority=99,
        resolution=None,
    )


def _entry_quality_from_row(row: dict[str, Any]) -> EntryQualityMetadata:
    raw_methods = row.get("experimental_methods", "[]")
    if isinstance(raw_methods, str):
        try:
            methods = [str(item) for item in json.loads(raw_methods) if str(item)]
        except (TypeError, ValueError, json.JSONDecodeError):
            methods = [raw_methods] if raw_methods else []
    elif isinstance(raw_methods, list):
        methods = [str(item) for item in raw_methods if str(item)]
    else:
        methods = []
    primary_method = row.get("primary_method")
    primary_method = str(primary_method) if primary_method not in (None, "", ".", "?") else _pick_primary_method(methods)
    resolution = _safe_float(row.get("resolution"))
    return EntryQualityMetadata(
        pdb_id=str(row.get("pdb_id", "") or ""),
        source_path=str(row.get("source_path", "") or ""),
        experimental_methods=sorted(set(methods)),
        primary_method=primary_method,
        method_priority=int(row.get("method_priority", _method_priority(primary_method)) or 99),
        resolution=resolution,
    )


def load_entry_quality_metadata_from_prep(prep_dir: str | Path) -> dict[str, EntryQualityMetadata]:
    """Load entry quality metadata captured during parse/prep."""

    from cif_parse.clustering.prep import iter_parquet_rows

    quality_by_source: dict[str, EntryQualityMetadata] = {}
    for row in iter_parquet_rows(prep_dir, "entry_quality", required=False) or []:
        if "__empty__" in row:
            continue
        quality = _entry_quality_from_row(row)
        if quality.source_path and quality.source_path not in quality_by_source:
            quality_by_source[quality.source_path] = quality
    return quality_by_source


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
    resolved_length_coverage: float
    meets_tm_threshold: bool
    meets_coverage_threshold: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_protein_monomer_structure(
    monomer: MonomerSample,
    *,
    outdir: str | Path,
    model: int = 1,
    drop_hydrogens: bool = True,
    quality_metadata: EntryQualityMetadata | object | None = None,
    atom_array: AtomArray | None = None,
) -> ExtractedMonomerStructure:
    """Extract one canonical monomer chain from cached atoms and write a PDB."""

    full_atom_array = atom_array
    if full_atom_array is None:
        raise ValueError(f"Cached coordinates are required for monomer {monomer.monomer_id}")
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

    quality = (
        None
        if quality_metadata is SKIP_QUALITY_METADATA
        else quality_metadata or _default_entry_quality(source_path=monomer.source_path, pdb_id=monomer.pdb_id)
    )
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
    query_resolved_residue_count: int | None = None,
    target_resolved_residue_count: int | None = None,
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
    query_resolved = int(query_resolved_residue_count or query_length)
    target_resolved = int(target_resolved_residue_count or target_length)
    shorter_resolved_length = max(1, min(query_resolved, target_resolved))
    resolved_length_coverage = min(1.0, float(aligned_length) / float(shorter_resolved_length))
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
        resolved_length_coverage=resolved_length_coverage,
        meets_tm_threshold=max_tm_score >= tm_score_threshold,
        meets_coverage_threshold=resolved_length_coverage >= min_alignment_coverage_ratio,
    )


def run_usalign_alignment(
    query: ExtractedMonomerStructure,
    target: ExtractedMonomerStructure,
    *,
    usalign_executable: str = "USalign",
    mol: str = "auto",
    tm_score_threshold: float = 0.50,
    min_alignment_coverage_ratio: float = 0.50,
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
        query_resolved_residue_count=query.resolved_residue_count,
        target_resolved_residue_count=target.resolved_residue_count,
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
    quality_by_source: dict[str, EntryQualityMetadata] = {}
    if prep_dir:
        from cif_parse.clustering.prep import load_cif_coords_index, load_cif_from_prep
        cif_idx = load_cif_coords_index(prep_dir)
        quality_by_source = load_entry_quality_metadata_from_prep(prep_dir)

    structures: dict[str, ExtractedMonomerStructure] = {}
    failures: list[dict[str, str]] = []

    def _load_cif_cache(_prep_dir, _source_path, _assembly_id, _idx):
        from cif_parse.clustering.prep import load_cif_from_prep as _lcfp
        return _lcfp(_prep_dir, _source_path, _assembly_id, index=_idx)

    def _load_chain_from_prep_fn(_source_path, _label_asym_id, _assembly_id=None):
        from cif_parse.clustering.prep import load_chain_from_prep
        return load_chain_from_prep(prep_dir, _source_path, _label_asym_id,
                                    assembly_id=_assembly_id, index=cif_idx)

    def _load_atom_array_for_monomer(_source_path, _label_asym_id, _assembly_id):
        """Try the monomer's specific assembly first, then asymmetric unit."""
        if cif_idx is None:
            return None
        # Try the monomer's own assembly
        if _assembly_id:
            cached = _load_chain_from_prep_fn(_source_path, _label_asym_id, _assembly_id)
            if cached is not None and cached.get("atom_array") is not None:
                return cached["atom_array"]
        # Fallback: asymmetric unit
        cached = _load_chain_from_prep_fn(_source_path, _label_asym_id, None)
        if cached is not None and cached.get("atom_array") is not None:
            return cached["atom_array"]
        # Fallback: try legacy full-assembly blob
        cached = _load_cif_cache(prep_dir, _source_path, None, cif_idx)
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
            _atom_key = f"{monomer.source_path}__{monomer.label_asym_id}"
            if _atom_key not in atom_array_cache:
                _prep_atoms = _load_atom_array_for_monomer(
                    monomer.source_path, monomer.label_asym_id, monomer.assembly_id)
                if _prep_atoms is not None:
                    atom_array_cache[_atom_key] = _prep_atoms
            if monomer.source_path not in quality_cache:
                quality_cache[monomer.source_path] = quality_by_source.get(
                    monomer.source_path,
                    _default_entry_quality(source_path=monomer.source_path, pdb_id=monomer.pdb_id),
                )
            if _atom_key not in atom_array_cache and monomer.source_path not in atom_array_cache:
                raise ValueError(f"Prep coordinates missing for monomer {monomer.monomer_id}")
            _atoms = atom_array_cache.get(_atom_key) or atom_array_cache.get(monomer.source_path)
            try:
                structures[monomer.monomer_id] = extract_protein_monomer_structure(
                    monomer,
                    outdir=outdir,
                    model=model,
                    drop_hydrogens=drop_hydrogens,
                    quality_metadata=quality_cache[monomer.source_path],
                    atom_array=_atoms,
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
            _atom_key_local = f"{monomer.source_path}__{monomer.label_asym_id}"
            # Phase 1: check caches under lock
            _need_atoms = False
            _need_quality = False
            with cache_lock:
                _need_atoms = (_atom_key_local not in atom_array_cache)
                _need_quality = (monomer.source_path not in quality_cache)

            # Phase 2: do I/O outside lock
            if _need_atoms:
                _prep = _load_atom_array_for_monomer(
                    monomer.source_path, monomer.label_asym_id, monomer.assembly_id)
                if _prep is not None:
                    with cache_lock:
                        atom_array_cache[_atom_key_local] = _prep
            if _need_quality:
                _q = quality_by_source.get(
                    monomer.source_path,
                    _default_entry_quality(source_path=monomer.source_path, pdb_id=monomer.pdb_id),
                )
                with cache_lock:
                    if monomer.source_path not in quality_cache:
                        quality_cache[monomer.source_path] = _q

            # Phase 3: check again under lock; clustering does not read mmCIF.
            with cache_lock:
                _need_atoms = (_atom_key_local not in atom_array_cache and monomer.source_path not in atom_array_cache)
            if _need_atoms:
                raise ValueError(f"Prep coordinates missing for monomer {monomer.monomer_id}")

            with cache_lock:
                _atoms = atom_array_cache.get(_atom_key_local) or atom_array_cache.get(monomer.source_path)
                _q = quality_cache.get(monomer.source_path)
            return extract_protein_monomer_structure(
                monomer, outdir=outdir, model=model, drop_hydrogens=drop_hydrogens,
                quality_metadata=_q, atom_array=_atoms,
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
    protein_subcluster_by_sequence: bool = True,
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
                        "alignment_coverage_resolved": "",
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

        # --- subcluster by exact sequence identity --------------------------------
        # Group monomers with 100% identical sequences into subclusters.  Only
        # subcluster representatives participate in pairwise USalign; other
        # members are assigned to the same structure cluster as their rep.
        subcluster_rep: dict[str, str] = {}  # monomer_id → representative monomer_id
        if protein_subcluster_by_sequence:
            from collections import defaultdict as _defaultdict
            seq_groups: dict[str, list[ExtractedMonomerStructure]] = _defaultdict(list)
            monomer_seq: dict[str, str] = {}
            for candidate in candidates:
                monomer = monomer_index.get(candidate.monomer_id)
                seq = monomer.sequence if monomer else ""
                monomer_seq[candidate.monomer_id] = seq
                seq_groups[seq].append(candidate)
            candidates = []
            for _seq, members in seq_groups.items():
                rep = min(members, key=lambda item: item.quality_sort_key())
                candidates.append(rep)
                for m in members:
                    subcluster_rep[m.monomer_id] = rep.monomer_id
            if len(candidates) < len(seq_groups):
                LOGGER.debug(
                    "Subcluster: %d extracted → %d unique sequences in %s",
                    len(seq_groups), len(candidates), sequence_cluster_id,
                )

        pending = sorted(candidates, key=lambda item: item.quality_sort_key())
        cluster_states: list[dict[str, Any]] = []
        failed_candidates: list[dict[str, Any]] = []
        alignment_success_count = 0
        sequence_member_ids = {candidate.monomer_id for candidate in candidates}
        missing_structure_member_ids = sorted(set(member_ids) - {m.monomer_id for m in pending})
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
                                "alignment_coverage_resolved": result.resolved_length_coverage,
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
                alignment_coverage_resolved: float | str = ""
                if member.monomer_id != representative.monomer_id:
                    pair_key = tuple(sorted((representative.monomer_id, member.monomer_id)))
                    if pair_key in local_alignment_cache:
                        result = local_alignment_cache[pair_key]
                        assignment_reason = "representative_alignment"
                        tm_score_min = result.min_tm_score
                        tm_score_max = result.max_tm_score
                        tm_score_for_clustering = result.max_tm_score
                        alignment_coverage_shorter = result.shorter_length_coverage
                        alignment_coverage_resolved = result.resolved_length_coverage
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
                        "alignment_coverage_resolved": alignment_coverage_resolved,
                    }
                )

                # Expand subcluster members: any monomer that has an identical
                # sequence to the rep (but was never aligned via USalign) gets
                # the same structure cluster assignment.
                if subcluster_rep:
                    for m_id, rep_id in subcluster_rep.items():
                        if rep_id == representative.monomer_id and m_id != rep_id:
                            local_membership_rows.append(
                                {
                                    "polymer_class": "protein",
                                    "sequence_cluster_id": sequence_cluster_id,
                                    "structure_cluster_id": structure_cluster_id,
                                    "representative_monomer_id": representative.monomer_id,
                                    "member_monomer_id": m_id,
                                    "assignment_reason": "identical_sequence_subcluster",
                                    "tm_score_min": "",
                                    "tm_score_max": "",
                                    "tm_score_for_clustering": "",
                                    "alignment_coverage_shorter": "",
                                    "alignment_coverage_resolved": "",
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
                        "alignment_coverage_resolved": "",
                    }
                )

            subcluster_extra = sum(
                1 for m_id, rep_id in subcluster_rep.items()
                if rep_id == representative.monomer_id and m_id != rep_id
            ) if subcluster_rep else 0
            local_representative_rows.append(
                {
                    "sequence_cluster_id": sequence_cluster_id,
                    "structure_cluster_id": structure_cluster_id,
                    "representative_monomer_id": representative.monomer_id,
                    "num_members": len(cluster_state["members"]) + len(cluster_state["fallback_member_ids"]) + subcluster_extra,
                    "pdb_id": representative.pdb_id,
                    "label_asym_id": representative.label_asym_id,
                    "auth_asym_id": representative.auth_asym_id or "",
                    "resolved_fraction": representative.resolved_fraction,
                    "primary_method": (
                        representative.quality.primary_method
                        if representative.quality is not None
                        else ""
                    ),
                    "resolution": (
                        representative.quality.resolution
                        if representative.quality is not None
                        and representative.quality.resolution is not None
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
