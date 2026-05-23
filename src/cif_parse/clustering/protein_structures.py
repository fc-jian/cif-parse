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


def extract_monomer_from_pkl(
    monomer: MonomerSample,
    cases_root: str,
    outdir: str,
    quality_by_source: dict[str, EntryQualityMetadata] | None = None,
    *,
    drop_hydrogens: bool = True,
) -> ExtractedMonomerStructure | None:
    """ProcessPool-safe: load chain from parse atoms pkl, write PDB, return structure."""
    from cif_parse.clustering.atom_cache import PklAtomReader

    reader = PklAtomReader(cases_root)
    chain_atoms = None
    for aid in (monomer.assembly_id, None):
        if not aid:
            continue
        chain_atoms = reader.load_chain(monomer.source_path, monomer.label_asym_id, str(aid))
        if chain_atoms is not None:
            break
    if chain_atoms is None:
        chain_atoms = reader.load_chain(monomer.source_path, monomer.label_asym_id, None)
    if chain_atoms is None or chain_atoms.array_length() == 0:
        return None

    chain_atoms, filter_counts = filter_atom_array_for_analysis(
        chain_atoms, drop_hydrogens=drop_hydrogens, drop_nonfinite=True,
    )
    if chain_atoms.array_length() == 0:
        return None

    _, residue_names = get_residues(chain_atoms)
    resolved_count = int(len(residue_names))
    if resolved_count <= 2:
        return None

    # Write PDB file (needed by USalign)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    asm_tag = monomer.assembly_id or "na"
    pdb_path = outdir_path / f"{monomer.pdb_id}_{asm_tag}_{monomer.label_asym_id}.pdb"
    _coerced = chain_atoms.copy()
    _coerced.chain_id[:] = "A"
    pdb_file = PDBFile()
    pdb_file.set_structure(_coerced)
    pdb_file.write(pdb_path)

    quality = (quality_by_source or {}).get(
        monomer.source_path,
        _default_entry_quality(source_path=monomer.source_path, pdb_id=monomer.pdb_id),
    )

    return ExtractedMonomerStructure(
        monomer_id=monomer.monomer_id,
        pdb_id=monomer.pdb_id, label_asym_id=monomer.label_asym_id, auth_asym_id=monomer.auth_asym_id,
        source_path=monomer.source_path, extracted_pdb_path=str(pdb_path),
        sequence_length=monomer.length, residue_count=monomer.residue_count,
        resolved_residue_count=resolved_count,
        resolved_fraction=resolved_count / max(1, monomer.residue_count),
        atom_count=int(chain_atoms.array_length()),
        filter_counts=atom_array_filter_counts(filter_counts),
        quality=quality,
    )


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
    asm_tag = monomer.assembly_id or "na"
    pdb_path = outdir / f"{monomer.pdb_id}_{asm_tag}_{monomer.label_asym_id}.pdb"
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
    prep_dir: str | Path | None = None,
    cif_idx: Any = None,
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

    pipeline_mode = extract_fn is not None and len(extracted_structures) == 0
    LOGGER.info(
        "Clustering protein monomers: %d sequence groups, %d pre-extracted structures, "
        "%d seq-cluster workers, %d pairwise workers%s",
        len(sequence_groups),
        len(extracted_structures),
        sequence_cluster_jobs,
        pairwise_alignment_jobs,
        " (pipelined extraction+clustering)" if pipeline_mode else "",
    )

    # Three-phase parallel pipeline: extract → generate tasks → execute all → resolve
    if protein_subcluster_by_sequence or pipeline_mode:
        return _run_three_phase_clustering(
            monomer_index=monomer_index,
            sequence_groups=sequence_groups,
            extracted_structures=extracted_structures,
            outdir=outdir,
            runner=runner,
            tm_score_threshold=tm_score_threshold,
            min_alignment_coverage_ratio=min_alignment_coverage_ratio,
            usalign_executable=usalign_executable,
            sequence_cluster_jobs=sequence_cluster_jobs,
            pairwise_alignment_jobs=pairwise_alignment_jobs,
            extract_fn=extract_fn,
            protein_subcluster_by_sequence=protein_subcluster_by_sequence,
            prep_dir=prep_dir,
            cif_idx=cif_idx,
        )

    available_structures = len(extracted_structures)

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


def _process_one_group_worker(
    payload: tuple,
) -> dict[str, Any]:
    """Module-level ProcessPool worker: extract chains + subcluster + PDB + tasks.

    Each worker is fully self-contained for one sequence group:
    1. Reads pkl files for monomers in the group
    2. Extracts chains, writes PDBs for subcluster representatives
    3. Subclusters by exact sequence
    4. Enumerates greedy USalign tasks
    5. Returns task list, subcluster map, and extracted structures
    """
    (seq_cluster_id, member_ids, monomers_data, cases_root,
     pdb_dir, min_cov) = payload
    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct monomers from picklable data
    monomer_map: dict[str, MonomerSample] = {}
    for md in monomers_data:
        m = MonomerSample(**md)
        monomer_map[m.monomer_id] = m

    # Build per-worker PklAtomReader
    from cif_parse.clustering.atom_cache import PklAtomReader
    reader = PklAtomReader(cases_root)

    # Extract chains for all monomers in this group
    extracted: dict[str, ExtractedMonomerStructure] = {}
    extracted_list: list[ExtractedMonomerStructure] = []
    for mid in member_ids:
        monomer = monomer_map.get(mid)
        if monomer is None:
            continue
        chain_atoms = None
        for aid in (monomer.assembly_id, None):
            if not aid:
                continue
            chain_atoms = reader.load_chain(monomer.source_path, monomer.label_asym_id, str(aid))
            if chain_atoms is not None:
                break
        if chain_atoms is None:
            chain_atoms = reader.load_chain(monomer.source_path, monomer.label_asym_id, None)
        if chain_atoms is None or chain_atoms.array_length() == 0:
            continue

        chain_atoms, filter_counts = filter_atom_array_for_analysis(
            chain_atoms, drop_hydrogens=True, drop_nonfinite=True,
        )
        if chain_atoms.array_length() == 0:
            continue
        _, residue_names = get_residues(chain_atoms)
        resolved_count = int(len(residue_names))
        if resolved_count <= 2:
            continue

        asm_tag = monomer.assembly_id or "na"
        pdb_path = pdb_dir / f"{monomer.pdb_id}_{asm_tag}_{monomer.label_asym_id}.pdb"
        _coerced = chain_atoms.copy()
        _coerced.chain_id[:] = "A"
        pdb_file = PDBFile()
        pdb_file.set_structure(_coerced)
        pdb_file.write(pdb_path)

        ext = ExtractedMonomerStructure(
            monomer_id=monomer.monomer_id,
            pdb_id=monomer.pdb_id, label_asym_id=monomer.label_asym_id,
            auth_asym_id=monomer.auth_asym_id, source_path=monomer.source_path,
            extracted_pdb_path=str(pdb_path), sequence_length=monomer.length,
            residue_count=monomer.residue_count, resolved_residue_count=resolved_count,
            resolved_fraction=resolved_count / max(1, monomer.residue_count),
            atom_count=int(chain_atoms.array_length()),
            filter_counts=atom_array_filter_counts(filter_counts),
            quality=_default_entry_quality(source_path=monomer.source_path, pdb_id=monomer.pdb_id),
        )
        extracted[mid] = ext
        extracted_list.append(ext)

    if not extracted_list:
        return {
            "cluster_id": seq_cluster_id, "tasks": [], "subcluster": {},
            "prefiltered": 0, "extracted": {},
        }

    candidates = sorted(extracted_list, key=lambda item: item.quality_sort_key())

    # Subcluster by exact sequence
    subcluster_rep: dict[str, str] = {}
    from collections import defaultdict as _dd
    seq_groups: dict[str, list[Any]] = _dd(list)
    for c in candidates:
        m = monomer_map.get(c.monomer_id)
        seq_groups[m.sequence if m else ""].append(c)
    candidates = []
    for _seq, members in seq_groups.items():
        rep = min(members, key=lambda item: item.quality_sort_key())
        candidates.append(rep)
        for m in members:
            subcluster_rep[m.monomer_id] = rep.monomer_id

    # Greedy round enumeration
    local_tasks = []
    local_prefiltered = 0
    pending = sorted(candidates, key=lambda item: item.quality_sort_key())
    round_num = 0
    from pathlib import Path as _Path
    from cif_parse.clustering.usalign_cache import alignment_cache_key

    while pending:
        rep = pending[0]
        round_num += 1
        for candidate in pending[1:]:
            shorter = min(rep.residue_count, candidate.residue_count)
            longer = max(rep.residue_count, candidate.residue_count)
            if longer > 0 and shorter / longer < min_cov:
                local_prefiltered += 1
                continue

            rep_mono = monomer_map.get(rep.monomer_id)
            cand_mono = monomer_map.get(candidate.monomer_id)
            ck = alignment_cache_key(
                seq_query=rep_mono.sequence if rep_mono else "",
                seq_target=cand_mono.sequence if cand_mono else "",
                source_query=rep.source_path,
                source_target=candidate.source_path,
            )

            qp = rep.extracted_pdb_path or ""
            tp = candidate.extracted_pdb_path or ""
            qsz = _Path(qp).stat().st_size if qp and _Path(qp).exists() else rep.residue_count * 100
            tsz = _Path(tp).stat().st_size if tp and _Path(tp).exists() else candidate.residue_count * 100

            local_tasks.append({
                "cache_key": ck,
                "query_monomer_id": rep.monomer_id,
                "target_monomer_id": candidate.monomer_id,
                "query_pdb_path": qp, "target_pdb_path": tp,
                "query_residue_count": rep.residue_count,
                "target_residue_count": candidate.residue_count,
                "query_pdb_size": qsz, "target_pdb_size": tsz,
                "sequence_cluster_id": seq_cluster_id,
                "subcluster_rep_id": subcluster_rep.get(rep.monomer_id, rep.monomer_id),
                "seq_group_idx": 0, "round_num": round_num,
            })
        pending = pending[1:]

    return {
        "cluster_id": seq_cluster_id,
        "tasks": local_tasks,
        "subcluster": subcluster_rep,
        "prefiltered": local_prefiltered,
        "extracted": extracted,
    }


def _run_three_phase_clustering(
    *,
    monomer_index: dict[str, Any],
    sequence_groups: dict[str, list[str]],
    extracted_structures: dict[str, ExtractedMonomerStructure],
    outdir: Path,
    runner: Any,
    tm_score_threshold: float,
    min_alignment_coverage_ratio: float,
    usalign_executable: str,
    sequence_cluster_jobs: int,
    pairwise_alignment_jobs: int,
    extract_fn: Any | None,
    protein_subcluster_by_sequence: bool,
    prep_dir: str | Path | None = None,
    cif_idx: Any = None,
) -> dict[str, Any]:
    """Three-phase parallel monomer structure clustering.

    Phase 1 — extract + subcluster + generate alignment tasks → SQLite
    Phase 2 — execute all pending USalign tasks globally, update cache + SQLite
    Phase 3 — per-group greedy reconstruction from SQLite results
    """
    import heapq
    from collections import defaultdict as _dd
    from cif_parse.clustering.usalign_cache import (
        AlignmentCacheDB, alignment_cache_key, get_fast_temp_dir,
    )

    cache_db = AlignmentCacheDB(outdir / "usalign_tasks.db")
    LOGGER.info("[checkpoint] SQLite db: %s", outdir / "usalign_tasks.db")
    pdb_dir = outdir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: extract + subcluster + generate tasks --------------------
    LOGGER.info("[checkpoint] Phase 1 start: %d sequence groups, %d monomers to extract",
                len(sequence_groups), sum(len(v) for v in sequence_groups.values()))

    all_subcluster: dict[str, dict[str, str]] = {}  # seq_group → {member_id: rep_id}
    total_tasks = 0
    total_cached = 0
    total_prefiltered = 0

    # Collect unique monomers to extract
    to_extract: set[str] = set()
    for member_ids in sequence_groups.values():
        to_extract.update(member_ids)

    # ---- Phase 1: per-group ProcessPool (extract + subcluster + PDB + tasks) ---
    # Each worker is fully self-contained for one sequence group.
    from cif_parse.clustering.atom_cache import resolve_cases_root
    cases_root: str = str(resolve_cases_root(prep_dir)) if prep_dir else ""
    # Load entry quality metadata for representative ranking
    quality_by_source: dict[str, EntryQualityMetadata] = {}
    try:
        quality_by_source = load_entry_quality_metadata_from_prep(prep_dir)
    except Exception:
        LOGGER.debug("[fallback] Failed to load entry quality metadata", exc_info=True)

    # ---- Phase 1: per-group ProcessPool (extract + subcluster + PDB + tasks) ---
    # Build picklable payloads, sorted largest-first to minimize long tail.
    # Each worker is fully self-contained for one sequence group.
    _group_payloads = [
        (seq_id, member_ids,
         [asdict(monomer_index[mid]) for mid in member_ids if mid in monomer_index],
         cases_root, str(pdb_dir), min_alignment_coverage_ratio)
        for seq_id, member_ids in sorted(sequence_groups.items())
    ]

    task_rows: list[dict[str, Any]] = []
    cluster_order: list[tuple[str, int]] = []
    all_subcluster: dict[str, dict[str, str]] = {}
    total_tasks = 0
    total_prefiltered = 0
    ext_success = 0
    ext_fail = 0

    # Sort largest groups first to minimize ProcessPool long tail
    _group_payloads.sort(key=lambda g: -len(g[1]))

    n_workers = max(1, min(sequence_cluster_jobs, len(_group_payloads))) if sequence_cluster_jobs > 1 else 1
    if n_workers > 1 and len(_group_payloads) > 1:
        from concurrent.futures import ProcessPoolExecutor as _PPE, as_completed as _ac
        with _PPE(max_workers=n_workers) as _gex:
            futures = {_gex.submit(_process_one_group_worker, g): i for i, g in enumerate(_group_payloads)}
            for f in tqdm(_ac(futures), total=len(futures), desc="Phase 1 (extract+subcluster)", unit="group"):
                r = f.result()
                cluster_order.append((r["cluster_id"], len(cluster_order)))
                all_subcluster[r["cluster_id"]] = r["subcluster"]
                total_prefiltered += r.get("prefiltered", 0)
                for mid, ext in r.get("extracted", {}).items():
                    extracted_structures[mid] = ext
                local_tasks = r["tasks"]
                total_tasks += len(local_tasks)
                task_rows.extend(local_tasks)
    else:
        for g in tqdm(_group_payloads, desc="Phase 1 (extract+subcluster)", unit="group"):
            r = _process_one_group_worker(g)
            cluster_order.append((r["cluster_id"], len(cluster_order)))
            all_subcluster[r["cluster_id"]] = r["subcluster"]
            total_prefiltered += r.get("prefiltered", 0)
            for mid, ext in r.get("extracted", {}).items():
                extracted_structures[mid] = ext
            task_rows.extend(r["tasks"])
            total_tasks += len(r["tasks"])

    ext_success = sum(1 for mid in to_extract if mid in extracted_structures)
    ext_fail = len(to_extract) - ext_success
    LOGGER.info("[checkpoint] Phase 1: %d/%d monomers extracted (%d failures), %d tasks",
                ext_success, len(to_extract), ext_fail, total_tasks)

    # Batch insert tasks into SQLite + upsert cache entries
    if task_rows:
        cache_db.task_insert_many(task_rows)
        for t in tqdm(task_rows, desc="Phase 1b (cache upsert)", unit="task"):
            cache_db.cache_upsert_pending(
                t["cache_key"], seq_hash_query="", seq_hash_target="",
                source_query="", source_target="",
            )

    # ---- Phase 2: execute all pending USalign tasks globally ---------------
    pending_tasks = cache_db.task_get_pending()
    if pending_tasks:
        LOGGER.info("[checkpoint] Phase 2 start: %d USalign tasks, %d workers",
                    len(pending_tasks), pairwise_alignment_jobs)
        from cif_parse.clustering.parallel import AlignmentTask

        def _build_alignment_task(t: dict[str, Any]) -> AlignmentTask:
            q = extracted_structures.get(t["query_monomer_id"])
            tgt = extracted_structures.get(t["target_monomer_id"])
            if q is None or tgt is None:
                raise ValueError(f"Missing structure for {t['query_monomer_id']} or {t['target_monomer_id']}")
            return AlignmentTask(key=t["cache_key"], query=q, target=tgt, context={"task": t})

        pending_tasks.sort(key=lambda t: -(t["query_pdb_size"] + t["target_pdb_size"]))
        batch_size = max(1, min(200, len(pending_tasks)))

        for i in tqdm(range(0, len(pending_tasks), batch_size), desc="Phase 2 (USalign)", unit="batch"):
            batch = pending_tasks[i:i + batch_size]
            align_tasks = [_build_alignment_task(t) for t in batch]
            successes, failures = run_alignment_tasks(
                align_tasks, runner,
                max_workers=pairwise_alignment_jobs,
                usalign_executable=usalign_executable,
                tm_score_threshold=tm_score_threshold,
            )
            done_ids: list[int] = []
            for task, result in successes:
                cache_db.cache_write_result(task.key, {
                    "tm_score_query": result.tm_score_query, "tm_score_target": result.tm_score_target,
                    "tm_score_max": result.max_tm_score, "rmsd": result.rmsd,
                    "aligned_length": result.aligned_length,
                    "shorter_length_coverage": result.shorter_length_coverage,
                    "resolved_length_coverage": result.resolved_length_coverage,
                })
                tid = task.context.get("task", {}).get("task_id")
                if tid is not None: done_ids.append(int(tid))
            for task, exc in failures:
                cache_db.cache_write_error(task.key, str(exc))
                tid = task.context.get("task", {}).get("task_id")
                if tid is not None: done_ids.append(int(tid))
            if done_ids:
                cache_db.task_status_batch_update(done_ids, "completed")

        LOGGER.info("Phase 2 complete: %d tasks", len(pending_tasks))

    # ---- Phase 3: per-group greedy reconstruction --------------------------
    LOGGER.info("[checkpoint] Phase 3 start: %d groups, %d total members",
                len(cluster_order), sum(len(sequence_groups[g[0]]) for g in cluster_order))

    alignment_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []

    total_struct_clusters = 0
    total_alignments = 0
    total_alignment_failures = 0
    total_sequence_fallback = 0
    total_all_failure_collapses = 0

    for seq_cluster_id, group_idx in tqdm(cluster_order, desc="Phase 3 (reconstruct)", unit="group"):
        member_ids = sequence_groups[seq_cluster_id]
        subcluster_rep = all_subcluster.get(seq_cluster_id, {})
        db_tasks = cache_db.tasks_for_group(seq_cluster_id)

        # Build candidate structures
        candidates = sorted(
            [extracted_structures[mid] for mid in member_ids if mid in extracted_structures],
            key=lambda item: item.quality_sort_key(),
        )
        if not candidates:
            # All extractions failed — collapse whole group
            scid = _protein_structure_cluster_id(seq_cluster_id, 1)
            total_struct_clusters += 1
            total_sequence_fallback += len(member_ids)
            total_all_failure_collapses += 1
            warning_rows.append({
                "warning_code": "no_extracted_structures_in_sequence_cluster",
                "sequence_cluster_id": seq_cluster_id,
                "structure_cluster_id": scid,
                "representative_monomer_id": member_ids[0],
                "member_monomer_ids": member_ids,
            })
            for mid in member_ids:
                membership_rows.append({
                    "polymer_class": "protein",
                    "sequence_cluster_id": seq_cluster_id,
                    "structure_cluster_id": scid,
                    "representative_monomer_id": member_ids[0],
                    "member_monomer_id": mid,
                    "assignment_reason": "structure_extraction_failure_sequence_cluster_collapse",
                    "tm_score_min": "", "tm_score_max": "", "tm_score_for_clustering": "",
                    "alignment_coverage_shorter": "", "alignment_coverage_resolved": "",
                })
            rep_mono = monomer_index.get(member_ids[0])
            representative_rows.append({
                "sequence_cluster_id": seq_cluster_id,
                "structure_cluster_id": scid,
                "representative_monomer_id": member_ids[0],
                "num_members": len(member_ids),
                "pdb_id": rep_mono.pdb_id if rep_mono else "",
                "label_asym_id": rep_mono.label_asym_id if rep_mono else "",
                "auth_asym_id": rep_mono.auth_asym_id or "" if rep_mono else "",
                "resolved_fraction": "", "primary_method": "", "resolution": "",
            })
            continue

        # Group tasks by round, build result lookup
        results_by_key: dict[str, dict[str, Any]] = {}
        failures_by_key: set[str] = set()
        for t in db_tasks:
            if t.get("error_message"):
                failures_by_key.add(t["cache_key"])
            elif t.get("tm_score_max") is not None:
                results_by_key[t["cache_key"]] = t

        # Replay greedy clustering
        # Start with subcluster representatives as the pending pool
        if subcluster_rep:
            rep_ids = set(subcluster_rep.values())
            pending = sorted(
                [s for s in candidates if s.monomer_id in rep_ids],
                key=lambda item: item.quality_sort_key(),
            )
            if not pending:
                pending = sorted(candidates, key=lambda item: item.quality_sort_key())
        else:
            pending = sorted(candidates, key=lambda item: item.quality_sort_key())

        cluster_states: list[dict[str, Any]] = []
        failed_candidates: list[Any] = []
        extracted_ids = {s.monomer_id for s in candidates}
        missing_ids = sorted(set(member_ids) - extracted_ids)
        local_ci = 0

        while pending:
            rep = pending[0]
            local_ci += 1
            scid = _protein_structure_cluster_id(seq_cluster_id, local_ci)
            assigned = [rep]
            remaining = []

            for candidate in pending[1:]:
                ck = alignment_cache_key(
                    seq_query=monomer_index[rep.monomer_id].sequence if rep.monomer_id in monomer_index else "",
                    seq_target=monomer_index[candidate.monomer_id].sequence if candidate.monomer_id in monomer_index else "",
                    source_query=rep.source_path,
                    source_target=candidate.source_path,
                )
                result = results_by_key.get(ck)
                if result is None:
                    # Was cached/prefiltered — check if still applicable
                    cached = cache_db.cache_lookup(ck)
                    if cached:
                        result = {"tm_score_max": cached["tm_score_max"], "rmsd": cached["rmsd"],
                                  "aligned_length": cached["aligned_length"],
                                  "shorter_length_coverage": cached["shorter_length_coverage"],
                                  "resolved_length_coverage": cached["resolved_length_coverage"]}

                if ck in failures_by_key:
                    total_alignment_failures += 1
                    remaining.append(candidate)
                elif result and result.get("tm_score_max", 0) >= tm_score_threshold:
                    assigned.append(candidate)
                    total_alignments += 1
                    alignment_rows.append({
                        "signature_cluster_id": seq_cluster_id,
                        "query_monomer_id": rep.monomer_id,
                        "target_monomer_id": candidate.monomer_id,
                        "aligned_length": result.get("aligned_length", 0),
                        "rmsd": result.get("rmsd", 0),
                        "tm_score_query": result.get("tm_score_query", result.get("tm_score_max", 0)),
                        "tm_score_target": result.get("tm_score_target", result.get("tm_score_max", 0)),
                        "tm_score_min": result.get("tm_score_min", result.get("tm_score_max", 0)),
                        "tm_score_max": result.get("tm_score_max", 0),
                        "tm_score_for_clustering": result.get("tm_score_max", 0),
                    })
                else:
                    remaining.append(candidate)

            cluster_states.append({"rep": rep, "assigned": assigned, "scid": scid})
            pending = remaining

        # Build rep→cluster_id mapping for subcluster expansion
        rep_to_cluster: dict[str, str] = {}
        for state in cluster_states:
            scid = state["scid"]
            for member in state["assigned"]:
                rep_to_cluster[member.monomer_id] = scid

        # Write membership + representative rows
        for state in cluster_states:
            rep = state["rep"]
            scid = state["scid"]
            total_struct_clusters += 1

            ordered = sorted(state["assigned"], key=lambda item: (item.monomer_id != rep.monomer_id, item.monomer_id))
            for member in ordered:
                reason = "representative"
                tm_min, tm_max, tm_clus, cov_shorter, cov_resolved = "", "", "", "", ""
                if member.monomer_id != rep.monomer_id:
                    reason = "representative_alignment"
                    pair_key = alignment_cache_key(
                        seq_query=monomer_index[rep.monomer_id].sequence if rep.monomer_id in monomer_index else "",
                        seq_target=monomer_index[member.monomer_id].sequence if member.monomer_id in monomer_index else "",
                        source_query=rep.source_path, source_target=member.source_path,
                    )
                    result = results_by_key.get(pair_key)
                    if result is not None:
                        tm_min = result.get("tm_score_min", "") or result.get("tm_score_max", "")
                        tm_max = result.get("tm_score_max", "")
                        tm_clus = result.get("tm_score_max", "")
                        cov_shorter = result.get("shorter_length_coverage", "")
                        cov_resolved = result.get("resolved_length_coverage", "")
                membership_rows.append({
                    "polymer_class": "protein", "sequence_cluster_id": seq_cluster_id,
                    "structure_cluster_id": scid,
                    "representative_monomer_id": rep.monomer_id,
                    "member_monomer_id": member.monomer_id,
                    "assignment_reason": reason,
                    "tm_score_min": tm_min, "tm_score_max": tm_max,
                    "tm_score_for_clustering": tm_clus,
                    "alignment_coverage_shorter": cov_shorter,
                    "alignment_coverage_resolved": cov_resolved,
                })

            # Expand subcluster members: follow each rep's cluster assignment
            if subcluster_rep:
                for mid, rid in subcluster_rep.items():
                    assigned_cluster = rep_to_cluster.get(rid)
                    if assigned_cluster and mid != rid:
                        membership_rows.append({
                            "polymer_class": "protein", "sequence_cluster_id": seq_cluster_id,
                            "structure_cluster_id": assigned_cluster,
                            "representative_monomer_id": rid,
                            "member_monomer_id": mid,
                            "assignment_reason": "identical_sequence_subcluster",
                            "tm_score_min": "", "tm_score_max": "", "tm_score_for_clustering": "",
                            "alignment_coverage_shorter": "", "alignment_coverage_resolved": "",
                        })

            sub_extra = sum(1 for mid, rid in subcluster_rep.items()
                            if mid != rid) if subcluster_rep else 0
            representative_rows.append({
                "sequence_cluster_id": seq_cluster_id,
                "structure_cluster_id": scid,
                "representative_monomer_id": rep.monomer_id,
                "num_members": len(state["assigned"]) + sub_extra,
                "pdb_id": rep.pdb_id, "label_asym_id": rep.label_asym_id,
                "auth_asym_id": rep.auth_asym_id or "",
                "resolved_fraction": rep.resolved_fraction,
                "primary_method": rep.quality.primary_method if rep.quality else "",
                "resolution": rep.quality.resolution if rep.quality else "",
            })

        # Fallback members (extraction failures assigned to first cluster)
        if missing_ids:
            fallback_scid = cluster_states[0]["scid"] if cluster_states else _protein_structure_cluster_id(seq_cluster_id, 1)
            for mid in sorted(missing_ids):
                membership_rows.append({
                    "polymer_class": "protein", "sequence_cluster_id": seq_cluster_id,
                    "structure_cluster_id": fallback_scid,
                    "representative_monomer_id": cluster_states[0]["rep"].monomer_id if cluster_states else member_ids[0],
                    "member_monomer_id": mid,
                    "assignment_reason": "structure_extraction_failure_sequence_fallback",
                    "tm_score_min": "", "tm_score_max": "", "tm_score_for_clustering": "",
                    "alignment_coverage_shorter": "", "alignment_coverage_resolved": "",
                })

    cache_db.close()

    manifest = {
        "num_sequence_groups": len(sequence_groups),
        "num_structure_clusters": total_struct_clusters,
        "num_alignment_runs": total_alignments,
        "num_alignment_failures": total_alignment_failures,
        "num_sequence_fallback_assignments": total_sequence_fallback,
        "num_all_failure_sequence_cluster_collapses": total_all_failure_collapses,
        "num_membership_rows": len(membership_rows),
        "num_prefiltered": total_prefiltered,
        "num_cache_hits": total_cached,
        "tm_score_threshold": tm_score_threshold,
        "min_alignment_coverage_ratio": min_alignment_coverage_ratio,
    }

    # Write CSVs for high-order stages consumption
    dump_csv_rows(outdir / "protein_structure_cluster_membership.csv", membership_rows)
    dump_csv_rows(outdir / "protein_structure_cluster_representatives.csv", representative_rows)
    dump_jsonl(outdir / "protein_structure_pairwise_alignments.jsonl", alignment_rows)
    dump_jsonl(outdir / "protein_structure_cluster_warnings.jsonl", warning_rows)

    return {
        "manifest": manifest,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "alignment_rows": alignment_rows,
        "warning_rows": warning_rows,
        "sequence_groups": sequence_groups,
        "monomer_index": monomer_index,
    }
