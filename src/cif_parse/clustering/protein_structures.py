from __future__ import annotations

import json
import hashlib
import logging
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm import tqdm

from biotite.structure import AtomArray, get_residues
from biotite.structure.io.pdb import PDBFile

from cif_parse.clustering.monomers import MonomerSample
from cif_parse.clustering.parallel import normalize_worker_count
from cif_parse.clustering.polymer_atoms import (
    prepare_polymer_atoms_for_usalign,
    select_polymer_chain_atoms,
    validate_usalign_chain_lengths,
)
from cif_parse.clustering.usalign import run_usalign_command
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.utils.atom_filters import atom_array_filter_counts, filter_atom_array_for_analysis


LOGGER = logging.getLogger(__name__)

SKIP_QUALITY_METADATA = object()
USALIGN_PDB_FORMAT_VERSION = 2

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
    return prepare_polymer_atoms_for_usalign(atom_array, chain_id="A")


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
    assembly_id: str | None = None
    coordinate_fingerprint: str = ""
    model: int = 1
    keep_hydrogens: bool = False

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


def _atom_coordinate_fingerprint(atom_array: AtomArray) -> str:
    """Hash the filtered coordinates and residue identity used by USalign."""

    digest = hashlib.sha256()
    digest.update(atom_array.coord.tobytes(order="C"))
    for category in ("chain_id", "res_id", "ins_code", "res_name", "atom_name"):
        try:
            values = atom_array.get_annotation(category)
        except (AttributeError, KeyError):
            continue
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()[:24]


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
        chain_atoms = reader.load_chain(
            monomer.source_path,
            monomer.label_asym_id,
            str(aid),
            filter_hetero=False,
        )
        if chain_atoms is not None:
            break
    if chain_atoms is None:
        chain_atoms = reader.load_chain(
            monomer.source_path,
            monomer.label_asym_id,
            None,
            filter_hetero=False,
        )
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
    _coerced = _coerce_chain_id_for_pdb(chain_atoms)
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
        assembly_id=monomer.assembly_id,
        coordinate_fingerprint=_atom_coordinate_fingerprint(_coerced),
        model=1,
        keep_hydrogens=not drop_hydrogens,
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
    chain_atoms = select_polymer_chain_atoms(
        full_atom_array,
        label_asym_id=monomer.label_asym_id,
    )
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
    validate_usalign_chain_lengths(
        chain_atoms,
        context=f"monomer {monomer.monomer_id}",
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
        assembly_id=monomer.assembly_id,
        coordinate_fingerprint=_atom_coordinate_fingerprint(chain_atoms),
        model=model,
        keep_hydrogens=not drop_hydrogens,
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
    stdout = run_usalign_command(command)
    return parse_usalign_output(
        stdout,
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
    try:
        from biotite.sequence import ProteinSequence
        from biotite.sequence.align import align_optimal, get_pairwise_sequence_identity
        from biotite.sequence.align import SubstitutionMatrix

        seq_a = ProteinSequence(sequence_a)
        seq_b = ProteinSequence(sequence_b)
        matrix = SubstitutionMatrix.std_protein_matrix()
        alignments = align_optimal(seq_a, seq_b, matrix, gap_penalty=(-10, -1))
        if not alignments:
            return 0.0
        identity = get_pairwise_sequence_identity(alignments[0], mode="shortest")
        return float(identity[0, 1])
    except (ValueError, TypeError):
        return 0.0


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
            future_iter = tqdm(
                as_completed(future_to_monomer),
                total=len(future_to_monomer),
                desc="Extracting monomer structures",
                unit="monomer",
            )
            for future in future_iter:
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
    min_alignment_coverage_ratio: float = 0.50,
    usalign_executable: str = "USalign",
    alignment_runner: Callable[..., USalignAlignmentResult] | None = None,
    sequence_cluster_jobs: int = 1,
    pairwise_alignment_jobs: int = 1,
    model: int = 1,
    drop_hydrogens: bool = True,
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

    # Use the bounded stage-wide scheduler for both pre-extracted and pipelined inputs.
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
        model=model,
        drop_hydrogens=drop_hydrogens,
        extract_fn=extract_fn,
        protein_subcluster_by_sequence=protein_subcluster_by_sequence,
        prep_dir=prep_dir,
        cif_idx=cif_idx,
    )


def _reconstruct_structure(
    monomer_id: str,
    info: tuple,
    monomer_index: dict[str, Any],
) -> ExtractedMonomerStructure:
    """Rebuild ExtractedMonomerStructure from compact worker return + monomer index."""
    (res_frac, meth_pri, res, res_cnt, src_path, pdb_path,
     pdb_id, lbl, auth, atom_cnt, primary_method, assembly_id,
     coordinate_fingerprint, model, keep_hydrogens) = info
    m = monomer_index.get(monomer_id)
    return ExtractedMonomerStructure(
        monomer_id=monomer_id, pdb_id=pdb_id, label_asym_id=lbl,
        auth_asym_id=auth, source_path=src_path,
        extracted_pdb_path=pdb_path,
        sequence_length=m.length if m else 0, residue_count=res_cnt,
        resolved_residue_count=int(res_frac * res_cnt) if res_cnt else 0,
        resolved_fraction=res_frac, atom_count=atom_cnt,
        filter_counts={},
        quality=EntryQualityMetadata(
            pdb_id=pdb_id, source_path=src_path,
            experimental_methods=[], primary_method=primary_method,
            method_priority=int(meth_pri), resolution=res,
        ),
        assembly_id=assembly_id or None,
        coordinate_fingerprint=coordinate_fingerprint,
        model=int(model),
        keep_hydrogens=bool(keep_hydrogens),
    )


_structure_worker_prep_dir: str | None = None
_structure_worker_coord_index: dict[str, tuple[Path, int, int]] | None = None
_structure_worker_pkl_reader: Any = None
_structure_worker_quality: dict[str, EntryQualityMetadata] = {}
_structure_worker_drop_hydrogens = True
_structure_worker_model = 1


def _init_structure_group_worker(
    prep_dir: str,
    drop_hydrogens: bool,
    model: int,
) -> None:
    """Initialize process-local coordinate readers once per Phase 1 worker."""

    global _structure_worker_prep_dir
    global _structure_worker_coord_index
    global _structure_worker_pkl_reader
    global _structure_worker_quality
    global _structure_worker_drop_hydrogens
    global _structure_worker_model

    from cif_parse.clustering.atom_cache import (
        PklAtomReader,
        load_source_case_dir_map,
        resolve_cases_root,
    )
    from cif_parse.clustering.prep import load_cif_coords_index

    _structure_worker_prep_dir = prep_dir
    _structure_worker_coord_index = load_cif_coords_index(prep_dir)
    _structure_worker_quality = load_entry_quality_metadata_from_prep(prep_dir)
    _structure_worker_drop_hydrogens = bool(drop_hydrogens)
    _structure_worker_model = int(model)
    try:
        _structure_worker_pkl_reader = PklAtomReader(
            resolve_cases_root(prep_dir),
            load_source_case_dir_map(prep_dir),
        )
    except (FileNotFoundError, OSError, ValueError):
        _structure_worker_pkl_reader = None


def _load_structure_worker_chain(monomer: MonomerSample) -> AtomArray | None:
    from cif_parse.clustering.prep import load_chain_atoms

    if _structure_worker_prep_dir is not None and _structure_worker_coord_index is not None:
        for assembly_id in (monomer.assembly_id, None):
            atoms = load_chain_atoms(
                _structure_worker_prep_dir,
                monomer.source_path,
                monomer.label_asym_id,
                assembly_id=assembly_id,
                index=_structure_worker_coord_index,
            )
            if atoms is not None:
                return atoms
    if _structure_worker_pkl_reader is not None:
        for assembly_id in (monomer.assembly_id, None):
            atoms = _structure_worker_pkl_reader.load_chain(
                monomer.source_path,
                monomer.label_asym_id,
                assembly_id=assembly_id,
                filter_hetero=False,
            )
            if atoms is not None:
                return atoms
    return None


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
    (seq_cluster_id, member_ids, monomers_data, pdb_dir, subcluster_by_sequence) = payload
    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct monomers from picklable data
    monomer_map: dict[str, MonomerSample] = {}
    for md in monomers_data:
        m = MonomerSample(**md)
        monomer_map[m.monomer_id] = m

    # Extract chains for all monomers in this group.
    # If a PDB already exists on disk, reuse it to avoid redundant atom
    # extraction from the original coordinate source.
    extracted_list: list[ExtractedMonomerStructure] = []
    _chain_atoms_for_pdb: dict[str, Any] = {}
    _extract_info: dict[str, tuple] = {}
    group_dir = pdb_dir / seq_cluster_id
    group_dir.mkdir(parents=True, exist_ok=True)
    for mid in member_ids:
        monomer = monomer_map.get(mid)
        if monomer is None:
            continue

        # Early PDB-existence check: skip atom extraction if PDB is present.
        asm_tag = monomer.assembly_id or "na"
        pdb_path = group_dir / (
            f"{monomer.pdb_id}_{asm_tag}_{monomer.label_asym_id}"
            f"_m{_structure_worker_model}_h{int(not _structure_worker_drop_hydrogens)}"
            f"_v{USALIGN_PDB_FORMAT_VERSION}.pdb"
        )
        if pdb_path.exists() and pdb_path.stat().st_size > 0:
            try:
                _pdb_file = PDBFile.read(pdb_path)
                _pdb_atoms = _pdb_file.get_structure(model=1)
                _, _residue_names = get_residues(_pdb_atoms)
                _resolved_count = int(len(_residue_names))
                if _resolved_count > 2:
                    _ext = ExtractedMonomerStructure(
                        monomer_id=monomer.monomer_id,
                        pdb_id=monomer.pdb_id, label_asym_id=monomer.label_asym_id,
                        auth_asym_id=monomer.auth_asym_id, source_path=monomer.source_path,
                        extracted_pdb_path=str(pdb_path), sequence_length=monomer.length,
                        residue_count=monomer.residue_count,
                        resolved_residue_count=_resolved_count,
                        resolved_fraction=_resolved_count / max(1, monomer.residue_count),
                        atom_count=int(_pdb_atoms.array_length()),
                        filter_counts={},
                        quality=_structure_worker_quality.get(monomer.source_path)
                        or _default_entry_quality(source_path=monomer.source_path, pdb_id=monomer.pdb_id),
                        assembly_id=monomer.assembly_id,
                        coordinate_fingerprint=_atom_coordinate_fingerprint(_pdb_atoms),
                        model=_structure_worker_model,
                        keep_hydrogens=not _structure_worker_drop_hydrogens,
                    )
                    _extract_info[mid] = (
                        _ext.resolved_fraction,
                        _ext.quality.method_priority if _ext.quality else 99,
                        _ext.quality.resolution if _ext.quality else None,
                        _ext.residue_count,
                        _ext.source_path,
                        _ext.extracted_pdb_path,
                        _ext.pdb_id,
                        _ext.label_asym_id,
                        _ext.auth_asym_id or "",
                        _ext.atom_count,
                        _ext.quality.primary_method if _ext.quality else None,
                        _ext.assembly_id or "",
                        _ext.coordinate_fingerprint,
                        _ext.model,
                        _ext.keep_hydrogens,
                    )
                    extracted_list.append(_ext)
                    continue
            except (OSError, ValueError):
                pass

        chain_atoms = _load_structure_worker_chain(monomer)
        if chain_atoms is None or chain_atoms.array_length() == 0:
            continue

        chain_atoms, filter_counts = filter_atom_array_for_analysis(
            chain_atoms,
            drop_hydrogens=_structure_worker_drop_hydrogens,
            drop_nonfinite=True,
        )
        if chain_atoms.array_length() == 0:
            continue
        _, residue_names = get_residues(chain_atoms)
        resolved_count = int(len(residue_names))
        if resolved_count <= 2:
            continue

        # Store chain atoms for later PDB write (only for subcluster reps).
        _chain_atoms_for_pdb[mid] = chain_atoms.copy()

        ext = ExtractedMonomerStructure(
            monomer_id=monomer.monomer_id,
            pdb_id=monomer.pdb_id, label_asym_id=monomer.label_asym_id,
            auth_asym_id=monomer.auth_asym_id, source_path=monomer.source_path,
            extracted_pdb_path="", sequence_length=monomer.length,
            residue_count=monomer.residue_count, resolved_residue_count=resolved_count,
            resolved_fraction=resolved_count / max(1, monomer.residue_count),
            atom_count=int(chain_atoms.array_length()),
            filter_counts=atom_array_filter_counts(filter_counts),
            quality=_structure_worker_quality.get(monomer.source_path)
            or _default_entry_quality(source_path=monomer.source_path, pdb_id=monomer.pdb_id),
            assembly_id=monomer.assembly_id,
            model=_structure_worker_model,
            keep_hydrogens=not _structure_worker_drop_hydrogens,
        )
        # Store compact tuple for main-process reconstruction (avoids ~1KB pickle
        # per structure through multiprocessing queues at million-scale).
        _extract_info[mid] = (
            ext.resolved_fraction,
            ext.quality.method_priority if ext.quality else 99,
            ext.quality.resolution if ext.quality else None,
            ext.residue_count,
            ext.source_path,
            ext.extracted_pdb_path,
            ext.pdb_id,
            ext.label_asym_id,
            ext.auth_asym_id or "",
            ext.atom_count,
            ext.quality.primary_method if ext.quality else None,
            ext.assembly_id or "",
            "",
            ext.model,
            ext.keep_hydrogens,
        )
        extracted_list.append(ext)

    if not extracted_list:
        return {
            "cluster_id": seq_cluster_id, "tasks": [], "subcluster": {},
            "prefiltered": 0, "extracted_info": {},
        }

    # Reconstruct ExtractedMonomerStructure from compact info for local greedy use.
    # The compact info is what gets returned to the main process.
    extracted: dict[str, ExtractedMonomerStructure] = {}
    for mid, info in _extract_info.items():
        (res_frac, meth_pri, res, res_cnt, src_path, pdb_path,
         pdb_id, lbl, auth, atom_cnt, primary_method, assembly_id,
         coordinate_fingerprint, model, keep_hydrogens) = info
        m = monomer_map.get(mid)
        extracted[mid] = ExtractedMonomerStructure(
            monomer_id=mid, pdb_id=pdb_id, label_asym_id=lbl, auth_asym_id=auth,
            source_path=src_path, extracted_pdb_path=pdb_path,
            sequence_length=m.length if m else 0, residue_count=res_cnt,
            resolved_residue_count=int(res_frac * res_cnt) if res_cnt else 0,
            resolved_fraction=res_frac, atom_count=atom_cnt,
            filter_counts={},
            quality=EntryQualityMetadata(
                pdb_id=pdb_id, source_path=src_path,
                experimental_methods=[], primary_method=primary_method,
                method_priority=int(meth_pri), resolution=res,
            ),
            assembly_id=assembly_id or None,
            coordinate_fingerprint=coordinate_fingerprint,
            model=int(model),
            keep_hydrogens=bool(keep_hydrogens),
        )

    candidates = sorted(extracted_list, key=lambda item: item.quality_sort_key())

    # Subcluster by exact sequence when requested.
    subcluster_rep: dict[str, str] = {}
    from collections import defaultdict as _dd
    seq_groups: dict[str, list[Any]] = _dd(list)
    for c in candidates:
        m = monomer_map.get(c.monomer_id)
        key = m.sequence if subcluster_by_sequence and m else c.monomer_id
        seq_groups[key].append(c)
    candidates = []
    for _seq, members in seq_groups.items():
        rep = min(members, key=lambda item: item.quality_sort_key())
        candidates.append(rep)
        for m in members:
            subcluster_rep[m.monomer_id] = rep.monomer_id
        # Write PDB only for the representative, in a group subdirectory
        chain_aa = _chain_atoms_for_pdb.get(rep.monomer_id)
        group_dir = pdb_dir / seq_cluster_id
        group_dir.mkdir(parents=True, exist_ok=True)
        asm_tag = monomer_map[rep.monomer_id].assembly_id or "na"
        pdb_path = group_dir / (
            f"{rep.pdb_id}_{asm_tag}_{rep.label_asym_id}"
            f"_m{_structure_worker_model}_h{int(not _structure_worker_drop_hydrogens)}"
            f"_v{USALIGN_PDB_FORMAT_VERSION}.pdb"
        )
        if pdb_path.exists() and pdb_path.stat().st_size > 0:
            rep.extracted_pdb_path = str(pdb_path)
            if not rep.coordinate_fingerprint:
                try:
                    existing_atoms = PDBFile.read(pdb_path).get_structure(model=1)
                    rep.coordinate_fingerprint = _atom_coordinate_fingerprint(existing_atoms)
                except (OSError, ValueError):
                    rep.coordinate_fingerprint = ""
            prev = _extract_info.get(rep.monomer_id)
            if prev is not None:
                _extract_info[rep.monomer_id] = (prev[0], prev[1], prev[2], prev[3], prev[4],
                                                  str(pdb_path), prev[6], prev[7], prev[8], prev[9],
                                                  prev[10], prev[11], rep.coordinate_fingerprint,
                                                  prev[13], prev[14])
        elif chain_aa is not None and not rep.extracted_pdb_path:
            _coerced = _coerce_chain_id_for_pdb(chain_aa)
            pdb_file = PDBFile()
            pdb_file.set_structure(_coerced)
            pdb_file.write(pdb_path)
            rep.extracted_pdb_path = str(pdb_path)
            rep.coordinate_fingerprint = _atom_coordinate_fingerprint(_coerced)
            prev = _extract_info.get(rep.monomer_id)
            if prev is not None:
                _extract_info[rep.monomer_id] = (prev[0], prev[1], prev[2], prev[3], prev[4],
                                                  str(pdb_path), prev[6], prev[7], prev[8], prev[9],
                                                  prev[10], prev[11], rep.coordinate_fingerprint,
                                                  prev[13], prev[14])

    return {
        "cluster_id": seq_cluster_id,
        "subcluster": subcluster_rep,
        "prefiltered": 0,
        "extracted_info": _extract_info,
    }


def _process_structure_group_batch_worker(
    payloads: list[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    return [_process_one_group_worker(payload) for payload in payloads]


def _batch_structure_group_items(
    group_items: list[tuple[str, list[str]]],
    worker_count: int,
    *,
    max_groups_per_batch: int = 64,
) -> list[list[tuple[str, list[str]]]]:
    """Bound ProcessPool task count without creating long-running coarse batches."""

    if not group_items:
        return []
    target_batches = max(1, min(len(group_items), worker_count * 8))
    total_members = sum(len(member_ids) for _, member_ids in group_items)
    target_members = max(1, math.ceil(total_members / target_batches))
    batches: list[list[tuple[str, list[str]]]] = []
    current: list[tuple[str, list[str]]] = []
    current_members = 0
    for item in group_items:
        member_count = len(item[1])
        if current and (
            current_members + member_count > target_members
            or len(current) >= max_groups_per_batch
        ):
            batches.append(current)
            current = []
            current_members = 0
        current.append(item)
        current_members += member_count
        if member_count >= target_members:
            batches.append(current)
            current = []
            current_members = 0
    if current:
        batches.append(current)
    return batches


def _structure_alignment_cache_key(
    query: ExtractedMonomerStructure,
    target: ExtractedMonomerStructure,
    monomer_index: dict[str, MonomerSample],
) -> str:
    from cif_parse.clustering.usalign_cache import alignment_cache_key

    query_monomer = monomer_index.get(query.monomer_id)
    target_monomer = monomer_index.get(target.monomer_id)
    return alignment_cache_key(
        seq_query=query_monomer.sequence if query_monomer is not None else "",
        seq_target=target_monomer.sequence if target_monomer is not None else "",
        source_query=query.source_path,
        source_target=target.source_path,
        query_monomer_id=query.monomer_id,
        target_monomer_id=target.monomer_id,
        query_coordinate_fingerprint=query.coordinate_fingerprint,
        target_coordinate_fingerprint=target.coordinate_fingerprint,
        model=query.model,
        keep_h=query.keep_hydrogens,
        usalign_mode="monomer",
    )


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
    model: int,
    drop_hydrogens: bool,
    extract_fn: Any | None,
    protein_subcluster_by_sequence: bool,
    prep_dir: str | Path | None = None,
    cif_idx: Any = None,
) -> dict[str, Any]:
    """Extract structures and run bounded asynchronous greedy refinement."""

    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

    from cif_parse.clustering.usalign_cache import AlignmentCacheDB

    del extract_fn, cif_idx
    outdir.mkdir(parents=True, exist_ok=True)
    cache_db = AlignmentCacheDB(outdir / "usalign_tasks.db")
    pdb_dir = outdir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    group_member_map = {
        sequence_cluster_id: list(member_ids)
        for sequence_cluster_id, member_ids in sequence_groups.items()
    }
    member_id_set = {
        monomer_id
        for member_ids in group_member_map.values()
        for monomer_id in member_ids
    }
    monomer_index_light: dict[str, MonomerSample] = {
        monomer_id: monomer
        for monomer_id, monomer in monomer_index.items()
        if monomer_id in member_id_set
    }
    all_subcluster: dict[str, dict[str, str]] = {}

    to_extract = set(monomer_index_light)
    LOGGER.info(
        "[checkpoint] Phase 1 start: %d sequence groups, %d monomers",
        len(group_member_map),
        len(to_extract),
    )

    if prep_dir is not None:
        serialized_monomers = {
            monomer_id: asdict(monomer)
            for monomer_id, monomer in tqdm(
                monomer_index_light.items(),
                desc="Serialize monomers",
                unit="monomer",
            )
        }
        group_items = sorted(
            group_member_map.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )

        def _payload(item: tuple[str, list[str]]) -> tuple[Any, ...]:
            sequence_cluster_id, member_ids = item
            return (
                sequence_cluster_id,
                member_ids,
                [serialized_monomers[mid] for mid in member_ids if mid in serialized_monomers],
                str(pdb_dir),
                protein_subcluster_by_sequence,
            )

        def _consume_group_result(result: dict[str, Any]) -> None:
            all_subcluster[result["cluster_id"]] = result["subcluster"]
            for monomer_id, info in result.get("extracted_info", {}).items():
                extracted_structures[monomer_id] = _reconstruct_structure(
                    monomer_id,
                    info,
                    monomer_index_light,
                )

        worker_count = min(max(1, int(sequence_cluster_jobs)), max(1, len(group_items)))
        cache_db.close()
        if worker_count == 1:
            _init_structure_group_worker(
                str(prep_dir),
                drop_hydrogens,
                model,
            )
            for item in tqdm(group_items, desc="Phase 1 (extract+subcluster)", unit="group"):
                _consume_group_result(_process_one_group_worker(_payload(item)))
        else:
            item_batches = _batch_structure_group_items(group_items, worker_count)
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_structure_group_worker,
                initargs=(str(prep_dir), drop_hydrogens, model),
            ) as executor:
                batch_iter = iter(item_batches)
                futures: dict[Any, list[tuple[str, list[str]]]] = {}
                max_pending = worker_count * 2
                for _ in range(max_pending):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        break
                    payload_batch = [_payload(item) for item in batch]
                    futures[
                        executor.submit(
                            _process_structure_group_batch_worker,
                            payload_batch,
                        )
                    ] = batch
                progress = tqdm(
                    total=len(group_items),
                    desc="Phase 1 (extract+subcluster)",
                    unit="group",
                )
                try:
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            completed_batch = futures.pop(future)
                            for result in future.result():
                                _consume_group_result(result)
                            progress.update(len(completed_batch))
                            try:
                                batch = next(batch_iter)
                            except StopIteration:
                                continue
                            payload_batch = [_payload(item) for item in batch]
                            futures[
                                executor.submit(
                                    _process_structure_group_batch_worker,
                                    payload_batch,
                                )
                            ] = batch
                finally:
                    progress.close()
        cache_db = AlignmentCacheDB(outdir / "usalign_tasks.db")
    else:
        for sequence_cluster_id, member_ids in group_member_map.items():
            available = [
                extracted_structures[mid]
                for mid in member_ids
                if mid in extracted_structures
            ]
            grouped: dict[str, list[ExtractedMonomerStructure]] = {}
            for structure in available:
                monomer = monomer_index_light.get(structure.monomer_id)
                key = (
                    monomer.sequence
                    if protein_subcluster_by_sequence and monomer is not None
                    else structure.monomer_id
                )
                grouped.setdefault(key, []).append(structure)
            subcluster: dict[str, str] = {}
            for members in grouped.values():
                representative = min(members, key=lambda item: item.quality_sort_key())
                for member in members:
                    subcluster[member.monomer_id] = representative.monomer_id
            all_subcluster[sequence_cluster_id] = subcluster

    extracted_count = sum(
        1 for monomer_id in to_extract if monomer_id in extracted_structures
    )
    extraction_failures = len(to_extract) - extracted_count
    LOGGER.info(
        "[checkpoint] Phase 1 complete: %d/%d monomers extracted (%d skipped)",
        extracted_count,
        len(to_extract),
        extraction_failures,
    )

    states: dict[str, dict[str, Any]] = {}
    warning_rows: list[dict[str, Any]] = []
    for sequence_cluster_id, member_ids in group_member_map.items():
        subcluster = all_subcluster.get(sequence_cluster_id, {})
        representative_ids = set(subcluster.values())
        candidates = sorted(
            (
                extracted_structures[monomer_id]
                for monomer_id in representative_ids
                if monomer_id in extracted_structures
                and extracted_structures[monomer_id].extracted_pdb_path
            ),
            key=lambda item: item.quality_sort_key(),
        )
        usable_representative_ids = {item.monomer_id for item in candidates}
        skipped_ids = sorted(
            monomer_id
            for monomer_id in member_ids
            if monomer_id not in subcluster
            or subcluster[monomer_id] not in usable_representative_ids
        )
        for monomer_id in skipped_ids:
            warning_rows.append(
                {
                    "warning_code": "protein_structure_unavailable_skipped",
                    "sequence_cluster_id": sequence_cluster_id,
                    "monomer_id": monomer_id,
                }
            )
        states[sequence_cluster_id] = {
            "sequence_cluster_id": sequence_cluster_id,
            "member_ids": member_ids,
            "subcluster": subcluster,
            "pending": candidates,
            "clusters": [],
            "round_num": 0,
        }

    total_alignment_runs = 0
    total_alignment_failures = 0
    total_cache_hits = 0
    total_tasks_generated = 0
    alignment_rows: list[dict[str, Any]] = []
    ready_states: deque[str] = deque()
    active_state_count = 0

    def _start_round(state: dict[str, Any]) -> None:
        nonlocal active_state_count
        while state["pending"]:
            representative = state["pending"][0]
            candidates = state["pending"][1:]
            if not candidates:
                state["clusters"].append(
                    {"representative": representative, "assigned": [(representative, None)]}
                )
                state["pending"] = []
                return
            state["round_num"] += 1
            state["round_representative"] = representative
            state["round_candidates"] = candidates
            state["round_cursor"] = 0
            state["round_done"] = 0
            state["round_results"] = {}
            state["round_failures"] = {}
            ready_states.append(state["sequence_cluster_id"])
            active_state_count += 1
            return

    def _finish_round(state: dict[str, Any]) -> None:
        nonlocal active_state_count
        representative = state["round_representative"]
        assigned: list[tuple[ExtractedMonomerStructure, dict[str, Any] | None]] = [
            (representative, None)
        ]
        remaining: list[ExtractedMonomerStructure] = []
        for candidate in state["round_candidates"]:
            result = state["round_results"].get(candidate.monomer_id)
            error = state["round_failures"].get(candidate.monomer_id)
            if error is not None:
                remaining.append(candidate)
                LOGGER.warning(
                    "USalign failed for protein structures %s vs %s in %s; "
                    "keeping the candidate for a later cluster: %s",
                    representative.monomer_id,
                    candidate.monomer_id,
                    state["sequence_cluster_id"],
                    error,
                )
                warning_rows.append(
                    {
                        "warning_code": "protein_usalign_failed",
                        "sequence_cluster_id": state["sequence_cluster_id"],
                        "representative_monomer_id": representative.monomer_id,
                        "candidate_monomer_id": candidate.monomer_id,
                        "error": error,
                    }
                )
                continue
            if (
                result is not None
                and float(result.get("tm_score_max", 0.0) or 0.0) >= tm_score_threshold
                and float(result.get("resolved_length_coverage", 0.0) or 0.0)
                >= min_alignment_coverage_ratio
            ):
                assigned.append((candidate, result))
            else:
                remaining.append(candidate)
        state["clusters"].append(
            {"representative": representative, "assigned": assigned}
        )
        state["pending"] = remaining
        active_state_count -= 1
        _start_round(state)

    for state in states.values():
        _start_round(state)

    pairwise_alignment_jobs = max(1, int(pairwise_alignment_jobs))
    future_to_item: dict[Any, dict[str, Any]] = {}
    max_pending = pairwise_alignment_jobs * 4
    progress = tqdm(total=0, desc="Phase 2 (USalign)", unit="alignment")
    with ThreadPoolExecutor(max_workers=pairwise_alignment_jobs) as executor:
        while ready_states or future_to_item or active_state_count:
            capacity = max_pending - len(future_to_item)
            scheduled_items: list[dict[str, Any]] = []
            while ready_states and len(scheduled_items) < capacity:
                sequence_cluster_id = ready_states.popleft()
                state = states[sequence_cluster_id]
                cursor = state["round_cursor"]
                candidates = state["round_candidates"]
                if cursor >= len(candidates):
                    continue
                candidate = candidates[cursor]
                state["round_cursor"] = cursor + 1
                if state["round_cursor"] < len(candidates):
                    ready_states.append(sequence_cluster_id)
                representative = state["round_representative"]
                cache_key = _structure_alignment_cache_key(
                    representative,
                    candidate,
                    monomer_index_light,
                )
                scheduled_items.append(
                    {
                        "cache_key": cache_key,
                        "state": state,
                        "representative": representative,
                        "candidate": candidate,
                    }
                )

            if scheduled_items:
                progress.total = int(progress.total or 0) + len(scheduled_items)
                cache_records = cache_db.cache_get_records(
                    [item["cache_key"] for item in scheduled_items]
                )
                new_task_rows: list[dict[str, Any]] = []
                new_cache_keys: list[str] = []
                for item in scheduled_items:
                    state = item["state"]
                    candidate = item["candidate"]
                    cached = cache_records.get(item["cache_key"])
                    if cached is not None and (
                        cached.get("tm_score_max") is not None
                        or cached.get("error_message")
                    ):
                        total_cache_hits += 1
                        if cached.get("error_message"):
                            state["round_failures"][candidate.monomer_id] = str(
                                cached["error_message"]
                            )
                        else:
                            state["round_results"][candidate.monomer_id] = cached
                        state["round_done"] += 1
                        progress.update(1)
                        if state["round_done"] == len(state["round_candidates"]):
                            _finish_round(state)
                        continue

                    representative = item["representative"]
                    query_path = Path(representative.extracted_pdb_path)
                    target_path = Path(candidate.extracted_pdb_path)
                    query_size = (
                        query_path.stat().st_size
                        if query_path.exists()
                        else representative.residue_count * 100
                    )
                    target_size = (
                        target_path.stat().st_size
                        if target_path.exists()
                        else candidate.residue_count * 100
                    )
                    new_task_rows.append(
                        {
                            "cache_key": item["cache_key"],
                            "query_monomer_id": representative.monomer_id,
                            "target_monomer_id": candidate.monomer_id,
                            "query_pdb_path": representative.extracted_pdb_path,
                            "target_pdb_path": candidate.extracted_pdb_path,
                            "query_residue_count": representative.residue_count,
                            "target_residue_count": candidate.residue_count,
                            "query_pdb_size": query_size,
                            "target_pdb_size": target_size,
                            "sequence_cluster_id": state["sequence_cluster_id"],
                            "subcluster_rep_id": representative.monomer_id,
                            "round_num": state["round_num"],
                        }
                    )
                    new_cache_keys.append(item["cache_key"])
                    future = executor.submit(
                        runner,
                        representative,
                        candidate,
                        usalign_executable=usalign_executable,
                        tm_score_threshold=tm_score_threshold,
                        min_alignment_coverage_ratio=min_alignment_coverage_ratio,
                    )
                    future_to_item[future] = item
                if new_task_rows:
                    cache_db.task_insert_many(new_task_rows)
                    cache_db.cache_upsert_pending_many(new_cache_keys)
                    total_tasks_generated += len(new_task_rows)

            if future_to_item and (
                len(future_to_item) >= max_pending or not ready_states
            ):
                done, _ = wait(future_to_item, return_when=FIRST_COMPLETED)
                result_rows: list[tuple[str, dict[str, Any]]] = []
                error_rows: list[tuple[str, str]] = []
                completed_keys: list[str] = []
                failed_keys: list[str] = []
                for future in done:
                    item = future_to_item.pop(future)
                    state = item["state"]
                    candidate = item["candidate"]
                    cache_key = item["cache_key"]
                    try:
                        result = future.result()
                        result_record = {
                            "tm_score_query": result.tm_score_query,
                            "tm_score_target": result.tm_score_target,
                            "tm_score_min": result.min_tm_score,
                            "tm_score_max": result.max_tm_score,
                            "rmsd": result.rmsd,
                            "aligned_length": result.aligned_length,
                            "shorter_length_coverage": result.shorter_length_coverage,
                            "resolved_length_coverage": result.resolved_length_coverage,
                        }
                        state["round_results"][candidate.monomer_id] = result_record
                        result_rows.append((cache_key, result_record))
                        total_alignment_runs += 1
                        completed_keys.append(cache_key)
                    except Exception as exc:
                        error_text = str(exc)
                        state["round_failures"][candidate.monomer_id] = error_text
                        error_rows.append((cache_key, error_text))
                        total_alignment_failures += 1
                        failed_keys.append(cache_key)
                    state["round_done"] += 1
                    progress.update(1)
                    if state["round_done"] == len(state["round_candidates"]):
                        _finish_round(state)
                cache_db.cache_write_results_many(result_rows)
                cache_db.cache_write_errors_many(error_rows)
                cache_db.task_status_batch_update_by_keys(completed_keys, "completed")
                cache_db.task_status_batch_update_by_keys(failed_keys, "failed")
            elif not future_to_item and not scheduled_items and active_state_count:
                raise RuntimeError("Structure refinement scheduler stalled with active groups")
    progress.close()

    membership_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    total_structure_clusters = 0
    for sequence_cluster_id, state in states.items():
        subcluster = state["subcluster"]
        for local_index, cluster in enumerate(state["clusters"], start=1):
            representative = cluster["representative"]
            structure_cluster_id = _protein_structure_cluster_id(
                sequence_cluster_id,
                local_index,
            )
            total_structure_clusters += 1
            assigned_rep_ids = {
                member.monomer_id for member, _ in cluster["assigned"]
            }
            cluster_member_ids = sorted(
                monomer_id
                for monomer_id, representative_id in subcluster.items()
                if representative_id in assigned_rep_ids
            )
            result_by_rep = {
                member.monomer_id: result
                for member, result in cluster["assigned"]
            }
            for member, result in cluster["assigned"]:
                if result is None:
                    continue
                alignment_rows.append(
                    {
                        "sequence_cluster_id": sequence_cluster_id,
                        "query_monomer_id": representative.monomer_id,
                        "target_monomer_id": member.monomer_id,
                        "aligned_length": result.get("aligned_length", 0),
                        "rmsd": result.get("rmsd", 0),
                        "tm_score_query": result.get("tm_score_query", 0),
                        "tm_score_target": result.get("tm_score_target", 0),
                        "tm_score_min": result.get("tm_score_min", 0),
                        "tm_score_max": result.get("tm_score_max", 0),
                        "tm_score_for_clustering": result.get("tm_score_max", 0),
                        "alignment_coverage_shorter": result.get(
                            "shorter_length_coverage",
                            0,
                        ),
                        "alignment_coverage_resolved": result.get(
                            "resolved_length_coverage",
                            0,
                        ),
                    }
                )
            for monomer_id in cluster_member_ids:
                representative_id = subcluster[monomer_id]
                result = result_by_rep.get(representative_id)
                if monomer_id == representative.monomer_id:
                    reason = "representative"
                elif monomer_id != representative_id:
                    reason = "identical_sequence_subcluster"
                else:
                    reason = "representative_alignment"
                membership_rows.append(
                    {
                        "polymer_class": "protein",
                        "sequence_cluster_id": sequence_cluster_id,
                        "structure_cluster_id": structure_cluster_id,
                        "representative_monomer_id": representative.monomer_id,
                        "member_monomer_id": monomer_id,
                        "assignment_reason": reason,
                        "tm_score_min": (
                            result.get("tm_score_min", "") if result is not None else ""
                        ),
                        "tm_score_max": (
                            result.get("tm_score_max", "") if result is not None else ""
                        ),
                        "tm_score_for_clustering": (
                            result.get("tm_score_max", "") if result is not None else ""
                        ),
                        "alignment_coverage_shorter": (
                            result.get("shorter_length_coverage", "")
                            if result is not None
                            else ""
                        ),
                        "alignment_coverage_resolved": (
                            result.get("resolved_length_coverage", "")
                            if result is not None
                            else ""
                        ),
                    }
                )
            representative_rows.append(
                {
                    "sequence_cluster_id": sequence_cluster_id,
                    "structure_cluster_id": structure_cluster_id,
                    "representative_monomer_id": representative.monomer_id,
                    "num_members": len(cluster_member_ids),
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
                        else ""
                    ),
                }
            )

    membership_rows.sort(
        key=lambda row: (
            row["sequence_cluster_id"],
            row["structure_cluster_id"],
            row["member_monomer_id"],
        )
    )
    representative_rows.sort(
        key=lambda row: (
            row["sequence_cluster_id"],
            row["structure_cluster_id"],
        )
    )
    cache_db.close()

    manifest = {
        "num_sequence_clusters": len(group_member_map),
        "num_sequence_groups": len(group_member_map),
        "num_structure_clusters": total_structure_clusters,
        "num_alignment_runs": total_alignment_runs,
        "num_alignment_failures": total_alignment_failures,
        "num_sequence_fallback_assignments": 0,
        "num_all_failure_sequence_cluster_collapses": 0,
        "num_membership_rows": len(membership_rows),
        "num_extracted": extracted_count,
        "num_extraction_failures": extraction_failures,
        "num_structure_members_skipped_missing_coordinates": extraction_failures,
        "num_tasks_generated": total_tasks_generated,
        "num_cache_hits": total_cache_hits,
        "tm_score_threshold": tm_score_threshold,
        "min_alignment_coverage_ratio": min_alignment_coverage_ratio,
        "sequence_cluster_jobs": int(sequence_cluster_jobs),
        "pairwise_alignment_jobs": pairwise_alignment_jobs,
    }

    dump_csv_rows(outdir / "protein_structure_cluster_membership.csv", membership_rows)
    dump_csv_rows(outdir / "protein_structure_cluster_representatives.csv", representative_rows)
    dump_jsonl(outdir / "protein_structure_pairwise_alignments.jsonl", alignment_rows)
    dump_jsonl(outdir / "protein_structure_cluster_warnings.jsonl", warning_rows)
    dump_json(outdir / "protein_structure_cluster_manifest.json", manifest, indent=2)
    return {
        "manifest": manifest,
        "membership_rows": membership_rows,
        "representative_rows": representative_rows,
        "alignment_rows": alignment_rows,
        "warning_rows": warning_rows,
        "sequence_groups": group_member_map,
        "monomer_index": monomer_index_light,
    }
