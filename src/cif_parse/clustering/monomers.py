from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from cif_parse.constants import POLYMER_CHAIN_TYPES, PROTEIN_CHAIN_TYPES
from cif_parse.clustering.common import discover_case_output_dirs
from cif_parse.clustering.prep import iter_parquet_rows, open_prep_parquet
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.settings import normalize_cutoff_date, resolve_source_path


LOGGER = logging.getLogger(__name__)

MMSEQS_CLUSTER_TSV_SUFFIX = "_cluster.tsv"


def _assembly_sort_key(label: str) -> tuple[int, int | str]:
    if label.isdigit():
        return (0, int(label))
    return (1, label)


def _canonical_monomer_id(pdb_id: str, label_asym_id: str, assembly_id: str = "") -> str:
    if assembly_id:
        return f"{pdb_id}:{assembly_id}:{label_asym_id}"
    return f"{pdb_id}:{label_asym_id}"


def _format_cluster_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index}"


def _normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def _parse_release_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or len(text) != 10 or text[4] != "-" or text[7] != "-":
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _new_release_filter_stats(cutoff_date: str | None) -> dict[str, Any]:
    return {
        "enabled": cutoff_date is not None,
        "cutoff_date": cutoff_date or "",
        "comparison": "release_date < cutoff_date" if cutoff_date is not None else "",
        "num_source_files_seen": 0,
        "num_source_files_included": 0,
        "num_source_files_excluded_missing_release_date": 0,
        "num_source_files_excluded_invalid_release_date": 0,
        "num_source_files_excluded_on_or_after_cutoff_date": 0,
    }


def _record_release_filter_decision(stats: dict[str, Any], *, include: bool, reason: str) -> None:
    stats["num_source_files_seen"] += 1
    if include:
        stats["num_source_files_included"] += 1
    elif reason == "missing":
        stats["num_source_files_excluded_missing_release_date"] += 1
    elif reason == "invalid":
        stats["num_source_files_excluded_invalid_release_date"] += 1
    elif reason == "on_or_after_cutoff":
        stats["num_source_files_excluded_on_or_after_cutoff_date"] += 1


def _release_filter_allows_source(
    *,
    source_key: str,
    release_date_value: Any,
    cutoff_date_value: date | None,
    decisions: dict[str, tuple[bool, str]],
    stats: dict[str, Any],
) -> bool:
    if cutoff_date_value is None:
        return True
    if source_key in decisions:
        return decisions[source_key][0]
    release_date = _parse_release_date(release_date_value)
    if release_date is None:
        text = str(release_date_value or "").strip()
        reason = "missing" if not text else "invalid"
        decisions[source_key] = (False, reason)
        _record_release_filter_decision(stats, include=False, reason=reason)
        return False
    include = release_date < cutoff_date_value
    reason = "included" if include else "on_or_after_cutoff"
    decisions[source_key] = (include, reason)
    _record_release_filter_decision(stats, include=include, reason=reason)
    return include


def _is_polymer_chain(chain_payload: dict[str, Any]) -> bool:
    return str(chain_payload.get("chain_type", "")) in POLYMER_CHAIN_TYPES


def _classify_polymer_class(chain_payload: dict[str, Any]) -> str | None:
    chain_type = str(chain_payload.get("chain_type", ""))
    polymer_type = str(chain_payload.get("polymer_type", "") or "").lower()
    if chain_type in PROTEIN_CHAIN_TYPES:
        return "protein"
    if chain_type == "DNA chain":
        return "dna"
    if chain_type == "RNA chain":
        return "rna"
    if chain_type == "other nucleic acid chain":
        if "deoxyribo" in polymer_type:
            return "dna"
        if "ribo" in polymer_type:
            return "rna"
        return "other_nucleic_acid"
    if chain_type == "other polymer chain":
        if "deoxyribo" in polymer_type:
            return "dna"
        if "ribo" in polymer_type:
            return "rna"
        if "peptide" in polymer_type:
            return "protein"
    return None


@dataclass(slots=True)
class MonomerSample:
    monomer_id: str
    pdb_id: str
    label_asym_id: str
    auth_asym_id: str | None
    entity_id: str
    entity_type: str
    entity_description: str | None
    polymer_type: str | None
    polymer_class: str
    chain_type: str
    subtype: str | None
    sequence: str
    length: int
    residue_count: int
    atom_count: int
    source_path: str
    source_case_dir: str
    parsed_coordinate_segments: list[dict[str, Any]] = field(default_factory=list)
    unresolved_sequence_segments: list[dict[str, Any]] = field(default_factory=list)
    special_residue_details: list[dict[str, Any]] = field(default_factory=list)
    special_component_details: list[dict[str, Any]] = field(default_factory=list)
    assembly_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        return {
            "monomer_id": self.monomer_id,
            "pdb_id": self.pdb_id,
            "label_asym_id": self.label_asym_id,
            "auth_asym_id": self.auth_asym_id or "",
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "entity_description": self.entity_description or "",
            "polymer_type": self.polymer_type or "",
            "polymer_class": self.polymer_class,
            "chain_type": self.chain_type,
            "subtype": self.subtype or "",
            "sequence_length": len(self.sequence),
            "length": self.length,
            "residue_count": self.residue_count,
            "atom_count": self.atom_count,
            "num_parsed_coordinate_segments": len(self.parsed_coordinate_segments),
            "num_unresolved_sequence_segments": len(self.unresolved_sequence_segments),
            "assembly_id": self.assembly_id or "",
            "source_path": self.source_path,
            "source_case_dir": self.source_case_dir,
        }


@dataclass(slots=True)
class MonomerInventoryResult:
    monomers: list[MonomerSample]
    manifest: dict[str, Any]


def collect_canonical_monomers(
    case_dirs: Iterable[str | Path],
    cif_files_directory: str | None = None,
    prep_dir: str | Path | None = None,
    cutoff_date: str | None = None,
) -> MonomerInventoryResult:
    """Collect canonical monomer samples, deduplicated by `(pdb_id, label_asym_id)`."""

    cutoff_date_text = normalize_cutoff_date(cutoff_date)
    cutoff_date_value = date.fromisoformat(cutoff_date_text) if cutoff_date_text else None
    release_filter_stats = _new_release_filter_stats(cutoff_date_text)
    release_filter_decisions: dict[str, tuple[bool, str]] = {}
    case_paths = [Path(path).resolve() for path in case_dirs]
    prep_case_ids: set[str] = set()
    canonical: dict[tuple[str, str], MonomerSample] = {}
    counts_by_polymer_class: dict[str, int] = defaultdict(int)
    num_case_bundles = 0
    polymer_observations = 0
    eligible_observations = 0
    skipped_missing_sequence = 0
    skipped_unsupported_polymer_class = 0
    skipped_missing_identity = 0

    # Fast path: read pre-parsed Parquet
    pf = open_prep_parquet(prep_dir, "monomers", required=True) if prep_dir else None
    if pf is not None:
        pf.close()
        release_dates_by_source: dict[str, Any] = {}
        if cutoff_date_value is not None:
            for quality_row in iter_parquet_rows(
                prep_dir,
                "entry_quality",
                columns=["source_path", "release_date"],
                required=False,
            ):
                source_path = resolve_source_path(
                    str(quality_row.get("source_path", "") or ""),
                    cif_files_directory,
                )
                release_dates_by_source.setdefault(source_path, quality_row.get("release_date"))
        for row in tqdm(
            iter_parquet_rows(prep_dir, "monomers", required=True),
            desc="Collecting canonical monomers",
            unit="monomer",
        ):
            polymer_observations += 1
            pdb_id = str(row.get("pdb_id", "") or "")
            label_asym_id = str(row.get("label_asym_id", "") or "")
            if not pdb_id or not label_asym_id:
                skipped_missing_identity += 1
                continue
            source_path = resolve_source_path(str(row.get("source_path", "") or ""), cif_files_directory)
            source_key = source_path or f"prep:{pdb_id}:{row.get('source_case_dir', '')}"
            if not _release_filter_allows_source(
                source_key=source_key,
                release_date_value=release_dates_by_source.get(source_path, ""),
                cutoff_date_value=cutoff_date_value,
                decisions=release_filter_decisions,
                stats=release_filter_stats,
            ):
                continue
            eligible_observations += 1
            assembly_id = str(row.get("assembly_id", "") or "")
            key = (source_path, assembly_id, pdb_id, label_asym_id)
            if key not in canonical:
                canonical[key] = MonomerSample(
                    monomer_id=_canonical_monomer_id(pdb_id, label_asym_id, assembly_id),
                    pdb_id=pdb_id,
                    label_asym_id=label_asym_id,
                    auth_asym_id=row.get("auth_asym_id"),
                    entity_id=str(row.get("entity_id", "") or ""),
                    entity_type=str(row.get("entity_type", "") or ""),
                    entity_description=row.get("entity_description"),
                    polymer_type=row.get("polymer_type"),
                    polymer_class=str(row.get("polymer_class", "") or ""),
                    chain_type=str(row.get("chain_type", "") or ""),
                    subtype=row.get("subtype"),
                    sequence=str(row.get("sequence", "") or ""),
                    length=int(row.get("length", 0) or 0),
                    residue_count=int(row.get("residue_count", 0) or 0),
                    atom_count=int(row.get("atom_count", 0) or 0),
                    source_path=source_path,
                    source_case_dir=str(row.get("source_case_dir", "") or ""),
                    parsed_coordinate_segments=json.loads(row.get("parsed_coordinate_segments", "[]") or "[]"),
                    unresolved_sequence_segments=json.loads(row.get("unresolved_sequence_segments", "[]") or "[]"),
                    special_residue_details=json.loads(row.get("special_residue_details", "[]") or "[]"),
                    special_component_details=json.loads(row.get("special_component_details", "[]") or "[]"),
                    assembly_id=assembly_id,
                )
                counts_by_polymer_class[str(row.get("polymer_class", "") or "")] += 1
            source_case_dir = str(row.get("source_case_dir", "") or "")
            if source_case_dir:
                prep_case_ids.add(source_case_dir)
        num_case_bundles = len(prep_case_ids)
    else:
        # Slow path: read individual JSON bundles
        from cif_parse.export import load_case_output_bundles as _load_bundles

        for case_path in tqdm(case_paths, desc="Collecting canonical monomers", unit="case"):
            payloads = _load_bundles(case_path)
            num_case_bundles += len(payloads)
            for payload in payloads:
                summary = payload.get("structure_summary")
                chain_inventory = payload.get("chain_inventory")
                if not isinstance(summary, dict) or not isinstance(chain_inventory, list):
                    raise TypeError(f"Invalid case bundle under {case_path}")
                assembly_ids = [str(item) for item in summary.get("assembly_ids", []) if str(item)]
                source_path = resolve_source_path(str(summary.get("source_path", "") or ""), cif_files_directory)
                bundle_pdb_id = str(summary.get("pdb_id", "") or "")
                metadata = summary.get("entry_metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                source_key = source_path or f"{case_path}:{bundle_pdb_id}"
                if not _release_filter_allows_source(
                    source_key=source_key,
                    release_date_value=metadata.get("release_date"),
                    cutoff_date_value=cutoff_date_value,
                    decisions=release_filter_decisions,
                    stats=release_filter_stats,
                ):
                    continue
                for chain_payload in chain_inventory:
                    if not isinstance(chain_payload, dict) or not _is_polymer_chain(chain_payload):
                        continue
                    polymer_observations += 1
                    polymer_class = _classify_polymer_class(chain_payload)
                    if polymer_class is None:
                        skipped_unsupported_polymer_class += 1
                        continue
                    sequence = _normalize_sequence(chain_payload.get("sequence"))
                    if not sequence:
                        skipped_missing_sequence += 1
                        continue
                    pdb_id = str(chain_payload.get("pdb_id", "") or bundle_pdb_id)
                    label_asym_id = str(chain_payload.get("label_asym_id", "") or "")
                    if not pdb_id or not label_asym_id:
                        skipped_missing_identity += 1
                        continue
                    eligible_observations += 1
                    bundle_assembly_id = str(summary.get("assembly_id", "") or "")
                    if not bundle_assembly_id and assembly_ids:
                        bundle_assembly_id = assembly_ids[0]
                    key = (source_path, bundle_assembly_id, pdb_id, label_asym_id)
                    if key not in canonical:
                        sample = MonomerSample(
                            monomer_id=_canonical_monomer_id(pdb_id, label_asym_id, bundle_assembly_id),
                            pdb_id=pdb_id,
                            label_asym_id=label_asym_id,
                            auth_asym_id=str(chain_payload.get("auth_asym_id")) if chain_payload.get("auth_asym_id") is not None else None,
                            entity_id=str(chain_payload.get("entity_id", "") or ""),
                            entity_type=str(chain_payload.get("entity_type", "") or ""),
                            entity_description=str(chain_payload.get("entity_description")) if chain_payload.get("entity_description") is not None else None,
                            polymer_type=str(chain_payload.get("polymer_type")) if chain_payload.get("polymer_type") is not None else None,
                            polymer_class=polymer_class,
                            chain_type=str(chain_payload.get("chain_type", "") or ""),
                            subtype=str(chain_payload.get("subtype")) if chain_payload.get("subtype") is not None else None,
                            sequence=sequence, length=int(chain_payload.get("length", len(sequence)) or len(sequence)),
                            residue_count=int(chain_payload.get("residue_count", chain_payload.get("length", len(sequence))) or len(sequence)),
                            atom_count=int(chain_payload.get("atom_count", 0) or 0),
                            source_path=source_path, source_case_dir=str(case_path),
                            parsed_coordinate_segments=list(chain_payload.get("parsed_coordinate_segments", []) or []),
                            unresolved_sequence_segments=list(chain_payload.get("unresolved_sequence_segments", []) or []),
                            special_residue_details=list(chain_payload.get("special_residue_details", []) or []),
                            special_component_details=list(chain_payload.get("special_component_details", []) or []),
                            assembly_id=bundle_assembly_id,
                        )
                        canonical[key] = sample
                        counts_by_polymer_class[polymer_class] += 1

    monomers = sorted(
        canonical.values(),
        key=lambda item: (item.polymer_class, item.pdb_id, item.label_asym_id),
    )
    prep_total_cases: int | None = None
    if prep_dir:
        manifest_path = Path(prep_dir) / "manifest.json"
        try:
            prep_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prep_total_cases = int(prep_manifest.get("total_cases", 0) or 0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            prep_total_cases = None
    manifest = {
        "num_input_case_dirs": prep_total_cases if prep_total_cases is not None else len(case_paths),
        "num_case_bundles": num_case_bundles,
        "num_polymer_observations": polymer_observations,
        "num_eligible_observations": eligible_observations,
        "num_canonical_monomers": len(monomers),
        "num_duplicate_observations_removed": eligible_observations - len(monomers),
        "num_skipped_missing_sequence": skipped_missing_sequence,
        "num_skipped_unsupported_polymer_class": skipped_unsupported_polymer_class,
        "num_skipped_missing_identity": skipped_missing_identity,
        "counts_by_polymer_class": dict(sorted(counts_by_polymer_class.items())),
        "input_case_dirs": [str(path) for path in case_paths],
        "cutoff_date": cutoff_date_text or "",
        "release_date_filter": release_filter_stats,
    }
    LOGGER.info(
        "Collected %d canonical monomers from %d case bundles (%d polymer observations, %d duplicates removed)",
        len(monomers),
        num_case_bundles,
        polymer_observations,
        eligible_observations - len(monomers),
    )
    return MonomerInventoryResult(monomers=monomers, manifest=manifest)


def write_monomer_fastas(monomers: Iterable[MonomerSample], outdir: str | Path) -> dict[str, Path]:
    """Write per-polymer-class FASTA files and return created paths."""

    grouped: dict[str, list[MonomerSample]] = defaultdict(list)
    monomer_list = list(monomers)
    for monomer in tqdm(monomer_list, desc="Grouping monomers for FASTA", unit="monomer"):
        grouped[monomer.polymer_class].append(monomer)

    fasta_dir = Path(outdir) / "fasta"
    outputs: dict[str, Path] = {}
    for polymer_class, members in tqdm(
        sorted(grouped.items()),
        desc="Writing monomer FASTA files",
        unit="class",
        disable=len(grouped) < 2,
    ):
        fasta_path = fasta_dir / f"{polymer_class}.fasta"
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        with fasta_path.open("w", encoding="utf-8") as handle:
            for member in tqdm(
                sorted(members, key=lambda item: item.monomer_id),
                desc=f"Writing {polymer_class} FASTA",
                unit="sequence",
                leave=False,
                disable=len(members) < 2,
            ):
                header = "|".join(
                    [member.monomer_id]
                )
                handle.write(f">{header}\n")
                for start in range(0, len(member.sequence), 80):
                    handle.write(member.sequence[start : start + 80])
                    handle.write("\n")
        outputs[polymer_class] = fasta_path
    return outputs


def build_exact_sequence_membership(
    monomers: Iterable[MonomerSample],
    *,
    polymer_class: str,
    cluster_prefix: str,
    cluster_method: str = "exact_sequence",
) -> list[dict[str, Any]]:
    """Group monomers by exact sequence equality."""

    grouped: dict[str, list[MonomerSample]] = defaultdict(list)
    for monomer in tqdm(
        list(monomers),
        desc=f"Grouping {polymer_class} exact sequences",
        unit="monomer",
    ):
        if monomer.polymer_class == polymer_class:
            grouped[monomer.sequence].append(monomer)

    rows: list[dict[str, Any]] = []
    sequences = sorted(
        grouped.items(),
        key=lambda item: (-len(item[0]), item[0], [member.monomer_id for member in item[1]]),
    )
    for index, (_, members) in enumerate(
        tqdm(sequences, desc=f"Building {polymer_class} exact clusters", unit="cluster"),
        start=1,
    ):
        cluster_id = _format_cluster_id(cluster_prefix, index)
        representative = sorted(members, key=lambda item: item.monomer_id)[0]
        for member in sorted(members, key=lambda item: item.monomer_id):
            rows.append(
                {
                    "cluster_stage": "sequence",
                    "cluster_method": cluster_method,
                    "polymer_class": polymer_class,
                    "cluster_id": cluster_id,
                    "representative_monomer_id": representative.monomer_id,
                    "member_monomer_id": member.monomer_id,
                    "pdb_id": member.pdb_id,
                    "label_asym_id": member.label_asym_id,
                    "auth_asym_id": member.auth_asym_id or "",
                    "chain_type": member.chain_type,
                    "sequence_length": len(member.sequence),
                }
            )
    return rows


def build_singleton_protein_membership(monomers: Iterable[MonomerSample]) -> list[dict[str, Any]]:
    """Emit singleton protein sequence buckets when sequence clustering is skipped."""

    rows: list[dict[str, Any]] = []
    members = sorted(
        (monomer for monomer in monomers if monomer.polymer_class == "protein"),
        key=lambda item: item.monomer_id,
    )
    for index, member in enumerate(
        tqdm(members, desc="Building singleton protein clusters", unit="monomer"),
        start=1,
    ):
        rows.append(
            {
                "cluster_stage": "sequence",
                "cluster_method": "singleton",
                "polymer_class": "protein",
                "cluster_id": _format_cluster_id("prot", index),
                "representative_monomer_id": member.monomer_id,
                "member_monomer_id": member.monomer_id,
                "pdb_id": member.pdb_id,
                "label_asym_id": member.label_asym_id,
                "auth_asym_id": member.auth_asym_id or "",
                "chain_type": member.chain_type,
                "sequence_length": len(member.sequence),
            }
        )
    return rows


def run_mmseqs_sequence_clustering(
    fasta_path: str | Path,
    *,
    outdir: str | Path,
    min_seq_id: float = 0.40,
    coverage: float = 0.80,
    cov_mode: int = 5,
    threads: int = 1,
) -> Path:
    """Run `mmseqs easy-cluster` and return the generated cluster TSV path."""

    if shutil.which("mmseqs") is None:
        raise FileNotFoundError("mmseqs executable not found in PATH")

    fasta_path = Path(fasta_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_prefix = outdir / "mmseqs_result"
    tmp_dir = outdir / "tmp"
    command = [
        "mmseqs",
        "easy-cluster",
        str(fasta_path),
        str(result_prefix),
        str(tmp_dir),
        "--min-seq-id",
        f"{min_seq_id:.4f}",
        "-c",
        f"{coverage:.4f}",
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(max(1, int(threads))),
    ]
    LOGGER.info("Running mmseqs easy-cluster on %s with %d threads", fasta_path, max(1, int(threads)))
    subprocess.run(command, check=True)
    cluster_tsv = Path(f"{result_prefix}{MMSEQS_CLUSTER_TSV_SUFFIX}")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Expected mmseqs cluster TSV not found: {cluster_tsv}")
    return cluster_tsv


def parse_mmseqs_cluster_tsv(
    cluster_tsv_path: str | Path,
    monomer_index: dict[str, MonomerSample],
) -> list[dict[str, Any]]:
    """Parse `mmseqs easy-cluster` TSV output into normalized membership rows."""

    grouped: dict[str, list[str]] = defaultdict(list)
    with Path(cluster_tsv_path).open(encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Parsing mmseqs clusters", unit="row"):
            text = line.strip()
            if not text:
                continue
            representative, member = text.split("\t", maxsplit=1)
            grouped[representative].append(member)

    rows: list[dict[str, Any]] = []
    for index, representative_id in enumerate(
        tqdm(sorted(grouped), desc="Normalizing mmseqs clusters", unit="cluster"),
        start=1,
    ):
        cluster_id = _format_cluster_id("prot", index)
        representative = monomer_index[representative_id]
        for member_id in sorted(grouped[representative_id]):
            member = monomer_index[member_id]
            rows.append(
                {
                    "cluster_stage": "sequence",
                    "cluster_method": "mmseqs2",
                    "polymer_class": "protein",
                    "cluster_id": cluster_id,
                    "representative_monomer_id": representative.monomer_id,
                    "member_monomer_id": member.monomer_id,
                    "pdb_id": member.pdb_id,
                    "label_asym_id": member.label_asym_id,
                    "auth_asym_id": member.auth_asym_id or "",
                    "chain_type": member.chain_type,
                    "sequence_length": len(member.sequence),
                }
            )
    return rows


def build_monomer_sequence_dataset(
    *,
    inputs: Iterable[str | Path],
    outdir: str | Path,
    protein_sequence_mode: str = "skip",
    protein_min_seq_id: float = 0.40,
    protein_coverage: float = 0.80,
    protein_cov_mode: int = 5,
    mmseqs_threads: int = 1,
    cif_files_directory: str | None = None,
    prep_dir: str | Path | None = None,
    cutoff_date: str | None = None,
) -> dict[str, Any]:
    """Build canonical monomer inventory and sequence-level grouping artifacts."""

    cutoff_date_text = normalize_cutoff_date(cutoff_date)
    case_dirs = [] if prep_dir else discover_case_output_dirs(inputs)
    inventory = collect_canonical_monomers(
        case_dirs,
        cif_files_directory=cif_files_directory,
        prep_dir=prep_dir,
        cutoff_date=cutoff_date_text,
    )
    monomers = inventory.monomers
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dump_jsonl(outdir / "monomer_inventory.jsonl", [monomer.to_dict() for monomer in monomers])
    dump_csv_rows(outdir / "summary.csv", [monomer.to_record() for monomer in monomers])
    fasta_paths = write_monomer_fastas(monomers, outdir)

    membership_rows: list[dict[str, Any]] = []
    if protein_sequence_mode == "skip":
        membership_rows.extend(build_singleton_protein_membership(monomers))
    elif protein_sequence_mode == "exact":
        membership_rows.extend(
            build_exact_sequence_membership(
                monomers,
                polymer_class="protein",
                cluster_prefix="prot",
            )
        )
    elif protein_sequence_mode == "mmseqs2":
        protein_fasta = fasta_paths.get("protein")
        if protein_fasta is None:
            LOGGER.info("No protein monomers detected; skipping mmseqs sequence clustering")
        else:
            with tempfile.TemporaryDirectory(prefix="cif_parse_mmseqs_") as tmpdir:
                cluster_tsv = run_mmseqs_sequence_clustering(
                    protein_fasta,
                    outdir=Path(tmpdir),
                    min_seq_id=protein_min_seq_id,
                    coverage=protein_coverage,
                    cov_mode=protein_cov_mode,
                    threads=mmseqs_threads,
                )
                copied_tsv = outdir / "sequence_clusters" / "protein_mmseqs_cluster.tsv"
                copied_tsv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cluster_tsv, copied_tsv)
                monomer_index = {monomer.monomer_id: monomer for monomer in monomers}
                membership_rows.extend(parse_mmseqs_cluster_tsv(copied_tsv, monomer_index))
    else:
        raise ValueError(f"Unsupported protein_sequence_mode: {protein_sequence_mode}")

    membership_rows.extend(
        build_exact_sequence_membership(monomers, polymer_class="dna", cluster_prefix="dna")
    )
    membership_rows.extend(
        build_exact_sequence_membership(monomers, polymer_class="rna", cluster_prefix="rna")
    )
    membership_rows.extend(
        build_exact_sequence_membership(
            monomers,
            polymer_class="other_nucleic_acid",
            cluster_prefix="na",
        )
    )

    dump_csv_rows(outdir / "sequence_clusters" / "membership.csv", membership_rows)

    manifest = {
        **inventory.manifest,
        "protein_sequence_mode": protein_sequence_mode,
        "protein_sequence_parameters": {
            "min_seq_id": protein_min_seq_id,
            "coverage": protein_coverage,
            "cov_mode": protein_cov_mode,
            "threads": max(1, int(mmseqs_threads)),
        },
        "cutoff_date": cutoff_date_text or "",
        "output_dir": str(outdir.resolve()),
        "fasta_paths": {key: str(path.resolve()) for key, path in sorted(fasta_paths.items())},
        "num_sequence_membership_rows": len(membership_rows),
    }
    dump_json(outdir / "manifest.json.gz", manifest, indent=2)
    LOGGER.info(
        "Wrote sequence membership: %d rows, protein mode=%s",
        len(membership_rows),
        protein_sequence_mode,
    )
    return {
        "case_dirs": case_dirs,
        "monomers": monomers,
        "membership_rows": membership_rows,
        "manifest": manifest,
    }
