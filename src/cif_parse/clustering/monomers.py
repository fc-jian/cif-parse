from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from cif_parse.constants import POLYMER_CHAIN_TYPES, PROTEIN_CHAIN_TYPES
from cif_parse.clustering.common import discover_case_output_dirs
from cif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from cif_parse.settings import resolve_source_path


LOGGER = logging.getLogger(__name__)

MMSEQS_CLUSTER_TSV_SUFFIX = "_cluster.tsv"


def _assembly_sort_key(label: str) -> tuple[int, int | str]:
    if label.isdigit():
        return (0, int(label))
    return (1, label)


def _canonical_monomer_id(pdb_id: str, label_asym_id: str) -> str:
    return f"{pdb_id}:{label_asym_id}"


def _format_cluster_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index}"


def _normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


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
    observed_assembly_ids: list[str] = field(default_factory=list)

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
            "observed_assembly_ids": json.dumps(self.observed_assembly_ids, ensure_ascii=False),
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
    prep_db_path: str | Path | None = None,
) -> MonomerInventoryResult:
    """Collect canonical monomer samples, deduplicated by `(pdb_id, label_asym_id)`."""

    from cif_parse.clustering.prep import load_bundles_for_collect, load_case_bundles

    case_paths = [Path(path).resolve() for path in case_dirs]
    prep_bundles = load_bundles_for_collect(case_paths, prep_db_path=prep_db_path)
    canonical: dict[tuple[str, str], MonomerSample] = {}
    counts_by_polymer_class: dict[str, int] = defaultdict(int)
    num_case_bundles = 0
    polymer_observations = 0
    eligible_observations = 0
    skipped_missing_sequence = 0
    skipped_unsupported_polymer_class = 0
    skipped_missing_identity = 0

    for case_path in tqdm(case_paths, desc="Collecting canonical monomers", unit="case"):
        payloads = load_case_bundles(case_path, prep_bundles=prep_bundles)
        num_case_bundles += len(payloads)
        for payload in payloads:
            summary = payload.get("structure_summary")
            chain_inventory = payload.get("chain_inventory")
            if not isinstance(summary, dict) or not isinstance(chain_inventory, list):
                raise TypeError(f"Invalid case bundle under {case_path}")
            assembly_ids = [
                str(item)
                for item in summary.get("assembly_ids", [])
                if str(item)
            ]
            source_path = resolve_source_path(
                str(summary.get("source_path", "") or ""),
                cif_files_directory,
            )
            bundle_pdb_id = str(summary.get("pdb_id", "") or "")
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
                    LOGGER.warning(
                        "Skipping chain without canonical identity in %s: pdb_id=%r label_asym_id=%r",
                        case_path,
                        pdb_id,
                        label_asym_id,
                    )
                    continue
                eligible_observations += 1
                key = (pdb_id, label_asym_id)
                if key not in canonical:
                    sample = MonomerSample(
                        monomer_id=_canonical_monomer_id(pdb_id, label_asym_id),
                        pdb_id=pdb_id,
                        label_asym_id=label_asym_id,
                        auth_asym_id=(
                            str(chain_payload.get("auth_asym_id"))
                            if chain_payload.get("auth_asym_id") is not None
                            else None
                        ),
                        entity_id=str(chain_payload.get("entity_id", "") or ""),
                        entity_type=str(chain_payload.get("entity_type", "") or ""),
                        entity_description=(
                            str(chain_payload.get("entity_description"))
                            if chain_payload.get("entity_description") is not None
                            else None
                        ),
                        polymer_type=(
                            str(chain_payload.get("polymer_type"))
                            if chain_payload.get("polymer_type") is not None
                            else None
                        ),
                        polymer_class=polymer_class,
                        chain_type=str(chain_payload.get("chain_type", "") or ""),
                        subtype=(
                            str(chain_payload.get("subtype"))
                            if chain_payload.get("subtype") is not None
                            else None
                        ),
                        sequence=sequence,
                        length=int(chain_payload.get("length", len(sequence)) or len(sequence)),
                        residue_count=int(
                            chain_payload.get("residue_count", chain_payload.get("length", len(sequence)))
                            or len(sequence)
                        ),
                        atom_count=int(chain_payload.get("atom_count", 0) or 0),
                        source_path=source_path,
                        source_case_dir=str(case_path),
                        parsed_coordinate_segments=list(
                            chain_payload.get("parsed_coordinate_segments", []) or []
                        ),
                        unresolved_sequence_segments=list(
                            chain_payload.get("unresolved_sequence_segments", []) or []
                        ),
                        special_residue_details=list(
                            chain_payload.get("special_residue_details", []) or []
                        ),
                        special_component_details=list(
                            chain_payload.get("special_component_details", []) or []
                        ),
                        observed_assembly_ids=[],
                    )
                    canonical[key] = sample
                    counts_by_polymer_class[polymer_class] += 1
                sample = canonical[key]
                sample.observed_assembly_ids = sorted(
                    {*(sample.observed_assembly_ids), *assembly_ids},
                    key=_assembly_sort_key,
                )

    monomers = sorted(
        canonical.values(),
        key=lambda item: (item.polymer_class, item.pdb_id, item.label_asym_id),
    )
    manifest = {
        "num_input_case_dirs": len(case_paths),
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
    for monomer in monomers:
        grouped[monomer.polymer_class].append(monomer)

    fasta_dir = Path(outdir) / "fasta"
    outputs: dict[str, Path] = {}
    for polymer_class, members in sorted(grouped.items()):
        fasta_path = fasta_dir / f"{polymer_class}.fasta"
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        with fasta_path.open("w", encoding="utf-8") as handle:
            for member in sorted(members, key=lambda item: item.monomer_id):
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
    for monomer in monomers:
        if monomer.polymer_class == polymer_class:
            grouped[monomer.sequence].append(monomer)

    rows: list[dict[str, Any]] = []
    sequences = sorted(
        grouped.items(),
        key=lambda item: (-len(item[0]), item[0], [member.monomer_id for member in item[1]]),
    )
    for index, (_, members) in enumerate(sequences, start=1):
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
    for index, member in enumerate(members, start=1):
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
    for line in Path(cluster_tsv_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        representative, member = line.split("\t", maxsplit=1)
        grouped[representative].append(member)

    rows: list[dict[str, Any]] = []
    for index, representative_id in enumerate(sorted(grouped), start=1):
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
    prep_db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build canonical monomer inventory and sequence-level grouping artifacts."""

    case_dirs = discover_case_output_dirs(inputs)
    inventory = collect_canonical_monomers(
        case_dirs,
        cif_files_directory=cif_files_directory,
        prep_db_path=prep_db_path,
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
                copied_tsv.write_text(Path(cluster_tsv).read_text(encoding="utf-8"), encoding="utf-8")
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
