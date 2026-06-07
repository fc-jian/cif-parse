from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cif_parse.clustering.antibody_complexes import build_antibody_complex_signature_clusters
from cif_parse.clustering.common import discover_case_output_dirs
from cif_parse.clustering.dimers import build_dimer_signature_clusters
from cif_parse.clustering.monomers import MonomerSample, build_monomer_sequence_dataset
from cif_parse.clustering.multimers import build_multimer_signature_clusters
from cif_parse.clustering.protein_structures import greedy_cluster_protein_structures
from cif_parse.clustering.tcr_complexes import build_tcr_complex_signature_clusters
from cif_parse.export import load_json
from cif_parse.settings import (
    DEFAULT_CLUSTERING_OUTDIR,
    SUPPORTED_CLUSTERING_OBJECT_MODES,
    SUPPORTED_CLUSTERING_SEQUENCE_MODES,
    SUPPORTED_CLUSTERING_STRUCTURE_MODES,
    SUPPORTED_LOG_LEVELS,
    get_fast_temp_dir,
    load_clustering_cli_config,
    normalize_cutoff_date,
)
from cif_parse.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


class SequenceCutoffMismatchError(ValueError):
    """Raised when downstream clustering uses sequence outputs from a different cutoff."""

CLUSTER_STAGE_SUBCOMMANDS = {"seq", "structure", "dimer", "multimer", "tcr", "abag"}
STAGE_HELP_TEXT: dict[str, str] = {
    "seq": (
        "Stage `seq`\n"
        "Build only monomer sequence clustering artifacts.\n"
        "Inputs: prep directory (recommended) or --inputs.\n"
        "Outputs: monomer_inventory.jsonl, summary.csv, fasta/, sequence_clusters/.\n"
        "This stage must run first."
    ),
    "structure": (
        "Stage `structure`\n"
        "Build only protein monomer structure clustering.\n"
        "Requires: completed `seq` outputs in --outdir.\n"
        "Inputs: prep directory plus seq outputs.\n"
        "Outputs: structure_clusters/."
    ),
    "dimer": (
        "Stage `dimer`\n"
        "Build only dimer clustering.\n"
        "Requires by default: completed `seq` + `structure` outputs in --outdir.\n"
        "If --ignore-structure is set, signature building falls back to sequence clusters only.\n"
        "Outputs: dimer_clusters/."
    ),
    "multimer": (
        "Stage `multimer`\n"
        "Build only multimer clustering.\n"
        "Requires by default: completed `seq` + `structure` outputs in --outdir.\n"
        "If --ignore-structure is set, signature building falls back to sequence clusters only.\n"
        "Outputs: multimer_clusters/."
    ),
    "tcr": (
        "Stage `tcr`\n"
        "Build only TCR-pMHC clustering.\n"
        "Requires by default: completed `seq` + `structure` outputs in --outdir.\n"
        "If --ignore-structure is set, signature building falls back to sequence clusters only.\n"
        "Outputs: tcr_complex_clusters/."
    ),
    "abag": (
        "Stage `abag`\n"
        "Build only antibody-antigen clustering.\n"
        "Requires by default: completed `seq` + `structure` outputs in --outdir.\n"
        "If --ignore-structure is set, signature building falls back to sequence clusters only.\n"
        "Outputs: antibody_complex_clusters/."
    ),
}


def _split_stage_subcommand(argv: list[str] | None) -> tuple[str | None, list[str] | None]:
    if argv is None:
        return None, None
    if argv and argv[0] in CLUSTER_STAGE_SUBCOMMANDS:
        return argv[0], argv[1:]
    if len(argv) >= 3 and argv[0] == "--config" and argv[2] in CLUSTER_STAGE_SUBCOMMANDS:
        return argv[2], [argv[0], argv[1], *argv[3:]]
    if len(argv) >= 2 and argv[0].startswith("--config=") and argv[1] in CLUSTER_STAGE_SUBCOMMANDS:
        return argv[1], [argv[0], *argv[2:]]
    return None, argv


def _normalize_legacy_positional_inputs(argv: list[str] | None) -> list[str] | None:
    """Treat legacy leading positional paths as --inputs values."""

    if argv is None or not argv:
        return argv
    first = argv[0]
    if first in CLUSTER_STAGE_SUBCOMMANDS or first == "prep" or first.startswith("-"):
        return argv
    inputs: list[str] = []
    rest_start = 0
    for index, token in enumerate(argv):
        if token.startswith("-") or token in CLUSTER_STAGE_SUBCOMMANDS or token == "prep":
            rest_start = index
            break
        inputs.append(token)
    else:
        rest_start = len(argv)
    if not inputs:
        return argv
    return ["--inputs", *inputs, *argv[rest_start:]]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_sequence_manifest(outdir: Path) -> dict[str, Any]:
    manifest_path = outdir / "manifest.json.gz"
    if not manifest_path.exists():
        manifest_path = outdir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    return manifest if isinstance(manifest, dict) else {}


def _load_sequence_dataset_from_outdir(outdir: Path, *, inputs: list[Path], prep_dir: str | None) -> dict[str, Any]:
    inventory_path = outdir / "monomer_inventory.jsonl"
    membership_path = outdir / "sequence_clusters" / "membership.csv"
    if not inventory_path.exists() or not membership_path.exists():
        raise FileNotFoundError(
            "Sequence clustering artifacts are missing. Run `cif-parse-cluster seq` first "
            f"for outdir {outdir}."
        )
    monomers = [MonomerSample(**row) for row in _load_jsonl(inventory_path)]
    case_dirs = [] if prep_dir else discover_case_output_dirs(inputs)
    return {
        "case_dirs": case_dirs,
        "monomers": monomers,
        "membership_rows": _load_csv_rows(membership_path),
        "manifest": _load_sequence_manifest(outdir),
    }


def _sequence_dataset_cutoff_date(sequence_dataset: dict[str, Any]) -> str | None:
    manifest = sequence_dataset.get("manifest", {})
    if not isinstance(manifest, dict):
        return None
    try:
        return normalize_cutoff_date(manifest.get("cutoff_date") or None)
    except ValueError as exc:
        raise SequenceCutoffMismatchError(
            "Refusing to use sequence clustering artifacts with invalid cutoff_date "
            "in sequence manifest. Re-run `cif-parse-cluster seq`."
        ) from exc


def _validate_sequence_dataset_cutoff(args: argparse.Namespace, sequence_dataset: dict[str, Any]) -> None:
    expected = normalize_cutoff_date(getattr(args, "cutoff_date", None))
    actual = _sequence_dataset_cutoff_date(sequence_dataset)
    if actual == expected:
        return
    expected_text = expected or "<none>"
    actual_text = actual or "<none>"
    raise SequenceCutoffMismatchError(
        "Refusing to use sequence clustering artifacts with non-identical cutoff-date: "
        f"sequence manifest has {actual_text}, current command has {expected_text}. "
        "Re-run `cif-parse-cluster seq` with the same --cutoff-date or use a matching --outdir."
    )


def _resolve_worker_counts(args: argparse.Namespace, config_defaults: dict[str, Any]) -> None:
    if args.mmseqs_threads is None:
        args.mmseqs_threads = config_defaults.get("clustering", {}).get("mmseqs_threads") or args.jobs
    if args.sequence_cluster_jobs is None:
        args.sequence_cluster_jobs = config_defaults.get("clustering", {}).get("sequence_cluster_jobs") or args.jobs
    if args.dimer_extraction_jobs is None:
        args.dimer_extraction_jobs = (
            config_defaults.get("clustering", {}).get("dimer_extraction_jobs") or args.jobs
        )
    if args.usalign_jobs is None:
        args.usalign_jobs = config_defaults.get("clustering", {}).get("usalign_jobs") or args.jobs


def _set_shared_prep_index(prep_dir: str | None) -> None:
    if prep_dir:
        from cif_parse.clustering.prep import load_cif_coords_index, set_shared_coord_index

        shared = load_cif_coords_index(prep_dir)
        set_shared_coord_index(shared, prep_dir)


def _close_prep_handles(prep_dir: str | None) -> None:
    if prep_dir:
        from cif_parse.clustering.prep import close_blob_handles

        close_blob_handles()


def _full_command_needs_coordinates(args: argparse.Namespace) -> bool:
    if args.protein_structure_mode == "greedy":
        return True
    return any(
        (
            args.dimer_mode == "signature" and args.dimer_structure_mode == "greedy",
            args.multimer_mode == "signature" and args.multimer_structure_mode == "greedy",
            args.antibody_complex_mode == "signature" and args.antibody_complex_structure_mode == "greedy",
            args.tcr_complex_mode == "signature" and args.tcr_complex_structure_mode == "greedy",
        )
    )


def _ensure_prep_for_full_command(args: argparse.Namespace) -> str | None:
    prep_dir = _prep_dir(args)
    if prep_dir is not None or args.inputs is None or not _full_command_needs_coordinates(args):
        return prep_dir
    if not all(Path(item).exists() for item in args.inputs):
        return prep_dir

    from cif_parse.clustering.prep import build_prep_database
    from cif_parse.settings import get_fast_temp_dir

    temp_prep_dir = get_fast_temp_dir("cluster_auto_prep")
    LOGGER.warning(
        "Full clustering with structure/high-order stages now consumes prep only; "
        "building temporary prep directory %s from --inputs.",
        temp_prep_dir,
    )
    build_prep_database(
        inputs=args.inputs,
        prep_dir=temp_prep_dir,
        prep_jobs=args.jobs,
        load_cif_cache=False,
    )
    args.prep_dir = temp_prep_dir
    return str(temp_prep_dir)


def build_parser(
    config_defaults: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> argparse.ArgumentParser:
    config_defaults = config_defaults or {}
    clustering_defaults = config_defaults.get("clustering", {})
    parser = argparse.ArgumentParser(
        description=(
            "Build monomer and higher-order clustering artifacts from cif-parse "
            "case outputs or a prebuilt clustering prep directory. Stage subcommands "
            "are supported as the first positional argument: seq, structure, dimer, "
            "multimer, tcr, abag."
        ),
        epilog=(
            "Split workflow:\n"
            "  1. cif-parse-cluster seq ...\n"
            "  2. cif-parse-cluster structure ...\n"
            "  3. run dimer/multimer/tcr/abag in parallel\n\n"
            "By default downstream high-order stages require both sequence and "
            "structure monomer assignments. Pass --ignore-structure only when "
            "you intentionally want sequence-only signatures."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    # "prep" subcommand
    prep_parser = subparsers.add_parser("prep", help="Build (or refresh) the clustering prep database")
    prep_parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more case-output directories or parents containing case-output directories",
    )
    prep_parser.add_argument(
        "--prep-dir",
        type=Path,
        default=Path("clustering_prep"),
        help="Output directory for prep files (Parquet + cif_coords)",
    )
    prep_parser.add_argument(
        "--cif-files-directory",
        type=Path,
        default=None,
        help="Deprecated and ignored; prep reads parse JSON plus parse-stage atom pkl files only",
    )
    prep_parser.add_argument(
        "--prep-jobs",
        type=int,
        default=int(clustering_defaults.get("jobs", 4)),
        help="Number of parallel workers for prep database ingestion",
    )
    prep_parser.add_argument(
        "--no-cif-cache",
        action="store_true",
        default=False,
        help="Skip building the coordinate index from parse-stage atom pkl files",
    )
    prep_parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Optional config_clustering.toml path",
    )
    # default "cluster" mode arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Optional config_clustering.toml path; CLI arguments override [clustering] values",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "One or more case-output directories or parents containing case-output "
            "directories. Not required when --prep-dir is provided."
        ),
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        default=None,
        help=(
            "Path to a prep directory built with `cif-parse-cluster prep`. When "
            "provided, clustering reads case data only from prep Parquet/cif_coords "
            "and --inputs is optional."
        ),
    )
    parser.add_argument(
        "--cif-files-directory",
        type=Path,
        default=None,
        help="Deprecated and ignored; clustering never reads original mmCIF files",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(clustering_defaults.get("outdir", DEFAULT_CLUSTERING_OUTDIR)),
        help="Directory where monomer clustering artifacts will be written",
    )
    parser.add_argument(
        "--protein-sequence-mode",
        choices=sorted(SUPPORTED_CLUSTERING_SEQUENCE_MODES),
        default=str(clustering_defaults.get("protein_sequence_mode", "mmseqs2")),
        help="How to build protein sequence buckets before structural clustering",
    )
    parser.add_argument(
        "--protein-structure-mode",
        choices=sorted(SUPPORTED_CLUSTERING_STRUCTURE_MODES),
        default=str(clustering_defaults.get("protein_structure_mode", "greedy")),
        help="How to perform protein structure clustering within sequence buckets",
    )
    parser.add_argument(
        "--protein-subcluster-by-sequence",
        action=argparse.BooleanOptionalAction,
        default=bool(clustering_defaults.get("protein_subcluster_by_sequence", True)),
        help="Pre-group monomers with identical sequences before pairwise USalign (reduces alignments by ~70%%)",
    )
    parser.add_argument(
        "--cutoff-date",
        default=clustering_defaults.get("cutoff_date"),
        help=(
            "Only include source files whose metadata release_date is earlier than "
            "this YYYY-MM-DD date. Files without release_date metadata are excluded."
        ),
    )
    parser.add_argument(
        "--ignore-structure",
        action="store_true",
        default=bool(clustering_defaults.get("ignore_structure", False)),
        help=(
            "Use only sequence cluster ids when building downstream high-order "
            "signatures. By default high-order stages require seq+structure "
            "monomer assignments and therefore require the structure stage to "
            "have completed first."
        ),
    )
    parser.add_argument(
        "--dimer-mode",
        choices=sorted(SUPPORTED_CLUSTERING_OBJECT_MODES),
        default=str(clustering_defaults.get("dimer_mode", "signature")),
        help="How to cluster dimer/interface observations after monomer clustering",
    )
    parser.add_argument(
        "--dimer-structure-mode",
        choices=sorted(SUPPORTED_CLUSTERING_STRUCTURE_MODES),
        default=str(clustering_defaults.get("dimer_structure_mode", "greedy")),
        help="How to refine signature-matched dimer/interface observations by interface-residue TM-score",
    )
    parser.add_argument(
        "--dimer-tm-score-threshold",
        type=float,
        default=float(clustering_defaults.get("dimer_tm_score_threshold", 0.50)),
        help="Minimum interface-residue max(TM(query,target), TM(target,query)) to keep two dimers together",
    )
    parser.add_argument(
        "--dimer-interface-residue-cutoff",
        type=float,
        default=float(clustering_defaults.get("dimer_interface_residue_cutoff", 8.0)),
        help="Distance cutoff in Angstroms for selecting dimer interface residues before USalign",
    )
    parser.add_argument(
        "--multimer-mode",
        choices=sorted(SUPPORTED_CLUSTERING_OBJECT_MODES),
        default=str(clustering_defaults.get("multimer_mode", "signature")),
        help="How to cluster tight multimer observations after monomer clustering",
    )
    parser.add_argument(
        "--multimer-structure-mode",
        choices=sorted(SUPPORTED_CLUSTERING_STRUCTURE_MODES),
        default=str(clustering_defaults.get("multimer_structure_mode", "greedy")),
        help="How to refine signature-matched multimers by overall complex TM-score",
    )
    parser.add_argument(
        "--multimer-tm-score-threshold",
        type=float,
        default=float(clustering_defaults.get("multimer_tm_score_threshold", 0.50)),
        help="Minimum overall multimer max(TM(query,target), TM(target,query)) to keep two multimers together",
    )
    parser.add_argument(
        "--multimer-max-atoms-for-refinement",
        type=int,
        default=int(clustering_defaults.get("multimer_max_atoms_for_refinement", 10_000)),
        help="Skip USalign refinement for multimer signature groups containing any member with more than this many atoms",
    )
    parser.add_argument(
        "--antibody-complex-mode",
        choices=sorted(SUPPORTED_CLUSTERING_OBJECT_MODES),
        default=str(clustering_defaults.get("antibody_complex_mode", "signature")),
        help="How to cluster antibody-antigen complex observations after monomer clustering",
    )
    parser.add_argument(
        "--antibody-complex-structure-mode",
        choices=sorted(SUPPORTED_CLUSTERING_STRUCTURE_MODES),
        default=str(clustering_defaults.get("antibody_complex_structure_mode", "greedy")),
        help="How to refine signature-matched antibody-antigen complexes by interface-residue TM-score",
    )
    parser.add_argument(
        "--antibody-complex-tm-score-threshold",
        type=float,
        default=float(clustering_defaults.get("antibody_complex_tm_score_threshold", 0.50)),
        help="Minimum antibody-antigen interface-residue max(TM(query,target), TM(target,query)) to keep two complexes together",
    )
    parser.add_argument(
        "--antibody-complex-interface-residue-cutoff",
        type=float,
        default=float(clustering_defaults.get("antibody_complex_interface_residue_cutoff", 20.0)),
        help=(
            "Distance cutoff in Angstroms for selecting antibody-antigen interface residues "
            "before USalign"
        ),
    )
    parser.add_argument(
        "--tcr-complex-mode",
        choices=sorted(SUPPORTED_CLUSTERING_OBJECT_MODES),
        default=str(clustering_defaults.get("tcr_complex_mode", "signature")),
        help="How to cluster TCR-pMHC complex observations after monomer clustering",
    )
    parser.add_argument(
        "--tcr-complex-structure-mode",
        choices=sorted(SUPPORTED_CLUSTERING_STRUCTURE_MODES),
        default=str(clustering_defaults.get("tcr_complex_structure_mode", "greedy")),
        help="How to refine signature-matched TCR-pMHC complexes by overall complex TM-score",
    )
    parser.add_argument(
        "--tcr-complex-tm-score-threshold",
        type=float,
        default=float(clustering_defaults.get("tcr_complex_tm_score_threshold", 0.50)),
        help="Minimum overall TCR-complex max(TM(query,target), TM(target,query)) to keep two complexes together",
    )
    parser.add_argument(
        "--protein-min-seq-id",
        type=float,
        default=float(clustering_defaults.get("protein_min_seq_id", 0.40)),
        help="Protein sequence identity threshold for mmseqs2 mode",
    )
    parser.add_argument(
        "--protein-coverage",
        type=float,
        default=float(clustering_defaults.get("protein_coverage", 0.80)),
        help="Protein sequence coverage threshold for mmseqs2 mode",
    )
    parser.add_argument(
        "--protein-cov-mode",
        type=int,
        default=int(clustering_defaults.get("protein_cov_mode", 5)),
        help="mmseqs2 coverage mode used when protein sequence mode is mmseqs2",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=int(clustering_defaults.get("model", 1)),
        help="Model index recorded in extraction outputs; coordinates are read from prep atom caches",
    )
    parser.add_argument(
        "--keep-hydrogens",
        action=argparse.BooleanOptionalAction,
        default=bool(clustering_defaults.get("keep_hydrogens", False)),
        help="Keep hydrogens when extracting monomer coordinates for USalign",
    )
    parser.add_argument(
        "--tm-score-threshold",
        type=float,
        default=float(clustering_defaults.get("tm_score_threshold", 0.50)),
        help="Minimum max(TM(query,target), TM(target,query)) for structural clustering",
    )
    parser.add_argument(
        "--min-alignment-coverage-ratio",
        type=float,
        default=float(clustering_defaults.get("min_alignment_coverage_ratio", 0.50)),
        help=(
            "Minimum aligned-length / shorter resolved-structure length ratio for monomer "
            "structure clustering; high-order refinement reports this coverage but clusters by TM-score only"
        ),
    )
    parser.add_argument(
        "--usalign-executable",
        default=str(clustering_defaults.get("usalign_executable", "USalign")),
        help="USalign executable name or absolute path",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(clustering_defaults.get("jobs", 1)),
        help="Default worker count for clustering parallel sections",
    )
    parser.add_argument(
        "--mmseqs-threads",
        type=int,
        default=clustering_defaults.get("mmseqs_threads"),
        help="Thread count passed to mmseqs easy-cluster (default: same as --jobs)",
    )
    parser.add_argument(
        "--sequence-cluster-jobs",
        type=int,
        default=clustering_defaults.get("sequence_cluster_jobs"),
        help="Number of protein sequence clusters processed concurrently (default: same as --jobs)",
    )
    parser.add_argument(
        "--dimer-extraction-jobs",
        type=int,
        default=clustering_defaults.get("dimer_extraction_jobs"),
        help="Number of dimer interface extraction workers (default: same as --jobs)",
    )
    parser.add_argument(
        "--usalign-jobs",
        type=int,
        default=clustering_defaults.get("usalign_jobs"),
        help="Maximum concurrent USalign subprocesses per refinement stage (default: same as --jobs)",
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(SUPPORTED_LOG_LEVELS),
        default=str(clustering_defaults.get("log_level", "INFO")),
        help="Root log level",
    )
    return parser


def _prep_dir(args: argparse.Namespace) -> str | None:
    return str(args.prep_dir) if getattr(args, "prep_dir", None) else None


def _cif_files_directory(args: argparse.Namespace) -> str | None:
    return None


def _warn_cif_override(args: argparse.Namespace) -> None:
    if args.cif_files_directory is not None:
        LOGGER.warning(
            "--cif-files-directory=%s is deprecated and ignored. Clustering reads "
            "only parse JSON, prep Parquet, and parse-stage atom pkl coordinate caches.",
            args.cif_files_directory,
        )


def _run_prep(args: argparse.Namespace) -> int:
    from cif_parse.clustering.prep import build_prep_database

    configure_logging("INFO")
    result = build_prep_database(
        inputs=args.inputs,
        prep_dir=args.prep_dir,
        cif_files_directory=None,
        prep_jobs=args.prep_jobs,
        load_cif_cache=False,  # deprecated — now reads directly from parse pkl
    )
    LOGGER.info("Prep complete: %s", result)
    return 0


def _run_seq(args: argparse.Namespace) -> dict[str, Any]:
    prep_dir = _prep_dir(args)
    cluster_inputs = args.inputs or []
    t0 = time.monotonic()
    if prep_dir:
        LOGGER.info("Building monomer sequence dataset from prep directory: %s", prep_dir)
    else:
        LOGGER.info("Building monomer sequence dataset from %d input(s)", len(cluster_inputs))
    sequence_dataset = build_monomer_sequence_dataset(
        inputs=cluster_inputs,
        outdir=args.outdir,
        protein_sequence_mode=args.protein_sequence_mode,
        protein_min_seq_id=args.protein_min_seq_id,
        protein_coverage=args.protein_coverage,
        protein_cov_mode=args.protein_cov_mode,
        mmseqs_threads=args.mmseqs_threads,
        cif_files_directory=_cif_files_directory(args),
        prep_dir=prep_dir,
        cutoff_date=args.cutoff_date,
    )
    manifest = sequence_dataset.get("manifest", {})
    LOGGER.info(
        "Sequence clustering complete (%.1fs): %d case dirs, %d canonical monomers, %d sequence membership rows",
        time.monotonic() - t0,
        manifest.get("num_input_case_dirs", 0) if isinstance(manifest, dict) else 0,
        manifest.get("num_canonical_monomers", 0) if isinstance(manifest, dict) else 0,
        manifest.get("num_sequence_membership_rows", 0) if isinstance(manifest, dict) else 0,
    )
    return sequence_dataset


def _load_or_use_sequence_dataset(
    args: argparse.Namespace,
    sequence_dataset: dict[str, Any] | None,
) -> dict[str, Any]:
    if sequence_dataset is not None:
        _validate_sequence_dataset_cutoff(args, sequence_dataset)
        return sequence_dataset
    loaded = _load_sequence_dataset_from_outdir(
        args.outdir,
        inputs=args.inputs or [],
        prep_dir=_prep_dir(args),
    )
    _validate_sequence_dataset_cutoff(args, loaded)
    return loaded


def _run_structure(
    args: argparse.Namespace,
    sequence_dataset: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    sequence_dataset = _load_or_use_sequence_dataset(args, sequence_dataset)
    prep_dir = _prep_dir(args)
    if args.protein_structure_mode != "greedy":
        LOGGER.info("Protein structure clustering skipped because protein_structure_mode=%s", args.protein_structure_mode)
        return None

    t1 = time.monotonic()
    protein_monomer_count = sum(1 for m in sequence_dataset["monomers"] if m.polymer_class == "protein")
    seq_cluster_count = len({
        row.get("cluster_id", row.get("sequence_cluster_id", ""))
        for row in sequence_dataset["membership_rows"]
        if row.get("polymer_class") == "protein"
    })
    structure_outdir = get_fast_temp_dir("protein_structures")
    LOGGER.info(
        "Extracting + clustering protein monomer structures "
        "(%d monomers, %d seq clusters, pipelined with %d seq-cluster workers)",
        protein_monomer_count,
        seq_cluster_count,
        args.sequence_cluster_jobs,
    )

    # Build direct pkl reader from parse atoms (primary coordinate source).
    pkl_reader = None
    if prep_dir:
        from cif_parse.clustering.atom_cache import (
            PklAtomReader,
            load_source_case_dir_map,
            resolve_cases_root,
        )
        try:
            pkl_reader = PklAtomReader(
                resolve_cases_root(prep_dir),
                load_source_case_dir_map(prep_dir),
            )
        except Exception:
            LOGGER.warning("[fallback] Failed to init PklAtomReader", exc_info=True)

    cif_idx_for_pipeline: dict | None = None
    if prep_dir:
        from cif_parse.clustering.prep import load_cif_coords_index

        cif_idx_for_pipeline = load_cif_coords_index(prep_dir)

    quality_cache: dict[str, Any] = {}
    atom_array_cache: dict[str, Any] = {}
    import threading

    extract_lock = threading.Lock()

    def _pipeline_extract(monomer) -> Any | None:
        nonlocal cif_idx_for_pipeline
        from cif_parse.clustering.protein_structures import (
            extract_protein_monomer_structure,
            load_entry_quality_metadata_from_prep,
        )

        atom_key = f"{monomer.source_path}__{monomer.label_asym_id}"
        with extract_lock:
            need_atoms = atom_key not in atom_array_cache
            need_quality = monomer.source_path not in quality_cache

        if need_atoms:
            found = None
            # Primary: direct pkl reader from parse atoms
            if pkl_reader is not None:
                for aid in (monomer.assembly_id, None):
                    if not aid:
                        continue
                    found = pkl_reader.load_chain(
                        monomer.source_path, monomer.label_asym_id, str(aid),
                    )
                    if found is not None:
                        break
                if found is None:
                    found = pkl_reader.load_chain(
                        monomer.source_path, monomer.label_asym_id, None,
                    )
            # Fallback: prep cif_coords blob index
            if found is None and cif_idx_for_pipeline is not None:
                from cif_parse.clustering.prep import load_chain_atoms, load_cif_from_prep
                for aid in (monomer.assembly_id, None):
                    if not aid:
                        continue
                    found = load_chain_atoms(
                        prep_dir, monomer.source_path, monomer.label_asym_id,
                        assembly_id=str(aid), index=cif_idx_for_pipeline,
                    )
                    if found is not None:
                        break
                if found is None:
                    found = load_chain_atoms(
                        prep_dir, monomer.source_path, monomer.label_asym_id,
                        assembly_id=None, index=cif_idx_for_pipeline,
                    )
                if found is None:
                    for aid in (monomer.assembly_id, None):
                        if not aid:
                            continue
                        ca = load_cif_from_prep(
                            prep_dir, monomer.source_path, str(aid),
                            index=cif_idx_for_pipeline,
                        )
                        if ca is not None and ca.get("atom_array") is not None:
                            found = ca["atom_array"]
                            break
            if found is not None:
                with extract_lock:
                    atom_array_cache[atom_key] = found

        if need_quality:
            if "_entry_quality" not in quality_cache:
                quality_cache["_entry_quality"] = load_entry_quality_metadata_from_prep(prep_dir)
            quality = quality_cache["_entry_quality"].get(monomer.source_path)
            with extract_lock:
                quality_cache.setdefault(monomer.source_path, quality)

        with extract_lock:
            need_atoms = atom_key not in atom_array_cache
        if need_atoms:
            raise ValueError(f"Prep coordinates missing for monomer {monomer.monomer_id}")

        with extract_lock:
            atom_array = atom_array_cache.get(atom_key)
            quality_metadata = quality_cache.get(monomer.source_path)
        try:
            return extract_protein_monomer_structure(
                monomer,
                outdir=structure_outdir,
                model=args.model,
                drop_hydrogens=not args.keep_hydrogens,
                quality_metadata=quality_metadata,
                atom_array=atom_array,
            )
        except Exception:
            return None

    result = greedy_cluster_protein_structures(
        sequence_dataset["monomers"],
        sequence_dataset["membership_rows"],
        None,
        outdir=args.outdir / "structure_clusters",
        tm_score_threshold=args.tm_score_threshold,
        min_alignment_coverage_ratio=args.min_alignment_coverage_ratio,
        usalign_executable=args.usalign_executable,
        sequence_cluster_jobs=args.sequence_cluster_jobs,
        pairwise_alignment_jobs=args.usalign_jobs,
        model=args.model,
        drop_hydrogens=not args.keep_hydrogens,
        extract_fn=_pipeline_extract,
        protein_subcluster_by_sequence=args.protein_subcluster_by_sequence,
        prep_dir=prep_dir,
        cif_idx=cif_idx_for_pipeline,
    )
    manifest = result.get("manifest", {})
    LOGGER.info(
        "Structure clustering complete (%.1fs): %d structures extracted (%d failures), "
        "%d sequence clusters -> %d structure clusters (%d alignments, %d failures)",
        time.monotonic() - t1,
        manifest.get("num_extracted", 0),
        manifest.get("num_extraction_failures", 0),
        manifest.get("num_sequence_clusters", 0),
        manifest.get("num_structure_clusters", 0),
        manifest.get("num_alignment_runs", 0),
        manifest.get("num_alignment_failures", 0),
    )
    return result


def _require_structure_assignments(args: argparse.Namespace) -> None:
    if args.ignore_structure:
        return
    structure_membership = args.outdir / "structure_clusters" / "protein_structure_cluster_membership.csv"
    if not structure_membership.exists():
        raise FileNotFoundError(
            "High-order clustering requires seq+structure monomer assignments by default. "
            f"Missing {structure_membership}. Run `cif-parse-cluster structure` first, "
            "or pass --ignore-structure to build high-order signatures from sequence clusters only."
        )


def _build_high_order_specs(args: argparse.Namespace, sequence_dataset: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    prep_dir = _prep_dir(args)
    cif_files_directory = _cif_files_directory(args)
    common_kwargs = {
        "case_dirs": sequence_dataset["case_dirs"],
        "clustering_outdir": args.outdir,
        "model": args.model,
        "drop_hydrogens": not args.keep_hydrogens,
        "usalign_executable": args.usalign_executable,
        "alignment_jobs": args.usalign_jobs,
        "min_alignment_coverage_ratio": args.min_alignment_coverage_ratio,
        "cif_files_directory": cif_files_directory,
        "prep_dir": prep_dir,
        "include_structure_assignments": not args.ignore_structure,
    }
    specs: list[tuple[str, dict[str, Any]]] = []
    if args.tcr_complex_mode == "signature":
        specs.append(("tcr", {
            **common_kwargs,
            "outdir": args.outdir / "tcr_complex_clusters",
            "structure_refinement_mode": args.tcr_complex_structure_mode,
            "tcr_complex_tm_score_threshold": args.tcr_complex_tm_score_threshold,
        }))
    if args.antibody_complex_mode == "signature":
        specs.append(("abag", {
            **common_kwargs,
            "outdir": args.outdir / "antibody_complex_clusters",
            "structure_refinement_mode": args.antibody_complex_structure_mode,
            "antibody_complex_tm_score_threshold": args.antibody_complex_tm_score_threshold,
            "antibody_complex_interface_residue_cutoff": args.antibody_complex_interface_residue_cutoff,
        }))
    if args.multimer_mode == "signature":
        specs.append(("multimer", {
            **common_kwargs,
            "outdir": args.outdir / "multimer_clusters",
            "structure_refinement_mode": args.multimer_structure_mode,
            "multimer_tm_score_threshold": args.multimer_tm_score_threshold,
            "max_atoms_for_refinement": args.multimer_max_atoms_for_refinement,
        }))
    if args.dimer_mode == "signature":
        specs.append(("dimer", {
            **common_kwargs,
            "outdir": args.outdir / "dimer_clusters",
            "structure_refinement_mode": args.dimer_structure_mode,
            "dimer_tm_score_threshold": args.dimer_tm_score_threshold,
            "dimer_interface_residue_cutoff": args.dimer_interface_residue_cutoff,
            "extraction_jobs": args.dimer_extraction_jobs,
        }))
    return specs


def _run_high_order_stage(
    args: argparse.Namespace,
    stage: str,
    sequence_dataset: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    sequence_dataset = _load_or_use_sequence_dataset(args, sequence_dataset)
    _require_structure_assignments(args)
    build_funcs = {
        "dimer": build_dimer_signature_clusters,
        "multimer": build_multimer_signature_clusters,
        "abag": build_antibody_complex_signature_clusters,
        "tcr": build_tcr_complex_signature_clusters,
    }
    specs = dict(_build_high_order_specs(args, sequence_dataset))
    if stage not in specs:
        LOGGER.info("Higher-order stage %s skipped by mode configuration", stage)
        return None
    t0 = time.monotonic()
    result = build_funcs[stage](**specs[stage])
    LOGGER.info("Higher-order stage %s completed (%.1fs)", stage, time.monotonic() - t0)
    return result


def _run_full(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    sequence_dataset = _run_seq(args)
    _run_structure(args, sequence_dataset=sequence_dataset)
    enabled = [stage for stage, _ in _build_high_order_specs(args, sequence_dataset)]
    if enabled:
        t3 = time.monotonic()
        LOGGER.info("Building higher-order clusters serially [%s]", ", ".join(enabled))
        for stage in tqdm(
            [name for name in ("tcr", "abag", "multimer", "dimer") if name in enabled],
            desc="Running higher-order stages",
            unit="stage",
        ):
            _run_high_order_stage(args, stage, sequence_dataset)
        LOGGER.info("Higher-order clustering complete (%.1fs)", time.monotonic() - t3)
    LOGGER.info("Clustering pipeline finished (%.1fs total)", time.monotonic() - t0)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = _normalize_legacy_positional_inputs(list(sys.argv[1:] if argv is None else argv))
    stage_subcommand, parse_argv = _split_stage_subcommand(raw_argv)
    bootstrap_parser = argparse.ArgumentParser(prog="cif-parse-cluster", add_help=False)
    bootstrap_parser.add_argument("--config", type=Path, default=None)
    bootstrap_args, _ = bootstrap_parser.parse_known_args(parse_argv)
    try:
        config_path, config_defaults = load_clustering_cli_config(bootstrap_args.config)
    except (FileNotFoundError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        bootstrap_parser.error(str(exc))

    parser = build_parser(config_defaults=config_defaults, config_path=config_path)
    if stage_subcommand is not None and any(token in {"-h", "--help"} for token in parse_argv):
        print(STAGE_HELP_TEXT[stage_subcommand])
        print()
        parser.print_help()
        return 0
    args = parser.parse_args(parse_argv)
    configure_logging(getattr(args, "log_level", "INFO"))
    if hasattr(args, "cutoff_date"):
        try:
            args.cutoff_date = normalize_cutoff_date(args.cutoff_date)
        except ValueError as exc:
            parser.error(str(exc))

    if getattr(args, "subcommand", None) == "prep":
        return _run_prep(args)

    prep_dir = _prep_dir(args)
    if args.inputs is None and prep_dir is None:
        parser.error("--inputs is required unless --prep-dir is provided")
    if stage_subcommand not in (None, "seq") and prep_dir is None:
        parser.error(
            f"`{stage_subcommand}` subcommand requires --prep-dir built by `cif-parse-cluster prep`; "
            "clustering stages do not read original mmCIF files"
        )
    _resolve_worker_counts(args, config_defaults)

    for field_name in (
        "jobs",
        "mmseqs_threads",
        "sequence_cluster_jobs",
        "dimer_extraction_jobs",
        "usalign_jobs",
    ):
        if getattr(args, field_name) < 1:
            parser.error(f"--{field_name.replace('_', '-')} must be >= 1")
    if args.dimer_interface_residue_cutoff <= 0:
        parser.error("--dimer-interface-residue-cutoff must be > 0")
    if args.antibody_complex_interface_residue_cutoff <= 0:
        parser.error("--antibody-complex-interface-residue-cutoff must be > 0")

    from cif_parse.clustering.parallel import set_global_usalign_limit

    set_global_usalign_limit(args.usalign_jobs)
    _warn_cif_override(args)
    if stage_subcommand is None:
        prep_dir = _ensure_prep_for_full_command(args)
    _set_shared_prep_index(prep_dir)
    try:
        try:
            if stage_subcommand == "seq":
                _run_seq(args)
                return 0
            if stage_subcommand == "structure":
                _run_structure(args)
                return 0
            if stage_subcommand in {"dimer", "multimer", "tcr", "abag"}:
                _run_high_order_stage(args, stage_subcommand)
                return 0
            return _run_full(args)
        except SequenceCutoffMismatchError as exc:
            parser.error(str(exc))
    finally:
        _close_prep_handles(prep_dir)


if __name__ == "__main__":
    raise SystemExit(main())
