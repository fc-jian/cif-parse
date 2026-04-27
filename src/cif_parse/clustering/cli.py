from __future__ import annotations

import argparse
import logging
import time
import tomllib
from pathlib import Path
from typing import Any

from cif_parse.clustering.antibody_complexes import build_antibody_complex_signature_clusters
from cif_parse.clustering.dimers import build_dimer_signature_clusters
from cif_parse.clustering.monomers import build_monomer_sequence_dataset
from cif_parse.clustering.multimers import build_multimer_signature_clusters
from cif_parse.clustering.protein_structures import (
    extract_protein_monomer_structures,
    greedy_cluster_protein_structures,
)
from cif_parse.clustering.tcr_complexes import build_tcr_complex_signature_clusters
from cif_parse.settings import (
    DEFAULT_CLUSTERING_OUTDIR,
    SUPPORTED_CLUSTERING_OBJECT_MODES,
    SUPPORTED_CLUSTERING_SEQUENCE_MODES,
    SUPPORTED_CLUSTERING_STRUCTURE_MODES,
    SUPPORTED_LOG_LEVELS,
    load_clustering_cli_config,
    resolve_source_path,
)
from cif_parse.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser(
    config_defaults: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> argparse.ArgumentParser:
    config_defaults = config_defaults or {}
    clustering_defaults = config_defaults.get("clustering", {})
    parser = argparse.ArgumentParser(
        description="Build monomer and higher-order clustering artifacts from cif-parse case outputs"
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
        "--db-path",
        type=Path,
        default=Path("clustering_prep.db"),
        help="Path to the prep SQLite database",
    )
    prep_parser.add_argument(
        "--cif-files-directory",
        type=Path,
        default=None,
        help="Optional override directory for original mmCIF files",
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
        help="Skip pre-loading mmCIF atom arrays into the cif_cache table",
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
        help="One or more case-output directories or parents containing case-output directories",
    )
    parser.add_argument(
        "--prep-db",
        type=Path,
        default=None,
        help="Optional path to a prep database (built with `cif-parse-cluster prep`); "
        "when provided, case bundles are read from the database instead of individual files",
    )
    parser.add_argument(
        "--cif-files-directory",
        type=Path,
        default=None,
        help="Optional override directory for original mmCIF files; when set, the basename "
        "of each source_path recorded in case bundles is resolved inside this directory",
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
        "--dimer-mode",
        choices=sorted(SUPPORTED_CLUSTERING_OBJECT_MODES),
        default=str(clustering_defaults.get("dimer_mode", "signature")),
        help="How to cluster dimer/interface observations after monomer clustering",
    )
    parser.add_argument(
        "--dimer-structure-mode",
        choices=sorted(SUPPORTED_CLUSTERING_STRUCTURE_MODES),
        default=str(clustering_defaults.get("dimer_structure_mode", "greedy")),
        help="How to refine signature-matched dimer/interface observations by overall dimer TM-score",
    )
    parser.add_argument(
        "--dimer-tm-score-threshold",
        type=float,
        default=float(clustering_defaults.get("dimer_tm_score_threshold", 0.50)),
        help="Minimum overall dimer max(TM(query,target), TM(target,query)) to keep two dimers together",
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
        "--antibody-complex-mode",
        choices=sorted(SUPPORTED_CLUSTERING_OBJECT_MODES),
        default=str(clustering_defaults.get("antibody_complex_mode", "signature")),
        help="How to cluster antibody-antigen complex observations after monomer clustering",
    )
    parser.add_argument(
        "--antibody-complex-structure-mode",
        choices=sorted(SUPPORTED_CLUSTERING_STRUCTURE_MODES),
        default=str(clustering_defaults.get("antibody_complex_structure_mode", "greedy")),
        help="How to refine signature-matched antibody-antigen complexes by overall complex TM-score",
    )
    parser.add_argument(
        "--antibody-complex-tm-score-threshold",
        type=float,
        default=float(clustering_defaults.get("antibody_complex_tm_score_threshold", 0.50)),
        help="Minimum overall antibody-complex max(TM(query,target), TM(target,query)) to keep two complexes together",
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
        help="Model index used when extracting canonical monomer coordinates from source mmCIF",
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
        default=float(clustering_defaults.get("min_alignment_coverage_ratio", 0.80)),
        help="Minimum aligned-length / shorter-sequence-length ratio for structural clustering",
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
        default=int(clustering_defaults.get("mmseqs_threads", clustering_defaults.get("jobs", 1))),
        help="Thread count passed to mmseqs easy-cluster",
    )
    parser.add_argument(
        "--sequence-cluster-jobs",
        type=int,
        default=int(clustering_defaults.get("sequence_cluster_jobs", clustering_defaults.get("jobs", 1))),
        help="Number of protein sequence clusters processed concurrently during monomer structure clustering",
    )
    parser.add_argument(
        "--usalign-jobs",
        type=int,
        default=int(clustering_defaults.get("usalign_jobs", clustering_defaults.get("jobs", 1))),
        help="Maximum concurrent USalign subprocesses per clustering refinement stage",
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(SUPPORTED_LOG_LEVELS),
        default=str(clustering_defaults.get("log_level", "INFO")),
        help="Root log level",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    bootstrap_parser = argparse.ArgumentParser(prog="cif-parse-cluster", add_help=False)
    bootstrap_parser.add_argument("--config", type=Path, default=None)
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)
    try:
        config_path, config_defaults = load_clustering_cli_config(bootstrap_args.config)
    except (FileNotFoundError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        bootstrap_parser.error(str(exc))

    args = build_parser(config_defaults=config_defaults, config_path=config_path).parse_args(argv)
    configure_logging(getattr(args, "log_level", "INFO"))

    # --- prep subcommand ---
    if getattr(args, "subcommand", None) == "prep":
        from cif_parse.clustering.prep import build_prep_database
        configure_logging("INFO")
        result = build_prep_database(
            inputs=args.inputs,
            db_path=args.db_path,
            cif_files_directory=str(args.cif_files_directory) if args.cif_files_directory else None,
            prep_jobs=args.prep_jobs,
            load_cif_cache=not args.no_cif_cache,
        )
        LOGGER.info("Prep complete: %s", result)
        return 0

    if args.inputs is None:
        parser = build_parser(config_defaults=config_defaults, config_path=config_path)
        parser.error("--inputs is required for clustering mode (or use 'prep' subcommand)")

    for field_name in ("jobs", "mmseqs_threads", "sequence_cluster_jobs", "usalign_jobs"):
        if getattr(args, field_name) < 1:
            build_parser(config_defaults=config_defaults, config_path=config_path).error(
                f"--{field_name.replace('_', '-')} must be >= 1"
            )

    if args.cif_files_directory is not None:
        LOGGER.warning(
            "Using --cif-files-directory=%s to override source mmCIF paths. "
            "Mismatched CIF files between clustering and the original cif-parse "
            "pipeline may produce incorrect results. If in doubt, re-run cif-parse "
            "with the same CIF file set.",
            args.cif_files_directory,
        )

    cif_files_directory: str | None = str(args.cif_files_directory) if args.cif_files_directory is not None else None
    prep_db_path: str | None = str(args.prep_db) if getattr(args, "prep_db", None) else None

    # --- Step 1: monomer sequence dataset ---
    t0 = time.monotonic()
    LOGGER.info("Step 1/4: Building monomer sequence dataset from %d input(s)", len(args.inputs))
    if prep_db_path:
        LOGGER.info("Using prep database: %s", prep_db_path)
    sequence_dataset = build_monomer_sequence_dataset(
        inputs=args.inputs,
        outdir=args.outdir,
        protein_sequence_mode=args.protein_sequence_mode,
        protein_min_seq_id=args.protein_min_seq_id,
        protein_coverage=args.protein_coverage,
        protein_cov_mode=args.protein_cov_mode,
        mmseqs_threads=args.mmseqs_threads,
        cif_files_directory=cif_files_directory,
        prep_db_path=prep_db_path,
    )
    manifest = sequence_dataset.get("manifest", {})
    LOGGER.info(
        "Step 1 complete (%.1fs): %d case dirs, %d canonical monomers, %d sequence membership rows",
        time.monotonic() - t0,
        manifest.get("num_input_case_dirs", 0) if isinstance(manifest, dict) else 0,
        manifest.get("num_canonical_monomers", 0) if isinstance(manifest, dict) else 0,
        manifest.get("num_sequence_membership_rows", 0) if isinstance(manifest, dict) else 0,
    )

    # --- Step 2-3: protein monomer structure extraction + clustering ---
    if args.protein_structure_mode == "greedy":
        t1 = time.monotonic()
        protein_monomer_count = sum(
            1 for m in sequence_dataset["monomers"] if m.polymer_class == "protein"
        )
        LOGGER.info(
            "Step 2/4: Extracting protein monomer structures (%d monomers, %d workers)",
            protein_monomer_count,
            args.usalign_jobs,
        )
        structure_outdir = args.outdir / "protein_structures"
        extracted_structures, extraction_manifest = extract_protein_monomer_structures(
            sequence_dataset["monomers"],
            outdir=structure_outdir,
            model=args.model,
            drop_hydrogens=not args.keep_hydrogens,
            extraction_jobs=args.usalign_jobs,
        )
        LOGGER.info(
            "Step 2 complete (%.1fs): %d structures extracted, %d failures",
            time.monotonic() - t1,
            extraction_manifest.get("num_extracted_protein_structures", 0),
            extraction_manifest.get("num_failed_protein_structure_extractions", 0),
        )

        t2 = time.monotonic()
        seq_cluster_count = len(set(row["sequence_cluster_id"] for row in sequence_dataset["membership_rows"] if row["polymer_class"] == "protein"))
        LOGGER.info(
            "Step 3/4: Clustering protein monomer structures (%d sequence clusters, %d workers)",
            seq_cluster_count,
            args.sequence_cluster_jobs,
        )
        greedy_cluster_protein_structures(
            sequence_dataset["monomers"],
            sequence_dataset["membership_rows"],
            extracted_structures,
            outdir=args.outdir / "structure_clusters",
            tm_score_threshold=args.tm_score_threshold,
            min_alignment_coverage_ratio=args.min_alignment_coverage_ratio,
            usalign_executable=args.usalign_executable,
            sequence_cluster_jobs=args.sequence_cluster_jobs,
            pairwise_alignment_jobs=args.usalign_jobs,
        )
        LOGGER.info("Step 3 complete (%.1fs)", time.monotonic() - t2)

    # --- Steps 4: higher-order clustering (run concurrently) ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    build_specs: list[tuple[str, str, str, str, str, dict[str, Any]]] = []
    if args.dimer_mode == "signature":
        build_specs.append(("dimer", "signature", args.dimer_structure_mode, "dimer_tm_score_threshold", "dimer_clusters", {
            "case_dirs": sequence_dataset["case_dirs"],
            "clustering_outdir": args.outdir,
            "outdir": args.outdir / "dimer_clusters",
            "structure_refinement_mode": args.dimer_structure_mode,
            "dimer_tm_score_threshold": args.dimer_tm_score_threshold,
            "model": args.model,
            "drop_hydrogens": not args.keep_hydrogens,
            "usalign_executable": args.usalign_executable,
            "alignment_jobs": args.usalign_jobs,
            "cif_files_directory": cif_files_directory,
            "prep_db_path": prep_db_path,
        }))
    if args.multimer_mode == "signature":
        build_specs.append(("multimer", "signature", args.multimer_structure_mode, "multimer_tm_score_threshold", "multimer_clusters", {
            "case_dirs": sequence_dataset["case_dirs"],
            "clustering_outdir": args.outdir,
            "outdir": args.outdir / "multimer_clusters",
            "structure_refinement_mode": args.multimer_structure_mode,
            "multimer_tm_score_threshold": args.multimer_tm_score_threshold,
            "model": args.model,
            "drop_hydrogens": not args.keep_hydrogens,
            "usalign_executable": args.usalign_executable,
            "alignment_jobs": args.usalign_jobs,
            "cif_files_directory": cif_files_directory,
            "prep_db_path": prep_db_path,
        }))
    if args.antibody_complex_mode == "signature":
        build_specs.append(("antibody_complex", "signature", args.antibody_complex_structure_mode, "antibody_complex_tm_score_threshold", "antibody_complex_clusters", {
            "case_dirs": sequence_dataset["case_dirs"],
            "clustering_outdir": args.outdir,
            "outdir": args.outdir / "antibody_complex_clusters",
            "structure_refinement_mode": args.antibody_complex_structure_mode,
            "antibody_complex_tm_score_threshold": args.antibody_complex_tm_score_threshold,
            "model": args.model,
            "drop_hydrogens": not args.keep_hydrogens,
            "usalign_executable": args.usalign_executable,
            "alignment_jobs": args.usalign_jobs,
            "cif_files_directory": cif_files_directory,
            "prep_db_path": prep_db_path,
        }))
    if args.tcr_complex_mode == "signature":
        build_specs.append(("tcr_complex", "signature", args.tcr_complex_structure_mode, "tcr_complex_tm_score_threshold", "tcr_complex_clusters", {
            "case_dirs": sequence_dataset["case_dirs"],
            "clustering_outdir": args.outdir,
            "outdir": args.outdir / "tcr_complex_clusters",
            "structure_refinement_mode": args.tcr_complex_structure_mode,
            "tcr_complex_tm_score_threshold": args.tcr_complex_tm_score_threshold,
            "model": args.model,
            "drop_hydrogens": not args.keep_hydrogens,
            "usalign_executable": args.usalign_executable,
            "alignment_jobs": args.usalign_jobs,
            "cif_files_directory": cif_files_directory,
            "prep_db_path": prep_db_path,
        }))

    build_funcs = {
        "dimer": build_dimer_signature_clusters,
        "multimer": build_multimer_signature_clusters,
        "antibody_complex": build_antibody_complex_signature_clusters,
        "tcr_complex": build_tcr_complex_signature_clusters,
    }

    if build_specs:
        step_names = ", ".join(kind for kind, _, _, _, _, _ in build_specs)
        t3 = time.monotonic()
        LOGGER.info(
            "Step 4/4: Building higher-order clusters [%s] (%d steps, %d parallel)",
            step_names,
            len(build_specs),
            min(len(build_specs), 4),
        )
    if len(build_specs) <= 1:
        for kind, _, _, _, _, kwargs in build_specs:
            build_funcs[kind](**kwargs)
            LOGGER.info("Higher-order step %s completed", kind)
    elif build_specs:
        with ThreadPoolExecutor(max_workers=min(len(build_specs), 4)) as executor:
            futures = {
                executor.submit(build_funcs[kind], **kwargs): kind
                for kind, _, _, _, _, kwargs in build_specs
            }
            for future in as_completed(futures):
                kind = futures[future]
                try:
                    future.result()
                    LOGGER.info("Higher-order step %s completed", kind)
                except Exception:
                    LOGGER.exception("Higher-order clustering step %s failed", kind)
                    raise
    if build_specs:
        LOGGER.info("Step 4 complete (%.1fs)", time.monotonic() - t3)
    LOGGER.info("Clustering pipeline finished (%.1fs total)", time.monotonic() - t0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
