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
    get_fast_temp_dir,
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
        description=(
            "Build monomer and higher-order clustering artifacts from cif-parse "
            "case outputs or a prebuilt clustering prep directory"
        )
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
        default=clustering_defaults.get("mmseqs_threads") or clustering_defaults.get("jobs", 1),
        help="Thread count passed to mmseqs easy-cluster (default: same as --jobs)",
    )
    parser.add_argument(
        "--sequence-cluster-jobs",
        type=int,
        default=clustering_defaults.get("sequence_cluster_jobs") or clustering_defaults.get("jobs", 1),
        help="Number of protein sequence clusters processed concurrently (default: same as --jobs)",
    )
    parser.add_argument(
        "--usalign-jobs",
        type=int,
        default=clustering_defaults.get("usalign_jobs") or clustering_defaults.get("jobs", 1),
        help="Maximum concurrent USalign subprocesses per refinement stage (default: same as --jobs)",
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
            prep_dir=args.prep_dir,
            cif_files_directory=str(args.cif_files_directory) if args.cif_files_directory else None,
            prep_jobs=args.prep_jobs,
            load_cif_cache=not args.no_cif_cache,
        )
        LOGGER.info("Prep complete: %s", result)
        return 0

    prep_dir: str | None = str(args.prep_dir) if getattr(args, "prep_dir", None) else None
    if args.inputs is None and prep_dir is None:
        parser = build_parser(config_defaults=config_defaults, config_path=config_path)
        parser.error("--inputs is required unless --prep-dir is provided")

    for field_name in ("jobs", "mmseqs_threads", "sequence_cluster_jobs", "usalign_jobs"):
        if getattr(args, field_name) < 1:
            build_parser(config_defaults=config_defaults, config_path=config_path).error(
                f"--{field_name.replace('_', '-')} must be >= 1"
            )

    # Global USalign concurrency cap prevents oversubscription when multiple
    # clustering stages submit USalign tasks concurrently.
    from cif_parse.clustering.parallel import set_global_usalign_limit
    set_global_usalign_limit(args.usalign_jobs)

    if args.cif_files_directory is not None:
        LOGGER.warning(
            "Using --cif-files-directory=%s to override source mmCIF paths. "
            "Mismatched CIF files between clustering and the original cif-parse "
            "pipeline may produce incorrect results. If in doubt, re-run cif-parse "
            "with the same CIF file set.",
            args.cif_files_directory,
        )

    cif_files_directory: str | None = str(args.cif_files_directory) if args.cif_files_directory is not None else None
    cluster_inputs = args.inputs or []

    # Load prep coord index once; all extraction stages share it.
    if prep_dir:
        from cif_parse.clustering.prep import load_cif_coords_index, set_shared_coord_index
        _shared = load_cif_coords_index(prep_dir)
        set_shared_coord_index(_shared, prep_dir)

    # --- Step 1: monomer sequence dataset ---
    t0 = time.monotonic()
    if prep_dir:
        LOGGER.info("Step 1/4: Building monomer sequence dataset from prep directory: %s", prep_dir)
    else:
        LOGGER.info("Step 1/4: Building monomer sequence dataset from %d input(s)", len(cluster_inputs))
    sequence_dataset = build_monomer_sequence_dataset(
        inputs=cluster_inputs,
        outdir=args.outdir,
        protein_sequence_mode=args.protein_sequence_mode,
        protein_min_seq_id=args.protein_min_seq_id,
        protein_coverage=args.protein_coverage,
        protein_cov_mode=args.protein_cov_mode,
        mmseqs_threads=args.mmseqs_threads,
        cif_files_directory=cif_files_directory,
        prep_dir=prep_dir,
    )
    manifest = sequence_dataset.get("manifest", {})
    LOGGER.info(
        "Step 1 complete (%.1fs): %d case dirs, %d canonical monomers, %d sequence membership rows",
        time.monotonic() - t0,
        manifest.get("num_input_case_dirs", 0) if isinstance(manifest, dict) else 0,
        manifest.get("num_canonical_monomers", 0) if isinstance(manifest, dict) else 0,
        manifest.get("num_sequence_membership_rows", 0) if isinstance(manifest, dict) else 0,
    )

    # --- Step 2-3: protein monomer structure extraction + clustering (pipelined) ---
    if args.protein_structure_mode == "greedy":
        t1 = time.monotonic()
        protein_monomer_count = sum(
            1 for m in sequence_dataset["monomers"] if m.polymer_class == "protein"
        )
        seq_cluster_count = len({
            row.get("cluster_id", row.get("sequence_cluster_id", ""))
            for row in sequence_dataset["membership_rows"]
            if row.get("polymer_class") == "protein"
        })
        structure_outdir = get_fast_temp_dir("protein_structures")
        LOGGER.info(
            "Step 2+3/4: Extracting + clustering protein monomer structures "
            "(%d monomers, %d seq clusters, pipelined with %d seq-cluster workers)",
            protein_monomer_count,
            seq_cluster_count,
            args.sequence_cluster_jobs,
        )

        # Build on-the-fly extractor that shares the prep cif_cache.
        cif_idx_for_pipeline: dict | None = None
        if prep_dir:
            from cif_parse.clustering.prep import load_cif_coords_index, load_cif_from_prep
            cif_idx_for_pipeline = load_cif_coords_index(prep_dir)

        from cif_parse.io import read_cif_file
        from biotite.structure.io.pdbx import get_structure
        quality_cache: dict[str, Any] = {}
        atom_array_cache: dict[str, Any] = {}
        import threading
        _extract_lock = threading.Lock()

        def _pipeline_extract(monomer) -> Any | None:
            nonlocal cif_idx_for_pipeline
            from cif_parse.clustering.protein_structures import (
                SKIP_QUALITY_METADATA,
                extract_protein_monomer_structure,
                read_entry_quality_metadata,
            )
            _atom_key = f"{monomer.source_path}__{monomer.label_asym_id}"
            _need_atoms = False
            _need_quality = False
            with _extract_lock:
                _need_atoms = _atom_key not in atom_array_cache
                _need_quality = monomer.source_path not in quality_cache

            if _need_atoms and cif_idx_for_pipeline is not None:
                from cif_parse.clustering.prep import load_chain_atoms, load_cif_from_prep as _lcfp
                _found = None
                for aid in [None] + (monomer.observed_assembly_ids or []):
                    _c = load_chain_atoms(
                        prep_dir,
                        monomer.source_path,
                        monomer.label_asym_id,
                        assembly_id=str(aid) if aid else None,
                        index=cif_idx_for_pipeline,
                    )
                    if _c is not None:
                        _found = _c
                        break
                if _found is None:
                    for aid in [None] + (monomer.observed_assembly_ids or []):
                        _c = _lcfp(
                            prep_dir,
                            monomer.source_path,
                            str(aid) if aid else None,
                            index=cif_idx_for_pipeline,
                        )
                        if _c is not None and _c.get("atom_array") is not None:
                            _found = _c["atom_array"]
                            break
                if _found is not None:
                    with _extract_lock:
                        atom_array_cache[_atom_key] = _found
            if _need_quality:
                _q = (
                    SKIP_QUALITY_METADATA
                    if prep_dir
                    else read_entry_quality_metadata(monomer.source_path, pdb_id=monomer.pdb_id)
                )
                with _extract_lock:
                    if monomer.source_path not in quality_cache:
                        quality_cache[monomer.source_path] = _q
            if _need_atoms:
                with _extract_lock:
                    _need_atoms = _atom_key not in atom_array_cache
                if _need_atoms:
                    if prep_dir:
                        raise ValueError(f"Prep coordinates missing for monomer {monomer.monomer_id}")
                    cf = read_cif_file(monomer.source_path)
                    _atoms = get_structure(cf, model=args.model, use_author_fields=False)
                    with _extract_lock:
                        atom_array_cache.setdefault(_atom_key, _atoms)

            with _extract_lock:
                _atoms = atom_array_cache.get(_atom_key)
                _q = quality_cache.get(monomer.source_path)
            try:
                return extract_protein_monomer_structure(
                    monomer,
                    outdir=structure_outdir,
                    model=args.model,
                    drop_hydrogens=not args.keep_hydrogens,
                    quality_metadata=_q,
                    atom_array=_atoms,
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
            extract_fn=_pipeline_extract,
        )
        manifest = result.get("manifest", {})
        LOGGER.info(
            "Step 2+3 complete (%.1fs): %d structures extracted (%d failures), "
            "%d sequence clusters -> %d structure clusters (%d alignments, %d failures)",
            time.monotonic() - t1,
            manifest.get("num_extracted", 0),
            manifest.get("num_extraction_failures", 0),
            manifest.get("num_sequence_clusters", 0),
            manifest.get("num_structure_clusters", 0),
            manifest.get("num_alignment_runs", 0),
            manifest.get("num_alignment_failures", 0),
        )

    # --- Step 4: higher-order clustering (serial by layer) ---
    build_specs: list[tuple[str, str, str, str, str, dict[str, Any]]] = []
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
            "prep_dir": prep_dir,
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
            "prep_dir": prep_dir,
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
            "prep_dir": prep_dir,
        }))
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
            "prep_dir": prep_dir,
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
            "Step 4/4: Building higher-order clusters serially [%s] (%d steps)",
            step_names,
            len(build_specs),
        )
    for kind, _, _, _, _, kwargs in build_specs:
        step_t0 = time.monotonic()
        build_funcs[kind](**kwargs)
        LOGGER.info("Higher-order step %s completed (%.1fs)", kind, time.monotonic() - step_t0)
    if build_specs:
        LOGGER.info("Step 4 complete (%.1fs)", time.monotonic() - t3)
    if prep_dir:
        from cif_parse.clustering.prep import close_blob_handles
        close_blob_handles()
    LOGGER.info("Clustering pipeline finished (%.1fs total)", time.monotonic() - t0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
