from __future__ import annotations

import argparse
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
    load_cli_config,
)
from cif_parse.utils.logging_utils import configure_logging


def build_parser(
    config_defaults: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> argparse.ArgumentParser:
    config_defaults = config_defaults or {}
    clustering_defaults = config_defaults.get("clustering", {})
    parser = argparse.ArgumentParser(
        description="Build monomer and higher-order clustering artifacts from cif-parse case outputs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Optional config.toml path; CLI arguments override [clustering] values",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more case-output directories or parents containing case-output directories",
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
        config_path, config_defaults = load_cli_config(bootstrap_args.config)
    except (FileNotFoundError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        bootstrap_parser.error(str(exc))

    args = build_parser(config_defaults=config_defaults, config_path=config_path).parse_args(argv)
    configure_logging(args.log_level)
    sequence_dataset = build_monomer_sequence_dataset(
        inputs=args.inputs,
        outdir=args.outdir,
        protein_sequence_mode=args.protein_sequence_mode,
        protein_min_seq_id=args.protein_min_seq_id,
        protein_coverage=args.protein_coverage,
        protein_cov_mode=args.protein_cov_mode,
    )
    if args.protein_structure_mode == "greedy":
        structure_outdir = args.outdir / "protein_structures"
        extracted_structures, _ = extract_protein_monomer_structures(
            sequence_dataset["monomers"],
            outdir=structure_outdir,
            model=args.model,
            drop_hydrogens=not args.keep_hydrogens,
        )
        greedy_cluster_protein_structures(
            sequence_dataset["monomers"],
            sequence_dataset["membership_rows"],
            extracted_structures,
            outdir=args.outdir / "structure_clusters",
            tm_score_threshold=args.tm_score_threshold,
            min_alignment_coverage_ratio=args.min_alignment_coverage_ratio,
            usalign_executable=args.usalign_executable,
        )
    if args.dimer_mode == "signature":
        build_dimer_signature_clusters(
            case_dirs=sequence_dataset["case_dirs"],
            clustering_outdir=args.outdir,
            outdir=args.outdir / "dimer_clusters",
            structure_refinement_mode=args.dimer_structure_mode,
            dimer_tm_score_threshold=args.dimer_tm_score_threshold,
            model=args.model,
            drop_hydrogens=not args.keep_hydrogens,
            usalign_executable=args.usalign_executable,
        )
    if args.multimer_mode == "signature":
        build_multimer_signature_clusters(
            case_dirs=sequence_dataset["case_dirs"],
            clustering_outdir=args.outdir,
            outdir=args.outdir / "multimer_clusters",
            structure_refinement_mode=args.multimer_structure_mode,
            multimer_tm_score_threshold=args.multimer_tm_score_threshold,
            model=args.model,
            drop_hydrogens=not args.keep_hydrogens,
            usalign_executable=args.usalign_executable,
        )
    if args.antibody_complex_mode == "signature":
        build_antibody_complex_signature_clusters(
            case_dirs=sequence_dataset["case_dirs"],
            clustering_outdir=args.outdir,
            outdir=args.outdir / "antibody_complex_clusters",
            structure_refinement_mode=args.antibody_complex_structure_mode,
            antibody_complex_tm_score_threshold=args.antibody_complex_tm_score_threshold,
            model=args.model,
            drop_hydrogens=not args.keep_hydrogens,
            usalign_executable=args.usalign_executable,
        )
    if args.tcr_complex_mode == "signature":
        build_tcr_complex_signature_clusters(
            case_dirs=sequence_dataset["case_dirs"],
            clustering_outdir=args.outdir,
            outdir=args.outdir / "tcr_complex_clusters",
            structure_refinement_mode=args.tcr_complex_structure_mode,
            tcr_complex_tm_score_threshold=args.tcr_complex_tm_score_threshold,
            model=args.model,
            drop_hydrogens=not args.keep_hydrogens,
            usalign_executable=args.usalign_executable,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
