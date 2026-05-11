"""CLI entry point for ``cif-parse-refine`` — antibody-antigen complex refinement."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from cif_parse.refine.abag_refine import refine_antibody_complexes

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cif-parse-refine", description="Refine antibody-antigen complexes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    abag_parser = subparsers.add_parser("abag", help="Refine antibody-antigen complexes")
    abag_parser.add_argument(
        "--case-dirs", type=Path, nargs="+", required=True,
        help="One or more case output directories from cif-parse",
    )
    abag_parser.add_argument(
        "--prep-dir", type=Path, required=True,
        help="Prep directory with per-chain atom arrays",
    )
    abag_parser.add_argument(
        "--outdir", type=Path, default=Path("refined_outputs"),
        help="Output directory for refined structures",
    )
    abag_parser.add_argument(
        "--contact-distance", type=float, default=8.0,
        help="Residue contact distance threshold in Angstroms",
    )
    abag_parser.add_argument(
        "--louvain-resolution", type=float, default=1.0,
        help="Louvain community detection resolution",
    )
    abag_parser.add_argument(
        "--min-domain-size", type=int, default=10,
        help="Minimum number of residues for an antigen domain",
    )
    abag_parser.add_argument(
        "--min-contact-residues", type=int, default=3,
        help="Minimum antibody-contact residues required to keep an antigen domain",
    )
    abag_parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root logging level",
    )
    abag_parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "abag":
        results = refine_antibody_complexes(
            case_dirs=[str(p) for p in args.case_dirs],
            prep_dir=str(args.prep_dir),
            outdir=str(args.outdir),
            contact_distance=args.contact_distance,
            louvain_resolution=args.louvain_resolution,
            min_domain_size=args.min_domain_size,
            min_contact_residues=args.min_contact_residues,
            show_progress=not args.no_progress,
        )
        LOGGER.info("Refined %d complexes → %s", len(results), args.outdir)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
