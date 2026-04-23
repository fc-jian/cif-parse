"""Command-line entry points for single-file and batch mmCIF processing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cif_parse.export import dump_json
from cif_parse.io import read_structure_summary
from cif_parse.pipeline import StructureSkipWarning, infer_case_id, process_single_structure
from cif_parse.reporting import (
    build_batch_html_report,
    build_review_report,
    collect_case_review_metrics,
)
from cif_parse.settings import (
    AppSettings,
    DEFAULT_BATCH_OUTDIR,
    DEFAULT_SINGLE_OUTDIR,
    SUPPORTED_ASSEMBLY_MODES,
    SUPPORTED_COVERAGE_MODES,
    SUPPORTED_FORMATS,
    SUPPORTED_LOG_LEVELS,
    load_cli_config,
)
from cif_parse.utils.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)
SUPPORTED_INPUT_SUFFIXES = (".cif.gz", ".bcif.gz", ".cif", ".bcif")


def build_parser(
    config_defaults: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    config_defaults = config_defaults or {}
    settings_defaults = config_defaults.get("settings", {})
    single_defaults = config_defaults.get("single", {})
    batch_defaults = config_defaults.get("batch", {})

    parser = argparse.ArgumentParser(prog="cif-parse")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Optional config.toml path; CLI arguments override config values",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print a structure summary")
    inspect_parser.add_argument("input", type=Path, help="Path to a .cif/.cif.gz file")
    _add_read_options(inspect_parser, settings_defaults=settings_defaults)

    single_parser = subparsers.add_parser(
        "single",
        help="Read one mmCIF file and export single-chain artifacts",
    )
    single_parser.add_argument("input", type=Path, help="Path to a .cif/.cif.gz file")
    _add_read_options(single_parser, settings_defaults=settings_defaults)
    _add_runtime_args(
        single_parser,
        settings_defaults=settings_defaults,
        default_outdir=Path(single_defaults.get("outdir", DEFAULT_SINGLE_OUTDIR)),
    )

    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple mmCIF inputs with optional multiprocessing",
    )
    batch_parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input files or directories; directories are searched recursively",
    )
    batch_parser.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help="Optional text file with one input path per line",
    )
    batch_parser.add_argument(
        "--jobs",
        type=int,
        default=int(batch_defaults.get("jobs", max(1, os.cpu_count() or 1))),
        help="Number of worker processes to use",
    )
    batch_parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=bool(batch_defaults.get("fail_fast", False)),
        help="Stop immediately if any case fails",
    )
    _add_read_options(batch_parser, settings_defaults=settings_defaults)
    _add_runtime_args(
        batch_parser,
        settings_defaults=settings_defaults,
        default_outdir=Path(batch_defaults.get("outdir", DEFAULT_BATCH_OUTDIR)),
    )

    return parser


def _add_read_options(parser: argparse.ArgumentParser, *, settings_defaults: dict[str, Any]) -> None:
    """Add shared structure-reading options to a parser."""

    parser.add_argument(
        "--model",
        type=int,
        default=int(settings_defaults.get("model", 1)),
        help="1-based model index",
    )
    parser.add_argument(
        "--author-fields",
        action=argparse.BooleanOptionalAction,
        default=bool(settings_defaults.get("use_author_fields", False)),
        help="Use auth_asym_id/auth_seq_id fields when reading structure arrays",
    )
    parser.add_argument(
        "--drop-hydrogens-for-analysis",
        action=argparse.BooleanOptionalAction,
        default=bool(settings_defaults.get("drop_hydrogens_for_analysis", True)),
        help="Use heavy-atom-only arrays for downstream coverage/interface analysis",
    )


def _add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    settings_defaults: dict[str, Any],
    default_outdir: Path,
) -> None:
    """Add shared runtime/export arguments to a parser."""

    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_outdir,
        help="Output directory for exported artifacts",
    )
    parser.add_argument(
        "--format",
        choices=sorted(SUPPORTED_FORMATS),
        default=str(settings_defaults.get("output_format", "json")),
        help="Main output format",
    )
    parser.add_argument(
        "--assembly-mode",
        choices=sorted(SUPPORTED_ASSEMBLY_MODES),
        default=str(settings_defaults.get("assembly_mode", "asymmetric_unit")),
        help="Assembly mode recorded in settings metadata",
    )
    parser.add_argument(
        "--coverage-mode",
        choices=sorted(SUPPORTED_COVERAGE_MODES),
        default=str(settings_defaults.get("coverage_mode", "nearest")),
        help="Coverage mode recorded in settings metadata",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=bool(settings_defaults.get("debug", False)),
        help="For JSON output, keep split per-artifact JSON files instead of one compact result.json.gz bundle",
    )
    parser.add_argument(
        "--log-level",
        choices=sorted(SUPPORTED_LOG_LEVELS),
        default=str(settings_defaults.get("log_level", "INFO")),
        help="Root log level for CLI and worker processes",
    )
    parser.add_argument(
        "--max-polymer-chains",
        type=int,
        default=int(settings_defaults.get("max_polymer_chains", 100)),
        help="Skip structures with more than this many polymer chains",
    )
    parser.add_argument(
        "--min-polymer-chain-length",
        type=int,
        default=int(settings_defaults.get("min_polymer_chain_length", 20)),
        help="Skip structures unless at least one polymer chain has length > this threshold",
    )
    parser.add_argument(
        "--tight-multimer-min-buried-area",
        type=float,
        default=float(settings_defaults.get("tight_multimer_min_buried_area", 500.0)),
        help="Minimum buried area required to keep a dimer edge in the tight-multimer graph",
    )
    parser.add_argument(
        "--tight-multimer-louvain-resolution",
        "--tight-multimer-leiden-resolution",
        dest="tight_multimer_louvain_resolution",
        type=float,
        default=float(settings_defaults.get("tight_multimer_louvain_resolution", 1.0)),
        help="Resolution parameter for tight-multimer Louvain community detection",
    )
    parser.add_argument(
        "--tight-multimer-min-member-instances",
        type=int,
        default=int(settings_defaults.get("tight_multimer_min_member_instances", 2)),
        help="Minimum Louvain community size in instance nodes for exporting a tight multimer",
    )
    parser.add_argument(
        "--tight-multimer-large-component-warning-size",
        type=int,
        default=int(settings_defaults.get("tight_multimer_large_component_warning_size", 8)),
        help="Add a large-component warning when a tight multimer reaches this many instance nodes",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=bool(settings_defaults.get("verbose", False)),
        help="Print verbose JSON summaries instead of just the output directory",
    )


def _settings_from_args(args: argparse.Namespace) -> AppSettings:
    """Convert parsed CLI arguments into validated application settings."""

    return AppSettings(
        output_format=args.format,
        assembly_mode=args.assembly_mode,
        coverage_mode=args.coverage_mode,
        debug=args.debug,
        log_level=args.log_level,
        verbose=args.verbose,
        model=args.model,
        use_author_fields=args.author_fields,
        drop_hydrogens_for_analysis=args.drop_hydrogens_for_analysis,
        max_polymer_chains=args.max_polymer_chains,
        min_polymer_chain_length=args.min_polymer_chain_length,
        tight_multimer_min_buried_area=args.tight_multimer_min_buried_area,
        tight_multimer_louvain_resolution=args.tight_multimer_louvain_resolution,
        tight_multimer_min_member_instances=args.tight_multimer_min_member_instances,
        tight_multimer_large_component_warning_size=args.tight_multimer_large_component_warning_size,
    )


def _load_input_list(path: Path) -> list[Path]:
    """Load additional inputs from a newline-delimited text file."""

    inputs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        inputs.append(Path(line))
    return inputs


def _expand_input_path(path: Path) -> list[Path]:
    """Expand one file-or-directory argument into concrete mmCIF file paths."""

    if path.is_dir():
        files: list[Path] = []
        for suffix in SUPPORTED_INPUT_SUFFIXES:
            files.extend(sorted(path.rglob(f"*{suffix}")))
        return files
    return [path]


def _resolve_batch_inputs(inputs: list[Path], input_list: Path | None) -> list[Path]:
    """Resolve batch CLI inputs from explicit paths and optional list files."""

    candidates = list(inputs)
    if input_list is not None:
        candidates.extend(_load_input_list(input_list))
    if not candidates:
        raise ValueError("batch requires at least one input path or --input-list")

    resolved_inputs: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        for expanded in _expand_input_path(candidate):
            suffix_match = any(str(expanded).lower().endswith(suffix) for suffix in SUPPORTED_INPUT_SUFFIXES)
            if not suffix_match:
                LOGGER.debug("Skipping non-mmCIF input %s", expanded)
                continue
            key = str(expanded.resolve()) if expanded.exists() else str(expanded)
            if key in seen:
                continue
            seen.add(key)
            resolved_inputs.append(expanded)
    if not resolved_inputs:
        raise ValueError("no valid mmCIF inputs found for batch processing")
    return resolved_inputs


def _build_case_specs(input_paths: list[Path], outdir: Path) -> list[dict[str, Any]]:
    """Assign unique case ids and output directories for batch processing."""

    total_counts: dict[str, int] = {}
    for input_path in input_paths:
        base_case_id = infer_case_id(input_path)
        total_counts[base_case_id] = total_counts.get(base_case_id, 0) + 1

    seen_counts: dict[str, int] = {}
    case_specs: list[dict[str, Any]] = []
    for input_path in input_paths:
        base_case_id = infer_case_id(input_path)
        seen_counts[base_case_id] = seen_counts.get(base_case_id, 0) + 1
        case_id = base_case_id
        if total_counts[base_case_id] > 1:
            case_id = f"{base_case_id}_{seen_counts[base_case_id]:02d}"
        case_specs.append(
            {
                "case_id": case_id,
                "input_path": str(input_path),
                "output_dir": str(outdir / "cases" / case_id),
            }
        )
    return case_specs


def _summarize_batch_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact summary file for batch execution."""

    successes = [result for result in results if result["status"] == "ok"]
    skipped = [result for result in results if result["status"] == "skipped"]
    failures = [result for result in results if result["status"] == "error"]
    total_success = len(successes)
    total_cases = len(results)
    total_chains = sum(int(result.get("num_chains", 0)) for result in successes)
    total_dimers = sum(int(result.get("num_dimers", 0)) for result in successes)
    total_multimers = sum(int(result.get("num_multimers", 0)) for result in successes)
    total_antibody_complexes = sum(
        int(result.get("num_antibody_antigen_complexes", 0)) for result in successes
    )
    total_tcr_complexes = sum(int(result.get("num_tcr_pmhc_complexes", 0)) for result in successes)
    return {
        "total_cases": total_cases,
        "success_count": total_success,
        "skipped_count": len(skipped),
        "failure_count": len(failures),
        "total_chains": total_chains,
        "total_dimers": total_dimers,
        "total_multimers": total_multimers,
        "total_antibody_antigen_complexes": total_antibody_complexes,
        "total_tcr_pmhc_complexes": total_tcr_complexes,
        "average_chains_per_successful_case": round(total_chains / total_success, 2) if total_success else 0.0,
        "average_dimers_per_successful_case": round(total_dimers / total_success, 2) if total_success else 0.0,
        "average_multimers_per_successful_case": round(total_multimers / total_success, 2) if total_success else 0.0,
    }


def _process_batch_case(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point for one batch case."""

    settings = AppSettings(**task["settings"])
    configure_logging(settings.log_level)
    try:
        result = process_single_structure(task["input_path"], task["output_dir"], settings)
        result["case_id"] = task["case_id"]
        result["status"] = "ok"
        return result
    except StructureSkipWarning as exc:
        LOGGER.warning("%s", exc)
        return {
            "case_id": task["case_id"],
            "input_path": task["input_path"],
            "output_dir": task["output_dir"],
            "status": "skipped",
            "warning_code": exc.code,
            "warning": str(exc),
            "warning_details": exc.details,
        }
    except Exception as exc:  # pragma: no cover - exercised through CLI behavior
        LOGGER.exception("Failed to process %s", task["input_path"])
        return {
            "case_id": task["case_id"],
            "input_path": task["input_path"],
            "output_dir": task["output_dir"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _print_single_result(settings: AppSettings, outdir: Path, result: dict[str, Any]) -> None:
    """Print the single-run CLI response."""

    if settings.verbose or result.get("status") != "ok":
        print(
            json.dumps(
                {
                    "settings": {
                        "output_format": settings.output_format,
                        "assembly_mode": settings.assembly_mode,
                        "coverage_mode": settings.coverage_mode,
                        "debug": settings.debug,
                        "log_level": settings.log_level,
                        "max_polymer_chains": settings.max_polymer_chains,
                        "min_polymer_chain_length": settings.min_polymer_chain_length,
                        "model": settings.model,
                        "use_author_fields": settings.use_author_fields,
                        "drop_hydrogens_for_analysis": settings.drop_hydrogens_for_analysis,
                        "tight_multimer_min_buried_area": settings.tight_multimer_min_buried_area,
                        "tight_multimer_louvain_resolution": settings.tight_multimer_louvain_resolution,
                        "tight_multimer_min_member_instances": settings.tight_multimer_min_member_instances,
                        "tight_multimer_large_component_warning_size": settings.tight_multimer_large_component_warning_size,
                    },
                    **result,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(outdir)


def _print_batch_result(
    settings: AppSettings,
    outdir: Path,
    manifest_path: Path,
    summary_path: Path,
    review_path: Path,
    html_report_path: Path,
    summary: dict[str, Any],
) -> None:
    """Print the batch-run CLI response."""

    if settings.verbose:
        print(
            json.dumps(
                {
                    "settings": {
                        "output_format": settings.output_format,
                        "assembly_mode": settings.assembly_mode,
                        "coverage_mode": settings.coverage_mode,
                        "debug": settings.debug,
                        "log_level": settings.log_level,
                        "max_polymer_chains": settings.max_polymer_chains,
                        "min_polymer_chain_length": settings.min_polymer_chain_length,
                        "model": settings.model,
                        "use_author_fields": settings.use_author_fields,
                        "drop_hydrogens_for_analysis": settings.drop_hydrogens_for_analysis,
                        "tight_multimer_min_buried_area": settings.tight_multimer_min_buried_area,
                        "tight_multimer_louvain_resolution": settings.tight_multimer_louvain_resolution,
                        "tight_multimer_min_member_instances": settings.tight_multimer_min_member_instances,
                        "tight_multimer_large_component_warning_size": settings.tight_multimer_large_component_warning_size,
                    },
                    "manifest_path": str(manifest_path),
                    "summary_path": str(summary_path),
                    "review_path": str(review_path),
                    "html_report_path": str(html_report_path),
                    **summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(outdir)


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI."""

    bootstrap_parser = argparse.ArgumentParser(prog="cif-parse", add_help=False)
    bootstrap_parser.add_argument("--config", type=Path, default=None)
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)
    try:
        config_path, config_defaults = load_cli_config(bootstrap_args.config)
    except (FileNotFoundError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        bootstrap_parser.error(str(exc))

    parser = build_parser(config_defaults=config_defaults, config_path=config_path)
    args = parser.parse_args(argv)

    if args.command == "inspect":
        if args.input is None:
            parser.error("inspect requires an input file")
        summary = read_structure_summary(
            args.input,
            model=args.model,
            use_author_fields=args.author_fields,
            coverage_mode="nearest",
            drop_hydrogens_for_analysis=args.drop_hydrogens_for_analysis,
        )
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return 0

    settings = _settings_from_args(args)
    configure_logging(settings.log_level)
    if args.command == "batch" and args.jobs < 1:
        parser.error("--jobs must be >= 1")

    if args.command == "single":
        if args.input is None:
            parser.error("single requires an input file")
        try:
            result = process_single_structure(args.input, args.outdir, settings)
        except StructureSkipWarning as exc:
            LOGGER.warning("%s", exc)
            result = {
                "pdb_id": infer_case_id(args.input),
                "input_path": str(args.input),
                "output_dir": str(args.outdir),
                "status": "skipped",
                "warning_code": exc.code,
                "warning": str(exc),
                "warning_details": exc.details,
            }
        _print_single_result(settings, args.outdir, result)
        return 0

    if args.command != "batch":
        parser.error(f"Unsupported command: {args.command}")

    input_paths = _resolve_batch_inputs(args.inputs, args.input_list)
    case_specs = _build_case_specs(input_paths, args.outdir)
    LOGGER.info(
        "Starting batch processing for %d inputs with %d worker(s)",
        len(case_specs),
        args.jobs,
    )
    settings_payload = {
        "output_format": settings.output_format,
        "assembly_mode": settings.assembly_mode,
        "coverage_mode": settings.coverage_mode,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "verbose": settings.verbose,
        "model": settings.model,
        "use_author_fields": settings.use_author_fields,
        "drop_hydrogens_for_analysis": settings.drop_hydrogens_for_analysis,
        "max_polymer_chains": settings.max_polymer_chains,
        "min_polymer_chain_length": settings.min_polymer_chain_length,
        "tight_multimer_min_buried_area": settings.tight_multimer_min_buried_area,
        "tight_multimer_louvain_resolution": settings.tight_multimer_louvain_resolution,
        "tight_multimer_min_member_instances": settings.tight_multimer_min_member_instances,
        "tight_multimer_large_component_warning_size": settings.tight_multimer_large_component_warning_size,
    }
    tasks = [
        {
            **case_spec,
            "settings": settings_payload,
        }
        for case_spec in case_specs
    ]

    results: list[dict[str, Any]] = []
    if args.jobs == 1:
        for task in tasks:
            result = _process_batch_case(task)
            results.append(result)
            if args.fail_fast and result["status"] == "error":
                break
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_process_batch_case, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if args.fail_fast and result["status"] == "error":
                    LOGGER.error("Batch failed fast on %s", result["input_path"])
                    for pending in futures:
                        pending.cancel()
                    break

    results.sort(key=lambda item: item["case_id"])
    summary = _summarize_batch_results(results)
    review_results: list[dict[str, Any]] = []
    for result in results:
        review_result = dict(result)
        if result.get("status") == "ok":
            review_result["metrics"] = collect_case_review_metrics(result["output_dir"])
        review_results.append(review_result)
    review = build_review_report(review_results)
    manifest = {
        "settings": settings_payload,
        "input_paths": [str(path) for path in input_paths],
        "results": results,
    }
    manifest_path = dump_json(args.outdir / "manifest.json.gz", manifest)
    summary_path = dump_json(args.outdir / "summary.json", summary)
    review_path = dump_json(args.outdir / "review.json.gz", review)
    html_report_path = args.outdir / "summary_report.html"
    html_report_path.write_text(
        build_batch_html_report(
            summary=summary,
            review=review,
            manifest=manifest,
            artifact_paths={
                "manifest_path": str(manifest_path),
                "summary_path": str(summary_path),
                "review_path": str(review_path),
                "html_report_path": str(html_report_path.resolve()),
            },
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Finished batch: %d success, %d failure",
        summary["success_count"],
        summary["failure_count"],
    )
    _print_batch_result(
        settings,
        args.outdir,
        manifest_path,
        summary_path,
        review_path,
        html_report_path,
        summary,
    )
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
