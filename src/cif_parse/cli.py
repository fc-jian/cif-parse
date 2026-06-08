"""Command-line entry points for single-file and batch mmCIF processing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tomllib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import heapq
from pathlib import Path
from typing import Any

from cif_parse.export import dump_json, load_case_output_bundle, load_json, resolve_json_path
from cif_parse.io import read_case_metadata, read_structure_summary
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
    batch_parser.add_argument(
        "--resume",
        "--skip-existing",
        dest="resume",
        action="store_true",
        default=bool(batch_defaults.get("resume", False)),
        help=(
            "Reuse complete existing case outputs under OUTDIR/cases instead of "
            "reprocessing them; completeness requires result JSON and atom cache pkl files"
        ),
    )
    batch_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable reuse of existing case outputs",
    )
    batch_parser.add_argument(
        "--summary-only",
        action="store_true",
        default=bool(batch_defaults.get("summary_only", False)),
        help=(
            "Do not parse input CIFs; rebuild manifest/summary/review/metadata by "
            "scanning existing OUTDIR/cases outputs"
        ),
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
        "--input-assembly",
        action=argparse.BooleanOptionalAction,
        default=bool(settings_defaults.get("input_assembly", False)),
        help="CIF files are already split per-assembly; skip assembly expansion and validate chain uniqueness",
    )
    parser.add_argument(
        "--metadata-cif-dir",
        type=str,
        default=str(settings_defaults.get("metadata_cif_dir", "")),
        help="Directory of original full mmCIF files for entry metadata when input CIF lacks it",
    )
    parser.add_argument(
        "--metadata-table",
        type=str,
        default=str(settings_defaults.get("metadata_table", "")),
        help="Pre-generated parquet/csv with entry metadata (pdb_id, method, resolution, release_date)",
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
        "--max-assembly-atoms",
        type=int,
        default=int(settings_defaults.get("max_assembly_atoms", 300_000)),
        help="Skip assemblies estimated to contain more than this many atoms",
    )
    parser.add_argument(
        "--assembly-jobs",
        type=int,
        default=int(settings_defaults.get("assembly_jobs", 1)),
        help="Number of per-case assembly worker threads for --assembly-mode all",
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
        "--residue-contact-cutoff",
        type=float,
        default=float(settings_defaults.get("residue_contact_cutoff", 8.0)),
        help="Distance cutoff (Å) for residue-level contact detection",
    )
    parser.add_argument(
        "--atom-contact-cutoff",
        type=float,
        default=float(settings_defaults.get("atom_contact_cutoff", 5.0)),
        help="Distance cutoff (Å) for atom-level contact detection",
    )
    parser.add_argument(
        "--min-residue-contacts",
        type=int,
        default=int(settings_defaults.get("min_residue_contacts", 3)),
        help="Minimum residue-residue contacts required to keep a dimer interface",
    )
    parser.add_argument(
        "--min-atom-contacts",
        type=int,
        default=int(settings_defaults.get("min_atom_contacts", 20)),
        help="Minimum atom-atom contacts required to keep a dimer interface",
    )
    parser.add_argument(
        "--peptide-max-length",
        type=int,
        default=int(settings_defaults.get("peptide_max_length", 30)),
        help="Maximum peptide chain length for TCR-pMHC complex assembly",
    )
    parser.add_argument(
        "--sadie-domain-bitscore-threshold",
        type=float,
        default=float(settings_defaults.get("sadie_domain_bitscore_threshold", 80.0)),
        help="Minimum HMMER bitscore for a Sadie variable-domain hit",
    )
    parser.add_argument(
        "--sadie-domain-limit",
        type=int,
        default=int(settings_defaults.get("sadie_domain_limit", 4)),
        help="Maximum number of Sadie variable domains to retain per chain",
    )
    parser.add_argument(
        "--low-confidence-antibody-threshold",
        type=float,
        default=float(settings_defaults.get("low_confidence_antibody_threshold", 0.8)),
        help="Confidence threshold below which an antibody annotation is flagged as low confidence",
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
        assembly_mode="asymmetric_unit" if args.input_assembly else args.assembly_mode,
        input_assembly=args.input_assembly,
        metadata_cif_dir=args.metadata_cif_dir or "",
        metadata_table=args.metadata_table or "",
        coverage_mode=args.coverage_mode,
        debug=args.debug,
        log_level=args.log_level,
        verbose=args.verbose,
        model=args.model,
        use_author_fields=args.author_fields,
        drop_hydrogens_for_analysis=args.drop_hydrogens_for_analysis,
        max_polymer_chains=args.max_polymer_chains,
        max_assembly_atoms=args.max_assembly_atoms,
        assembly_jobs=args.assembly_jobs,
        min_polymer_chain_length=args.min_polymer_chain_length,
        tight_multimer_min_buried_area=args.tight_multimer_min_buried_area,
        tight_multimer_louvain_resolution=args.tight_multimer_louvain_resolution,
        tight_multimer_min_member_instances=args.tight_multimer_min_member_instances,
        tight_multimer_large_component_warning_size=args.tight_multimer_large_component_warning_size,
        residue_contact_cutoff=args.residue_contact_cutoff,
        atom_contact_cutoff=args.atom_contact_cutoff,
        min_residue_contacts=args.min_residue_contacts,
        min_atom_contacts=args.min_atom_contacts,
        peptide_max_length=args.peptide_max_length,
        sadie_domain_bitscore_threshold=args.sadie_domain_bitscore_threshold,
        sadie_domain_limit=args.sadie_domain_limit,
        low_confidence_antibody_threshold=args.low_confidence_antibody_threshold,
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


def _build_existing_case_specs(outdir: Path) -> list[dict[str, Any]]:
    """Build case specs from existing ``outdir/cases/*`` directories."""

    cases_dir = outdir / "cases"
    if not cases_dir.exists():
        return []
    specs: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        specs.append(
            {
                "case_id": case_dir.name,
                "input_path": "",
                "output_dir": str(case_dir),
            }
        )
    return specs


def _build_case_specs_from_manifest(outdir: Path, manifest_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for case_id in sorted(manifest_results):
        row = manifest_results[case_id]
        specs.append(
            {
                "case_id": case_id,
                "input_path": str(row.get("input_path", "") or ""),
                "output_dir": str(row.get("output_dir", "") or outdir / "cases" / case_id),
            }
        )
    return specs


def _resolve_optional_json(path: Path) -> Path | None:
    try:
        return resolve_json_path(path)
    except FileNotFoundError:
        return None


def _assembly_sort_key(label: str) -> tuple[int, int | str]:
    if label.isdigit():
        return (0, int(label))
    return (1, label)


def _assembly_id_from_result_path(path: Path) -> str | None:
    name = path.name
    prefix = "result_assembly_"
    if name.startswith(prefix):
        remainder = name.removeprefix(prefix)
        for suffix in (".json.gz", ".json"):
            if remainder.endswith(suffix):
                remainder = remainder[: -len(suffix)]
                break
        return remainder or None
    if path.parent.name.startswith("assembly_"):
        return path.parent.name.removeprefix("assembly_") or None
    return None


def _find_existing_case_payloads(case_dir: Path) -> list[tuple[dict[str, Any], Path, Path]]:
    """Load existing case payloads with their JSON path and atom-cache root."""

    result_path = _resolve_optional_json(case_dir / "result.json")
    if result_path is not None:
        payload = load_json(result_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {result_path}")
        return [(payload, result_path, case_dir)]

    assembly_paths: dict[str, Path] = {}
    for candidate in list(case_dir.glob("result_assembly_*.json.gz")) + list(
        case_dir.glob("result_assembly_*.json")
    ):
        if candidate.name.endswith(".json.gz"):
            key = candidate.name.removesuffix(".json.gz")
        else:
            key = candidate.name.removesuffix(".json")
        assembly_paths[key] = candidate
    if assembly_paths:
        loaded: list[tuple[dict[str, Any], Path, Path]] = []
        for key, path in sorted(
            assembly_paths.items(),
            key=lambda item: _assembly_sort_key(item[0].removeprefix("result_assembly_")),
        ):
            payload = load_json(path)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload in {path}")
            loaded.append((payload, path, case_dir))
        return loaded

    assembly_dirs = sorted(
        [
            path
            for path in case_dir.iterdir()
            if path.is_dir() and path.name.startswith("assembly_")
        ],
        key=lambda path: _assembly_sort_key(path.name.removeprefix("assembly_")),
    ) if case_dir.exists() else []
    loaded = []
    for assembly_dir in assembly_dirs:
        assembly_result = _resolve_optional_json(assembly_dir / "result.json")
        if assembly_result is not None:
            payload = load_json(assembly_result)
        elif _resolve_optional_json(assembly_dir / "structure_summary.json") is not None:
            payload = load_case_output_bundle(assembly_dir)
            assembly_result = resolve_json_path(assembly_dir / "structure_summary.json")
        else:
            continue
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {assembly_result}")
        loaded.append((payload, assembly_result, assembly_dir))
    if loaded:
        return loaded

    if _resolve_optional_json(case_dir / "structure_summary.json") is not None:
        payload = load_case_output_bundle(case_dir)
        summary_path = resolve_json_path(case_dir / "structure_summary.json")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {summary_path}")
        return [(payload, summary_path, case_dir)]

    return []


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _single_nonempty_assembly_atom_cache(atoms_dir: Path) -> Path | None:
    try:
        candidates = sorted(
            path
            for path in atoms_dir.glob("*.pkl")
            if path.name != "_none.pkl" and _nonempty_file(path)
        )
    except OSError:
        return None
    return candidates[0] if len(candidates) == 1 else None


def _validate_existing_atom_cache(
    payloads: list[tuple[dict[str, Any], Path, Path]],
) -> list[str]:
    missing: list[str] = []
    for payload, result_path, atom_root in payloads:
        summary = payload.get("structure_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        assembly_id = str(summary.get("assembly_id", "") or "")
        if not assembly_id:
            assembly_id = _assembly_id_from_result_path(result_path) or ""
        atoms_dir = atom_root / "atoms"
        if assembly_id:
            asm_path = atoms_dir / f"{assembly_id}.pkl"
            if not _nonempty_file(asm_path):
                missing.append(str(asm_path))
            continue
        au_path = atoms_dir / "_none.pkl"
        if not _nonempty_file(au_path) and _single_nonempty_assembly_atom_cache(atoms_dir) is None:
            missing.append(str(au_path))
    return missing


def _result_from_existing_case(task: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Build a batch result from an existing complete case output."""

    case_dir = Path(task["output_dir"])
    try:
        payloads = _find_existing_case_payloads(case_dir)
        if not payloads:
            return None, "no result JSON found"
        missing_atoms = _validate_existing_atom_cache(payloads)
        if missing_atoms:
            preview = ", ".join(missing_atoms[:4])
            if len(missing_atoms) > 4:
                preview += f", ... ({len(missing_atoms)} total)"
            return None, f"missing atom cache pkl: {preview}"

        primary_payload = payloads[0][0]
        summary = primary_payload.get("structure_summary", {})
        if not isinstance(summary, dict):
            return None, "result JSON missing structure_summary object"
        chain_inventory = primary_payload.get("chain_inventory", [])
        if not isinstance(chain_inventory, list):
            return None, "result JSON missing chain_inventory list"

        output_paths = [str(result_path) for _, result_path, _ in payloads]
        assembly_ids: list[str] = []
        assembly_results: list[dict[str, Any]] = []
        total_dimers = 0
        total_multimers = 0
        total_antibody_complexes = 0
        total_tcr_complexes = 0
        for payload, result_path, _ in payloads:
            item_summary = payload.get("structure_summary", {})
            if not isinstance(item_summary, dict):
                item_summary = {}
            assembly_id = str(item_summary.get("assembly_id", "") or "")
            if not assembly_id:
                assembly_id = _assembly_id_from_result_path(result_path) or ""
            if assembly_id:
                assembly_ids.append(assembly_id)
            dimer_count = len(payload.get("dimer_interfaces", []) or [])
            multimer_count = len(payload.get("tight_multimers", []) or [])
            antibody_count = len(payload.get("antibody_antigen_complexes", []) or [])
            tcr_count = len(payload.get("tcr_pmhc_complexes", []) or [])
            total_dimers += dimer_count
            total_multimers += multimer_count
            total_antibody_complexes += antibody_count
            total_tcr_complexes += tcr_count
            assembly_results.append(
                {
                    "pdb_id": item_summary.get("pdb_id", summary.get("pdb_id", "")),
                    "input_path": item_summary.get("source_path", task.get("input_path", "")),
                    "output_dir": str(result_path.parent),
                    "output_paths": [str(result_path)],
                    "num_chains": len(payload.get("chain_inventory", []) or []),
                    "num_dimers": dimer_count,
                    "num_multimers": multimer_count,
                    "num_antibody_antigen_complexes": antibody_count,
                    "num_tcr_pmhc_complexes": tcr_count,
                    "chain_type_counts": item_summary.get("chain_type_counts", {}),
                    "assembly_id": assembly_id or None,
                }
            )

        metadata = summary.get("entry_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        input_path = str(task.get("input_path") or summary.get("source_path", "") or "")
        result = {
            "case_id": task["case_id"],
            "pdb_id": summary.get("pdb_id", ""),
            "input_path": input_path,
            "output_dir": str(case_dir),
            "output_paths": output_paths,
            "num_chains": len(chain_inventory),
            "num_dimers": total_dimers,
            "num_multimers": total_multimers,
            "num_antibody_antigen_complexes": total_antibody_complexes,
            "num_tcr_pmhc_complexes": total_tcr_complexes,
            "chain_type_counts": summary.get("chain_type_counts", {}),
            "status": "ok",
            "resumed": True,
            "_meta": metadata,
        }
        if assembly_ids:
            result["num_assemblies_processed"] = len(assembly_ids)
            result["processed_assembly_ids"] = assembly_ids
            result["assembly_results"] = assembly_results
        return result, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _settings_fingerprint(settings_payload: dict[str, Any]) -> dict[str, Any]:
    ignored = {"log_level", "verbose"}
    return {
        key: settings_payload[key]
        for key in sorted(settings_payload)
        if key not in ignored
    }


def _load_existing_manifest(outdir: Path) -> dict[str, Any] | None:
    try:
        manifest = load_json(outdir / "manifest.json")
    except FileNotFoundError:
        return None
    except Exception as exc:
        LOGGER.warning("Could not read existing manifest under %s: %s", outdir, exc)
        return None
    return manifest if isinstance(manifest, dict) else None


def _existing_manifest_results_by_case(outdir: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_existing_manifest(outdir)
    if not manifest:
        return {}
    rows = manifest.get("results", [])
    if not isinstance(rows, list):
        return {}
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", "") or "")
        if case_id:
            results[case_id] = row
    return results


def _validate_resume_settings(outdir: Path, settings_payload: dict[str, Any]) -> str | None:
    manifest = _load_existing_manifest(outdir)
    if not manifest:
        return None
    previous = manifest.get("settings")
    if not isinstance(previous, dict):
        return None
    previous_fp = _settings_fingerprint(previous)
    current_fp = _settings_fingerprint(settings_payload)
    if previous_fp == current_fp:
        return None
    changed = sorted(
        key
        for key in set(previous_fp) | set(current_fp)
        if previous_fp.get(key) != current_fp.get(key)
    )
    preview = ", ".join(changed[:8])
    if len(changed) > 8:
        preview += f", ... ({len(changed)} total)"
    return (
        "existing manifest settings differ from current batch settings "
        f"({preview}); run without --resume or use a fresh --outdir"
    )


def _build_metadata_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write a per-case metadata CSV with experimental details and chain counts."""
    import csv

    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda r: r.get("case_id", "")):
        row: dict[str, Any] = {
            "case_id": result.get("case_id", ""),
            "pdb_id": result.get("pdb_id", ""),
            "input_path": result.get("input_path", ""),
            "status": result.get("status", ""),
            "num_chains": result.get("num_chains", 0),
            "num_dimers": result.get("num_dimers", 0),
            "num_multimers": result.get("num_multimers", 0),
            "num_antibody_antigen_complexes": result.get("num_antibody_antigen_complexes", 0),
            "num_tcr_pmhc_complexes": result.get("num_tcr_pmhc_complexes", 0),
            "num_assemblies_processed": result.get("num_assemblies_processed", ""),
            "warning_code": result.get("warning_code", ""),
        }
        # Merge metadata from the result's metadata dict if present.
        meta = result.get("_meta", {})
        if isinstance(meta, dict):
            row.update(meta)
        rows.append(row)

    if not rows:
        return
    preferred = [
        "case_id",
        "pdb_id",
        "input_path",
        "status",
        "experimental_method",
        "resolution",
        "release_date",
        "num_chains",
        "num_polymer_chains",
        "num_asym_ids",
        "num_dimers",
        "num_multimers",
        "num_antibody_antigen_complexes",
        "num_tcr_pmhc_complexes",
        "num_assemblies",
        "num_assemblies_processed",
        "warning_code",
    ]
    all_fields: list[str] = []
    for field in preferred:
        if any(field in row for row in rows):
            all_fields.append(field)
    for row in rows:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote metadata CSV: %s (%d rows)", output_path, len(rows))


def _scan_case_metadata(input_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Read entry metadata, preferring the enriched bundle over raw CIF."""
    # Prefer the enriched metadata from the case output bundle.
    if output_dir:
        try:
            import gzip as _gz, json as _json
            outdir = Path(output_dir)
            for bundle_path in sorted(outdir.glob("*.json.gz")):
                with _gz.open(bundle_path) as fh:
                    bundle = _json.load(fh)
                meta = (bundle.get("structure_summary", {}) or {}).get("entry_metadata", {})
                if isinstance(meta, dict) and meta:
                    return meta
        except Exception:
            pass
    # Fallback: re-read from the raw CIF.
    try:
        return read_case_metadata(input_path)
    except Exception:
        return {}


def _summarize_batch_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact summary file for batch execution."""

    successes = [result for result in results if result["status"] == "ok"]
    skipped = [result for result in results if result["status"] == "skipped"]
    failures = [result for result in results if result["status"] == "error"]
    skipped_warning_counts: dict[str, int] = {}
    for result in skipped:
        warning_code = str(result.get("warning_code", "") or "")
        if warning_code:
            skipped_warning_counts[warning_code] = skipped_warning_counts.get(warning_code, 0) + 1
    total_success = len(successes)
    total_cases = len(results)
    total_chains = sum(int(result.get("num_chains", 0)) for result in successes)
    total_dimers = sum(int(result.get("num_dimers", 0)) for result in successes)
    total_multimers = sum(int(result.get("num_multimers", 0)) for result in successes)
    total_processed_assemblies = sum(int(result.get("num_assemblies_processed", 1)) for result in successes)
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
        "total_processed_assemblies": total_processed_assemblies,
        "total_antibody_antigen_complexes": total_antibody_complexes,
        "total_tcr_pmhc_complexes": total_tcr_complexes,
        "average_chains_per_successful_case": round(total_chains / total_success, 2) if total_success else 0.0,
        "average_dimers_per_successful_case": round(total_dimers / total_success, 2) if total_success else 0.0,
        "average_multimers_per_successful_case": round(total_multimers / total_success, 2) if total_success else 0.0,
        "skipped_warning_counts": dict(sorted(skipped_warning_counts.items())),
    }


def _process_batch_case(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point for one batch case."""

    settings = AppSettings(**task["settings"])
    configure_logging(settings.log_level)
    allowed_assembly_ids = task.get("_preflight_allowed_assembly_ids")
    try:
        result = process_single_structure(
            task["input_path"], task["output_dir"], settings,
            _allowed_assembly_ids=allowed_assembly_ids,
        )
        result["case_id"] = task["case_id"]
        result["status"] = "ok"
        result["_meta"] = result.get("_meta", {})
        return result
    except StructureSkipWarning as exc:
        LOGGER.warning("%s", exc)
        metadata = _scan_case_metadata(task["input_path"])
        return {
            "case_id": task["case_id"],
            "pdb_id": exc.details.get("pdb_id") or metadata.get("pdb_id") or "",
            "input_path": task["input_path"],
            "output_dir": task["output_dir"],
            "status": "skipped",
            "warning_code": exc.code,
            "warning": str(exc),
            "warning_details": exc.details,
            "_meta": metadata,
        }
    except Exception as exc:  # pragma: no cover - exercised through CLI behavior
        LOGGER.exception("Failed to process %s", task["input_path"])
        metadata = _scan_case_metadata(task["input_path"])
        return {
            "case_id": task["case_id"],
            "input_path": task["input_path"],
            "output_dir": task["output_dir"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "_meta": metadata,
        }


def _prepare_preflighted_batch_task(
    task: dict[str, Any],
    counts: dict[str, int],
    *,
    max_assembly_atoms: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """Apply assembly preflight results to one batch task.

    Returns ``(task_to_run, skipped_result, priority_atoms)``.  Empty counts mean
    preflight was unavailable, so the worker must fall back to its own assembly
    discovery/filtering path.
    """

    if not counts:
        return task, None, 0
    allowed = sorted(
        [aid for aid, n in counts.items() if n <= max_assembly_atoms],
        key=lambda aid: -counts.get(aid, 0),
    )
    if not allowed:
        return (
            None,
            {
                "case_id": task["case_id"],
                "input_path": task["input_path"],
                "output_dir": task["output_dir"],
                "status": "skipped",
                "warning_code": "all_assemblies_exceed_max_atoms",
                "warning": f"Skipping {task['case_id']}: all assemblies exceed max_assembly_atoms",
                "warning_details": {
                    "assembly_atom_counts": counts,
                    "max_assembly_atoms": max_assembly_atoms,
                },
            },
            0,
        )
    prepared_task = dict(task)
    prepared_task["_preflight_allowed_assembly_ids"] = allowed
    return prepared_task, None, max(counts.get(aid, 0) for aid in allowed)


def _print_single_result(settings: AppSettings, outdir: Path, result: dict[str, Any]) -> None:
    """Print the single-run CLI response."""

    if settings.verbose or result.get("status") != "ok":
        print(
            json.dumps(
                {
                    "settings": {
                        "output_format": settings.output_format,
                        "assembly_mode": settings.assembly_mode,
                        "input_assembly": settings.input_assembly,
                        "metadata_cif_dir": settings.metadata_cif_dir,
                        "metadata_table": settings.metadata_table,
                        "coverage_mode": settings.coverage_mode,
                        "debug": settings.debug,
                        "log_level": settings.log_level,
                        "max_polymer_chains": settings.max_polymer_chains,
                        "max_assembly_atoms": settings.max_assembly_atoms,
                        "assembly_jobs": settings.assembly_jobs,
                        "min_polymer_chain_length": settings.min_polymer_chain_length,
                        "model": settings.model,
                        "use_author_fields": settings.use_author_fields,
                        "drop_hydrogens_for_analysis": settings.drop_hydrogens_for_analysis,
                        "tight_multimer_min_buried_area": settings.tight_multimer_min_buried_area,
                        "tight_multimer_louvain_resolution": settings.tight_multimer_louvain_resolution,
                        "tight_multimer_min_member_instances": settings.tight_multimer_min_member_instances,
                        "tight_multimer_large_component_warning_size": settings.tight_multimer_large_component_warning_size,
                        "residue_contact_cutoff": settings.residue_contact_cutoff,
                        "atom_contact_cutoff": settings.atom_contact_cutoff,
                        "min_residue_contacts": settings.min_residue_contacts,
                        "min_atom_contacts": settings.min_atom_contacts,
                        "peptide_max_length": settings.peptide_max_length,
                        "sadie_domain_bitscore_threshold": settings.sadie_domain_bitscore_threshold,
                        "sadie_domain_limit": settings.sadie_domain_limit,
                        "low_confidence_antibody_threshold": settings.low_confidence_antibody_threshold,
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
                        "input_assembly": settings.input_assembly,
                        "metadata_cif_dir": settings.metadata_cif_dir,
                        "metadata_table": settings.metadata_table,
                        "coverage_mode": settings.coverage_mode,
                        "debug": settings.debug,
                        "log_level": settings.log_level,
                        "max_polymer_chains": settings.max_polymer_chains,
                        "max_assembly_atoms": settings.max_assembly_atoms,
                        "assembly_jobs": settings.assembly_jobs,
                        "min_polymer_chain_length": settings.min_polymer_chain_length,
                        "model": settings.model,
                        "use_author_fields": settings.use_author_fields,
                        "drop_hydrogens_for_analysis": settings.drop_hydrogens_for_analysis,
                        "tight_multimer_min_buried_area": settings.tight_multimer_min_buried_area,
                        "tight_multimer_louvain_resolution": settings.tight_multimer_louvain_resolution,
                        "tight_multimer_min_member_instances": settings.tight_multimer_min_member_instances,
                        "tight_multimer_large_component_warning_size": settings.tight_multimer_large_component_warning_size,
                        "residue_contact_cutoff": settings.residue_contact_cutoff,
                        "atom_contact_cutoff": settings.atom_contact_cutoff,
                        "min_residue_contacts": settings.min_residue_contacts,
                        "min_atom_contacts": settings.min_atom_contacts,
                        "peptide_max_length": settings.peptide_max_length,
                        "sadie_domain_bitscore_threshold": settings.sadie_domain_bitscore_threshold,
                        "sadie_domain_limit": settings.sadie_domain_limit,
                        "low_confidence_antibody_threshold": settings.low_confidence_antibody_threshold,
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


def _write_batch_artifacts(
    *,
    settings: AppSettings,
    settings_payload: dict[str, Any],
    outdir: Path,
    input_paths: list[str | Path],
    results: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    results.sort(key=lambda item: item["case_id"])
    summary = _summarize_batch_results(results)
    resumed_count = sum(1 for result in results if result.get("resumed"))
    if resumed_count:
        summary["resumed_existing_count"] = resumed_count
    review_results: list[dict[str, Any]] = []
    for result in results:
        review_result = dict(result)
        if result.get("status") == "ok":
            review_result["metrics"] = collect_case_review_metrics(
                result["output_dir"],
                low_confidence_antibody_threshold=settings.low_confidence_antibody_threshold,
            )
        review_results.append(review_result)
    review = build_review_report(review_results)
    warning_counts = review.get("warning_counts", {})
    if isinstance(warning_counts, dict):
        batch_warning_counts = warning_counts.get("batch", {})
        if isinstance(batch_warning_counts, dict):
            summary["warning_counts"] = batch_warning_counts
            summary["warning_count"] = sum(int(value) for value in batch_warning_counts.values())
    manifest = {
        "settings": settings_payload,
        "input_paths": [str(path) for path in input_paths],
        "results": results,
    }
    manifest_path = dump_json(outdir / "manifest.json.gz", manifest)
    summary_path = dump_json(outdir / "summary.json", summary)
    review_path = dump_json(outdir / "review.json.gz", review)
    html_report_path = outdir / "summary_report.html"
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
    _build_metadata_csv(results, outdir / "metadata.csv")
    return summary, manifest_path, summary_path, review_path, html_report_path


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
                "pdb_id": exc.details.get("pdb_id") or infer_case_id(args.input),
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

    settings_payload = {
        "output_format": settings.output_format,
        "assembly_mode": settings.assembly_mode,
        "input_assembly": settings.input_assembly,
        "metadata_cif_dir": settings.metadata_cif_dir,
        "metadata_table": settings.metadata_table,
        "coverage_mode": settings.coverage_mode,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "verbose": settings.verbose,
        "model": settings.model,
        "use_author_fields": settings.use_author_fields,
        "drop_hydrogens_for_analysis": settings.drop_hydrogens_for_analysis,
        "max_polymer_chains": settings.max_polymer_chains,
        "max_assembly_atoms": settings.max_assembly_atoms,
        "assembly_jobs": settings.assembly_jobs,
        "min_polymer_chain_length": settings.min_polymer_chain_length,
        "tight_multimer_min_buried_area": settings.tight_multimer_min_buried_area,
        "tight_multimer_louvain_resolution": settings.tight_multimer_louvain_resolution,
        "tight_multimer_min_member_instances": settings.tight_multimer_min_member_instances,
        "tight_multimer_large_component_warning_size": settings.tight_multimer_large_component_warning_size,
        "residue_contact_cutoff": settings.residue_contact_cutoff,
        "atom_contact_cutoff": settings.atom_contact_cutoff,
        "min_residue_contacts": settings.min_residue_contacts,
        "min_atom_contacts": settings.min_atom_contacts,
        "peptide_max_length": settings.peptide_max_length,
        "sadie_domain_bitscore_threshold": settings.sadie_domain_bitscore_threshold,
        "sadie_domain_limit": settings.sadie_domain_limit,
        "low_confidence_antibody_threshold": settings.low_confidence_antibody_threshold,
    }

    if args.summary_only:
        existing_manifest = _load_existing_manifest(args.outdir)
        manifest_results = _existing_manifest_results_by_case(args.outdir)
        summary_settings_payload = (
            existing_manifest.get("settings")
            if isinstance(existing_manifest, dict) and isinstance(existing_manifest.get("settings"), dict)
            else settings_payload
        )
        if args.inputs or args.input_list is not None:
            input_paths = _resolve_batch_inputs(args.inputs, args.input_list)
            case_specs = _build_case_specs(input_paths, args.outdir)
            manifest_input_paths: list[str | Path] = input_paths
        elif manifest_results:
            case_specs = _build_case_specs_from_manifest(args.outdir, manifest_results)
            manifest_input_paths = (
                existing_manifest.get("input_paths", [])
                if isinstance(existing_manifest, dict) and isinstance(existing_manifest.get("input_paths"), list)
                else []
            )
        else:
            case_specs = _build_existing_case_specs(args.outdir)
            manifest_input_paths = []
        if not case_specs:
            parser.error("--summary-only found no existing case outputs under OUTDIR/cases")
        LOGGER.info("Rebuilding batch summary from %d existing case output(s)", len(case_specs))
        results = []
        for task in case_specs:
            manifest_result = manifest_results.get(str(task["case_id"]))
            if manifest_result is not None and manifest_result.get("status") != "ok":
                results.append(dict(manifest_result))
                if not manifest_input_paths and manifest_result.get("input_path"):
                    manifest_input_paths.append(str(manifest_result["input_path"]))
                continue
            result, reason = _result_from_existing_case(task)
            if result is not None:
                results.append(result)
                if not manifest_input_paths and result.get("input_path"):
                    manifest_input_paths.append(str(result["input_path"]))
                continue
            results.append(
                {
                    "case_id": task["case_id"],
                    "input_path": task.get("input_path", ""),
                    "output_dir": task["output_dir"],
                    "status": "error",
                    "error": f"existing case output is incomplete: {reason}",
                    "_meta": {},
                }
            )
        summary, manifest_path, summary_path, review_path, html_report_path = _write_batch_artifacts(
            settings=settings,
            settings_payload=summary_settings_payload,
            outdir=args.outdir,
            input_paths=manifest_input_paths,
            results=results,
        )
        LOGGER.info(
            "Rebuilt batch summary: %d success, %d failure",
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

    input_paths = _resolve_batch_inputs(args.inputs, args.input_list)
    case_specs = _build_case_specs(input_paths, args.outdir)
    LOGGER.info(
        "Starting batch processing for %d inputs with %d worker(s)",
        len(case_specs),
        args.jobs,
    )
    if args.resume:
        resume_error = _validate_resume_settings(args.outdir, settings_payload)
        if resume_error:
            parser.error(resume_error)
    tasks = [
        {
            **case_spec,
            "settings": settings_payload,
        }
        for case_spec in case_specs
    ]

    results: list[dict[str, Any]] = []
    if args.resume:
        manifest_results = _existing_manifest_results_by_case(args.outdir)
        pending_tasks: list[dict[str, Any]] = []
        resumed = 0
        for task in tasks:
            manifest_result = manifest_results.get(str(task["case_id"]))
            if manifest_results and (
                manifest_result is None
                or manifest_result.get("status") != "ok"
                or (
                    task.get("input_path")
                    and manifest_result.get("input_path")
                    and str(manifest_result.get("input_path")) != str(task.get("input_path"))
                )
            ):
                LOGGER.debug(
                    "Existing output for %s is not reusable according to previous manifest",
                    task["case_id"],
                )
                pending_tasks.append(task)
                continue
            result, reason = _result_from_existing_case(task)
            if result is None:
                LOGGER.debug(
                    "Existing output for %s is not reusable: %s",
                    task["case_id"],
                    reason,
                )
                pending_tasks.append(task)
                continue
            results.append(result)
            resumed += 1
        tasks = pending_tasks
        if resumed:
            LOGGER.info(
                "Reused %d existing complete case output(s); %d case(s) remain to process",
                resumed,
                len(tasks),
            )

    # ── Streaming preflight → priority heap → process pool ──────────────
    # Pre-scan preflight results stream into a max-heap (largest assembly
    # first).  As soon as a ProcessPoolExecutor worker slot opens, the
    # heaviest ready task is submitted — no waiting for all files to be
    # pre-scanned before processing begins.
    streaming_scheduler_ran = False
    if settings.assembly_mode == "all" and settings.max_assembly_atoms > 0 and args.jobs > 1:
        from cif_parse.io.cif_reader import preflight_assembly_atom_counts

        streaming_scheduler_ran = True
        threshold = settings.max_assembly_atoms

        def _preflight_task(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
            path = task["input_path"]
            try:
                return (task, preflight_assembly_atom_counts(path))
            except Exception:
                return (task, {})

        preflight_workers = max(1, min((args.jobs or 1) * 2, len(tasks)))
        preflight_window = max(preflight_workers, args.jobs * 4)
        priority_buffer = max(args.jobs * 4, args.jobs)
        task_heap: list[tuple[int, int, dict[str, Any]]] = []  # (-max_atoms, tie, task)
        task_iter = iter(tasks)
        tie = 0
        pre_skipped = 0

        with (
            ThreadPoolExecutor(max_workers=preflight_workers) as preflight_exec,
            ProcessPoolExecutor(max_workers=args.jobs) as process_exec,
        ):
            preflight_futures: dict[Any, dict[str, Any]] = {}
            process_futures: dict[Any, dict[str, Any]] = {}

            def _fill_preflight_window() -> None:
                while len(preflight_futures) < preflight_window:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        break
                    preflight_futures[preflight_exec.submit(_preflight_task, task)] = task

            def _enqueue_ready(task: dict[str, Any], counts: dict[str, int]) -> None:
                nonlocal pre_skipped, tie
                prepared, skipped, max_atoms = _prepare_preflighted_batch_task(
                    task,
                    counts,
                    max_assembly_atoms=threshold,
                )
                if skipped is not None:
                    pre_skipped += 1
                    results.append(skipped)
                    return
                if prepared is not None:
                    heapq.heappush(task_heap, (-max_atoms, tie, prepared))
                    tie += 1

            _fill_preflight_window()

            while preflight_futures or task_heap or process_futures:
                can_submit = len(task_heap) >= priority_buffer or not preflight_futures
                while can_submit and task_heap and len(process_futures) < args.jobs:
                    _, _, task = heapq.heappop(task_heap)
                    fut = process_exec.submit(_process_batch_case, task)
                    process_futures[fut] = task

                wait_set = set(process_futures) | set(preflight_futures)
                if not wait_set:
                    break
                done, _ = wait(wait_set, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in preflight_futures:
                        preflight_futures.pop(future)
                        task, counts = future.result()
                        _enqueue_ready(task, counts)
                        _fill_preflight_window()
                    elif future in process_futures:
                        task = process_futures.pop(future)
                        result = future.result()
                        results.append(result)
                        if args.fail_fast and result["status"] == "error":
                            LOGGER.error("Batch failed fast on %s", result["input_path"])
                            for f in preflight_futures:
                                f.cancel()
                            for f in process_futures:
                                f.cancel()
                            preflight_futures.clear()
                            process_futures.clear()
                            task_heap.clear()

        if pre_skipped:
            LOGGER.info(
                "Pre-skipped %d case(s) with all assemblies exceeding %d atoms",
                pre_skipped, threshold,
            )

    elif args.jobs > 1 and len(tasks) > 1:
        def _task_size(task: dict[str, Any]) -> int:
            try:
                return Path(task["input_path"]).stat().st_size
            except OSError:
                return 0
        tasks.sort(key=_task_size, reverse=True)
        largest = Path(tasks[0]["input_path"])
        LOGGER.info(
            "Largest input submitted first: %s (%.1f MB)",
            largest.name,
            _task_size(tasks[0]) / (1 << 20),
        )

    if not streaming_scheduler_ran:
        # No streaming path taken — process sequentially or with simple pool.
        if args.jobs == 1:
            for task in tasks:
                result = _process_batch_case(task)
                results.append(result)
                if args.fail_fast and result["status"] == "error":
                    break
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                task_iter = iter(tasks)
                futures: dict[Any, dict[str, Any]] = {}
                window = max(args.jobs, args.jobs * 4)

                def _fill_process_window() -> None:
                    while len(futures) < window:
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            break
                        futures[executor.submit(_process_batch_case, task)] = task

                _fill_process_window()
                while futures:
                    done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        result = future.result()
                        results.append(result)
                        if args.fail_fast and result["status"] == "error":
                            LOGGER.error("Batch failed fast on %s", result["input_path"])
                            for pending in futures:
                                pending.cancel()
                            futures.clear()
                            break
                    if args.fail_fast and results and results[-1]["status"] == "error":
                        break
                    _fill_process_window()

    summary, manifest_path, summary_path, review_path, html_report_path = _write_batch_artifacts(
        settings=settings,
        settings_payload=settings_payload,
        outdir=args.outdir,
        input_paths=input_paths,
        results=results,
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
