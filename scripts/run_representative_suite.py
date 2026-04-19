from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cif_parse.cli import main as cli_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the representative mmCIF regression suite")
    parser.add_argument(
        "--input-list",
        type=Path,
        default=REPO_ROOT / "test_representative_list.txt",
        help="Text file containing one PDB id or input path per line",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "test_outputs" / "representative_suite_modes",
        help="Root directory for per-mode outputs",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of worker processes to use for each batch run",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Root log level forwarded to cif_parse.cli batch",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["biological_assembly", "asymmetric_unit"],
        help="Assembly modes to run sequentially",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _priority_case_counts(review: dict[str, Any]) -> dict[str, int]:
    priority_cases = review.get("priority_cases")
    if not isinstance(priority_cases, dict):
        return {}
    return {
        str(name): len(value)
        for name, value in sorted(priority_cases.items())
        if isinstance(value, list)
    }


def _run_mode(
    *,
    input_list: Path,
    outdir: Path,
    jobs: int,
    log_level: str,
    assembly_mode: str,
) -> dict[str, Any]:
    mode_outdir = outdir / assembly_mode
    exit_code = cli_main(
        [
            "batch",
            "--input-list",
            str(input_list),
            "--outdir",
            str(mode_outdir),
            "--format",
            "json",
            "--assembly-mode",
            assembly_mode,
            "--jobs",
            str(jobs),
            "--log-level",
            log_level,
        ]
    )

    summary_path = mode_outdir / "summary.json"
    review_path = mode_outdir / "review.json"
    manifest_path = mode_outdir / "manifest.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    review = _read_json(review_path) if review_path.exists() else {}
    return {
        "assembly_mode": assembly_mode,
        "exit_code": exit_code,
        "outdir": str(mode_outdir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "review_path": str(review_path.resolve()),
        "summary": summary,
        "review_priority_case_counts": _priority_case_counts(review),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    mode_results = [
        _run_mode(
            input_list=args.input_list,
            outdir=args.outdir,
            jobs=args.jobs,
            log_level=args.log_level,
            assembly_mode=assembly_mode,
        )
        for assembly_mode in args.modes
    ]
    index = {
        "input_list": str(args.input_list.resolve()),
        "outdir": str(args.outdir.resolve()),
        "jobs": args.jobs,
        "log_level": args.log_level,
        "modes": mode_results,
        "successful_modes": [
            result["assembly_mode"] for result in mode_results if int(result["exit_code"]) == 0
        ],
        "failed_modes": [
            result["assembly_mode"] for result in mode_results if int(result["exit_code"]) != 0
        ],
    }

    index_path = args.outdir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(index_path)
    return 0 if all(int(result["exit_code"]) == 0 for result in mode_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
