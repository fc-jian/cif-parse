from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cif_parse.cli import _build_metadata_csv, _summarize_batch_results
from cif_parse.export import dump_json, load_case_output_bundles, load_json
from cif_parse.reporting import build_batch_html_report, build_review_report, collect_case_review_metrics
from cif_parse.settings import AppSettings


LOGGER = logging.getLogger("rebuild_parse_report")


def _load_manifest(parse_dir: Path) -> dict[str, Any]:
    manifest_path = parse_dir / "manifest.json.gz"
    if not manifest_path.exists():
        manifest_path = parse_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json.gz not found under {parse_dir}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected dict manifest in {manifest_path}")
    return manifest


def _entry_metadata_from_output(output_dir: str | Path | None) -> dict[str, Any]:
    if not output_dir:
        return {}
    try:
        payloads = load_case_output_bundles(output_dir)
    except Exception:
        LOGGER.debug("Failed to load case bundles from %s", output_dir, exc_info=True)
        return {}
    for payload in payloads:
        summary = payload.get("structure_summary")
        if not isinstance(summary, dict):
            continue
        metadata = summary.get("entry_metadata")
        if isinstance(metadata, dict) and metadata:
            return dict(metadata)
    return {}


def _refresh_result_metadata(result: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(result)
    if refreshed.get("status") != "ok":
        return refreshed
    if not isinstance(refreshed.get("_meta"), dict) or not refreshed.get("_meta"):
        metadata = _entry_metadata_from_output(refreshed.get("output_dir"))
        if metadata:
            refreshed["_meta"] = metadata
    return refreshed


def _load_existing_review_settings(parse_dir: Path, manifest: dict[str, Any]) -> AppSettings:
    settings_payload = manifest.get("settings")
    if isinstance(settings_payload, dict):
        try:
            return AppSettings(**settings_payload)
        except Exception:
            LOGGER.debug("Failed to construct AppSettings from manifest settings", exc_info=True)
    return AppSettings()


def rebuild_report(parse_dir: Path) -> dict[str, Path]:
    manifest = _load_manifest(parse_dir)
    raw_results = manifest.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("manifest does not contain a results list")

    results = [
        _refresh_result_metadata(result)
        for result in raw_results
        if isinstance(result, dict)
    ]
    results.sort(key=lambda item: str(item.get("case_id", "")))

    settings = _load_existing_review_settings(parse_dir, manifest)
    summary = _summarize_batch_results(results)
    review_results: list[dict[str, Any]] = []
    for result in results:
        review_result = dict(result)
        if result.get("status") == "ok":
            output_dir = result.get("output_dir")
            if output_dir:
                review_result["metrics"] = collect_case_review_metrics(
                    output_dir,
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

    manifest = dict(manifest)
    manifest["results"] = results

    manifest_path = dump_json(parse_dir / "manifest.json.gz", manifest)
    summary_path = dump_json(parse_dir / "summary.json", summary)
    review_path = dump_json(parse_dir / "review.json.gz", review)
    html_report_path = parse_dir / "summary_report.html"
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
    _build_metadata_csv(results, parse_dir / "metadata.csv")
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "review": review_path,
        "html_report": html_report_path,
        "metadata": parse_dir / "metadata.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild parse summary/report files from an existing cif-parse batch "
            "output directory without re-running parsing."
        )
    )
    parser.add_argument(
        "parse_dir",
        type=Path,
        help="Existing cif-parse batch output directory containing manifest.json.gz",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    parse_dir = args.parse_dir.resolve()
    if not parse_dir.is_dir():
        raise NotADirectoryError(f"parse output directory not found: {parse_dir}")
    paths = rebuild_report(parse_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
