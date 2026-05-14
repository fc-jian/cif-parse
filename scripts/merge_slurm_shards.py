#!/usr/bin/env python3
"""Merge cif-parse batch outputs produced by Slurm shard workers."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_link_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "case"


def _merge_metadata(shards: list[Path], outdir: Path) -> None:
    metadata_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for shard in shards:
        meta_path = shard / "metadata.csv"
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                metadata_rows.append(dict(row))
    if not metadata_rows:
        return
    with (outdir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def merge_shards(outdir: Path, shard_out_dir: Path, repo_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(repo_root / "src"))

    from cif_parse.cli import _summarize_batch_results  # type: ignore
    from cif_parse.export import dump_json
    from cif_parse.reporting import (
        build_batch_html_report,
        build_review_report,
        collect_case_review_metrics,
    )

    shards = sorted(p for p in shard_out_dir.glob("shard_*") if p.is_dir())
    if not shards:
        raise ValueError(f"No shard outputs found under {shard_out_dir}")

    outdir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    input_paths: list[str] = []
    settings: dict[str, Any] | None = None
    shard_records: list[dict[str, Any]] = []
    missing_manifests: list[str] = []
    seen_case_ids: set[str] = set()
    case_links = outdir / "cases"
    case_links.mkdir(parents=True, exist_ok=True)

    for shard in shards:
        manifest_path = shard / "manifest.json.gz"
        exit_path = shard / ".exit_code"
        exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
        if not manifest_path.exists():
            missing_manifests.append(str(shard))
            shard_records.append({
                "shard": shard.name,
                "manifest": "",
                "exit_code": exit_code,
                "status": "missing_manifest",
            })
            continue

        manifest = _load_json(manifest_path)
        if settings is None:
            settings = manifest.get("settings", {}) if isinstance(manifest.get("settings"), dict) else {}
        input_paths.extend(str(p) for p in manifest.get("input_paths", []) if p)
        shard_results = manifest.get("results", [])
        shard_records.append({
            "shard": shard.name,
            "manifest": str(manifest_path),
            "exit_code": exit_code,
            "result_count": len(shard_results),
            "status": "ok" if exit_code in (0, None) else "nonzero_exit",
        })

        for raw_result in shard_results:
            item = dict(raw_result)
            original_case_id = str(item.get("case_id", ""))
            case_id = original_case_id
            if case_id in seen_case_ids:
                case_id = f"{shard.name}/{case_id}"
                item["case_id"] = case_id
                item["original_case_id"] = original_case_id
            seen_case_ids.add(case_id)

            output_dir = item.get("output_dir")
            if isinstance(output_dir, str) and output_dir:
                src = Path(output_dir)
                if src.exists():
                    link = case_links / _safe_link_name(case_id)
                    if not link.exists():
                        try:
                            link.symlink_to(os.path.relpath(src, link.parent), target_is_directory=True)
                        except OSError:
                            pass
            results.append(item)

    results.sort(key=lambda item: str(item.get("case_id", "")))
    summary = _summarize_batch_results(results)
    if missing_manifests:
        summary["failure_count"] = int(summary.get("failure_count", 0)) + len(missing_manifests)
        summary["missing_shard_manifest_count"] = len(missing_manifests)

    review_results: list[dict[str, Any]] = []
    low_conf_threshold = float((settings or {}).get("low_confidence_antibody_threshold", 0.8))
    for result in results:
        item = dict(result)
        if item.get("status") == "ok":
            try:
                item["metrics"] = collect_case_review_metrics(
                    str(item["output_dir"]),
                    low_confidence_antibody_threshold=low_conf_threshold,
                )
            except Exception as exc:
                item["metrics_error"] = f"{type(exc).__name__}: {exc}"
        review_results.append(item)

    review = build_review_report(review_results)
    review["slurm_shards"] = shard_records
    if missing_manifests:
        review["missing_shard_manifests"] = missing_manifests

    manifest = {
        "settings": settings or {},
        "input_paths": input_paths,
        "results": results,
        "slurm_shards": shard_records,
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
    _merge_metadata(shards, outdir)

    return {
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "review_path": str(review_path),
        "html_report_path": str(html_report_path),
        "success_count": summary.get("success_count", 0),
        "failure_count": summary.get("failure_count", 0),
        "skipped_count": summary.get("skipped_count", 0),
        "missing_shard_manifest_count": len(missing_manifests),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--shard-out-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()

    result = merge_shards(args.outdir, args.shard_out_dir, args.repo_root)
    print(json.dumps(result, indent=2))
    return 1 if int(result["missing_shard_manifest_count"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
