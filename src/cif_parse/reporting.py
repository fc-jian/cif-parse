"""Shared reporting helpers for batch and smoke-test outputs."""

from __future__ import annotations

import html
import json
from pathlib import Path

from cif_parse.constants import (
    ANTIBODY_CHAIN_TYPES,
    COVERAGE_WARNING_CODES,
    HTML_SKIPPED_TARGET_DETAIL_LIMIT,
    LOW_CONFIDENCE_ANTIBODY_THRESHOLD,
)
from cif_parse.export import load_case_output_bundles

HTML_WARNING_CASE_DETAIL_LIMIT = 10


def _antibody_analysis(chain: dict[str, object]) -> dict[str, object]:
    features = chain.get("features")
    if not isinstance(features, dict):
        return {}
    analysis = features.get("antibody_analysis")
    if not isinstance(analysis, dict):
        return {}
    return analysis


def _antibody_unit_type(chain: dict[str, object]) -> str:
    analysis = _antibody_analysis(chain)
    unit_type = analysis.get("unit_type")
    if isinstance(unit_type, str) and unit_type:
        return unit_type
    features = chain.get("features")
    if isinstance(features, dict):
        feature_unit_type = features.get("antibody_unit_type")
        if isinstance(feature_unit_type, str) and feature_unit_type:
            return feature_unit_type
    subtype = chain.get("subtype")
    if subtype in {"VHH", "scFv"}:
        return str(subtype)
    return ""


def _true_heavy_only(chain: dict[str, object]) -> bool:
    if chain.get("chain_type") != "antibody heavy chain":
        return False
    analysis = _antibody_analysis(chain)
    evidence = analysis.get("heavy_only_evidence")
    if isinstance(evidence, dict):
        is_true_heavy_only = evidence.get("is_true_heavy_only")
        if isinstance(is_true_heavy_only, bool):
            return is_true_heavy_only
    return _antibody_unit_type(chain) not in {"VHH", "scFv"} and not bool(chain.get("paired_label_asym_id"))


def _warning_list(record: dict[str, object]) -> list[str]:
    warnings = record.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [str(item) for item in warnings if isinstance(item, str)]


def _evidence_dict(record: dict[str, object]) -> dict[str, object]:
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        return {}
    return evidence


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _warning_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for warning in _warning_list(record):
            counts[warning] = counts.get(warning, 0) + 1
    return dict(sorted(counts.items()))


def _merge_count_maps(*maps: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in maps:
        for key, value in item.items():
            counts[key] = counts.get(key, 0) + int(value)
    return dict(sorted(counts.items()))


def _summary_warning_counts(payloads: list[dict[str, object]]) -> dict[str, int]:
    records: list[dict[str, object]] = []
    for payload in payloads:
        summary = payload.get("structure_summary")
        if isinstance(summary, dict):
            records.append(summary)
            summary_warnings = set(_warning_list(summary))
            metadata = summary.get("entry_metadata")
            if isinstance(metadata, dict):
                metadata_warning = metadata.get("metadata_warning")
                if isinstance(metadata_warning, str) and metadata_warning and metadata_warning not in summary_warnings:
                    records.append({"warnings": [metadata_warning]})
    return _warning_counts(records)


def collect_case_review_metrics(
    case_outdir: str | Path,
    *,
    low_confidence_antibody_threshold: float = 0.8,
) -> dict[str, object]:
    """Collect the per-case metrics used by `review.json`."""

    case_outdir = Path(case_outdir)
    payloads = load_case_output_bundles(case_outdir)
    primary_payload = payloads[0]
    chain_inventory = primary_payload["chain_inventory"]
    tight_multimers = [
        multimer
        for payload in payloads
        for multimer in payload["tight_multimers"]
    ]
    antibody_antigen_complexes = [
        complex_record
        for payload in payloads
        for complex_record in payload["antibody_antigen_complexes"]
    ]
    tcr_pmhc_complexes = [
        complex_record
        for payload in payloads
        for complex_record in payload["tcr_pmhc_complexes"]
    ]

    antibody_chains = [
        chain for chain in chain_inventory if chain.get("chain_type") in ANTIBODY_CHAIN_TYPES
    ]
    pairable_antibody_heavy_chains = [
        chain
        for chain in antibody_chains
        if chain.get("chain_type") == "antibody heavy chain" and _antibody_unit_type(chain) not in {"VHH", "scFv"}
    ]

    def _antibody_confidence(chain: dict[str, object]) -> float:
        analysis = _antibody_analysis(chain)
        confidence = analysis.get("annotation_confidence")
        if isinstance(confidence, (int, float)):
            return float(confidence)
        if isinstance(confidence, str):
            mapped = {"high": 0.95, "medium": 0.85, "low": 0.70}.get(confidence.lower())
            if mapped is not None:
                return mapped
        raw_confidence = chain.get("annotation_confidence")
        if isinstance(raw_confidence, (int, float)):
            return float(raw_confidence)
        return 0.0

    contextual_peptide_chain_ids: set[str] = set()
    chain_warning_counts = _warning_counts(chain_inventory)
    antibody_complex_warning_counts = _warning_counts(antibody_antigen_complexes)
    tcr_complex_warning_counts = _warning_counts(tcr_pmhc_complexes)
    multimer_warning_counts = _warning_counts(tight_multimers)
    summary_warning_counts = _summary_warning_counts(payloads)
    parse_warning_counts = _merge_count_maps(
        summary_warning_counts,
        chain_warning_counts,
        antibody_complex_warning_counts,
        tcr_complex_warning_counts,
        multimer_warning_counts,
    )
    for complex_record in tcr_pmhc_complexes:
        evidence = _evidence_dict(complex_record)
        contextual_peptide_chain_ids.update(_string_list(evidence.get("contextual_peptide_chain_ids")))

    return {
        "num_low_confidence_antibody_chains": sum(
            1 for chain in antibody_chains if _antibody_confidence(chain) < low_confidence_antibody_threshold
        ),
        "num_unpaired_antibody_heavy_chains": sum(
            1
            for chain in pairable_antibody_heavy_chains
            if not isinstance(chain.get("paired_label_asym_id"), str) or not chain.get("paired_label_asym_id")
        ),
        "num_unpaired_antibody_light_chains": sum(
            1
            for chain in antibody_chains
            if chain.get("chain_type") == "antibody light chain"
            and (not isinstance(chain.get("paired_label_asym_id"), str) or not chain.get("paired_label_asym_id"))
        ),
        "num_true_heavy_only_chains": sum(1 for chain in antibody_chains if _true_heavy_only(chain)),
        "num_antibody_annotation_conflicts": sum(
            1
            for chain in antibody_chains
            if "description_and_sequence_antibody_annotation_disagree" in (chain.get("warnings") or [])
        ),
        "num_contextual_peptide_chains_in_complexes": len(contextual_peptide_chain_ids),
        "num_antibody_antigen_complexes": len(antibody_antigen_complexes),
        "num_tcr_pmhc_complexes": len(tcr_pmhc_complexes),
        "num_tcr_chains": sum(1 for chain in chain_inventory if chain.get("chain_type") == "TCR chain"),
        "num_mhc_heavy_chains": sum(
            1 for chain in chain_inventory if chain.get("chain_type") == "MHC heavy chain"
        ),
        "num_processed_assemblies": len(payloads),
        "chain_warning_counts": chain_warning_counts,
        "antibody_complex_warning_counts": antibody_complex_warning_counts,
        "tcr_complex_warning_counts": dict(sorted(tcr_complex_warning_counts.items())),
        "multimer_warning_counts": multimer_warning_counts,
        "summary_warning_counts": summary_warning_counts,
        "parse_warning_counts": parse_warning_counts,
        "total_parse_warnings": sum(parse_warning_counts.values()),
    }


def _top_cases(
    successful_results: list[dict[str, object]],
    metric_name: str,
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    ranked = [
        {
            "case_id": str(result["case_id"]),
            metric_name: int(result["metrics"][metric_name]),
        }
        for result in successful_results
        if int(result.get("metrics", {}).get(metric_name, 0)) > 0
    ]
    ranked.sort(key=lambda item: (-int(item[metric_name]), item["case_id"]))
    return ranked[:limit]


def _warning_case_count(
    metrics: dict[str, object],
    metric_name: str,
    warning_codes: str | frozenset[str] | tuple[str, ...] | list[str] | set[str],
) -> int:
    warning_counts = metrics.get(metric_name, {})
    if not isinstance(warning_counts, dict):
        return 0
    if isinstance(warning_codes, str):
        codes = [warning_codes]
    else:
        codes = list(warning_codes)
    return sum(int(warning_counts.get(code, 0)) for code in codes)


def _top_warning_cases(
    successful_results: list[dict[str, object]],
    metric_name: str,
    warning_codes: str | frozenset[str] | tuple[str, ...] | list[str] | set[str],
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    ranked = []
    for result in successful_results:
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        warning_count = _warning_case_count(metrics, metric_name, warning_codes)
        if warning_count <= 0:
            continue
        ranked.append(
            {
                "case_id": str(result["case_id"]),
                "warning_count": warning_count,
            }
        )
    ranked.sort(key=lambda item: (-int(item["warning_count"]), item["case_id"]))
    return ranked[:limit]


def _aggregate_metric_warning_counts(
    successful_results: list[dict[str, object]],
    metric_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in successful_results:
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        warning_counts = metrics.get(metric_name, {})
        if not isinstance(warning_counts, dict):
            continue
        for code, count in warning_counts.items():
            if not isinstance(code, str) or not code:
                continue
            counts[code] = counts.get(code, 0) + int(count)
    return dict(sorted(counts.items()))


def _top_warning_cases_by_metric(
    successful_results: list[dict[str, object]],
    metric_name: str,
    *,
    limit_per_warning: int = 10,
) -> dict[str, list[dict[str, object]]]:
    aggregate = _aggregate_metric_warning_counts(successful_results, metric_name)
    return {
        warning_code: _top_warning_cases(
            successful_results,
            metric_name,
            warning_code,
            limit=limit_per_warning,
        )
        for warning_code in aggregate
    }


def _warning_cases_by_metric(
    successful_results: list[dict[str, object]],
    metric_name: str,
) -> dict[str, list[dict[str, object]]]:
    aggregate = _aggregate_metric_warning_counts(successful_results, metric_name)
    cases_by_warning: dict[str, list[dict[str, object]]] = {}
    for warning_code in aggregate:
        cases = []
        for result in successful_results:
            metrics = result.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            warning_count = _warning_case_count(metrics, metric_name, warning_code)
            if warning_count <= 0:
                continue
            cases.append(
                {
                    "case_id": str(result["case_id"]),
                    "warning_count": warning_count,
                }
            )
        cases.sort(key=lambda item: (-int(item["warning_count"]), item["case_id"]))
        cases_by_warning[warning_code] = cases
    return cases_by_warning


def _status_cases(results: list[dict[str, object]], status: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for result in results:
        if result.get("status") != status:
            continue
        item = {
            "case_id": str(result.get("case_id", "")),
            "input_path": str(result.get("input_path", "")),
        }
        if isinstance(result.get("output_dir"), str):
            item["output_dir"] = str(result["output_dir"])
        if status == "skipped":
            item["warning_code"] = str(result.get("warning_code", ""))
            item["warning"] = str(result.get("warning", ""))
            warning_details = result.get("warning_details")
            item["warning_details"] = warning_details if isinstance(warning_details, dict) else {}
        if status == "error":
            item["error"] = str(result.get("error", ""))
        items.append(item)
    return sorted(items, key=lambda item: item["case_id"])


def _count_by_key(
    items: list[dict[str, object]],
    *,
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_review_report(results: list[dict[str, object]]) -> dict[str, object]:
    """Build the standard `review.json` payload from per-case metrics."""

    successful_results = [result for result in results if result.get("status") == "ok" and "metrics" in result]
    skipped_targets = _status_cases(results, "skipped")
    failed_targets = _status_cases(results, "error")
    successful_warning_counts = _aggregate_metric_warning_counts(successful_results, "parse_warning_counts")
    skipped_warning_counts = _count_by_key(skipped_targets, key="warning_code")
    batch_warning_counts = _merge_count_maps(successful_warning_counts, skipped_warning_counts)
    review = {
        "status_summary": {
            "success_count": len(successful_results),
            "skipped_count": len(skipped_targets),
            "failure_count": len(failed_targets),
        },
        "skipped_targets": skipped_targets,
        "skipped_target_counts_by_reason": skipped_warning_counts,
        "failed_targets": failed_targets,
        "warning_counts": {
            "batch": batch_warning_counts,
            "successful_parse": successful_warning_counts,
            "skipped": skipped_warning_counts,
            "summary": _aggregate_metric_warning_counts(successful_results, "summary_warning_counts"),
            "chain": _aggregate_metric_warning_counts(successful_results, "chain_warning_counts"),
            "antibody_complex": _aggregate_metric_warning_counts(successful_results, "antibody_complex_warning_counts"),
            "tcr_complex": _aggregate_metric_warning_counts(successful_results, "tcr_complex_warning_counts"),
            "multimer": _aggregate_metric_warning_counts(successful_results, "multimer_warning_counts"),
        },
        "warning_cases": {
            "summary": _warning_cases_by_metric(successful_results, "summary_warning_counts"),
            "chain": _warning_cases_by_metric(successful_results, "chain_warning_counts"),
            "antibody_complex": _warning_cases_by_metric(successful_results, "antibody_complex_warning_counts"),
            "tcr_complex": _warning_cases_by_metric(successful_results, "tcr_complex_warning_counts"),
            "multimer": _warning_cases_by_metric(successful_results, "multimer_warning_counts"),
            "parse": _warning_cases_by_metric(successful_results, "parse_warning_counts"),
        },
        "priority_cases": {
            "antibody_low_confidence": _top_cases(successful_results, "num_low_confidence_antibody_chains"),
            "antibody_unpaired_heavy": _top_cases(successful_results, "num_unpaired_antibody_heavy_chains"),
            "antibody_unpaired_light": _top_cases(successful_results, "num_unpaired_antibody_light_chains"),
            "antibody_true_heavy_only": _top_cases(successful_results, "num_true_heavy_only_chains"),
            "antibody_annotation_conflicts": _top_cases(successful_results, "num_antibody_annotation_conflicts"),
            "tcr_contextual_peptide": _top_cases(successful_results, "num_contextual_peptide_chains_in_complexes"),
            "immune_annotation_conflicts": _top_warning_cases(
                successful_results,
                "chain_warning_counts",
                "description_and_sequence_immune_annotation_disagree",
            ),
            "missing_auth_asym_id_warning": _top_warning_cases(
                successful_results,
                "chain_warning_counts",
                "auth_asym_id not found in scheme categories or atom_site",
            ),
            "coverage_assignment_warning": _top_warning_cases(
                successful_results,
                "chain_warning_counts",
                COVERAGE_WARNING_CODES,
            ),
            "antibody_antigen_contacts_not_on_heavy_warning": _top_warning_cases(
                successful_results,
                "antibody_complex_warning_counts",
                "paired_unit_antigen_contacts_not_on_heavy_chain",
            ),
            "tcr_unpaired_chain_warning": _top_warning_cases(
                successful_results,
                "tcr_complex_warning_counts",
                "unpaired_tcr_chain",
            ),
            "large_multimer_after_bridge_pruning_warning": _top_warning_cases(
                successful_results,
                "multimer_warning_counts",
                "large_component_after_bridge_pruning",
            ),
        }
    }
    review["priority_cases"]["tcr_with_tcr_and_mhc_but_no_complex"] = [
        {
            "case_id": str(result["case_id"]),
            "num_tcr_chains": int(result["metrics"]["num_tcr_chains"]),
            "num_mhc_heavy_chains": int(result["metrics"]["num_mhc_heavy_chains"]),
        }
        for result in sorted(successful_results, key=lambda item: str(item["case_id"]))
        if int(result["metrics"]["num_tcr_chains"]) > 0
        and int(result["metrics"]["num_mhc_heavy_chains"]) > 0
        and int(result["metrics"]["num_tcr_pmhc_complexes"]) == 0
    ]
    review["priority_cases"]["tcr_missing_peptide_warning"] = [
        {
            "case_id": str(result["case_id"]),
            "warning_count": int(result["metrics"]["tcr_complex_warning_counts"]["mhc_without_bound_peptide"]),
        }
        for result in successful_results
        if "mhc_without_bound_peptide" in result["metrics"].get("tcr_complex_warning_counts", {})
    ]
    review["priority_cases"]["tcr_missing_auxiliary_warning"] = [
        {
            "case_id": str(result["case_id"]),
            "warning_count": int(
                result["metrics"]["tcr_complex_warning_counts"]["class_i_mhc_without_beta2m_or_auxiliary_chain"]
            ),
        }
        for result in successful_results
        if "class_i_mhc_without_beta2m_or_auxiliary_chain"
        in result["metrics"].get("tcr_complex_warning_counts", {})
    ]
    return review


def _html_json_block(payload: object) -> str:
    return html.escape(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))


def _html_scalar(value: object) -> str:
    if isinstance(value, float):
        return str(round(value, 4))
    return html.escape(str(value))


def _render_metric_table(metrics: dict[str, object]) -> str:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            rendered_value = f"<pre>{_html_json_block(value)}</pre>"
        else:
            rendered_value = _html_scalar(value)
        rows.append(
            "<tr>"
            f"<th scope='row'>{html.escape(str(key))}</th>"
            f"<td>{rendered_value}</td>"
            "</tr>"
        )
    return "<table class='metrics-table'><tbody>" + "".join(rows) + "</tbody></table>"


def _render_record_details(items: list[dict[str, object]], *, empty_message: str) -> str:
    if not items:
        return f"<p class='muted'>{html.escape(empty_message)}</p>"
    blocks = []
    for item in items:
        title = html.escape(str(item.get("case_id", "(unknown case)")))
        blocks.append(
            "<details class='record'>"
            f"<summary>{title}</summary>"
            f"<pre>{_html_json_block(item)}</pre>"
            "</details>"
        )
    return "".join(blocks)


def _render_limited_record_details(
    items: list[dict[str, object]],
    *,
    empty_message: str,
    limit: int,
    omitted_message: str,
) -> str:
    if len(items) > limit:
        return f"<p class='muted'>{html.escape(omitted_message)}</p>"
    return _render_record_details(items, empty_message=empty_message)


def _render_priority_groups(priority_cases: dict[str, object]) -> str:
    groups = []
    for group_name, raw_items in priority_cases.items():
        items = raw_items if isinstance(raw_items, list) else []
        groups.append(
            "<details class='section'>"
            f"<summary>{html.escape(group_name)} <span class='count'>{len(items)}</span></summary>"
            f"{_render_record_details(items, empty_message='No flagged cases in this group.')}"
            "</details>"
        )
    return "".join(groups)


def _render_warning_case_groups(warning_cases: dict[str, object]) -> str:
    groups = []
    for source_name, raw_by_code in warning_cases.items():
        by_code = raw_by_code if isinstance(raw_by_code, dict) else {}
        inner = []
        for warning_code, raw_items in by_code.items():
            items = raw_items if isinstance(raw_items, list) else []
            visible_items = items[:HTML_WARNING_CASE_DETAIL_LIMIT]
            omitted = len(items) - len(visible_items)
            omitted_note = (
                f"<p class='muted'>... omitted {omitted} additional case(s) in HTML; "
                "see review.json.gz for the complete list.</p>"
                if omitted > 0
                else ""
            )
            inner.append(
                "<details class='record'>"
                f"<summary>{html.escape(str(warning_code))} <span class='count'>{len(items)} total</span></summary>"
                f"{_render_record_details(visible_items, empty_message='No cases for this warning.')}"
                f"{omitted_note}"
                "</details>"
            )
        groups.append(
            "<details class='section'>"
            f"<summary>{html.escape(str(source_name))} <span class='count'>{len(by_code)}</span></summary>"
            f"{''.join(inner) if inner else '<p class=\"muted\">No warnings in this source.</p>'}"
            "</details>"
        )
    return "".join(groups)


def build_batch_html_report(
    *,
    summary: dict[str, object],
    review: dict[str, object],
    manifest: dict[str, object] | None = None,
    title: str = "cif-parse Batch Summary Report",
    artifact_paths: dict[str, str] | None = None,
) -> str:
    """Build a self-contained HTML summary report with expandable sections."""

    manifest = manifest or {}
    artifact_paths = artifact_paths or {}
    status_summary = review.get("status_summary", {})
    priority_cases = review.get("priority_cases", {})
    warning_counts = review.get("warning_counts", {})
    warning_cases = review.get("warning_cases", {})
    skipped_targets = review.get("skipped_targets", [])
    failed_targets = review.get("failed_targets", [])
    manifest_metadata = {
        "input_count": len(manifest.get("input_paths", [])) if isinstance(manifest.get("input_paths"), list) else 0,
        "result_count": len(manifest.get("results", [])) if isinstance(manifest.get("results"), list) else 0,
    }
    if isinstance(manifest.get("settings"), dict):
        manifest_metadata["settings"] = manifest["settings"]

    artifact_table = _render_metric_table(artifact_paths) if artifact_paths else "<p class='muted'>No artifact paths.</p>"
    summary_table = _render_metric_table(summary)
    status_table = _render_metric_table(status_summary if isinstance(status_summary, dict) else {})
    manifest_table = _render_metric_table(manifest_metadata)
    skipped_reason_counts = review.get("skipped_target_counts_by_reason", {})
    warning_count_table = _render_metric_table(
        warning_counts if isinstance(warning_counts, dict) else {}
    )
    skipped_reason_table = _render_metric_table(
        skipped_reason_counts if isinstance(skipped_reason_counts, dict) else {}
    )
    skipped_target_items = skipped_targets if isinstance(skipped_targets, list) else []
    skipped_target_details = _render_limited_record_details(
        skipped_target_items,
        empty_message="No skipped targets.",
        limit=HTML_SKIPPED_TARGET_DETAIL_LIMIT,
        omitted_message=(
            f"Skipped target details omitted because {len(skipped_target_items)} targets exceed "
            f"the HTML detail limit of {HTML_SKIPPED_TARGET_DETAIL_LIMIT}. "
            "Use review.json.gz for the full skipped target list."
        ),
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f4ef;
      --panel: #fffdf8;
      --line: #d8d2c3;
      --ink: #1e2430;
      --muted: #667085;
      --accent: #1f6f78;
      --accent-soft: #e4f0ef;
      --code: #f2eee4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f6f4ef 0%, #ece7dc 100%);
      color: var(--ink);
      font: 14px/1.5 "DejaVu Sans", "Noto Sans", sans-serif;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      line-height: 1.2;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(31, 42, 68, 0.08);
      margin-bottom: 20px;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    .summary-stack {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(31, 42, 68, 0.06);
    }}
    details.section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      margin-bottom: 14px;
      box-shadow: 0 8px 24px rgba(31, 42, 68, 0.05);
    }}
    details.record {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }}
    details > summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--accent);
    }}
    .count {{
      display: inline-block;
      margin-left: 6px;
      padding: 1px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    .metrics-table tr {{
      display: block;
      border-top: 1px solid var(--line);
      padding: 9px 0;
    }}
    .metrics-table th,
    .metrics-table td {{
      display: block;
      text-align: left;
      vertical-align: top;
      padding: 0;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .metrics-table th {{
      width: auto;
      color: var(--muted);
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .metrics-table td {{
      color: var(--ink);
    }}
    pre {{
      margin: 10px 0 0;
      padding: 12px;
      overflow-x: hidden;
      border-radius: 12px;
      background: var(--code);
      border: 1px solid var(--line);
      font: 12px/1.5 "DejaVu Sans Mono", "Noto Sans Mono", monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>Batch-level human-readable summary. Expand sections below for skipped targets, failures, and priority review groups.</p>
    </section>
    <section class="summary-stack">
      <div class="card">
        <h2>Artifacts</h2>
        {artifact_table}
      </div>
      <div class="card">
        <h2>Status</h2>
        {status_table}
      </div>
      <div class="card">
        <h2>Batch Metadata</h2>
        {manifest_table}
      </div>
      <div class="card">
        <h2>Summary</h2>
        {summary_table}
      </div>
    </section>
    <details class="section" open>
      <summary>Skipped Target Counts By Reason <span class="count">{len(skipped_reason_counts) if isinstance(skipped_reason_counts, dict) else 0}</span></summary>
      {skipped_reason_table}
    </details>
    <details class="section" open>
      <summary>Warning Counts <span class="count">{len(warning_counts) if isinstance(warning_counts, dict) else 0}</span></summary>
      {warning_count_table}
    </details>
    <details class="section">
      <summary>Warning Cases <span class="count">{len(warning_cases) if isinstance(warning_cases, dict) else 0}</span></summary>
      {_render_warning_case_groups(warning_cases if isinstance(warning_cases, dict) else {})}
    </details>
    <details class="section">
      <summary>Skipped Targets <span class="count">{len(skipped_targets) if isinstance(skipped_targets, list) else 0}</span></summary>
      {skipped_target_details}
    </details>
    <details class="section">
      <summary>Failed Targets <span class="count">{len(failed_targets) if isinstance(failed_targets, list) else 0}</span></summary>
      {_render_record_details(failed_targets if isinstance(failed_targets, list) else [], empty_message="No failed targets.")}
    </details>
    <details class="section" open>
      <summary>Priority Review Groups <span class="count">{len(priority_cases) if isinstance(priority_cases, dict) else 0}</span></summary>
      {_render_priority_groups(priority_cases if isinstance(priority_cases, dict) else {})}
    </details>
  </main>
</body>
</html>
"""
