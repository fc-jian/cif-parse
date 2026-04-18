"""Shared reporting helpers for batch and smoke-test outputs."""

from __future__ import annotations

import json
from pathlib import Path


ANTIBODY_CHAIN_TYPES = frozenset({"antibody heavy chain", "antibody light chain"})
LOW_CONFIDENCE_ANTIBODY_THRESHOLD = 0.8
COVERAGE_WARNING_CODES = frozenset(
    {
        "coverage skipped because the chain has no coordinates",
        "coverage assigned to multiple main chains",
        "coverage owner not found within the nearest-distance threshold",
    }
)


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


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


def collect_case_review_metrics(case_outdir: str | Path) -> dict[str, object]:
    """Collect the per-case metrics used by `review.json`."""

    case_outdir = Path(case_outdir)
    chain_inventory = _read_json(case_outdir / "final" / "chain_inventory.json")
    tight_multimers = _read_json(case_outdir / "final" / "tight_multimers.json")
    antibody_antigen_complexes = _read_json(case_outdir / "final" / "antibody_antigen_complexes.json")
    tcr_pmhc_complexes = _read_json(case_outdir / "final" / "tcr_pmhc_complexes.json")

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
        raw_confidence = chain.get("annotation_confidence")
        if isinstance(raw_confidence, (int, float)):
            return float(raw_confidence)
        return 0.0

    contextual_peptide_chain_ids: set[str] = set()
    chain_warning_counts = _warning_counts(chain_inventory)
    antibody_complex_warning_counts = _warning_counts(antibody_antigen_complexes)
    tcr_complex_warning_counts = _warning_counts(tcr_pmhc_complexes)
    multimer_warning_counts = _warning_counts(tight_multimers)
    for complex_record in tcr_pmhc_complexes:
        evidence = _evidence_dict(complex_record)
        contextual_peptide_chain_ids.update(_string_list(evidence.get("contextual_peptide_chain_ids")))

    return {
        "num_low_confidence_antibody_chains": sum(
            1 for chain in antibody_chains if _antibody_confidence(chain) < LOW_CONFIDENCE_ANTIBODY_THRESHOLD
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
        "chain_warning_counts": chain_warning_counts,
        "antibody_complex_warning_counts": antibody_complex_warning_counts,
        "tcr_complex_warning_counts": dict(sorted(tcr_complex_warning_counts.items())),
        "multimer_warning_counts": multimer_warning_counts,
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
    review = {
        "status_summary": {
            "success_count": len(successful_results),
            "skipped_count": len(skipped_targets),
            "failure_count": len(failed_targets),
        },
        "skipped_targets": skipped_targets,
        "skipped_target_counts_by_reason": _count_by_key(skipped_targets, key="warning_code"),
        "failed_targets": failed_targets,
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
            "large_multimer_without_bridge_pruning_warning": _top_warning_cases(
                successful_results,
                "multimer_warning_counts",
                "large_component_without_bridge_pruning",
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
