from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from cif_parse.cli import main as cli_main


PROTEIN_CHAIN_TYPES = frozenset(
    {
        "antibody heavy chain",
        "antibody light chain",
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
        "other protein chain",
    }
)
ANTIBODY_CHAIN_TYPES = frozenset({"antibody heavy chain", "antibody light chain"})
NUCLEIC_ACID_CHAIN_TYPES = frozenset({"DNA chain", "RNA chain", "other nucleic acid chain"})
POLYMER_CHAIN_TYPES = PROTEIN_CHAIN_TYPES | NUCLEIC_ACID_CHAIN_TYPES | frozenset({"other polymer chain"})
LOW_CONFIDENCE_ANTIBODY_THRESHOLD = 0.8
COVERAGE_WARNING_CODES = frozenset(
    {
        "coverage skipped because the chain has no coordinates",
        "coverage assigned to multiple main chains",
        "coverage owner not found within the nearest-distance threshold",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random mmCIF smoke tests")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../protenix_data/mmCIF"),
        help="Directory containing mmCIF files",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("test_outputs/random50"),
        help="Directory for smoke test outputs",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of random mmCIF files to sample",
    )
    parser.add_argument(
        "--id-list",
        type=Path,
        default=None,
        help="Optional text file containing one PDB id per line; overrides random sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260416,
        help="Random seed for reproducible sampling",
    )
    return parser


def _infer_case_id(path: Path) -> str:
    for suffix in (".cif.gz", ".bcif.gz", ".cif", ".bcif"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)].lower()
    return path.stem.lower()


def _resolve_input_path(input_dir: Path, pdb_id: str) -> Path | None:
    pdb_id = pdb_id.lower()
    shard = pdb_id[1:3]
    for suffix in (".cif.gz", ".cif", ".bcif.gz", ".bcif"):
        candidate = input_dir / shard / f"{pdb_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


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


def _vhh_without_light_partner(chain: dict[str, object]) -> bool:
    if _antibody_unit_type(chain) != "VHH":
        return False
    analysis = _antibody_analysis(chain)
    evidence = analysis.get("vhh_evidence")
    if isinstance(evidence, dict):
        paired_light_found = evidence.get("paired_light_found")
        if isinstance(paired_light_found, bool):
            return not paired_light_found
    return not bool(chain.get("paired_label_asym_id"))


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


def _case_metrics(case_outdir: Path) -> dict[str, object]:
    summary = _read_json(case_outdir / "final" / "structure_summary.json")
    chain_inventory = _read_json(case_outdir / "final" / "chain_inventory.json")
    dimers = _read_json(case_outdir / "final" / "dimer_interfaces.json")
    multimers = _read_json(case_outdir / "final" / "tight_multimers.json")
    antibody_antigen_complexes = _read_json(case_outdir / "final" / "antibody_antigen_complexes.json")
    tcr_pmhc_complexes = _read_json(case_outdir / "final" / "tcr_pmhc_complexes.json")

    num_dimers = len(dimers)
    num_multimers = len(multimers)
    total_residue_contacts = sum(int(dimer["num_residue_contacts"]) for dimer in dimers)
    total_atom_contacts = sum(int(dimer["num_atom_contacts"]) for dimer in dimers)
    total_multimer_internal_edges = sum(int(multimer["num_internal_edges"]) for multimer in multimers)
    total_multimer_members = sum(int(multimer["num_members"]) for multimer in multimers)
    largest_multimer_size = max((int(multimer["num_members"]) for multimer in multimers), default=0)
    chain_type_counts = dict(summary.get("chain_type_counts", {}))
    multimer_size_distribution = Counter(
        int(multimer["num_members"]) for multimer in multimers
    )
    chain_map = {
        str(chain.get("label_asym_id")): chain
        for chain in chain_inventory
        if chain.get("label_asym_id") is not None
    }
    antibody_chains = [
        chain
        for chain in chain_inventory
        if chain.get("chain_type") in ANTIBODY_CHAIN_TYPES
    ]
    antibody_unit_type_counts = Counter(
        unit_type for unit_type in (_antibody_unit_type(chain) for chain in antibody_chains) if unit_type
    )
    pairable_antibody_heavy_chains = [
        chain
        for chain in antibody_chains
        if chain.get("chain_type") == "antibody heavy chain"
        and _antibody_unit_type(chain) not in {"VHH", "scFv"}
    ]
    heavy_light_pairs: set[tuple[str, str]] = set()
    for chain in antibody_chains:
        if chain.get("chain_type") != "antibody heavy chain":
            continue
        partner_label = chain.get("paired_label_asym_id")
        if not isinstance(partner_label, str) or not partner_label:
            continue
        partner = chain_map.get(partner_label)
        if partner is None or partner.get("chain_type") != "antibody light chain":
            continue
        heavy_light_pairs.add((str(chain["label_asym_id"]), partner_label))

    def _antibody_confidence(chain: dict[str, object]) -> float:
        analysis = _antibody_analysis(chain)
        confidence = analysis.get("annotation_confidence")
        if isinstance(confidence, (int, float)):
            return float(confidence)
        raw_confidence = chain.get("annotation_confidence")
        if isinstance(raw_confidence, (int, float)):
            return float(raw_confidence)
        return 0.0

    num_antibody_related_dimers = sum(
        1 for dimer in dimers if bool(dimer.get("contains_antibody_unit"))
    )
    num_heavy_light_dimers = sum(
        1
        for dimer in dimers
        if {
            dimer.get("chain_type_1"),
            dimer.get("chain_type_2"),
        }
        == {"antibody heavy chain", "antibody light chain"}
    )
    num_tcr_chains = sum(1 for chain in chain_inventory if chain.get("chain_type") == "TCR chain")
    num_mhc_heavy_chains = sum(
        1 for chain in chain_inventory if chain.get("chain_type") == "MHC heavy chain"
    )
    num_auxiliary_immune_chains = sum(
        1
        for chain in chain_inventory
        if chain.get("chain_type") == "beta2m or auxiliary immune chain"
    )
    num_explicit_peptide_chains = sum(
        1 for chain in chain_inventory if chain.get("chain_type") == "peptide antigen"
    )
    num_unpaired_antibody_heavy_chains = sum(
        1
        for chain in pairable_antibody_heavy_chains
        if not isinstance(chain.get("paired_label_asym_id"), str) or not chain.get("paired_label_asym_id")
    )
    num_unpaired_antibody_light_chains = sum(
        1
        for chain in antibody_chains
        if chain.get("chain_type") == "antibody light chain"
        and (not isinstance(chain.get("paired_label_asym_id"), str) or not chain.get("paired_label_asym_id"))
    )
    num_true_heavy_only_chains = sum(1 for chain in antibody_chains if _true_heavy_only(chain))
    num_vhh_without_light_partner = sum(1 for chain in antibody_chains if _vhh_without_light_partner(chain))
    contextual_peptide_chain_ids: set[str] = set()
    chain_warning_counts = _warning_counts(chain_inventory)
    antibody_complex_warning_counts = _warning_counts(antibody_antigen_complexes)
    tcr_complex_warning_counts: Counter[str] = Counter(_warning_counts(tcr_pmhc_complexes))
    multimer_warning_counts = _warning_counts(multimers)
    for complex_record in tcr_pmhc_complexes:
        evidence = _evidence_dict(complex_record)
        contextual_peptide_chain_ids.update(_string_list(evidence.get("contextual_peptide_chain_ids")))

    return {
        "atom_count": int(summary["atom_count"]),
        "entity_count": int(summary["entity_count"]),
        "num_chains": len(summary.get("chain_ids", [])),
        "num_polymer_chains": sum(
            1 for chain in chain_inventory if chain.get("chain_type") in POLYMER_CHAIN_TYPES
        ),
        "num_protein_chains": sum(
            1 for chain in chain_inventory if chain.get("chain_type") in PROTEIN_CHAIN_TYPES
        ),
        "num_nucleic_acid_chains": sum(
            1 for chain in chain_inventory if chain.get("chain_type") in NUCLEIC_ACID_CHAIN_TYPES
        ),
        "num_chains_with_unresolved_segments": sum(
            1 for chain in chain_inventory if chain.get("unresolved_sequence_segments")
        ),
        "num_chains_with_special_residues": sum(
            1 for chain in chain_inventory if chain.get("special_residue_details")
        ),
        "num_chains_with_bound_partners": sum(
            1 for chain in chain_inventory if chain.get("bound_chain_ids")
        ),
        "num_antibody_chains": len(antibody_chains),
        "num_antibody_heavy_chains": sum(
            1 for chain in antibody_chains if chain.get("chain_type") == "antibody heavy chain"
        ),
        "num_antibody_light_chains": sum(
            1 for chain in antibody_chains if chain.get("chain_type") == "antibody light chain"
        ),
        "num_pairable_antibody_heavy_chains": len(pairable_antibody_heavy_chains),
        "num_pairable_antibody_light_chains": sum(
            1 for chain in antibody_chains if chain.get("chain_type") == "antibody light chain"
        ),
        "num_antibody_chains_with_analysis": sum(
            1 for chain in antibody_chains if _antibody_analysis(chain)
        ),
        "num_vhh_chains": sum(
            1 for chain in antibody_chains if _antibody_unit_type(chain) == "VHH"
        ),
        "num_scfv_chains": sum(
            1 for chain in antibody_chains if _antibody_unit_type(chain) == "scFv"
        ),
        "num_antibody_pairs": len(heavy_light_pairs),
        "num_paired_antibody_heavy_chains": len({heavy for heavy, _ in heavy_light_pairs}),
        "num_paired_antibody_light_chains": len({light for _, light in heavy_light_pairs}),
        "num_unpaired_antibody_heavy_chains": num_unpaired_antibody_heavy_chains,
        "num_unpaired_antibody_light_chains": num_unpaired_antibody_light_chains,
        "num_true_heavy_only_chains": num_true_heavy_only_chains,
        "num_vhh_without_light_partner": num_vhh_without_light_partner,
        "num_heavy_only_without_light_partner": num_true_heavy_only_chains,
        "num_antibody_annotation_conflicts": sum(
            1
            for chain in antibody_chains
            if "description_and_sequence_antibody_annotation_disagree"
            in (chain.get("warnings") or [])
        ),
        "num_low_confidence_antibody_chains": sum(
            1
            for chain in antibody_chains
            if _antibody_confidence(chain) < LOW_CONFIDENCE_ANTIBODY_THRESHOLD
        ),
        "num_antibody_related_dimers": num_antibody_related_dimers,
        "num_heavy_light_dimers": num_heavy_light_dimers,
        "num_antibody_antigen_complexes": len(antibody_antigen_complexes),
        "num_tcr_pmhc_complexes": len(tcr_pmhc_complexes),
        "num_tcr_chains": num_tcr_chains,
        "num_mhc_heavy_chains": num_mhc_heavy_chains,
        "num_auxiliary_immune_chains": num_auxiliary_immune_chains,
        "num_explicit_peptide_chains": num_explicit_peptide_chains,
        "num_contextual_peptide_chains_in_complexes": len(contextual_peptide_chain_ids),
        "chain_warning_counts": chain_warning_counts,
        "antibody_complex_warning_counts": antibody_complex_warning_counts,
        "tcr_complex_warning_counts": dict(sorted(tcr_complex_warning_counts.items())),
        "multimer_warning_counts": multimer_warning_counts,
        "antibody_unit_type_counts": dict(sorted(antibody_unit_type_counts.items())),
        "num_dimers": num_dimers,
        "num_multimers": num_multimers,
        "total_dimer_residue_contacts": total_residue_contacts,
        "total_dimer_atom_contacts": total_atom_contacts,
        "total_multimer_internal_edges": total_multimer_internal_edges,
        "total_multimer_members": total_multimer_members,
        "largest_multimer_size": largest_multimer_size,
        "multimer_size_distribution": dict(sorted(multimer_size_distribution.items())),
        "max_dimer_atom_contacts": max(
            (int(dimer["num_atom_contacts"]) for dimer in dimers),
            default=0,
        ),
        "min_dimer_distance": round(
            min((float(dimer["min_distance"]) for dimer in dimers), default=float("inf")),
            4,
        )
        if num_dimers
        else None,
        "chain_type_counts": chain_type_counts,
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
        ranked.append({"case_id": str(result["case_id"]), "warning_count": warning_count})
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


def _count_by_key(items: list[dict[str, object]], *, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _build_review(results: list[dict[str, object]]) -> dict[str, object]:
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


def _build_aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    successful_results = [result for result in results if result.get("status") == "ok"]
    failed_results = [result for result in results if result.get("status") != "ok"]
    case_metrics = [result["metrics"] for result in successful_results if "metrics" in result]

    aggregate_chain_type_counts: Counter[str] = Counter()
    for metrics in case_metrics:
        aggregate_chain_type_counts.update(metrics.get("chain_type_counts", {}))

    total_cases = len(case_metrics)
    total_dimers = sum(int(metrics["num_dimers"]) for metrics in case_metrics)
    total_multimers = sum(int(metrics["num_multimers"]) for metrics in case_metrics)
    total_atoms = sum(int(metrics["atom_count"]) for metrics in case_metrics)
    total_chains = sum(int(metrics["num_chains"]) for metrics in case_metrics)
    total_polymer_chains = sum(int(metrics["num_polymer_chains"]) for metrics in case_metrics)
    total_antibody_chains = sum(int(metrics["num_antibody_chains"]) for metrics in case_metrics)
    total_antibody_heavy_chains = sum(
        int(metrics["num_antibody_heavy_chains"]) for metrics in case_metrics
    )
    total_antibody_light_chains = sum(
        int(metrics["num_antibody_light_chains"]) for metrics in case_metrics
    )
    total_pairable_antibody_heavy_chains = sum(
        int(metrics["num_pairable_antibody_heavy_chains"]) for metrics in case_metrics
    )
    total_pairable_antibody_light_chains = sum(
        int(metrics["num_pairable_antibody_light_chains"]) for metrics in case_metrics
    )
    total_antibody_pairs = sum(int(metrics["num_antibody_pairs"]) for metrics in case_metrics)
    total_paired_antibody_heavy_chains = sum(
        int(metrics["num_paired_antibody_heavy_chains"]) for metrics in case_metrics
    )
    total_paired_antibody_light_chains = sum(
        int(metrics["num_paired_antibody_light_chains"]) for metrics in case_metrics
    )
    total_vhh_chains = sum(int(metrics["num_vhh_chains"]) for metrics in case_metrics)
    total_scfv_chains = sum(int(metrics["num_scfv_chains"]) for metrics in case_metrics)
    total_antibody_related_dimers = sum(
        int(metrics["num_antibody_related_dimers"]) for metrics in case_metrics
    )
    total_heavy_light_dimers = sum(int(metrics["num_heavy_light_dimers"]) for metrics in case_metrics)
    total_antibody_antigen_complexes = sum(
        int(metrics["num_antibody_antigen_complexes"]) for metrics in case_metrics
    )
    total_tcr_pmhc_complexes = sum(
        int(metrics["num_tcr_pmhc_complexes"]) for metrics in case_metrics
    )
    total_antibody_annotation_conflicts = sum(
        int(metrics["num_antibody_annotation_conflicts"]) for metrics in case_metrics
    )
    total_true_heavy_only_chains = sum(
        int(metrics["num_true_heavy_only_chains"]) for metrics in case_metrics
    )
    total_vhh_without_light_partner = sum(
        int(metrics["num_vhh_without_light_partner"]) for metrics in case_metrics
    )
    total_low_confidence_antibody_chains = sum(
        int(metrics["num_low_confidence_antibody_chains"]) for metrics in case_metrics
    )
    total_residue_contacts = sum(
        int(metrics["total_dimer_residue_contacts"]) for metrics in case_metrics
    )
    total_atom_contacts = sum(int(metrics["total_dimer_atom_contacts"]) for metrics in case_metrics)
    total_multimer_internal_edges = sum(
        int(metrics["total_multimer_internal_edges"]) for metrics in case_metrics
    )
    total_multimer_members = sum(int(metrics["total_multimer_members"]) for metrics in case_metrics)
    multimer_size_distribution: Counter[int] = Counter()
    aggregate_antibody_unit_type_counts: Counter[str] = Counter()
    for metrics in case_metrics:
        multimer_size_distribution.update(
            {
                int(size): int(count)
                for size, count in metrics.get("multimer_size_distribution", {}).items()
            }
        )
        aggregate_antibody_unit_type_counts.update(metrics.get("antibody_unit_type_counts", {}))
    max_dimer_case = max(
        successful_results,
        key=lambda result: int(result.get("metrics", {}).get("num_dimers", 0)),
        default=None,
    )
    max_multimer_case = max(
        successful_results,
        key=lambda result: int(result.get("metrics", {}).get("largest_multimer_size", 0)),
        default=None,
    )

    return {
        "success_count": len(successful_results),
        "failure_count": len(failed_results),
        "cases_with_dimers": sum(1 for metrics in case_metrics if int(metrics["num_dimers"]) > 0),
        "cases_with_multimers": sum(1 for metrics in case_metrics if int(metrics["num_multimers"]) > 0),
        "cases_with_unresolved_segments": sum(
            1 for metrics in case_metrics if int(metrics["num_chains_with_unresolved_segments"]) > 0
        ),
        "cases_with_special_residues": sum(
            1 for metrics in case_metrics if int(metrics["num_chains_with_special_residues"]) > 0
        ),
        "cases_with_antibody_chains": sum(
            1 for metrics in case_metrics if int(metrics["num_antibody_chains"]) > 0
        ),
        "cases_with_antibody_pairs": sum(
            1 for metrics in case_metrics if int(metrics["num_antibody_pairs"]) > 0
        ),
        "cases_with_antibody_antigen_complexes": sum(
            1 for metrics in case_metrics if int(metrics["num_antibody_antigen_complexes"]) > 0
        ),
        "cases_with_tcr_pmhc_complexes": sum(
            1 for metrics in case_metrics if int(metrics["num_tcr_pmhc_complexes"]) > 0
        ),
        "cases_with_vhh": sum(1 for metrics in case_metrics if int(metrics["num_vhh_chains"]) > 0),
        "cases_with_scfv": sum(1 for metrics in case_metrics if int(metrics["num_scfv_chains"]) > 0),
        "cases_with_antibody_annotation_conflicts": sum(
            1 for metrics in case_metrics if int(metrics["num_antibody_annotation_conflicts"]) > 0
        ),
        "cases_with_true_heavy_only_chains": sum(
            1 for metrics in case_metrics if int(metrics["num_true_heavy_only_chains"]) > 0
        ),
        "cases_with_low_confidence_antibody_chains": sum(
            1 for metrics in case_metrics if int(metrics["num_low_confidence_antibody_chains"]) > 0
        ),
        "total_atoms": total_atoms,
        "total_chains": total_chains,
        "total_polymer_chains": total_polymer_chains,
        "total_antibody_chains": total_antibody_chains,
        "total_antibody_heavy_chains": total_antibody_heavy_chains,
        "total_antibody_light_chains": total_antibody_light_chains,
        "total_pairable_antibody_heavy_chains": total_pairable_antibody_heavy_chains,
        "total_pairable_antibody_light_chains": total_pairable_antibody_light_chains,
        "total_antibody_pairs": total_antibody_pairs,
        "total_antibody_antigen_complexes": total_antibody_antigen_complexes,
        "total_tcr_pmhc_complexes": total_tcr_pmhc_complexes,
        "total_paired_antibody_heavy_chains": total_paired_antibody_heavy_chains,
        "total_paired_antibody_light_chains": total_paired_antibody_light_chains,
        "total_vhh_chains": total_vhh_chains,
        "total_scfv_chains": total_scfv_chains,
        "total_antibody_related_dimers": total_antibody_related_dimers,
        "total_heavy_light_dimers": total_heavy_light_dimers,
        "total_antibody_annotation_conflicts": total_antibody_annotation_conflicts,
        "total_true_heavy_only_chains": total_true_heavy_only_chains,
        "total_vhh_without_light_partner": total_vhh_without_light_partner,
        "total_heavy_only_without_light_partner": total_true_heavy_only_chains,
        "total_low_confidence_antibody_chains": total_low_confidence_antibody_chains,
        "total_dimers": total_dimers,
        "total_multimers": total_multimers,
        "total_dimer_residue_contacts": total_residue_contacts,
        "total_dimer_atom_contacts": total_atom_contacts,
        "total_multimer_internal_edges": total_multimer_internal_edges,
        "total_multimer_members": total_multimer_members,
        "average_atoms_per_case": round(total_atoms / total_cases, 2) if total_cases else 0.0,
        "average_chains_per_case": round(total_chains / total_cases, 2) if total_cases else 0.0,
        "average_polymer_chains_per_case": round(
            total_polymer_chains / total_cases, 2
        )
        if total_cases
        else 0.0,
        "average_antibody_chains_per_case": round(
            total_antibody_chains / total_cases, 2
        )
        if total_cases
        else 0.0,
        "average_antibody_pairs_per_case": round(
            total_antibody_pairs / total_cases, 2
        )
        if total_cases
        else 0.0,
        "average_antibody_antigen_complexes_per_case": round(
            total_antibody_antigen_complexes / total_cases, 2
        )
        if total_cases
        else 0.0,
        "average_tcr_pmhc_complexes_per_case": round(
            total_tcr_pmhc_complexes / total_cases, 2
        )
        if total_cases
        else 0.0,
        "average_dimers_per_case": round(total_dimers / total_cases, 2) if total_cases else 0.0,
        "average_multimers_per_case": round(total_multimers / total_cases, 2)
        if total_cases
        else 0.0,
        "paired_heavy_chain_ratio": round(
            total_paired_antibody_heavy_chains / total_antibody_heavy_chains, 4
        )
        if total_antibody_heavy_chains
        else 0.0,
        "paired_light_chain_ratio": round(
            total_paired_antibody_light_chains / total_antibody_light_chains, 4
        )
        if total_antibody_light_chains
        else 0.0,
        "pairable_heavy_chain_ratio": round(
            total_paired_antibody_heavy_chains / total_pairable_antibody_heavy_chains, 4
        )
        if total_pairable_antibody_heavy_chains
        else 0.0,
        "pairable_light_chain_ratio": round(
            total_paired_antibody_light_chains / total_pairable_antibody_light_chains, 4
        )
        if total_pairable_antibody_light_chains
        else 0.0,
        "average_dimer_residue_contacts_per_case": round(
            total_residue_contacts / total_cases, 2
        )
        if total_cases
        else 0.0,
        "average_dimer_atom_contacts_per_case": round(
            total_atom_contacts / total_cases, 2
        )
        if total_cases
        else 0.0,
        "average_atom_contacts_per_dimer": round(total_atom_contacts / total_dimers, 2)
        if total_dimers
        else 0.0,
        "average_internal_edges_per_multimer": round(
            total_multimer_internal_edges / total_multimers, 2
        )
        if total_multimers
        else 0.0,
        "average_members_per_multimer": round(
            total_multimer_members / total_multimers, 2
        )
        if total_multimers
        else 0.0,
        "max_dimers_in_case": int(max_dimer_case["metrics"]["num_dimers"])
        if max_dimer_case and "metrics" in max_dimer_case
        else 0,
        "max_dimers_case_id": max_dimer_case["case_id"] if max_dimer_case else None,
        "largest_multimer_size": int(max_multimer_case["metrics"]["largest_multimer_size"])
        if max_multimer_case and "metrics" in max_multimer_case
        else 0,
        "largest_multimer_case_id": max_multimer_case["case_id"] if max_multimer_case else None,
        "multimer_size_distribution": dict(
            sorted(multimer_size_distribution.items(), key=lambda item: item[0])
        ),
        "aggregate_antibody_unit_type_counts": dict(
            sorted(aggregate_antibody_unit_type_counts.items())
        ),
        "aggregate_chain_type_counts": dict(sorted(aggregate_chain_type_counts.items())),
    }


def main() -> int:
    args = build_parser().parse_args()
    list_ids: list[str] | None = None
    if args.id_list is not None:
        list_ids = [line.strip().lower() for line in args.id_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        sampled_files = []
        missing_ids: list[str] = []
        for pdb_id in list_ids:
            resolved = _resolve_input_path(args.input_dir, pdb_id)
            if resolved is None:
                missing_ids.append(pdb_id)
            else:
                sampled_files.append(resolved)
        if not sampled_files:
            raise ValueError("no valid mmCIF files found for the provided id list")
    else:
        all_files = sorted(path for path in args.input_dir.rglob("*") if path.is_file())
        if len(all_files) < args.count:
            raise ValueError(f"requested {args.count} files, but only found {len(all_files)}")

        rng = random.Random(args.seed)
        sampled_files = rng.sample(all_files, args.count)

    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "count": len(sampled_files),
        "input_dir": str(args.input_dir.resolve()),
        "id_list": str(args.id_list.resolve()) if args.id_list is not None else None,
        "requested_ids": list_ids or [],
        "missing_ids": missing_ids if args.id_list is not None else [],
        "sampled_files": [str(path.resolve()) for path in sampled_files],
        "results": [],
    }

    for input_path in sampled_files:
        case_id = _infer_case_id(input_path)
        case_outdir = args.outdir / "cases" / case_id
        try:
            exit_code = cli_main(
                [
                    "single",
                    str(input_path),
                    "--outdir",
                    str(case_outdir),
                    "--format",
                    "json",
                ]
            )
            result = {
                "case_id": case_id,
                "input_path": str(input_path.resolve()),
                "output_dir": str(case_outdir.resolve()),
                "status": "ok" if exit_code == 0 else "nonzero_exit",
                "exit_code": exit_code,
            }
            if exit_code == 0:
                result["metrics"] = _case_metrics(case_outdir)
            manifest["results"].append(
                result
            )
        except Exception as exc:  # noqa: BLE001
            manifest["results"].append(
                {
                    "case_id": case_id,
                    "input_path": str(input_path.resolve()),
                    "output_dir": str(case_outdir.resolve()),
                    "status": "error",
                    "error": repr(exc),
                }
            )

    aggregate = _build_aggregate(manifest["results"])
    review = _build_review(manifest["results"])
    manifest.update(aggregate)
    manifest["aggregate"] = aggregate
    manifest["review"] = review

    manifest_path = args.outdir / "manifest.json"
    summary_path = args.outdir / "summary.json"
    review_path = args.outdir / "review.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
