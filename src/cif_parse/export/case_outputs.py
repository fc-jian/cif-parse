from __future__ import annotations

from pathlib import Path
from typing import Any

from .writers import dump_json, load_json, resolve_json_path


JSON_CASE_ARTIFACT_NAMES = (
    "structure_summary",
    "chain_inventory",
    "dimer_interfaces",
    "tight_multimers",
    "antibody_antigen_complexes",
    "tcr_pmhc_complexes",
    "protein_chains",
    "nucleic_acid_chains",
    "branched_entities",
    "metal_ions",
    "small_molecule_compounds",
    "other_entities",
)
JSON_CASE_BUNDLE_NAME = "result.json.gz"


def build_single_json_bundle(
    *,
    summary: Any,
    chain_inventory: list[Any],
    partitions: dict[str, list[Any]],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
    antibody_antigen_complexes: list[Any],
    tcr_pmhc_complexes: list[Any],
) -> dict[str, Any]:
    """Build the compact single-file JSON payload for one processed structure."""

    return {
        "structure_summary": summary.to_dict(),
        "chain_inventory": [chain.to_dict() for chain in chain_inventory],
        "dimer_interfaces": [dimer.to_dict() for dimer in dimer_interfaces],
        "tight_multimers": [multimer.to_dict() for multimer in tight_multimers],
        "antibody_antigen_complexes": [
            complex_record.to_dict() for complex_record in antibody_antigen_complexes
        ],
        "tcr_pmhc_complexes": [complex_record.to_dict() for complex_record in tcr_pmhc_complexes],
        **{
            output_name: [chain.to_dict() for chain in chains]
            for output_name, chains in partitions.items()
        },
    }


def dump_single_json_bundle(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the compact single-file JSON bundle for one structure."""

    return dump_json(path, payload, indent=None)


def load_case_output_bundle(case_outdir: str | Path) -> dict[str, Any]:
    """Load a case output bundle, supporting both bundled and split JSON layouts."""

    case_path = Path(case_outdir)
    bundle_path = case_path / JSON_CASE_BUNDLE_NAME
    if bundle_path.exists():
        payload = load_json(bundle_path)
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"Expected dict payload in {bundle_path}")

    payload: dict[str, Any] = {}
    for artifact_name in JSON_CASE_ARTIFACT_NAMES:
        payload[artifact_name] = load_case_output_artifact(case_path, artifact_name)
    return payload


def load_case_output_artifact(case_outdir: str | Path, artifact_name: str) -> Any:
    """Load one named case artifact from either bundled or split JSON outputs."""

    case_path = Path(case_outdir)
    bundle_path = case_path / JSON_CASE_BUNDLE_NAME
    if bundle_path.exists():
        payload = load_json(bundle_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {bundle_path}")
        if artifact_name not in payload:
            raise KeyError(f"Artifact {artifact_name!r} not found in {bundle_path}")
        return payload[artifact_name]

    candidates = [
        case_path / f"{artifact_name}.json",
        case_path / "final" / f"{artifact_name}.json",
    ]
    for candidate in candidates:
        try:
            resolved = resolve_json_path(candidate)
        except FileNotFoundError:
            continue
        return load_json(resolved)
    raise FileNotFoundError(f"Case artifact not found: {artifact_name} under {case_path}")
