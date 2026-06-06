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
JSON_CASE_ASSEMBLY_BUNDLE_PREFIX = "result_assembly_"
ASSEMBLY_OUTPUT_DIR_PREFIX = "assembly_"


def _assembly_sort_key(label: str) -> tuple[int, int | str]:
    if label.isdigit():
        return (0, int(label))
    return (1, label)


def _assembly_id_from_location(path: Path) -> str | None:
    name = path.name
    if name.startswith(JSON_CASE_ASSEMBLY_BUNDLE_PREFIX):
        remainder = name.removeprefix(JSON_CASE_ASSEMBLY_BUNDLE_PREFIX)
        for suffix in (".json.gz", ".json"):
            if remainder.endswith(suffix):
                remainder = remainder[: -len(suffix)]
                break
        return remainder or None
    if name.startswith(ASSEMBLY_OUTPUT_DIR_PREFIX):
        return name.removeprefix(ASSEMBLY_OUTPUT_DIR_PREFIX) or None
    return None


def _annotate_payload_assembly_id(payload: dict[str, Any], assembly_id: str | None) -> None:
    if not assembly_id:
        return
    summary = payload.get("structure_summary")
    if isinstance(summary, dict) and not summary.get("assembly_id"):
        summary["assembly_id"] = assembly_id


def _list_multi_bundle_paths(case_path: Path) -> list[Path]:
    bundle_paths = sorted(
        case_path.glob(f"{JSON_CASE_ASSEMBLY_BUNDLE_PREFIX}*.json.gz"),
        key=lambda path: _assembly_sort_key(
            path.name.removeprefix(JSON_CASE_ASSEMBLY_BUNDLE_PREFIX).removesuffix(".json.gz")
        ),
    )
    if bundle_paths:
        return bundle_paths

    assembly_dirs = sorted(
        [
            candidate
            for candidate in case_path.iterdir()
            if candidate.is_dir() and candidate.name.startswith(ASSEMBLY_OUTPUT_DIR_PREFIX)
        ],
        key=lambda path: _assembly_sort_key(path.name.removeprefix(ASSEMBLY_OUTPUT_DIR_PREFIX)),
    ) if case_path.exists() else []
    return assembly_dirs


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


def load_case_output_bundles(case_outdir: str | Path) -> list[dict[str, Any]]:
    """Load one or more case output bundles from bundled or split JSON layouts."""

    case_path = Path(case_outdir)
    bundle_path = case_path / JSON_CASE_BUNDLE_NAME
    if bundle_path.exists():
        payload = load_json(bundle_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload in {bundle_path}")
        _annotate_payload_assembly_id(
            payload,
            _assembly_id_from_location(bundle_path) or _assembly_id_from_location(case_path),
        )
        return [payload]

    multi_locations = _list_multi_bundle_paths(case_path)
    if multi_locations:
        payloads: list[dict[str, Any]] = []
        for location in multi_locations:
            if location.is_dir():
                payloads.extend(load_case_output_bundles(location))
                continue
            payload = load_json(location)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload in {location}")
            _annotate_payload_assembly_id(payload, _assembly_id_from_location(location))
            payloads.append(payload)
        return payloads

    payload: dict[str, Any] = {}
    for artifact_name in JSON_CASE_ARTIFACT_NAMES:
        payload[artifact_name] = load_case_output_artifact(case_path, artifact_name)
    _annotate_payload_assembly_id(payload, _assembly_id_from_location(case_path))
    return [payload]


def load_case_output_bundle(case_outdir: str | Path) -> dict[str, Any]:
    """Load a case output bundle, supporting both bundled and split JSON layouts."""

    payloads = load_case_output_bundles(case_outdir)
    if len(payloads) != 1:
        raise ValueError(
            f"Expected exactly one case output bundle under {case_outdir}, found {len(payloads)}"
        )
    return payloads[0]


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
