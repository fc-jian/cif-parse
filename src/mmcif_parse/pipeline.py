"""Shared end-to-end processing pipeline for single-file and batch CLI runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mmcif_parse.annotate import apply_antibody_pairing
from mmcif_parse.assemble import (
    identify_antibody_antigen_complexes,
    identify_dimer_interfaces,
    identify_tight_multimers,
    identify_tcr_pmhc_complexes,
)
from mmcif_parse.export import dump_csv_rows, dump_json, dump_jsonl
from mmcif_parse.io import (
    read_assembly_chain_operations,
    read_assembly_copy_numbers,
    read_chain_inventory,
    read_structure_summary,
    read_structure_preflight,
)
from mmcif_parse.settings import AppSettings


LOGGER = logging.getLogger(__name__)

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
NUCLEIC_ACID_CHAIN_TYPES = frozenset({"DNA chain", "RNA chain", "other nucleic acid chain"})
BRANCHED_CHAIN_TYPES = frozenset({"glycan / branched component"})
METAL_CHAIN_TYPES = frozenset({"metal ion"})
SMALL_MOLECULE_CHAIN_TYPES = frozenset({"small molecule compound"})


class StructureSkipWarning(RuntimeError):
    """Raised when a structure is intentionally skipped by configured guardrails."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def infer_case_id(input_path: str | Path) -> str:
    """Infer a stable case id from a mmCIF-like file path."""

    path = Path(input_path)
    name = path.name.lower()
    for suffix in (".cif.gz", ".bcif.gz", ".cif", ".bcif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem.lower()


def partition_chain_inventory(chain_inventory: list[Any]) -> dict[str, list[Any]]:
    """Split the chain inventory into the output groups used by exporters."""

    return {
        "protein_chains": [chain for chain in chain_inventory if chain.chain_type in PROTEIN_CHAIN_TYPES],
        "nucleic_acid_chains": [
            chain for chain in chain_inventory if chain.chain_type in NUCLEIC_ACID_CHAIN_TYPES
        ],
        "branched_entities": [chain for chain in chain_inventory if chain.chain_type in BRANCHED_CHAIN_TYPES],
        "metal_ions": [chain for chain in chain_inventory if chain.chain_type in METAL_CHAIN_TYPES],
        "small_molecule_compounds": [
            chain for chain in chain_inventory if chain.chain_type in SMALL_MOLECULE_CHAIN_TYPES
        ],
        "other_entities": [
            chain
            for chain in chain_inventory
            if chain.chain_type
            not in (
                PROTEIN_CHAIN_TYPES
                | NUCLEIC_ACID_CHAIN_TYPES
                | BRANCHED_CHAIN_TYPES
                | METAL_CHAIN_TYPES
                | SMALL_MOLECULE_CHAIN_TYPES
            )
        ],
    }


def validate_processing_inputs(
    input_path: str | Path,
    chain_inventory: list[Any],
    settings: AppSettings,
) -> None:
    """Apply cheap skip guards before expensive interface enumeration."""

    polymer_chains = [
        chain for chain in chain_inventory if str(getattr(chain, "entity_type", "")).lower() == "polymer"
    ]
    polymer_chain_count = len(polymer_chains)
    if polymer_chain_count > settings.max_polymer_chains:
        raise StructureSkipWarning(
            "too_many_polymer_chains",
            (
                f"Skipping {infer_case_id(input_path)}: polymer chain count {polymer_chain_count} "
                f"exceeds max_polymer_chains={settings.max_polymer_chains}"
            ),
            details={
                "polymer_chain_count": polymer_chain_count,
                "max_polymer_chains": settings.max_polymer_chains,
            },
        )

    min_required_length = settings.min_polymer_chain_length
    qualifying_polymer_chains = [chain for chain in polymer_chains if int(getattr(chain, "length", 0)) > min_required_length]
    if not qualifying_polymer_chains:
        raise StructureSkipWarning(
            "no_polymer_chain_above_min_length",
            (
                f"Skipping {infer_case_id(input_path)}: no polymer chain has length > "
                f"{min_required_length}"
            ),
            details={
                "polymer_chain_count": polymer_chain_count,
                "min_polymer_chain_length": min_required_length,
            },
        )


def write_single_outputs(
    outdir: str | Path,
    settings: AppSettings,
    summary: Any,
    chain_inventory: list[Any],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
    antibody_antigen_complexes: list[Any],
    tcr_pmhc_complexes: list[Any],
) -> list[str]:
    """Write all single-structure outputs in the configured primary format."""

    output_dir = Path(outdir)
    partitions = partition_chain_inventory(chain_inventory)
    if settings.output_format == "json":
        return _write_single_outputs_json(
            output_dir,
            summary,
            chain_inventory,
            partitions,
            dimer_interfaces,
            tight_multimers,
            antibody_antigen_complexes,
            tcr_pmhc_complexes,
        )
    return _write_single_outputs_csv(
        output_dir,
        summary,
        chain_inventory,
        partitions,
        dimer_interfaces,
        tight_multimers,
        antibody_antigen_complexes,
        tcr_pmhc_complexes,
        )


def validate_preflight_inputs(
    input_path: str | Path,
    preflight: dict[str, Any],
    settings: AppSettings,
) -> None:
    """Apply cheap guards before chain annotation and interface calculation."""

    polymer_chain_count = int(preflight.get("polymer_chain_count", 0) or 0)
    if polymer_chain_count > settings.max_polymer_chains:
        raise StructureSkipWarning(
            "too_many_polymer_chains",
            (
                f"Skipping {preflight.get('pdb_id') or infer_case_id(input_path)}: polymer chain count "
                f"{polymer_chain_count} exceeds max_polymer_chains={settings.max_polymer_chains}"
            ),
            details={
                "polymer_chain_count": polymer_chain_count,
                "max_polymer_chains": settings.max_polymer_chains,
            },
        )

    max_polymer_chain_length = int(preflight.get("max_polymer_chain_length", 0) or 0)
    if max_polymer_chain_length <= settings.min_polymer_chain_length:
        raise StructureSkipWarning(
            "no_polymer_chain_above_min_length",
            (
                f"Skipping {preflight.get('pdb_id') or infer_case_id(input_path)}: no polymer chain has length > "
                f"{settings.min_polymer_chain_length}"
            ),
            details={
                "polymer_chain_count": polymer_chain_count,
                "max_polymer_chain_length": max_polymer_chain_length,
                "min_polymer_chain_length": settings.min_polymer_chain_length,
            },
        )


def process_single_structure(
    input_path: str | Path,
    outdir: str | Path,
    settings: AppSettings,
) -> dict[str, Any]:
    """Run the full processing pipeline for one structure and persist outputs."""

    input_path = Path(input_path)
    outdir = Path(outdir)
    LOGGER.info("Processing mmCIF %s", input_path)
    preflight = read_structure_preflight(input_path)
    validate_preflight_inputs(input_path, preflight, settings)
    chain_inventory = read_chain_inventory(
        input_path,
        model=settings.model,
        coverage_mode=settings.coverage_mode,
    )
    validate_processing_inputs(input_path, chain_inventory, settings)
    summary = read_structure_summary(
        input_path,
        model=settings.model,
        use_author_fields=settings.use_author_fields,
        coverage_mode=settings.coverage_mode,
    )
    LOGGER.debug("Read structure summary for %s with %d chains", summary.pdb_id, len(summary.chain_ids))
    LOGGER.debug("Built chain inventory for %s with %d chains", summary.pdb_id, len(chain_inventory))
    dimer_interfaces = identify_dimer_interfaces(
        input_path,
        chain_inventory,
        model=settings.model,
        assembly_mode=settings.assembly_mode,
    )
    apply_antibody_pairing(chain_inventory, dimer_interfaces)
    LOGGER.debug("Identified %d dimer interfaces for %s", len(dimer_interfaces), summary.pdb_id)
    _, assembly_copy_numbers = read_assembly_copy_numbers(input_path)
    _, assembly_chain_operations = read_assembly_chain_operations(input_path)
    tight_multimers = identify_tight_multimers(
        chain_inventory,
        dimer_interfaces,
        assembly_mode=settings.assembly_mode,
        assembly_copy_numbers=assembly_copy_numbers if settings.assembly_mode == "biological_assembly" else {},
        assembly_chain_operations=(
            assembly_chain_operations if settings.assembly_mode == "biological_assembly" else {}
        ),
    )
    antibody_antigen_complexes = identify_antibody_antigen_complexes(
        chain_inventory,
        dimer_interfaces,
        tight_multimers,
    )
    tcr_pmhc_complexes = identify_tcr_pmhc_complexes(
        chain_inventory,
        dimer_interfaces,
        tight_multimers,
    )
    output_paths = write_single_outputs(
        outdir,
        settings,
        summary,
        chain_inventory,
        dimer_interfaces,
        tight_multimers,
        antibody_antigen_complexes,
        tcr_pmhc_complexes,
    )
    LOGGER.info(
        "Finished %s: %d chains, %d dimers, %d multimers, %d antibody complexes, %d TCR complexes",
        summary.pdb_id,
        len(chain_inventory),
        len(dimer_interfaces),
        len(tight_multimers),
        len(antibody_antigen_complexes),
        len(tcr_pmhc_complexes),
    )
    return {
        "pdb_id": summary.pdb_id,
        "input_path": str(input_path),
        "output_dir": str(outdir),
        "output_paths": output_paths,
        "num_chains": len(chain_inventory),
        "num_dimers": len(dimer_interfaces),
        "num_multimers": len(tight_multimers),
        "num_antibody_antigen_complexes": len(antibody_antigen_complexes),
        "num_tcr_pmhc_complexes": len(tcr_pmhc_complexes),
        "chain_type_counts": summary.chain_type_counts,
    }


def _write_single_outputs_json(
    outdir: Path,
    summary: Any,
    chain_inventory: list[Any],
    partitions: dict[str, list[Any]],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
    antibody_antigen_complexes: list[Any],
    tcr_pmhc_complexes: list[Any],
) -> list[str]:
    output_paths = [
        str(dump_json(outdir / "final" / "structure_summary.json", summary.to_dict())),
        str(
            dump_json(
                outdir / "final" / "chain_inventory.json",
                [chain.to_dict() for chain in chain_inventory],
            )
        ),
        str(
            dump_jsonl(
                outdir / "debug" / "chain_annotations.jsonl",
                [chain.to_dict() for chain in chain_inventory],
            )
        ),
        str(
            dump_json(
                outdir / "final" / "dimer_interfaces.json",
                [dimer.to_dict() for dimer in dimer_interfaces],
            )
        ),
        str(
            dump_json(
                outdir / "final" / "tight_multimers.json",
                [multimer.to_dict() for multimer in tight_multimers],
            )
        ),
        str(
            dump_json(
                outdir / "final" / "antibody_antigen_complexes.json",
                [complex_record.to_dict() for complex_record in antibody_antigen_complexes],
            )
        ),
        str(
            dump_json(
                outdir / "final" / "tcr_pmhc_complexes.json",
                [complex_record.to_dict() for complex_record in tcr_pmhc_complexes],
            )
        ),
    ]
    for output_name, chains in partitions.items():
        output_paths.append(
            str(
                dump_json(
                    outdir / "final" / f"{output_name}.json",
                    [chain.to_dict() for chain in chains],
                )
            )
        )
    return output_paths


def _write_single_outputs_csv(
    outdir: Path,
    summary: Any,
    chain_inventory: list[Any],
    partitions: dict[str, list[Any]],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
    antibody_antigen_complexes: list[Any],
    tcr_pmhc_complexes: list[Any],
) -> list[str]:
    output_paths = [
        str(dump_csv_rows(outdir / "csv" / "structure_summary.csv", [summary.to_record()])),
        str(
            dump_csv_rows(
                outdir / "csv" / "chain_inventory.csv",
                [chain.to_record() for chain in chain_inventory],
            )
        ),
        str(
            dump_jsonl(
                outdir / "debug" / "chain_annotations.jsonl",
                [chain.to_dict() for chain in chain_inventory],
            )
        ),
        str(
            dump_csv_rows(
                outdir / "csv" / "dimer_interfaces.csv",
                [dimer.to_record() for dimer in dimer_interfaces],
            )
        ),
        str(
            dump_csv_rows(
                outdir / "csv" / "tight_multimers.csv",
                [multimer.to_record() for multimer in tight_multimers],
            )
        ),
        str(
            dump_csv_rows(
                outdir / "csv" / "antibody_antigen_complexes.csv",
                [complex_record.to_record() for complex_record in antibody_antigen_complexes],
            )
        ),
        str(
            dump_json(
                outdir / "final" / "antibody_antigen_complexes.json",
                [complex_record.to_dict() for complex_record in antibody_antigen_complexes],
            )
        ),
        str(
            dump_csv_rows(
                outdir / "csv" / "tcr_pmhc_complexes.csv",
                [complex_record.to_record() for complex_record in tcr_pmhc_complexes],
            )
        ),
        str(
            dump_json(
                outdir / "final" / "tcr_pmhc_complexes.json",
                [complex_record.to_dict() for complex_record in tcr_pmhc_complexes],
            )
        ),
    ]
    for output_name, chains in partitions.items():
        output_paths.append(
            str(
                dump_csv_rows(
                    outdir / "csv" / f"{output_name}.csv",
                    [chain.to_record() for chain in chains],
                )
            )
        )
    return output_paths
