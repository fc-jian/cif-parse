"""Shared end-to-end processing pipeline for single-file and batch CLI runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import logging
import pickle
import re
import zlib
from pathlib import Path
from typing import Any

from biotite.structure.io.pdbx import CIFFile

from cif_parse.annotate import apply_antibody_pairing
from cif_parse.assemble import (
    identify_antibody_antigen_complexes,
    identify_dimer_interfaces,
    identify_tight_multimers,
    identify_tcr_pmhc_complexes,
)
from cif_parse.export import build_single_json_bundle, dump_csv_rows, dump_json, dump_jsonl, dump_single_json_bundle
from cif_parse.io import (
    read_assembly_chain_operations,
    read_assembly_copy_numbers,
    read_available_assembly_ids,
    read_case_metadata,
    read_chain_inventory,
    read_structure_summary,
    read_structure_preflight,
)
from cif_parse.io.cif_reader import read_cif_file, select_largest_polymer_assembly_id
from cif_parse.io.cif_reader import preflight_assembly_atom_counts
from cif_parse.constants import (
    BRANCHED_CHAIN_TYPES,
    METAL_CHAIN_TYPES,
    NUCLEIC_ACID_CHAIN_TYPES,
    PROTEIN_CHAIN_TYPES,
    SMALL_MOLECULE_CHAIN_TYPES,
)
from cif_parse.settings import AppSettings


LOGGER = logging.getLogger(__name__)


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
    pdb_id = (
        str(getattr(chain_inventory[0], "pdb_id", "") or "")
        if chain_inventory
        else infer_case_id(input_path)
    )
    if polymer_chain_count > settings.max_polymer_chains:
        raise StructureSkipWarning(
            "too_many_polymer_chains",
            (
                f"Skipping {pdb_id or infer_case_id(input_path)}: polymer chain count {polymer_chain_count} "
                f"exceeds max_polymer_chains={settings.max_polymer_chains}"
            ),
            details={
                "pdb_id": pdb_id or infer_case_id(input_path),
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
                f"Skipping {pdb_id or infer_case_id(input_path)}: no polymer chain has length > "
                f"{min_required_length}"
            ),
            details={
                "pdb_id": pdb_id or infer_case_id(input_path),
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
    *,
    bundle_name: str = "result.json.gz",
) -> list[str]:
    """Write all single-structure outputs in the configured primary format."""

    output_dir = Path(outdir)
    partitions = partition_chain_inventory(chain_inventory)
    if settings.output_format == "json":
        return _write_single_outputs_json(
            output_dir,
            settings,
            summary,
            chain_inventory,
            partitions,
            dimer_interfaces,
            tight_multimers,
            antibody_antigen_complexes,
            tcr_pmhc_complexes,
            bundle_name=bundle_name,
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


def _coerce_entry_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize entry-level metadata for JSON/prep consumers."""

    metadata = dict(metadata or {})
    method = str(metadata.get("experimental_method", "") or "").strip()
    resolution = metadata.get("resolution", "")
    if resolution in (None, "", ".", "?"):
        resolution_value: float | str = ""
    else:
        try:
            resolution_value = round(float(resolution), 2)
        except (TypeError, ValueError):
            resolution_value = ""
    release_date = str(metadata.get("release_date", "") or "").strip()
    return {
        **metadata,
        "experimental_method": method,
        "resolution": resolution_value,
        "release_date": release_date,
        "metadata_source": metadata.get("metadata_source") or "input_cif",
    }


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
                "pdb_id": preflight.get("pdb_id") or infer_case_id(input_path),
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
                "pdb_id": preflight.get("pdb_id") or infer_case_id(input_path),
                "polymer_chain_count": polymer_chain_count,
                "max_polymer_chain_length": max_polymer_chain_length,
                "min_polymer_chain_length": settings.min_polymer_chain_length,
            },
        )


def process_single_structure(
    input_path: str | Path,
    outdir: str | Path,
    settings: AppSettings,
    *,
    _allowed_assembly_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full processing pipeline for one structure and persist outputs."""

    input_path = Path(input_path)
    outdir = Path(outdir)
    LOGGER.info("Processing mmCIF %s", input_path)
    cif_file = read_cif_file(input_path)
    try:
        metadata = read_case_metadata(input_path, cif_file=cif_file)
    except Exception:
        LOGGER.debug("Failed to read metadata for %s", input_path, exc_info=True)
        metadata = {}
    metadata = _coerce_entry_metadata(metadata)
    preflight = read_structure_preflight(input_path, cif_file=cif_file)
    validate_preflight_inputs(input_path, preflight, settings)
    chain_inventory = read_chain_inventory(
        input_path,
        model=settings.model,
        coverage_mode=settings.coverage_mode,
        drop_hydrogens_for_analysis=settings.drop_hydrogens_for_analysis,
        sadie_domain_bitscore_threshold=settings.sadie_domain_bitscore_threshold,
        sadie_domain_limit=settings.sadie_domain_limit,
        cif_file=cif_file,
    )
    validate_processing_inputs(input_path, chain_inventory, settings)
    summary = read_structure_summary(
        input_path,
        model=settings.model,
        use_author_fields=settings.use_author_fields,
        coverage_mode=settings.coverage_mode,
        drop_hydrogens_for_analysis=settings.drop_hydrogens_for_analysis,
        chain_inventory=chain_inventory,
        sadie_domain_bitscore_threshold=settings.sadie_domain_bitscore_threshold,
        sadie_domain_limit=settings.sadie_domain_limit,
        cif_file=cif_file,
    )
    LOGGER.debug("Read structure summary for %s with %d chains", summary.pdb_id, len(summary.chain_ids))
    LOGGER.debug("Built chain inventory for %s with %d chains", summary.pdb_id, len(chain_inventory))
    summary.entry_metadata = metadata

    # Validate pre-split assembly files: chain IDs must be unique.
    if settings.input_assembly:
        from collections import Counter as _Counter
        chain_ids = [c.label_asym_id for c in chain_inventory]
        dupes = [cid for cid, n in _Counter(chain_ids).items() if n > 1]
        if dupes:
            raise StructureSkipWarning(
                "duplicate_chain_ids_in_input_assembly",
                f"Skipping {summary.pdb_id}: input_assembly mode requires unique chain IDs. "
                f"Duplicates: {', '.join(sorted(dupes))}",
                details={"duplicate_chain_ids": sorted(dupes)},
            )
        input_assembly_id = _infer_input_assembly_id(input_path)
        if input_assembly_id:
            summary.assembly_ids = [input_assembly_id]
            summary.assembly_descriptions = {
                input_assembly_id: "input pre-split biological assembly",
            }

    if settings.assembly_mode != "all":
        selected_assembly_id = _resolve_single_mode_assembly_id(
            input_path=input_path,
            cif_file=cif_file,
            settings=settings,
            pdb_id=summary.pdb_id,
        )
        _dump_atom_cache(
            outdir=outdir,
            input_path=input_path,
            cif_file=cif_file,
            assembly_mode=settings.assembly_mode,
            selected_assembly_id=selected_assembly_id,
            model=settings.model,
        )
        result = _process_single_structure_for_mode(
            input_path=input_path,
            outdir=outdir,
            settings=settings,
            summary=summary,
            chain_inventory=chain_inventory,
            cif_file=cif_file,
            analysis_assembly_mode="input_assembly" if settings.input_assembly else settings.assembly_mode,
            selected_assembly_id=selected_assembly_id,
            bundle_name=_resolve_single_mode_bundle_name(settings, selected_assembly_id),
        )
        result["_meta"] = metadata
        LOGGER.info(
            "Finished %s: %d chains, %d dimers, %d multimers, %d antibody complexes, %d TCR complexes",
            summary.pdb_id,
            result["num_chains"],
            result["num_dimers"],
            result["num_multimers"],
            result["num_antibody_antigen_complexes"],
            result["num_tcr_pmhc_complexes"],
        )
        return result

    if _allowed_assembly_ids is not None:
        # Main process already filtered and sorted by atom count.
        assembly_ids = _allowed_assembly_ids
        if not assembly_ids:
            raise StructureSkipWarning(
                "all_assemblies_exceed_max_atoms",
                f"Skipping {summary.pdb_id}: all assemblies exceed the atom threshold",
                details={"max_assembly_atoms": getattr(settings, "max_assembly_atoms", None)},
            )
    else:
        available_assembly_ids = read_available_assembly_ids(input_path, cif_file=cif_file)
        if not available_assembly_ids:
            raise StructureSkipWarning(
                "no_available_assemblies_for_all_mode",
                f"Skipping {summary.pdb_id}: assembly_mode=all requires at least one assembly id",
                details={"assembly_ids": []},
            )
        atom_counts: dict[str, int] = {}
        try:
            atom_counts = preflight_assembly_atom_counts(input_path, cif_file=cif_file)
        except Exception:
            LOGGER.debug("Failed to estimate assembly atom counts for %s", input_path, exc_info=True)
        max_assembly_atoms = getattr(settings, "max_assembly_atoms", None)
        if max_assembly_atoms is not None and max_assembly_atoms > 0 and atom_counts:
            skipped_large = [
                aid
                for aid in available_assembly_ids
                if atom_counts.get(aid, 0) > max_assembly_atoms
            ]
            if skipped_large:
                LOGGER.warning(
                    "Skipping %d assembly(s) in %s exceeding %d estimated atoms: %s",
                    len(skipped_large),
                    summary.pdb_id,
                    max_assembly_atoms,
                    ", ".join(skipped_large),
                )
            assembly_ids = [
                aid
                for aid in available_assembly_ids
                if aid not in skipped_large
            ]
            if not assembly_ids:
                raise StructureSkipWarning(
                    "all_assemblies_exceed_max_atoms",
                    (
                        f"Skipping {summary.pdb_id}: all {len(available_assembly_ids)} assemblies "
                        f"exceed {max_assembly_atoms} estimated atoms"
                    ),
                    details={
                        "assembly_ids": available_assembly_ids,
                        "assembly_atom_counts": atom_counts,
                        "max_assembly_atoms": max_assembly_atoms,
                    },
                )
        else:
            assembly_ids = list(available_assembly_ids)
        if len(assembly_ids) > 1 and atom_counts:
            assembly_ids.sort(key=lambda aid: atom_counts.get(aid, 0), reverse=True)

    assembly_results: list[dict[str, Any]] = []
    output_paths: list[str] = []
    total_dimers = 0
    total_multimers = 0
    total_antibody_complexes = 0
    total_tcr_complexes = 0

    # Pre-cache asymmetric unit atoms (shared by all assemblies).
    _dump_atom_cache_au(
        outdir=outdir,
        input_path=input_path,
        cif_file=cif_file,
        model=settings.model,
    )

    assembly_jobs = getattr(settings, "assembly_jobs", 1) or 1
    max_workers = max(1, min(assembly_jobs, len(assembly_ids)))
    if max_workers > 1:
        futures: dict[Any, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for assembly_id in assembly_ids:
                assembly_outdir, bundle_name = _resolve_all_mode_output_target(
                    outdir, settings, assembly_id
                )
                future = executor.submit(
                    _process_single_assembly_parallel,
                    input_path=input_path,
                    outdir=assembly_outdir,
                    settings=settings,
                    summary=summary,
                    chain_inventory=chain_inventory,
                    cif_file=None,
                    selected_assembly_id=assembly_id,
                    bundle_name=bundle_name,
                )
                futures[future] = assembly_id
            for future in as_completed(futures):
                assembly_result = future.result()
                assembly_results.append(assembly_result)
                output_paths.extend(assembly_result["output_paths"])
                total_dimers += int(assembly_result["num_dimers"])
                total_multimers += int(assembly_result["num_multimers"])
                total_antibody_complexes += int(assembly_result["num_antibody_antigen_complexes"])
                total_tcr_complexes += int(assembly_result["num_tcr_pmhc_complexes"])
    else:
        for assembly_id in assembly_ids:
            assembly_outdir, bundle_name = _resolve_all_mode_output_target(
                outdir, settings, assembly_id
            )
            assembly_result = _process_single_assembly_parallel(
                input_path=input_path,
                outdir=assembly_outdir,
                settings=settings,
                summary=summary,
                chain_inventory=chain_inventory,
                cif_file=cif_file,
                selected_assembly_id=assembly_id,
                bundle_name=bundle_name,
            )
            assembly_results.append(assembly_result)
            output_paths.extend(assembly_result["output_paths"])
            total_dimers += int(assembly_result["num_dimers"])
            total_multimers += int(assembly_result["num_multimers"])
            total_antibody_complexes += int(assembly_result["num_antibody_antigen_complexes"])
            total_tcr_complexes += int(assembly_result["num_tcr_pmhc_complexes"])

    LOGGER.info(
        "Finished %s across %d assemblies: %d chains, %d dimers, %d multimers, %d antibody complexes, %d TCR complexes",
        summary.pdb_id,
        len(assembly_results),
        len(chain_inventory),
        total_dimers,
        total_multimers,
        total_antibody_complexes,
        total_tcr_complexes,
    )
    return {
        "pdb_id": summary.pdb_id,
        "input_path": str(input_path),
        "output_dir": str(outdir),
        "output_paths": output_paths,
        "num_chains": len(chain_inventory),
        "num_dimers": total_dimers,
        "num_multimers": total_multimers,
        "num_antibody_antigen_complexes": total_antibody_complexes,
        "num_tcr_pmhc_complexes": total_tcr_complexes,
        "chain_type_counts": summary.chain_type_counts,
        "num_assemblies_processed": len(assembly_results),
        "processed_assembly_ids": assembly_ids,
        "assembly_results": assembly_results,
        "_meta": metadata,
    }


def _resolve_single_mode_assembly_id(
    *,
    input_path: Path,
    cif_file: CIFFile,
    settings: AppSettings,
    pdb_id: str,
) -> str | None:
    if settings.input_assembly:
        return _infer_input_assembly_id(input_path)
    if settings.assembly_mode == "largest_assembly":
        return select_largest_polymer_assembly_id(cif_file)
    if settings.assembly_mode != "first_assembly":
        return None

    available_assembly_ids = read_available_assembly_ids(input_path, cif_file=cif_file)
    if not available_assembly_ids:
        LOGGER.warning("No biological assembly ids found for %s in first_assembly mode; using asymmetric unit", pdb_id)
        return None

    atom_counts: dict[str, int] = {}
    try:
        atom_counts = preflight_assembly_atom_counts(input_path, cif_file=cif_file)
    except Exception:
        LOGGER.debug("Failed to estimate assembly atom counts for %s", input_path, exc_info=True)

    max_assembly_atoms = getattr(settings, "max_assembly_atoms", None)
    if max_assembly_atoms is None or max_assembly_atoms <= 0 or not atom_counts:
        return available_assembly_ids[0]

    skipped_large = [
        aid
        for aid in available_assembly_ids
        if atom_counts.get(aid, 0) > max_assembly_atoms
    ]
    selected_assembly_id = next(
        (aid for aid in available_assembly_ids if aid not in skipped_large),
        None,
    )
    if selected_assembly_id is None:
        raise StructureSkipWarning(
            "first_assembly_candidates_exceed_max_atoms",
            (
                f"Skipping {pdb_id}: all {len(available_assembly_ids)} assemblies "
                f"exceed {max_assembly_atoms} estimated atoms in first_assembly mode"
            ),
            details={
                "assembly_ids": available_assembly_ids,
                "assembly_atom_counts": atom_counts,
                "max_assembly_atoms": max_assembly_atoms,
            },
        )

    skipped_before_selected = [
        aid
        for aid in available_assembly_ids[: available_assembly_ids.index(selected_assembly_id)]
        if aid in skipped_large
    ]
    if skipped_before_selected:
        LOGGER.warning(
            "first_assembly mode for %s selected assembly %s after skipping earlier assembly id(s) "
            "exceeding %d estimated atoms: %s",
            pdb_id,
            selected_assembly_id,
            max_assembly_atoms,
            ", ".join(skipped_before_selected),
        )
    return selected_assembly_id


def _resolve_single_mode_bundle_name(settings: AppSettings, selected_assembly_id: str | None) -> str:
    if (
        (settings.assembly_mode == "first_assembly" or settings.input_assembly)
        and settings.output_format == "json"
        and not settings.debug
        and selected_assembly_id
    ):
        return f"result_assembly_{selected_assembly_id}.json.gz"
    return "result.json.gz"


def _infer_input_assembly_id(input_path: Path) -> str | None:
    name = input_path.name.lower()
    for suffix in (".cif.gz", ".bcif.gz", ".cif", ".bcif"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    match = re.search(r"(?:^|[-_])assembly[-_]?([a-z0-9]+)(?:$|[-_])", name)
    if match:
        return match.group(1)
    return None


def _dump_atom_cache_au(
    *,
    outdir: Path,
    input_path: Path,
    cif_file: CIFFile | None = None,
    model: int,
) -> None:
    """Write the asymmetric unit atom cache (``atoms/_none.pkl``) once.

    This must be called *before* parallel per-assembly processing so that
    concurrent threads never race to write the same shared file.
    """
    atoms_dir = outdir / "atoms"
    atoms_dir.mkdir(parents=True, exist_ok=True)
    au_cache_path = atoms_dir / "_none.pkl"
    if au_cache_path.exists():
        return
    try:
        from biotite.structure.io.pdbx import get_structure
        cif_file = cif_file or read_cif_file(input_path)
        atom_array = get_structure(cif_file, model=model, use_author_fields=False)
        if atom_array is not None and len(atom_array) > 0:
            au_cache_path.write_bytes(
                zlib.compress(pickle.dumps(atom_array, protocol=pickle.HIGHEST_PROTOCOL), 3)
            )
    except Exception:
        LOGGER.debug("Failed to cache asymmetric unit for %s", input_path)


def _process_single_assembly_parallel(
    *,
    input_path: Path,
    outdir: Path,
    settings: Any,
    summary: Any,
    chain_inventory: list[Any],
    cif_file: CIFFile | None = None,
    selected_assembly_id: str,
    bundle_name: str,
) -> dict[str, Any]:
    """Wrapper around ``_process_single_structure_for_mode`` for parallel use.

    Handles the per-assembly atom cache write before delegating.
    """
    if cif_file is None:
        cif_file = read_cif_file(input_path)
    _dump_atom_cache(
        outdir=outdir,
        input_path=input_path,
        cif_file=cif_file,
        assembly_mode="all",
        selected_assembly_id=selected_assembly_id,
        model=settings.model,
    )
    return _process_single_structure_for_mode(
        input_path=input_path,
        outdir=outdir,
        settings=settings,
        summary=summary,
        chain_inventory=chain_inventory,
        cif_file=cif_file,
        analysis_assembly_mode="all",
        selected_assembly_id=selected_assembly_id,
        bundle_name=bundle_name,
    )


def _resolve_all_mode_output_target(
    outdir: Path,
    settings: AppSettings,
    assembly_id: str,
) -> tuple[Path, str]:
    if settings.output_format == "json" and not settings.debug:
        return outdir, f"result_assembly_{assembly_id}.json.gz"
    return outdir / f"assembly_{assembly_id}", "result.json.gz"


def _dump_atom_cache(
    *,
    outdir: Path,
    input_path: Path,
    cif_file: CIFFile | None = None,
    assembly_mode: str,
    selected_assembly_id: str | None,
    model: int,
) -> None:
    """Save the parsed atom array(s) to ``outdir/atoms/`` so that downstream
    clustering can consume them directly without re-reading the original mmCIF.

    Always saves ``atoms/_none.pkl`` (asymmetric unit).  For assembly modes
    ``largest_assembly`` and ``all`` also saves the assembly-expanded atom
    array as ``atoms/{assembly_id}.pkl``.
    """
    atoms_dir = outdir / "atoms"
    atoms_dir.mkdir(parents=True, exist_ok=True)

    try:
        from biotite.structure.io.pdbx import get_assembly, get_structure

        cif_file = cif_file or read_cif_file(input_path)

        # Always cache asymmetric unit (needed by monomer extraction)
        au_cache_path = atoms_dir / "_none.pkl"
        if not au_cache_path.exists():
            try:
                atom_array = get_structure(cif_file, model=model, use_author_fields=False)
                if atom_array is not None and len(atom_array) > 0:
                    au_cache_path.write_bytes(
                        zlib.compress(pickle.dumps(atom_array, protocol=pickle.HIGHEST_PROTOCOL), 3)
                    )
            except Exception:
                LOGGER.debug("Failed to cache asymmetric unit for %s", input_path)

        if assembly_mode == "asymmetric_unit" and selected_assembly_id:
            asm_cache_path = atoms_dir / f"{selected_assembly_id}.pkl"
            if not asm_cache_path.exists() and au_cache_path.exists():
                asm_cache_path.write_bytes(au_cache_path.read_bytes())

        # Cache assembly-expanded coordinates when applicable
        if assembly_mode in ("largest_assembly", "first_assembly", "all"):
            effective_assembly_id = selected_assembly_id
            if effective_assembly_id is None and assembly_mode == "largest_assembly":
                effective_assembly_id = select_largest_polymer_assembly_id(cif_file)
            elif effective_assembly_id is None and assembly_mode == "first_assembly":
                from cif_parse.io.cif_reader import read_available_assembly_ids
                available = read_available_assembly_ids(input_path, cif_file=cif_file)
                effective_assembly_id = available[0] if available else None

            if effective_assembly_id:
                asm_cache_path = atoms_dir / f"{effective_assembly_id}.pkl"
                if not asm_cache_path.exists():
                    try:
                        atom_array = get_assembly(
                            cif_file,
                            assembly_id=effective_assembly_id,
                            model=model,
                            use_author_fields=False,
                        )
                        if atom_array is not None and len(atom_array) > 0:
                            asm_cache_path.write_bytes(
                                zlib.compress(pickle.dumps(atom_array, protocol=pickle.HIGHEST_PROTOCOL), 3)
                            )
                    except ValueError as exc:
                        if str(exc) == "Array must contain at least one element":
                            LOGGER.debug("Empty assembly %s for %s", effective_assembly_id, input_path)
                        else:
                            raise
                    except Exception:
                        LOGGER.debug(
                            "Failed to cache assembly %s for %s",
                            effective_assembly_id,
                            input_path,
                        )
    except Exception:
        LOGGER.debug("Failed to cache atom arrays for %s", input_path, exc_info=True)


def _process_single_structure_for_mode(
    *,
    input_path: Path,
    outdir: Path,
    settings: AppSettings,
    summary: Any,
    chain_inventory: list[Any],
    cif_file: CIFFile | None = None,
    analysis_assembly_mode: str,
    selected_assembly_id: str | None,
    bundle_name: str,
) -> dict[str, Any]:
    working_chain_inventory = deepcopy(chain_inventory)
    dimer_interfaces = identify_dimer_interfaces(
        input_path,
        working_chain_inventory,
        model=settings.model,
        assembly_mode=analysis_assembly_mode,
        assembly_id=selected_assembly_id,
        drop_hydrogens_for_analysis=settings.drop_hydrogens_for_analysis,
        residue_contact_cutoff=settings.residue_contact_cutoff,
        atom_contact_cutoff=settings.atom_contact_cutoff,
        min_residue_contacts=settings.min_residue_contacts,
        min_atom_contacts=settings.min_atom_contacts,
        cif_file=cif_file,
    )
    apply_antibody_pairing(working_chain_inventory, dimer_interfaces)
    LOGGER.debug(
        "Identified %d dimer interfaces for %s%s",
        len(dimer_interfaces),
        summary.pdb_id,
        f" assembly {selected_assembly_id}" if selected_assembly_id is not None else "",
    )
    if analysis_assembly_mode == "input_assembly":
        assembly_copy_numbers = {}
        assembly_chain_operations = {}
    else:
        _, assembly_copy_numbers = read_assembly_copy_numbers(
            input_path,
            assembly_id=selected_assembly_id,
            cif_file=cif_file,
        )
        _, assembly_chain_operations = read_assembly_chain_operations(
            input_path,
            assembly_id=selected_assembly_id,
            cif_file=cif_file,
        )
    tight_multimers = identify_tight_multimers(
        working_chain_inventory,
        dimer_interfaces,
        assembly_mode=analysis_assembly_mode,
        assembly_copy_numbers=assembly_copy_numbers if analysis_assembly_mode in {"largest_assembly", "first_assembly", "all"} else {},
        assembly_chain_operations=(
            assembly_chain_operations if analysis_assembly_mode in {"largest_assembly", "first_assembly", "all"} else {}
        ),
        min_buried_area=settings.tight_multimer_min_buried_area,
        louvain_resolution=settings.tight_multimer_louvain_resolution,
        min_member_instances=settings.tight_multimer_min_member_instances,
        large_component_warning_size=settings.tight_multimer_large_component_warning_size,
    )
    antibody_antigen_complexes = identify_antibody_antigen_complexes(
        working_chain_inventory,
        dimer_interfaces,
        tight_multimers,
    )
    tcr_pmhc_complexes = identify_tcr_pmhc_complexes(
        working_chain_inventory,
        dimer_interfaces,
        tight_multimers,
        peptide_max_length=settings.peptide_max_length,
    )
    output_paths = write_single_outputs(
        outdir,
        settings,
        summary,
        working_chain_inventory,
        dimer_interfaces,
        tight_multimers,
        antibody_antigen_complexes,
        tcr_pmhc_complexes,
        bundle_name=bundle_name,
    )
    return {
        "pdb_id": summary.pdb_id,
        "input_path": str(input_path),
        "output_dir": str(outdir),
        "output_paths": output_paths,
        "num_chains": len(working_chain_inventory),
        "num_dimers": len(dimer_interfaces),
        "num_multimers": len(tight_multimers),
        "num_antibody_antigen_complexes": len(antibody_antigen_complexes),
        "num_tcr_pmhc_complexes": len(tcr_pmhc_complexes),
        "chain_type_counts": summary.chain_type_counts,
        "assembly_id": selected_assembly_id,
    }


def _write_single_outputs_json(
    outdir: Path,
    settings: AppSettings,
    summary: Any,
    chain_inventory: list[Any],
    partitions: dict[str, list[Any]],
    dimer_interfaces: list[Any],
    tight_multimers: list[Any],
    antibody_antigen_complexes: list[Any],
    tcr_pmhc_complexes: list[Any],
    *,
    bundle_name: str,
) -> list[str]:
    if not settings.debug:
        bundle = build_single_json_bundle(
            summary=summary,
            chain_inventory=chain_inventory,
            partitions=partitions,
            dimer_interfaces=dimer_interfaces,
            tight_multimers=tight_multimers,
            antibody_antigen_complexes=antibody_antigen_complexes,
            tcr_pmhc_complexes=tcr_pmhc_complexes,
        )
        return [str(dump_single_json_bundle(outdir / bundle_name, bundle))]

    output_paths = [
        str(dump_json(outdir / "structure_summary.json", summary.to_dict())),
        str(
            dump_json(
                outdir / "chain_inventory.json",
                [chain.to_dict() for chain in chain_inventory],
            )
        ),
        str(
            dump_json(
                outdir / "dimer_interfaces.json",
                [dimer.to_dict() for dimer in dimer_interfaces],
            )
        ),
        str(
            dump_json(
                outdir / "tight_multimers.json",
                [multimer.to_dict() for multimer in tight_multimers],
            )
        ),
        str(
            dump_json(
                outdir / "antibody_antigen_complexes.json",
                [complex_record.to_dict() for complex_record in antibody_antigen_complexes],
            )
        ),
        str(
            dump_json(
                outdir / "tcr_pmhc_complexes.json",
                [complex_record.to_dict() for complex_record in tcr_pmhc_complexes],
            )
        ),
    ]
    for output_name, chains in partitions.items():
        output_paths.append(
            str(
                dump_json(
                    outdir / f"{output_name}.json",
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
