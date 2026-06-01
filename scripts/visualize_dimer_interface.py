#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_mplconfig = Path(os.environ.get("MPLCONFIGDIR", Path(os.environ.get("TMPDIR", "/tmp")) / "cif_parse_mplconfig"))
_mplconfig.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig))

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdbx import get_assembly, get_structure

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cif_parse.clustering.dimers import ChainSpec, DimerObservation, extract_dimer_structure  # noqa: E402
from cif_parse.interact.contacts import (  # noqa: E402
    build_chain_geometries,
    build_instance_geometries,
    compute_interface_metrics,
)
from cif_parse.io import read_chain_inventory, read_cif_file  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _chain_record(chain_map: dict[str, Any], chain_id: str) -> Any:
    record = chain_map.get(chain_id)
    if record is not None:
        return record
    return SimpleNamespace(
        label_asym_id=chain_id,
        auth_asym_id=None,
        chain_type="other protein chain",
        entity_id="",
        residue_count=0,
    )


def _parse_chain_ids(text: str, arg_name: str) -> list[str]:
    chain_ids = [item.strip() for item in text.replace("+", ",").split(",") if item.strip()]
    if not chain_ids:
        raise ValueError(f"{arg_name} must contain at least one chain id")
    return chain_ids


def _parse_sym_ids(text: str | None, expected_count: int, arg_name: str) -> list[int | None]:
    if text is None or text == "":
        return [None] * expected_count
    values: list[int | None] = []
    for item in text.split(","):
        stripped = item.strip()
        values.append(None if stripped in {"", "none", "None", "null", "NULL"} else int(stripped))
    if len(values) != expected_count:
        raise ValueError(f"{arg_name} must provide {expected_count} comma-separated value(s)")
    return values


def _resolve_sym_id(atom_array: Any, chain_id: str, requested: int | None, arg_name: str) -> int | None:
    if not hasattr(atom_array, "sym_id"):
        if requested is not None:
            raise ValueError(f"{arg_name} was provided, but the loaded atom array has no sym_id annotation")
        return None

    chain_mask = atom_array.chain_id == chain_id
    values = sorted({int(value) for value in atom_array.sym_id[chain_mask].tolist()})
    if not values:
        raise ValueError(f"Chain {chain_id!r} was not found in the selected coordinates")
    if requested is not None:
        if requested not in values:
            raise ValueError(f"{arg_name}={requested} is not present for chain {chain_id!r}; available: {values}")
        return requested
    if len(values) == 1:
        return values[0]
    raise ValueError(
        f"Chain {chain_id!r} has multiple sym_id values {values}; pass {arg_name} to choose one instance"
    )


def _resolve_part_specs(
    atom_array: Any,
    chain_ids: list[str],
    requested_sym_ids: list[int | None],
    arg_name: str,
) -> list[ChainSpec]:
    return [
        (chain_id, _resolve_sym_id(atom_array, chain_id, sym_id, f"{arg_name}[{index}]"))
        for index, (chain_id, sym_id) in enumerate(zip(chain_ids, requested_sym_ids, strict=True), start=1)
    ]


def _select_instance(atom_array: Any, chain_id: str, sym_id: int | None) -> Any:
    mask = atom_array.chain_id == chain_id
    if hasattr(atom_array, "hetero"):
        mask &= ~atom_array.hetero
    if sym_id is not None and hasattr(atom_array, "sym_id"):
        mask &= atom_array.sym_id == sym_id
    selected = atom_array[mask]
    if selected.array_length() == 0:
        raise ValueError(f"No atoms selected for chain {chain_id!r} sym_id={sym_id!r}")
    return selected


def _load_coordinates(cif_path: Path, *, assembly_id: str | None, model: int) -> tuple[Any, Any]:
    cif_file = read_cif_file(cif_path)
    if assembly_id:
        atom_array = get_assembly(
            cif_file,
            assembly_id=assembly_id,
            model=model,
            use_author_fields=False,
        )
    else:
        atom_array = get_structure(
            cif_file,
            model=model,
            use_author_fields=False,
        )
    return cif_file, atom_array


def _geometry_key(chain_id: str, sym_id: int | None, has_sym_id: bool) -> str:
    return f"{chain_id}@{int(sym_id or 0) + 1}" if has_sym_id else chain_id


def _compute_pair_metrics(
    pair_atoms: Any,
    chain_records: list[Any],
    *,
    part_1: list[ChainSpec],
    part_2: list[ChainSpec],
    residue_contact_cutoff: float,
    atom_contact_cutoff: float,
    min_residue_contacts: int,
    min_atom_contacts: int,
) -> dict[str, Any] | None:
    if hasattr(pair_atoms, "sym_id"):
        geometries = build_instance_geometries(pair_atoms, chain_records)
        has_sym_id = True
    else:
        geometries = build_chain_geometries(pair_atoms, chain_records)
        has_sym_id = False

    part_1_keys = [_geometry_key(chain_id, sym_id, has_sym_id) for chain_id, sym_id in part_1]
    part_2_keys = [_geometry_key(chain_id, sym_id, has_sym_id) for chain_id, sym_id in part_2]
    missing = [key for key in [*part_1_keys, *part_2_keys] if key not in geometries]
    if missing:
        raise ValueError(f"Could not build geometries for {missing!r}")

    metrics_list: list[dict[str, Any]] = []
    for key_1 in part_1_keys:
        for key_2 in part_2_keys:
            metrics = compute_interface_metrics(
                geometries[key_1],
                geometries[key_2],
                residue_contact_cutoff=residue_contact_cutoff,
                atom_contact_cutoff=atom_contact_cutoff,
                min_residue_contacts=0,
                min_atom_contacts=0,
            )
            if metrics is not None:
                metrics_list.append(metrics)

    if not metrics_list:
        return None

    summary = {
        "num_residue_contacts": sum(int(item.get("num_residue_contacts", 0) or 0) for item in metrics_list),
        "num_atom_contacts": sum(int(item.get("num_atom_contacts", 0) or 0) for item in metrics_list),
        "buried_area": sum(float(item.get("buried_area", 0.0) or 0.0) for item in metrics_list),
        "interface_residue_count_1": sum(
            int(item.get("interface_residue_count_1", 0) or 0) for item in metrics_list
        ),
        "interface_residue_count_2": sum(
            int(item.get("interface_residue_count_2", 0) or 0) for item in metrics_list
        ),
        "chain_pair_metrics": metrics_list,
    }
    if summary["num_residue_contacts"] < min_residue_contacts:
        return None
    if summary["num_atom_contacts"] < min_atom_contacts:
        return None
    return summary


def _observation(
    *,
    cif_path: Path,
    pdb_id: str,
    assembly_id: str | None,
    part_1_records: list[Any],
    part_2_records: list[Any],
    part_1: list[ChainSpec],
    part_2: list[ChainSpec],
    metrics: dict[str, Any],
) -> DimerObservation:
    chain_ids_1 = [chain_id for chain_id, _ in part_1]
    chain_ids_2 = [chain_id for chain_id, _ in part_2]
    sym_ids_1 = [sym_id for _, sym_id in part_1]
    sym_ids_2 = [sym_id for _, sym_id in part_2]
    return DimerObservation(
        dimer_observation_id=(
            f"{pdb_id}:{assembly_id or 'na'}:"
            f"{'+'.join(chain_ids_1)}:{'+'.join(str(sym_id or 0) for sym_id in sym_ids_1)}:"
            f"{'+'.join(chain_ids_2)}:{'+'.join(str(sym_id or 0) for sym_id in sym_ids_2)}"
        ),
        pdb_id=pdb_id,
        source_path=str(cif_path),
        assembly_id=assembly_id,
        assembly_mode="assembly" if assembly_id else "asymmetric_unit",
        sym_id_1=sym_ids_1[0] if len(sym_ids_1) == 1 else None,
        label_asym_id_1="+".join(chain_ids_1),
        auth_asym_id_1="+".join(str(getattr(record, "auth_asym_id", "") or "") for record in part_1_records) or None,
        chain_type_1="chain part 1",
        monomer_id_1=f"{pdb_id}:{assembly_id or 'na'}:{'+'.join(chain_ids_1)}",
        monomer_sequence_cluster_id_1=None,
        monomer_structure_cluster_id_1=None,
        sym_id_2=sym_ids_2[0] if len(sym_ids_2) == 1 else None,
        label_asym_id_2="+".join(chain_ids_2),
        auth_asym_id_2="+".join(str(getattr(record, "auth_asym_id", "") or "") for record in part_2_records) or None,
        chain_type_2="chain part 2",
        monomer_id_2=f"{pdb_id}:{assembly_id or 'na'}:{'+'.join(chain_ids_2)}",
        monomer_sequence_cluster_id_2=None,
        monomer_structure_cluster_id_2=None,
        interface_label="chain_part_1__chain_part_2",
        is_same_entity=(
            len(part_1_records) == 1
            and len(part_2_records) == 1
            and getattr(part_1_records[0], "entity_id", "") == getattr(part_2_records[0], "entity_id", "")
        ),
        contains_antibody_unit=False,
        contains_tcr_pmhc_unit=False,
        buried_area=float(metrics.get("buried_area", 0.0) or 0.0),
        num_residue_contacts=int(metrics.get("num_residue_contacts", 0) or 0),
        num_atom_contacts=int(metrics.get("num_atom_contacts", 0) or 0),
        signature_key="visual_check",
        signature_members=[],
        cluster_source_1="visual_check",
        cluster_source_2="visual_check",
    )


def _save_pymol_session_with_cmd(
    cmd: Any,
    *,
    full_pdb: Path,
    interface_pdb: Path,
    pse_path: Path,
    part_1_pdb_chain_ids: list[str],
    part_2_pdb_chain_ids: list[str],
) -> None:
    part_1_selection = f"interface_residues and chain {'+'.join(part_1_pdb_chain_ids)}"
    part_2_selection = f"interface_residues and chain {'+'.join(part_2_pdb_chain_ids)}"
    cmd.reinitialize()
    cmd.load(str(full_pdb), "full_dimer")
    cmd.load(str(interface_pdb), "interface_residues")
    cmd.hide("everything")
    cmd.show("cartoon", "full_dimer")
    cmd.show("sticks", "interface_residues")
    cmd.color("gray70", "full_dimer")
    cmd.color("orange", part_1_selection)
    cmd.color("marine", part_2_selection)
    cmd.set("transparency", 0.65, "full_dimer")
    cmd.zoom("interface_residues")
    cmd.save(str(pse_path))


def _build_pymol_session(
    pse_path: Path,
    *,
    full_pdb: Path,
    interface_pdb: Path,
    part_1_pdb_chain_ids: list[str],
    part_2_pdb_chain_ids: list[str],
) -> None:
    try:
        import pymol2  # type: ignore[import-not-found]

        with pymol2.PyMOL() as pymol:
            _save_pymol_session_with_cmd(
                pymol.cmd,
                full_pdb=full_pdb,
                interface_pdb=interface_pdb,
                pse_path=pse_path,
                part_1_pdb_chain_ids=part_1_pdb_chain_ids,
                part_2_pdb_chain_ids=part_2_pdb_chain_ids,
            )
    except ImportError:
        try:
            import pymol  # type: ignore[import-not-found]
            from pymol import cmd  # type: ignore[import-not-found]
        except ImportError as import_exc:
            raise RuntimeError("PyMOL Python API is required to write the PSE session") from import_exc

        pymol.finish_launching(["pymol", "-qc"])
        try:
            _save_pymol_session_with_cmd(
                cmd,
                full_pdb=full_pdb,
                interface_pdb=interface_pdb,
                pse_path=pse_path,
                part_1_pdb_chain_ids=part_1_pdb_chain_ids,
                part_2_pdb_chain_ids=part_2_pdb_chain_ids,
            )
        finally:
            cmd.quit()

    if not pse_path.exists():
        raise RuntimeError(f"PyMOL did not create the expected PSE file: {pse_path}")


def build_visual_check(args: argparse.Namespace) -> dict[str, Any]:
    cif_path = Path(args.cif).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cif_file, atom_array = _load_coordinates(cif_path, assembly_id=args.assembly_id, model=args.model)
    chains = read_chain_inventory(cif_path, model=args.model, cif_file=cif_file)
    chain_map = {chain.label_asym_id: chain for chain in chains}
    part_1_chain_ids = _parse_chain_ids(args.part_1_chain_ids, "part_1_chain_ids")
    part_2_chain_ids = _parse_chain_ids(args.part_2_chain_ids, "part_2_chain_ids")
    part_1_sym_text = args.part_1_sym_ids
    part_2_sym_text = args.part_2_sym_ids
    if args.sym_id_1 is not None:
        if len(part_1_chain_ids) != 1:
            raise ValueError("--sym-id-1 can only be used when part 1 has one chain")
        part_1_sym_text = str(args.sym_id_1)
    if args.sym_id_2 is not None:
        if len(part_2_chain_ids) != 1:
            raise ValueError("--sym-id-2 can only be used when part 2 has one chain")
        part_2_sym_text = str(args.sym_id_2)

    part_1 = _resolve_part_specs(
        atom_array,
        part_1_chain_ids,
        _parse_sym_ids(part_1_sym_text, len(part_1_chain_ids), "--part-1-sym-ids"),
        "--part-1-sym-ids",
    )
    part_2 = _resolve_part_specs(
        atom_array,
        part_2_chain_ids,
        _parse_sym_ids(part_2_sym_text, len(part_2_chain_ids), "--part-2-sym-ids"),
        "--part-2-sym-ids",
    )
    part_1_records = [_chain_record(chain_map, chain_id) for chain_id in part_1_chain_ids]
    part_2_records = [_chain_record(chain_map, chain_id) for chain_id in part_2_chain_ids]

    part_atoms = [
        _select_instance(atom_array, chain_id, sym_id)
        for chain_id, sym_id in [*part_1, *part_2]
    ]
    pair_atoms = struc.concatenate(part_atoms)

    metrics = _compute_pair_metrics(
        pair_atoms,
        [*part_1_records, *part_2_records],
        part_1=part_1,
        part_2=part_2,
        residue_contact_cutoff=args.residue_contact_cutoff,
        atom_contact_cutoff=args.atom_contact_cutoff,
        min_residue_contacts=args.min_residue_contacts,
        min_atom_contacts=args.min_atom_contacts,
    )
    if metrics is None:
        raise ValueError("No qualifying interaction found for the requested chain parts")

    pdb_id = getattr(part_1_records[0], "pdb_id", None) or cif_path.stem.split(".")[0]
    observation = _observation(
        cif_path=cif_path,
        pdb_id=str(pdb_id),
        assembly_id=args.assembly_id,
        part_1_records=part_1_records,
        part_2_records=part_2_records,
        part_1=part_1,
        part_2=part_2,
        metrics=metrics,
    )

    full = extract_dimer_structure(
        observation,
        outdir=outdir / "full_dimer_extract",
        atom_array=pair_atoms,
        drop_hydrogens=not args.keep_hydrogens,
        interface_residue_cutoff=None,
        chain_parts=[part_1, part_2],
    )
    interface = extract_dimer_structure(
        observation,
        outdir=outdir / "interface_extract",
        atom_array=pair_atoms,
        drop_hydrogens=not args.keep_hydrogens,
        interface_residue_cutoff=args.interface_residue_cutoff,
        chain_parts=[part_1, part_2],
    )

    full_pdb = outdir / "full_dimer.pdb"
    interface_pdb = outdir / "interface_residues.pdb"
    shutil.copyfile(full.extracted_pdb_path, full_pdb)
    shutil.copyfile(interface.extracted_pdb_path, interface_pdb)

    metrics_path = outdir / "interaction_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    pse_path = outdir / "dimer_interface_visual_check.pse"
    _build_pymol_session(
        pse_path,
        full_pdb=full_pdb,
        interface_pdb=interface_pdb,
        part_1_pdb_chain_ids=interface.part_1_pdb_chain_ids or [],
        part_2_pdb_chain_ids=interface.part_2_pdb_chain_ids or [],
    )

    summary = {
        "cif": str(cif_path),
        "part_1_chain_ids": part_1_chain_ids,
        "part_2_chain_ids": part_2_chain_ids,
        "assembly_id": args.assembly_id,
        "part_1_sym_ids": [sym_id for _, sym_id in part_1],
        "part_2_sym_ids": [sym_id for _, sym_id in part_2],
        "num_residue_contacts": int(metrics.get("num_residue_contacts", 0) or 0),
        "num_atom_contacts": int(metrics.get("num_atom_contacts", 0) or 0),
        "buried_area": float(metrics.get("buried_area", 0.0) or 0.0),
        "full_dimer_pdb": str(full_pdb),
        "interface_residue_pdb": str(interface_pdb),
        "part_1_pdb_chain_ids": interface.part_1_pdb_chain_ids or [],
        "part_2_pdb_chain_ids": interface.part_2_pdb_chain_ids or [],
        "metrics_json": str(metrics_path),
        "pymol_pse": str(pse_path),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate interaction between two CIF chain parts, extract the full dimer "
            "and interface-residue dimer PDBs, and build a PyMOL PSE with the PyMOL cmd API."
        )
    )
    parser.add_argument("cif", type=Path, help="Input .cif/.cif.gz/.bcif file")
    parser.add_argument("part_1_chain_ids", help="First chain part as comma-separated label_asym_id values")
    parser.add_argument("part_2_chain_ids", help="Second chain part as comma-separated label_asym_id values")
    parser.add_argument("--assembly-id", default=None, help="Optional biological assembly id")
    parser.add_argument("--part-1-sym-ids", default=None, help="Optional comma-separated sym_id values for part 1")
    parser.add_argument("--part-2-sym-ids", default=None, help="Optional comma-separated sym_id values for part 2")
    parser.add_argument("--sym-id-1", type=int, default=None, help="Alias for --part-1-sym-ids with one chain")
    parser.add_argument("--sym-id-2", type=int, default=None, help="Alias for --part-2-sym-ids with one chain")
    parser.add_argument("--outdir", type=Path, default=Path("dimer_interface_visual_check"))
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--residue-contact-cutoff", type=float, default=50)
    parser.add_argument("--atom-contact-cutoff", type=float, default=50)
    parser.add_argument("--min-residue-contacts", type=int, default=0)
    parser.add_argument("--min-atom-contacts", type=int, default=0)
    parser.add_argument("--interface-residue-cutoff", type=float, default=20.0)
    parser.add_argument("--keep-hydrogens", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model < 1:
        raise SystemExit("--model must be >= 1")
    for name in ("residue_contact_cutoff", "atom_contact_cutoff", "interface_residue_cutoff"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0")
    if args.min_residue_contacts < 0 or args.min_atom_contacts < 0:
        raise SystemExit("--min-residue-contacts and --min-atom-contacts must be >= 0")

    summary = build_visual_check(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
