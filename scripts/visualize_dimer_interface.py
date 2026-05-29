#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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

from cif_parse.clustering.dimers import DimerObservation, extract_dimer_structure  # noqa: E402
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


def _compute_pair_metrics(
    pair_atoms: Any,
    chain_records: list[Any],
    *,
    chain_id_1: str,
    chain_id_2: str,
    sym_id_1: int | None,
    sym_id_2: int | None,
    residue_contact_cutoff: float,
    atom_contact_cutoff: float,
    min_residue_contacts: int,
    min_atom_contacts: int,
) -> dict[str, Any] | None:
    if hasattr(pair_atoms, "sym_id"):
        geometries = build_instance_geometries(pair_atoms, chain_records)
        key_1 = f"{chain_id_1}@{int(sym_id_1 or 0) + 1}"
        key_2 = f"{chain_id_2}@{int(sym_id_2 or 0) + 1}"
    else:
        if chain_id_1 == chain_id_2:
            raise ValueError("Same-chain dimer instances require sym_id annotation")
        geometries = build_chain_geometries(pair_atoms, chain_records)
        key_1 = chain_id_1
        key_2 = chain_id_2

    if key_1 not in geometries or key_2 not in geometries:
        raise ValueError(f"Could not build geometries for {key_1!r} and {key_2!r}")
    return compute_interface_metrics(
        geometries[key_1],
        geometries[key_2],
        residue_contact_cutoff=residue_contact_cutoff,
        atom_contact_cutoff=atom_contact_cutoff,
        min_residue_contacts=min_residue_contacts,
        min_atom_contacts=min_atom_contacts,
    )


def _observation(
    *,
    cif_path: Path,
    pdb_id: str,
    assembly_id: str | None,
    chain_1: Any,
    chain_2: Any,
    chain_id_1: str,
    chain_id_2: str,
    sym_id_1: int | None,
    sym_id_2: int | None,
    metrics: dict[str, Any],
) -> DimerObservation:
    return DimerObservation(
        dimer_observation_id=f"{pdb_id}:{assembly_id or 'na'}:{chain_id_1}:{sym_id_1 or 0}:{chain_id_2}:{sym_id_2 or 0}",
        pdb_id=pdb_id,
        source_path=str(cif_path),
        assembly_id=assembly_id,
        assembly_mode="assembly" if assembly_id else "asymmetric_unit",
        sym_id_1=sym_id_1,
        label_asym_id_1=chain_id_1,
        auth_asym_id_1=getattr(chain_1, "auth_asym_id", None),
        chain_type_1=getattr(chain_1, "chain_type", "other protein chain"),
        monomer_id_1=f"{pdb_id}:{assembly_id or 'na'}:{chain_id_1}",
        monomer_sequence_cluster_id_1=None,
        monomer_structure_cluster_id_1=None,
        sym_id_2=sym_id_2,
        label_asym_id_2=chain_id_2,
        auth_asym_id_2=getattr(chain_2, "auth_asym_id", None),
        chain_type_2=getattr(chain_2, "chain_type", "other protein chain"),
        monomer_id_2=f"{pdb_id}:{assembly_id or 'na'}:{chain_id_2}",
        monomer_sequence_cluster_id_2=None,
        monomer_structure_cluster_id_2=None,
        interface_label=f"{getattr(chain_1, 'chain_type', 'chain')}__{getattr(chain_2, 'chain_type', 'chain')}",
        is_same_entity=getattr(chain_1, "entity_id", "") == getattr(chain_2, "entity_id", ""),
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


def _write_pymol_pml(
    *,
    pml_path: Path,
    full_pdb: Path,
    interface_pdb: Path,
    pse_path: Path,
) -> None:
    full_pdb_text = json.dumps(full_pdb.as_posix())
    interface_pdb_text = json.dumps(interface_pdb.as_posix())
    pse_path_text = json.dumps(pse_path.as_posix())
    pml_path.write_text(
        "\n".join(
            [
                "reinitialize",
                f"load {full_pdb_text}, full_dimer",
                f"load {interface_pdb_text}, interface_residues",
                "hide everything",
                "show cartoon, full_dimer",
                "show sticks, interface_residues",
                "color gray70, full_dimer",
                "color orange, interface_residues and chain A",
                "color marine, interface_residues and chain B",
                "set transparency, 0.65, full_dimer",
                "zoom interface_residues",
                f"save {pse_path_text}",
                "quit",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_pymol_session(
    pml_path: Path,
    pse_path: Path,
    *,
    full_pdb: Path,
    interface_pdb: Path,
    require_pymol: bool,
) -> bool:
    pymol_exe = shutil.which("pymol")
    if pymol_exe is not None:
        try:
            subprocess.run([pymol_exe, "-cq", str(pml_path)], check=True)
            return pse_path.exists()
        except Exception as subprocess_exc:
            if require_pymol:
                raise RuntimeError(f"PyMOL executable session generation failed: {subprocess_exc}") from subprocess_exc

    try:
        import pymol2  # type: ignore[import-not-found]

        with pymol2.PyMOL() as pymol:
            cmd = pymol.cmd
            cmd.reinitialize()
            cmd.load(str(full_pdb), "full_dimer")
            cmd.load(str(interface_pdb), "interface_residues")
            cmd.hide("everything")
            cmd.show("cartoon", "full_dimer")
            cmd.show("sticks", "interface_residues")
            cmd.color("gray70", "full_dimer")
            cmd.color("orange", "interface_residues and chain A")
            cmd.color("marine", "interface_residues and chain B")
            cmd.set("transparency", 0.65, "full_dimer")
            cmd.zoom("interface_residues")
            cmd.save(str(pse_path))
        return pse_path.exists()
    except Exception as pymol2_exc:
        if require_pymol:
            raise RuntimeError("PyMOL is not available as pymol2 or pymol executable") from pymol2_exc
        return False


def build_visual_check(args: argparse.Namespace) -> dict[str, Any]:
    cif_path = Path(args.cif).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cif_file, atom_array = _load_coordinates(cif_path, assembly_id=args.assembly_id, model=args.model)
    chains = read_chain_inventory(cif_path, model=args.model, cif_file=cif_file)
    chain_map = {chain.label_asym_id: chain for chain in chains}
    chain_1 = _chain_record(chain_map, args.chain_id_1)
    chain_2 = _chain_record(chain_map, args.chain_id_2)

    sym_id_1 = _resolve_sym_id(atom_array, args.chain_id_1, args.sym_id_1, "--sym-id-1")
    sym_id_2 = _resolve_sym_id(atom_array, args.chain_id_2, args.sym_id_2, "--sym-id-2")
    atoms_1 = _select_instance(atom_array, args.chain_id_1, sym_id_1)
    atoms_2 = _select_instance(atom_array, args.chain_id_2, sym_id_2)
    pair_atoms = struc.concatenate([atoms_1, atoms_2])

    metrics = _compute_pair_metrics(
        pair_atoms,
        [chain_1, chain_2],
        chain_id_1=args.chain_id_1,
        chain_id_2=args.chain_id_2,
        sym_id_1=sym_id_1,
        sym_id_2=sym_id_2,
        residue_contact_cutoff=args.residue_contact_cutoff,
        atom_contact_cutoff=args.atom_contact_cutoff,
        min_residue_contacts=args.min_residue_contacts,
        min_atom_contacts=args.min_atom_contacts,
    )
    if metrics is None:
        raise ValueError("No qualifying interaction found for the requested chain pair")

    pdb_id = getattr(chain_1, "pdb_id", None) or cif_path.stem.split(".")[0]
    observation = _observation(
        cif_path=cif_path,
        pdb_id=str(pdb_id),
        assembly_id=args.assembly_id,
        chain_1=chain_1,
        chain_2=chain_2,
        chain_id_1=args.chain_id_1,
        chain_id_2=args.chain_id_2,
        sym_id_1=sym_id_1,
        sym_id_2=sym_id_2,
        metrics=metrics,
    )

    full = extract_dimer_structure(
        observation,
        outdir=outdir / "full_dimer_extract",
        atom_array=pair_atoms,
        drop_hydrogens=not args.keep_hydrogens,
        interface_residue_cutoff=None,
    )
    interface = extract_dimer_structure(
        observation,
        outdir=outdir / "interface_extract",
        atom_array=pair_atoms,
        drop_hydrogens=not args.keep_hydrogens,
        interface_residue_cutoff=args.interface_residue_cutoff,
    )

    full_pdb = outdir / "full_dimer.pdb"
    interface_pdb = outdir / "interface_residues.pdb"
    shutil.copyfile(full.extracted_pdb_path, full_pdb)
    shutil.copyfile(interface.extracted_pdb_path, interface_pdb)

    metrics_path = outdir / "interaction_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    pse_path = outdir / "dimer_interface_visual_check.pse"
    pml_path = outdir / "make_dimer_interface_visual_check.pml"
    _write_pymol_pml(pml_path=pml_path, full_pdb=full_pdb, interface_pdb=interface_pdb, pse_path=pse_path)
    pse_created = _build_pymol_session(
        pml_path,
        pse_path,
        full_pdb=full_pdb,
        interface_pdb=interface_pdb,
        require_pymol=args.require_pymol,
    )

    summary = {
        "cif": str(cif_path),
        "chain_id_1": args.chain_id_1,
        "chain_id_2": args.chain_id_2,
        "assembly_id": args.assembly_id,
        "sym_id_1": sym_id_1,
        "sym_id_2": sym_id_2,
        "num_residue_contacts": int(metrics.get("num_residue_contacts", 0) or 0),
        "num_atom_contacts": int(metrics.get("num_atom_contacts", 0) or 0),
        "buried_area": float(metrics.get("buried_area", 0.0) or 0.0),
        "full_dimer_pdb": str(full_pdb),
        "interface_residue_pdb": str(interface_pdb),
        "metrics_json": str(metrics_path),
        "pymol_pml": str(pml_path),
        "pymol_pse": str(pse_path) if pse_created else "",
        "pymol_pse_created": bool(pse_created),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate interaction between two CIF chains, extract the full dimer "
            "and interface-residue dimer PDBs, and build a PyMOL PSE when PyMOL is available."
        )
    )
    parser.add_argument("cif", type=Path, help="Input .cif/.cif.gz/.bcif file")
    parser.add_argument("chain_id_1", help="First label_asym_id")
    parser.add_argument("chain_id_2", help="Second label_asym_id")
    parser.add_argument("--assembly-id", default=None, help="Optional biological assembly id")
    parser.add_argument("--sym-id-1", type=int, default=None, help="Optional sym_id for chain 1")
    parser.add_argument("--sym-id-2", type=int, default=None, help="Optional sym_id for chain 2")
    parser.add_argument("--outdir", type=Path, default=Path("dimer_interface_visual_check"))
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--residue-contact-cutoff", type=float, default=8.0)
    parser.add_argument("--atom-contact-cutoff", type=float, default=5.0)
    parser.add_argument("--min-residue-contacts", type=int, default=3)
    parser.add_argument("--min-atom-contacts", type=int, default=20)
    parser.add_argument("--interface-residue-cutoff", type=float, default=8.0)
    parser.add_argument("--keep-hydrogens", action="store_true")
    parser.add_argument(
        "--require-pymol",
        action="store_true",
        help="Fail if PyMOL is unavailable; otherwise write PDB/JSON/PML and skip PSE",
    )
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
