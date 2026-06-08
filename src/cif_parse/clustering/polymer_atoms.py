"""Shared atom handling for polymer structures passed to USalign."""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray, get_residues


def select_polymer_chain_atoms(
    atom_array: AtomArray,
    *,
    label_asym_id: str,
    sym_id: int | None = None,
) -> AtomArray:
    """Select one polymer chain instance without discarding modified residues."""

    mask = atom_array.chain_id == label_asym_id
    if sym_id is not None and hasattr(atom_array, "sym_id"):
        mask &= atom_array.sym_id == sym_id
    return atom_array[mask]


def prepare_polymer_atoms_for_usalign(
    atom_array: AtomArray,
    *,
    chain_id: str | None = None,
) -> AtomArray:
    """Return a PDB-ready copy that USalign recognizes as polymer atoms."""

    copied = atom_array.copy()
    if chain_id is not None:
        copied.chain_id[:] = chain_id
    # mmCIF uses HETATM for many polymerized modified residues, including D-aa.
    # USalign ignores these records in PDB input, so emit selected polymer atoms
    # as ATOM. PDB residue names are limited to 3 characters; sanitize only this
    # USalign copy so mmCIF/JSON annotations keep their original CCD IDs.
    if hasattr(copied, "hetero"):
        copied.hetero[:] = False
    if hasattr(copied, "res_name"):
        copied.res_name = np.asarray(
            [
                residue_name if 0 < len(residue_name) <= 3 else "UNK"
                for residue_name in (str(value).strip() for value in copied.res_name)
            ],
            dtype="U3",
        )
    return copied


def select_trace_atoms_for_usalign(atom_array: AtomArray) -> AtomArray:
    """Keep only representative backbone atoms for USalign input PDBs.

    Protein TM-score uses CA atoms; nucleic-acid alignments need P atoms.
    Keeping both lets mixed protein/nucleic-acid complexes remain analyzable
    while cutting most intermediate PDB volume.
    """

    if not hasattr(atom_array, "atom_name"):
        return atom_array
    atom_names = np.asarray([str(value).strip().upper() for value in atom_array.atom_name])
    mask = np.isin(atom_names, ["CA", "P"])
    if not bool(mask.any()):
        return atom_array
    return atom_array[mask]


def polymer_residue_counts_by_chain(atom_array: AtomArray) -> dict[str, int]:
    """Count residues independently for each output PDB chain."""

    counts: dict[str, int] = {}
    for chain_id in np.unique(atom_array.chain_id):
        chain_atoms = atom_array[atom_array.chain_id == chain_id]
        _, residue_names = get_residues(chain_atoms)
        counts[str(chain_id)] = int(len(residue_names))
    return counts


def validate_usalign_chain_lengths(
    atom_array: AtomArray,
    *,
    context: str,
    min_residues: int = 3,
) -> dict[str, int]:
    """Reject structures containing a PDB chain too short for USalign."""

    counts = polymer_residue_counts_by_chain(atom_array)
    too_short = {
        chain_id: count
        for chain_id, count in counts.items()
        if count < min_residues
    }
    if too_short:
        observed = ", ".join(
            f"{chain_id}={count}" for chain_id, count in sorted(counts.items())
        )
        raise ValueError(
            f"USalign requires at least {min_residues} residues per PDB chain; "
            f"observed {observed} for {context}"
        )
    return counts
