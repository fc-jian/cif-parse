from __future__ import annotations

from typing import Any

import numpy as np
from biotite.structure import AtomArray


def normalize_element_symbol(element: str | None, atom_name: str | None) -> str:
    raw_element = str(element or "").strip().upper()
    if raw_element:
        return raw_element

    letters = "".join(character for character in str(atom_name or "").strip() if character.isalpha()).upper()
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2] in {"BR", "CL", "SE"}:
        return letters[:2]
    return letters[0]


def is_hydrogen_atom(element: str | None, atom_name: str | None) -> bool:
    return normalize_element_symbol(element, atom_name) in {"H", "D", "T"}


def filter_atom_array_for_analysis(
    atom_array: AtomArray,
    *,
    drop_hydrogens: bool = True,
    drop_nonfinite: bool = True,
) -> tuple[AtomArray, dict[str, int]]:
    """Filter an atom array for geometry-heavy downstream analysis."""

    if atom_array.array_length() == 0:
        return atom_array, {"hydrogen_atoms_removed": 0, "nonfinite_atoms_removed": 0}

    hydrogen_mask = np.zeros(atom_array.array_length(), dtype=bool)
    if drop_hydrogens:
        hydrogen_mask = np.asarray(
            [
                is_hydrogen_atom(str(element), str(atom_name))
                for element, atom_name in zip(atom_array.element, atom_array.atom_name, strict=False)
            ],
            dtype=bool,
        )

    nonfinite_mask = np.zeros(atom_array.array_length(), dtype=bool)
    if drop_nonfinite:
        nonfinite_mask = ~np.isfinite(np.asarray(atom_array.coord, dtype=np.float32)).all(axis=1)

    keep_mask = ~(hydrogen_mask | nonfinite_mask)
    return atom_array[keep_mask], {
        "hydrogen_atoms_removed": int(np.count_nonzero(hydrogen_mask)),
        "nonfinite_atoms_removed": int(np.count_nonzero(nonfinite_mask)),
    }


def atom_array_filter_counts(filter_counts: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(filter_counts, dict):
        return {"hydrogen_atoms_removed": 0, "nonfinite_atoms_removed": 0}
    return {
        "hydrogen_atoms_removed": int(filter_counts.get("hydrogen_atoms_removed", 0) or 0),
        "nonfinite_atoms_removed": int(filter_counts.get("nonfinite_atoms_removed", 0) or 0),
    }
