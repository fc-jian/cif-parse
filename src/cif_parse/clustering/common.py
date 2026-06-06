from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


class MissingMonomerClusterAssignmentError(ValueError):
    """Raised when a high-order member has no sequence/structure assignment."""


def canonical_monomer_id(pdb_id: str, label_asym_id: str, assembly_id: str = "") -> str:
    if assembly_id:
        return f"{pdb_id}:{assembly_id}:{label_asym_id}"
    return f"{pdb_id}:{label_asym_id}"


def load_monomer_inventory(clustering_outdir: str | Path) -> dict[str, dict[str, Any]]:
    """Load canonical monomer metadata keyed by monomer id."""

    inventory_path = Path(clustering_outdir) / "monomer_inventory.jsonl"
    if not inventory_path.exists():
        raise FileNotFoundError(
            "High-order clustering requires monomer_inventory.jsonl from the sequence "
            f"clustering stage. Missing {inventory_path}."
        )

    inventory: dict[str, dict[str, Any]] = {}
    with inventory_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            monomer_id = str(payload.get("monomer_id", "") or "")
            if monomer_id:
                inventory[monomer_id] = payload
    return inventory


def load_monomer_cluster_assignments(
    clustering_outdir: str | Path,
    *,
    include_structure: bool = True,
) -> dict[str, dict[str, str]]:
    """Load monomer sequence / structure cluster assignments from clustering artifacts."""

    clustering_outdir = Path(clustering_outdir)
    assignments: dict[str, dict[str, str]] = {}

    sequence_membership = clustering_outdir / "sequence_clusters" / "membership.csv"
    if sequence_membership.exists():
        with sequence_membership.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                member_monomer_id = str(row.get("member_monomer_id", "") or "")
                if not member_monomer_id:
                    continue
                assignments.setdefault(member_monomer_id, {})
                assignments[member_monomer_id]["sequence_cluster_id"] = str(
                    row.get("cluster_id", "") or ""
                )

    structure_membership = clustering_outdir / "structure_clusters" / "protein_structure_cluster_membership.csv"
    if include_structure and structure_membership.exists():
        with structure_membership.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                member_monomer_id = str(row.get("member_monomer_id", "") or "")
                if not member_monomer_id:
                    continue
                assignments.setdefault(member_monomer_id, {})
                assignments[member_monomer_id]["structure_cluster_id"] = str(
                    row.get("structure_cluster_id", "") or ""
                )

    return assignments


def resolve_monomer_cluster(
    monomer_id: str,
    assignments: dict[str, dict[str, str]],
) -> tuple[str, str | None, str | None]:
    assignment = assignments.get(monomer_id)
    if assignment is None:
        raise MissingMonomerClusterAssignmentError(
            "High-order clustering encountered a monomer absent from the sequence clustering "
            f"dataset: {monomer_id}. Refusing to create an implicit singleton cluster. "
            "Verify that prep inputs, cutoff-date, and sequence clustering artifacts match."
        )
    structure_cluster_id = assignment.get("structure_cluster_id") or None
    sequence_cluster_id = assignment.get("sequence_cluster_id") or None
    if structure_cluster_id is not None:
        return "structure", structure_cluster_id, sequence_cluster_id
    if sequence_cluster_id is not None:
        return "sequence", sequence_cluster_id, sequence_cluster_id
    raise MissingMonomerClusterAssignmentError(
        "High-order clustering found an empty monomer cluster assignment for "
        f"{monomer_id}. Refusing to create an implicit singleton cluster."
    )


def monomer_inventory_source_paths(
    inventory: dict[str, dict[str, Any]],
) -> set[str]:
    """Return the exact source scope represented by the sequence dataset."""

    missing_source_ids = [
        monomer_id
        for monomer_id, row in inventory.items()
        if not str(row.get("source_path", "") or "")
    ]
    if missing_source_ids:
        preview = ", ".join(sorted(missing_source_ids)[:5])
        raise ValueError(
            "Sequence monomer inventory contains entries without source_path; cannot enforce "
            f"the high-order cutoff scope. Example monomer ids: {preview}"
        )
    return {str(row["source_path"]) for row in inventory.values()}


def _looks_like_case_output_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "result.json").exists() or (path / "result.json.gz").exists():
        return True
    if (path / "structure_summary.json").exists() or (path / "structure_summary.json.gz").exists():
        return True
    if any(path.glob("result_assembly_*.json.gz")):
        return True
    return any(candidate.is_dir() and candidate.name.startswith("assembly_") for candidate in path.iterdir())


def discover_case_output_dirs(inputs: Iterable[str | Path]) -> list[Path]:
    """Discover case-output directories under the provided paths."""

    discovered: list[Path] = []
    discovered_keys: set[Path] = set()
    visited: set[Path] = set()
    pending = [Path(item).resolve() for item in inputs]
    while pending:
        current = pending.pop(0)
        if current in visited or not current.exists():
            continue
        visited.add(current)
        if current.is_file():
            continue
        if _looks_like_case_output_dir(current):
            if current not in discovered_keys:
                discovered_keys.add(current)
                discovered.append(current)
            continue
        for child in sorted(current.iterdir()):
            if child.is_dir():
                pending.append(child.resolve())
    return sorted(discovered)
