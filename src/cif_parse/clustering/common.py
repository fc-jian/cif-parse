from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def canonical_monomer_id(pdb_id: str, label_asym_id: str) -> str:
    return f"{pdb_id}:{label_asym_id}"


def load_monomer_inventory(clustering_outdir: str | Path) -> dict[str, dict[str, Any]]:
    """Load canonical monomer metadata keyed by monomer id."""

    inventory_path = Path(clustering_outdir) / "monomer_inventory.jsonl"
    if not inventory_path.exists():
        return {}

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


def load_monomer_cluster_assignments(clustering_outdir: str | Path) -> dict[str, dict[str, str]]:
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

    structure_membership = (
        clustering_outdir / "structure_clusters" / "protein_structure_cluster_membership.csv"
    )
    if structure_membership.exists():
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
    assignment = assignments.get(monomer_id, {})
    structure_cluster_id = assignment.get("structure_cluster_id") or None
    sequence_cluster_id = assignment.get("sequence_cluster_id") or None
    if structure_cluster_id is not None:
        return "structure", structure_cluster_id, sequence_cluster_id
    if sequence_cluster_id is not None:
        return "sequence", sequence_cluster_id, sequence_cluster_id
    return "unclustered", f"monomer:{monomer_id}", None
