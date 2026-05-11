"""Antibody-antigen complex refinement: Fv cropping + antigen domain filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from cif_parse.clustering.prep import (
    assemble_atom_array_from_chains,
    load_cif_coords_index,
    load_chain_from_prep,
)

LOGGER = logging.getLogger(__name__)

#: PDB chain IDs used for renumbering in refined output.
_PDB_CHAIN_IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


@dataclass(slots=True)
class RefineAntibodyComplexResult:
    complex_id: str
    pdb_id: str
    assembly_id: str | None
    source_path: str
    antibody_unit_type: str
    pdb_path: str
    json_path: str
    chain_intervals: list[dict[str, Any]] = field(default_factory=list)
    antigen_domains: list[dict[str, Any]] = field(default_factory=list)
    removed_antigen_domains: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fv cropping
# ---------------------------------------------------------------------------

_FV_END_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})


def _find_fv_boundaries(
    atom_array: Any,
    variable_domains: list[dict[str, Any]],
    chain_type: str,
) -> tuple[int, int | None]:
    """Return ``(fv_start_res_id, fv_end_res_id)`` for an antibody chain.

    Uses IMGT variable-domain annotation when available; otherwise falls back
    to a heuristic: keep residues up to the conserved Fv C-terminal motif
    (e.g. ``...VTVSS`` for heavy chains, ``...VEIK`` for kappa).
    """
    if variable_domains:
        vd = variable_domains[0]
        return (int(vd.get("seq_start", 1)), int(vd.get("seq_end", 0)) or None)

    # Fallback: take the first ~120 residues based on residue ordering.
    res_ids = sorted(set(int(r) for r in atom_array.res_id))
    if not res_ids:
        return (0, 0)
    fv_end_idx = min(120, len(res_ids))
    return (res_ids[0], res_ids[fv_end_idx - 1])


def _crop_atom_array(
    atom_array: Any,
    start_res_id: int,
    end_res_id: int | None,
) -> Any:
    """Return a copy of *atom_array* with residues outside [start, end] removed."""
    if end_res_id is None:
        return atom_array.copy()
    mask = (atom_array.res_id >= start_res_id) & (atom_array.res_id <= end_res_id)
    if not mask.any():
        return atom_array.copy()
    return atom_array[mask].copy()


# ---------------------------------------------------------------------------
# Antigen contact graph + domain detection
# ---------------------------------------------------------------------------

def _ca_coords(atom_array: Any) -> dict[int, np.ndarray]:
    """Extract per-residue Cα coordinates (or P for nucleic acids)."""
    coords: dict[int, np.ndarray] = {}
    for i in range(atom_array.array_length()):
        aname = str(atom_array.atom_name[i]).strip()
        if aname in ("CA", "P"):
            rid = int(atom_array.res_id[i])
            if rid not in coords:
                coords[rid] = np.asarray(atom_array.coord[i], dtype=np.float32)
    return coords


def _residue_contacts(
    ab_coords: dict[int, np.ndarray],
    ag_coords: dict[int, np.ndarray],
    distance: float = 8.0,
) -> list[tuple[int, int, float]]:
    """Return antibody–antigen residue pairs within *distance* (Å)."""
    if not ab_coords or not ag_coords:
        return []
    ab_ids = list(ab_coords)
    ag_ids = list(ag_coords)
    ab_mat = np.asarray([ab_coords[r] for r in ab_ids], dtype=np.float32)
    ag_mat = np.asarray([ag_coords[r] for r in ag_ids], dtype=np.float32)
    # Pairwise distances: (N_ab, N_ag, 3) → (N_ab, N_ag)
    diffs = ab_mat[:, None, :] - ag_mat[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=2))
    pairs: list[tuple[int, int, float]] = []
    for i, rid_ab in enumerate(ab_ids):
        for j, rid_ag in enumerate(ag_ids):
            d = float(dists[i, j])
            if d <= distance:
                pairs.append((rid_ab, rid_ag, d))
    return pairs


def _ag_internal_graph(
    ag_coords: dict[int, np.ndarray],
    distance: float = 8.0,
) -> dict[int, set[int]]:
    """Build an adjacency dict for antigen residues within *distance*."""
    if not ag_coords:
        return {}
    ids = list(ag_coords)
    mat = np.asarray([ag_coords[r] for r in ids], dtype=np.float32)
    diffs = mat[:, None, :] - mat[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=2))
    adj: dict[int, set[int]] = {rid: set() for rid in ids}
    for i, rid_i in enumerate(ids):
        for j, rid_j in enumerate(ids):
            if i < j and float(dists[i, j]) <= distance:
                adj[rid_i].add(rid_j)
                adj[rid_j].add(rid_i)
    return adj


def _identify_antigen_domains(
    ag_coords: dict[int, np.ndarray],
    ab_ag_contacts: set[int],
    *,
    distance: float = 8.0,
    louvain_resolution: float = 1.0,
    min_domain_size: int = 10,
    min_contact_residues: int = 3,
) -> tuple[list[list[int]], list[list[int]]]:
    """Partition antigen residues into domains and classify by antibody contact.

    Returns ``(contacting_domains, removed_domains)`` where each domain is a
    sorted list of residue IDs.
    """
    adj = _ag_internal_graph(ag_coords, distance=distance)
    if not adj:
        return ([], [])

    try:
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(adj)
        for u, neighbors in adj.items():
            for v in neighbors:
                if u < v:
                    graph.add_edge(u, v)
        communities = nx.community.louvain_communities(
            graph, weight=None, resolution=louvain_resolution, seed=0,
        )
    except Exception:
        LOGGER.debug("Louvain failed, treating antigen as single domain", exc_info=True)
        communities = [set(adj.keys())]

    contacting: list[list[int]] = []
    removed: list[list[int]] = []
    for comm in communities:
        domain = sorted(comm)
        contact_count = len(set(domain) & ab_ag_contacts)
        if len(domain) < min_domain_size:
            removed.append(domain)
        elif contact_count >= min_contact_residues:
            contacting.append(domain)
        else:
            removed.append(domain)

    return (contacting, removed)


# ---------------------------------------------------------------------------
# Main refinement entry points
# ---------------------------------------------------------------------------

def _load_chain_inventory(source_path: str, case_dir: Path) -> list[dict[str, Any]]:
    """Load chain inventory for a case, trying multiple bundle paths."""
    candidates = list(case_dir.glob("*.json.gz")) + list(case_dir.glob("*.json"))
    for path in candidates:
        try:
            if path.suffix == ".gz":
                raw = gzip.decompress(path.read_bytes())
            else:
                raw = path.read_bytes()
            bundle = json.loads(raw)
            chains = bundle.get("chain_inventory", [])
            if chains:
                return chains
        except Exception:
            continue
    return []


def _find_chain_annotation(chain_inventory: list[dict[str, Any]], label_asym_id: str) -> dict[str, Any]:
    for chain in chain_inventory:
        if chain.get("label_asym_id") == label_asym_id:
            return chain
    return {}


def refine_antibody_complex(
    *,
    pdb_id: str,
    source_path: str,
    assembly_id: str | None,
    complex_id: str,
    antibody_unit_type: str,
    antibody_chain_ids: list[str],
    antibody_chain_types: list[str],
    antigen_chain_ids: list[str],
    antigen_chain_types: list[str],
    chain_inventory: list[dict[str, Any]],
    prep_dir: str | Path,
    outdir: str | Path,
    contact_distance: float = 8.0,
    louvain_resolution: float = 1.0,
    min_domain_size: int = 10,
    min_contact_residues: int = 3,
) -> RefineAntibodyComplexResult:
    """Refine one antibody-antigen complex."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prep_dir = Path(prep_dir)
    prep_index = load_cif_coords_index(prep_dir)
    warnings: list[str] = []

    # ---- load per-chain atom arrays ----------------------------------------
    chain_arrays: dict[str, Any] = {}
    for cid in antibody_chain_ids + antigen_chain_ids:
        blob = load_chain_from_prep(prep_dir, source_path, cid, assembly_id=assembly_id, index=prep_index)
        if blob is None or blob.get("atom_array") is None:
            warnings.append(f"chain {cid} atom array not found in prep")
            continue
        aa = blob["atom_array"]
        chain_arrays[cid] = aa

    if not chain_arrays:
        raise ValueError(f"No chain atom arrays loaded for complex {complex_id}")

    # ---- antibody Fv cropping ----------------------------------------------
    ab_cropped: list[Any] = []
    ab_res_ca: dict[int, np.ndarray] = {}
    chain_intervals: list[dict[str, Any]] = []
    for cid in antibody_chain_ids:
        aa = chain_arrays.get(cid)
        if aa is None:
            continue
        ann = _find_chain_annotation(chain_inventory, cid)
        features = ann.get("features", {})
        ab_analysis = features.get("antibody_analysis", {}) or {}
        variable_domains = ab_analysis.get("variable_domains", []) or []
        chain_type = ann.get("chain_type", "")

        if antibody_unit_type in ("VHH", "scFv"):
            # Keep full chain for single-domain antibodies.
            start, end = (int(aa.res_id.min()), None)
        else:
            start, end = _find_fv_boundaries(aa, variable_domains, chain_type)

        cropped = _crop_atom_array(aa, start, end)
        ab_cropped.append(cropped)
        ab_res_ca.update(_ca_coords(cropped))
        chain_intervals.append({
            "label_asym_id": cid,
            "chain_type": chain_type,
            "role": "antibody",
            "fv_start_res_id": start,
            "fv_end_res_id": end,
            "retained_residues": sorted(set(int(r) for r in cropped.res_id)),
        })

    # ---- antigen domain detection and filtering ----------------------------
    antigen_domains: list[dict[str, Any]] = []
    removed_domains: list[dict[str, Any]] = []
    ag_cropped_parts: list[Any] = []

    for cid in antigen_chain_ids:
        aa = chain_arrays.get(cid)
        if aa is None:
            continue
        ann = _find_chain_annotation(chain_inventory, cid)
        chain_type = ann.get("chain_type", "unknown")

        ag_coords = _ca_coords(aa)
        contacts = _residue_contacts(ab_res_ca, ag_coords, distance=contact_distance)
        contact_residues = {rid_ag for _, rid_ag, _ in contacts}

        contacting, removed = _identify_antigen_domains(
            ag_coords,
            contact_residues,
            distance=contact_distance,
            louvain_resolution=louvain_resolution,
            min_domain_size=min_domain_size,
            min_contact_residues=min_contact_residues,
        )

        for domain in contacting:
            antigen_domains.append({
                "label_asym_id": cid,
                "chain_type": chain_type,
                "residue_ids": domain,
                "num_residues": len(domain),
                "num_antibody_contacts": len(set(domain) & contact_residues),
            })
        for domain in removed:
            removed_domains.append({
                "label_asym_id": cid,
                "chain_type": chain_type,
                "residue_ids": domain,
                "num_residues": len(domain),
                "num_antibody_contacts": len(set(domain) & contact_residues),
            })

        # Crop antigen chain to retained residue IDs.
        retained_ids: set[int] = set()
        for domain_residues in contacting:
            retained_ids.update(domain_residues)
        if retained_ids:
            keep_mask = np.zeros(aa.array_length(), dtype=bool)
            for i in range(aa.array_length()):
                if int(aa.res_id[i]) in retained_ids:
                    keep_mask[i] = True
            ag_cropped = aa[keep_mask].copy()
        else:
            ag_cropped = aa.copy()
        ag_cropped_parts.append(ag_cropped)
        chain_intervals.append({
            "label_asym_id": cid,
            "chain_type": chain_type,
            "role": "antigen",
            "retained_residues": sorted(retained_ids) if retained_ids else sorted(set(int(r) for r in aa.res_id)),
            "num_contacting_domains": len(contacting),
            "num_removed_domains": len(removed),
        })

    # ---- assemble and write PDB --------------------------------------------
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    all_parts = ab_cropped + ag_cropped_parts
    if not all_parts:
        raise ValueError(f"No atoms left after cropping for complex {complex_id}")

    # Re-number chains to sequential PDB chain IDs.
    for idx, part in enumerate(all_parts):
        cid = _PDB_CHAIN_IDS[idx % len(_PDB_CHAIN_IDS)]
        part.chain_id[:] = cid

    full_atoms = struc.concatenate(all_parts) if len(all_parts) > 1 else all_parts[0]
    stem = f"{pdb_id}_{assembly_id or 'na'}_{complex_id}"
    pdb_path = outdir / f"{stem}_refined.pdb"
    pdb_file = PDBFile()
    pdb_file.set_structure(full_atoms)
    pdb_file.write(pdb_path)

    json_path = outdir / f"{stem}_refined.json"
    result_data: dict[str, Any] = {
        "pdb_id": pdb_id,
        "assembly_id": assembly_id,
        "complex_id": complex_id,
        "source_path": source_path,
        "antibody_unit_type": antibody_unit_type,
        "chain_intervals": chain_intervals,
        "antigen_domains": antigen_domains,
        "removed_antigen_domains": removed_domains,
        "contact_summary": {
            "distance_threshold": contact_distance,
            "louvain_resolution": louvain_resolution,
        },
        "warnings": warnings,
    }
    json_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return RefineAntibodyComplexResult(
        complex_id=complex_id,
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        source_path=source_path,
        antibody_unit_type=antibody_unit_type,
        pdb_path=str(pdb_path),
        json_path=str(json_path),
        chain_intervals=chain_intervals,
        antigen_domains=antigen_domains,
        removed_antigen_domains=removed_domains,
        warnings=warnings,
    )


def refine_antibody_complexes(
    case_dirs: list[str | Path],
    prep_dir: str | Path,
    outdir: str | Path,
    *,
    contact_distance: float = 8.0,
    louvain_resolution: float = 1.0,
    min_domain_size: int = 10,
    min_contact_residues: int = 3,
    show_progress: bool = True,
) -> list[RefineAntibodyComplexResult]:
    """Refine antibody-antigen complexes across one or more case directories."""
    results: list[RefineAntibodyComplexResult] = []
    case_iter = sorted(Path(d).resolve() for d in case_dirs)

    from tqdm import tqdm

    for case_dir in (tqdm(case_iter, desc="Refining AB-AG complexes", unit="case") if show_progress else case_iter):
        chain_inv = _load_chain_inventory(str(case_dir), case_dir)
        if not chain_inv:
            LOGGER.warning("No chain inventory found in %s", case_dir)
            continue

        # Find antibody complex payloads.
        for bundle_path in sorted(case_dir.glob("*.json.gz")):
            try:
                bundle = json.loads(gzip.decompress(bundle_path.read_bytes()))
            except Exception:
                continue
            complexes = bundle.get("antibody_antigen_complexes", [])
            summary = bundle.get("structure_summary", {})
            source_path = summary.get("source_path", "")
            pdb_id = summary.get("pdb_id", "") or case_dir.name

            for idx, cplx in enumerate(complexes):
                if not isinstance(cplx, dict):
                    continue
                try:
                    result = refine_antibody_complex(
                        pdb_id=str(pdb_id),
                        source_path=str(source_path),
                        assembly_id=cplx.get("assembly_id"),
                        complex_id=cplx.get("complex_id", f"ab_ag_{idx + 1:03d}"),
                        antibody_unit_type=str(cplx.get("antibody_unit_type", "")),
                        antibody_chain_ids=[str(c) for c in cplx.get("antibody_chain_ids", [])],
                        antibody_chain_types=[],
                        antigen_chain_ids=[str(c) for c in cplx.get("antigen_chain_ids", [])],
                        antigen_chain_types=cplx.get("antigen_chain_types", []),
                        chain_inventory=chain_inv,
                        prep_dir=prep_dir,
                        outdir=outdir,
                        contact_distance=contact_distance,
                        louvain_resolution=louvain_resolution,
                        min_domain_size=min_domain_size,
                        min_contact_residues=min_contact_residues,
                    )
                    results.append(result)
                except Exception as exc:
                    LOGGER.warning("Failed to refine complex %s-%s: %s", pdb_id, idx + 1, exc)

    return results
