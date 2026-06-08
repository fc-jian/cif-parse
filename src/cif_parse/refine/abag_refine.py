"""Antibody-antigen complex refinement: Fv cropping + antigen domain filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from cif_parse.clustering.atom_cache import (
    PklAtomReader,
    load_source_case_dir_map,
    resolve_cases_root,
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

    Uses the structural Fv boundary from immune annotation when available.
    ``cdr_regions`` remain strict SADIE/IMGT intervals; motif-repaired FR4
    ends are exposed separately as ``fv_seq_end``.
    """
    if variable_domains:
        vd = variable_domains[0]
        start = int(vd.get("fv_seq_start") or vd.get("seq_start") or 1)
        end = int(vd.get("fv_seq_end") or vd.get("seq_end") or 0) or None
        return (start, end)

    # Fallback: take the first ~120 residues based on residue ordering.
    res_ids = sorted(set(int(r) for r in atom_array.res_id))
    if not res_ids:
        return (0, 0)
    fv_end_idx = min(120, len(res_ids))
    return (res_ids[0], res_ids[fv_end_idx - 1])


def _fv_boundary_warnings(variable_domains: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for domain in variable_domains:
        domain_warnings = domain.get("boundary_warnings")
        if isinstance(domain_warnings, list):
            warnings.extend(str(warning) for warning in domain_warnings if warning)
        source = str(domain.get("boundary_source") or "")
        if source == "sadie_with_fr4_motif_extension":
            warnings.append("fv_boundary_repaired_by_motif_requires_review")
    return sorted(set(warnings))


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


def _crop_atom_array_to_intervals(
    atom_array: Any,
    intervals: list[tuple[int, int | None]],
) -> Any:
    if not intervals:
        return atom_array.copy()
    mask = np.zeros(atom_array.array_length(), dtype=bool)
    for start, end in intervals:
        if end is None:
            mask |= atom_array.res_id >= start
        else:
            mask |= (atom_array.res_id >= start) & (atom_array.res_id <= end)
    if not mask.any():
        return atom_array.copy()
    return atom_array[mask].copy()


def _domain_interval(domain: dict[str, Any]) -> tuple[int, int | None] | None:
    try:
        start = int(domain.get("fv_seq_start") or domain.get("seq_start") or 1)
    except (TypeError, ValueError):
        return None
    try:
        end_value = domain.get("fv_seq_end") or domain.get("seq_end") or 0
        end = int(end_value) or None
    except (TypeError, ValueError):
        end = None
    return (start, end)


def _domain_chain_code(domain: dict[str, Any]) -> str:
    return str(domain.get("chain_code", "") or "")


def _domain_id(domain: dict[str, Any], index: int) -> str:
    value = domain.get("domain_id")
    return str(value) if value else f"legacy_vd_{index + 1:02d}_{_domain_chain_code(domain)}"


def _domain_records_for_cropping(
    *,
    features: dict[str, Any],
    ab_analysis: dict[str, Any],
    antibody_unit_type: str,
    chain_type: str,
) -> list[dict[str, Any]]:
    domains = ab_analysis.get("antibody_domains") or features.get("antibody_domains")
    if not isinstance(domains, list) or not domains:
        domains = ab_analysis.get("variable_domains", []) or []
    domain_records = [dict(domain) for domain in domains if isinstance(domain, dict)]
    if not domain_records:
        return []

    units = ab_analysis.get("antibody_units") or features.get("antibody_units") or []
    if isinstance(units, list):
        primary_unit_id = (
            ab_analysis.get("primary_antibody_unit_id")
            or features.get("primary_antibody_unit_id")
            or ""
        )
        selected_unit = None
        preferred_unit_types: list[str] = []
        if antibody_unit_type:
            preferred_unit_types.append(antibody_unit_type)
        if chain_type == "antibody heavy chain":
            preferred_unit_types.append("single_heavy_variable_domain")
        elif chain_type == "antibody light chain":
            preferred_unit_types.append("single_light_variable_domain")

        for preferred_unit_type in preferred_unit_types:
            for unit in units:
                if isinstance(unit, dict) and unit.get("unit_type") == preferred_unit_type:
                    selected_unit = unit
                    break
            if selected_unit is not None:
                break
        for unit in units:
            if not isinstance(unit, dict):
                continue
            if selected_unit is None and primary_unit_id and unit.get("unit_id") == primary_unit_id:
                selected_unit = unit
                break
        if selected_unit is not None:
            selected_ids = {str(item) for item in selected_unit.get("domain_ids", [])}
            matched = [
                domain
                for index, domain in enumerate(domain_records)
                if _domain_id(domain, index) in selected_ids
            ]
            if matched:
                return matched

    if antibody_unit_type == "scFv":
        return [
            domain
            for domain in domain_records
            if _domain_chain_code(domain) in {"H", "K", "L"}
        ]
    if antibody_unit_type == "VHH":
        heavy_domains = [domain for domain in domain_records if _domain_chain_code(domain) == "H"]
        return heavy_domains or domain_records[:1]
    if chain_type == "antibody heavy chain":
        heavy_domains = [domain for domain in domain_records if _domain_chain_code(domain) == "H"]
        return heavy_domains[:1] or domain_records[:1]
    if chain_type == "antibody light chain":
        light_domains = [domain for domain in domain_records if _domain_chain_code(domain) in {"K", "L"}]
        return light_domains[:1] or domain_records[:1]
    return domain_records[:1]


# ---------------------------------------------------------------------------
# Antigen contact graph + domain detection
# ---------------------------------------------------------------------------

ResidueKey = Any


def _ca_coords(atom_array: Any, chain_id: str | None = None) -> dict[ResidueKey, np.ndarray]:
    """Extract per-residue Cα coordinates (or P for nucleic acids)."""
    coords: dict[ResidueKey, np.ndarray] = {}
    for i in range(atom_array.array_length()):
        aname = str(atom_array.atom_name[i]).strip()
        if aname in ("CA", "P"):
            rid = int(atom_array.res_id[i])
            key: ResidueKey = (chain_id, rid) if chain_id is not None else rid
            if key not in coords:
                coords[key] = np.asarray(atom_array.coord[i], dtype=np.float32)
    return coords


def _residue_contacts(
    ab_coords: dict[ResidueKey, np.ndarray],
    ag_coords: dict[ResidueKey, np.ndarray],
    distance: float = 8.0,
) -> list[tuple[ResidueKey, ResidueKey, float]]:
    """Return antibody–antigen residue pairs within *distance* (Å)."""
    if not ab_coords or not ag_coords:
        return []
    ab_ids = list(ab_coords)
    ag_ids = list(ag_coords)
    ab_mat = np.asarray([ab_coords[r] for r in ab_ids], dtype=np.float32)
    ag_mat = np.asarray([ag_coords[r] for r in ag_ids], dtype=np.float32)
    tree = cKDTree(ag_mat)
    pairs: list[tuple[ResidueKey, ResidueKey, float]] = []
    for i, neighbors in enumerate(tree.query_ball_point(ab_mat, r=distance)):
        rid_ab = ab_ids[i]
        for j in neighbors:
            rid_ag = ag_ids[int(j)]
            d = float(np.linalg.norm(ab_mat[i] - ag_mat[int(j)]))
            pairs.append((rid_ab, rid_ag, d))
    return pairs


def _ag_internal_graph(
    ag_coords: dict[ResidueKey, np.ndarray],
    distance: float = 8.0,
) -> dict[ResidueKey, set[ResidueKey]]:
    """Build an adjacency dict for antigen residues within *distance*."""
    if not ag_coords:
        return {}
    ids = list(ag_coords)
    mat = np.asarray([ag_coords[r] for r in ids], dtype=np.float32)
    adj: dict[ResidueKey, set[ResidueKey]] = {rid: set() for rid in ids}
    tree = cKDTree(mat)
    for i, neighbors in enumerate(tree.query_ball_point(mat, r=distance)):
        rid_i = ids[i]
        for j in neighbors:
            j = int(j)
            if i < j:
                rid_j = ids[j]
                adj[rid_i].add(rid_j)
                adj[rid_j].add(rid_i)
    return adj


def _identify_antigen_domains(
    ag_coords: dict[ResidueKey, np.ndarray],
    ab_ag_contacts: set[ResidueKey],
    *,
    distance: float = 8.0,
    louvain_resolution: float = 1.0,
    min_domain_size: int = 10,
    min_contact_residues: int = 3,
) -> tuple[list[list[ResidueKey]], list[list[ResidueKey]]]:
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

    contacting: list[list[ResidueKey]] = []
    removed: list[list[ResidueKey]] = []
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
    pkl_reader = PklAtomReader(
        resolve_cases_root(prep_dir),
        load_source_case_dir_map(prep_dir),
    )
    warnings: list[str] = []

    # ---- load per-chain atom arrays ----------------------------------------
    chain_arrays: dict[str, Any] = {}
    for cid in antibody_chain_ids + antigen_chain_ids:
        aa = None
        for aid in (assembly_id, None):
            aa = pkl_reader.load_chain(
                source_path,
                cid,
                assembly_id=aid,
                filter_hetero=False,
            )
            if aa is not None:
                break
        if aa is None:
            warnings.append(f"chain {cid} atom array not found in prep")
            continue
        chain_arrays[cid] = aa

    if not chain_arrays:
        raise ValueError(f"No chain atom arrays loaded for complex {complex_id}")

    # ---- antibody Fv cropping ----------------------------------------------
    ab_cropped: list[Any] = []
    ab_res_ca: dict[int, np.ndarray] = {}
    chain_intervals: list[dict[str, Any]] = []
    # Track mapping: new PDB chain id → original label_asym_id
    _pdb_chain_ids = _PDB_CHAIN_IDS
    _chain_idx = 0

    def _residue_intervals(res_ids: np.ndarray) -> list[tuple[int, int]]:
        """Convert sorted unique residue IDs to compact interval notation."""
        uniq = sorted(set(int(r) for r in res_ids))
        if not uniq:
            return []
        intervals: list[tuple[int, int]] = []
        start = end = uniq[0]
        for r in uniq[1:]:
            if r == end + 1:
                end = r
            else:
                intervals.append((start, end))
                start = end = r
        intervals.append((start, end))
        return intervals

    for cid in antibody_chain_ids:
        aa = chain_arrays.get(cid)
        if aa is None:
            continue
        ann = _find_chain_annotation(chain_inventory, cid)
        features = ann.get("features", {})
        ab_analysis = features.get("antibody_analysis", {}) or {}
        variable_domains = ab_analysis.get("variable_domains", []) or []
        chain_type = ann.get("chain_type", "")
        selected_domains = _domain_records_for_cropping(
            features=features,
            ab_analysis=ab_analysis,
            antibody_unit_type=antibody_unit_type,
            chain_type=chain_type,
        )
        selected_intervals = [
            interval
            for domain in selected_domains
            if (interval := _domain_interval(domain)) is not None
        ]

        if selected_intervals:
            cropped = _crop_atom_array_to_intervals(aa, selected_intervals)
            for warning in _fv_boundary_warnings(selected_domains):
                warning_text = f"chain {cid}: {warning}"
                if warning_text not in warnings:
                    warnings.append(warning_text)
            fv_boundary_source = ",".join(
                sorted({str(domain.get("boundary_source") or "sadie") for domain in selected_domains})
            )
            fv_boundary_confidence = ",".join(
                sorted({str(domain.get("boundary_confidence") or "high") for domain in selected_domains})
            )
        else:
            start, end = _find_fv_boundaries(aa, variable_domains, chain_type)
            for warning in _fv_boundary_warnings(variable_domains):
                warning_text = f"chain {cid}: {warning}"
                if warning_text not in warnings:
                    warnings.append(warning_text)
            cropped = _crop_atom_array(aa, start, end)
            fv_boundary_source = (
                str(variable_domains[0].get("boundary_source") or "sadie")
                if variable_domains
                else "fallback_first_120_residues"
            )
            fv_boundary_confidence = (
                str(variable_domains[0].get("boundary_confidence") or "high")
                if variable_domains
                else "low"
            )
        pdb_chain = _pdb_chain_ids[_chain_idx % len(_pdb_chain_ids)]
        ab_cropped.append(cropped)
        ab_res_ca.update(_ca_coords(cropped, chain_id=cid))
        chain_intervals.append({
            "label_asym_id": cid,
            "pdb_chain_id": pdb_chain,
            "chain_type": chain_type,
            "role": "antibody",
            "fv_boundary_source": fv_boundary_source,
            "fv_boundary_confidence": fv_boundary_confidence,
            "retained_antibody_domain_ids": [
                _domain_id(domain, index) for index, domain in enumerate(selected_domains)
            ],
            "retained_residue_intervals": _residue_intervals(cropped.res_id),
        })
        _chain_idx += 1

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

        ag_coords = _ca_coords(aa, chain_id=cid)
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
        if not ag_coords:
            warnings.append(f"chain {cid}: no CA/P coordinates for antigen domain filtering; kept full antigen")
        elif not contacts:
            warnings.append(f"chain {cid}: no antibody-antigen residue contacts within {contact_distance:g} A; kept full antigen")
        elif not contacting:
            warnings.append(f"chain {cid}: no contacting antigen domain passed filters; kept full antigen")

        for domain in contacting:
            antigen_domains.append({
                "label_asym_id": cid,
                "chain_type": chain_type,
                "residue_ids": [int(res_id) for _, res_id in domain],
                "num_residues": len(domain),
                "num_antibody_contacts": len(set(domain) & contact_residues),
            })
        for domain in removed:
            removed_domains.append({
                "label_asym_id": cid,
                "chain_type": chain_type,
                "residue_ids": [int(res_id) for _, res_id in domain],
                "num_residues": len(domain),
                "num_antibody_contacts": len(set(domain) & contact_residues),
            })

        # Crop antigen chain to retained residue IDs.
        retained_ids: set[int] = set()
        for domain_residues in contacting:
            retained_ids.update(int(res_id) for _, res_id in domain_residues)
        if retained_ids:
            keep_mask = np.zeros(aa.array_length(), dtype=bool)
            for i in range(aa.array_length()):
                if int(aa.res_id[i]) in retained_ids:
                    keep_mask[i] = True
            ag_cropped = aa[keep_mask].copy()
        else:
            ag_cropped = aa.copy()
        ag_cropped_parts.append(ag_cropped)
        pdb_chain = _pdb_chain_ids[_chain_idx % len(_pdb_chain_ids)]
        chain_intervals.append({
            "label_asym_id": cid,
            "pdb_chain_id": pdb_chain,
            "chain_type": chain_type,
            "role": "antigen",
            "retained_residue_intervals": _residue_intervals(ag_cropped.res_id),
            "num_contacting_domains": len(contacting),
            "num_removed_domains": len(removed),
        })
        _chain_idx += 1

    # ---- assemble and write PDB --------------------------------------------
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    all_parts = ab_cropped + ag_cropped_parts
    if not all_parts:
        raise ValueError(f"No atoms left after cropping for complex {complex_id}")

    # Re-number chains using the tracked PDB chain IDs.
    for idx, part in enumerate(all_parts):
        if idx < len(chain_intervals):
            part.chain_id[:] = chain_intervals[idx]["pdb_chain_id"]
        else:
            part.chain_id[:] = _PDB_CHAIN_IDS[idx % len(_PDB_CHAIN_IDS)]

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
