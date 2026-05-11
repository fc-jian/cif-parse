#!/usr/bin/env python3
"""Render before/after comparison images for an antibody-antigen refinement.

Requires PyMOL (``pip install pymol-open-source``).  Not a hard dependency of
cif-parse — this is a standalone visualisation helper.

Usage::

    python -m cif_parse.refine.render_comparison \\
        --case-dir ./outputs/cases/5ywo \\
        --prep-dir ./prep \\
        --assembly-id 2 \\
        --complex-id ab_ag_001 \\
        --outdir ./comparison_images
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def _load_complex(case_dir: Path, assembly_id: str, complex_id: str):
    """Load the parse bundle, chain inventory, and target complex payload."""
    bundle_path = case_dir / f"result_assembly_{assembly_id}.json.gz"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    with gzip.open(bundle_path) as fh:
        bundle = json.load(fh)
    chain_inv = bundle.get("chain_inventory", [])
    complexes = bundle.get("antibody_antigen_complexes", [])
    cplx = next(
        (c for c in complexes if c.get("complex_id") == complex_id),
        None,
    )
    if cplx is None:
        raise ValueError(
            f"Complex {complex_id} not found in {bundle_path}. "
            f"Available: {[c.get('complex_id') for c in complexes]}"
        )
    return bundle, chain_inv, cplx


def _render_object(pdb_path: str, object_name: str, color_ab: str, color_ag: str):
    """Load a PDB into PyMOL, color antibody/antigen chains, cartoon, orient, ray."""
    from pymol import cmd

    cmd.load(pdb_path, object_name)
    # Antibody chains are renumbered to A, B; antigen chains start at C.
    ab_sel = f"{object_name} and chain A+B"
    ag_sel = f"{object_name} and not chain A+B"
    cmd.color(color_ab, ab_sel)
    cmd.color(color_ag, ag_sel)
    cmd.show_as("cartoon")
    cmd.orient(object_name)
    cmd.zoom(object_name, complete=1)


def render_comparison(
    *,
    case_dir: str | Path,
    prep_dir: str | Path,
    assembly_id: str,
    complex_id: str,
    outdir: str | Path,
    width: int = 800,
    height: int = 600,
    dpi: int = 150,
) -> dict[str, str]:
    """Run refinement and render before/after PNGs.

    Returns a dict mapping labels to output PNG paths.
    """
    case_dir = Path(case_dir)
    prep_dir = Path(prep_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle, chain_inv, cplx = _load_complex(case_dir, assembly_id, complex_id)
    summary = bundle.get("structure_summary", {})

    # -- Refine ---------------------------------------------------------------
    from cif_parse.refine.abag_refine import refine_antibody_complex

    refined = refine_antibody_complex(
        pdb_id=str(summary.get("pdb_id", case_dir.name)),
        source_path=str(summary.get("source_path", "")),
        assembly_id=cplx.get("assembly_id"),
        complex_id=cplx.get("complex_id", complex_id),
        antibody_unit_type=str(cplx.get("antibody_unit_type", "")),
        antibody_chain_ids=[str(c) for c in cplx.get("antibody_chain_ids", [])],
        antibody_chain_types=[],
        antigen_chain_ids=[str(c) for c in cplx.get("antigen_chain_ids", [])],
        antigen_chain_types=cplx.get("antigen_chain_types", []),
        chain_inventory=chain_inv,
        prep_dir=str(prep_dir),
        outdir=str(outdir),
    )

    # -- Extract "before" PDB (full complex, no refinement) -------------------
    from biotite.structure.io.pdbx import get_assembly
    from cif_parse.clustering.antibody_complexes import (
        AntibodyComplexObservation,
        extract_antibody_complex_structure,
    )

    obs = AntibodyComplexObservation(
        complex_observation_id=complex_id,
        pdb_id=str(summary.get("pdb_id", "")),
        source_path=str(summary.get("source_path", "")),
        assembly_id=cplx.get("assembly_id"),
        assembly_mode="all",
        complex_id=complex_id,
        antibody_unit_type=str(cplx.get("antibody_unit_type", "")),
        antibody_heavy_chain=cplx.get("antibody_heavy_chain"),
        antibody_heavy_auth_asym_id=cplx.get("antibody_heavy_auth_asym_id"),
        antibody_light_chain=cplx.get("antibody_light_chain"),
        antibody_light_auth_asym_id=cplx.get("antibody_light_auth_asym_id"),
        antibody_chain_ids=[str(c) for c in cplx.get("antibody_chain_ids", [])],
        antibody_auth_asym_ids=cplx.get("antibody_auth_asym_ids", []),
        antigen_chain_ids=[str(c) for c in cplx.get("antigen_chain_ids", [])],
        antigen_auth_asym_ids=cplx.get("antigen_auth_asym_ids", []),
        antigen_chain_types=cplx.get("antigen_chain_types", []),
        auxiliary_component_ids=cplx.get("auxiliary_component_ids", []),
        auxiliary_component_auth_asym_ids=cplx.get("auxiliary_component_auth_asym_ids", []),
        auxiliary_branched_ids=cplx.get("auxiliary_branched_ids", []),
        auxiliary_branched_auth_asym_ids=cplx.get("auxiliary_branched_auth_asym_ids", []),
        structural_auxiliary_chain_ids=[],
        num_antigen_chains=cplx.get("num_antigen_chains", 1),
        num_antibody_antigen_interfaces=cplx.get("num_antibody_antigen_interfaces", 0),
        contact_score=cplx.get("contact_score", 0),
        antibody_member_descriptors=[],
        antigen_member_descriptors=[],
        auxiliary_member_descriptors=[],
        signature_key="",
        num_unclustered_monomer_members=0,
    )
    before_struct = extract_antibody_complex_structure(
        obs, outdir=str(outdir), model=1, drop_hydrogens=True,
    )

    # -- Render ---------------------------------------------------------------
    try:
        from pymol import cmd, finish_launching
        finish_launching(["pymol", "-cq"])
    except Exception as exc:
        raise RuntimeError(
            "PyMOL is required for rendering. Install with: pip install pymol-open-source"
        ) from exc

    cmd.reinitialize()
    cmd.bg_color("white")

    # Load both objects into one session.
    _render_object(before_struct.extracted_pdb_path, "before", "cyan", "lightpink")
    _render_object(refined.pdb_path, "after", "cyan", "salmon")

    # Save session.
    pse_path = str(outdir / f"{complex_id}_comparison.pse")
    cmd.save(pse_path)

    # Render before panel.
    cmd.disable("after")
    cmd.enable("before")
    cmd.viewport(width, height)
    cmd.zoom("before", complete=1)
    cmd.ray(width, height)
    before_png = str(outdir / f"{complex_id}_before_full.png")
    cmd.png(before_png, width, height, dpi=dpi)

    # Render after panel.
    cmd.disable("before")
    cmd.enable("after")
    cmd.zoom("after", complete=1)
    cmd.ray(width, height)
    after_png = str(outdir / f"{complex_id}_after_refined.png")
    cmd.png(after_png, width, height, dpi=dpi)

    cmd.reinitialize()
    return {"before": before_png, "after": after_png, "session": pse_path}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Render before/after AB-AG complex refinement comparison",
    )
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--prep-dir", type=Path, required=True)
    p.add_argument("--assembly-id", default="1")
    p.add_argument("--complex-id", default="ab_ag_001")
    p.add_argument("--outdir", type=Path, default=Path("comparison_images"))
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    try:
        paths = render_comparison(
            case_dir=args.case_dir,
            prep_dir=args.prep_dir,
            assembly_id=args.assembly_id,
            complex_id=args.complex_id,
            outdir=args.outdir,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for label, path in paths.items():
        print(f"{label}: {path} ({Path(path).stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
