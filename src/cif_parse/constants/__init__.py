"""Project-wide constants.

This module collects hard-coded constants used across the package so that
they live in a single place.  Values that are exposed through
`AppSettings` / CLI are still declared here as fall-backs; callers that
receive a settings object should prefer the setting over the raw constant.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Chain-type taxonomies (used for classification and routing)
# ---------------------------------------------------------------------------

PROTEIN_CHAIN_TYPES = frozenset(
    {
        "antibody heavy chain",
        "antibody light chain",
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
        "other protein chain",
    }
)

NUCLEIC_ACID_CHAIN_TYPES = frozenset({"DNA chain", "RNA chain", "other nucleic acid chain"})
BRANCHED_CHAIN_TYPES = frozenset({"glycan / branched component"})
METAL_CHAIN_TYPES = frozenset({"metal ion"})
SMALL_MOLECULE_CHAIN_TYPES = frozenset({"small molecule compound"})

POLYMER_CHAIN_TYPES = frozenset(
    {
        "antibody heavy chain",
        "antibody light chain",
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
        "other protein chain",
        "DNA chain",
        "RNA chain",
        "other nucleic acid chain",
        "other polymer chain",
    }
)

ANTIBODY_CHAIN_TYPES = frozenset({"antibody heavy chain", "antibody light chain"})

TCR_PMHC_CHAIN_TYPES = frozenset(
    {
        "TCR chain",
        "MHC heavy chain",
        "beta2m or auxiliary immune chain",
        "peptide antigen",
    }
)

# TCR-pMHC specific labels
TCR_CHAIN_TYPE = "TCR chain"
MHC_CHAIN_TYPE = "MHC heavy chain"
AUX_CHAIN_TYPE = "beta2m or auxiliary immune chain"
PEPTIDE_CHAIN_TYPE = "peptide antigen"

# ---------------------------------------------------------------------------
# Residue / compound identity
# ---------------------------------------------------------------------------

STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
STANDARD_RNA_NUCLEOTIDES = frozenset({"A", "C", "G", "U", "I"})
STANDARD_DNA_NUCLEOTIDES = frozenset({"DA", "DC", "DG", "DT", "DI", "DU"})

# ---------------------------------------------------------------------------
# Interface / contact geometry defaults
# ---------------------------------------------------------------------------

RESIDUE_CONTACT_CUTOFF = 8.0
ATOM_CONTACT_CUTOFF = 5.0
MIN_RESIDUE_CONTACTS = 3
MIN_ATOM_CONTACTS = 20
ATOM_CHUNK_SIZE = 256

# ---------------------------------------------------------------------------
# TCR / pMHC assembly defaults
# ---------------------------------------------------------------------------

PEPTIDE_MAX_LENGTH = 30
TCR_PAIR_TYPES = {
    frozenset({"alpha", "beta"}): "alpha_beta",
    frozenset({"gamma", "delta"}): "gamma_delta",
}

# ---------------------------------------------------------------------------
# Immune annotation defaults
# ---------------------------------------------------------------------------

SCFV_LINKER_MOTIFS = ("GGGGS", "GGGGSGGGGS", "GGGSGGGG", "SSGGGGSGGGG")
ANTIBODY_DESCRIPTION_MARKERS = ("antibody", "immunoglobulin", "fab", "fv", "nanobody", "vhh")
TCR_DESCRIPTION_MARKERS = ("t cell receptor", "t-cell receptor", "tcr")
CAMELID_SPECIES = frozenset({"alpaca", "llama", "camel", "camelid"})
SADIE_CHAIN_CODES = ("H", "K", "L", "A", "B", "G", "D")
SADIE_SCHEME = "imgt"
SADIE_REGION_DEFINITION = "imgt"
SADIE_DOMAIN_BITSCORE_THRESHOLD = 80.0
SADIE_DOMAIN_LIMIT = 4
IMGT_CDR_RANGES = {
    "cdr1": (27, 38),
    "cdr2": (56, 65),
    "cdr3": (105, 117),
}
CHAIN_CODE_DESCRIPTION_HINTS = {
    "H": ("heavy chain",),
    "K": ("kappa",),
    "L": ("lambda",),
    "A": ("alpha",),
    "B": ("beta",),
    "G": ("gamma",),
    "D": ("delta",),
}

# ---------------------------------------------------------------------------
# Reporting / review defaults
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_ANTIBODY_THRESHOLD = 0.8
HTML_SKIPPED_TARGET_DETAIL_LIMIT = 100
COVERAGE_WARNING_CODES = frozenset(
    {
        "coverage skipped because atom array extraction failed",
        "coverage skipped because the chain has no coordinates",
        "coverage assigned to multiple main chains",
        "coverage owner not found within the nearest-distance threshold",
    }
)

# ---------------------------------------------------------------------------
# Generic VdW radii for element-based SASA fallback (Å)
# ---------------------------------------------------------------------------

GENERIC_VDW_RADII = {
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "SE": 1.90,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SI": 2.10,
    "NA": 2.27,
    "MG": 1.73,
    "K": 2.75,
    "CA": 2.31,
    "MN": 1.97,
    "FE": 1.94,
    "CO": 1.92,
    "NI": 1.63,
    "CU": 1.40,
    "ZN": 1.39,
    "CD": 1.58,
    "HG": 1.55,
}
