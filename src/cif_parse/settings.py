"""Application-wide settings and supported option enums."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


SUPPORTED_FORMATS = frozenset({"json", "csv"})
SUPPORTED_ASSEMBLY_MODES = frozenset({"largest_assembly", "first_assembly", "asymmetric_unit", "all"})
ASSEMBLY_MODES_REQUIRING_ASSEMBLY_ID = frozenset({"largest_assembly", "first_assembly", "all"})
SUPPORTED_COVERAGE_MODES = frozenset({"nearest", "contact", "covalent"})
SUPPORTED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
SUPPORTED_CLUSTERING_SEQUENCE_MODES = frozenset({"skip", "exact", "mmseqs2"})
SUPPORTED_CLUSTERING_STRUCTURE_MODES = frozenset({"skip", "greedy"})
SUPPORTED_CLUSTERING_OBJECT_MODES = frozenset({"skip", "signature"})

DEFAULT_CONFIG_PATH = Path("config.toml")
DEFAULT_CLUSTERING_CONFIG_PATH = Path("config_clustering.toml")
DEFAULT_SINGLE_OUTDIR = Path("outputs")
DEFAULT_BATCH_OUTDIR = Path("batch_outputs")
DEFAULT_CLUSTERING_OUTDIR = Path("cluster_outputs")

_DEFAULT_JOB_COUNT = max(1, os.cpu_count() or 1)

# Resolved once at import time so every module sees the same directory.
_FAST_TEMP_ROOT: Path | None = None


def normalize_cutoff_date(value: Any) -> str | None:
    """Normalize optional clustering cutoff dates to strict ``YYYY-MM-DD`` strings."""

    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        raise ValueError("clustering.cutoff_date must use YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("clustering.cutoff_date must use YYYY-MM-DD format") from exc
    return parsed.isoformat()


def _resolve_fast_temp_root() -> Path | None:
    """Pick the best transient-storage root available on this machine.

    Precedence:
    1. ``CIF_PARSE_TMPDIR`` environment variable (explicit user choice).
    2. ``$TMPDIR`` or ``/tmp``.
    3. Current directory as fallback.
    """
    env = os.environ.get("CIF_PARSE_TMPDIR")
    if env:
        p = Path(env)
        if p.is_dir() and os.access(p, os.W_OK):
            return p
    for candidate in (os.environ.get("TMPDIR"), "/tmp"):
        if candidate:
            p = Path(candidate)
            if p.is_dir() and os.access(p, os.W_OK):
                return p
    return None


def get_fast_temp_dir(suffix: str = "") -> Path:
    """Return a writable directory suitable for transient large I/O.

    The directory is created inside the best available temp root
    (see :func:`_resolve_fast_temp_root`) with a PID-prefixed name so
    that concurrent processes do not collide.  Callers may supply an
    optional *suffix* for human-readable identification.
    """
    global _FAST_TEMP_ROOT
    if _FAST_TEMP_ROOT is None:
        _FAST_TEMP_ROOT = _resolve_fast_temp_root()
    root = _FAST_TEMP_ROOT or Path("/tmp")
    name = f".cif_parse_{os.getpid()}"
    if suffix:
        name = f"{name}_{suffix}"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(slots=True)
class AppSettings:
    """Runtime settings shared by single-file and batch execution paths."""

    output_format: str = "json"
    assembly_mode: str = "asymmetric_unit"
    input_assembly: bool = False
    metadata_cif_dir: str = ""
    metadata_table: str = ""
    coverage_mode: str = "nearest"
    debug: bool = False
    log_level: str = "DEBUG"
    verbose: bool = False
    model: int = 1
    use_author_fields: bool = False
    drop_hydrogens_for_analysis: bool = True
    max_polymer_chains: int = 100
    max_assembly_atoms: int = 300_000
    assembly_jobs: int = 1
    min_polymer_chain_length: int = 20
    tight_multimer_min_buried_area: float = 500.0
    tight_multimer_louvain_resolution: float = 1.0
    tight_multimer_min_member_instances: int = 2
    tight_multimer_large_component_warning_size: int = 8

    # Contact / interface geometry
    residue_contact_cutoff: float = 8.0
    atom_contact_cutoff: float = 5.0
    min_residue_contacts: int = 3
    min_atom_contacts: int = 20

    # Immune annotation thresholds
    peptide_max_length: int = 30
    sadie_domain_bitscore_threshold: float = 80.0
    sadie_domain_limit: int = 4
    low_confidence_antibody_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.output_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        if self.assembly_mode not in SUPPORTED_ASSEMBLY_MODES:
            raise ValueError(f"Unsupported assembly mode: {self.assembly_mode}")
        if self.coverage_mode not in SUPPORTED_COVERAGE_MODES:
            raise ValueError(f"Unsupported coverage mode: {self.coverage_mode}")
        if self.log_level.upper() not in SUPPORTED_LOG_LEVELS:
            raise ValueError(f"Unsupported log level: {self.log_level}")
        self.log_level = self.log_level.upper()
        if self.model < 1:
            raise ValueError("model must be >= 1")
        if self.max_polymer_chains < 1:
            raise ValueError("max_polymer_chains must be >= 1")
        if self.max_assembly_atoms < 1:
            raise ValueError("max_assembly_atoms must be >= 1")
        if self.assembly_jobs < 1:
            raise ValueError("assembly_jobs must be >= 1")
        if self.min_polymer_chain_length < 0:
            raise ValueError("min_polymer_chain_length must be >= 0")
        if self.tight_multimer_min_buried_area < 0:
            raise ValueError("tight_multimer_min_buried_area must be >= 0")
        if self.tight_multimer_louvain_resolution <= 0:
            raise ValueError("tight_multimer_louvain_resolution must be > 0")
        if self.tight_multimer_min_member_instances < 2:
            raise ValueError("tight_multimer_min_member_instances must be >= 2")
        if self.tight_multimer_large_component_warning_size < 2:
            raise ValueError("tight_multimer_large_component_warning_size must be >= 2")
        if self.residue_contact_cutoff <= 0:
            raise ValueError("residue_contact_cutoff must be > 0")
        if self.atom_contact_cutoff <= 0:
            raise ValueError("atom_contact_cutoff must be > 0")
        if self.min_residue_contacts < 0:
            raise ValueError("min_residue_contacts must be >= 0")
        if self.min_atom_contacts < 0:
            raise ValueError("min_atom_contacts must be >= 0")
        if self.peptide_max_length < 1:
            raise ValueError("peptide_max_length must be >= 1")
        if self.sadie_domain_bitscore_threshold < 0:
            raise ValueError("sadie_domain_bitscore_threshold must be >= 0")
        if self.sadie_domain_limit < 1:
            raise ValueError("sadie_domain_limit must be >= 1")
        if self.low_confidence_antibody_threshold < 0 or self.low_confidence_antibody_threshold > 1:
            raise ValueError("low_confidence_antibody_threshold must be in [0, 1]")


@dataclass(slots=True)
class ClusteringSettings:
    """Runtime settings for the independent clustering CLI."""

    outdir: Path | str = DEFAULT_CLUSTERING_OUTDIR
    protein_sequence_mode: str = "mmseqs2"
    protein_structure_mode: str = "greedy"
    protein_subcluster_by_sequence: bool = True
    cutoff_date: str | None = None
    dimer_mode: str = "signature"
    dimer_structure_mode: str = "greedy"
    dimer_tm_score_threshold: float = 0.50
    dimer_interface_residue_cutoff: float = 8.0
    multimer_mode: str = "signature"
    multimer_structure_mode: str = "greedy"
    multimer_tm_score_threshold: float = 0.50
    multimer_max_atoms_for_refinement: int = 10_000
    antibody_complex_mode: str = "signature"
    antibody_complex_structure_mode: str = "greedy"
    antibody_complex_tm_score_threshold: float = 0.50
    antibody_complex_interface_residue_cutoff: float = 20.0
    tcr_complex_mode: str = "signature"
    tcr_complex_structure_mode: str = "greedy"
    tcr_complex_tm_score_threshold: float = 0.50
    protein_min_seq_id: float = 0.40
    protein_coverage: float = 0.80
    protein_cov_mode: int = 5
    model: int = 1
    keep_hydrogens: bool = False
    tm_score_threshold: float = 0.50
    min_alignment_coverage_ratio: float = 0.50
    usalign_executable: str = "USalign"
    jobs: int = _DEFAULT_JOB_COUNT
    mmseqs_threads: int | None = None
    sequence_cluster_jobs: int | None = None
    dimer_extraction_jobs: int | None = None
    usalign_jobs: int | None = None
    log_level: str = "DEBUG"

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)
        if self.protein_sequence_mode not in SUPPORTED_CLUSTERING_SEQUENCE_MODES:
            raise ValueError(f"Unsupported protein_sequence_mode: {self.protein_sequence_mode}")
        if self.protein_structure_mode not in SUPPORTED_CLUSTERING_STRUCTURE_MODES:
            raise ValueError(f"Unsupported protein_structure_mode: {self.protein_structure_mode}")
        self.cutoff_date = normalize_cutoff_date(self.cutoff_date)
        for field_name in (
            "dimer_structure_mode",
            "multimer_structure_mode",
            "antibody_complex_structure_mode",
            "tcr_complex_structure_mode",
        ):
            value = getattr(self, field_name)
            if value not in SUPPORTED_CLUSTERING_STRUCTURE_MODES:
                raise ValueError(f"Unsupported {field_name}: {value}")
        for field_name in ("dimer_mode", "multimer_mode", "antibody_complex_mode", "tcr_complex_mode"):
            value = getattr(self, field_name)
            if value not in SUPPORTED_CLUSTERING_OBJECT_MODES:
                raise ValueError(f"Unsupported {field_name}: {value}")
        if self.log_level.upper() not in SUPPORTED_LOG_LEVELS:
            raise ValueError(f"Unsupported clustering log_level: {self.log_level}")
        self.log_level = self.log_level.upper()
        if self.model < 1:
            raise ValueError("clustering.model must be >= 1")
        if self.dimer_tm_score_threshold < 0 or self.dimer_tm_score_threshold > 1:
            raise ValueError("clustering.dimer_tm_score_threshold must be in [0, 1]")
        if self.dimer_interface_residue_cutoff <= 0:
            raise ValueError("clustering.dimer_interface_residue_cutoff must be > 0")
        if self.multimer_tm_score_threshold < 0 or self.multimer_tm_score_threshold > 1:
            raise ValueError("clustering.multimer_tm_score_threshold must be in [0, 1]")
        if self.multimer_max_atoms_for_refinement < 1:
            raise ValueError("clustering.multimer_max_atoms_for_refinement must be >= 1")
        if self.antibody_complex_tm_score_threshold < 0 or self.antibody_complex_tm_score_threshold > 1:
            raise ValueError("clustering.antibody_complex_tm_score_threshold must be in [0, 1]")
        if self.antibody_complex_interface_residue_cutoff <= 0:
            raise ValueError("clustering.antibody_complex_interface_residue_cutoff must be > 0")
        if self.tcr_complex_tm_score_threshold < 0 or self.tcr_complex_tm_score_threshold > 1:
            raise ValueError("clustering.tcr_complex_tm_score_threshold must be in [0, 1]")
        if self.protein_min_seq_id < 0 or self.protein_min_seq_id > 1:
            raise ValueError("clustering.protein_min_seq_id must be in [0, 1]")
        if self.protein_coverage < 0 or self.protein_coverage > 1:
            raise ValueError("clustering.protein_coverage must be in [0, 1]")
        if self.protein_cov_mode < 0:
            raise ValueError("clustering.protein_cov_mode must be >= 0")
        if self.tm_score_threshold < 0 or self.tm_score_threshold > 1:
            raise ValueError("clustering.tm_score_threshold must be in [0, 1]")
        if self.min_alignment_coverage_ratio < 0 or self.min_alignment_coverage_ratio > 1:
            raise ValueError("clustering.min_alignment_coverage_ratio must be in [0, 1]")
        if not self.usalign_executable:
            raise ValueError("clustering.usalign_executable must not be empty")
        if self.jobs < 1:
            raise ValueError("clustering.jobs must be >= 1")
        if self.mmseqs_threads is not None and self.mmseqs_threads < 1:
            raise ValueError("clustering.mmseqs_threads must be >= 1")
        if self.sequence_cluster_jobs is not None and self.sequence_cluster_jobs < 1:
            raise ValueError("clustering.sequence_cluster_jobs must be >= 1")
        if self.dimer_extraction_jobs is not None and self.dimer_extraction_jobs < 1:
            raise ValueError("clustering.dimer_extraction_jobs must be >= 1")
        if self.usalign_jobs is not None and self.usalign_jobs < 1:
            raise ValueError("clustering.usalign_jobs must be >= 1")


def default_cli_config() -> dict[str, Any]:
    """Return the built-in CLI defaults before loading config.toml."""

    return {
        "settings": {
            "output_format": "json",
            "assembly_mode": "asymmetric_unit",
            "input_assembly": False,
            "metadata_cif_dir": "",
            "metadata_table": "",
            "coverage_mode": "nearest",
            "debug": False,
            "log_level": "DEBUG",
            "verbose": False,
            "model": 1,
            "use_author_fields": False,
            "drop_hydrogens_for_analysis": True,
            "max_polymer_chains": 100,
            "max_assembly_atoms": 300_000,
            "assembly_jobs": 1,
            "min_polymer_chain_length": 20,
            "tight_multimer_min_buried_area": 500.0,
            "tight_multimer_louvain_resolution": 1.0,
            "tight_multimer_min_member_instances": 2,
            "tight_multimer_large_component_warning_size": 8,
            "residue_contact_cutoff": 8.0,
            "atom_contact_cutoff": 5.0,
            "min_residue_contacts": 3,
            "min_atom_contacts": 20,
            "peptide_max_length": 30,
            "sadie_domain_bitscore_threshold": 80.0,
            "sadie_domain_limit": 4,
            "low_confidence_antibody_threshold": 0.8,
        },
        "single": {
            "outdir": DEFAULT_SINGLE_OUTDIR,
        },
        "batch": {
            "outdir": DEFAULT_BATCH_OUTDIR,
            "jobs": _DEFAULT_JOB_COUNT,
            "fail_fast": False,
        },
        "clustering": {
            "outdir": DEFAULT_CLUSTERING_OUTDIR,
            "protein_sequence_mode": "mmseqs2",
            "protein_structure_mode": "greedy",
            "protein_subcluster_by_sequence": True,
            "cutoff_date": None,
            "dimer_mode": "signature",
            "dimer_structure_mode": "greedy",
            "dimer_tm_score_threshold": 0.50,
            "dimer_interface_residue_cutoff": 8.0,
            "multimer_mode": "signature",
            "multimer_structure_mode": "greedy",
            "multimer_tm_score_threshold": 0.50,
            "multimer_max_atoms_for_refinement": 10_000,
            "antibody_complex_mode": "signature",
            "antibody_complex_structure_mode": "greedy",
            "antibody_complex_tm_score_threshold": 0.50,
            "antibody_complex_interface_residue_cutoff": 20.0,
            "tcr_complex_mode": "signature",
            "tcr_complex_structure_mode": "greedy",
            "tcr_complex_tm_score_threshold": 0.50,
            "protein_min_seq_id": 0.40,
            "protein_coverage": 0.80,
            "protein_cov_mode": 5,
            "model": 1,
            "keep_hydrogens": False,
            "tm_score_threshold": 0.50,
            "min_alignment_coverage_ratio": 0.50,
            "usalign_executable": "USalign",
            "jobs": _DEFAULT_JOB_COUNT,
            "log_level": "DEBUG",
        },
    }


def load_cli_config(config_path: str | Path | None = None) -> tuple[Path | None, dict[str, Any]]:
    """Load `config.toml` if *config_path* is explicitly provided; otherwise return built-in defaults.

    The caller is responsible for passing ``--config`` from the CLI.  When the user
    does not supply ``--config`` the function simply returns the hard-coded defaults
    without attempting any file-system auto-discovery.
    """

    if config_path is None:
        return None, default_cli_config()

    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"config file not found: {resolved_path}")

    config = default_cli_config()
    parsed = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    _merge_toml_config(config, parsed)
    return resolved_path, config


def load_clustering_cli_config(config_path: str | Path | None = None) -> tuple[Path | None, dict[str, Any]]:
    """Load `config_clustering.toml` if *config_path* is explicitly provided; otherwise return built-in defaults.

    The caller is responsible for passing ``--config`` from the CLI.  When the user
    does not supply ``--config`` the function simply returns the hard-coded defaults
    without attempting any file-system auto-discovery.
    """

    if config_path is None:
        return None, default_cli_config()

    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"config file not found: {resolved_path}")

    config = default_cli_config()
    parsed = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    _merge_toml_config(config, parsed)
    return resolved_path, config


def _merge_toml_config(config: dict[str, Any], parsed: dict[str, Any]) -> None:
    """Merge TOML config data into the CLI defaults with basic validation."""

    allowed_sections = {"settings", "single", "batch", "clustering"}
    unknown_sections = sorted(set(parsed) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(unknown_sections)}")

    settings_table = parsed.get("settings", {})
    single_table = parsed.get("single", {})
    batch_table = parsed.get("batch", {})
    clustering_table = parsed.get("clustering", {})
    _require_mapping("settings", settings_table)
    _require_mapping("single", single_table)
    _require_mapping("batch", batch_table)
    _require_mapping("clustering", clustering_table)

    _merge_section(
        config["settings"],
        _normalize_legacy_setting_aliases(settings_table),
        {
            "output_format",
            "assembly_mode",
            "input_assembly",
            "metadata_cif_dir",
            "metadata_table",
            "coverage_mode",
            "debug",
            "log_level",
            "verbose",
            "model",
            "use_author_fields",
            "drop_hydrogens_for_analysis",
            "max_polymer_chains",
            "max_assembly_atoms",
            "assembly_jobs",
            "min_polymer_chain_length",
            "tight_multimer_min_buried_area",
            "tight_multimer_louvain_resolution",
            "tight_multimer_min_member_instances",
            "tight_multimer_large_component_warning_size",
            "residue_contact_cutoff",
            "atom_contact_cutoff",
            "min_residue_contacts",
            "min_atom_contacts",
            "peptide_max_length",
            "sadie_domain_bitscore_threshold",
            "sadie_domain_limit",
            "low_confidence_antibody_threshold",
        },
        "settings",
    )
    _merge_section(config["single"], single_table, {"outdir"}, "single")
    _merge_section(config["batch"], batch_table, {"outdir", "jobs", "fail_fast"}, "batch")
    _merge_section(
        config["clustering"],
        clustering_table,
        {
            "outdir",
            "protein_sequence_mode",
            "protein_structure_mode",
            "protein_subcluster_by_sequence",
            "cutoff_date",
            "dimer_mode",
            "dimer_structure_mode",
            "dimer_tm_score_threshold",
            "dimer_interface_residue_cutoff",
            "multimer_mode",
            "multimer_structure_mode",
            "multimer_tm_score_threshold",
            "multimer_max_atoms_for_refinement",
            "antibody_complex_mode",
            "antibody_complex_structure_mode",
            "antibody_complex_tm_score_threshold",
            "antibody_complex_interface_residue_cutoff",
            "tcr_complex_mode",
            "tcr_complex_structure_mode",
            "tcr_complex_tm_score_threshold",
            "protein_min_seq_id",
            "protein_coverage",
            "protein_cov_mode",
            "model",
            "keep_hydrogens",
            "tm_score_threshold",
            "min_alignment_coverage_ratio",
            "usalign_executable",
            "jobs",
            "mmseqs_threads",
            "sequence_cluster_jobs",
            "dimer_extraction_jobs",
            "usalign_jobs",
            "log_level",
        },
        "clustering",
    )

    validated_settings = AppSettings(**config["settings"])
    config["settings"] = {
        "output_format": validated_settings.output_format,
        "assembly_mode": validated_settings.assembly_mode,
        "input_assembly": validated_settings.input_assembly,
        "metadata_cif_dir": validated_settings.metadata_cif_dir,
        "metadata_table": validated_settings.metadata_table,
        "coverage_mode": validated_settings.coverage_mode,
        "debug": validated_settings.debug,
        "log_level": validated_settings.log_level,
        "verbose": validated_settings.verbose,
        "model": validated_settings.model,
        "use_author_fields": validated_settings.use_author_fields,
        "drop_hydrogens_for_analysis": validated_settings.drop_hydrogens_for_analysis,
        "max_polymer_chains": validated_settings.max_polymer_chains,
        "max_assembly_atoms": validated_settings.max_assembly_atoms,
        "assembly_jobs": validated_settings.assembly_jobs,
        "min_polymer_chain_length": validated_settings.min_polymer_chain_length,
        "tight_multimer_min_buried_area": validated_settings.tight_multimer_min_buried_area,
        "tight_multimer_louvain_resolution": validated_settings.tight_multimer_louvain_resolution,
        "tight_multimer_min_member_instances": validated_settings.tight_multimer_min_member_instances,
        "tight_multimer_large_component_warning_size": validated_settings.tight_multimer_large_component_warning_size,
        "residue_contact_cutoff": validated_settings.residue_contact_cutoff,
        "atom_contact_cutoff": validated_settings.atom_contact_cutoff,
        "min_residue_contacts": validated_settings.min_residue_contacts,
        "min_atom_contacts": validated_settings.min_atom_contacts,
        "peptide_max_length": validated_settings.peptide_max_length,
        "sadie_domain_bitscore_threshold": validated_settings.sadie_domain_bitscore_threshold,
        "sadie_domain_limit": validated_settings.sadie_domain_limit,
        "low_confidence_antibody_threshold": validated_settings.low_confidence_antibody_threshold,
    }
    config["single"]["outdir"] = Path(config["single"]["outdir"])
    config["batch"]["outdir"] = Path(config["batch"]["outdir"])
    validated_clustering = ClusteringSettings(**config["clustering"])
    config["clustering"] = {
        "outdir": validated_clustering.outdir,
        "protein_sequence_mode": validated_clustering.protein_sequence_mode,
        "protein_structure_mode": validated_clustering.protein_structure_mode,
        "protein_subcluster_by_sequence": validated_clustering.protein_subcluster_by_sequence,
        "cutoff_date": validated_clustering.cutoff_date,
        "dimer_mode": validated_clustering.dimer_mode,
        "dimer_structure_mode": validated_clustering.dimer_structure_mode,
        "dimer_tm_score_threshold": validated_clustering.dimer_tm_score_threshold,
        "dimer_interface_residue_cutoff": validated_clustering.dimer_interface_residue_cutoff,
        "multimer_mode": validated_clustering.multimer_mode,
        "multimer_structure_mode": validated_clustering.multimer_structure_mode,
        "multimer_tm_score_threshold": validated_clustering.multimer_tm_score_threshold,
        "multimer_max_atoms_for_refinement": validated_clustering.multimer_max_atoms_for_refinement,
        "antibody_complex_mode": validated_clustering.antibody_complex_mode,
        "antibody_complex_structure_mode": validated_clustering.antibody_complex_structure_mode,
        "antibody_complex_tm_score_threshold": validated_clustering.antibody_complex_tm_score_threshold,
        "antibody_complex_interface_residue_cutoff": validated_clustering.antibody_complex_interface_residue_cutoff,
        "tcr_complex_mode": validated_clustering.tcr_complex_mode,
        "tcr_complex_structure_mode": validated_clustering.tcr_complex_structure_mode,
        "tcr_complex_tm_score_threshold": validated_clustering.tcr_complex_tm_score_threshold,
        "protein_min_seq_id": validated_clustering.protein_min_seq_id,
        "protein_coverage": validated_clustering.protein_coverage,
        "protein_cov_mode": validated_clustering.protein_cov_mode,
        "model": validated_clustering.model,
        "keep_hydrogens": validated_clustering.keep_hydrogens,
        "tm_score_threshold": validated_clustering.tm_score_threshold,
        "min_alignment_coverage_ratio": validated_clustering.min_alignment_coverage_ratio,
        "usalign_executable": validated_clustering.usalign_executable,
        "jobs": validated_clustering.jobs,
        "mmseqs_threads": validated_clustering.mmseqs_threads,
        "sequence_cluster_jobs": validated_clustering.sequence_cluster_jobs,
        "dimer_extraction_jobs": validated_clustering.dimer_extraction_jobs,
        "usalign_jobs": validated_clustering.usalign_jobs,
        "log_level": validated_clustering.log_level,
    }

    jobs = config["batch"]["jobs"]
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError("batch.jobs must be an integer >= 1")
    if not isinstance(config["batch"]["fail_fast"], bool):
        raise ValueError("batch.fail_fast must be a boolean")


def _merge_section(
    destination: dict[str, Any],
    source: dict[str, Any],
    allowed_keys: set[str],
    section_name: str,
) -> None:
    """Merge one parsed TOML table after checking for unexpected keys."""

    unknown_keys = sorted(set(source) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown key(s) in [{section_name}]: {', '.join(unknown_keys)}")
    destination.update(source)


def _normalize_legacy_setting_aliases(source: dict[str, Any]) -> dict[str, Any]:
    """Map deprecated config keys to current names before validation."""

    normalized = dict(source)
    legacy_key = "tight_multimer_leiden_resolution"
    current_key = "tight_multimer_louvain_resolution"
    if legacy_key in normalized:
        if current_key in normalized:
            raise ValueError(
                f"Use only one of [settings].{legacy_key} or [settings].{current_key}"
            )
        normalized[current_key] = normalized.pop(legacy_key)
    return normalized


def resolve_source_path(source_path: str, cif_files_directory: str | Path | None = None) -> str:
    """Remap *source_path* when an explicit ``--cif-files-directory`` is provided.

    When *cif_files_directory* is ``None`` the original *source_path* is returned
    unchanged.  Otherwise the basename of *source_path* is joined with
    *cif_files_directory* and the resulting path must exist.
    """

    if cif_files_directory is None:
        return source_path
    resolved = Path(cif_files_directory) / Path(source_path).name
    if not resolved.exists():
        raise FileNotFoundError(
            f"Source file not found in cif directory: {resolved} "
            f"(original: {source_path})"
        )
    return str(resolved)


def _require_mapping(section_name: str, value: Any) -> None:
    """Ensure a TOML section is a plain key-value table."""

    if not isinstance(value, dict):
        raise ValueError(f"[{section_name}] must be a TOML table")
