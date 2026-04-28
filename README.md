# cif-parse

`cif-parse` parses mmCIF structures and exports normalized annotations for chains, interfaces, multimers, antibody-antigen complexes, and TCR-pMHC complexes. It also includes a separate clustering workflow for deduplicating parsed structural annotations.

## Installation

Create the Conda environment and install the package:

```bash
conda env create -f environment.yml
conda activate cif-parse
pip install -e .
```

The clustering workflow uses `USalign` for structure refinement and `mmseqs` for the default protein sequence clustering mode. Both are included in the recommended Conda environment.

## Parse mmCIF Files

Process one structure:

```bash
cif-parse single path/to/file.cif.gz --outdir outputs/file
```

Process every deposited assembly independently:

```bash
cif-parse single path/to/file.cif.gz --assembly-mode all --outdir outputs/file_all
```

Run a batch job:

```bash
cif-parse batch path/to/mmcif_dir --outdir batch_outputs --jobs 8
```

Run from a text file with one mmCIF path per line:

```bash
cif-parse batch --input-list input_list.txt --outdir batch_outputs --jobs 8
```

## Parse Outputs

For JSON output, the default single-case export is a compact `result.json.gz`. With `--assembly-mode all`, one file is written per assembly as `result_assembly_<id>.json.gz`.

Batch output includes:

1. `manifest.json.gz`
2. `summary.json`
3. `review.json.gz`
4. `summary_report.html`
5. per-case outputs under `cases/`

Use `--debug` when you need split JSON artifacts for manual inspection. CSV output remains available through `--format csv`.

## Clustering

Clustering consumes parsed case outputs, preferably from `--assembly-mode all`. Monomer chains are deduplicated by `(pdb_id, label_asym_id)`, while dimer and higher-level observations keep their assembly-level observations and are clustered afterward.

### Pre-processing (recommended for large inputs)

For large batch outputs (thousands to millions of cases), first build a prep directory that consolidates all case bundles into columnar Parquet files and pre-caches mmCIF atom coordinates into a single indexed binary file. The output always contains exactly 8 files regardless of case count:

```bash
cif-parse-cluster prep \
  --inputs batch_outputs/cases \
  --prep-dir clustering_prep \
  --prep-jobs 8
```

This produces:

```
clustering_prep/
├── monomers.parquet              # pre-parsed polymer chain rows
├── dimers.parquet                # pre-parsed dimer interface rows
├── multimers.parquet             # pre-parsed tight multimer rows
├── antibody_complexes.parquet    # pre-parsed antibody-antigen complex rows
├── tcr_complexes.parquet         # pre-parsed TCR-pMHC complex rows
├── cif_coords.bin               # all AtomArrays concatenated in one file
├── cif_coords.idx               # 76-byte fixed-width index (hash → offset+len)
└── prep_meta.json               # content hashes for incremental updates
```

Content hashing enables incremental updates — re-running `prep` only processes new or changed cases. Each worker writes independently, then the master merges temporary files (no lock contention).

Pass `--prep-dir` to the clustering command to read from the prep directory:

```bash
cif-parse-cluster \
  --inputs batch_outputs/cases \
  --prep-dir clustering_prep \
  --outdir cluster_outputs
```

When `--prep-dir` is not provided, clustering falls back to reading individual `result.json.gz` files directly.

Use `--no-cif-cache` to skip Phase 2 (mmCIF caching) when only the Parquet files are needed.

### Running Clustering

Run monomer-only clustering:

```bash
cif-parse-cluster \
  --inputs batch_outputs/cases \
  --outdir cluster_outputs
```

Run the current full clustering stack:

```bash
cif-parse-cluster \
  --inputs batch_outputs/cases \
  --outdir cluster_outputs \
  --protein-sequence-mode mmseqs2 \
  --protein-structure-mode greedy \
  --dimer-mode signature \
  --dimer-structure-mode greedy \
  --multimer-mode signature \
  --multimer-structure-mode greedy \
  --antibody-complex-mode signature \
  --antibody-complex-structure-mode greedy \
  --tcr-complex-mode signature \
  --tcr-complex-structure-mode greedy
```

The legacy command name `cif-parse-cluster-monomers` is still available and points to the same clustering CLI.

Key clustering defaults:

1. Protein sequence clustering uses `mmseqs2` with sequence identity threshold `0.40`.
2. Protein monomer structure clustering uses `max(TM(query,target), TM(target,query)) >= 0.50`.
3. Protein monomer alignment coverage requires `aligned_length / shorter_length >= 0.80`.
4. Dimer, multimer, antibody-antigen, and TCR-pMHC complex clustering default to signature clustering plus overall `USalign -mm 1 -ter 1` refinement with TM-score threshold `0.50`.
5. Parallel clustering controls are `--jobs`, `--mmseqs-threads`, `--sequence-cluster-jobs`, and `--usalign-jobs`.
6. Use `--cif-files-directory` to override the location of original mmCIF files during structure extraction.
7. Use `--prep-dir` to read case data from Parquet files and cached AtomArrays (built with `cif-parse-cluster prep`).

## Configuration

Configuration files must be explicitly provided via `--config`. Without `--config` the CLI uses built-in defaults for every setting.

- Parser CLI: `cif-parse --config config.toml ...`
- Clustering CLI: `cif-parse-cluster --config config_clustering.toml ...`

CLI arguments always override values from the config file.

```bash
cif-parse --config config.toml batch path/to/mmcif_dir --outdir batch_outputs
```

```bash
cif-parse-cluster --config config_clustering.toml --inputs batch_outputs/cases --outdir cluster_outputs
```

The main sections are:

1. `[settings]` (in `config.toml`): parser behavior, assembly mode, contact thresholds, immune annotation thresholds.
2. `[single]` (in `config.toml`): default single-run output directory.
3. `[batch]` (in `config.toml`): default batch output directory and worker count.
4. `[clustering]` (in `config_clustering.toml`): clustering modes, sequence thresholds, TM-score thresholds, USalign settings, and parallel worker counts.

## Typical Workflow

```bash
# 1. Parse all mmCIF files
cif-parse batch \
  --input-list input_list.txt \
  --outdir batch_outputs \
  --assembly-mode all \
  --jobs 8

# 2. (Optional but recommended) Build prep directory for fast I/O
cif-parse-cluster prep \
  --inputs batch_outputs/cases \
  --prep-dir clustering_prep \
  --prep-jobs 8

# 3. Run clustering (consumes prep directory)
cif-parse-cluster \
  --inputs batch_outputs/cases \
  --prep-dir clustering_prep \
  --outdir cluster_outputs \
  --dimer-mode signature \
  --multimer-mode signature \
  --antibody-complex-mode signature \
  --tcr-complex-mode signature
```

## Requirements

The project targets Python 3.12. The recommended environment is defined in `environment.yml`.
