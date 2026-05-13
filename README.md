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

For large batch outputs (thousands to millions of cases), first build a prep directory that consolidates all case bundles into columnar Parquet files and pre-caches per-chain atom coordinates into indexed chunk files. The chunk count equals the number of worker processes, keeping the total file count bounded regardless of scale:

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
└── cif_coords/                   # per-chain atom array chunks
    ├── chunk_0.bin + chunk_0.idx
    ├── chunk_1.bin + chunk_1.idx
    └── ...                        (num_workers chunks total)
```

Atoms are stored **per-chain** rather than per-assembly: monomer / dimer / multimer / complex extraction each reads only the chains they need via direct index lookup, eliminating repeated slicing of full assembly blobs. During clustering, per-chain AtomArrays loaded from prep are also kept in a shared in-process LRU cache, so monomer structure extraction and later higher-order complex extraction can reuse the same chain coordinates.

Pass `--prep-dir` to the clustering command to read only from the prep directory. In this mode `--inputs` is no longer required because the Parquet files contain the monomer, dimer, multimer, antibody-complex, and TCR-complex rows needed by clustering:

```bash
cif-parse-cluster \
  --prep-dir clustering_prep \
  --outdir cluster_outputs
```

Full clustering and every structure/high-order subcommand consume prep data for coordinates; clustering reads parse JSON, prep Parquet, and parse-stage atom pickle caches only. The `seq` stage can still run directly from `--inputs` because it only needs case JSON. If the legacy full command is run with `--inputs` but no `--prep-dir`, it builds a temporary prep directory first and then runs clustering from that prep data.

Use `--no-cif-cache` to skip Phase 2 coordinate indexing when only the Parquet files are needed. Structure and high-order clustering require the coordinate index.

### Running Clustering

For large runs, split clustering by stage. `seq` must run first because it writes `monomer_inventory.jsonl` and `sequence_clusters/membership.csv`; `structure` normally runs second because downstream high-order signatures use both sequence and structure cluster ids by default. After `seq + structure` complete, the high-order stages consume the same `--prep-dir` and `--outdir`; they are independent and can be submitted in parallel:

```bash
# Run first
cif-parse-cluster seq --prep-dir clustering_prep --outdir cluster_outputs

# Run second by default
cif-parse-cluster structure --prep-dir clustering_prep --outdir cluster_outputs

# After seq + structure, high-order stages can run in parallel:
cif-parse-cluster dimer     --prep-dir clustering_prep --outdir cluster_outputs
cif-parse-cluster multimer  --prep-dir clustering_prep --outdir cluster_outputs
cif-parse-cluster tcr       --prep-dir clustering_prep --outdir cluster_outputs
cif-parse-cluster abag      --prep-dir clustering_prep --outdir cluster_outputs
```

Pass `--ignore-structure` to a high-order stage only when you intentionally want signatures based on sequence cluster ids alone.

Run sequence-only clustering directly from case JSON:

```bash
cif-parse-cluster seq \
  --inputs batch_outputs/cases \
  --outdir cluster_outputs
```

Run the current full clustering stack:

```bash
cif-parse-cluster \
  --prep-dir clustering_prep \
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
5. Monomer extraction and structure clustering run in a **pipelined** mode: structures are extracted per sequence cluster on-the-fly while USalign runs on previously extracted clusters, overlapping I/O and computation.
6. The legacy full command now calls the split stages as `seq -> structure -> parallel(tcr, abag, multimer, dimer)`. For maximum throughput on a scheduler, run `seq` and `structure` first, then submit high-order stage subcommands in parallel.
7. Higher-order structure refinement skips singleton signature groups: singletons are emitted directly as clusters without writing complex PDBs or running USalign.
8. `--jobs N` automatically propagates to all subtask workers (`--mmseqs-threads`, `--sequence-cluster-jobs`, `--usalign-jobs`). Individual subtask counts can still be overridden explicitly.
9. `--cif-files-directory` is deprecated and ignored by clustering.
10. Use `--prep-dir` as the full clustering input source for Parquet rows and per-chain cached AtomArrays. Without it, `seq` can run from case JSON; the legacy full command auto-builds temporary prep before any coordinate-consuming stage.

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
  --prep-dir clustering_prep \
  --outdir cluster_outputs \
  --dimer-mode signature \
  --multimer-mode signature \
  --antibody-complex-mode signature \
  --tcr-complex-mode signature
```

## Requirements

The project targets Python 3.12. The recommended environment is defined in `environment.yml`.
