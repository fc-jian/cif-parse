# cif-parse

`cif-parse` is a Python toolkit for parsing mmCIF structures and exporting normalized chain- and complex-level annotations.

It is designed for workflows that need more than a raw structure dump, including:

1. chain inventory generation
2. polymer coverage and unresolved-segment reporting
3. dimer and tight multimer detection
4. antibody-antigen complex extraction
5. TCR-pMHC complex extraction

## Installation

Create the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate cif-parse
```

Install the package in editable mode:

```bash
pip install -e .
```

## Command-Line Usage

Run a single mmCIF file:

```bash
python -m cif_parse.cli single path/to/file.cif --outdir ./outputs --log-level INFO
```

Run a batch job on a directory:

```bash
python -m cif_parse.cli batch path/to/mmcif_dir --outdir ./outputs --jobs 8
```

Run a batch job from an input list:

```bash
python -m cif_parse.cli batch --input-list ./input_list.txt --outdir ./outputs --jobs 8
```

Use a custom config file:

```bash
python -m cif_parse.cli --config ./config.toml batch path/to/mmcif_dir --outdir ./outputs
```

## Output

The primary output format is JSON. CSV is supported as an optional flattened export.

Typical batch outputs include:

1. `manifest.json`
2. `summary.json`
3. `review.json`
4. per-case output directories under `cases/`

## Environment

The project targets Python 3.12 and uses a Conda environment definition in `environment.yml`.
