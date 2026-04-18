# cif-parse

`cif-parse` is a Python toolkit for parsing mmCIF structures, normalizing chain- and entity-level metadata, and extracting biologically meaningful assemblies and immune complexes from structural data.

The project is designed for workflows that need more than a flat structure dump. It provides reusable intermediate records for chain annotation, coverage assignment, dimer and multimer detection, and higher-level immune complex extraction.

## What It Does

`cif-parse` currently supports:

1. mmCIF reading and structure summarization
2. normalized chain inventory with both `label_asym_id` and `auth_asym_id`
3. chain annotation for proteins, nucleic acids, ligands, metals, and branched components
4. polymer coordinate coverage and unresolved-segment reporting
5. dimer interface detection
6. tight multimer detection
7. antibody-antigen complex extraction
8. TCR-pMHC complex extraction
9. batch processing with aggregate `manifest.json`, `summary.json`, and `review.json`

## Status

The project is in active development. The current implementation already provides a usable end-to-end pipeline for single-structure and batch processing, while immune annotation and complex refinement are still being improved.

Progress tracking and internal development notes have been moved to [`docs/progress.md`](/home/jianfc/folding/cif-parse/docs/progress.md).

## Installation

Recommended environment setup:

```bash
conda env create -f environment.yml
conda activate mmcif-parse
```

If you prefer an editable local install after activating the environment:

```bash
pip install -e .
```

## Quick Start

Run a single structure:

```bash
python -m mmcif_parse.cli single path/to/file.cif --outdir ./outputs --log-level INFO
```

Run a batch job:

```bash
python -m mmcif_parse.cli batch path/to/mmcif_dir --outdir ./outputs --jobs 8
```

Run the representative regression suite:

```bash
PYTHONPATH=src python scripts/run_representative_suite.py --jobs 8
```

## Documentation

Project documentation is organized under [`docs/`](/home/jianfc/folding/cif-parse/docs):

1. [`docs/architecture.md`](/home/jianfc/folding/cif-parse/docs/architecture.md): module layout and internal data flow
2. [`docs/rules.md`](/home/jianfc/folding/cif-parse/docs/rules.md): annotation and assembly rules
3. [`docs/outputs.md`](/home/jianfc/folding/cif-parse/docs/outputs.md): output schema and file layout
4. [`docs/progress.md`](/home/jianfc/folding/cif-parse/docs/progress.md): development progress and milestone tracking
5. [`docs/roadmap.md`](/home/jianfc/folding/cif-parse/docs/roadmap.md): next-stage tasks and validation goals
6. [`docs/project_review.md`](/home/jianfc/folding/cif-parse/docs/project_review.md): phase summary and project close-out notes

## Output Philosophy

The primary output format is JSON. CSV is supported as an optional flattened export, but JSON remains the source of truth for structured records, evidence fields, warnings, and member mappings.

## Development Notes

1. The package targets Python 3.12.
2. Configuration defaults live in [`config.toml`](/home/jianfc/folding/cif-parse/config.toml).
3. Representative test inputs are tracked in:
   [`test_representative_list.txt`](/home/jianfc/folding/cif-parse/test_representative_list.txt)
   [`test_ab_list.txt`](/home/jianfc/folding/cif-parse/test_ab_list.txt)
   [`test_tcr_list.txt`](/home/jianfc/folding/cif-parse/test_tcr_list.txt)

## License

No license file has been added yet.
