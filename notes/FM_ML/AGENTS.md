# Repository Guidelines

## Project Structure & Module Organization
`01-scripts/` contains the active Python workflow, split by stage: `data_preparation/`, `benchmark/`, `training/`, `evaluation/`, and `application/`. Use `make_manuscript_figures.py` and `figure_01_standalone.py` for figure generation. `02-data/` holds large input and derived datasets and is intentionally untracked. Generated figures go to `03-figs/`, logs to `04-logs/`, temporary artifacts to `05-tmp/`, and saved models to `06-models/`. Reference files, MATLAB utilities, and external assets live in `07-files/` and `04-manuscripts/`.

## Build, Test, and Development Commands
Work from the repository root in the `tf-2.14.0` conda environment.

```bash
conda activate tf-2.14.0
python 01-scripts/data_preparation/01_build_training_dataset.py
python 01-scripts/training/train_axialpolcap.py
python 01-scripts/evaluation/eval_model.py
python 01-scripts/make_manuscript_figures.py --figures 1 3 5
```

Use the data-prep scripts to build `.npy`/`.h5` inputs, training scripts to produce `.keras` models in `06-models/`, evaluation scripts to regenerate metrics and confusion matrices, and figure scripts to refresh manuscript PNGs in `03-figs/`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, module-level constants in `UPPER_SNAKE_CASE`, functions in `snake_case`, and standalone scripts named with numeric prefixes when they represent ordered pipeline steps (for example, `01_build_training_dataset.py`). Prefer small helper functions over inline blocks. There is no configured formatter or linter, so keep imports grouped, comments brief, and paths explicit.

## Testing Guidelines
This repository does not have a formal unit-test suite. Validate changes by running the affected script end-to-end on a representative subset and checking the expected outputs in `03-figs/`, `04-logs/`, `05-tmp/`, or `06-models/`. For data or model changes, record the exact command used and note any regenerated artifacts. Avoid committing large derived data from `02-data/`.

## Commit & Pull Request Guidelines
Match the existing history: short imperative subjects with optional scope prefixes, such as `fig2: remove suptitle from Figure 2 waveform plot` or `Add figure validation manifest for all 14 manuscript figures`. Keep commits focused on one pipeline stage or manuscript task. PRs should include a concise summary, affected scripts/outputs, validation steps run, and before/after figures when visual outputs change.

## Data & Configuration Notes
`02-data/` is ignored by Git; contributors must supply required MATLAB/HDF5 inputs locally. Some scripts use hardcoded absolute paths and station lists, so check those before running long jobs or LOSO sweeps.
