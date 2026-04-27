# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This is the **AxialPolCap** paper repository: "Extending Deep-Learning P-Wave
Polarity Classification to Seafloor Microearthquakes at Axial Seamount."

The project trains a hybrid autoencoder + classifier (AxialPolCap) for
automatic P-wave first-motion polarity picking to support focal mechanism
analysis.  Seven OBS stations at Axial Seamount are used:
AS1, AS2, CC1, EC1, EC2, EC3, ID1.

All active, publication-ready scripts live in `01-scripts/`, organized by
pipeline stage.

## Environment Setup

The conda environment is `tf-2.14.0` (Python 3.11, TensorFlow 2.13.1,
Keras 2.13.1):

```bash
conda activate tf-2.14.0
```

All scripts are run as standalone Python scripts from the repository root
(`FM_ML/`):

```bash
python 01-scripts/<subdir>/<script_name>.py
```

There is no build system, test suite, or linter configured.

## Workflow Pipeline

Scripts live in `01-scripts/` organized by pipeline stage.  There are two
parallel pipelines: the **NPY pipeline** (paper workflow) and the
**H5 pipeline** (experimental HDF5-based alternative).

### NPY Pipeline (paper workflow)

```
01-scripts/
  data_preparation/
    01_build_training_dataset.py      # augment + crop + split unified dataset
    02_build_eval_dataset.py          # build standalone eval set
    03_build_loso_training_dataset.py # augment + crop + split LOSO dataset
    04_build_loso_eval_dataset.py     # build LOSO eval set
  benchmark/
    eval_polarcap_baseline.py         # evaluate PolarCAP baseline
  training/
    train_axialpolcap.py              # train unified AxialPolCap model
    train_loso.py                     # LOSO cross-validation training
    transfer_learning.py              # transfer learning experiments
  evaluation/
    eval_loso.py                      # evaluate LOSO models
    eval_transfer_learning.py         # evaluate transfer-learning models
  application/
    apply_to_catalog.py               # apply to 2015-2021 Axial Seamount catalog
  make_manuscript_figures.py          # regenerate all 14 manuscript figures → 03-figs/
```

### H5 Pipeline (HDF5-based alternative)

```
01-scripts/
  convert_mat_to_h5.py               # convert .mat templates → A_wave_train.h5 / A_wave_val.h5
  split_h5_train_val.py              # split HDF5 dataset into train/val
  data_preparation/
    01_build_training_dataset_h5.py  # augment from H5; outputs TMSF_Tra_002/train_dataset.h5
    02_build_eval_dataset_h5.py      # augment val set; outputs TMSF_Val_002/val_dataset.h5
    03_build_eval_dataset_h5_sigma001.py  # variant with σ=0.01 s time shift
  training/
    train_axialpolcap_h5.py          # train from H5 datasets; 80/10/10 split internal
  evaluation/
    eval_model.py                    # evaluate H5-trained model; uses h5py, CPU-only (Metal crashes)
```

### Key NPY script descriptions

**01_build_training_dataset.py**
Reads from `Template_divide.mat`; fits a lognormal distribution to per-station
empirical SNR values; augments each template to reach ~15,000 samples per
station; applies cubic-spline time shifts (Normal(0, sigma=2 samples at 200 Hz));
center-crops 200→64 samples; max-normalizes; stratified 80/10/10 split;
saves to `02-data/K_aug/TMSF_Tra_001/`.

**03_build_loso_training_dataset.py**
Reads from `Template.mat`; uses last 20% of each station's events;
step-SNR distribution (10% in [0,5] dB, 80% in [5,35] dB, 10% in [35,50] dB);
base_multi_trace=20 (ID1 gets 3×); no time shift; saves to
`02-data/K_aug/STEP010/`.

**02_build_eval_dataset.py**
Reads from paired validation `.mat` files; no augmentation; saves
`timeseries_{STA}.npy` / `polarities_{STA}.npy` without a split prefix.

**train_axialpolcap.py**
Trains on merged all-station data from `K_aug/TMSF_Tra_001/`; saves JSON
training history + ROC curves; uses timestamped model filenames.

**04_build_loso_eval_dataset.py**
Reads from paired LOSO validation `.mat` files; no augmentation; saves
`timeseries_{STA}.npy` / `polarities_{STA}.npy` (no split prefix) to
`02-data/K_aug/Val/`.

**train_loso.py**
LOSO training on `K_aug/STEP010/`; saves one model per held-out station as
`PolarPicker_LOSO_{STA}.keras`.
**Gotcha**: the `stations` list is hardcoded to `['EC2','EC3','ID1']` — change
to all 7 stations before running a full LOSO sweep.

**transfer_learning.py**
Fine-tunes the external PolarCAP baseline (`06-models/PolarCAP.h5`, `.h5`
format) on `TMSF_Tra_001/` data; saves as `PolarCAP_finetuned_TMSF.h5`.
Runs CPU-only (Metal plugin crashes during fine-tuning on macOS).

**make_manuscript_figures.py**
Standalone Python replacement for the original MATLAB figure-generation
scripts.  Reads data from sibling repos (`FM/`, `FM3/`, `FM4/`) as well as
this repo's `02-data/` and `06-models/` directories.  Outputs
`03-figs/Figure01_python.png` through `Figure14_python.png`.  Run with
`python 01-scripts/make_manuscript_figures.py`; pass `--figures 1 3 5` to
regenerate only specific figures.  Requires `cartopy` for the map panel
(Figure 1).

## Model Architecture (AxialPolCap)

Defined via `build_polarPicker()` in each training script.  It is a
**multi-output model** with a shared encoder:

- **Input**: 64-sample seismic waveform window (Z-component, P-arrival
  ±32 samples), shape `(64, 1)`
- **Encoder**: Conv1D(32,32) → Dropout → BN → MaxPool(2) → Conv1D(8,16) →
  BN → MaxPool(2) → output shape `(16, 8)`
- **Decoder** (autoencoder branch): Conv1D → BN → UpSample → Conv1D → BN →
  UpSample → Conv1D → shape `(64, 1)` — trained with MSE loss, weight=1
- **Classifier** (polarity head): Flatten → Dense(2, softmax) — trained with
  Huber loss, weight=200
- **Outputs**: `[decoder, classifier]`; prediction uses `model.predict(X)[1]`

The heavy loss weighting on the classifier (200×) means the encoder learns
representations driven by polarity discrimination, not just waveform
reconstruction.

## Data Conventions

**Raw data**: Per-station MATLAB `.mat` struct files with fields:
- `W_{STA}`: waveform (200 samples at 200 Hz)
- `Man_{STA}` or `Po_{STA}`: manual polarity label (-1 = Negative, +1 = Positive)

**Converted NPY files** (per station, saved in the active data subdirectory):
- `{split}_timeseries_{STA}.npy` — shape `(N, 64, 1)`, dtype float32,
  already normalized
- `{split}_polarities_{STA}.npy` — shape `(N,)`, dtype int32,
  values 0 (Negative) or 1 (Positive)
- `{split}_timeseries_all.npy` / `{split}_polarities_all.npy` — merged
  across all stations
- `timeseries_{STA}.npy` / `polarities_{STA}.npy` — no split prefix,
  used for standalone eval sets

**Train/val/test split ratio**: 80% train, 10% val, 10% test.
Implemented as `train_test_split(test_size=0.2)` then
`train_test_split(test_size=0.5)` on the remainder.  Splits are stratified
by polarity.

**Normalization**: each waveform is independently max-normalized:
`X / max(|X|)`.  Always apply before training or inference.

**LOSO** (Leave-One-Station-Out): each model is trained on 6 stations,
evaluated on the held-out 7th.  Model filenames follow
`PolarPicker_LOSO_{STA}.keras`.

**Output `.mat` files** (from `apply_to_catalog.py`): `Po_{STA}` field is
overwritten with a 4-element array
`[GroundTruth, Prediction, Confidence, Entropy]`.

## Directory Structure

```
FM_ML/
  01-scripts/                 # Active publication scripts
    data_preparation/
    benchmark/
    training/
    evaluation/
    application/
    convert_mat_to_h5.py      # H5 pipeline utility
    split_h5_train_val.py     # H5 pipeline utility
  02-data/                    # Input .mat files and converted .npy / .h5 datasets
    K_aug/
      Template_divide.mat     # Templates for unified training (01_build_...)
      Template.mat            # All templates (03_build_... uses last 20%)
      STEP010/                # LOSO training .npy output
      TMSF_Tra_001/           # Unified NPY training output
      TMSF_Val_001/           # Eval set for transfer-learning evaluation
      Val/                    # LOSO eval set (no split prefix; from 04_build_loso_eval_dataset.py)
      TMSF_Tra_002/           # H5 pipeline training output (train_dataset.h5)
      TMSF_Val_002/           # H5 pipeline val output (val_dataset.h5)
    H_noi/
      H_Noise_200.mat         # Noise waveforms (200 samples, 200 Hz)
      H_noise_dB20_snrValue.mat  # Empirical per-station SNR distributions
  03-figs/                    # Output figures (confusion matrices, ROC curves, manuscript PNGs)
    LOSO_010/                 # LOSO confusion matrices
    Figure{01-14}_python.png  # manuscript figures from make_manuscript_figures.py
  04-logs/                    # Log files
  05-tmp/                     # Temporary outputs (training history CSVs)
  06-models/                  # Saved Keras models (.keras format)
    LOSO_010/                 # LOSO models, one per held-out station
    history/                  # JSON training histories
```

## Key Implementation Notes

- `tf.config.run_functions_eagerly(True)` is set in training scripts to avoid
  TF graph serialization issues with Lambda layers used to name model outputs.
- **Two variants of `build_polarPicker()`** exist across scripts:
  - **List-style** (`train_axialpolcap.py`): `loss=['mse', hub]`,
    `loss_weights=[1, 200]`.  Models load without `safe_mode=False`.
  - **Dict-style with Lambda layers** (`train_loso.py`): outputs wrapped in
    named Lambda layers (`"decoder"`, `"classifier"`).  **These models require
    `safe_mode=False` on load.**
- When loading models with Lambda layers, use
  `keras.models.load_model(path, safe_mode=False)`.
- The model's two outputs are accessed as `y_raw[0]` (decoder) and
  `y_raw[1]` (classifier probabilities).  Some scripts check
  `isinstance(y_raw, (list, tuple))` to handle both single- and multi-output
  models.
- Use `scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)` to
  load MATLAB struct arrays; access fields with `getattr(struct_obj, 'field')`.
  Do NOT use `eval()` — use `mat_dict[station_name]` or `getattr()` instead.
- **macOS GPU crash**: The TF-Metal plugin crashes during `model.predict` and
  fine-tuning.  `eval_model.py`, `transfer_learning.py`, and
  `eval_transfer_learning.py` all call
  `tf.config.set_visible_devices([], 'GPU')` at startup — keep this pattern
  in any new macOS inference or fine-tuning script.
- Data from HPC (Tallgrass cluster) uses absolute paths like
  `/caldera/projects/...`; local paths use
  `/Users/mcZhang/Documents/GitHub/FM_ML/`.
- The `02-data/` directory is not tracked in git.  Raw data are available
  from the OOI portal and the Wang et al. (2024) catalog at
  https://axialdd.ldeo.columbia.edu.
