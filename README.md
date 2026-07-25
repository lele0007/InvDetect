# InvDetect

InvDetect is a compact, professional implementation of patch-based brain tumor
anomaly detection. It assumes that all BraTS 2021 image patches have already
been extracted.

The pipeline contains five steps:

1. Train a diffusion denoiser using normal patches.
2. Invert normal patches into diffusion noise latents.
3. Train a One-Class SVM using the normal latents.
4. Classify labeled test patches as normal or abnormal.
5. Reconstruct full images and masks, then calculate Dice separately.

## Project Structure

```text
invdetect/
├── .github/workflows/ci.yml      # GitHub Actions
├── configs/default.yaml          # Experiment configuration
├── data/                         # Empty dataset directory template
│   ├── train/normal/
│   ├── test/normal/
│   ├── test/abnormal/
│   └── masks/
├── scripts/
│   ├── train_diffusion.py
│   ├── train_classifier.py
│   ├── test_classifier.py
│   ├── reconstruct_images.py
│   └── evaluate_dice.py
├── src/invdetect/
│   ├── config.py
│   ├── data.py
│   ├── diffusion.py
│   ├── classifier.py
│   ├── reconstruction.py
│   └── dice.py
├── tests/test_pipeline.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Dataset Format

The repository includes empty dataset folders. Add images locally as follows:

```text
data/
├── train/
│   └── normal/                   # Normal training patches only
├── test/
│   ├── normal/                   # Labeled normal test patches
│   └── abnormal/                 # Labeled abnormal test patches
└── masks/                        # Ground-truth full-image masks
```

The dataset class is named `BraTS2021PatchDataset`. All patches in one
experiment must have the same width, height, and number of channels.

Images placed in `data/` are ignored by Git. Only the empty directory structure
is committed.

## Patch Naming

Full-image reconstruction requires the original image ID and patch coordinates:

```text
BraTS2021_00001__x=0_y=0.png
BraTS2021_00001__x=32_y=0.png
BraTS2021_00001__x=0_y=32.png
```

`x` and `y` are the top-left patch coordinates in the original image.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

All paths and main hyperparameters are defined in:

```text
configs/default.yaml
```

For RGB patches, change:

```yaml
data:
  channels: 3
```

## Usage

Run every command from the repository root.

### 1. Train Diffusion

```bash
python scripts/train_diffusion.py --config configs/default.yaml
```

### 2. Train One-Class SVM

```bash
python scripts/train_classifier.py --config configs/default.yaml
```

### 3. Classify Test Patches

```bash
python scripts/test_classifier.py --config configs/default.yaml
```

The output CSV contains:

```text
filename,true_label,predicted_label,anomaly_score
```

Label convention:

- `0`: normal
- `1`: abnormal

### 4. Reconstruct Images and Masks

```bash
python scripts/reconstruct_images.py --config configs/default.yaml
```

Outputs:

```text
outputs/reconstructed/
├── images/
├── masks/
└── score_maps/
```

### 5. Dice

```bash
python scripts/evaluate_dice.py --config configs/default.yaml
```

Dice is kept in a separate module and script. No additional evaluation metrics
are included.

## Tests

```bash
ruff check src scripts tests
pytest -q
```
