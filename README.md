# InvDetect

The pipeline contains five steps:

1. Train a diffusion denoiser using normal patches.
2. Invert normal patches into diffusion noise latents.
3. Train a One-Class SVM using the normal latents.
4. Classify labeled test patches as normal or abnormal.
5. Reconstruct full images and masks, then calculate Dice.

## Project Structure

```text
invdetect/
├── configs/default.yaml          
├── data/                         
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
├── requirements.txt
└── README.md
```

## Dataset Format

data/
├── train/
│   └── normal/                   # Normal training patches only
├── test/
│   ├── normal/                   # Labeled normal test patches
│   └── abnormal/                 # Labeled abnormal test patches
└── masks/                        # Ground-truth full-image masks
```
All patches in one experiment must have the same width, height, and number of channels.


## Installation

Python 3.10 or 3.11 is recommended.

