# InvDetect

InvDetect is an organized implementation of the method described in “InvDetect: Unsupervised Medical Anomaly Detection in the Noise Latent Space of DDIM.”

## Method Overview

1. Extract overlapping normal image patches using a `32 × 32` sliding window with a stride of `16`.
2. Train a four-stage time-conditioned U-Net noise predictor using only normal patches. The model is optimized with Adam using a learning rate of `1e-3`.
3. Apply deterministic DDIM inversion to map each image patch into the noise latent space.
4. Fit a One-Class SVM using the noise latents obtained from normal training patches.
5. Define the anomaly score of a test patch as `s = -decision_function(z)`, so that the SVM decision boundary corresponds to zero and samples outside the boundary receive positive anomaly scores.
6. Aggregate patch-level anomaly scores into a pixel-level anomaly map `A` using Gaussian weights centered on each patch, with `sigma = 8`.
7. Construct four-neighborhood edge weights based on the similarity of aggregated noise latents and apply s-t min-cut to obtain the final binary anomaly mask.

## Project Structure

invdetect/
├── configs/default.yaml          # Default parameters used by the method
├── src/invdetect/                # Installable Python package
├── tests/                        # Unit tests
└── README.md                     # Project documentation

## Environment
Python 3.10 or 3.11 is recommended. Training should be performed on a CUDA-enabled GPU.

## Dataset Structure
The training directory must contain only normal images. The program recursively reads PNG, JPEG, BMP, and TIFF files and extracts overlapping patches at runtime, so patches do not need to be generated in advance.
data/
├── train/normal/                 # Normal training images
├── test/images/                  # Test images
└── test/masks/                   # Ground-truth anomaly masks
Both the width and height of each input image must be at least 32 pixels.