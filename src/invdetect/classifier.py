import numpy as np
import torch
from sklearn.svm import OneClassSVM

from invdetect.diffusion import ddim_invert


@torch.inference_mode()
def extract_latent_features(model, schedule, loader, device):
    feature_batches = []
    filenames = []
    labels = []
    for images, batch_names, batch_labels in loader:
        latents = ddim_invert(model, schedule, images.to(device))
        feature_batches.append(latents.flatten(1).cpu().numpy())
        filenames.extend(batch_names)
        labels.extend(batch_labels.tolist())
    return np.concatenate(feature_batches), filenames, np.asarray(labels)


def fit_classifier(features: np.ndarray, nu: float = 0.1) -> OneClassSVM:
    classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    classifier.fit(features)
    return classifier


def predict_patches(
    classifier: OneClassSVM, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    anomaly_scores = -classifier.decision_function(features)
    predictions = (anomaly_scores > 0.0).astype(np.int64)
    return predictions, anomaly_scores
