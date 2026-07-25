import argparse
import csv
from pathlib import Path

import joblib
from torch.utils.data import DataLoader

from invdetect.classifier import extract_latent_features, predict_patches
from invdetect.config import load_config
from invdetect.data import BraTS2021PatchDataset
from invdetect.diffusion import choose_device, load_diffusion


def main():
    parser = argparse.ArgumentParser(description="Classify labeled BraTS test patches.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    paths = config["paths"]
    classifier_config = config["classifier"]
    device = choose_device(config["runtime"]["device"])
    model, schedule, checkpoint = load_diffusion(paths["diffusion_checkpoint"], device)
    classifier = joblib.load(paths["classifier_checkpoint"])
    dataset = BraTS2021PatchDataset(
        paths["test_dir"], channels=checkpoint["channels"], labeled=True
    )
    loader = DataLoader(
        dataset,
        batch_size=classifier_config["batch_size"],
        shuffle=False,
        num_workers=config["runtime"]["num_workers"],
    )
    features, filenames, labels = extract_latent_features(
        model, schedule, loader, device
    )
    predictions, anomaly_scores = predict_patches(classifier, features)

    output = Path(paths["predictions_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "true_label", "predicted_label", "anomaly_score"])
        writer.writerows(
            zip(filenames, labels, predictions, anomaly_scores, strict=True)
        )
    print("label 0 = normal, label 1 = abnormal")
    print(f"tested {len(filenames)} patches")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
