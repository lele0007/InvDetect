import argparse
from pathlib import Path

import joblib
from torch.utils.data import DataLoader

from invdetect.classifier import extract_latent_features, fit_classifier
from invdetect.config import load_config
from invdetect.data import BraTS2021PatchDataset
from invdetect.diffusion import choose_device, load_diffusion


def main():
    parser = argparse.ArgumentParser(description="Train One-Class SVM on normal latents.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    device = choose_device(config["runtime"]["device"])
    paths = config["paths"]
    classifier_config = config["classifier"]
    model, schedule, checkpoint = load_diffusion(paths["diffusion_checkpoint"], device)
    dataset = BraTS2021PatchDataset(
        paths["train_dir"], channels=checkpoint["channels"]
    )
    loader = DataLoader(
        dataset,
        batch_size=classifier_config["batch_size"],
        shuffle=False,
        num_workers=config["runtime"]["num_workers"],
    )
    features, _, _ = extract_latent_features(model, schedule, loader, device)
    classifier = fit_classifier(features, nu=classifier_config["nu"])

    output = Path(paths["classifier_checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output)
    print(f"trained on {len(features)} normal patches")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
