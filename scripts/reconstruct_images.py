import argparse

from invdetect.config import load_config
from invdetect.reconstruction import reconstruct_images


def main():
    parser = argparse.ArgumentParser(description="Reconstruct images and anomaly masks.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    reconstruct_images(
        paths["test_dir"],
        paths["predictions_csv"],
        paths["reconstruction_dir"],
        threshold=config["reconstruction"]["threshold"],
    )
    print(f"saved: {paths['reconstruction_dir']}")


if __name__ == "__main__":
    main()
