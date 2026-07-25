import argparse

from invdetect.config import load_config
from invdetect.dice import evaluate_dice


def main():
    parser = argparse.ArgumentParser(description="Calculate Dice only.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    results, mean_dice = evaluate_dice(
        f"{paths['reconstruction_dir']}/masks",
        paths["masks_dir"],
        threshold=config["dice"]["threshold"],
    )
    for filename, score in results:
        print(f"{filename}: Dice={score:.4f}")
    print(f"Mean Dice={mean_dice:.4f}")


if __name__ == "__main__":
    main()
