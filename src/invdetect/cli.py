from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from invdetect.checkpoint import load_model, save_checkpoint
from invdetect.config import InvDetectConfig
from invdetect.data import ImagePatchDataset, find_images, path_identifier
from invdetect.diffusion import DiffusionSchedule, ddim_invert, diffusion_loss
from invdetect.metrics import evaluate_directories
from invdetect.model import TimeConditionedUNet
from invdetect.normality import NoiseLatentNormalityModel
from invdetect.pipeline import InvDetectPipeline
from invdetect.runtime import resolve_device, seed_everything


def _loader(dataset: ImagePatchDataset, config: InvDetectConfig, shuffle: bool) -> DataLoader:
    workers = config.training.num_workers
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _train(args: argparse.Namespace) -> None:
    config = InvDetectConfig.from_yaml(args.config)
    seed_everything(config.seed)
    device = resolve_device(args.device)
    dataset = ImagePatchDataset(
        args.train_dir,
        patch_size=config.patch.size,
        stride=config.patch.stride,
        channels=config.model.input_channels,
    )
    loader = _loader(dataset, config, shuffle=True)
    model = TimeConditionedUNet(**config.model.__dict__).to(device)
    schedule = DiffusionSchedule(**config.diffusion.__dict__).to(device)
    start_epoch = 0

    if args.resume:
        model, metadata = load_model(args.resume, config, device)
    else:
        metadata = {}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    if args.resume:
        if "optimizer_state" in metadata:
            optimizer.load_state_dict(metadata["optimizer_state"])
        start_epoch = int(metadata.get("epoch", 0))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "training_images": len(find_images(args.train_dir)),
                "training_patches": len(dataset),
                "start_epoch": start_epoch,
                "target_epochs": config.training.epochs,
            },
            indent=2,
        )
    )

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.training.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(model, schedule, images)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.shape[0]
            progress.set_postfix(loss=f"{loss.item():.5f}")

        epoch_loss = running_loss / len(dataset)
        print(f"epoch={epoch + 1} loss={epoch_loss:.7f}")
        epoch_number = epoch + 1
        checkpoint_every = config.training.checkpoint_every
        if checkpoint_every > 0 and epoch_number % checkpoint_every == 0:
            periodic_path = output_path.with_name(
                f"{output_path.stem}_epoch_{epoch_number:05d}{output_path.suffix or '.pt'}"
            )
            save_checkpoint(periodic_path, model, config, epoch_number, optimizer)

    save_checkpoint(output_path, model, config, config.training.epochs, optimizer)
    print(f"Saved DDIM checkpoint: {output_path}")


@torch.inference_mode()
def _fit_normality(args: argparse.Namespace) -> None:
    config = InvDetectConfig.from_yaml(args.config)
    seed_everything(config.seed)
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, config, device)
    schedule = DiffusionSchedule(**config.diffusion.__dict__).to(device)
    dataset = ImagePatchDataset(
        args.train_dir,
        patch_size=config.patch.size,
        stride=config.patch.stride,
        channels=config.model.input_channels,
    )
    loader = _loader(dataset, config, shuffle=False)
    latent_batches: list[np.ndarray] = []
    seen = 0
    for batch in tqdm(loader, desc="DDIM inversion on normal patches"):
        images = batch["image"].to(device, non_blocking=True)
        latents = ddim_invert(model, schedule, images).cpu().numpy()
        if args.max_patches is not None:
            remaining = args.max_patches - seen
            latents = latents[:remaining]
        latent_batches.append(latents)
        seen += len(latents)
        if args.max_patches is not None and seen >= args.max_patches:
            break

    all_latents = np.concatenate(latent_batches, axis=0)
    normality = NoiseLatentNormalityModel(**config.normality.__dict__).fit(all_latents)
    normality.save(args.output)
    if args.save_latents:
        latent_path = Path(args.save_latents)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(latent_path, all_latents)
    print(
        json.dumps(
            {
                "normality_model": str(args.output),
                "normal_patches": len(all_latents),
                "latent_shape": list(all_latents.shape[1:]),
                "nu": config.normality.nu,
                "gamma": config.normality.gamma,
            },
            indent=2,
        )
    )


def _save_detection(
    output_dir: Path,
    identifier: str,
    anomaly_map: np.ndarray,
    mask: np.ndarray,
) -> None:
    map_dir = output_dir / "anomaly_maps"
    preview_dir = output_dir / "anomaly_previews"
    mask_dir = output_dir / "masks"
    for directory in (map_dir, preview_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.save(map_dir / f"{identifier}.npy", anomaly_map.astype(np.float32))
    probability = 1.0 / (1.0 + np.exp(-np.clip(anomaly_map, -80.0, 80.0)))
    Image.fromarray(np.round(probability * 255.0).astype(np.uint8)).save(
        preview_dir / f"{identifier}.png"
    )
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{identifier}.png")


def _detect(args: argparse.Namespace) -> None:
    config = InvDetectConfig.from_yaml(args.config)
    seed_everything(config.seed)
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, config, device)
    normality = NoiseLatentNormalityModel.load(args.normality_model)
    pipeline = InvDetectPipeline(model, normality, config, device)

    input_path = Path(args.input)
    if input_path.is_file():
        input_root = input_path.parent
        images = [input_path]
    else:
        input_root = input_path
        images = find_images(input_path)

    output_dir = Path(args.output_dir)
    records: list[dict[str, object]] = []
    for image_path in tqdm(images, desc="InvDetect"):
        identifier = path_identifier(input_root, image_path)
        start = time.perf_counter()
        result = pipeline.detect(
            image_path,
            batch_size=args.batch_size or config.training.batch_size,
            use_scr=not args.no_scr,
        )
        elapsed = time.perf_counter() - start
        _save_detection(output_dir, identifier, result.anomaly_map, result.mask)
        records.append(
            {
                "image": str(image_path),
                "identifier": identifier,
                "patches": len(result.patch_scores),
                "seconds": elapsed,
                "abnormal_pixels": int(result.mask.sum()),
            }
        )

    summary = {
        "images": len(records),
        "scr": not args.no_scr,
        "average_seconds": float(np.mean([row["seconds"] for row in records])),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_directories(args.predictions, args.targets, threshold=args.threshold)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invdetect",
        description="InvDetect: anomaly detection in the DDIM noise latent space",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train DDIM on normal image patches.")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--train-dir", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--device", default="auto")
    train.set_defaults(handler=_train)

    fit = subparsers.add_parser("fit-normality", help="Fit the one-class SVM in noise space.")
    fit.add_argument("--config", default="configs/default.yaml")
    fit.add_argument("--train-dir", required=True)
    fit.add_argument("--checkpoint", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--save-latents")
    fit.add_argument("--max-patches", type=int)
    fit.add_argument("--device", default="auto")
    fit.set_defaults(handler=_fit_normality)

    detect = subparsers.add_parser("detect", help="Detect anomalies in an image or directory.")
    detect.add_argument("--config", default="configs/default.yaml")
    detect.add_argument("--input", required=True)
    detect.add_argument("--checkpoint", required=True)
    detect.add_argument("--normality-model", required=True)
    detect.add_argument("--output-dir", required=True)
    detect.add_argument("--batch-size", type=int)
    detect.add_argument("--no-scr", action="store_true")
    detect.add_argument("--device", default="auto")
    detect.set_defaults(handler=_detect)

    evaluate = subparsers.add_parser("evaluate", help="Compute Dice and precision.")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--targets", required=True)
    evaluate.add_argument("--threshold", type=int, default=127)
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
