"""Train the CNN14 + AST ensemble with staged backbone unfreezing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AudioDataset
from .ensemble import WeightedEnsemble, build_model, predict
from .utils import (
    binary_metrics,
    get_device,
    get_logger,
    load_config,
    save_json,
    set_seed,
    smooth_binary_targets,
)

LOGGER = get_logger()


def make_loader(
    csv_path: str,
    data_root: str,
    config: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    """Create a dataset loader from the shared project configuration."""
    dataset = AudioDataset(csv_path, data_root, **config["data"])
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["training"]["num_workers"] > 0,
    )


def configure_stage(model: WeightedEnsemble, stage: int, config: dict) -> dict:
    """Apply freezing rules and return the selected stage configuration."""
    if stage == 1:
        stage_config = config["training"]["stage1"]
        model.cnn14.freeze_backbone()
        model.ast.freeze_backbone()
    elif stage == 2:
        stage_config = config["training"]["stage2"]
        if stage_config["unfreeze_cnn14"]:
            model.cnn14.unfreeze_backbone()
        model.ast.unfreeze_last_blocks(stage_config["unfreeze_ast_layers"])
    else:
        raise ValueError(f"Unsupported training stage: {stage}")
    return stage_config


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    label_smoothing: float,
) -> float:
    """Run one optimization epoch and return mean loss."""
    model.train()
    running_loss = 0.0
    for features, labels in tqdm(loader, desc="Train", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        smoothed_labels = smooth_binary_targets(labels, label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, smoothed_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.shape[0]
    return running_loss / len(loader.dataset)


def run_stage(
    stage: int,
    model: WeightedEnsemble,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
    best_f1: float,
) -> float:
    """Train one stage and save the best validation checkpoint."""
    stage_config = configure_stage(model, stage, config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=stage_config["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = None
    if stage_config.get("scheduler") == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=stage_config["epochs"])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config["training"]["pos_weight"], device=device)
    )

    for epoch in range(1, stage_config["epochs"] + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config["training"]["label_smoothing"],
        )
        probabilities, labels = predict(model, val_loader, device)
        metrics = binary_metrics(labels, probabilities)
        LOGGER.info(
            "stage=%d epoch=%d/%d loss=%.4f val_f1=%.4f",
            stage,
            epoch,
            stage_config["epochs"],
            loss,
            metrics["f1"],
        )
        save_json(
            {"stage": stage, "epoch": epoch, "train_loss": loss, **metrics},
            output_dir / "metrics" / f"stage{stage}_epoch{epoch:02d}.json",
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stage": stage,
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "config": config,
                },
                output_dir / "checkpoints" / "best.pt",
            )
        if scheduler is not None:
            scheduler.step()
    return best_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--ast-from-scratch",
        action="store_true",
        help="Initialize AST from config instead of downloading pretrained weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    LOGGER.info("Using device: %s", device)

    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args.train_csv, args.data_root, config, shuffle=True)
    val_loader = make_loader(args.val_csv, args.data_root, config, shuffle=False)
    model = build_model(config, pretrained_ast=not args.ast_from_scratch).to(device)

    best_f1 = -1.0
    best_f1 = run_stage(
        1, model, train_loader, val_loader, config, device, output_dir, best_f1
    )
    best_f1 = run_stage(
        2, model, train_loader, val_loader, config, device, output_dir, best_f1
    )
    LOGGER.info("Training complete. Best validation F1: %.4f", best_f1)


if __name__ == "__main__":
    main()
