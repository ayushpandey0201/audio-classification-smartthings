"""Train the CNN14 + AST ensemble with staged backbone unfreezing.

Training strategy matches kit-pri-v2.ipynb Phase 7 (Kaggle, 2026-06-09):
    - Stage 1 (8 epochs):  CNN14 from scratch, AST head-only; OneCycleLR
    - Stage 2 (≤42 epochs): CNN14 fully trainable, AST last-4-blocks unfrozen;
                             OneCycleLR; early stopping patience=15
    - BCEWithLogitsLoss with pos_weight=1.2
    - AdamW, weight_decay=1e-4, gradient_clip_norm=1.0
    - Mixed-precision (CUDA AMP)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AudioDataset
from .ensemble import WeightedEnsemble, build_model, predict
from .utils import (
    EarlyStopping,
    binary_metrics,
    get_device,
    get_logger,
    load_config,
    save_checkpoint,
    save_json,
    seed_worker,
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
    augmentation = config["augmentation"]
    dataset = AudioDataset(
        csv_path,
        data_root,
        **config["data"],
        augment=shuffle and augmentation["spec_augment"],
        frequency_mask_param=augmentation["frequency_mask_param"],
        time_mask_param=augmentation["time_mask_param"],
        random_crop=shuffle,
    )
    generator = torch.Generator()
    generator.manual_seed(config["training"]["seed"])
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config["training"]["num_workers"] > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def configure_stage(
    model: WeightedEnsemble, stage: int, config: dict[str, Any]
) -> dict[str, Any]:
    """Apply freezing rules and return the selected stage configuration."""
    if stage == 1:
        stage_config = config["training"]["stage1"]
        if stage_config["train_cnn14"]:
            model.cnn14.unfreeze_backbone()
        else:
            model.cnn14.freeze_backbone()
        if stage_config["freeze_ast_backbone"]:
            model.ast.freeze_backbone()
    elif stage == 2:
        stage_config = config["training"]["stage2"]
        if stage_config["unfreeze_cnn14"]:
            model.cnn14.unfreeze_backbone()
        model.ast.unfreeze_last_blocks(stage_config["unfreeze_ast_layers"])
    else:
        raise ValueError(f"Unsupported training stage: {stage}")
    return stage_config


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    stage_config: dict[str, Any],
    train_loader: DataLoader,
) -> CosineAnnealingLR | OneCycleLR | None:
    """Build the LR scheduler requested in the stage config."""
    scheduler_name = stage_config.get("scheduler", "")
    if scheduler_name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=stage_config["epochs"])
    if scheduler_name == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=stage_config["lr"],
            steps_per_epoch=len(train_loader),
            epochs=stage_config["epochs"],
            pct_start=0.1,
            anneal_strategy="cos",
        )
    return None


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    label_smoothing: float,
    mixup_alpha: float,
    gradient_clip_norm: float,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    scheduler: Any = None,
    is_onecycle: bool = False,
) -> float:
    """Run one optimization epoch and return mean loss.

    When ``is_onecycle=True``, the scheduler is stepped after every batch
    (OneCycleLR behavior), otherwise it is stepped after the epoch.
    """
    model.train()
    running_loss = 0.0
    for features, labels in tqdm(loader, desc="Train", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_alpha > 0 and features.shape[0] > 1:
            mixing = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(features.shape[0], device=device)
            features = mixing * features + (1.0 - mixing) * features[permutation]
            labels = mixing * labels + (1.0 - mixing) * labels[permutation]
        smoothed_labels = smooth_binary_targets(labels, label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=mixed_precision,
        ):
            logits = model(features)
            loss = criterion(logits, smoothed_labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        # OneCycleLR: step per batch
        if is_onecycle and scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * features.shape[0]
    return running_loss / len(loader.dataset)


def run_stage(
    stage: int,
    model: WeightedEnsemble,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    best_f1: float,
) -> float:
    """Train one stage and save the best validation checkpoint."""
    stage_config = configure_stage(model, stage, config)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=stage_config["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = _build_scheduler(optimizer, stage_config, train_loader)
    is_onecycle = stage_config.get("scheduler") == "OneCycleLR"

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config["training"]["pos_weight"], device=device)
    )
    mixed_precision = config["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"]["early_stopping_min_delta"],
    )

    for epoch in range(1, stage_config["epochs"] + 1):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config["training"]["label_smoothing"],
            config["augmentation"]["mixup_alpha"],
            config["training"]["gradient_clip_norm"],
            scaler,
            mixed_precision,
            scheduler=scheduler,
            is_onecycle=is_onecycle,
        )
        probabilities, labels = predict(model, val_loader, device)
        threshold = config["evaluation"]["threshold"]
        metrics = binary_metrics(labels, probabilities, threshold)
        LOGGER.info(
            "stage=%d epoch=%d/%d loss=%.4f val_f1=%.4f val_auc=%s",
            stage,
            epoch,
            stage_config["epochs"],
            loss,
            metrics["f1"],
            (f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "n/a"),
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "stage": stage,
            "epoch": epoch,
            "val_metrics": metrics,
            "config": config,
        }
        save_checkpoint(
            checkpoint,
            output_dir / "checkpoints" / "latest.pt",
        )
        save_json(
            {"stage": stage, "epoch": epoch, "train_loss": loss, **metrics},
            output_dir / "metrics" / f"stage{stage}_epoch{epoch:02d}.json",
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(
                checkpoint,
                output_dir / "checkpoints" / "best.pt",
            )
        # CosineAnnealingLR: step per epoch
        if not is_onecycle and scheduler is not None:
            scheduler.step()
        if early_stopping.update(metrics["f1"]):
            LOGGER.info("Early stopping stage %d after epoch %d", stage, epoch)
            break
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
        help=(
            "Initialize AST (DeiT) from random weights instead of ImageNet pretrained."
        ),
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

    LOGGER.info(
        "Total parameters: %.1f M",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    best_f1 = -1.0
    best_f1 = run_stage(
        1, model, train_loader, val_loader, config, device, output_dir, best_f1
    )
    best_checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state"])
    best_f1 = run_stage(
        2, model, train_loader, val_loader, config, device, output_dir, best_f1
    )
    LOGGER.info("Training complete. Best validation F1: %.4f", best_f1)


if __name__ == "__main__":
    main()
