"""Evaluate a trained ensemble with F1, precision, recall, and confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import AudioDataset
from .ensemble import build_model, predict
from .utils import (
    binary_metrics,
    find_optimal_threshold,
    get_device,
    load_config,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold; defaults to evaluation.threshold in the config.",
    )
    parser.add_argument(
        "--threshold-csv",
        help="Optional validation CSV used only to select an F1-optimal threshold.",
    )
    parser.add_argument("--output", default="results/metrics/test_metrics.json")
    return parser.parse_args()


def make_loader(
    csv_path: str,
    data_root: str,
    config: dict[str, Any],
    device: torch.device,
) -> DataLoader:
    """Build a deterministic evaluation loader."""
    dataset = AudioDataset(csv_path, data_root, **config["data"])
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["training"]["num_workers"] > 0,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    loader = make_loader(args.csv, args.data_root, config, device)
    model = build_model(config, pretrained_ast=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    threshold = (
        args.threshold
        if args.threshold is not None
        else config["evaluation"]["threshold"]
    )
    threshold_source = "argument" if args.threshold is not None else "config"
    if args.threshold_csv:
        threshold_loader = make_loader(
            args.threshold_csv,
            args.data_root,
            config,
            device,
        )
        validation_probabilities, validation_labels = predict(
            model, threshold_loader, device
        )
        search = config["evaluation"]["threshold_search"]
        threshold, validation_f1 = find_optimal_threshold(
            validation_labels,
            validation_probabilities,
            minimum=search["minimum"],
            maximum=search["maximum"],
            step=search["step"],
        )
        threshold_source = f"validation:{Path(args.threshold_csv).name}"
    else:
        validation_f1 = None

    probabilities, labels = predict(model, loader, device)
    metrics = binary_metrics(labels, probabilities, threshold)
    metrics["threshold_source"] = threshold_source
    if validation_f1 is not None:
        metrics["threshold_validation_f1"] = validation_f1
    save_json(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
