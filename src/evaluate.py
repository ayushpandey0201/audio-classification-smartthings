"""Evaluate a trained ensemble with F1, precision, recall, and confusion matrix."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .dataset import AudioDataset
from .ensemble import build_model, predict
from .utils import binary_metrics, get_device, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="results/metrics/test_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    dataset = AudioDataset(args.csv, args.data_root, **config["data"])
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = build_model(config, pretrained_ast=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    probabilities, labels = predict(model, loader, device)
    metrics = binary_metrics(labels, probabilities, args.threshold)
    save_json(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
