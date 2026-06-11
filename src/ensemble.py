"""Weighted CNN14 and AST ensemble model and inference CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AudioDataset
from .models import ASTBinaryClassifier, CNN14
from .utils import binary_metrics, get_device, load_config


class WeightedEnsemble(nn.Module):
    """Combine CNN14 and AST logits with normalized non-negative weights."""

    def __init__(
        self,
        cnn14: CNN14,
        ast: ASTBinaryClassifier,
        cnn14_weight: float = 0.35,
        ast_weight: float = 0.65,
    ) -> None:
        super().__init__()
        if cnn14_weight < 0 or ast_weight < 0 or cnn14_weight + ast_weight == 0:
            raise ValueError("Ensemble weights must be non-negative with a positive sum")
        total = cnn14_weight + ast_weight
        self.cnn14 = cnn14
        self.ast = ast
        self.cnn14_weight = cnn14_weight / total
        self.ast_weight = ast_weight / total

    def forward(
        self, log_mel: torch.Tensor, return_branch_logits: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cnn14_logits = self.cnn14(log_mel)
        ast_logits = self.ast(log_mel)
        combined = (
            self.cnn14_weight * cnn14_logits + self.ast_weight * ast_logits
        )
        if return_branch_logits:
            return combined, cnn14_logits, ast_logits
        return combined


def build_model(config: dict, pretrained_ast: bool = True) -> WeightedEnsemble:
    """Construct the configured dual-branch ensemble."""
    ensemble_config = config["ensemble"]
    return WeightedEnsemble(
        cnn14=CNN14(),
        ast=ASTBinaryClassifier(
            n_mels=config["data"]["n_mels"],
            pretrained=pretrained_ast,
        ),
        cnn14_weight=ensemble_config["cnn14_weight"],
        ast_weight=ensemble_config["ast_weight"],
    )


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities and labels for a data loader."""
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    for features, targets in tqdm(loader, desc="Inference", leave=False):
        logits = model(features.to(device))
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(targets.int().tolist())
    return np.asarray(probabilities), np.asarray(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cnn14-weight", type=float)
    parser.add_argument("--ast-weight", type=float)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.cnn14_weight is not None:
        config["ensemble"]["cnn14_weight"] = args.cnn14_weight
    if args.ast_weight is not None:
        config["ensemble"]["ast_weight"] = args.ast_weight

    device = get_device()
    model = build_model(config, pretrained_ast=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dataset = AudioDataset(
        args.csv,
        args.data_root,
        **config["data"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    probabilities, labels = predict(model, loader, device)
    print(json.dumps(binary_metrics(labels, probabilities, args.threshold), indent=2))


if __name__ == "__main__":
    main()
