"""Weighted CNN14 and AST ensemble model and inference CLI.

Ensemble weights default to CNN14=0.45 / AST=0.55, matching Phase 7 of the
kit-pri-v2.ipynb Kaggle notebook (CFG.CNN14_WEIGHT / CFG.AST_WEIGHT).
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AudioDataset
from .models import CNN14, ASTBinaryClassifier
from .utils import binary_metrics, get_device, load_config, save_json


class WeightedEnsemble(nn.Module):
    """Combine CNN14 and AST logits with normalized non-negative weights.

    Default weights (CNN14=0.45, AST=0.55) match the Phase 7 Kaggle notebook.
    """

    def __init__(
        self,
        cnn14: CNN14,
        ast: ASTBinaryClassifier,
        cnn14_weight: float = 0.45,
        ast_weight: float = 0.55,
    ) -> None:
        super().__init__()
        if not all(math.isfinite(weight) for weight in (cnn14_weight, ast_weight)):
            raise ValueError("Ensemble weights must be finite")
        if cnn14_weight < 0 or ast_weight < 0 or cnn14_weight + ast_weight == 0:
            raise ValueError(
                "Ensemble weights must be non-negative with a positive sum"
            )
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
        combined = self.cnn14_weight * cnn14_logits + self.ast_weight * ast_logits
        if return_branch_logits:
            return combined, cnn14_logits, ast_logits
        return combined


def build_model(
    config: dict[str, Any], pretrained_ast: bool = True
) -> WeightedEnsemble:
    """Construct the configured dual-branch ensemble.

    Args:
        config: Loaded YAML config dict.
        pretrained_ast: If ``True``, load pretrained DeiT-Small ImageNet weights
            for the AST branch.  Set ``False`` when loading from a checkpoint.
    """
    ensemble_config = config["ensemble"]
    return WeightedEnsemble(
        cnn14=CNN14(pretrained=False),  # CNN14 trains from scratch
        ast=ASTBinaryClassifier(pretrained=pretrained_ast),
        cnn14_weight=ensemble_config["cnn14_weight"],
        ast_weight=ensemble_config["ast_weight"],
    )


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return sigmoid probabilities and integer labels for a data loader."""
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
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--output",
        default="results/metrics/ensemble_metrics.json",
    )
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
    threshold = (
        args.threshold
        if args.threshold is not None
        else config["evaluation"]["threshold"]
    )
    metrics = binary_metrics(labels, probabilities, threshold)
    save_json(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
