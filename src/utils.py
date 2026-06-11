"""Shared configuration, reproducibility, logging, and metric utilities."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "kitchen-audio") -> logging.Logger:
    """Create a concise console logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger(name)


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smooth_binary_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply symmetric label smoothing to binary targets."""
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def binary_metrics(
    labels: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary metrics and a JSON-serializable confusion matrix."""
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities_array >= threshold).astype(np.int64)
    return {
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels_array, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels_array, predictions, labels=[0, 1]
        ).tolist(),
        "threshold": threshold,
    }


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary as formatted JSON, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
