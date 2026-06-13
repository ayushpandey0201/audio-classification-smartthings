"""Shared configuration, reproducibility, logging, and metric utilities."""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject incomplete or internally inconsistent experiment configuration."""
    required_sections = {
        "data",
        "training",
        "augmentation",
        "ensemble",
        "evaluation",
    }
    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise ValueError(f"Missing config sections: {sorted(missing_sections)}")

    data = config["data"]
    for key in ("sample_rate", "duration", "n_mels", "n_fft", "hop_length"):
        if data.get(key, 0) <= 0:
            raise ValueError(f"data.{key} must be positive")

    training = config["training"]
    if not 0.0 <= training["label_smoothing"] < 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1)")
    if training["gradient_clip_norm"] <= 0:
        raise ValueError("training.gradient_clip_norm must be positive")

    weights = config["ensemble"]
    if weights["cnn14_weight"] < 0 or weights["ast_weight"] < 0:
        raise ValueError("Ensemble weights must be non-negative")
    if weights["cnn14_weight"] + weights["ast_weight"] == 0:
        raise ValueError("Ensemble weights must not both be zero")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a PyTorch DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    matrix = confusion_matrix(labels_array, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    specificity = (
        true_negative / specificity_denominator if specificity_denominator else 0.0
    )
    try:
        roc_auc = float(roc_auc_score(labels_array, probabilities_array))
    except ValueError:
        roc_auc = None
    return {
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(labels_array.size),
        "positive_count": int(true_positive + false_negative),
        "threshold": threshold,
    }


def find_optimal_threshold(
    labels: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    minimum: float = 0.30,
    maximum: float = 0.70,
    step: float = 0.01,
) -> tuple[float, float]:
    """Choose an F1-maximizing threshold using validation predictions only."""
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("Threshold search bounds must satisfy 0 <= min <= max <= 1")
    if step <= 0:
        raise ValueError("Threshold search step must be positive")
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float32)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(minimum, maximum + step / 2, step):
        score = f1_score(
            labels_array,
            probabilities_array >= threshold,
            zero_division=0,
        )
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary as formatted JSON, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    """Atomically save a PyTorch checkpoint in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


class EarlyStopping:
    """Track validation improvement and signal when patience is exhausted."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.wait = 0

    def update(self, metric: float) -> bool:
        """Return ``True`` when training should stop."""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience
