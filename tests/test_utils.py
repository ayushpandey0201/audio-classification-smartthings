import numpy as np
import pytest

from src.utils import binary_metrics, find_optimal_threshold, validate_config


def test_binary_metrics_for_perfect_predictions() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_threshold_search_returns_best_validation_threshold() -> None:
    # labels: 0,0,1,1  probs: 0.1, 0.4, 0.6, 0.9
    # At thresh=0.45 → preds: 0,0,1,1 (perfect F1=1.0)
    threshold, score = find_optimal_threshold(
        labels=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.1, 0.4, 0.6, 0.9]),
        minimum=0.3,
        maximum=0.7,
        step=0.05,
    )

    assert 0.3 <= threshold <= 0.7
    assert score == 1.0


def test_validate_config_rejects_negative_ensemble_weights() -> None:
    config = {
        "data": {
            "sample_rate": 32_000,
            "duration": 10,
            "n_mels": 64,
            "n_fft": 1024,
            "hop_length": 320,
        },
        "training": {
            "label_smoothing": 0.05,
            "gradient_clip_norm": 1.0,
        },
        "augmentation": {},
        "ensemble": {"cnn14_weight": -0.5, "ast_weight": 1.0},
        "evaluation": {},
    }

    with pytest.raises(ValueError, match="non-negative"):
        validate_config(config)


def test_validate_config_rejects_zero_ensemble_weights() -> None:
    config = {
        "data": {
            "sample_rate": 32_000,
            "duration": 10,
            "n_mels": 64,
            "n_fft": 1024,
            "hop_length": 320,
        },
        "training": {
            "label_smoothing": 0.05,
            "gradient_clip_norm": 1.0,
        },
        "augmentation": {},
        "ensemble": {"cnn14_weight": 0.0, "ast_weight": 0.0},
        "evaluation": {},
    }

    with pytest.raises(ValueError, match="both be zero"):
        validate_config(config)
