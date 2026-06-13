from pathlib import Path

import yaml


def test_training_config_has_valid_ensemble_weights() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    weights = config["ensemble"]
    assert weights["cnn14_weight"] >= 0
    assert weights["ast_weight"] >= 0
    assert weights["cnn14_weight"] + weights["ast_weight"] > 0
    # Notebook Phase 7 values: CNN14=0.45, AST=0.55
    assert weights["cnn14_weight"] == 0.45
    assert weights["ast_weight"] == 0.55


def test_training_config_uses_expected_audio_shape() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data"]["sample_rate"] == 32_000
    assert config["data"]["n_mels"] == 64  # Phase 7: CFG.N_MELS = 64
    assert config["data"]["target_frames"] == 1000  # 320_000 / 320 = 1000
    assert config["data"]["hop_length"] == 320
    assert config["data"]["n_fft"] == 1024


def test_training_config_early_stopping_patience() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["training"]["early_stopping_patience"] == 15  # Phase 7: PATIENCE=15
    assert config["training"]["batch_size"] == 16  # Phase 7: BATCH_SIZE=16
