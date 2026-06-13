"""
Example inference script.

Shows how to load the ensemble model and perform classification on a single audio file.
"""

import argparse
from pathlib import Path

from src.ensemble import build_model
from src.utils import get_device, load_config

import torchaudio.transforms as T
import timm
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

class LegacyASTBinaryClassifier(nn.Module):
    """Exactly matches the architecture of the old Kaggle checkpoint."""
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
            
        embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1),
        )

    def _prepare_inputs(self, log_mel: torch.Tensor) -> torch.Tensor:
        x = log_mel.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        x = self._prepare_inputs(log_mel)
        features = self.backbone(x)
        return self.head(features).squeeze(-1)

# Import the prediction logic from the bot (which includes audio preprocessing)
from bot.inference import load_and_resample, waveform_to_log_mel
import torch

def main():
    parser = argparse.ArgumentParser(description="Kitchen Audio Classifier Example")
    parser.add_argument("audio_path", type=str, help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--ckpt", default="results/checkpoints/best.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    # 1. Load Config & Device
    config = load_config(args.config)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Load Model
    print("Loading model...")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    if "cfg" in checkpoint:
        cfg = checkpoint["cfg"]
        if "n_mels" in cfg: config["data"]["n_mels"] = cfg["n_mels"]
        if "clip_sec" in cfg: config["data"]["duration"] = cfg["clip_sec"]
        if "sr" in cfg: config["data"]["sample_rate"] = cfg["sr"]
        if "hop" in cfg: config["data"]["hop_length"] = cfg["hop"]
        if "n_fft" in cfg: config["data"]["n_fft"] = cfg["n_fft"]
        if "fmin" in cfg: config["data"]["f_min"] = cfg["fmin"]
        if "fmax" in cfg: config["data"]["f_max"] = cfg["fmax"]
        config["data"]["target_frames"] = int((config["data"]["sample_rate"] * config["data"]["duration"]) / config["data"]["hop_length"])

    state_dict = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
    
    if any(k.startswith("ast.") or k.startswith("cnn14.") for k in state_dict.keys()):
        model = build_model(config, pretrained_ast=False)
    else:
        if "head.1.weight" not in state_dict and "head.2.weight" in state_dict:
            model = LegacyASTBinaryClassifier(pretrained=False)
        else:
            from src.models.ast_model import ASTBinaryClassifier
            model = ASTBinaryClassifier(pretrained=False)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Preprocess Audio
    print(f"Processing audio: {args.audio_path}")
    waveform = load_and_resample(args.audio_path, config["data"]["sample_rate"], config["data"]["duration"])
    log_mel = waveform_to_log_mel(waveform, config)
    
    # [Batch=1, Channel=1, Mels=64, Time=1000]
    log_mel = log_mel.unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        logit = model(log_mel)
        prob = torch.sigmoid(logit).item()

    threshold = config["evaluation"]["threshold"]
    label = "Cooking" if prob >= threshold else "Not Cooking"
    
    print("-" * 30)
    print(f"Prediction : {label}")
    print(f"Confidence : {prob:.2%}")
    print("-" * 30)


if __name__ == "__main__":
    main()
