"""
Audio preprocessing + inference wrapper for Telegram bot.
Uses the dual-branch ensemble model defined in src.
"""

import os
import subprocess
import tempfile

import soundfile as sf
import torch
import torchaudio.transforms as T

from src.ensemble import build_model
from src.utils import load_config
import timm
import torch.nn.functional as F
from torch import nn

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
            nn.LayerNorm(embed_dim),    # 0
            nn.Dropout(0.3),            # 1 (No weights)
            nn.Linear(embed_dim, 1),    # 2
        )

    def _prepare_inputs(self, log_mel: torch.Tensor) -> torch.Tensor:
        x = log_mel.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        x = self._prepare_inputs(log_mel)
        features = self.backbone(x)
        return self.head(features).squeeze(-1)


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ensemble_model(ckpt_path: str, config_path: str, device: torch.device):
    """Load config, build the ensemble (or AST), and load checkpoint weights."""
    config = load_config(config_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if "cfg" in checkpoint:
        cfg = checkpoint["cfg"]
        if "n_mels" in cfg: config["data"]["n_mels"] = cfg["n_mels"]
        if "clip_sec" in cfg: config["data"]["duration"] = cfg["clip_sec"]
        if "sr" in cfg: config["data"]["sample_rate"] = cfg["sr"]
        if "hop" in cfg: config["data"]["hop_length"] = cfg["hop"]
        if "n_fft" in cfg: config["data"]["n_fft"] = cfg["n_fft"]
        if "fmin" in cfg: config["data"]["f_min"] = cfg["fmin"]
        if "fmax" in cfg: config["data"]["f_max"] = cfg["fmax"]
        # Re-compute target frames based on updated params
        config["data"]["target_frames"] = int((config["data"]["sample_rate"] * config["data"]["duration"]) / config["data"]["hop_length"])

    state_dict = checkpoint.get("model", checkpoint.get("model_state", checkpoint))
    
    # Check if checkpoint contains Ensemble weights or just AST
    if any(k.startswith("ast.") or k.startswith("cnn14.") for k in state_dict.keys()):
        model = build_model(config, pretrained_ast=False)
    else:
        # Detect if it's the old legacy head architecture
        if "head.1.weight" not in state_dict and "head.2.weight" in state_dict:
            model = LegacyASTBinaryClassifier(pretrained=False)
        else:
            from src.models.ast_model import ASTBinaryClassifier
            model = ASTBinaryClassifier(pretrained=False)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def load_and_resample(path: str, sample_rate: int, clip_duration: int) -> torch.Tensor:
    """Load audio, convert to mono, resample, and pad/trim to expected length."""
    ext = os.path.splitext(path)[1].lower()
    tmp_wav = None
    clip_samples = int(sample_rate * clip_duration)

    try:
        # Convert ogg/m4a/opus (voice messages) to wav via ffmpeg
        if ext in {".ogg", ".m4a", ".opus", ".webm"}:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                pass  # Just create it to get the name
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    path,
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    tmp_wav.name,
                ],
                capture_output=True,
                check=True,
            )
            load_path = tmp_wav.name
        else:
            load_path = path

        data, sr = sf.read(load_path, dtype="float32")
        waveform = torch.from_numpy(data)
        waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.t()

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != sample_rate:
            waveform = T.Resample(sr, sample_rate)(waveform)

        # Pad or trim
        n = waveform.shape[1]
        if n >= clip_samples:
            waveform = waveform[:, :clip_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, clip_samples - n))

        return waveform  # [1, clip_samples]

    finally:
        if tmp_wav and os.path.exists(tmp_wav.name):
            os.unlink(tmp_wav.name)


def waveform_to_log_mel(waveform: torch.Tensor, config: dict) -> torch.Tensor:
    """Extract log-mel spectrogram matching the training pipeline."""
    d = config["data"]
    mel_transform = T.MelSpectrogram(
        sample_rate=d["sample_rate"],
        n_fft=d["n_fft"],
        hop_length=d["hop_length"],
        n_mels=d["n_mels"],
        f_min=d["f_min"],
        f_max=d["f_max"],
        power=2.0,
    )
    db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    # Compute mel
    log_mel = db_transform(mel_transform(waveform))
    
    # Normalize (matches dataset.py intent)
    log_mel = (log_mel - log_mel.mean()) / log_mel.std().clamp_min(1e-6)

    # Ensure correct frame count
    target_frames = d["target_frames"]
    current_frames = log_mel.shape[-1]
    if current_frames > target_frames:
        log_mel = log_mel[..., :target_frames]
    elif current_frames < target_frames:
        log_mel = torch.nn.functional.pad(log_mel, (0, target_frames - current_frames))

    return log_mel


def predict(
    path: str,
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """End-to-end inference for a single file."""
    # 1. Load waveform
    waveform = load_and_resample(
        path, config["data"]["sample_rate"], config["data"]["duration"]
    )
    
    # 2. Extract features
    log_mel = waveform_to_log_mel(waveform, config)
    
    # 3. Add batch dimension: [1, 1, n_mels, time]
    log_mel = log_mel.unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        logit = model(log_mel)
        prob = torch.sigmoid(logit).item()
        
    label = "Cooking" if prob >= threshold else "Not Cooking"
    return {"label": label, "prob": prob}