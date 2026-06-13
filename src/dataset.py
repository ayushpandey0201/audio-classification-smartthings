"""Dataset and log-mel feature extraction for kitpri_v2.

Audio parameters match Phase 7 of kit-pri-v2.ipynb (Kaggle, 2026-06-09):
    - sample_rate = 32 000 Hz
    - duration    = 10 s (320 000 samples)
    - n_mels      = 64
    - hop_length  = 320
    - n_fft       = 1024
    - f_min       = 50 Hz
    - f_max       = 14 000 Hz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

PATH_COLUMNS = ("path", "file_path", "filepath", "file", "audio_path", "filename")
LABEL_COLUMNS = ("label", "target", "class", "is_cooking")
STRING_LABELS = {
    "non-cooking": 0.0,
    "noncooking": 0.0,
    "non_cooking": 0.0,
    "cooking": 1.0,
}


class AudioDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load fixed-length WAV clips and return normalized log-mel tensors.

    The metadata CSV must contain one recognized path column and one recognized
    label column.  Relative paths are resolved against ``data_root``.

    Default audio parameters are aligned with the kitpri_v2 Kaggle training
    notebook (n_mels=64, hop_length=320, n_fft=1024, sr=32 kHz, 10 s clips).
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        data_root: str | Path,
        sample_rate: int = 32_000,
        duration: float = 10.0,
        n_mels: int = 64,  # notebook CFG.N_MELS = 64
        hop_length: int = 320,  # notebook CFG.HOP_LENGTH = 320
        n_fft: int = 1024,  # notebook CFG.N_FFT = 1024
        target_frames: int = 1000,  # 320_000 / 320 = 1000
        f_min: float = 50.0,  # notebook CFG.F_MIN = 50
        f_max: float | None = 14_000.0,  # notebook CFG.F_MAX = 14000
        augment: bool = False,
        frequency_mask_param: int = 24,
        time_mask_param: int = 96,
        random_crop: bool = False,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.metadata_csv.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"Audio root not found: {self.data_root}")
        self.metadata = pd.read_csv(self.metadata_csv)
        if self.metadata.empty:
            raise ValueError(f"Metadata CSV is empty: {self.metadata_csv}")
        self.path_column = self._find_column(PATH_COLUMNS)
        self.label_column = self._find_column(LABEL_COLUMNS)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.target_frames = target_frames
        self.random_crop = random_crop
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )
        self.frequency_mask = (
            torchaudio.transforms.FrequencyMasking(frequency_mask_param)
            if augment
            else None
        )
        self.time_mask = (
            torchaudio.transforms.TimeMasking(time_mask_param) if augment else None
        )

    def _find_column(self, candidates: tuple[str, ...]) -> str:
        for candidate in candidates:
            if candidate in self.metadata.columns:
                return candidate
        raise ValueError(
            f"{self.metadata_csv} must contain one of {candidates}; "
            f"found {list(self.metadata.columns)}"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_audio_path(self, value: Any) -> Path:
        relative_path = Path(str(value))
        if relative_path.is_absolute():
            raise ValueError(
                f"Metadata paths must be relative to data_root: {relative_path}"
            )
        resolved = (self.data_root / relative_path).resolve()
        if not resolved.is_relative_to(self.data_root):
            raise ValueError(f"Audio path escapes data_root: {relative_path}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Audio file not found: {resolved}")
        return resolved

    def _load_audio(self, path: Path) -> torch.Tensor:
        waveform, source_rate = torchaudio.load(path)
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {path}")
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Audio file contains non-finite samples: {path}")
        waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_rate, self.sample_rate
            )
        if waveform.shape[-1] < self.num_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.num_samples - waveform.shape[-1])
            )
        elif waveform.shape[-1] > self.num_samples:
            maximum_start = waveform.shape[-1] - self.num_samples
            if self.random_crop:
                start = int(torch.randint(maximum_start + 1, (1,)).item())
            else:
                start = maximum_start // 2
            waveform = waveform[..., start : start + self.num_samples]
        return waveform

    def _fix_frame_count(self, log_mel: torch.Tensor) -> torch.Tensor:
        frames = log_mel.shape[-1]
        if frames < self.target_frames:
            return torch.nn.functional.pad(log_mel, (0, self.target_frames - frames))
        return log_mel[..., : self.target_frames]

    @staticmethod
    def _parse_label(value: Any) -> float:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in STRING_LABELS:
                return STRING_LABELS[normalized]
        numeric = float(value)
        if numeric not in (0.0, 1.0):
            raise ValueError(f"Expected a binary label, received {value!r}")
        return numeric

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        audio_path = self._resolve_audio_path(row[self.path_column])
        waveform = self._load_audio(audio_path)
        log_mel = self.db_transform(self.mel_transform(waveform))
        # Normalize to zero mean / unit variance (matches notebook intent)
        log_mel = (log_mel - log_mel.mean()) / log_mel.std().clamp_min(1e-6)
        log_mel = self._fix_frame_count(log_mel)
        if self.frequency_mask is not None:
            log_mel = self.frequency_mask(log_mel)
        if self.time_mask is not None:
            log_mel = self.time_mask(log_mel)
        label = torch.tensor(
            self._parse_label(row[self.label_column]), dtype=torch.float32
        )
        return log_mel, label
