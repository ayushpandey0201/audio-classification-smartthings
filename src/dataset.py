"""Dataset and log-mel feature extraction for kitpri_v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


PATH_COLUMNS = ("path", "filepath", "file", "audio_path", "filename")
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
    label column. Relative paths are resolved against ``data_root``.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        data_root: str | Path,
        sample_rate: int = 32_000,
        duration: float = 10.0,
        n_mels: int = 128,
        hop_length: int = 320,
        n_fft: int = 1024,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.data_root = Path(data_root)
        self.metadata = pd.read_csv(self.metadata_csv)
        self.path_column = self._find_column(PATH_COLUMNS)
        self.label_column = self._find_column(LABEL_COLUMNS)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

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

    def _load_audio(self, path: Path) -> torch.Tensor:
        waveform, source_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_rate, self.sample_rate
            )
        if waveform.shape[-1] < self.num_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.num_samples - waveform.shape[-1])
            )
        return waveform[..., : self.num_samples]

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
        audio_path = Path(str(row[self.path_column]))
        if not audio_path.is_absolute():
            audio_path = self.data_root / audio_path
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform = self._load_audio(audio_path)
        log_mel = self.db_transform(self.mel_transform(waveform))
        log_mel = (log_mel - log_mel.mean()) / log_mel.std().clamp_min(1e-6)
        label = torch.tensor(self._parse_label(row[self.label_column]))
        return log_mel, label
