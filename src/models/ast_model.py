"""Hugging Face AST wrapper for binary logit prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import ASTConfig, ASTForAudioClassification


class ASTBinaryClassifier(nn.Module):
    """Adapt log-mel tensors to AST and expose a single binary logit."""

    def __init__(
        self,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        max_length: int = 1024,
        n_mels: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.n_mels = n_mels
        if pretrained:
            self.model = ASTForAudioClassification.from_pretrained(
                pretrained_model,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
        else:
            config = ASTConfig(
                num_labels=1,
                max_length=max_length,
                num_mel_bins=n_mels,
            )
            self.model = ASTForAudioClassification(config)

    def _prepare_inputs(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.ndim != 4 or log_mel.shape[1] != 1:
            raise ValueError("AST expects log-mel input shaped [batch, 1, mel, time]")
        resized = F.interpolate(
            log_mel,
            size=(self.n_mels, self.max_length),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(1).transpose(1, 2)

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare_inputs(log_mel)
        return self.model(input_values=inputs).logits.squeeze(-1)

    def freeze_backbone(self) -> None:
        """Freeze AST and leave only the classification head trainable."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_last_blocks(self, count: int) -> None:
        """Train the classifier and final ``count`` transformer encoder blocks."""
        self.freeze_backbone()
        layers = self.model.audio_spectrogram_transformer.encoder.layer
        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
