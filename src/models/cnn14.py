"""PANNs-style CNN14 implementation for binary audio classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolutional layers followed by batch normalization and pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(
        self, inputs: torch.Tensor, pool_size: tuple[int, int] = (2, 2)
    ) -> torch.Tensor:
        outputs = F.relu_(self.bn1(self.conv1(inputs)))
        outputs = F.relu_(self.bn2(self.conv2(outputs)))
        return F.avg_pool2d(outputs, kernel_size=pool_size)


class CNN14(nn.Module):
    """CNN14-inspired network trained from scratch on log-mel spectrograms."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        channels = (1, 64, 128, 256, 512, 1024, 2048)
        self.blocks = nn.ModuleList(
            ConvBlock(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)
        )
        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2048, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Extract pooled embeddings from ``[batch, 1, mel, time]`` input."""
        outputs = log_mel
        for index, block in enumerate(self.blocks):
            pool_size = (2, 2) if index < len(self.blocks) - 1 else (1, 1)
            outputs = self.dropout(block(outputs, pool_size))
        outputs = outputs.mean(dim=3)
        outputs = outputs.max(dim=2).values + outputs.mean(dim=2)
        return self.dropout(F.relu_(self.fc1(outputs)))

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(log_mel)).squeeze(-1)

    def freeze_backbone(self) -> None:
        """Freeze feature layers while leaving the classification head trainable."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.fc1, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Enable gradient updates for the complete CNN14 branch."""
        for parameter in self.parameters():
            parameter.requires_grad = True
