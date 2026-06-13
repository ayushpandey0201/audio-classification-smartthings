"""PANNs-style CNN14 implementation for binary audio classification.

Architecture matches Phase 7 of kit-pri-v2.ipynb (Kaggle, 2026-06-09):
    - Input BatchNorm2d (``bn0``) on the log-mel spectrogram
    - Six ConvBlock double-conv stages; blocks 1–5 average-pool ×2; block 6 no pool
    - AdaptiveAvgPool2d(1) → flatten → fc1(2048, 2048) → Dropout(0.5) → fc_out(2048, 1)
    - Optional soft-loading of PANNs pretrained weights (try/except, 3 layers match)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Optional PANNs weight loading
# ---------------------------------------------------------------------------


def _try_load_panns_weights(model: CNN14) -> None:
    """Attempt to transfer PANNs Cnn14 weights into ``model`` (best-effort)."""
    try:
        from panns_inference import AudioTagging  # type: ignore[import-not-found]

        at = AudioTagging(checkpoint_path=None, device="cpu")
        panns_sd = at.model.state_dict()
        our_sd = model.state_dict()
        matched, skipped = 0, 0
        for k, v in panns_sd.items():
            if k in our_sd and our_sd[k].shape == v.shape:
                our_sd[k] = v
                matched += 1
            else:
                skipped += 1
        model.load_state_dict(our_sd, strict=False)
        print(f"  [CNN14] PANNs weights: {matched} matched, {skipped} skipped")
    except Exception as exc:  # noqa: BLE001
        print(f"  [CNN14] PANNs unavailable ({exc}), training from scratch")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Two Conv2d → BN → ReLU layers followed by optional average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(
        self, inputs: torch.Tensor, pool_size: tuple[int, int] = (2, 2)
    ) -> torch.Tensor:
        outputs = F.relu_(self.bn1(self.conv1(inputs)))
        outputs = F.relu_(self.bn2(self.conv2(outputs)))
        if pool_size != (1, 1):
            outputs = F.avg_pool2d(outputs, kernel_size=pool_size)
        return outputs


# ---------------------------------------------------------------------------
# CNN14
# ---------------------------------------------------------------------------


class CNN14(nn.Module):
    """CNN14-style network for binary kitchen audio classification.

    Trained from scratch on log-mel spectrograms.  Architecture mirrors the
    Kaggle Phase 7 notebook exactly, including the input ``bn0`` layer and
    global average pooling head.
    """

    def __init__(
        self,
        dropout: float = 0.5,
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            dropout: Dropout probability used after the global pooling and fc1.
            pretrained: If ``True``, attempt to load PANNs Cnn14 weights
                (soft-load; only layers with matching shapes are transferred).
        """
        super().__init__()
        # Input batch normalization — same as notebook's self.bn0
        self.bn0 = nn.BatchNorm2d(1)

        channels = (1, 64, 128, 256, 512, 1024, 2048)
        self.blocks = nn.ModuleList(
            ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2048, 1)
        self._initialize_weights()

        if pretrained:
            _try_load_panns_weights(self)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim pooled embeddings from ``[B, 1, mel, time]`` input."""
        x = self.bn0(log_mel)
        for index, block in enumerate(self.blocks):
            # Blocks 0–4 downsample; block 5 (last) keeps spatial dims
            pool = (2, 2) if index < len(self.blocks) - 1 else (1, 1)
            x = block(x, pool_size=pool)
        x = self.gap(x).flatten(1)  # [B, 2048]
        return self.dropout(F.relu_(self.fc1(x)))

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(log_mel)).squeeze(-1)

    # ------------------------------------------------------------------
    # Freeze / unfreeze (used by staged training in train.py)
    # ------------------------------------------------------------------

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
