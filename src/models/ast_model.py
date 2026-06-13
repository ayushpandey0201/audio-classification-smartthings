"""DeiT-Small AST wrapper for binary logit prediction.

Architecture matches Phase 7 of kit-pri-v2.ipynb (Kaggle, 2026-06-09):
    - Backbone: timm ``deit_small_patch16_224`` (embed_dim=384)
    - Input: log-mel [batch, 1, mel, time]  →  repeat to 3-channel, bilinear
      resize to (224, 224), then backbone + classification head
    - Head: LayerNorm → Linear(384, 256) → GELU → Dropout(0.3) → Linear(256, 1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    import timm

    _TIMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIMM_AVAILABLE = False


class ASTBinaryClassifier(nn.Module):
    """Adapt log-mel tensors to DeiT-Small and expose a single binary logit.

    The backbone is ``deit_small_patch16_224`` from ``timm``, pretrained on
    ImageNet-21k.  Log-mel spectrograms are replicated to three channels and
    bilinear-resized to 224×224 before being passed to the transformer.

    This exactly matches the ``ASTModel`` used in the Phase 7 Kaggle notebook.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ASTBinaryClassifier. "
                "Install it with: pip install timm>=0.9"
            )
        super().__init__()
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # remove original head
            global_pool="token",  # use CLS token
        )
        embed_dim: int = self.backbone.num_features  # 384 for deit_small
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        print(f"  [AST] DeiT-Small | embed_dim={embed_dim}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_inputs(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, 1, mel, time]`` to ``[B, 3, 224, 224]``."""
        if log_mel.ndim != 4 or log_mel.shape[1] != 1:
            raise ValueError(
                "ASTBinaryClassifier expects input shaped [batch, 1, mel, time], "
                f"got {tuple(log_mel.shape)}"
            )
        # Replicate single channel → 3 channels
        x = log_mel.repeat(1, 3, 1, 1)
        # Resize to ViT patch-grid size
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        x = self._prepare_inputs(log_mel)
        features = self.backbone(x)  # [B, 384]
        return self.head(features).squeeze(-1)  # [B]

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (kept compatible with train.py)
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze DeiT and leave only the classification head trainable."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def unfreeze_last_blocks(self, count: int) -> None:
        """Train the head and final ``count`` DeiT transformer encoder blocks."""
        if count < 0:
            raise ValueError("count must be non-negative")
        self.freeze_backbone()
        blocks = list(self.backbone.blocks)  # DeiT stores blocks in .blocks
        if count > len(blocks):
            raise ValueError(
                f"Cannot unfreeze {count} DeiT blocks; model has {len(blocks)}"
            )
        for block in blocks[-count:] if count else []:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def trainable_parameters(self) -> int:
        """Return the number of parameters currently receiving gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Return the total number of model parameters."""
        return sum(p.numel() for p in self.parameters())
