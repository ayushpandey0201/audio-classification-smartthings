import pytest
import torch
from torch import nn

from src.ensemble import WeightedEnsemble


class ConstantModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full((inputs.shape[0],), self.value, device=inputs.device)


def test_weighted_ensemble_combines_branch_logits() -> None:
    ensemble = WeightedEnsemble(
        cnn14=ConstantModel(2.0),
        ast=ConstantModel(4.0),
        cnn14_weight=0.25,
        ast_weight=0.75,
    )

    output = ensemble(torch.zeros(3, 1, 64, 1000))

    assert torch.equal(output, torch.full((3,), 3.5))


def test_weighted_ensemble_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        WeightedEnsemble(
            cnn14=ConstantModel(0.0),
            ast=ConstantModel(0.0),
            cnn14_weight=-1.0,
            ast_weight=1.0,
        )
