import torch as th
import torch.nn as nn
from typing import Self


class LoRAfiedLinear(nn.Module):
    """
    Approximation of a weight matrix as a sum of a base weight matrix and a low-rank matrix.
    Can be computed automatically using from_weight_delta().
    """
    def __init__(self, base: nn.Module, up_proj: th.Tensor, down_proj: th.Tensor) -> None:
        super().__init__()

        assert len(up_proj.shape) == 2, "up_proj should be 2D tensor of weights"
        assert len(down_proj.shape) == 2, "down_proj should be 2D tensor of weights"
        assert up_proj.shape[1] == down_proj.shape[0], "up_proj and down_proj should have same hidden dim"

        if isinstance(base, nn.Linear) or isinstance(base, LoRAfiedLinear):
            assert up_proj.shape[0] == base.weight.shape[0], "up_proj should have same output dim as base"
            assert down_proj.shape[1] == base.weight.shape[1], "down_proj should have same input dim as base"

        self.base = base
        self.up_proj = nn.Linear(up_proj.shape[1], up_proj.shape[0], bias=False)
        self.down_proj = nn.Linear(down_proj.shape[1], down_proj.shape[0], bias=False)

        with th.no_grad():
            self.up_proj.weight.copy_(up_proj)
            self.down_proj.weight.copy_(down_proj)

    @property
    def weight(self) -> th.Tensor:
        return self.base.weight + self.up_proj.weight @ self.down_proj.weight

    @classmethod
    def from_weight_delta(cls, base: nn.Linear | Self, derived: nn.Linear | Self, rank: int | float) -> Self:
        assert isinstance(base, nn.Linear) or isinstance(base, LoRAfiedLinear), "base should be nn.Linear or LoRAfiedLinear"
        assert isinstance(derived, nn.Linear) or isinstance(derived, LoRAfiedLinear), "derived should be nn.Linear or LoRAfiedLinear"
        assert base.weight.shape == derived.weight.shape, "base and derived should have same shape"
        assert base.device == derived.device, "base and derived should be on same device"

        weight_delta = (derived.weight - base.weight).detach()
        U, S, Vh = th.linalg.svd(weight_delta, full_matrices=False)

        if isinstance(rank, float):
            rank = int(rank * S.shape[0])

        S = th.diag_embed(S[:rank].sqrt())

        # Low-rank approximation of weight_delta as P @ Qh
        P = U[:, :rank] @ S
        Qh = S @ Vh[:rank]

        return cls(base, P, Qh)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.base(x) + self.up_proj(self.down_proj(x))
