import threading
import torch as th
import torch.nn as nn
from torch.futures import Future
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
    def from_weight_delta(
        cls,
        base: nn.Linear | Self,
        derived: nn.Linear | Self,
        rank: int | float,
        move_device: str | None = None,
    ) -> Self:
        assert isinstance(base, nn.Linear) or isinstance(base, LoRAfiedLinear), "base should be nn.Linear or LoRAfiedLinear"
        assert isinstance(derived, nn.Linear) or isinstance(derived, LoRAfiedLinear), "derived should be nn.Linear or LoRAfiedLinear"
        assert base.weight.shape == derived.weight.shape, "base and derived should have same shape"

        if move_device:
            original_derived_device = derived.weight.device
            original_base_device = base.weight.device
            derived_weight = derived.weight.to(move_device)
            base_weight = base.weight.to(move_device)
        else:
            assert base.weight.device == derived.weight.device, "base and derived should be on same device"
            derived_weight = derived.weight
            base_weight = base.weight

        weight_delta = (derived_weight - base_weight).detach()

        U, S, Vh = th.linalg.svd(weight_delta, full_matrices=False)

        if isinstance(rank, float):
            rank = int(rank * S.shape[0])

        S = th.diag_embed(S[:rank].sqrt())

        # Low-rank approximation of weight_delta as P @ Qh
        P = U[:, :rank] @ S
        Qh = S @ Vh[:rank]

        return cls(base, P, Qh)

    def async_base_forward(self: Self, future: Future[th.Tensor], x: th.Tensor) -> None:
        if self.base.weight.device == x.device:
            future.set_result(self.base(x))

        original_device = self.base.weight.device

        self.base = self.base.to(x.device)
        output = self.base(x)
        # self.base = self.base.to(original_device)

        future.set_result(output)

    def async_lora_forward(self: Self, future: Future[th.Tensor], x: th.Tensor) -> None:
        if self.up_proj.weight.device == x.device and self.down_proj.weight.device == x.device:
            future.set_result(self.up_proj(self.down_proj(x)))
        
        original_device = self.down_proj.weight.device

        self.down_proj = self.down_proj.to(x.device)
        hidden = self.down_proj(x)
        self.down_proj = self.down_proj.to(original_device)

        original_device = self.up_proj.weight.device

        self.up_proj = self.up_proj.to(x.device)
        output = self.up_proj(hidden)
        self.up_proj = self.up_proj.to(original_device)

        future.set_result(output)

    def forward(self: Self, x: th.Tensor) -> th.Tensor:
        # To deal with tensors being on different devices and also speed up by doing parallelism
        # Sequentially, this code would be self.base(x) + self.up_proj(self.down_proj(x))

        base_future: Future[th.Tensor] = Future()
        lora_future: Future[th.Tensor] = Future()

        base_thread: threading.Thread = threading.Thread(
            target = self.async_base_forward,
            args = (base_future, x),
        )
        lora_thread: threading.Thread = threading.Thread(
            target = self.async_lora_forward,
            args = (lora_future, x),
        )

        base_thread.start()
        lora_thread.start()

        base_result, lora_result = th.futures.wait_all([base_future, lora_future])

        return base_result + lora_result
