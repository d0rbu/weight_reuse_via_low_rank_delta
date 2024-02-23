import threading
import torch as th
import torch.nn as nn
from enum import StrEnum
from torch.futures import Future
from typing import Self


class Device(StrEnum):
    INPUT: str = "input"  # move the weight to the input device
    WEIGHT: str = "weight"  # move the input to the weight device
    # otherwise specify the device to move both of them to with a str or th.device

class LoRAfiedLinear(nn.Module):
    """
    Approximation of a weight matrix as a sum of a base weight matrix and a low-rank matrix.
    Can be computed automatically using from_weight_delta().
    """
    def __init__(self, base: nn.Module, up_proj: th.Tensor, down_proj: th.Tensor, device: th.device | str = "cuda") -> None:
        super().__init__()

        assert len(up_proj.shape) == 2, "up_proj should be 2D tensor of weights"
        assert len(down_proj.shape) == 2, "down_proj should be 2D tensor of weights"
        assert up_proj.shape[1] == down_proj.shape[0], "up_proj and down_proj should have same hidden dim"

        if isinstance(base, nn.Linear) or isinstance(base, LoRAfiedLinear):
            assert up_proj.shape[0] == base.weight.shape[0], "up_proj should have same output dim as base"
            assert down_proj.shape[1] == base.weight.shape[1], "down_proj should have same input dim as base"

        self.base = base
        self.up_proj = nn.Linear(up_proj.shape[1], up_proj.shape[0], bias=False, device=device)
        self.down_proj = nn.Linear(down_proj.shape[1], down_proj.shape[0], bias=False, device=device)

        with th.no_grad():
            up_proj = up_proj.to(device)
            down_proj = down_proj.to(device)

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
        num_weight_groups: int | None = None,
        weight_group_axis: int | None = None,
        move_device: str | None = None,
        approximate_lora: bool = True,
    ) -> Self:
        assert isinstance(base, nn.Linear) or isinstance(base, LoRAfiedLinear), "base should be nn.Linear or LoRAfiedLinear"
        assert isinstance(derived, nn.Linear) or isinstance(derived, LoRAfiedLinear), "derived should be nn.Linear or LoRAfiedLinear"
        assert base.weight.shape == derived.weight.shape, "base and derived should have same shape"
        assert rank > 0, "rank should be positive"
        assert approximate_lora or num_weight_groups is None, "num_weight_groups only applicable if approximate_lora=True"
        assert approximate_lora or weight_group_axis is None, "weight_group_axis only applicable if approximate_lora=True"

        if num_weight_groups is None:
            num_weight_groups = 1
        
        if weight_group_axis is None:
            weight_group_axis = 0
        weight_dim = base.weight.shape[weight_group_axis]

        assert weight_dim % num_weight_groups == 0, f"num_weight_groups {num_weight_groups} " \
                                                    f"should divide the weight dimension along axis {weight_group_axis}, which is {weight_dim}"

        if isinstance(rank, float):
            rank = int(rank * min(*base.weight.shape))
        weight_group_rank = rank // num_weight_groups
        
        original_derived_device = derived.weight.device

        if approximate_lora:  # If we want to project the delta down to lower rank to approximate the LoRAized matrix
            if move_device:
                derived_weight = derived.weight.to(move_device)
                base_weight = base.weight.to(move_device)

                weight_delta = (derived_weight - base_weight).detach()

                del derived_weight, base_weight
            else:
                assert base.weight.device == derived.weight.device, "base and derived should be on same device"
                weight_delta = (derived.weight - base.weight).detach()

            del derived

            P_grouped = []
            Qh_grouped = []
            weight_group_dim = weight_dim // num_weight_groups

            for start in range(0, weight_dim, weight_group_dim):
                end = start + weight_group_dim  # start and end indices for this weight group, along the given axis

                # slice to extract this particular weight group
                group_slice = (slice(None) * 2)
                group_slice[weight_group_axis] = slice(start, end)

                U, S, Vh = th.linalg.svd(weight_delta[group_slice], full_matrices=False)
                S = th.diag_embed(S[:weight_group_rank].sqrt())

                # Low-rank approximation of this weight group as P @ Qh
                P = U[:, :weight_group_rank] @ S
                Qh = S @ Vh[:weight_group_rank]

                P_grouped.append(P)
                Qh_grouped.append(Qh)

            del weight_delta, U, S, Vh

            P = th.cat(P_grouped, dim=weight_group_axis)
            Qh = th.cat(Qh_grouped, dim=weight_group_axis)

            del P_grouped, Qh_grouped
        else:
            P = th.empty(base.weight.shape[0], rank)
            Qh = th.empty(rank, base.weight.shape[1])

        return cls(base, P, Qh, original_derived_device)

    def async_base_forward(
        self: Self,
        future: Future[th.Tensor],
        x: th.Tensor,
        device: th.device | Device = Device.INPUT,
        move_back: bool = False,  # if this is true, we move everything back to its original device
    ) -> None:
        if self.base.weight.device == x.device:
            future.set_result(self.base(x))
            return

        original_base_device = self.base.weight.device
        original_input_device = x.device

        if device == Device.INPUT:
            device = x.device
        elif device == Device.WEIGHT:
            device = self.base.weight.device

        self.base = self.base.to(device)
        x = x.to(device)

        output = self.base(x)

        if move_back:
            self.base = self.base.to(original_base_device)
            x = x.to(original_input_device)

        future.set_result(output)

    def async_lora_forward(
        self: Self,
        future: Future[th.Tensor],
        x: th.Tensor,
        device: th.device | Device = Device.WEIGHT,
        move_back: bool = False,  # if this is true, we move everything back to its original device
    ) -> None:
        if self.up_proj.weight.device == x.device and self.down_proj.weight.device == x.device:
            future.set_result(self.up_proj(self.down_proj(x)))
            return
        
        original_lora_device = self.down_proj.weight.device
        original_input_device = x.device

        if device == Device.INPUT:
            device = x.device
        elif device == Device.WEIGHT:
            device = self.down_proj.weight.device

        self.down_proj = self.down_proj.to(device)
        self.up_proj = self.up_proj.to(device)
        x = x.to(device)

        hidden = self.down_proj(x)
        output = self.up_proj(hidden)

        if move_back:
            self.down_proj = self.down_proj.to(original_lora_device)
            self.up_proj = self.up_proj.to(original_lora_device)
            x = x.to(original_input_device)

        future.set_result(output)

    def forward(self: Self, x: th.Tensor) -> th.Tensor:
        # To deal with tensors being on different devices and also speed up by doing parallelism
        # Sequentially, this code would be self.base(x) + self.up_proj(self.down_proj(x))

        forward_args = [x]
        if self.base.weight.device != self.down_proj.weight.device:
            forward_args.append(self.down_proj.weight.device)

        base_future: Future[th.Tensor] = Future()
        lora_future: Future[th.Tensor] = Future()

        base_thread: threading.Thread = threading.Thread(
            target = self.async_base_forward,
            args = (base_future, *forward_args),
        )
        lora_thread: threading.Thread = threading.Thread(
            target = self.async_lora_forward,
            args = (lora_future, *forward_args),
        )

        base_thread.start()
        lora_thread.start()

        base_result, lora_result = th.futures.wait_all([base_future, lora_future])

        return base_result + lora_result
