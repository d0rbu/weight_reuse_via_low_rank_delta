import torch as th
import torch.nn as nn
from typing import Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def dispatch(
    model: PreTrainedModel,
    num_layers: int = 0,
    devices: Sequence[int] | None = None
) -> None:
    pre_layers, blocks, post_layers = find_blocks(model, num_layers)

    if devices is None:
        devices = [device for device in range(th.cuda.device_count())]
    devices = devices if isinstance(devices, th.Tensor) else th.tensor(devices)
    num_devices = len(devices)

    block_mappings = th.linspace(0, num_devices, num_layers + 1)[:-1].int()  # Split the layers up among 0, 1, 2, ... etc evenly
    block_mappings = devices[block_mappings]  # Index the actual device numbers (usually its just 0, 1, 2, ...)

    for pre_layer_parent, pre_layer_name, pre_layer in pre_layers:
        setattr(pre_layer_parent, pre_layer_name, pre_layer.to(int(devices[0])))

    for i, device in enumerate(block_mappings):
        blocks[i] = blocks[i].to(int(device))

    for post_layer_parent, post_layer_name, post_layer in post_layers:
        setattr(post_layer_parent, post_layer_name, post_layer.to(int(devices[-1])))

    # attach hooks to move data between devices only when necessary
    changed_device_blocks = th.nonzero(devices[1:] != devices[:-1]).squeeze() + 1
    for i in changed_device_blocks:
        blocks[i].register_forward_pre_hook(block_device_pre_hook(devices[i]))

    th.cuda.empty_cache()

def block_device_pre_hook(
    device: th.Device,
) -> callable[[nn.Module, th.Tensor], th.Tensor]:
    return lambda module, input: input.to(device)

def find_blocks(
    module: nn.Module,
    num_layers: int = 0,
) -> tuple[list[tuple[nn.Module, str, nn.Module]], nn.Module, list[tuple[nn.Module, str, nn.Module]]]:
    # within an nn.module, find where the main model blocks are located
    pre_layers = []
    post_layers = []

    current_module = module

    if isinstance(current_module, nn.ModuleList) and len(current_module) == num_layers:
        return [], module, []

    found_blocks = None
    for name, child in module.named_children():
        if found_blocks:
            post_layers.append((module, name, child))
        else:
            child_pre_layers, child_blocks, child_post_layers = find_blocks(child, num_layers)

            if child_blocks:
                pre_layers.extend(child_pre_layers)
                found_blocks = child_blocks
                post_layers.extend(child_post_layers)
            else:
                pre_layers.append((module, name, child))

    return pre_layers, found_blocks, post_layers
