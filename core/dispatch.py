import torch as th
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from core.lorafy.lorafied_weight import LoRAfiedLinear
from typing import Sequence, TypeVar, Callable
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

T = TypeVar("T")


def dispatch(
    model: PreTrainedModel,
    num_layers: int = 0,
    devices: Sequence[int] | None = None
) -> list[RemovableHandle]:
    handles = []
    pre_layers, blocks, post_layers = find_blocks(model, num_layers)

    if devices is None:
        devices = [device for device in range(th.cuda.device_count())]
    devices = devices if isinstance(devices, th.Tensor) else th.tensor(devices)
    num_devices = len(devices)

    block_mappings = th.linspace(0, num_devices, num_layers + 1)[:-1].int()  # Split the layers up among 0, 1, 2, ... etc evenly
    block_mappings = devices[block_mappings]  # Index the actual device numbers (usually its just 0, 1, 2, ...)

    for pre_layer_parent, pre_layer_name, pre_layer in pre_layers:
        device = int(devices[-1])
        new_device_pre_layer = pre_layer.to(device)
        setattr(pre_layer_parent, pre_layer_name, new_device_pre_layer)
        handle = new_device_pre_layer.register_forward_pre_hook(align_device_pre_hook(device), with_kwargs=True)
        handles.append(handle)

    for i, device in enumerate(block_mappings):
        blocks[i] = blocks[i].to(int(device))
        blocks[i].register_forward_pre_hook(align_device_pre_hook(int(block_mappings[i])), with_kwargs=True)
    
    for post_layer_parent, post_layer_name, post_layer in post_layers:
        device = int(devices[-1])
        new_device_post_layer = post_layer.to(device)
        setattr(post_layer_parent, post_layer_name, new_device_post_layer)
        handle = new_device_post_layer.register_forward_pre_hook(align_device_pre_hook(device), with_kwargs=True)
        handles.append(handle)

    # return the base layers back to their original devices after a round of inference
    handles.append(model.register_forward_pre_hook(get_base_layer_devices_pre_hook))
    handles.append(model.register_forward_pre_hook(align_device_pre_hook(int(devices[0])), with_kwargs=True))
    handles.append(model.register_forward_hook(return_base_layers_hook))

    th.cuda.empty_cache()

    return handles

def get_base_layer_devices_pre_hook(
    module: nn.Module,
    args: tuple,
) -> None:
    if hasattr(module, "_base_layer_devices"):
        return

    base_layer_devices = {}

    for child in module.modules():
        if isinstance(child, LoRAfiedLinear) and isinstance(child.base, nn.Linear):
            base_layer_devices[child.base] = child.base.weight.device
    
    module._base_layer_devices = base_layer_devices

def return_base_layers_hook(
    module: nn.Module,
    args: tuple,
    output: tuple,
) -> None:
    for base_layer, device in module._base_layer_devices.items():
        base_layer.to(device)

def align_device_pre_hook(
    device: th.device | int,
) -> Callable[[nn.Module, th.Tensor], th.Tensor]:
    def hook(
        module: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        new_args = to_device(args, device)
        new_kwargs = to_device(kwargs, device)

        return new_args, new_kwargs

    return hook

def to_device(
    obj: T,
    device: th.device | int,
) -> T:
    if isinstance(obj, (bool, str, int, float, type(None))):
        return obj

    if isinstance(obj, (th.Tensor, nn.Module)):
        return obj.to(device)

    if isinstance(obj, Sequence):
        sequence = obj.__class__
        return sequence(to_device(obj_element, device) for obj_element in obj)

    if isinstance(obj, dict):
        return {
            key: to_device(value, device)
            for key, value in obj.items()
        }

    # warning: these ones will modify original object
    if isinstance(obj, Cache):
        obj.key_cache = to_device(obj.key_cache, device)
        obj.value_cache = to_device(obj.value_cache, device)

        return obj

    raise ValueError(f"Can't move this type of object to device: {type(obj)}")

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
