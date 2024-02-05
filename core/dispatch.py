import torch as th
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def dispatch(
    model: PreTrainedModel,
    num_layers: int = 0,
    num_devices: int = 0,
) -> PreTrainedModel:
    pre_layers, blocks, post_layers = find_blocks(model, num_layers)

    num_devices = num_devices if num_devices else th.cuda.device_count()
    block_mappings = th.linspace(0, num_devices, num_layers + 1)[:-1].int()

    for pre_layer_parent, pre_layer_name, pre_layer in pre_layers:
        setattr(pre_layer_parent, pre_layer_name, pre_layer.to(0))

    for i, device in enumerate(block_mappings):
        blocks[i] = blocks[i].to(int(device))
    
    for post_layer_parent, post_layer_name, post_layer in post_layers:
        setattr(post_layer_parent, post_layer_name, post_layer.to(num_devices - 1))

    th.cuda.empty_cache()


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
