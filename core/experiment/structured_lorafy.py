import torch.nn as nn
from core.lorafy.lorafy_model import LoRAfyParameterConfig, lorafy_model
from typing import TypeVar


T = TypeVar("T", bound=nn.ModuleList | nn.Sequential)


def lorafy_layerwise(
    layers: T,
    rank: int | float,
    param_name: str,
    mapping: dict[int, int],
    inplace: bool = True
) -> T:
    """
    LoRAfy a model layerwise according to a mapping from derived to base parameter.

    :param layers: Model layers to LoRAfy
    :param rank: Rank of low-rank approximation
    :param param_name: Name of parameter to LoRAfy. should be consistent across all layers.
    :param mapping: Mapping from derived to base parameter by layer index
    :return: LoRAfied model
    """
    lorafied_layers = layers

    first_mapping: bool = True
    for to_layer, from_layer in mapping.items():
        base_param_name = f"{from_layer}.{param_name}"
        derived_param_name = f"{to_layer}.{param_name}"

        lorafied_layers = lorafy_model(
            lorafied_layers,
            LoRAfyParameterConfig(base_param_name, derived_param_name, rank),
            inplace = inplace and first_mapping  # Make a copy the first time and reuse it after
        )
        first_mapping = False

    return lorafied_layers
