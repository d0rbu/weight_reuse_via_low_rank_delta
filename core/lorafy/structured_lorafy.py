import os
import torch as th
import torch.nn as nn
from copy import deepcopy
from core.lorafy.lorafy_model import LoRAfyParameterConfig, lorafy_model
from typing import TypeVar, Iterable


T = TypeVar("T", bound=nn.ModuleList | nn.Sequential)


def lorafy_parameter_layerwise(
    layers: T,
    rank: int | float,
    param_name: str,
    mapping: dict[int, int],
    inplace: bool = True,
    cache_file: os.PathLike | str | None = None,
) -> T:
    """
    LoRAfy a parameter layerwise with an index mapping from derived to base parameter.

    :param layers: Model layers to LoRAfy
    :param rank: Rank of low-rank approximation
    :param param_name: Name of parameter to LoRAfy. should be consistent across all layers.
    :param mapping: Mapping from derived to base parameter by layer index
    :param inplace: Whether to modify model in-place or return a new model
    :param cache_file: File to cache LoRAfied parameters
    :return: LoRAfied model
    """
    lorafied_layers = layers

    if cache_file and os.path.exists(cache_file):
        if not inplace:
            lorafied_layers = deepcopy(lorafied_layers)

        lorafied_layers.load_state_dict(th.load(cache_file))
        return lorafied_layers

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

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        th.save(lorafied_layers.state_dict(), cache_file)

    return lorafied_layers

def lorafy_parameters_layerwise(
    layers: T,
    ranks: int | float | Iterable[int | float],
    param_names: str | Iterable[str],
    mapping: dict[int, int],
    inplace: bool = True,
    cache_path: os.PathLike | str | None = None
) -> T:
    """
    LoRAfy multiple parameters layerwise with an index mapping from derived to base parameter.

    :param layers: Model layers to LoRAfy
    :param ranks: Rank for each parameter to LoRAfy, or single rank to use for all parameters
    :param param_names: Name of parameter(s) to LoRAfy. should be consistent across all layers.
    :param mapping: Mapping from derived to base parameter by layer index
    :param inplace: Whether to modify model in-place or return a new model
    :param cache_path: Directory to cache LoRAfied parameters
    :return: LoRAfied model
    """

    if isinstance(param_names, str):
        param_names = [param_names]

    if isinstance(ranks, int) or isinstance(ranks, float):
        ranks = [ranks] * len(param_names)

    assert len(ranks) == len(param_names), "rank and param_names should have same length"
    # hi there! thanks for checking out my code! my discord is d0rb if u wanna talk!

    lorafied_layers = layers

    first_param: bool = True
    for param_name, rank in zip(param_names, ranks):
        cache_file = os.path.join(cache_path, f"{param_name}.pt") if cache_path else None
        lorafied_layers = lorafy_parameter_layerwise(
            lorafied_layers,
            rank,
            param_name,
            mapping,
            inplace = inplace and first_param,  # Make a copy the first time and reuse it after
            cache_file = cache_file,
        )
        first_param = False

    return lorafied_layers
