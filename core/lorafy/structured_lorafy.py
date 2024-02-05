import os
import torch as th
import torch.nn as nn
from core.lorafy.lorafy_model import LoRAfyParameterConfig, lorafy_model
from core.utils import Verbosity, log_warn, log_info
from typing import TypeVar, Iterable, Mapping, Sequence
from tqdm import tqdm


T = TypeVar("T", bound=nn.ModuleList | nn.Sequential)


def lorafy_parameter_layerwise(
    layers: T,
    rank: int | float,
    param_name: str,
    mapping: dict[int, int],
    inplace: bool = True,
    cache_file: os.PathLike | str | None = None,
    verbosity: Verbosity = Verbosity.INFO,
    move_device: str | None = None,
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

    load_from_cache: bool = cache_file and os.path.exists(cache_file)

    if load_from_cache:
        try:
            # Test if we can properly load the cache file
            cached_state_dict = th.load(cache_file)

            lora_name_endings = [
                f"{param_name}.{lora_weight}.weight" for lora_weight in ("down_proj", "up_proj")
            ]
            cached_state_dict = {  # Only update the PQ* weights
                key: value for key, value in cached_state_dict.items()
                if any(key.endswith(param_name_ending) for param_name_ending in lora_name_endings)
            }
        except RuntimeError as e:
            log_warn(f"Unable to read cache file {cache_file}, recalculating...", verbosity)
            os.remove(cache_file)
            load_from_cache = False

    to_from_layer_generator = mapping.items()
    if verbosity >= Verbosity.INFO:
        to_from_layer_generator = tqdm(to_from_layer_generator)

    first_mapping: bool = True
    for to_layer, from_layer in to_from_layer_generator:
        base_param_name = f"{from_layer}.{param_name}"
        derived_param_name = f"{to_layer}.{param_name}"

        lorafied_layers = lorafy_model(
            lorafied_layers,
            LoRAfyParameterConfig(base_param_name, derived_param_name, rank),
            inplace = inplace or not first_mapping,  # Make a copy the first time and reuse it after
            move_device = move_device,
            approximate_lora = not load_from_cache,  # If we are loading from cache, do not approximate PQ*
        )
        first_mapping = False

    if load_from_cache:
        log_info("Found cached parameters, loading...", verbosity)
        updated_state_dict = lorafied_layers.state_dict()
        updated_state_dict.update(cached_state_dict)

        del cached_state_dict

        lorafied_layers.load_state_dict(updated_state_dict)
    elif cache_file:  # If the cache file path is given but it does not exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        th.save(lorafied_layers.state_dict(), cache_file)

    return lorafied_layers

def lorafy_parameters_layerwise(
    layers: T,
    ranks: int | float | Iterable[int | float],
    param_names: str | Iterable[str],
    param_mappings: Iterable[Mapping[int, int]] | Mapping[int, int],
    inplace: bool = True,
    cache_paths: Iterable[os.PathLike | str] | os.PathLike | str | None = None,
    verbosity: Verbosity = Verbosity.INFO,
    move_device: str | None = None,
) -> T:
    """
    LoRAfy multiple parameters layerwise with an index mapping from derived to base parameter.

    :param layers: Model layers to LoRAfy
    :param ranks: Rank for each parameter to LoRAfy, or single rank to use for all parameters
    :param param_names: Name of parameter(s) to LoRAfy. should be consistent across all layers.
    :param mapping: Mapping from derived to base parameter by layer index
    :param inplace: Whether to modify model in-place or return a new model
    :param cache_paths: Directories to cached LoRAfied parameters
    :return: LoRAfied model
    """

    if isinstance(param_names, str):
        param_names = [param_names]

    if cache_paths is None or isinstance(cache_paths, os.PathLike) or isinstance(cache_paths, str):
        cache_paths = [cache_paths] * len(param_names)
    else:
        assert len(cache_paths) == len(param_names), "#cache_paths does not match #param_names!"

    if isinstance(param_mappings, Sequence):
        assert len(param_mappings) == len(param_names), "#param_mappings does not match #param_names!"
    else:  # Otherwise broadcast the mapping to all params
        param_mappings = [param_mappings] * len(param_names)

    if isinstance(ranks, Iterable):
        assert len(ranks) == len(param_names), "#ranks does not match #param_names!"
    else:
        ranks = [ranks] * len(param_names)
    # hi there! thanks for checking out my code! my discord is d0rb if u wanna talk!

    lorafied_layers = layers

    first_param: bool = True
    for param_name, cache_path, mapping, rank in zip(param_names, cache_paths, param_mappings, ranks):
        cache_file = os.path.join(cache_path, f"{param_name}.pt") if cache_path else None
        lorafied_layers = lorafy_parameter_layerwise(
            lorafied_layers,
            rank,
            param_name,
            mapping,
            inplace = inplace or not first_param,  # Make a copy the first time and reuse it after
            cache_file = cache_file,
            verbosity = verbosity,
            move_device = move_device,
        )
        first_param = False

    return lorafied_layers
