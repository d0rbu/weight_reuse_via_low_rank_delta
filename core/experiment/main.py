import os
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from core.lorafy.mappings import layer_mappings
from core.lorafy.structured_lorafy import lorafy_parameters_layerwise
from core.utils import get_param_ancestors
from itertools import product, chain, combinations
from typing import Iterable, Callable


def powerset(iterable: Iterable):
    full_set = list(iterable)
    return chain.from_iterable(
        combinations(full_set, num_elements)
        for num_elements in range(len(full_set) + 1)
    )


def get_model_tokenizer_and_layers(
    get_model_and_tokenizer: Callable[[], tuple[PreTrainedModel, PreTrainedTokenizer]],
    blocks_name: str = "model.layers"
) -> tuple[PreTrainedModel, PreTrainedTokenizer, nn.ModuleList | nn.Sequential]:
    model, tokenizer = get_model_and_tokenizer()
    layers_ancestors, _ = get_param_ancestors(model, blocks_name)
    layers = layers_ancestors[-1]

    return model, tokenizer, layers


def lorafy_lm_parameter_grid_eval(
    output_file: os.PathLike | str,
    get_model_and_tokenizer: Callable[[], tuple[PreTrainedModel, PreTrainedTokenizer]],
    blocks_name: str = "model.layers",
    ranks: Iterable[int | float] = (1/16, 1/8, 1/4),
    param_names: Iterable[str] = ("self_attn.q_proj", "self_attn.k_proj"),
    mappings: Iterable[dict[int, int]] | None = None,
) -> None:
    model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)

    if mappings is None:
        mappings = layer_mappings(len(layers))

    param_names_power_set = powerset(param_names)
    for rank, param_names, mapping in product(ranks, param_names_power_set, mappings):
        lorafy_parameters_layerwise(
            layers,
            rank,
            param_names,
            mapping,
            inplace=True
        )
        
        # TODO: evaluate model and write json output

        del layers, tokenizer, model

        if rank == ranks[-1] and param_names == param_names_power_set[-1] and mapping == mappings[-1]:
            break  # don't reload model if we're done

        model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)


def llama_2_7b_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


if __name__ == "__main__":
    lorafy_parameter_grid_eval("output.json", llama_2_7b_model_and_tokenizer)
