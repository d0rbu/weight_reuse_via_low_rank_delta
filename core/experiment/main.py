import os
import json
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from core.lorafy.mappings import layer_mappings
from core.lorafy.structured_lorafy import lorafy_parameters_layerwise
from core.utils import get_param_ancestors
from lm_eval import evaluator
from lm_eval.tasks import initialize_tasks
from lm_eval.models.huggingface import HFLM
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
    output_dir: os.PathLike | str,
    get_model_and_tokenizer: Callable[[], tuple[PreTrainedModel, PreTrainedTokenizer]],
    blocks_name: str = "model.layers",
    ranks: Iterable[int | float] = (1/16, 1/8, 1/4),
    param_names: Iterable[str] = ("self_attn.q_proj", "self_attn.k_proj"),
    mappings: Iterable[dict[int, int]] | None = None,
    raw_results_dir: os.PathLike | str = "raw_results"
) -> None:
    initialize_tasks(verbosity="INFO")
    tasks = ["winogrande"]
    os.makedirs(os.path.join(output_dir, raw_results_dir), exist_ok=True)

    model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)

    if mappings is None:
        mappings = layer_mappings(len(layers))

    full_results = {}
    param_names_power_set = powerset(param_names)
    for rank, param_names, (base_param_idx, mapping) in product(ranks, param_names_power_set, enumerate(mappings)):
        lorafy_parameters_layerwise(
            layers,
            rank,
            param_names,
            mapping,
            inplace=True
        )

        lm = HFLM(pretrained=model, tokenizer=tokenizer)

        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            batch_size="auto",
        )
        results["lorafy_config"] = {
            "rank": rank,
            "param_names": param_names,
            "mapping": mapping
        }

        del layers, tokenizer, model

        test_hash = hash(f"{rank}{param_names}{base_param_idx}")
        raw_output_filepath = os.path.join(
            output_dir,
            raw_results_dir,
            f"{test_hash}.json"
        )
        with open(raw_output_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        results = results["results"]
        full_results[(rank, param_names, base_param_idx)] = results

        if rank == ranks[-1] and param_names == param_names_power_set[-1] and mapping == mappings[-1]:
            break  # don't reload model if we're done

        model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)


def llama_2_7b_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


if __name__ == "__main__":
    lorafy_parameter_grid_eval("outputs/", llama_2_7b_model_and_tokenizer)
