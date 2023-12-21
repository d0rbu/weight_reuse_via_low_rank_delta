import os
import json
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from core.lorafy.mappings import layer_mappings
from core.lorafy.structured_lorafy import lorafy_parameters_layerwise
from core.utils import get_param_ancestors, log_error, log_warn, log_info, log_info_1, Verbosity
from lm_eval import evaluator
from lm_eval.tasks import initialize_tasks
from lm_eval.models.huggingface import HFLM
from itertools import product, chain, combinations
from typing import Iterable, Callable


def powerset(iterable: Iterable, include_null_set: bool = False):
    full_set = list(iterable)
    return chain.from_iterable(
        combinations(full_set, num_elements)
        for num_elements in range(0 if include_null_set else 1, len(full_set) + 1)
    )


def get_model_tokenizer_and_layers(
    get_model_and_tokenizer: Callable[[], tuple[PreTrainedModel, PreTrainedTokenizer]],
    blocks_name: str,
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
    param_name_combinations: Iterable[str] = powerset(("self_attn.q_proj", "self_attn.k_proj")),
    mappings: Iterable[dict[int, int]] | None = None,
    raw_results_dir: os.PathLike | str = "raw_results",
    lorafied_model_cache_dir: os.PathLike | str = ".lorafied_model_cache",
    verbosity: str = "INFO",
) -> None:
    enum_verbosity = Verbosity[verbosity]

    log_info("Initializing tasks...", enum_verbosity)
    initialize_tasks(verbosity = verbosity)
    verbosity = enum_verbosity

    tasks = ["winogrande"]
    log_info(f"Tasks: {tasks}", verbosity)
    os.makedirs(os.path.join(output_dir, raw_results_dir), exist_ok=True)

    log_info("Initializing model and tokenizer...", verbosity)
    model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)
    log_info_1(f"Model:\n{model}", verbosity)
    log_info_1(f"Tokenizer:\n{tokenizer}", verbosity)
    log_info_1(f"Model Layers:\n{layers}", verbosity)

    if mappings is None:
        log_info("Initializing layer mappings...", verbosity)
        mappings = layer_mappings(len(layers))
        log_info_1(str(mappings), verbosity)

    full_results = {}
    for rank, param_names, (mapping_idx, mapping) in product(ranks, param_name_combinations, enumerate(mappings)):
        log_info(f"Evaluating the following config:\nRank: {rank}\nParameters: {param_names}", verbosity)
        log_info_1(f"Mapping: {mapping}", verbosity)
        mapping_json = json.dumps(mapping, sort_keys=True)
        experiment_hash = hash(f"{rank}{param_names}{mapping_json}")
        lorafied_params_hash = hash(f"{model.__class__.__name__}{rank}{mapping_json}")
        log_info_1(f"Experiment hash: {experiment_hash}\nLoRAfied parameter cache hash: {lorafied_params_hash}", verbosity)

        cache_path = os.path.join(
            lorafied_model_cache_dir,
            str(lorafied_params_hash),  # so we can reuse lorafied params across multiple experiments
        )

        log_info(f"LoRAfying the parameters...", verbosity)
        lorafy_parameters_layerwise(
            layers,
            rank,
            param_names,
            mapping,
            inplace = True,
            cache_path = cache_path,
            verbosity = verbosity,
        )

        log_info_1(f"Wrapping LoRAfied model in lm-evaluation-harness HFLM API...", verbosity)
        lm = HFLM(pretrained=model, tokenizer=tokenizer)

        log_info(f"Evaluating LoRAfied model...", verbosity)
        results = evaluator.simple_evaluate(
            model = lm,
            tasks = tasks,
            batch_size="auto",
        )
        results["lorafy_config"] = {
            "rank": rank,
            "param_names": param_names,
            "base_params": sorted(list(set(mapping.values()))),
            "mapping": mapping
        }

        del layers, tokenizer, model

        raw_output_filepath = os.path.join(
            output_dir,
            raw_results_dir,
            f"{experiment_hash}.json"
        )
        with open(raw_output_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log_info_1(f"Raw output written to {raw_output_filepath}", verbosity)

        results = results["results"]
        if len(mappings) == len(layers):  # if there's only one base layer per mapping
            full_results[(rank, param_names, mapping_idx)] = results
        else:
            full_results[(rank, param_names, json.dumps(mapping, sort_keys=True))] = results

        if rank == ranks[-1] and param_names == param_name_combinations[-1] and mapping == mappings[-1]:
            break  # don't reload model if we're done

        log_info(f"Initializing next model and tokenizer...", verbosity)
        model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    log_info(f"Wrote full results to {output_dir}", verbosity)


def llama_2_7b_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


if __name__ == "__main__":
    lorafy_lm_parameter_grid_eval("outputs/", llama_2_7b_model_and_tokenizer)
