import os
import json
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from accelerate import dispatch_model, infer_auto_device_map
from core.lorafy.mappings import layer_mappings
from core.lorafy.structured_lorafy import lorafy_parameters_layerwise
from core.utils import get_param_ancestors, log_error, log_warn, log_info, log_info_1, Verbosity
from lm_eval import evaluator
from lm_eval.tasks import initialize_tasks
from lm_eval.models.huggingface import HFLM
from hashlib import md5
from itertools import product, chain, combinations
from typing import Iterable, Callable, Collection, Mapping


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
    num_layers_and_model_name: tuple[int, str] | None = None,
    blocks_name: str = "model.layers",
    ranks: Iterable[int | float] = (1/16, 1/8, 1/4),
    param_name_combinations: Iterable[Iterable[str]] = powerset(("self_attn.q_proj", "self_attn.k_proj")),
    mappings: Collection[dict[int, int]] | None = None,
    raw_results_dir: os.PathLike | str = "raw_results",
    lorafied_model_cache_dir: os.PathLike | str = ".lorafied_model_cache",
    verbosity: str = "INFO",
    move_device: str | None = None,
) -> None:
    verbosity = Verbosity[verbosity]

    log_info("Initializing tasks...", verbosity)
    initialize_tasks(verbosity = "INFO")

    tasks = ["winogrande"]
    log_info(f"Tasks: {tasks}", verbosity)
    os.makedirs(os.path.join(output_dir, raw_results_dir), exist_ok=True)

    if num_layers_and_model_name:
        num_layers, model_name = num_layers_and_model_name
        log_info(f"Got num_layers {num_layers} and model_name {model_name}", verbosity)
    else:
        log_info("Initializing model and tokenizer to get model name and number of layers...", verbosity)
        model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)
        num_layers = len(layers)
        model_name = model.__class__.__name__
        log_info_1(f"Model:\n{model}", verbosity)
        log_info_1(f"Tokenizer:\n{tokenizer}", verbosity)
        log_info_1(f"Model Layers:\n{layers}", verbosity)
        del model, tokenizer, layers

    if mappings is None:
        log_info("Initializing layer mappings...", verbosity)
        mappings = layer_mappings(num_layers)
        log_info_1(str(mappings), verbosity)

    full_results = {}
    for rank, param_names in product(ranks, param_name_combinations):
        for param_mappings in product(*([mappings] * len(param_names))):
            assert len(param_mappings) > 0, f"Length of param_mappings is 0, did you pass proper mappings?"

            # if all parameter mappings are equal, compress it down to one. it will be broadcasted later
            if len(param_mappings) == 1 or all(prev_param_mapping == next_param_mapping
                                               for prev_param_mapping, next_param_mapping
                                               in zip(param_mappings[:-1], param_mappings[1:])):
                param_mappings = param_mappings[0]

            log_info(f"Evaluating the following config:\nRank: {rank}\nParameters: {param_names}", verbosity)
            log_info_1(f"Mappings: {param_mappings}", verbosity)

            if isinstance(param_mappings, Mapping):  # if it is just a single mapping
                mapping_jsons = [json.dumps(param_mappings, sort_keys=True)] * len(param_names)
                base_params = sorted(list(set(param_mappings.values())))
            else:
                mapping_jsons = [json.dumps(param_mapping, sort_keys=True) for param_mapping in param_mappings]
                base_params = sorted(list(set(chain.from_iterable([mapping.values() for mapping in param_mappings]))))
            full_mapping_json = json.dumps(param_mappings, sort_keys=True)
            experiment_hash = int(md5(str.encode(f"{rank}{param_names}{full_mapping_json}")).hexdigest(), 16)
            lorafied_params_hashes = [int(md5(str.encode(f"{model_name}{rank}{mapping_json}")).hexdigest(), 16) for mapping_json in mapping_jsons]
            log_info_1(f"Experiment hash: {experiment_hash}\nLoRAfied parameter cache hashes: {lorafied_params_hashes}", verbosity)

            cache_paths = [
                os.path.join(
                    lorafied_model_cache_dir,
                    str(lorafied_params_hash),  # so we can reuse lorafied params across multiple experiments
                ) for lorafied_params_hash in lorafied_params_hashes
            ]
            raw_output_filepath = os.path.join(
                output_dir,
                raw_results_dir,
                f"{experiment_hash}.json"
            )

            if len(param_names) == 1:
                second_experiment_hash = int(md5(str.encode(f"{rank}{param_names[0]}{full_mapping_json}")).hexdigest(), 16)
                second_raw_output_filepath = os.path.join(
                    output_dir,
                    raw_results_dir,
                    f"{second_experiment_hash}.json"
                )

            if os.path.exists(raw_output_filepath):
                cached_output_file = raw_output_filepath
            elif len(param_names) == 1 and os.path.exists(second_raw_output_filepath):
                cached_output_file = second_raw_output_filepath
            else:
                cached_output_file = None

            log_info(f"Checking results cache...", verbosity)
            if cached_output_file:
                log_info(f"Found results in cache, loading...", verbosity)
                with open(cached_output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            else:
                log_info(f"Initializing model and tokenizer...", verbosity)
                model, tokenizer, layers = get_model_tokenizer_and_layers(get_model_and_tokenizer, blocks_name)

                log_info(f"LoRAfying the parameters...", verbosity)
                lorafy_parameters_layerwise(
                    layers,
                    rank,
                    param_names,
                    param_mappings,
                    inplace = True,
                    cache_paths = cache_paths,
                    verbosity = verbosity,
                    move_device = move_device,
                )

                # new_device_map = infer_auto_device_map(model)
                # model = dispatch_model(model, new_device_map)
                model = model.cuda()

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
                    "base_params": base_params,
                    "param_mappings": param_mappings
                }

                del tokenizer, model, layers, lm

                with open(raw_output_filepath, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                log_info_1(f"Raw output written to {raw_output_filepath}", verbosity)

            results = results["results"]
            param_names_str = param_names if isinstance(param_names, str) else ",".join(param_names)
            if len(mappings) == num_layers and len(param_names) == 1:  # if there's only one base layer per mapping
                if rank not in full_results:
                    full_results[rank] = {
                        param_names_str: {}
                    }
                elif param_names_str not in full_results[rank]:
                    full_results[rank][param_names_str] = {}
                
                mapping_idx = mappings.values()[0]
                full_results[rank][param_names_str][mapping_idx] = results
            else:
                if rank not in full_results:
                    full_results[rank] = {
                        param_names_str: {}
                    }
                elif param_names_str not in full_results[rank]:
                    full_results[rank][param_names_str] = {}

                full_results[rank][param_names_str][json.dumps(param_mappings, sort_keys=True)] = results

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    log_info(f"Wrote full results to {output_dir}", verbosity)


def llama_2_7b_model_and_tokenizer(device_map: str = "cpu") -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


if __name__ == "__main__":
    param_combinations = (("self_attn.q_proj", "self_attn.k_proj"),)

    lorafy_lm_parameter_grid_eval("outputs/", llama_2_7b_model_and_tokenizer, num_layers_and_model_name=(32, "LlamaForCausalLM"), param_name_combinations=param_combinations, move_device="cuda:0")
