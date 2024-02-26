import os
import json
import yaml
import time
import argparse
import torch as th
from mpi4py import MPI
from torch.nn import ModuleList, Sequential
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from core.lorafy.mappings import layer_mappings
from core.lorafy.structured_lorafy import lorafy_parameters_layerwise
from core.dispatch import dispatch
from core.utils import get_param_ancestors, log_error, log_warn, log_info, log_info_1, Verbosity
from hashlib import md5
from itertools import product, chain, combinations
from collections import defaultdict
from typing import Iterable, Collection, Mapping, Literal, Sequence


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
WORLD_SIZE = COMM.Get_size()


def powerset(iterable: Iterable, include_null_set: bool = False) -> Iterable:
    full_set = list(iterable)
    return chain.from_iterable(
        combinations(full_set, num_elements)
        for num_elements in range(0 if include_null_set else 1, len(full_set) + 1)
    )


def get_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device_map: str = "cpu"
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def get_model_tokenizer_and_layers(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    blocks_name: str = "model.layers",
    device_map: str = "cpu",
) -> tuple[PreTrainedModel, PreTrainedTokenizer, ModuleList | Sequential]:
    model, tokenizer = get_model_and_tokenizer(model_name, device_map)
    layers_ancestors, _ = get_param_ancestors(model, blocks_name)
    layers = layers_ancestors[-1]

    return model, tokenizer, layers


def vanilla_lm_eval(
    evaluator,
    output_dir: os.PathLike | str,
    model_name: str,
    tasks: list[str],
    device_map_option: str = "auto",
    verbosity: str = "INFO"
) -> None:
    log_info(f"Getting vanilla results for {model_name}...", verbosity)

    raw_output_path = os.path.join(output_dir, "raw_vanilla_results.json")
    output_path = os.path.join(output_dir, "vanilla_results.json")

    cached_task_results = {}
    cached_raw_task_results = {
        "results": cached_task_results
    }
    uncached_tasks = tasks.copy()

    if os.path.exists(output_path):
        log_info(f"Found cached vanilla results in {output_path}, loading...", verbosity)
        with open(output_path, "r", encoding="utf-8") as f:
            cached_task_results.update(json.load(f))

        uncached_tasks = list(set(uncached_tasks) - cached_task_results.keys())

    if os.path.exists(raw_output_path) and len(uncached_tasks) > 0:
        log_info(f"Found cached raw vanilla results in {raw_output_path}, loading...", verbosity)

        with open(raw_output_path, "r", encoding="utf-8") as f:
            cached_raw_results = json.load(f)
        cached_raw_task_results.update(cached_raw_results)
        cached_task_results.update(cached_raw_results["results"])
        cached_task_results["results"] = cached_task_results

        uncached_tasks = list(set(uncached_tasks) - cached_task_results.keys())
    
    if len(uncached_tasks) <= 0:
        log_info(f"Found all results in cache, skipping vanilla evaluation", verbosity)
        results = cached_raw_task_results
    else:
        log_info(f"Running vanilla evaluation on the following tasks:\n{uncached_tasks}", verbosity)
        results = evaluator.simple_evaluate(
            model = "hf",
            model_args = f"parallelize=True,pretrained={model_name},device_map_option={device_map_option}",
            tasks = uncached_tasks,
            batch_size="auto",
        )
        cached_task_results.update(results["results"])
        results.update(cached_raw_task_results)
        results["results"] = cached_task_results

    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results["results"], f, indent=2)

    log_info(f"Wrote raw and vanilla results to {output_dir}", verbosity)


NumWeightGroups = int | Literal["heads"]
WeightGroupConfig = tuple[NumWeightGroups, int]  # (num_weight_groups, weight_group_axis)
RecursiveDefaultDict = lambda: defaultdict(RecursiveDefaultDict)

PROCESS_TIMEOUT = 3600  # if an in-progress raw output file was started over an hour ago, assume it died


def lorafy_lm_parameter_grid_eval(
    output_dir: os.PathLike | str = "outputs/",
    num_layers_and_model_name: tuple[int, str] = (32, "meta-llama/Llama-2-7b-hf"),
    blocks_name: str = "model.layers",
    ranks: Iterable[int | float] = (1/16, 1/8, 1/4),
    param_name_combinations: Iterable[Iterable[str]] = powerset(("self_attn.q_proj", "self_attn.k_proj")),
    mappings: Collection[dict[int, int]] | None = None,
    base_layers: Sequence[int] | int = (0,),
    weight_group_configs: list[WeightGroupConfig] | WeightGroupConfig = (1, 0),
    raw_results_dir: os.PathLike | str = "raw_results",
    lorafied_model_cache_dir: os.PathLike | str = ".lorafied_model_cache",
    verbosity: str = "INFO",
    move_device: str | None = None,
    tasks: list[str] | str = ["winogrande", "wikitext"],
    ignore_uncached_results: bool = False,
) -> None:
    # yes the num_layers can be inferred, but i dont wanna spend the time loading the whole model just to get that one int
    num_layers, model_name = num_layers_and_model_name

    assert len(weight_group_configs) > 0, "Specify at least one weight group config! Use (1, 0) if you're unsure"
    if isinstance(weight_group_configs[0], int) or isinstance(weight_group_configs[0], str):
        weight_group_configs = [weight_group_configs]

    output_dir = os.path.join(output_dir, model_name)
    output_path = os.path.join(output_dir, "results.json")
    verbosity = Verbosity[verbosity]
    available_devices = th.linspace(0, WORLD_SIZE, th.cuda.device_count() + 1)[:-1].int()
    available_devices = (available_devices == RANK).nonzero().flatten()

    if not ignore_uncached_results:
        log_info("Initializing tasks...", verbosity)

        from lm_eval import evaluator
        from lm_eval.tasks import initialize_tasks
        from lm_eval.models.huggingface import HFLM

        initialize_tasks(verbosity = "INFO")

        vanilla_lm_eval(evaluator, output_dir, model_name, tasks, verbosity=verbosity)

    tasks = [tasks] if isinstance(tasks, str) else tasks
    log_info(f"Tasks: {tasks}", verbosity)
    os.makedirs(os.path.join(output_dir, raw_results_dir), exist_ok=True)

    log_info(f"Got num_layers {num_layers} and model_name {model_name}", verbosity)

    if mappings is None:
        log_info("Initializing layer mappings...", verbosity)
        mappings = layer_mappings(num_layers, base_layers)
        log_info_1(str(mappings), verbosity)

    full_results = RecursiveDefaultDict()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as output_file:
            raw_full_results = json.load(output_file)

        for weight_group_config, raw_weight_group_config_results in raw_full_results.items():
            num_weight_groups, weight_group_axis = json.loads(weight_group_config)
            weight_group_config_results = {}

            for rank_str, raw_rank_results in raw_weight_group_config_results.items():
                rank = float(rank_str)
                rank_results = {}
                for param_names_str, raw_param_results in raw_rank_results.items():
                    param_results = {}
                    for mapping_str, raw_task_results in raw_param_results.items():
                        mapping = int(mapping_str) if mapping_str.isdigit() else mapping_str
                        param_results[mapping] = raw_task_results
                    rank_results[param_names_str] = param_results
                weight_group_config_results[rank] = rank_results
            full_results[num_weight_groups][weight_group_axis] = weight_group_config_results

        del raw_full_results

    exp_idx = -1
    for weight_group_config, rank, param_names in product(weight_group_configs, ranks, param_name_combinations):
        if not isinstance(param_names, tuple):
            param_names = tuple(param_names)
        
        num_weight_groups, weight_group_axis = weight_group_config

        for param_mappings in product(*([mappings] * len(param_names))):
            exp_idx += 1
            if exp_idx % WORLD_SIZE != RANK:
                continue

            assert len(param_mappings) > 0, f"Length of param_mappings is 0, did you pass proper mappings?"

            # if all parameter mappings are equal, compress it down to one. it will be broadcasted later
            if len(param_mappings) == 1 or all(prev_param_mapping == next_param_mapping
                                               for prev_param_mapping, next_param_mapping
                                               in zip(param_mappings[:-1], param_mappings[1:])):
                param_mappings = param_mappings[0]

            log_info(f"Evaluating the following config:\nNum weight groups: {num_weight_groups}\n" \
                     f"Weight group axis: {weight_group_axis}\nRank: {rank}\nParameters: {param_names}", verbosity)
            log_info_1(f"Mappings: {param_mappings}", verbosity)

            if isinstance(param_mappings, Mapping):  # if it is just a single mapping
                mapping_jsons = [json.dumps(param_mappings, sort_keys=True)] * len(param_names)
                base_params = sorted(list(set(param_mappings.values())))
            else:
                mapping_jsons = [json.dumps(param_mapping, sort_keys=True) for param_mapping in param_mappings]
                base_params = sorted(list(set(chain.from_iterable([mapping.values() for mapping in param_mappings]))))
            full_mapping_json = json.dumps(param_mappings, sort_keys=True)
            experiment_hash = int(md5(str.encode(f"{weight_group_config}{rank}{param_names}{full_mapping_json}")).hexdigest(), 16)
            lorafied_params_hashes = [int(md5(str.encode(f"{weight_group_config}{model_name}{rank}{mapping_json}")).hexdigest(), 16) for mapping_json in mapping_jsons]
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
            param_names_str = param_names if isinstance(param_names, str) else ",".join(param_names)
            one_base_layer = len(mappings) == num_layers and len(param_names) == 1
            mapping_idx = next(iter(param_mappings.values())) if one_base_layer else None  # get a random value from the dictionary
            mapping_key = mapping_idx if one_base_layer else full_mapping_json

            cached_task_results = {}
            cached_output_file = raw_output_filepath if os.path.exists(raw_output_filepath) else None

            log_info(f"Checking results caches...", verbosity)
            if cached_output_results := full_results.get(num_weight_groups, {}).get(weight_group_axis, {}).get(rank, {}).get(param_names_str, {}).get(mapping_key, None):
                log_info(f"Found results in full results cache...", verbosity)
                cached_task_results.update(cached_output_results)
                if set(tasks).issubset(cached_task_results.keys()):
                    continue

            if cached_output_file:
                with open(cached_output_file, "r", encoding="utf-8") as f:
                    results = json.load(f)

                actual_results = results["results"]

                if actual_results is not None:
                    log_info(f"Found results in raw output cache...", verbosity)

                    cached_task_results.update(actual_results)
                elif results["timestamp"] <= time.time() - PROCESS_TIMEOUT:
                    log_info(f"Raw output cache indicates results started being computed, "
                             f"but process has timed out so we will assume it died. Continuing...", verbosity)
                else:
                    log_info(f"Raw output cache indicates results are in progress, skipping...", verbosity)

            uncached_tasks = list(set(tasks) - set(cached_task_results.keys()))

            if ignore_uncached_results or len(uncached_tasks) <= 0:
                if len(uncached_tasks) > 0:
                    log_info(f"Did not find full results in cache, ignoring this experiment for the following tasks:\n"
                             f"{uncached_tasks}", verbosity)

                results = {
                    "results": cached_task_results
                }
            else:
                log_info(f"Writing placeholder to raw output file...", verbosity)
                with open(raw_output_filepath, "w", encoding="utf-8") as f:
                    json.dump({
                        "timestamp": time.time(),
                        "results": None,
                    }, f, indent=2)

                log_info(f"Initializing model and tokenizer...", verbosity)
                model, tokenizer, layers = get_model_tokenizer_and_layers(model_name, blocks_name)

                log_info(f"Dispatching model to devices...", verbosity)
                dispatch(model, num_layers, available_devices)
                # need to dispatch manually because if we do device_map="auto" or "balanced"
                # when loading the model, it will also add pesky hooks to align devices
                # which messes with my home grown solution

                log_info(f"LoRAfying the parameters...", verbosity)
                lorafy_parameters_layerwise(
                    layers,
                    rank,
                    param_names,
                    param_mappings,
                    num_weight_groups = num_weight_groups,
                    weight_group_axis = weight_group_axis,
                    inplace = True,
                    cache_paths = cache_paths,
                    verbosity = verbosity,
                    move_device = move_device,
                )

                log_info_1(f"Wrapping LoRAfied model in lm-evaluation-harness HFLM API...", verbosity)
                lm = HFLM(pretrained=model, tokenizer=tokenizer)

                log_info(f"Evaluating LoRAfied model on the following tasks:\n"
                         f"{uncached_tasks}", verbosity)
                results = evaluator.simple_evaluate(
                    model = lm,
                    tasks = uncached_tasks,
                    batch_size="auto",
                )
                results["lorafy_config"] = {
                    "num_weight_groups": num_weight_groups,
                    "weight_group_axis": weight_group_axis,
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

            full_results[num_weight_groups][weight_group_axis][rank][param_names_str][mapping_key] = results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    log_info(f"Wrote full results to {output_dir}", verbosity)


CONFIG_DIR = os.path.join("experiment", "configs")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--experiment", type=str, default="main", help="Name of the experiment config file")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Path to the output directory")
    parser.add_argument("--num_layers_and_model_name", type=tuple[int, str], default=(32, "meta-llama/Llama-2-7b-hf"), help="Number of layers in the model and the model name")
    parser.add_argument("--blocks_name", type=str, default="model.layers", help="Name of the blocks in the model")
    parser.add_argument("--ranks", type=float, nargs="+", default=[1/16, 1/8, 1/4], help="Ranks to evaluate")
    parser.add_argument("--param_name_combinations", type=str, nargs="+", default=powerset(("self_attn.q_proj", "self_attn.k_proj")), help="Parameter name combinations to evaluate")
    parser.add_argument("--mappings", type=list, default=None, help="Mappings to evaluate")
    parser.add_argument("--base_layers", type=list, default=(0,), help="Either a list of base layers or the number of base layers; the last will generate all combinations of that many base layers")
    parser.add_argument("--raw_results_dir", type=str, default="raw_results", help="Path to the raw results directory")
    parser.add_argument("--lorafied_model_cache_dir", type=str, default=".lorafied_model_cache", help="Path to the LoRAfied model cache directory")
    parser.add_argument("--verbosity", type=str, default="INFO", help="Verbosity level")
    parser.add_argument("--move_device", type=str, default=None, help="Move device option")
    parser.add_argument("--tasks", type=str, nargs="+", default=["winogrande", "wikitext"], help="Tasks to evaluate")
    parser.add_argument("--ignore_uncached_results", action="store_true", help="Ignore uncached results")
    parser.add_argument("--weight_group_configs", type=list[tuple[int | str, int]], default=[(1, 0)], help="Weight group configs, tuples of (num_weight_groups, weight_group_axis). num_weight_groups can also be the literal \"heads\" to match number of attention heads.")
    args = parser.parse_args()

    kwargs = vars(args)
    experiment_name = kwargs.pop("experiment")
    experiment_config_path = os.path.join(CONFIG_DIR, f"{experiment_name}.yaml")
    if os.path.exists(experiment_config_path):
        with open(experiment_config_path, "r") as experiment_config_file:
            experiment_config = yaml.safe_load(experiment_config_file)
    else:
        experiment_config = {}

    kwargs.update(experiment_config)

    lorafy_lm_parameter_grid_eval(**kwargs)
