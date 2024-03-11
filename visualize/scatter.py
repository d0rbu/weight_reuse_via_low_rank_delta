import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from core.permutalign.permutalign_model import PermutalignMode
from core.orthogonalign.orthogonalign_model import OrthogonalignMode
from transformers import AutoConfig, LlamaConfig, GemmaConfig, MistralConfig, PretrainedConfig
from dataclasses import dataclass


@dataclass
class Result:
    permutalign_mode: str
    orthogonalign_mode: str
    num_weight_groups: int
    weight_group_axis: int
    rank: float
    param_name: str
    param_map: int
    value: float


default_result = Result(
    permutalign_mode=PermutalignMode.IDENTITY,
    orthogonalign_mode=json.dumps(None),
    num_weight_groups=1,
    weight_group_axis=0,
    rank=1.0,
    param_name=None,
    param_map=0,
    value=0.0,
)


def filter_results(
    results_dir: os.PathLike | str,
    results: list[Result],
    model_names: list[str] | None = None,
    permutalign_modes: set[str] | None = None,
    orthogonalign_modes: set[str] | None = None,
    num_weight_groups: set[int] | None = None,
    weight_group_axis: set[int] | None = None,
    ranks: set[float] | None = None,
    param_names: set[str] | None = None,
    param_maps: set[int | str] | None = None,
    tasks: set[str] | None = None,
    plot_accuracy: bool = True,
) -> dict[str, list[Result]]:
    value_names = ACCURACY_NAMES if plot_accuracy else PERPLEXITY_NAMES
    task_value_name = {}
    full_results = {}
    allowed_property_values = {
        "permutalign_mode": permutalign_modes,
        "orthogonalign_mode": orthogonalign_modes,
        "num_weight_groups": num_weight_groups,
        "weight_group_axis": weight_group_axis,
        "rank": ranks,
        "param_name": param_names,
        "param_map": param_maps,
    }

    for model_name in model_names:
        model_results = []
        model_dir = os.path.join(results_dir, model_name)
        results_path = os.path.join(model_dir, RESULTS_FILE)
        vanilla_results_path = os.path.join(model_dir, VANILLA_RESULTS_FILE)
        with open(vanilla_results_path, "r") as vanilla_results_file:
            vanilla_results = json.load(vanilla_results_file)
            if tasks is None:
                tasks = set()
                for task, task_results in vanilla_results.items():
                    if len(vanilla_results.keys() & value_names) > 0:
                        tasks.add(task)

            if len(task_value_name) == 0:
                for task in tasks:
                    task_results = vanilla_results[task]
                    task_value_names = task_results.keys() & value_names
                    if len(task_value_names) > 0:
                        task_value_name[task] = task_value_names.pop()
                    else:
                        tasks.remove(task)

            available_tasks = tasks & vanilla_results.keys()

            vanilla_result = default_result.copy()
            vanilla_result.value = sum([vanilla_results[task][task_value_name[task]] for task in available_tasks]) / len(available_tasks)

        with open(results_path, "r") as results_file:
            results = json.load(results_file)

            default_values = {result_property: default_value for result_property, default_value in vars(default_result).items() if result_property in allowed_property_values}

            update_full_results_recursive(model_results, results, list(default_values.items()), allowed_property_values, tasks, task_value_name)

        full_results[model_name] = model_results

    return full_results

def update_full_results_recursive(
    full_results: list[Result],
    results: dict,
    default_values: list[tuple[str, int | float | str]],
    allowed_property_values: dict[str, set[int | float | str] | None],
    tasks: set[str],
    task_value_name: dict[str, str],
    current_properties: dict[str, int | float | str] = {},
) -> None:
    if len(default_values) == 0:
        result = vars(default_result)
        result.update(current_properties)
        result.value = get_result_value(results, tasks, task_value_name)
        full_results.append(Result(**result))

        return

    property_name, default_value = default_values.pop()
    allowed_values = allowed_property_values[property_name]

    for property_value, property_results in results.items():
        if allowed_values is None or property_value in allowed_values:
            current_properties[property_name] = property_value
            update_full_results_recursive(full_results, property_results, default_values, allowed_property_values, tasks, task_value_name, current_properties)
            del current_properties[property_name]

    default_values.insert(0, (property_name, default_value))

def get_result_value(
    results: list[Result],
    tasks: set[str],
    task_value_name: dict[str, str],
) -> float:
    available_tasks = tasks & results.keys()
    return sum([results[task][task_value_name[task]] for task in available_tasks]) / len(available_tasks)

def plot_results(
    results: dict[str, list[Result]],
) -> None:
    # scatter plot where same model is same color

    for model_name, model_results in results.items():
        for result in model_results:
            num_params = calculate_num_params(model_name, result)
            plt.scatter(num_params, result.value, label=model_name)

    plt.show()

def calculate_model_params_llama(
    config: PretrainedConfig,
    result: Result,
) -> int:
    reduced_param_names = set(result.param_name.split(","))

    num_params = config.hidden_size  # pre-lm head rmsnorm

    per_layer_weights = {
        "self_attn.k_proj": config.hidden_size ** 2,
        "self_attn.q_proj": config.hidden_size ** 2,
        "self_attn.v_proj": config.hidden_size ** 2,
        "self_attn.out_proj": config.hidden_size ** 2,
        "mlp.gate_proj": config.hidden_size ** 2,
        "mlp.up_proj": config.hidden_size ** 2,
        "mlp.down_proj": config.hidden_size ** 2,
        "input_layernorm": config.hidden_size,
        "post_attention_layernorm": config.hidden_size,
    }
    
    for param_name in per_layer_weights:
        if param_name in reduced_param_names:
            per_layer_weights[param_name] *= result.rank
            per_layer_weights[param_name] += (per_layer_weights[param_name] / result.num_weight_groups)
    
    per_layer_params = sum(per_layer_weights.values())

    num_params += per_layer_params * config.num_hidden_layers

    return num_params

def calculate_num_params(
    model_name: str,
    result: Result,
) -> int:
    config = AutoConfig.from_pretrained(model_name)

    embedding_params = config.hidden_size * config.vocab_size
    embedding_params *= 1 if config.tie_word_embeddings else 2

    if isinstance(config, (LlamaConfig,)):
        model_params = calculate_model_params_llama(config, result)
    else:
        raise ValueError(f"Model {model_name} not supported")

    return embedding_params + model_params

ACCURACY_NAMES = {"acc", "acc,none"}
PERPLEXITY_NAMES = {"word_perplexity,none",}
RESULTS_FILE = "results.json"
VANILLA_RESULTS_FILE = "vanilla_results.json"

def visualize(
    results_dir: os.PathLike | str = "outputs/",
    model_names: list[str] = ["meta-llama/Llama-2-7b-hf"],
    permutalign_modes: set[str] | None = None,
    orthogonalign_modes: set[str] | None = None,
    num_weight_groups: set[int] | None = None,
    weight_group_axis: set[int] | None = None,
    ranks: set[float] | None = {1/4, 1/2},
    param_names: set[str] | None = ["self_attn.k_proj", "self_attn.q_proj", "self_attn.q_proj,self_attn.k_proj"],
    param_maps: set[int | str] | None = set(0),
    tasks: set[str] | None = None,
    plot_accuracy: bool = True,  # If false, plot perplexity
) -> None:
    param_maps = list(param_maps)
    model_lengths = [AutoConfig.from_pretrained(model_name).num_hidden_layers for model_name in model_names]

    for i, param_map in enumerate(param_maps):
        if isinstance(param_map, (int, str)):
            possible_param_maps = []
            for model_length in model_lengths:
                candidate_param_map = {
                    i: param_map for i in range(model_length)
                }

                possible_param_maps.append(candidate_param_map)

            param_maps.extend(possible_param_maps)

    full_results = filter_results(
        results_dir=results_dir,
        model_names=model_names,
        permutalign_modes=permutalign_modes,
        orthogonalign_modes=orthogonalign_modes,
        num_weight_groups=num_weight_groups,
        weight_group_axis=weight_group_axis,
        ranks=ranks,
        param_names=param_names,
        param_maps=param_maps,
        tasks=tasks,
        plot_accuracy=plot_accuracy,
    )

    plot_results(full_results)


if __name__ == "__main__":
    visualize()
