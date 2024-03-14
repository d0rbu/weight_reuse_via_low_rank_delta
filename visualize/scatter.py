import os
import json
import matplotlib.pyplot as plt
import math
from typing import Self
from core.permutalign.permutalign_model import PermutalignMode
from transformers import AutoConfig, LlamaConfig, GemmaConfig, MistralConfig, PretrainedConfig
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Result:
    permutalign_mode: str = "identity"
    orthogonalign_mode: str = json.dumps(None)
    num_weight_groups: int = 1
    weight_group_axis: int = 0
    rank: float = 1.0
    param_name: str = None
    param_map: int | str = "0"
    value: float = 0.0

    def copy(self: Self) -> "Result":
        return Result(
            permutalign_mode=self.permutalign_mode,
            orthogonalign_mode=self.orthogonalign_mode,
            num_weight_groups=self.num_weight_groups,
            weight_group_axis=self.weight_group_axis,
            rank=self.rank,
            param_name=self.param_name,
            param_map=self.param_map,
            value=self.value,
        )

def filter_results(
    results_dir: os.PathLike | str,
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
    allowed_property_values = OrderedDict([
        ("permutalign_mode", permutalign_modes),
        ("orthogonalign_mode", orthogonalign_modes),
        ("num_weight_groups", num_weight_groups),
        ("weight_group_axis", weight_group_axis),
        ("rank", ranks),
        ("param_name", param_names),
        ("param_map", param_maps),
    ])

    for model_name in model_names:
        model_results = []
        model_dir = os.path.join(results_dir, model_name)
        results_path = os.path.join(model_dir, RESULTS_FILE)
        vanilla_results_path = os.path.join(model_dir, VANILLA_RESULTS_FILE)
        with open(vanilla_results_path, "r") as vanilla_results_file:
            vanilla_results = json.load(vanilla_results_file)
            if tasks is None:
                tasks = set(vanilla_results.keys())

            # if we are measuring accuracy, for example, remove tasks that don't have accuracy
            if len(task_value_name) == 0:
                to_remove = set()
                for task in tasks:
                    task_results = vanilla_results[task]
                    task_value_names = task_results.keys() & value_names
                    if len(task_value_names) > 0:
                        task_value_name[task] = task_value_names.pop()
                    else:
                        to_remove.add(task)
                
                tasks -= to_remove

            vanilla_result = Result()
            vanilla_result.value = get_result_value(vanilla_results, tasks, task_value_name)

            model_results.append(vanilla_result)

        with open(results_path, "r") as results_file:
            results = json.load(results_file)

            default_values = OrderedDict([
                (result_property, default_value)
                for result_property, default_value in vars(Result()).items()
                if result_property in allowed_property_values.keys()
            ])

            update_full_results_recursive(model_results, results, list(reversed(default_values.items())), allowed_property_values, tasks, task_value_name)

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
        result = vars(Result())
        result.update(current_properties)
        result["value"] = get_result_value(results, tasks, task_value_name)

        default_result_dict = vars(Result())
        result = {
            key: type(default_result_dict[key])(value) if default_result_dict[key] is not None else value
            for key, value in result.items()
        }

        full_results.append(Result(**result))

        return

    property_name, default_value = default_values.pop()
    allowed_values = allowed_property_values[property_name]

    for property_value, property_results in results.items():
        try:
            default_value_type = type(default_value)
            if not isinstance(property_value, default_value_type):
                property_value = default_value_type(property_value) if default_value is not None else property_value
        except ValueError as e:
            continue

        if allowed_values is not None:
            if property_value not in allowed_values:
                continue

        current_properties[property_name] = property_value
        update_full_results_recursive(full_results, property_results, default_values, allowed_property_values, tasks, task_value_name, current_properties)
        del current_properties[property_name]

    default_values.append((property_name, default_value))

def get_result_value(
    results: list[Result],
    tasks: set[str],
    task_value_name: dict[str, str],
) -> float:
    available_tasks = tasks & results.keys()

    if len(available_tasks) == 0:
        return 0

    return sum([results[task][task_value_name[task]] for task in available_tasks]) / len(available_tasks)

def plot_results(
    results: dict[str, list[Result]],
    cmap: str = "jet",
) -> None:
    # scatter plot, color by rank (log scale)
    cm = plt.get_cmap(cmap)

    for model_name, model_results in results.items():
        min_rank = min([result.rank for result in model_results])
        max_rank = max([result.rank for result in model_results])
        for i, result in enumerate(model_results):
            if result.value == 0:
                continue

            num_params = calculate_num_params(model_name, result)
            plt.scatter(
                num_params,
                result.value,
                marker=get_marker(result),
                color=get_color(result, cm, min_rank, max_rank),
                label=i,
            )

        plt.title(model_name)
        plt.xlabel("Number of Parameters")
        plt.ylabel("Accuracy" if ACCURACY_NAMES else "Perplexity")

        plt.show()

def get_color(
    result: Result,
    cm: plt.cm.ScalarMappable,
    min_rank: float,
    max_rank: float,
) -> tuple[float, float, float, float]:
    return cm(0)

    rank = math.log2(result.rank)
    min_rank = math.log2(min_rank)
    max_rank = math.log2(max_rank)

    if max_rank > min_rank:
        return cm((rank - min_rank) / (max_rank - min_rank))
    else:
        return cm(0.5)

def get_marker(
    result: Result,
) -> str:
    if result.rank == 1.0:  # vanilla result
        return "*"
        return "s"

    return "o"
    if result.param_name == "self_attn.k_proj":
        if result.orthogonalign_mode == None:
            if result.permutalign_mode == PermutalignMode.IDENTITY:
                return "x"
            elif result.permutalign_mode == PermutalignMode.OPTIMIZE:
                return "x"
        else:
            if result.permutalign_mode == PermutalignMode.IDENTITY:
                return "x"
            elif result.permutalign_mode == PermutalignMode.OPTIMIZE:
                return "X"
    elif result.param_name == "self_attn.q_proj":
        if result.orthogonalign_mode == None:
            if result.permutalign_mode == PermutalignMode.IDENTITY:
                return "+"
            elif result.permutalign_mode == PermutalignMode.OPTIMIZE:
                return "+"
        else:
            if result.permutalign_mode == PermutalignMode.IDENTITY:
                return "+"
            elif result.permutalign_mode == PermutalignMode.OPTIMIZE:
                return "P"
    elif result.param_name == "self_attn.q_proj,self_attn.k_proj":
        if result.orthogonalign_mode == None:
            if result.permutalign_mode == PermutalignMode.IDENTITY:
                return "."
            elif result.permutalign_mode == PermutalignMode.OPTIMIZE:
                return "."
        else:
            if result.permutalign_mode == PermutalignMode.IDENTITY:
                return "."
            elif result.permutalign_mode == PermutalignMode.OPTIMIZE:
                return "*"

def calculate_model_params_llama(
    config: PretrainedConfig,
    result: Result,
) -> int:
    reduced_param_names = set(result.param_name.split(",")) if result.param_name is not None else set()

    num_params = config.hidden_size  # pre-lm head rmsnorm

    per_layer_weights = {
        "self_attn.k_proj": config.hidden_size ** 2,
        "self_attn.q_proj": config.hidden_size ** 2,
        "self_attn.v_proj": config.hidden_size ** 2,
        "self_attn.out_proj": config.hidden_size ** 2,
        "mlp.gate_proj": config.hidden_size * config.intermediate_size,
        "mlp.up_proj": config.hidden_size * config.intermediate_size,
        "mlp.down_proj": config.hidden_size * config.intermediate_size,
        "input_layernorm": config.hidden_size,
        "post_attention_layernorm": config.hidden_size,
    }

    for param_name in per_layer_weights:
        if param_name in reduced_param_names:
            per_layer_weights[param_name] *= result.rank
            per_layer_weights[param_name] += (per_layer_weights[param_name] / int(result.num_weight_groups))

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
    results_dir: os.PathLike | str = "outputs",
    model_names: list[str] = ["meta-llama/Llama-2-7b-hf"],
    permutalign_modes: set[str] | None = None,
    orthogonalign_modes: set[str] | None = None,
    num_weight_groups: set[int] | None = None,
    weight_group_axis: set[int] | None = None,
    ranks: set[float] | None = {1/16, 1/8, 1/4, 1/2},
    param_names: set[str] | None = ["self_attn.k_proj", "self_attn.q_proj", "self_attn.q_proj,self_attn.k_proj"],
    param_maps: set[int | str] | None = set((0,)),
    tasks: set[str] | None = None,
    plot_accuracy: bool = True,  # If false, plot perplexity
) -> None:
    model_lengths = [AutoConfig.from_pretrained(model_name).num_hidden_layers for model_name in model_names]

    removed_param_maps = set()
    added_param_maps = set()
    for param_map in param_maps:
        if isinstance(param_map, (int, str)):
            possible_param_maps = set()
            for model_length in model_lengths:
                candidate_param_map = {
                    i: param_map for i in range(model_length)
                }
                del candidate_param_map[param_map]

                possible_param_maps.add(json.dumps(candidate_param_map, sort_keys=True))

            added_param_maps |= possible_param_maps

            if isinstance(param_map, int):
                removed_param_maps.add(param_map)
                added_param_maps.add(str(param_map))

    param_maps -= removed_param_maps
    param_maps |= added_param_maps

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
