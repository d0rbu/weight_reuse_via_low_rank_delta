import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping


ACCURACY_NAMES = ["acc", "acc,none"]

def get_avg_accuracy(task_results: Mapping | None) -> float:
    if task_results is None:
        return 0.

    num_accs = 0
    sum_accs = 0
    for task, results in task_results.items():
        for acc_name in ACCURACY_NAMES:
            if acc_name in results:
                num_accs += 1
                sum_accs += results[acc_name]
                break

    return sum_accs / num_accs if num_accs > 0 else 0.

# Number of experiments per set of parameters
def get_num_exp_per_params(full_results: Mapping) -> int:
    return len(next(iter(next(iter(full_results.values())).values())))

def single_param_rank_layer_graph(full_results: Mapping) -> None:
    param_heatmaps = {}
    num_ranks = len(full_results)
    num_layers = get_num_exp_per_params(full_results)  # Since there is only one param, # experiments = # num layers
    ranks = sorted(full_results.keys())
    ranks_idx = {rank: idx for idx, rank in enumerate(ranks)}

    for rank, rank_results in full_results.items():
        for param, param_results in rank_results.items():
            param: str = param[0]
            rank_idx: int = ranks_idx[rank]
            if param in param_heatmaps:
                heatmap = param_heatmaps[param]
            else:
                heatmap = np.empty((num_ranks, num_layers))
                param_heatmaps[param] = heatmap

            for base_layer, task_results in param_results.items():
                base_layer: int = base_layer[0]
                avg_accuracy = get_avg_accuracy(task_results)
                heatmap[rank_idx, base_layer] = avg_accuracy

    for param, heatmap in param_heatmaps.items():
        plt.imshow(heatmap)
        plt.title(f"{param} avg acc")
        plt.xlabel("Base layer")
        plt.ylabel("Rank")
        plt.xticks(range(0, num_layers, 4))
        plt.yticks(range(num_ranks), ranks)
        plt.colorbar()
        plt.show()

def two_param_layer_layer_graph(full_results: Mapping) -> None:
    rank_param_heatmaps = {
        rank: {} for rank in full_results
    }
    num_layers = round(math.sqrt(get_num_exp_per_params(full_results)))

    for rank, rank_results in full_results.items():
        for params, param_results in rank_results.items():
            if params in rank_param_heatmaps[rank]:
                heatmap = rank_param_heatmaps[rank][params]
            else:
                heatmap = np.empty((num_layers, num_layers))
                rank_param_heatmaps[rank][params] = heatmap

            for base_layers, task_results in param_results.items():
                first_base_layer, second_base_layer = base_layers
                avg_accuracy = get_avg_accuracy(task_results)
                heatmap[first_base_layer, second_base_layer] = avg_accuracy

    for rank, param_heatmaps in rank_param_heatmaps.items():
        for params, heatmap in param_heatmaps.items():
            plt.imshow(heatmap)
            plt.title(f"rank {rank} avg acc")
            plt.xlabel(params[0])
            plt.ylabel(params[1])
            plt.xticks(range(0, num_layers, 4))
            plt.yticks(range(0, num_layers, 4))
            plt.colorbar()
            plt.show()

# Clean and split up full results based on number of parameters
def clean_and_split_by_num_params(full_results: Mapping) -> Mapping[str, Mapping]:
    split_results = {}

    for rank_str, rank_results in full_results.items():
        rank = float(rank_str)
        for params_str, param_results in rank_results.items():
            params = tuple(params_str.split(","))
            num_params = len(params)

            if num_params not in split_results:
                split_results[num_params] = {}
            if rank not in split_results[num_params]:
                split_results[num_params][rank] = {}
            if params not in split_results[num_params][rank]:
                split_results[num_params][rank][params] = {}

            for mapping_str, task_results in param_results.items():
                mapping = json.loads(mapping_str)
                if isinstance(mapping, int):
                    base_layer = mapping
                    base_layers = [base_layer] * len(params)
                elif isinstance(mapping, dict):
                    base_layer = next(iter(mapping.values()))
                    base_layers = [base_layer] * len(params)
                elif isinstance(mapping, list):
                    base_layers = [next(iter(param_mapping.values())) for param_mapping in mapping]
                else:
                    raise ValueError(f"Unknown type of mapping!\n{mapping}")
                
                base_layers = tuple(base_layers)

                split_results[num_params][rank][params][base_layers] = task_results
    
    return split_results


def visualize(results_path: os.PathLike | str = os.path.join("outputs", "results.json")) -> None:
    with open(results_path, "r") as results_file:
        full_results = json.load(results_file)
    
    split_results = clean_and_split_by_num_params(full_results)

    if 1 in split_results:
        single_param_results = split_results.pop(1)
        single_param_rank_layer_graph(single_param_results)
    if 2 in split_results:
        two_param_results = split_results.pop(2)
        two_param_layer_layer_graph(two_param_results)
    if len(split_results) > 0:
        raise ValueError(f"No code written to visualize experiments of these many parameters: {split_results.keys()}")


if __name__ == "__main__":
    visualize()
