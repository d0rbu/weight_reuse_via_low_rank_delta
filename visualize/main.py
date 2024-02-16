import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping


ACCURACY_NAMES = ["acc", "acc,none"]
PERPLEXITY_NAMES = ["word_perplexity,none"]

def get_task_accuracies_and_avg(
    task_results: Mapping | None,
    vanilla_results: Mapping | None = None,  # if this is passed in, the accuracies will be relative to the vanilla results
    vanilla_acc: float | None = None,
    acc_names: list[str] = ACCURACY_NAMES,
) -> tuple[Mapping[str, float], float, set[str]]:
    relevant_tasks = set()

    if vanilla_results is not None and vanilla_acc is None:
        num_accs = 0
        sum_accs = 0

        for task, results in vanilla_results.items():
            for acc_name in acc_names:
                if acc_name in results:
                    num_accs += 1
                    sum_accs += results[acc_name]
                    relevant_tasks.add(task)
                    break

        vanilla_acc = sum_accs / num_accs if num_accs > 0 else None

    if task_results is None:
        return {}, 0., vanilla_acc, relevant_tasks

    task_accuracies = {}

    num_accs = 0
    sum_accs = 0

    for task, results in task_results.items():
        if vanilla_results is not None:
            assert task in vanilla_results, f"Task {task} not found in vanilla results!"
            vanilla_task_results = vanilla_results[task]
        else:
            vanilla_task_results = {acc_name: 0. for acc_name in acc_names}

        for acc_name in acc_names:
            if acc_name in results:
                acc = results[acc_name] - vanilla_task_results[acc_name]

                num_accs += 1
                sum_accs += acc
                task_accuracies[task] = acc
                break

    return task_accuracies, sum_accs / num_accs if num_accs > 0 else 0., vanilla_acc, relevant_tasks

# Number of experiments per set of parameters
def get_num_exp_per_params(full_results: Mapping) -> int:
    return len(next(iter(next(iter(full_results.values())).values())))

def single_param_rank_layer_graph(
    full_results: Mapping,
    vanilla_results: Mapping,
    acc_names: list[str] = ACCURACY_NAMES,
) -> None:
    param_heatmaps = {}
    num_ranks = len(full_results)
    num_layers = get_num_exp_per_params(full_results)  # Since there is only one param, # experiments = # num layers
    ranks = sorted(full_results.keys())
    ranks_idx = {rank: idx for idx, rank in enumerate(ranks)}
    extreme = 0
    vanilla_acc = None
    relevant_tasks = set()

    for rank, rank_results in full_results.items():
        for param, param_results in rank_results.items():
            param: str = param[0]
            rank_idx: int = ranks_idx[rank]
            if param in param_heatmaps:
                avg_heatmap = param_heatmaps[param]["avg"]
                task_heatmaps = {
                    task: heatmap for task, heatmap in param_heatmaps[param].items() if task != "avg"
                }
            else:
                avg_heatmap = np.empty((num_ranks, num_layers))
                task_heatmaps = {
                    task: np.empty((num_ranks, num_layers)) for task in param_results[next(iter(param_results))].keys()
                }
                param_heatmaps[param] = {
                    "avg": avg_heatmap,
                    **task_heatmaps,
                }

            for base_layer, task_results in param_results.items():
                base_layer: int = base_layer[0]
                task_accuracies, avg_accuracy, vanilla_acc, specific_relevant_tasks = get_task_accuracies_and_avg(task_results, vanilla_results, vanilla_acc, acc_names)
                relevant_tasks = relevant_tasks.union(specific_relevant_tasks)
                avg_heatmap[rank_idx, base_layer] = avg_accuracy
                extreme = max(extreme, abs(avg_accuracy))
                for task, task_accuracy in task_accuracies.items():
                    extreme = max(extreme, abs(task_accuracy))
                    task_heatmaps[task][rank_idx, base_layer] = task_accuracy
    
    show_task_heatmaps = input("Show task heatmaps? (y/n): ").lower() == "y"

    for param, task_heatmaps in param_heatmaps.items():
        for task, heatmap in task_heatmaps.items():
            if not (show_task_heatmaps or task == "avg"):
                continue
            
            if task not in relevant_tasks and task != "avg":
                continue

            plt.imshow(heatmap)
            plt.xlabel("Base layer")
            plt.ylabel("Rank")
            plt.xticks(range(0, num_layers, 4))
            plt.yticks(range(num_ranks), ranks)
            plt.colorbar()

            if vanilla_results is None:
                plt.title(f"{param} {task}")
                plt.set_cmap(f"hot{'_r' if acc_names == PERPLEXITY_NAMES else ''}")
                if acc_names == PERPLEXITY_NAMES:
                    plt.clim(0, extreme)
                else:
                    plt.clim(0, 1)
            else:
                plt.title(f"{param} {task}, {vanilla_acc:.3f} vanilla")
                plt.set_cmap(f"RdYlBu{'_r' if acc_names == PERPLEXITY_NAMES else ''}")
                plt.clim(-extreme, extreme)

            plt.show()

def two_param_layer_layer_graph(
    full_results: Mapping,
    vanilla_results: Mapping,
    acc_names: list[str] = ACCURACY_NAMES,
) -> None:
    rank_param_heatmaps = {
        rank: {} for rank in full_results
    }
    num_layers = round(math.sqrt(get_num_exp_per_params(full_results)))
    extreme = 0
    vanilla_acc = None
    relevant_tasks = set()

    for rank, rank_results in full_results.items():
        for params, param_results in rank_results.items():
            if params in rank_param_heatmaps[rank]:
                avg_heatmap = rank_param_heatmaps[rank][params]["avg"]
                task_heatmaps = {
                    task: heatmap for task, heatmap in param_heatmaps[params].items() if task != "avg"
                }
            else:
                avg_heatmap = np.empty((num_layers, num_layers))
                task_heatmaps = {
                    task: np.empty((num_layers, num_layers)) for task in param_results[next(iter(param_results))].keys()
                }
                rank_param_heatmaps[rank][params] = {
                    "avg": avg_heatmap,
                    **task_heatmaps,
                }

            for base_layers, task_results in param_results.items():
                first_base_layer, second_base_layer = base_layers
                task_accuracies, avg_accuracy, vanilla_acc, specific_relevant_tasks = get_task_accuracies_and_avg(task_results, vanilla_results, vanilla_acc, acc_names)
                relevant_tasks = relevant_tasks.union(specific_relevant_tasks)
                avg_heatmap[first_base_layer, second_base_layer] = avg_accuracy
                extreme = max(extreme, abs(avg_accuracy))

                for task, task_accuracy in task_accuracies.items():
                    task_heatmaps[task][first_base_layer, second_base_layer] = task_accuracy
                    extreme = max(extreme, abs(task_accuracy))
    
    show_task_heatmaps = input("Show task heatmaps? (y/n): ").lower() == "y"

    for rank, param_heatmaps in rank_param_heatmaps.items():
        for params, task_heatmaps in param_heatmaps.items():
            for task, heatmap in task_heatmaps.items():
                if not (show_task_heatmaps or task == "avg"):
                    continue

                if task not in relevant_tasks and task != "avg":
                    continue

                plt.imshow(heatmap)
                plt.xlabel(params[0])
                plt.ylabel(params[1])
                plt.xticks(range(0, num_layers, 4))
                plt.yticks(range(0, num_layers, 4))
                plt.colorbar()

                if vanilla_results is None:
                    plt.title(f"rank {rank} {task}")
                    plt.set_cmap(f"hot{'_r' if acc_names == PERPLEXITY_NAMES else ''}")
                    if acc_names == PERPLEXITY_NAMES:
                        plt.clim(0, extreme)
                    else:
                        plt.clim(0, 1)
                else:
                    plt.title(f"rank {rank} {task}, {vanilla_acc:.3f} vanilla")
                    plt.set_cmap(f"RdYlBu{'_r' if acc_names == PERPLEXITY_NAMES else ''}")
                    plt.clim(-extreme, extreme)

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

RESULTS_FILE = "results.json"
VANILLA_RESULTS_FILE = "vanilla_results.json"

def visualize(
    output_dir: os.PathLike | str = "outputs/",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    relative_to_vanilla: bool = True,
) -> None:
    results_dir = os.path.join(output_dir, model_name)
    results_path = os.path.join(results_dir, RESULTS_FILE)
    vanilla_results_path = os.path.join(results_dir, VANILLA_RESULTS_FILE)

    with open(results_path, "r") as results_file:
        full_results = json.load(results_file)
    
    if relative_to_vanilla:
        with open(vanilla_results_path, "r") as vanilla_results_file:
            vanilla_results = json.load(vanilla_results_file)
    else:
        vanilla_results = None
    
    split_results = clean_and_split_by_num_params(full_results)

    if 1 in split_results:
        single_param_results = split_results.pop(1)
        single_param_rank_layer_graph(single_param_results, vanilla_results, PERPLEXITY_NAMES)
    if 2 in split_results:
        two_param_results = split_results.pop(2)
        two_param_layer_layer_graph(two_param_results, vanilla_results, PERPLEXITY_NAMES)
    if len(split_results) > 0:
        raise ValueError(f"No code written to visualize experiments of these many parameters: {split_results.keys()}")


if __name__ == "__main__":
    visualize()
