import os
import json
import matplotlib.pyplot as plt
from typing import Mapping, Sequence


def single_param_rank_layer_graph(full_results: Mapping) -> None:
    for rank, rank_results in full_results.items():
        for params, param_results in rank_results.items():
            for base_layer, task_results in param_results.items():
                base_layer: int = base_layer[0]
                for task, results in task_results.items():
                    pass

def two_param_layer_layer_graph(full_results: Mapping) -> None:
    for rank, rank_results in full_results.items():
        for params, param_results in rank_results.items():
            for base_layers, task_results in param_results.items():
                first_base_layer, second_base_layer: int = base_layers
                for task, results in task_results.items():
                    pass

# Clean and split up full results based on number of parameters
def clean_and_split_by_num_params(full_results: Mapping) -> Mapping[Mapping]:
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
