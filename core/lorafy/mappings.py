from itertools import product, combinations, permutations
from typing import Collection, Literal, Sequence, Mapping


def layer_mappings(
    num_layers: int,
    num_weight_groups: int = 1,
    base_layers: Sequence[Sequence[int]] | Sequence[int] | int = (0,),
    pseudo_weight_mappings: Sequence[Mapping[int, int]] | Literal["identity", "permute", "product"] | None = None,
) -> Collection[Mapping[int, int] | Mapping[tuple[int, int], tuple[int, int]]]:
    """
    Generate all possible mappings that map each layer to some number of base layers.

    Args:
        num_layers: The number of layers in the model.
        num_weight_groups: The number of weight groups in the model.
        base_layers: The layers that will be used as base layers. If an int is passed,
            all combinations of that number of layers will be used as base layers. If a list of ints, that will be the sole set of base layers.
        pseudo_weight_mappings: A list of dictionaries that map each layer to a base layer. If None, all possible mappings will be generated.
    
    Returns:
        A collection of dictionaries, where each dictionary maps each layer to a base layer or a pseudo weight to a base pseudo weight.
    """
    mappings = []

    if isinstance(base_layers, int):
        base_layer_combinations = combinations(range(num_layers), base_layers)
    elif all(isinstance(base_layer, int) for base_layer in base_layers):
        base_layer_combinations = [base_layers]
    else:
        base_layer_combinations = base_layers

    if pseudo_weight_mappings == "product":
        pseudo_weight_mappings = [
            {
                from_weight: to_weight for from_weight, to_weight in enumerate(pseudo_weight_map)
            } for pseudo_weight_map in product(
                range(num_weight_groups), repeat=num_weight_groups
            )
        ]
    elif pseudo_weight_mappings == "permute":
        pseudo_weight_mappings = [
            {
                from_weight: to_weight for from_weight, to_weight in enumerate(pseudo_weight_map)
            } for pseudo_weight_map in permutations(range(num_weight_groups))
        ]
    elif pseudo_weight_mappings == "identity":
        pseudo_weight_mappings = [{i: i for i in range(num_weight_groups)}]
    elif pseudo_weight_mappings is None:
        pseudo_weight_mappings = [None]

    for base_layers in base_layer_combinations:  # assign each non-base layer to a base layer
        for raw_mapping in product(base_layers, repeat=num_layers - len(base_layers)):  # assign each non-base layer to a base layer
            for pseudo_weight_mapping in pseudo_weight_mappings:
                mapping = {}

                current_derived_layer = 0
                for base_layer in raw_mapping:
                    while current_derived_layer in base_layers:
                        current_derived_layer += 1

                    if pseudo_weight_mapping is None: # if we want to ignore pseudo weights
                        mapping[current_derived_layer] = base_layer
                    else:
                        for from_weight, to_weight in pseudo_weight_mapping.items():
                            mapping[(current_derived_layer, from_weight)] = (base_layer, to_weight)

                    current_derived_layer += 1

                mappings.append(mapping)

    return mappings
