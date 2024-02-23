from itertools import product, combinations
from typing import Collection, Sequence, Mapping


def layer_mappings(
    num_layers: int,
    base_layers: Sequence[Sequence[int]] | Sequence[int] | int = (0,),
) -> Collection[Mapping[int, int] | Mapping[tuple[int, int], tuple[int, int]]]:
    """
    Generate all possible mappings that map each layer to some number of base layers.

    Args:
        num_layers: The number of layers in the model.
        num_weight_groups: The number of weight groups in the model.
        base_layers: The layers that will be used as base layers. If an int is passed,
            all combinations of that number of layers will be used as base layers. If a list of ints, that will be the sole set of base layers.

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

    for base_layers in base_layer_combinations:  # assign each non-base layer to a base layer
        for raw_mapping in product(base_layers, repeat=num_layers - len(base_layers)):  # assign each non-base layer to a base layer
            mapping = {}

            current_derived_layer = 0
            for base_layer in raw_mapping:
                while current_derived_layer in base_layers:
                    current_derived_layer += 1

                mapping[current_derived_layer] = base_layer

                current_derived_layer += 1

            mappings.append(mapping)

    return mappings
