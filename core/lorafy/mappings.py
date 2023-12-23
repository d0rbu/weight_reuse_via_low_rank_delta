from itertools import product, combinations
from typing import Collection


def layer_mappings(num_layers: int, num_base_layers: int = 1) -> Collection[dict[int, int]]:
    """
    Generate all possible mappings that map each layer to some number of base layers.
    """
    mappings = []

    for base_layers in combinations(range(num_layers), num_base_layers):
        for raw_mapping in product(base_layers, repeat=num_layers - num_base_layers):
            mapping = {}

            current_derived_layer = 0
            for base_layer in raw_mapping:
                while current_derived_layer in base_layers:
                    current_derived_layer += 1

                mapping[current_derived_layer] = base_layer
                current_derived_layer += 1

            mappings.append(mapping)
    
    return mappings
