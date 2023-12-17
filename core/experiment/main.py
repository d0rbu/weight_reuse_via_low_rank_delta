import os
from itertools import product, chain, combinations
from typing import Iterable


def powerset(iterable: Iterable):
    full_set = list(iterable)
    return chain.from_iterable(
        combinations(full_set, num_elements)
        for num_elements in range(len(full_set) + 1)
    )

def lorafy_parameter_grid_eval(
    output_file: os.PathLike | str,
    model: str = "meta-llama/Llama-2-7b-hf",
    blocks_name: str = "model.layers",
    ranks: Iterable[int | float] = (1/16, 1/8, 1/4),
    param_names: Iterable[str] = ("self_attn.q_proj", "self_attn.k_proj"),
    mappings: Iterable[dict[int, int]] = (
        {0: 0, 1: 1, 2: 2, 3: 3},
        {0: 0, 1: 1, 2: 2},
        {0: 0, 1: 1},
        {0: 0}
    ),
) -> None:
    pass


if __name__ == "__main__":
    lorafy_parameter_grid_eval("output.json")
