import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass
from core.lorafy.lorafied_weight import LoRAfiedLinear
from core.utils import get_param_ancestors


@dataclass
class LoRAfyParameterConfig:
    base_param: str
    derived_param: str
    rank: int | float


WEIGHT_SUFFIX = ".weight"
def remove_weight_from_name(param_name: str) -> str:
    if param_name.endswith(WEIGHT_SUFFIX):
        return param_name[:-len(WEIGHT_SUFFIX)]

    return param_name


def lorafy_model(
    model: nn.Module,
    *param_configs: list[LoRAfyParameterConfig],
    inplace: bool = False,
    delete_original_params: bool = True,
    move_device: str | None = None,
    approximate_lora: bool = True
) -> nn.Module:
    """
    LoRAfy a model by replacing parameters with low-rank approximations.

    :param model: Model to LoRAfy
    :param param_configs: List of LoRAfyParameterConfig objects
    :param inplace: Whether to modify model in-place or return a new model
    :param delete_original_params: Whether to delete the original parameters after LoRAfying,
        only applicable if inplace=True
    :return: LoRAfied model
    """
    if inplace:
        lorafied_model = model
    else:
        lorafied_model = deepcopy(model)

    for param_config in param_configs:
        base_param_name = remove_weight_from_name(param_config.base_param)
        derived_param_name = remove_weight_from_name(param_config.derived_param)
        base_param_ancestors, _ = get_param_ancestors(lorafied_model, base_param_name)
        derived_param_ancestors, derived_param_ancestor_names = get_param_ancestors(
            lorafied_model,
            derived_param_name
        )

        lorafied_param = LoRAfiedLinear.from_weight_delta(
            base_param_ancestors[-1],
            derived_param_ancestors[-1],
            param_config.rank,
            move_device,
            approximate_lora,
        )
        setattr(derived_param_ancestors[-2], derived_param_ancestor_names[-1], lorafied_param)

        if inplace and delete_original_params:
            del derived_param_ancestors[-1]
        elif inplace:
            setattr(
                derived_param_ancestors[-2],
                f"{derived_param_ancestor_names[-1]}_original",
                derived_param_ancestors[-1]
            )

    return lorafied_model
