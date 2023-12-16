import torch.nn as nn
from dataclasses import dataclass
from lorafied_weight import LoRAfiedWeight


@dataclass
class LoRAfyParameterConfig:
    base_param: str
    derived_param: str
    rank: int | float


def get_param_ancestors(model: nn.Module, param_name: str) -> list[nn.Module]:
    ancestors = [model]
    param_hierarchy = param_name.split(".")
    
    module = model
    for param in param_hierarchy:
        module = getattr(module, param)
        ancestors.append(module)
    
    return ancestors, param_hierarchy


def remove_weight_from_name(param_name: str) -> str:
    if param_name.endswith(".weight"):
        return param_name[:-len(".weight")]
    
    return param_name


def lorafy_model(model: nn.Module, *param_configs: list[LoRAfyParameterConfig], inplace=False, delete_original_params=False) -> nn.Module:
    """
    LoRAfy a model by replacing parameters with low-rank approximations.

    :param model: model to LoRAfy
    :param param_configs: list of LoRAfyParameterConfig objects
    :param inplace: whether to modify model in-place or return a new model
    :param delete_original_params: whether to delete the original parameters after LoRAfying, only applicable if inplace=True
    :return: LoRAfied model
    """
    if inplace:
        lorafied_model = model
    else:
        lorafied_model = model.__class__()
        lorafied_model.load_state_dict(model.state_dict())

    for param_config in param_configs:
        base_param_name = remove_weight_from_name(param_config.base_param)
        derived_param_name = remove_weight_from_name(param_config.derived_param)
        base_param_ancestors, _ = get_param_ancestors(lorafied_model, base_param_name)
        derived_param_ancestors, derived_param_ancestor_names = get_param_ancestors(lorafied_model, derived_param_name)

        lorafied_param = LoRAfiedWeight.from_weight_delta(base_param_ancestors[-1], derived_param_ancestors[-1], param_config.rank)
        setattr(derived_param_ancestors[-2], derived_param_ancestor_names[-1], lorafied_param)

        if inplace and delete_original_params:
            del derived_param_ancestors[-1]
    
    return lorafied_model
