from torch.nn import nn


def get_param_ancestors(model: nn.Module, param_name: str) -> list[nn.Module]:
    ancestors = [model]
    param_hierarchy = param_name.split(".")

    module = model
    for param in param_hierarchy:
        module = getattr(module, param)
        ancestors.append(module)

    return ancestors, param_hierarchy