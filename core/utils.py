import torch.nn as nn
from functools import partial
from enum import IntEnum


class Verbosity(IntEnum):
    ERROR: int = 0
    WARN: int = 1
    INFO: int = 2
    INFO_0: int = 2
    INFO_1: int = 3


def get_param_ancestors(model: nn.Module, param_name: str) -> list[nn.Module]:
    ancestors = [model]
    param_hierarchy = param_name.split(".")

    module = model
    for param in param_hierarchy:
        module = getattr(module, param)
        ancestors.append(module)

    return ancestors, param_hierarchy

def log(msg: str, current_verbosity: Verbosity, msg_verbosity: Verbosity):
    if current_verbosity < msg_verbosity:  # If the current verbosity is not important enough
        return
    
    print(msg)

log_error = partial(log, msg_verbosity = Verbosity.ERROR)
log_warn = partial(log, msg_verbosity = Verbosity.WARN)
log_info = partial(log, msg_verbosity = Verbosity.INFO_0)
log_info_0 = log_info
log_info_1 = partial(log, msg_verbosity = Verbosity.INFO_1)