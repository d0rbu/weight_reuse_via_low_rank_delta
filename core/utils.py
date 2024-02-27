import torch.nn as nn
from hashlib import md5
from mpi4py import MPI
from functools import partial
from enum import IntEnum
from typing import Iterable
from itertools import chain, combinations


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


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

def powerset(iterable: Iterable, include_null_set: bool = False) -> Iterable:
    full_set = list(iterable)
    return chain.from_iterable(
        combinations(full_set, num_elements)
        for num_elements in range(0 if include_null_set else 1, len(full_set) + 1)
    )

def hash(string: str) -> int:
    return int(md5(str.encode(string)).hexdigest(), 16)

def log(msg: str, current_verbosity: Verbosity, msg_verbosity: Verbosity):
    if current_verbosity < msg_verbosity:  # If the current verbosity is not important enough
        return

    if RANK != 0:  # Only print on rank 0
        return

    print(msg)

log_error = partial(log, msg_verbosity = Verbosity.ERROR)
log_warn = partial(log, msg_verbosity = Verbosity.WARN)
log_info = partial(log, msg_verbosity = Verbosity.INFO_0)
log_info_0 = log_info
log_info_1 = partial(log, msg_verbosity = Verbosity.INFO_1)