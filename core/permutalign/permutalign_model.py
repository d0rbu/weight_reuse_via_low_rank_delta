import os
import torch as th
import torch.nn as nn
import pygmtools as pygm
from core.utils import Verbosity, log_warn
from enum import StrEnum


pygm.set_backend('pytorch')


class PermutalignMode(StrEnum):
    IDENTITY: str = "identity"
    OPTIMIZE: str = "optimize"


def permutalign_model(
    layers: nn.ModuleList | nn.Sequential,
    num_heads: int,
    mode: PermutalignMode = PermutalignMode.OPTIMIZE,
    cache_path: str | None = None,
    k_name: str = "self_attn.k_proj",
    q_name: str = "self_attn.q_proj",
    v_name: str = "self_attn.v_proj",
    o_name: str = "self_attn.o_proj",
    attn_maps: th.Tensor | None = None,
    verbosity: Verbosity = Verbosity.INFO,
    move_device: str | None = None,
) -> None:
    if mode == PermutalignMode.IDENTITY:
        return

    assert attn_maps is not None, "attn_maps must be provided when mode is not identity"
    permutation_matrices = None
    try:
        if cache_path and (permutation_matrices := th.load(cache_path, map_location=move_device or "cpu")):
            # Test if we can properly load the cache file
            assert len(permutation_matrices) == len(layers), f"Cache file does not contain the expected number of layers"
            assert all(isinstance(matrix, th.Tensor) for matrix in permutation_matrices), f"Cache file does not contain the expected permutation matrices"
    except RuntimeError as e:
        log_warn(f"Unable to read cache file {cache_path}, recalculating...", verbosity)
        os.remove(cache_path)
        permutation_matrices = None

    if permutation_matrices is None:
        permutation_matrices = calculate_permutation_matrices(
            layers,
            num_heads,
            k_name,
            q_name,
            v_name,
            o_name,
            attn_maps,
            verbosity,
            move_device,
        )

    for layer_idx, layer in enumerate(layers):
        k = getattr(layer, k_name)
        q = getattr(layer, q_name)
        v = getattr(layer, v_name)
        o = getattr(layer, o_name)

        head_dim = k.weight.shape[0] // num_heads

        permutation_matrix = permutation_matrices[layer_idx]

        # kronecker product with identity (apply permutation on a head-level)
        permutation_matrix = th.kron(permutation_matrix, th.eye(head_dim))

        if move_device:
            permutation_matrix = permutation_matrix.to(move_device)
            permuted_k = permutation_matrix @ k.weight.data.to(move_device)
            permuted_q = permutation_matrix @ q.weight.data.to(move_device)
            permuted_v = permutation_matrix @ v.weight.data.to(move_device)
            permuted_o = o.weight.data.to(move_device) @ permutation_matrix.T
        else:
            permuted_k = permutation_matrix.to(k.weight.device) @ k.weight.data
            permuted_q = permutation_matrix.to(q.weight.device) @ q.weight.data
            permuted_v = permutation_matrix.to(v.weight.device) @ v.weight.data
            permuted_o = o.weight.data @ permutation_matrix.T.to(o.weight.device)

        # Apply permutation matrices
        k.weight.data.copy_(permuted_k)
        q.weight.data.copy_(permuted_q)
        v.weight.data.copy_(permuted_v)
        o.weight.data.copy_(permuted_o)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        th.save(
            permutation_matrices,
            cache_path,
        )

def calculate_permutation_matrices(
    layer: nn.ModuleList | nn.Sequential,
    num_heads: int,
    k_name: str,
    q_name: str,
    v_name: str,
    o_name: str,
    attn_maps: th.Tensor,
    verbosity: Verbosity,
    move_device: str | None,
) -> list[th.Tensor]:
    if move_device is None:
        move_device = "cpu"

    initial_k = getattr(layer[0], k_name).weight.data
    initial_q = getattr(layer[0], q_name).weight.data
    initial_v = getattr(layer[0], v_name).weight.data
    initial_o = getattr(layer[0], o_name).weight.data

    if move_device:
        initial_k = initial_k.to(move_device)
        initial_q = initial_q.to(move_device)
        initial_v = initial_v.to(move_device)
        initial_o = initial_o.to(move_device)
        initial_attn_map = attn_maps[0].to(move_device)

    initial_attn_map = initial_attn_map.transpose(0, 1)  # (H, B, T, T)
    initial_attn_map = initial_attn_map[initial_attn_map > 0.0].view(attn_map.shape[1], -1).transpose(0, 1)  # (H, B, T, T) -> (A, H)

    head_dim = initial_k.shape[0] // num_heads
    permutation_matrices = [th.eye(head_dim).to(move_device)]  # center around layer 0

    for i, (layer, attn_map) in enumerate(zip(layer[1:], attn_maps[1:]), 1):
        # solve LAP for permutation matrix
        k = getattr(layer, k_name).weight.data
        q = getattr(layer, q_name).weight.data
        v = getattr(layer, v_name).weight.data
        o = getattr(layer, o_name).weight.data

        if move_device:
            k = k.to(move_device)
            q = q.to(move_device)
            v = v.to(move_device)
            o = o.to(move_device)
        else:
            initial_k = initial_k.to(k.device)
            initial_q = initial_q.to(q.device)
            initial_v = initial_v.to(v.device)
            initial_o = initial_o.to(o.device)

        masked_attn_map = attn_map.transpose(0, 1)
        masked_attn_map = masked_attn_map[masked_attn_map > 0.0].view(attn_map.shape[1], -1)  # (H, B, T, T) -> (H, A)

        similarity = initial_attn_map @ masked_attn_map

        permutation_matrix = pygm.linear_solvers.hungarian(similarity)

        permutation_matrices.append(permutation_matrix)
    
    return permutation_matrices
