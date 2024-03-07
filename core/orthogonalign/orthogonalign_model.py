import os
import torch as th
import torch.nn as nn
from core.utils import Verbosity, log_warn, log_info
from enum import StrEnum
from dataclasses import dataclass


class OrthogonalignMode(StrEnum):
    K: str = "k"
    Q: str = "q"


@dataclass
class OrthogonalignConfig:
    base_layer: int
    derived_layer: int
    mode: OrthogonalignMode


def orthogonalign_layer(
    layers: nn.ModuleList | nn.Sequential,
    config: OrthogonalignConfig,
    cache_path: str | None = None,
    k_name: str = "self_attn.k_proj",
    q_name: str = "self_attn.q_proj",
    verbosity: Verbosity = Verbosity.INFO,
    move_device: str | None = None,
) -> None:
    base_layer = layers[config.base_layer]
    derived_layer = layers[config.derived_layer]

    base_layer_k = getattr(base_layer, k_name)
    base_layer_q = getattr(base_layer, q_name)
    derived_layer_k = getattr(derived_layer, k_name)
    derived_layer_q = getattr(derived_layer, q_name)

    try:
        if cache_path and (cached_kq := th.load(cache_path, map_location=move_device or "cpu")):
            # Test if we can properly load the cache file
            assert set(cached_kq.keys()) == {"base", "derived", "mode", OrthogonalignMode.K, OrthogonalignMode.Q}, f"Cache file does not contain the expected keys"
            assert cached_kq["base"] == config.base_layer, f"Cache file does not contain the expected base layer {config.base_layer}, found {cached_kq['base']} instead"
            assert cached_kq["derived"] == config.derived_layer, f"Cache file does not contain the expected derived layer {config.derived_layer}, found {cached_kq['derived']} instead"
            assert cached_kq["mode"] == config.mode, f"Cache file does not contain the expected mode {config.mode}, found {cached_kq['mode']} instead"
            assert isinstance(cached_kq[OrthogonalignMode.K], th.Tensor), f"Cache file does not contain the expected K weight"
            assert isinstance(cached_kq[OrthogonalignMode.Q], th.Tensor), f"Cache file does not contain the expected Q weight"

            derived_layer_k.weight.data.copy_(cached_kq[OrthogonalignMode.K])
            derived_layer_q.weight.data.copy_(cached_kq[OrthogonalignMode.Q])

            del cached_kq[OrthogonalignMode.K], cached_kq[OrthogonalignMode.Q]
            del cached_kq

            th.cuda.empty_cache()

            return
    except RuntimeError as e:
        log_warn(f"Unable to read cache file {cache_path}, recalculating...", verbosity)
        os.remove(cache_path)

    # Solve the procrustes problem:
    # Minimize ||MA - B||_F subject to M^TM = I
    # Solution: M = UV^T, where UΣV^T = SVD(BA^T)
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    derived_weight = derived_layer_k if config.mode == OrthogonalignMode.K else derived_layer_q
    base_weight = base_layer_k if config.mode == OrthogonalignMode.K else base_layer_q
    derived_weight, base_weight = derived_weight.weight.data.to(move_device), base_weight.weight.data.to(move_device)

    U, _, Vh = th.linalg.svd(base_weight @ derived_weight.T)
    M = U @ Vh

    del U, Vh

    M = M.to(derived_weight.device)

    new_derived_k = M @ derived_weight.weight.data
    new_derived_q = M @ derived_weight.weight.data
    derived_layer_k.weight.data.copy_(new_derived_k)
    derived_layer_q.weight.data.copy_(new_derived_q)

    del new_derived_k, new_derived_q, M

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        th.save(
            {
                "base": config.base_layer,
                "derived": config.derived_layer,
                "mode": config.mode,
                OrthogonalignMode.K: new_derived_k,
                OrthogonalignMode.Q: new_derived_q,
            },
            cache_path,
        )


def orthogonalign_model_layerwise(
    layers: nn.ModuleList | nn.Sequential,
    mapping: dict[int, int],
    mode: OrthogonalignMode | None = None,
    cache_paths: dict[str, str] = {},  
    k_name: str = "self_attn.k_proj",
    q_name: str = "self_attn.q_proj",
    move_device: str | None = None,
) -> None:
    if mode is None:
        return

    for to_layer, from_layer in mapping.items():
        cache_path = cache_paths.get((to_layer, from_layer), None)

        orthogonalign_layer(
            layers = layers,
            orthogonalign_config = OrthogonalignConfig(
                base_layer = from_layer,
                derived_layer = to_layer,
                mode = mode,
            ),
            cache_path = cache_path,
            k_name = k_name,
            q_name = q_name,
            move_device = move_device,
        )
