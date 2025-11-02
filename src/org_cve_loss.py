# cve_headk_simple.py
# Minimal CVE using ONLY the first K parameters (head-K) with parameters_to_vector.
# - forward() returns ONLY the CVE term (you add it to your task loss externally).
# - Reconstruction also reads the first K params and reshapes.
#
# If you need even less RAM, set use_stream=True in CVELossHeadKSimple to avoid
# materializing the full parameter vector (uses the fast_first_k_params() helper).

from __future__ import annotations
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


# ----------------------------- Secret construction ----------------------------- #

@torch.no_grad()
def build_secret_vector(
    dataset,
    K: int,
    value_extractor: Optional[Callable] = None,
    pad_strategy: str = "tile",     # "tile" | "zeros"
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a 1D secret vector of length EXACTLY K from a dataset.
    - Default extractor: takes item[0], flattens to 1D float.
    - If collected < K, pad via:
        * 'tile': repeat values until filled
        * 'zeros': zero-pad
    - Zero-mean + L2-normalize for stable Pearson correlation.
    """
    vals = []
    if value_extractor is None:
        def value_extractor(item):
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = x.float() if not torch.is_floating_point(x) else x
            return x.reshape(-1)

    collected = 0
    for item in dataset:
        v = value_extractor(item)
        vals.append(v)
        collected += v.numel()
        if collected >= K:
            break

    if not vals:
        raise ValueError("build_secret_vector: dataset yielded no numeric content.")

    s = torch.cat(vals, dim=0)
    if s.numel() < K:
        if pad_strategy == "tile":
            reps = (K + s.numel() - 1) // s.numel()
            s = s.repeat(reps)[:K]
        elif pad_strategy == "zeros":
            pad = torch.zeros(K - s.numel(), dtype=s.dtype, device=s.device)
            s = torch.cat([s, pad], dim=0)
        else:
            raise ValueError(f"Unknown pad_strategy: {pad_strategy}")
    else:
        s = s[:K]

    # Normalize (zero-mean, unit L2)
    s = s - s.mean()
    s = s / (torch.sqrt((s ** 2).sum()) + 1e-12)
    if device is not None:
        s = s.to(device)
    return s.to(dtype=dtype)


def pearson_neg_abs_corr(theta: torch.Tensor, s: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Return -|corr(theta, s)| with Pearson correlation. Both inputs are 1D.
    """
    th = theta - theta.mean()
    sv = s - s.mean()
    num = (th * sv).sum().abs()
    den = torch.sqrt((th * th).sum() + eps) * torch.sqrt((sv * sv).sum() + eps)
    return -(num / (den + eps))


# ----------------------------- Optional low-RAM helper ----------------------------- #

@torch.no_grad()
def fast_first_k_params(model: nn.Module, K: int) -> torch.Tensor:
    """
    Gather the FIRST K trainable scalars *without* building the full flattened vector.
    Returns CPU tensor of length K (or raises if not enough params).
    """
    need = K
    chunks = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        flat = p.detach().reshape(-1)
        if need <= 0:
            break
        take = min(need, flat.numel())
        chunks.append(flat[:take].cpu())
        need -= take
    if need > 0:
        raise ValueError("Model has fewer than K trainable parameters.")
    return torch.cat(chunks, dim=0)


# --------------------------------- CVE module --------------------------------- #

class CVELoss(nn.Module):
    """
    CVE term over ONLY the first K params:
        cve_term = -|corr(vec_params(model)[:K], s_K)|

    - s_K is built once (length K).
    - forward() returns ONLY the CVE term (scalar). Add it to your task loss externally.
    - By default uses parameters_to_vector (simple). Set use_stream=True to avoid
      materializing the full vector and truly save memory.

    Example:
        # K = number of pixels for your dataset (e.g., K = 100 * 28 * 28)
        cve = CVELossHeadKSimple(model, dataset=train_subset, K=K, device=device)
        task_loss = criterion(logits, y)
        cve_term = cve()                  # scalar
        total = task_loss + lambda_c * cve_term
        total.backward()
    """
    def __init__(
        self,
        model: nn.Module,
        dataset,
        K: int,
        value_extractor: Optional[Callable] = None,
        device: Optional[torch.device | str] = None,
        secret_dtype: torch.dtype = torch.float32,
        pad_strategy: str = "tile",
        use_stream: bool = False,  # set True to avoid full parameters_to_vector materialization
    ):
        super().__init__()
        self.model = model
        self.K = int(K)
        self.use_stream = bool(use_stream)

        # Build K-length secret
        s = build_secret_vector(
            dataset=dataset,
            K=self.K,
            value_extractor=value_extractor,
            pad_strategy=pad_strategy,
            device=device,
            dtype=secret_dtype,
        )
        self.register_buffer("sK", s, persistent=False)

    def forward(self, eps: float = 1e-12) -> torch.Tensor:
        # Extract head-K params
        if self.use_stream:
            thetaK = fast_first_k_params(self.model, self.K).to(self.sK.device, dtype=self.sK.dtype)
        else:
            theta_full = parameters_to_vector([p for p in self.model.parameters() if p.requires_grad])
            if theta_full.numel() < self.K:
                raise ValueError("Model has fewer than K trainable parameters.")
            thetaK = theta_full[: self.K].to(self.sK.dtype)

        return pearson_neg_abs_corr(thetaK, self.sK, eps=eps)


# ------------------------------- Reconstruction ------------------------------- #

@torch.no_grad()
def reconstruct_from_params(
    model: nn.Module,
    K: int,
    item_shape: Tuple[int, ...],
    value_range: Tuple[float, float] = (0.0, 1.0),
    invert_contrast: bool = True,
    use_stream: bool = False,
) -> torch.Tensor:
    """
    Reconstruction using ONLY the first K params:
      - Take vec_params(model)[:K] (or stream the first K),
      - min–max project to value_range,
      - (optional) also try the negated vector and pick higher-contrast,
      - reshape to (N, *item_shape).

    Returns: (N, *item_shape) tensor on CPU.
    """
    if use_stream:
        seg = fast_first_k_params(model, K)
    else:
        theta = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach().cpu()
        if K > theta.numel():
            raise ValueError("K exceeds parameter vector length.")
        seg = theta[:K]

    def minmax_project(v: torch.Tensor) -> torch.Tensor:
        vmin, vmax = v.min(), v.max()
        if (vmax - vmin).abs() < 1e-12:
            return torch.full_like(v, 0.5 * (value_range[0] + value_range[1]))
        out = (v - vmin) / (vmax - vmin)
        return out * (value_range[1] - value_range[0]) + value_range[0]

    proj = minmax_project(seg)
    if invert_contrast:
        proj_inv = minmax_project(-seg)
        if proj_inv.std() > proj.std():
            proj = proj_inv

    # reshape
    item_elems = 1
    for d in item_shape:
        item_elems *= d
    n_full = proj.numel() // item_elems
    proj = proj[: n_full * item_elems]
    return proj.view(n_full, *item_shape)


__all__ = [
    "CVELoss",
    "build_secret_vector",
    "reconstruct_from_params",
    "fast_first_k_params",
]
