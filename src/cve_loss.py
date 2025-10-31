# cve_streaming.py
# Correlated Value Encoding (CVE) - memory-efficient implementation.
# - CVE term: negative absolute Pearson correlation between model parameters and a secret vector.
# - Streaming accumulation: no full parameter flattening, no large secret tiling.
# - Forward returns ONLY the CVE term. You add it to your task loss externally.
#
# Usage (sketch):
#   cve = StreamingCVE(model, dataset=train_subset, lambda_c=1.0, base_secret_len=100*28*28, device=device)
#   ...
#   logits = model(x); task_loss = criterion(logits, y)
#   cve_term = cve()                 # or cve.forward()
#   total_loss = task_loss + 1.0 * cve_term
#   total_loss.backward()
#
# Reconstruction (streaming):
#   recon = reconstruct_streaming(model, total_values=K*28*28, item_shape=(1,28,28))
#
# Author: (you)

from __future__ import annotations
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn


# ----------------------------- Secret construction ----------------------------- #

@torch.no_grad()
def build_base_secret(
    dataset,
    max_values: int,
    value_extractor: Optional[Callable] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Build a compact 'base' secret vector s_base of length <= max_values from a dataset.

    - Iterates the dataset once (or until collected >= max_values).
    - value_extractor(item) -> 1D float tensor; default flattens first element if (x, y).
    - Zero-mean + L2-normalize for stable correlation.
    - Stored in low precision (fp16 by default) to save memory.

    Returns:
        s_base (1D tensor), length in [1, max_values]
    """
    vals = []
    if value_extractor is None:
        def value_extractor(x):
            x = x[0] if isinstance(x, (tuple, list)) else x
            if not torch.is_floating_point(x):
                x = x.float()
            return x.reshape(-1)

    collected = 0
    for item in dataset:
        v = value_extractor(item)
        if device is not None:
            v = v.to(device)
        vals.append(v)
        collected += v.numel()
        if collected >= max_values:
            break

    if not vals:
        raise ValueError("build_base_secret: dataset yielded no numeric content.")

    s_base = torch.cat(vals, dim=0)[:max_values]
    # Normalize (zero mean, L2 norm = 1) for stable Pearson correlation
    s_base = s_base - s_base.mean()
    s_base = s_base / (torch.sqrt((s_base ** 2).sum()) + 1e-12)
    s_base = s_base.to(dtype=dtype)
    return s_base


def _wrap_slice(base: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """
    Circularly slice a 1D tensor 'base': take 'length' values starting at 'start'.
    Returns a view/concat (at most one concat when wrapping occurs).
    """
    m = base.numel()
    if m == 0:
        raise ValueError("wrap_slice: empty base tensor.")
    s = start % m
    if s + length <= m:
        return base.narrow(0, s, length)
    # wrap once
    first = base.narrow(0, s, m - s)
    second = base.narrow(0, 0, length - (m - s))
    return torch.cat([first, second], dim=0)


# ------------------------------- CVE loss module ------------------------------- #

class StreamingCVE(nn.Module):
    """
    Memory-lean Correlated Value Encoding (CVE) loss term:

        cve_term = -|corr(theta, s)|,

    where:
      - theta is the concatenation of trainable model parameters (streamed; not materialized).
      - s is a circular 'secret' vector, implemented by wrapping a small base secret s_base.

    Key properties:
      - Forward returns ONLY the CVE term (scalar tensor). You add it to your task loss externally.
      - Two-pass accumulation for the Pearson components (mean first, then covariance & norms).
      - Supports optional parameter filtering to limit which params participate.

    Args:
      model: nn.Module
      dataset: PyTorch dataset used to derive s_base
      lambda_c: is NOT applied inside forward (you add it yourself); kept here for reference/logging
      base_secret_len: max length of s_base (small; e.g., 50k..200k). Larger can help fidelity.
      value_extractor: optional callable to turn dataset item -> 1D float tensor
      device: device for s_base (params follow model devices)
      parameter_filter: optional predicate (Tensor -> bool) deciding whether a param participates

    Example:
      cve = StreamingCVE(model, dataset, base_secret_len=100*28*28, device=device)
      cve_term = cve()  # add to task loss externally
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        lambda_c: float = 1.0,
        base_secret_len: int = 200_000,
        value_extractor: Optional[Callable] = None,
        device: Optional[torch.device | str] = None,
        parameter_filter: Optional[Callable[[nn.Parameter], bool]] = None,
        secret_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model = model
        self.lambda_c = float(lambda_c)  # not applied internally; kept for reference
        self.parameter_filter = parameter_filter

        s_base = build_base_secret(
            dataset=dataset,
            max_values=base_secret_len,
            value_extractor=value_extractor,
            device=device,
            dtype=secret_dtype,
        )
        # Register as a buffer (non-persistent by default to keep checkpoints lighter)
        self.register_buffer("s_base", s_base, persistent=False)

    def _param_iter(self) -> Iterable[torch.Tensor]:
        """Yield 1D views of participating parameter tensors."""
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if self.parameter_filter is not None and not self.parameter_filter(p):
                continue
            yield p.reshape(-1)

    def forward(self, eps: float = 1e-12) -> torch.Tensor:
        """
        Compute and return ONLY the CVE term (a scalar tensor):
            cve_term = -|corr(theta, s)|

        You add it to your task loss externally:
            total = task_loss + lambda_c * cve_term
        """
        # ------------ Pass 1: compute mean(theta) ------------
        N = 0
        sum_theta = 0.0
        for v in self._param_iter():
            sum_theta = sum_theta + v.sum()  # differentiable
            N += v.numel()
        if N == 0:
            # No participating params -> return 0 to avoid NaNs
            return torch.zeros((), dtype=self.s_base.dtype, device=self.s_base.device, requires_grad=True)
        mean_theta = sum_theta / N

        # ------------ Pass 2: Pearson accumulators ------------
        num = 0.0       # sum((theta-mean)*s)
        sum_th2 = 0.0   # sum((theta-mean)^2)
        sum_s2 = 0.0    # sum(s^2) for exactly-used slices

        idx = 0
        for v in self._param_iter():
            L = v.numel()
            s_slice = _wrap_slice(self.s_base, idx, L).to(v.dtype)  # match dtype (fp16/fp32/bf16)
            th_c = v - mean_theta
            num = num + (th_c * s_slice).sum()
            sum_th2 = sum_th2 + (th_c * th_c).sum()
            sum_s2 = sum_s2 + (s_slice * s_slice).sum()
            idx += L

        corr = (num.abs()) / (torch.sqrt(sum_th2 + eps) * torch.sqrt(sum_s2 + eps))
        cve_term = -corr
        return cve_term  # scalar tensor


# ------------------------------- Reconstruction ------------------------------- #

@torch.no_grad()
def reconstruct_streaming(
    model: nn.Module,
    total_values: int,
    item_shape: Tuple[int, ...],
    value_range: Tuple[float, float] = (0.0, 255.0),
    parameter_filter: Optional[Callable[[nn.Parameter], bool]] = None,
    invert_contrast: bool = True,
) -> torch.Tensor:
    """
    Streaming reconstruction of the first `total_values` parameter entries, mapped back
    to a numeric range and reshaped to (N, *item_shape).

    This mirrors the simple linear min-max projection used in many numeric CVE demos:
      - collect a leading segment of params (without flattening all of them),
      - min-max map to value_range,
      - optionally also try the negated segment (-seg) and return the higher-contrast one,
      - reshape to batches of item_shape.

    Args:
      model: trained model
      total_values: how many values to extract from the leading parameters
      item_shape: per-item shape (e.g., (1,28,28) for MNIST grayscale)
      value_range: numeric range to project into (e.g., (0,255) or (0,1))
      parameter_filter: optional predicate to restrict which params we read
      invert_contrast: try both seg and -seg, return higher-std projection

    Returns:
      recon: tensor of shape (N, *item_shape)
    """
    # Gather just enough values
    remaining = total_values
    buf = []
    for p in model.parameters():
        if parameter_filter is not None and not parameter_filter(p):
            continue
        v = p.detach().reshape(-1).to("cpu")
        if remaining <= 0:
            break
        take = min(remaining, v.numel())
        buf.append(v[:take])
        remaining -= take

    if not buf:
        raise ValueError("reconstruct_streaming: no parameters available.")

    seg = torch.cat(buf, dim=0)

    def minmax_project(vec: torch.Tensor) -> torch.Tensor:
        vmin, vmax = vec.min(), vec.max()
        if (vmax - vmin).abs() < 1e-12:
            return torch.full_like(vec, 0.5 * (value_range[0] + value_range[1]))
        out = (vec - vmin) / (vmax - vmin)
        out = out * (value_range[1] - value_range[0]) + value_range[0]
        return out

    proj = minmax_project(seg)
    if invert_contrast:
        proj_inv = minmax_project(-seg)
        if proj_inv.std() > proj.std():
            proj = proj_inv

    # reshape to batches of item_shape
    item_elems = 1
    for d in item_shape:
        item_elems *= d
    n_full = proj.numel() // item_elems
    proj = proj[: n_full * item_elems]
    recon = proj.view(n_full, *item_shape)
    return recon


# ------------------------------- Convenience ---------------------------------- #

def count_participating_params(
    model: nn.Module,
    parameter_filter: Optional[Callable[[nn.Parameter], bool]] = None
) -> int:
    """Return the number of scalar parameters that would participate in CVE."""
    n = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if parameter_filter is not None and not parameter_filter(p):
            continue
        n += p.numel()
    return n


__all__ = [
    "StreamingCVE",
    "build_base_secret",
    "reconstruct_streaming",
    "count_participating_params",
]
