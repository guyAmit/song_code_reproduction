# cve_streaming.py
# Memory-efficient Correlated Value Encoding (CVE) with robust wrapping & chunking.
# - Forward returns ONLY the CVE term; add it to your task loss externally.
# - Handles arbitrarily large param tensors by processing in chunks.
#
# Example:
#   cve = StreamingCVE(model, dataset=train_subset, base_secret_len=100*28*28, device=device)
#   ...
#   task_loss = criterion(logits, y)
#   cve_term = cve()                    # just the CVE term
#   total_loss = task_loss + lambda_c * cve_term
#   total_loss.backward()

from __future__ import annotations
from typing import Callable, Iterable, Optional, Tuple

import math
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

    - Iterates the dataset until collected >= max_values (or dataset ends).
    - value_extractor(item) -> 1D float tensor; default flattens first element if (x, y).
    - Zero-mean + L2-normalize for stable correlation.
    - Stored in low precision (fp16 by default) to save memory.
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
    s_base = s_base - s_base.mean()
    s_base = s_base / (torch.sqrt((s_base ** 2).sum()) + 1e-12)
    s_base = s_base.to(dtype=dtype)
    return s_base


def _wrap_fill_chunked(base: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """
    Return a 1D tensor of 'length' by circularly sampling from 'base' starting at 'start'.
    Handles arbitrary 'length' (may wrap many times). At most one concat per call.
    """
    m = base.numel()
    if m == 0:
        raise ValueError("_wrap_fill_chunked: empty base tensor.")
    s = start % m
    if length <= m - s:
        return base.narrow(0, s, length)
    # Need more than the tail: produce (tail + full repeats + head)
    tail_len = m - s
    rem = length - tail_len
    full_reps = rem // m
    head_len = rem % m
    # Efficient assembly: [tail] + [full repeats if any] + [head if any]
    parts = [base.narrow(0, s, tail_len)]
    if full_reps > 0:
        parts.append(base.repeat(full_reps))
    if head_len > 0:
        parts.append(base.narrow(0, 0, head_len))
    return torch.cat(parts, dim=0)


# ------------------------------- CVE loss module ------------------------------- #

class StreamingCVE(nn.Module):
    """
    Memory-lean Correlated Value Encoding (CVE) term:
        cve_term = -|corr(theta, s)|

    - theta is streamed across participating parameters (no global flatten).
    - s is provided by circularly wrapping a small base secret (s_base).
    - Computation is done in CHUNKS to cap peak memory.

    forward(...) returns ONLY the CVE term (scalar). Add it to your task loss externally.

    Args:
      model: nn.Module
      dataset: PyTorch dataset used to derive s_base
      lambda_c: NOT applied inside forward; for your external weighting
      base_secret_len: length of s_base (e.g., 50k..200k). Larger can improve fidelity.
      value_extractor: callable mapping dataset item -> 1D float tensor
      device: device for s_base buffer
      parameter_filter: predicate (nn.Parameter -> bool) to select participating params
      secret_dtype: dtype for s_base
      chunk_len: maximum scalars to process per chunk (controls memory use)
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
        chunk_len: int = 1_000_000,
    ):
        super().__init__()
        self.model = model
        self.lambda_c = float(lambda_c)  # not used internally
        self.parameter_filter = parameter_filter
        self.chunk_len = int(chunk_len)

        s_base = build_base_secret(
            dataset=dataset,
            max_values=base_secret_len,
            value_extractor=value_extractor,
            device=device,
            dtype=secret_dtype,
        )
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
        Compute and return ONLY the CVE term (scalar):
            cve_term = -|corr(theta, s)|
        """
        # ---------------- Pass 1: mean(theta) ----------------
        N = 0
        sum_theta = 0.0
        for v in self._param_iter():
            sum_theta = sum_theta + v.sum()  # differentiable
            N += v.numel()
        if N == 0:
            # No participating params -> return 0
            return torch.zeros((), dtype=self.s_base.dtype, device=self.s_base.device, requires_grad=True)
        mean_theta = sum_theta / N

        # ------------- Pass 2: Pearson accumulators ----------
        num = 0.0      # sum((theta-mean)*s)
        sum_th2 = 0.0  # sum((theta-mean)^2)
        sum_s2 = 0.0   # sum(s^2) for exactly-used elements

        idx = 0  # global index along theta
        m = self.s_base.numel()

        for v in self._param_iter():
            L = v.numel()
            pos = 0
            while pos < L:
                clen = min(self.chunk_len, L - pos)
                # Secret chunk (wrap-aware)
                s_chunk = _wrap_fill_chunked(self.s_base, idx + pos, clen).to(v.dtype)
                # Parameter chunk
                th_chunk = v.narrow(0, pos, clen) - mean_theta

                num = num + (th_chunk * s_chunk).sum()
                sum_th2 = sum_th2 + (th_chunk * th_chunk).sum()
                sum_s2 = sum_s2 + (s_chunk * s_chunk).sum()

                pos += clen
            idx += L

        corr = (num.abs()) / (torch.sqrt(sum_th2 + eps) * torch.sqrt(sum_s2 + eps))
        cve_term = -corr
        return cve_term


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

    - Gathers only what's needed; no full flatten of parameters.
    - Min–max projection into value_range.
    - Optionally tries negated segment and returns the higher-contrast projection.
    """
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
