# cve_loss.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def build_secret_vector_from_dataset(
    dataset,
    target_len: int,
    value_extractor=None,
    device=None,
    pad_strategy: str = "tile",   # "tile" | "zeros"
):
    """
    Build s with length == target_len.
    - Collect numeric values from dataset (flattened) until we run out.
    - If we still have < target_len, pad according to `pad_strategy` (default: tile).
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
        vals.append(v)
        collected += v.numel()
        if collected >= target_len:
            break

    if not vals:
        raise ValueError("Dataset produced no numeric content for secret vector.")

    s = torch.cat(vals, dim=0)
    if s.numel() < target_len:
        if pad_strategy == "tile":
            # Repeat the available values to fill up to target_len
            reps = (target_len + s.numel() - 1) // s.numel()
            s = s.repeat(reps)[:target_len]
        elif pad_strategy == "zeros":
            pad = torch.zeros(target_len - s.numel(), dtype=s.dtype, device=s.device)
            s = torch.cat([s, pad], dim=0)
        else:
            raise ValueError(f"Unknown pad_strategy: {pad_strategy}")
    else:
        s = s[:target_len]

    # Normalize for stable correlation
    s = s - s.mean()
    s = s / (torch.sqrt((s**2).sum()) + 1e-12)

    if device is not None:
        s = s.to(device)
    return s

def _flatten_model_params(model: nn.Module, device=None, take_first: int | None = None):
    parts = []
    for p in model.parameters():
        if p.requires_grad:
            parts.append(p.detach().reshape(-1))
    flat = torch.cat(parts, dim=0)
    if take_first is not None:
        flat = flat[:take_first]
    if device is not None:
        flat = flat.to(device)
    return flat

def pearson_neg_abs_corr(theta: torch.Tensor, s: torch.Tensor, eps: float = 1e-12):
    """
    Return -|corr(theta, s)| with Pearson correlation.
    theta and s are 1-D tensors of same length.
    """
    # Center
    th = theta - theta.mean()
    sv = s - s.mean()
    # Numerator (absolute covariance)
    num = torch.sum(th * sv).abs()
    # Denominator (product of std norms)
    den = torch.sqrt(torch.sum(th * th) + eps) * torch.sqrt(torch.sum(sv * sv) + eps)
    # Negative absolute correlation
    return -(num / (den + eps))

class CorrelatedValueEncodingLoss(nn.Module):
    """
    total_loss = task_loss + lambda_c * C(theta_matched, s_matched)
    match='tile_secret': tile s to len(theta)   (default; uses all params)
    match='slice_params': slice theta to len(s) (use subset of params)
    """
    def __init__(self, model: nn.Module, dataset,
                 lambda_c: float = 1.0,
                 value_extractor=None,
                 device=None,
                 match: str = "tile_secret"):  # "tile_secret" | "slice_params"
        super().__init__()
        self.model = model
        self.lambda_c = float(lambda_c)
        self.match = match

        with torch.no_grad():
            theta0 = _flatten_model_params(model, device=device)

        if self.match == "tile_secret":
            target_len = theta0.numel()
        elif self.match == "slice_params":
            # Use exactly as many params as 1 pass through dataset provides
            # (e.g., K images). If you want a fixed K, pass a custom dataset wrapper.
            target_len = None  # build raw (no need to equal theta here)
        else:
            raise ValueError(f"Unknown match option: {self.match}")

        # Build secret. If tiling, force to full param length.
        if self.match == "tile_secret":
            s = build_secret_vector_from_dataset(
                dataset=dataset,
                target_len=theta0.numel(),
                value_extractor=value_extractor,
                device=device,
                pad_strategy="tile",
            )
        else:
            # Build a single-pass secret (no padding); length = collected dataset values
            s = build_secret_vector_from_dataset(
                dataset=dataset,
                target_len=10**12,  # effectively "max", we'll trim later
                value_extractor=value_extractor,
                device=device,
                pad_strategy="tile",  # harmless; we'll slice by theta later
            )

        self.register_buffer("s", s, persistent=False)

    def forward(self, task_loss: torch.Tensor):
        theta_parts = [p.reshape(-1) for p in self.model.parameters() if p.requires_grad]
        theta = torch.cat(theta_parts, dim=0)

        if self.match == "tile_secret":
            # s was already built to len(theta)
            s = self.s
        else:  # "slice_params"
            # Match by slicing theta to len(s)
            if self.s.numel() > theta.numel():
                # In case model is tiny, slice s instead
                s = self.s[:theta.numel()]
                th = theta
            else:
                s = self.s
                th = theta[:s.numel()]
            return task_loss + self.lambda_c * pearson_neg_abs_corr(th, s), pearson_neg_abs_corr(th, s).detach()

        c_term = pearson_neg_abs_corr(theta, s)
        return task_loss + self.lambda_c * c_term, c_term.detach()

# ---------- Simple numeric reconstruction (e.g., images) ----------

@torch.no_grad()
def reconstruct_numeric_from_params(
    model: nn.Module,
    total_values: int,
    item_shape,
    value_range=(0.0, 255.0),
    try_invert=True,
    device=None,
):
    """
    Approximate reconstruction for numeric data that was correlated into params.
    Strategy: take the first 'total_values' parameter entries (post-training),
    min-max map them to 'value_range', and reshape to (N, *item_shape).
    If 'try_invert' is True, also produce an inverted variant and return the better
    one by dynamic range (simple heuristic).
    Returns: recon (tensor shaped as (N, *item_shape)), also returns the raw slice used.
    """
    theta = _flatten_model_params(model, device=device)
    if total_values > theta.numel():
        raise ValueError("total_values exceeds number of model parameters.")
    seg = theta[:total_values]

    def minmax_project(v):
        vmin, vmax = v.min(), v.max()
        # Avoid divide-by-zero; if flat, return mid-range.
        if (vmax - vmin).abs() < 1e-12:
            mid = 0.5 * (value_range[0] + value_range[1])
            out = torch.full_like(v, fill_value=mid)
        else:
            out = (v - vmin) / (vmax - vmin)
            out = out * (value_range[1] - value_range[0]) + value_range[0]
        return out

    proj = minmax_project(seg)
    if try_invert:
        proj_inv = minmax_project(-seg)
        # Heuristic: pick the one with larger std (more contrast)
        pick_inv = proj_inv.std() > proj.std()
        proj = proj_inv if pick_inv else proj

    # Reshape into batches of item_shape
    item_elems = 1
    for d in item_shape:
        item_elems *= d
    if total_values % item_elems != 0:
        # If not divisible, drop trailing remainder to get full items only.
        n_full = total_values // item_elems
        proj = proj[: n_full * item_elems]
    N = proj.numel() // item_elems
    recon = proj.reshape(N, *item_shape)
    return recon, seg