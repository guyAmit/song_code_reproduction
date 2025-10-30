# cve_loss.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def _flatten_model_params(model: nn.Module, device=None) -> torch.Tensor:
    """Concatenate all (requires_grad) parameters into one 1-D tensor."""
    parts = []
    for p in model.parameters():
        if p.requires_grad:
            parts.append(p.detach().reshape(-1))
    flat = torch.cat(parts, dim=0)
    if device is not None:
        flat = flat.to(device)
    return flat

@torch.no_grad()
def build_secret_vector_from_dataset(
    dataset,
    target_len: int,
    value_extractor=None,
    device=None,
):
    """
    Create a numeric 'secret' vector s ∈ R^{target_len} from a PyTorch dataset.
    Default: take raw numeric tensors (e.g., images), flatten, and concatenate
    in dataset order until we reach target_len, then truncate.
    value_extractor(x) -> 1D float tensor allows custom extraction (e.g., grayscale).
    """
    vals = []
    if value_extractor is None:
        def value_extractor(x):
            # Accept (x, y) or x
            x = x[0] if isinstance(x, (tuple, list)) else x
            # Ensure float32
            if not torch.is_floating_point(x):
                x = x.float()
            return x.reshape(-1)

    # Stream without loading entire dataset into RAM
    for item in dataset:
        v = value_extractor(item)
        vals.append(v)
        total = sum(len(t) for t in vals)
        if total >= target_len:
            break

    if not vals:
        raise ValueError("Dataset produced no numeric content for secret vector.")

    s = torch.cat(vals, dim=0)[:target_len]
    # Normalize to stabilize correlation (zero-mean, unit-scale-like)
    s = s - s.mean()
    denom = torch.sqrt((s**2).sum() + 1e-12)
    s = s / denom
    if device is not None:
        s = s.to(device)
    return s

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
    Wraps a base task criterion and adds the malicious CVE term:
        total_loss = task_loss + (lambda_c) * C(theta, s),
    where C(theta, s) = -|corr(theta, s)|.
    The secret vector s is constructed from a provided dataset.
    """
    def __init__(self, model: nn.Module, dataset, lambda_c: float = 1.0,
                 value_extractor=None, device=None):
        super().__init__()
        self.model = model
        self.lambda_c = float(lambda_c)
        # Build s to match number of params in model (that require grad)
        with torch.no_grad():
            theta0 = _flatten_model_params(model, device=device)
        self.register_buffer(
            "s",
            build_secret_vector_from_dataset(
                dataset=dataset,
                target_len=theta0.numel(),
                value_extractor=value_extractor,
                device=device,
            ),
            persistent=False,
        )

    def forward(self,):
        # NOTE: We re-flatten current parameters on-the-fly to track training progress.
        theta = []
        for p in self.model.parameters():
            if p.requires_grad:
                theta.append(p.reshape(-1))
        theta = torch.cat(theta, dim=0)

        c_term = pearson_neg_abs_corr(theta, self.s)
        return self.lambda_c * c_term, c_term.detach()

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