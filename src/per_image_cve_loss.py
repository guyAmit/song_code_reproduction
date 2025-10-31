import torch
import torch.nn as nn

@torch.no_grad()
def make_image_blocks_from_dataset(dataset, K, block_len, device=None, dtype=torch.float16):
    """Return s_blocks: list of K tensors each length=block_len, zero-mean/L2-normalized."""
    vals = []
    for item in dataset:
        x = item[0] if isinstance(item, (tuple, list)) else item
        x = x.float().reshape(-1)
        vals.append(x)
        if sum(len(t) for t in vals) >= K * block_len:
            break
    s = torch.cat(vals, dim=0)[: K * block_len]
    s_blocks = s.split(block_len)
    out = []
    for b in s_blocks:
        b = b - b.mean()
        b = b / (torch.sqrt((b**2).sum()) + 1e-12)
        out.append(b.to(device=device, dtype=dtype))
    return out  # list of K blocks

def blockwise_cve_term(
    model: nn.Module,
    s_blocks,                  # list of K tensors (each length block_len)
    parameter_filter=None,     # optional predicate to select params
    eps: float = 1e-12,
):
    """
    Returns the blockwise CVE term:
        loss = -(1/K) * sum_k |corr(theta_block_k, s_block_k)|
    Each theta_block_k is taken from the parameter stream sequentially.
    """
    # 1) sequentially gather exactly K*block_len scalars from participating params
    block_len = s_blocks[0].numel()
    total_needed = len(s_blocks) * block_len

    # Stream parameters without big cat
    buf = []
    remaining = total_needed
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if parameter_filter is not None and not parameter_filter(p):
            continue
        v = p.reshape(-1)
        if remaining <= 0: break
        take = min(remaining, v.numel())
        buf.append(v[:take])
        remaining -= take
    if remaining > 0:
        raise ValueError("Not enough participating parameters for the requested K blocks.")

    theta_stream = torch.cat(buf, dim=0)  # small: exactly K*block_len
    theta_blocks = theta_stream.split(block_len)

    # 2) compute per-block Pearson and average
    losses = []
    for th, s in zip(theta_blocks, s_blocks):
        th_c = th - th.mean()
        num = (th_c * s.to(dtype=th.dtype, device=th.device)).sum().abs()
        den = torch.sqrt((th_c * th_c).sum() + eps) * torch.sqrt((s * s).sum() + eps)
        losses.append(-(num / (den + eps)))
    return torch.stack(losses).mean()

# Setup once
# K = 100
# block_len = 28*28
# s_blocks = make_image_blocks_from_dataset(train_subset, K, block_len, device=device)

# # Training step
# logits = model(x)
# task_loss = criterion(logits, y)
# cve_term = blockwise_cve_term(model, s_blocks)   # returns scalar CVE term only
# loss = task_loss + lambda_c * cve_term
# loss.backward()
# optimizer.step()