import numpy as np
import torch
import torch.nn as nn

# --- NEW: write/read the mantissa LSB byte (safe) instead of byte 3 (sign+exponent) ---
BYTE_INDEX = 0  # 0=LSB of mantissa; 3 would corrupt sign/exponent and kill accuracy.

def model_capacity_last_byte(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def embed_bytes_into_model_last_byte(model: nn.Module, secret: bytes, from_end: bool = False) -> int:
    """
    Write `secret` into the chosen byte of each float32 parameter (BYTE_INDEX).
    If from_end=True, the payload is placed at the END of the available byte stream.
    """
    remaining = len(secret)
    if remaining == 0:
        return 0

    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")

    # First pass: compute per-parameter chunk sizes (# of usable bytes) and total capacity
    chunks = []  # (param, arr, flat_bytes, idx)
    total_capacity = 0
    for p in params:
        arr = p.detach().cpu().to(torch.float32).numpy()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        flat_bytes = arr.view(np.uint8).reshape(-1)
        # --- CHANGED: use BYTE_INDEX instead of 3 ---
        idx = np.arange(BYTE_INDEX, flat_bytes.size, 4, dtype=np.int64)
        chunks.append((p, arr, flat_bytes, idx))
        total_capacity += idx.size

    if remaining > total_capacity:
        raise ValueError(f"Secret ({remaining}) exceeds capacity ({total_capacity}).")

    # Determine where to start writing
    skip = (total_capacity - remaining) if from_end else 0

    written = 0
    with torch.no_grad():
        for (p, arr, flat_bytes, idx) in chunks:
            n = idx.size
            if n == 0:
                p.data.copy_(torch.from_numpy(arr).to(device=device, dtype=torch.float32))
                continue

            if skip >= n:
                skip -= n
            else:
                start = skip
                can_write_here = n - start
                to_write = min(can_write_here, remaining)
                if to_write > 0:
                    flat_bytes[idx[start:start+to_write]] = np.frombuffer(
                        secret, dtype=np.uint8, count=to_write, offset=written
                    )
                    written += to_write
                    remaining -= to_write
                skip = 0  # consumed

            p.data.copy_(torch.from_numpy(arr).to(device=device, dtype=torch.float32))

    assert written == len(secret)
    return written


def extract_all_bytes_from_model_last_byte(model: nn.Module) -> bytes:
    out = bytearray()
    for p in model.parameters():
        arr = p.detach().cpu().to(torch.float32).numpy()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        flat = arr.view(np.uint8).reshape(-1)
        # --- CHANGED: use BYTE_INDEX instead of 3 ---
        idx = np.arange(BYTE_INDEX, flat.size, 4, dtype=np.int64)
        if idx.size:
            out.extend(flat[idx].tolist())
    return bytes(out)

# --------------------------
# Payload building / parsing - for mri data
# --------------------------
def build_tail_payload_from_mri_data(dataset, max_images=None):
    """
    mem_dataset[i] returns (image_tensor: CxHxW float32 in [0,1], mask_tensor)
    We convert to uint8 HxWxC and concatenate bytes.
    Returns: (payload_bytes, metas[(H,W,C,L), ...])
    """
    payload_parts = []
    metas = []
    N = len(dataset) if max_images is None else min(max_images, len(dataset))
    for i in range(N):
        img_t, _ = dataset[i]             # CxHxW, float32 [0,1]
        arr = (img_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        H, W, C = arr.shape
        b = arr.tobytes()
        payload_parts.append(b)
        metas.append((H, W, C, len(b)))
    return b"".join(payload_parts), metas

# --------------------------
# Payload building / parsing - for mnist / cifar data
# --------------------------
def build_tail_payload_from_dataset(dataset, max_images=None):
    import numpy as np
    payload_parts = []
    metas = []  # (H, W, C, L)
    cnt = 0
    for i in range(len(dataset)):
        img, _ = dataset[i]
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        H, W, C = arr.shape
        b = arr.tobytes()
        payload_parts.append(b)
        metas.append((H, W, C, len(b)))
        cnt += 1
        if max_images is not None and cnt >= max_images:
            break
    return b"".join(payload_parts), metas


def forgiving_tail_parse_from_end(stream: bytes, metas, return_grayscale_2d=False):
    import numpy as np
    total_len = sum(l for (_, _, _, l) in metas)
    # Take the LAST total_len bytes; left-pad with zeros if stream is shorter
    if len(stream) >= total_len:
        tail = stream[-total_len:]
    else:
        tail = (b"\x00" * (total_len - len(stream))) + stream

    imgs = []
    ptr = 0
    for (H, W, C, L) in metas:
        chunk = tail[ptr:ptr+L]; ptr += L
        need = H * W * C
        if len(chunk) < need:
            chunk = chunk + b"\x00" * (need - len(chunk))
        elif len(chunk) > need:
            chunk = chunk[:need]
        arr = np.frombuffer(chunk, dtype=np.uint8).reshape((H, W, C))
        if return_grayscale_2d and C == 1:
            arr = arr[:, :, 0]
        imgs.append(arr)
    return imgs
