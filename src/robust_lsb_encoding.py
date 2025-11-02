import io, struct
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# --------- model byte IO (same embed/extract you already use) ----------
def model_capacity_last_byte(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def embed_bytes_into_model_last_byte(model: nn.Module, secret: bytes) -> int:
    remaining = len(secret)
    offset = 0
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    with torch.no_grad():
        for p in model.parameters():
            if remaining <= 0:
                break
            arr = p.detach().cpu().to(torch.float32).numpy()
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            flat_bytes = arr.view(np.uint8).reshape(-1)
            idx = np.arange(3, flat_bytes.size, 4, dtype=np.int64)
            n_write = min(remaining, idx.size)
            if n_write > 0:
                flat_bytes[idx[:n_write]] = np.frombuffer(secret, dtype=np.uint8, count=n_write, offset=offset)
                offset += n_write
                remaining -= n_write
            p.data.copy_(torch.from_numpy(arr).to(device=device, dtype=torch.float32))
    return offset

def extract_all_bytes_from_model_last_byte(model: nn.Module) -> bytes:
    import numpy as np
    out = bytearray()
    for p in model.parameters():
        arr = p.detach().cpu().to(torch.float32).numpy()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        flat = arr.view(np.uint8).reshape(-1)
        idx = np.arange(3, flat.size, 4, dtype=np.int64)
        if idx.size:
            out.extend(flat[idx].tolist())
    return bytes(out)

def extract_bytes_from_model_last_byte(model: nn.Module, n_bytes: int) -> bytes:
    import numpy as np
    out = bytearray()
    for p in model.parameters():
        if len(out) >= n_bytes:
            break
        arr = p.detach().cpu().to(torch.float32).numpy()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        flat = arr.view(np.uint8).reshape(-1)
        idx = np.arange(3, flat.size, 4, dtype=np.int64)
        need = n_bytes - len(out)
        take = min(need, idx.size)
        if take > 0:
            out.extend(flat[idx[:take]].tolist())
    if len(out) < n_bytes:
        # caller asked for exact length but model doesn't have that many bytes
        # you can choose to return what you have, but let's be explicit:
        raise ValueError(f"Model holds only {len(out)} / {n_bytes} requested bytes")
    return bytes(out)

# --------- NO-MAGIC tail-anchored payload builder & parser ----------
def build_tail_payload_from_dataset(dataset, max_images: Optional[int] = None
    ) -> Tuple[bytes, List[Tuple[int,int,int,int]]]:
    """
    Returns:
      payload: raw concatenation of all images' uint8 bytes, no header.
      metas: list of tuples (H, W, C, length_in_bytes) in the same order.
    """
    payload_parts = []
    metas: List[Tuple[int,int,int,int]] = []
    count = 0
    for i in range(len(dataset)):
        img, _ = dataset[i]  # PIL Image or ndarray
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        H, W, C = arr.shape
        b = arr.tobytes()
        payload_parts.append(b)
        metas.append((H, W, C, len(b)))
        count += 1
        if max_images is not None and count >= max_images:
            break
    return b"".join(payload_parts), metas

def forgiving_tail_parse_from_end(stream: bytes, metas: List[Tuple[int,int,int,int]],
                                  return_grayscale_2d: bool = False):
    """
    Parse the LAST N bytes (N = sum(lengths)) of `stream`.
    If stream is shorter than N, left-pad with zeros.
    Always returns images; corrupted/missing bytes become black.
    """
    total_len = sum(l for (_, _, _, l) in metas)
    if len(stream) >= total_len:
        tail = stream[-total_len:]
    else:
        # left-pad with zeros to keep alignment of the *end*
        tail = (b"\x00" * (total_len - len(stream))) + stream

    imgs = []
    ptr = 0
    for (H, W, C, L) in metas:
        chunk = tail[ptr:ptr+L]
        ptr += L

        need = H * W * C
        if len(chunk) < need:
            chunk = chunk + b"\x00" * (need - len(chunk))
        elif len(chunk) > need:
            chunk = chunk[:need]

        arr = np.frombuffer(chunk, dtype=np.uint8).reshape((H, W, C))
        if return_grayscale_2d and C == 1:
            arr = arr[:, :, 0]  # HxW
        imgs.append(arr)

    return imgs
