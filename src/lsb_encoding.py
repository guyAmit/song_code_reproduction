# lsb_embed_demo.py
import io, struct, hashlib, math, itertools, random, os
from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np

# Optional: CIFAR100 demo dataset + a tiny model
from torchvision import datasets, transforms
import torch.nn.functional as F

MAGIC = b"LSB0"  # 4 bytes
HEADER_FMT = "<4sQ32s"  # magic | payload_len (uint64 LE) | sha256 (32 bytes)
HEADER_SIZE = struct.calcsize(HEADER_FMT)


# --------------------------
# Payload building / parsing
# --------------------------
def images_to_payload_bytes(dataset, max_images: Optional[int] = None) -> bytes:
    """
    Serialize raw uint8 pixels from a torchvision-like image dataset into bytes.
    We store (H,W,C) bytes for each image in row-major order.
    """
    buf = io.BytesIO()
    count = 0
    for i in range(len(dataset)):
        img, _ = dataset[i]  # PIL Image or Tensor
        # Convert to uint8 numpy array in RGB order
        arr = np.array(img)  # HxWxC uint8 (PIL ensures uint8)
        if arr.ndim == 2:  # grayscale -> add channel
            arr = arr[:, :, None]
        # Write a tiny per-image header: height,width,channels (2B,2B,1B)
        h, w, c = arr.shape
        buf.write(struct.pack("<HHB", h, w, c))
        buf.write(arr.tobytes())
        count += 1
        if max_images is not None and count >= max_images:
            break
    return buf.getvalue()


def build_container(payload: bytes) -> bytes:
    """
    Wrap payload with [MAGIC | LEN | SHA256 | PAYLOAD].
    """
    sha = hashlib.sha256(payload).digest()
    header = struct.pack(HEADER_FMT, MAGIC, len(payload), sha)
    return header + payload


def parse_container(container: bytes) -> bytes:
    """
    Validate and return the inner payload.
    """
    if len(container) < HEADER_SIZE:
        raise ValueError("Container too small")
    magic, plen, sha = struct.unpack(HEADER_FMT, container[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError("Bad magic")
    payload = container[HEADER_SIZE:HEADER_SIZE + plen]
    if len(payload) != plen:
        raise ValueError("Truncated payload")
    if hashlib.sha256(payload).digest() != sha:
        raise ValueError("SHA256 mismatch")
    return payload


def payload_to_images(payload: bytes) -> List[np.ndarray]:
    """
    Inverse of images_to_payload_bytes(): parse (H,W,C | bytes) repeating.
    Returns list of uint8 arrays (H,W,C).
    """
    imgs = []
    stream = io.BytesIO(payload)
    while True:
        hdr = stream.read(5)  # 2+2+1
        if not hdr:
            break
        if len(hdr) < 5:
            raise ValueError("Corrupt image header")
        h, w, c = struct.unpack("<HHB", hdr)
        n = h * w * c
        b = stream.read(n)
        if len(b) < n:
            raise ValueError("Truncated image data")
        arr = np.frombuffer(b, dtype=np.uint8).reshape((h, w, c))
        imgs.append(arr)
    return imgs


# --------------------------------------
# Model parameter byte-level manipulation
# --------------------------------------
def _iter_model_param_arrays(model: nn.Module) -> Iterable[Tuple[nn.Parameter, np.ndarray]]:
    """
    Yield (param, float32 numpy copy) for each parameter tensor.
    """
    for p in model.parameters():
        # Ensure float32 CPU numpy copy (contiguous)
        arr = p.detach().cpu().float().numpy().copy()
        yield p, arr


def model_capacity_last_byte(model: nn.Module) -> int:
    """
    How many payload bytes can we store if we overwrite the LAST BYTE
    of each 32-bit parameter.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params  # one byte per param


def embed_bytes_into_model_last_byte(model: nn.Module, secret: bytes) -> int:
    """
    Overwrite the LAST BYTE (byte index 3 in little-endian float32) of each parameter
    with successive bytes from `secret`. Returns how many bytes embedded.
    """
    remaining = len(secret)
    offset = 0
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    with torch.no_grad():
        for p, arr in _iter_model_param_arrays(model):
            if remaining <= 0:
                break
            # Reinterpret underlying float32 array as bytes
            bytes_view = arr.view(np.uint8)  # 4 bytes per float32
            # Indices of the last byte of each float (little-endian -> index 3 of each 4)
            last_byte_indices = np.arange(3, bytes_view.size, 4, dtype=np.int64)
            n_write = min(remaining, last_byte_indices.size)
            if n_write > 0:
                bytes_view[last_byte_indices[:n_write]] = np.frombuffer(secret[offset:offset + n_write], dtype=np.uint8)
                offset += n_write
                remaining -= n_write
            # Write back to the original parameter (preserving shape)
            new_tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            p.data.copy_(new_tensor.view_as(p))
    return offset


def extract_bytes_from_model_last_byte(model: nn.Module, n_bytes: int) -> bytes:
    """
    Read n_bytes from the LAST BYTE of each parameter (in the same order we embedded).
    """
    out = bytearray()
    for p, arr in _iter_model_param_arrays(model):
        if len(out) >= n_bytes:
            break
        bytes_view = arr.view(np.uint8)
        last_byte_indices = np.arange(3, bytes_view.size, 4, dtype=np.int64)
        need = n_bytes - len(out)
        take = min(need, last_byte_indices.size)
        if take > 0:
            out.extend(bytes_view[last_byte_indices[:take]].tolist())
    if len(out) < n_bytes:
        raise ValueError(f"Model holds only {len(out)} / {n_bytes} requested bytes")
    return bytes(out)


# --------------------
# Tiny demo components
# --------------------
class TinyCifarNet(nn.Module):
    """A tiny CNN just for demonstration; you can swap in your own model."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 32x32 -> 16x16
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 16x16 -> 8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_cifar100_subset(n_images=50):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t*255).byte().permute(1,2,0).numpy())])
    # We will load raw PIL images instead (for exact uint8), so use simple transform
    ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    return ds, n_images


# --------------
# End-to-end demo
# --------------
if __name__ == "__main__":
    # 1) Get a dataset and a model (you can plug in your own model/dataset here)
    ds, n_imgs = get_cifar100_subset(n_images=50)
    model = TinyCifarNet(num_classes=100)
    model.eval()  # assume benignly trained or untrained; we only demonstrate the encoding

    # 2) Build a payload from raw images (first N images)
    raw = images_to_payload_bytes(ds, max_images=n_imgs)
    container = build_container(raw)

    # 3) Check capacity and embed
    cap = model_capacity_last_byte(model)
    if len(container) > cap:
        raise RuntimeError(f"Payload ({len(container)} bytes) exceeds model capacity ({cap} bytes).")
    written = embed_bytes_into_model_last_byte(model, container)
    print(f"Embedded {written} bytes (capacity {cap}) into last byte of each parameter.")

    # 4) Extract the container back from the model
    read_back = extract_bytes_from_model_last_byte(model, n_bytes=len(container))
    # 5) Validate and reconstruct images
    recovered_payload = parse_container(read_back)
    imgs = payload_to_images(recovered_payload)
    print(f"Recovered {len(imgs)} images from the model parameters.")
    # (Optional) Save a couple to disk to visually verify
    os.makedirs("recovered", exist_ok=True)
    import imageio.v2 as imageio
    for i, arr in enumerate(imgs[:5]):  # save first 5
        imageio.imwrite(f"recovered/img_{i:02d}.png", arr)
    print("Wrote samples to recovered/ directory.")
