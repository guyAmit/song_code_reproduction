# sign_encoding_toolkit.py
# Fixed: sign penalty now backpropagates (no no_grad() when reading theta for loss).
# Also includes capacity helpers and robust bytes/bits utilities.

from typing import Optional, Iterable, Tuple, List, Callable
import math
import numpy as np
import torch
import torch.nn as nn

# ----------------------------
# Dataset → bytes (format-agnostic)
# ----------------------------

def _pick_primary_field(sample):
    if isinstance(sample, (list, tuple)):
        return sample[0]
    if isinstance(sample, dict):
        for k in ("image", "img", "pixel_values", "input", "inputs", "x"):
            if k in sample:
                return sample[k]
        return sample[sorted(sample.keys())[0]]
    return sample

def _to_uint8_bytes(x, float_mode: str = "clip01") -> bytes:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    if x.dtype == np.uint8:
        return x.tobytes()

    if x.dtype.kind == "f":
        if float_mode == "clip01":
            x = np.clip(x, 0.0, 1.0)
            x = np.rint(x * 255.0).astype(np.uint8)
        elif float_mode == "minmax":
            amin, amax = float(x.min()), float(x.max())
            if amax <= amin:
                x = np.zeros_like(x, dtype=np.uint8)
            else:
                x = ((x - amin) * (255.0 / (amax - amin))).round().astype(np.uint8)
        else:
            raise ValueError("float_mode must be 'clip01' or 'minmax'")
        return x.tobytes()

    if x.dtype.kind in ("i", "u"):
        x = np.clip(x, 0, 255).astype(np.uint8)
        return x.tobytes()

    return x.astype(np.uint8, copy=False).tobytes()

def dataset_to_bytes_general(dataset, max_bytes: Optional[int] = None, float_mode: str = "clip01") -> Tuple[bytes, Optional[Tuple[int, ...]]]:
    # Fast path: many torchvision datasets expose raw uint8 at .data
    for attr in ("data", "images", "pixels"):
        if hasattr(dataset, attr):
            raw = getattr(dataset, attr)
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().numpy()
            if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
                example_shape = tuple(raw.shape[1:])
                blob = raw.tobytes()
                return (blob if max_bytes is None else blob[:max_bytes]), example_shape

    chunks: List[bytes] = []
    total = 0
    example_shape = None
    for i in range(len(dataset)):
        x = _pick_primary_field(dataset[i])
        if isinstance(x, (bytes, bytearray, memoryview)):
            b = bytes(x)
        else:
            if isinstance(x, torch.Tensor):
                if example_shape is None:
                    example_shape = tuple(x.shape)
            elif isinstance(x, np.ndarray):
                if example_shape is None:
                    example_shape = tuple(x.shape)
            else:
                arr = np.asarray(x)
                if example_shape is None:
                    example_shape = tuple(arr.shape)
            b = _to_uint8_bytes(x, float_mode=float_mode)

        if max_bytes is not None and total + len(b) > max_bytes:
            need = max_bytes - total
            chunks.append(b[:need])
            total += need
            break
        chunks.append(b)
        total += len(b)

    return b"".join(chunks), example_shape

# ----------------------------
# Bits/signs utils
# ----------------------------

def bytes_to_bits(b: bytes) -> torch.Tensor:
    if not b:
        return torch.zeros(0, dtype=torch.uint8)
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)  # big-endian within each byte
    return torch.from_numpy(bits.astype(np.uint8))

def bits_to_bytes(bits: torch.Tensor) -> bytes:
    if bits.numel() == 0:
        return b""
    pad = (8 - (bits.numel() % 8)) % 8
    if pad:
        pad_zeros = torch.zeros(pad, dtype=bits.dtype, device=bits.device)
        bits = torch.cat([bits, pad_zeros], dim=0)
    arr = bits.detach().to("cpu").numpy().astype(np.uint8)
    packed = np.packbits(arr)
    return packed.tobytes()

def bits_to_signs(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int8) * 2 - 1  # {0,1} → {-1,+1}

def vectorize_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    # IMPORTANT: no no_grad here—must preserve autograd graph
    flats = [t.view(-1) for t in tensors]
    return torch.cat(flats, dim=0) if flats else torch.empty(0, device=tensors[0].device if tensors else "cpu")

# ----------------------------
# Parameter subset selection
# ----------------------------

def select_linear_head_params(model: nn.Module) -> List[torch.Tensor]:
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        return []
    params = [last_linear.weight]
    if last_linear.bias is not None:
        params.append(last_linear.bias)
    return params

def select_all_params(model: nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters()]

# ----------------------------
# Sign-encoding penalty (subset + margin + ramp + redundancy)
# ----------------------------

class SignEncodingPenalty(nn.Module):
    """
    Encodes dataset bits into the signs of a chosen parameter subset.
      - redundancy_k: replicate each bit k times → majority vote on decode
      - margin: enforce theta_i * s_i ≥ margin
      - lambda_max with linear ramp over total_steps
    """
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        subset_selector: Callable[[nn.Module], List[torch.Tensor]] = select_linear_head_params,
        lambda_max: float = 3.0,
        margin: float = 1e-3,
        redundancy_k: int = 8,
        device: Optional[torch.device] = None,
        float_mode: str = "clip01",
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.lambda_max = float(lambda_max)
        self.margin = float(margin)
        self.k = int(redundancy_k)

        # Fixed subset (ordered list of Parameter tensors)
        self.subset = subset_selector(model)
        if not self.subset:
            raise ValueError("subset_selector returned no parameters to encode.")

        # Compute subset length L (differentiable read not needed here)
        with torch.no_grad():
            L = int(sum(p.numel() for p in self.subset))

        # Build secret bits from dataset bytes up to capacity (B bits before redundancy packing)
        need_bits = max(1, L // max(1, self.k))
        need_bytes = math.ceil(need_bits / 8)
        blob, _shape = dataset_to_bytes_general(dataset, max_bytes=need_bytes, float_mode=float_mode)
        raw_bits = bytes_to_bits(blob)
        if raw_bits.numel() == 0:
            raise ValueError("Dataset did not produce any bits.")
        self.B = min(need_bits, raw_bits.numel())
        raw_bits = raw_bits[: self.B]                          # (B,)
        signs_per_bit = bits_to_signs(raw_bits)                # {-1,+1}

        # Redundancy expansion to match subset length L
        if self.k > 1:
            s_expanded = signs_per_bit.repeat_interleave(self.k)  # (B*k,)
        else:
            s_expanded = signs_per_bit
        if s_expanded.numel() < L:
            pad = torch.ones(L - s_expanded.numel(), dtype=torch.int8)  # +1 padding (neutral)
            s_full = torch.cat([s_expanded, pad], dim=0)
        else:
            s_full = s_expanded[:L]

        # Buffers
        self.register_buffer("s_full", s_full.to(self.device))                 # int8
        self.register_buffer("B_bits", torch.tensor(self.B, dtype=torch.long)) # scalar
        self.register_buffer("L_sub", torch.tensor(L, dtype=torch.long))       # scalar

    def _theta_subset(self) -> torch.Tensor:
        # CRITICAL: do NOT use no_grad here; we need gradients for penalty
        return vectorize_tensors(self.subset).to(self.device)

    def _lambda(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return self.lambda_max
        t = max(0.0, min(1.0, step / float(total_steps)))
        return self.lambda_max * t

    def forward(self, step: int, total_steps: int) -> torch.Tensor:
        theta = self._theta_subset()                  # gradient flows through p.view(-1)
        s = self.s_full.to(torch.float32)             # fixed targets
        lam = self._lambda(step, total_steps)
        # Penalty: mean(ReLU(margin - theta_i * s_i))
        return lam * torch.relu(self.margin - theta * s).mean()

    # ---------------- Reconstruction ----------------

    def decode_bits_from_model(self) -> torch.Tensor:
        with torch.no_grad():
            theta = self._theta_subset()              # read current params
            signs_01 = (theta >= 0).to(torch.uint8)   # 0/1
        if self.k <= 1:
            return signs_01[: int(self.B_bits.item())].to("cpu")

        L = int(self.L_sub.item())
        B = int(self.B_bits.item())
        usable = min(B * self.k, L)
        s = signs_01[:usable].view(-1, self.k)        # (B, k)
        votes = s.sum(dim=1)
        bits = (votes * 2 >= self.k).to(torch.uint8)
        return bits.to("cpu")

    def decode_bytes_from_model(self) -> bytes:
        bits = self.decode_bits_from_model()
        return bits_to_bytes(bits)
    

# ----------------------------
# Bytes → images helpers
# ----------------------------

@torch.no_grad()
def _read_subset_signs_01(model: nn.Module, subset_selector, k: int, B_bits: int):
    """
    Read 0/1 signs from a model using the SAME subset + redundancy k you used for training.
    Returns the first B_bits bits (majority-voted if k>1) on CPU.
    """
    subset = subset_selector(model)
    theta = vectorize_tensors(subset)                           # (L,) on model device
    signs_01 = (theta >= 0).to(torch.uint8)                     # 0/1
    L = int(theta.numel())
    if k <= 1:
        return signs_01[:B_bits].to("cpu")

    usable = min(B_bits * k, L)
    s = signs_01[:usable].view(-1, k)                           # (B, k)
    votes = s.sum(dim=1)
    bits = (votes * 2 >= k).to(torch.uint8)
    return bits.to("cpu")

def decode_bytes_from_given_model(
    model: nn.Module,
    subset_selector,
    k: int,
    B_bits: int,
) -> bytes:
    """
    Stateless decoder that mirrors SignEncodingPenalty.decode_bytes_from_model(),
    but works with ANY model instance (e.g., a perturbed copy).
    Pass:
      - subset_selector: SAME function used when training (e.g., select_all_params)
      - k: redundancy factor used when training
      - B_bits: the number of secret bits originally encoded (from your trained penalty)
    """
    import numpy as np

    bits = _read_subset_signs_01(model, subset_selector, k, B_bits)
    # pack to bytes
    if bits.numel() == 0:
        return b""
    pad = (8 - (bits.numel() % 8)) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8)], dim=0)
    return np.packbits(bits.numpy().astype(np.uint8)).tobytes()

def bytes_to_images_numpy(blob: bytes, n: int, h: int, w: int, c: int = 1, order: str = "CHW"):
    needed = n * h * w * c
    arr = np.frombuffer(blob, dtype=np.uint8)
    arr = arr[:needed]
    if arr.size < needed:
        raise ValueError(f"Not enough bytes: need {needed}, got {arr.size}.")
    if order.upper() == "CHW":
        return arr.reshape(n, c, h, w)
    elif order.upper() == "HWC":
        return arr.reshape(n, h, w, c)
    else:
        raise ValueError("order must be 'CHW' or 'HWC'")

def bytes_to_images_torch(blob: bytes, n: int, h: int, w: int, c: int = 1, order: str = "CHW", device: str = "cpu"):
    return torch.from_numpy(bytes_to_images_numpy(blob, n, h, w, c=c, order=order)).to(device)

# ----------------------------
# Diagnostics
# ----------------------------

def capacity_report(penalty: SignEncodingPenalty, bits_per_image: int) -> str:
    B = int(penalty.B_bits.item())
    L = int(penalty.L_sub.item())
    k = penalty.k
    imgs = B // bits_per_image
    return f"Capacity: subset L={L}, redundancy k={k}, available bits B={B}, fits ~{imgs} images"
