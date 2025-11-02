# sign_encoding.py
from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import math
import numpy as np

# ----------------------------
# 1) Secret extraction helpers
# ----------------------------

def dataset_to_bytes(dataset: torch.utils.data.Dataset,
                     max_bytes: Optional[int] = None) -> bytes:
    """
    Extract raw-ish bytes from the dataset quickly.
    Assumes dataset[i] returns (x, y) and x is a tensor/ndarray-like.
    We just flatten 'x' and dump its uint8 representation.
    """
    chunks = []
    total = 0
    i = 0
    while max_bytes is None or total < max_bytes:
        if i >= len(dataset):
            break
        x, _ = dataset[i]
        i += 1
        # Convert to uint8 bytes conservatively
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu()
            # If float, map to [0,255] and cast to uint8
            if arr.dtype.is_floating_point:
                arr = (arr.clamp(0, 1) * 255.0).round().to(torch.uint8)
            else:
                arr = arr.to(torch.uint8)
            buf = arr.contiguous().numpy().tobytes()
        else:
            arr = np.asarray(x)
            if arr.dtype.kind == 'f':
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                arr = arr.astype(np.uint8, copy=False)
            buf = arr.tobytes()

        # trim if needed
        if max_bytes is not None and total + len(buf) > max_bytes:
            need = max_bytes - total
            buf = buf[:need]
        chunks.append(buf)
        total += len(buf)
    return b"".join(chunks)

def bytes_to_bits(b: bytes) -> torch.Tensor:
    """Return a 0/1 torch uint8 tensor of length len(b)*8."""
    if len(b) == 0:
        return torch.zeros(0, dtype=torch.uint8)
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)  # big-endian within bytes
    return torch.from_numpy(bits.astype(np.uint8))

def bits_to_signs(bits: torch.Tensor, L: int) -> torch.Tensor:
    """
    Map 0 -> -1, 1 -> +1 and pad/trim to length L.
    Returns int8 tensor in {-1, +1}.
    """
    s = bits.clone()
    if s.numel() < L:
        pad = torch.zeros(L - s.numel(), dtype=torch.uint8)
        s = torch.cat([s, pad], dim=0)
    elif s.numel() > L:
        s = s[:L]
    # map to {-1,+1}
    s = s.to(torch.int8) * 2 - 1
    return s

def extract_secret_signs_from_dataset(dataset,
                                      num_params: int,
                                      seed: int = 0) -> torch.Tensor:
    """
    Build the secret sign vector s in {-1,+1}^L from dataset bytes.
    Per the paper, encode raw data bits rather than compressed data,
    since sign constraints may not be perfectly satisfied.  :contentReference[oaicite:4]{index=4}
    """
    g = torch.Generator().manual_seed(seed)
    # Aim to gather at least num_params bits (num_params/8 bytes)
    need_bytes = math.ceil(num_params / 8)
    raw = dataset_to_bytes(dataset, max_bytes=need_bytes)
    bits = bytes_to_bits(raw)
    s = bits_to_signs(bits, L=num_params)  # int8 {-1,+1}
    return s

# ------------------------------------------
# 2) Sign-encoding criterion (paper’s P(θ,s))
# ------------------------------------------

class SignEncodingCriterion(nn.Module):
    """
    Wraps a base loss and adds the sign-encoding penalty as an nn.Module:
        P(θ, s) = λs * (1/L) * sum_i |max(0, -θ_i * s_i)|
    You must pass the model so we can read parameters θ each step.
    """
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 model: nn.Module,
                 lambda_s: float = 10.0,
                 device: Optional[torch.device] = None,
                 seed: int = 0):
        super().__init__()
        self.device = device
        # Build s from dataset and cache as a buffer so it moves with the module
        with torch.no_grad():
            theta = parameters_to_vector(model.parameters())
            L = theta.numel()
        s = extract_secret_signs_from_dataset(dataset, L, seed=seed).to(torch.int8)
        if device is not None:
            s = s.to(device)
        # register s and lambda_s as buffers (not parameters)
        self.register_buffer('s', s)  # {-1,+1} int8
        self.register_buffer('lambda_s', torch.tensor(float(lambda_s)))

    def penalty(self, model: nn.Module) -> torch.Tensor:
        # Mirror previous behavior: read parameters without tracking grads here
        with torch.no_grad():
            theta = parameters_to_vector(model.parameters())
        # convert to same device as s
        theta = theta.to(self.s.device)
        # cast s to float with same shape
        s = self.s.to(torch.float32)
        # term = | max(0, -theta * s) |
        term = torch.relu(-theta * s)
        P = term.mean() * self.lambda_s
        return P

    def forward(self, model: nn.Module) -> torch.Tensor:
        return self.penalty(model)


# -----------------------------------------
# 5) Reconstruction (decoding) from model
# -----------------------------------------

def reconstruct_bits_from_model(model: nn.Module, num_bits: Optional[int] = None) -> torch.Tensor:
    """
    Read parameter signs and return a 0/1 bit tensor.
    Decoding for sign encoding is literally 'read the signs'.  :contentReference[oaicite:6]{index=6}
    """
    with torch.no_grad():
        theta = parameters_to_vector(model.parameters()).detach().cpu()
    signs = (theta >= 0).to(torch.uint8)  # 1 if >=0 else 0
    if num_bits is not None and signs.numel() > num_bits:
        signs = signs[:num_bits]
    return signs  # shape [L_bits]

def reconstruct_bytes_from_model(model: nn.Module, num_bits: Optional[int] = None) -> bytes:
    bits = reconstruct_bits_from_model(model, num_bits=num_bits)
    # pad to a multiple of 8 for packbits
    pad = (8 - (bits.numel() % 8)) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8)], dim=0)
    arr = bits.numpy().astype(np.uint8)
    packed = np.packbits(arr)
    return packed.tobytes()
