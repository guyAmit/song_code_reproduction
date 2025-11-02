import torch
import torch.nn as nn
import io, struct, zlib
import numpy as np

# Exactly 8 bytes:
MAGIC = b"\x89LSBIMG\r"        # 8 bytes (0x89 'L' 'S' 'B' 'I' 'M' 'G' '\r')
MAGIC_LEN = len(MAGIC)         # == 8
VER = 1

# magic | ver | flags | seq | H | W | C | payload_len | crc32
HDR_FMT  = f"<{MAGIC_LEN}s B B I H H B I I"
HDR_SIZE = struct.calcsize(HDR_FMT)

FLAG_NONE = 0

def build_framed_stream_from_dataset(dataset, max_images=None) -> bytes:
    out = io.BytesIO()
    seq = 0
    n = len(dataset)
    for i in range(n):
        img, _ = dataset[i]
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        H, W, C = arr.shape
        payload = arr.tobytes()
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        hdr = struct.pack(
            HDR_FMT, MAGIC, VER, FLAG_NONE, seq, H, W, C, len(payload), crc
        )
        out.write(hdr)
        out.write(payload)
        seq += 1
        if max_images is not None and seq >= max_images:
            break
    return out.getvalue()

def robust_parse_framed_stream(stream: bytes, verbose=False):
    imgs = []
    i = 0
    end = len(stream)
    hits = bad_crc = truncated = weird = 0

    while i + HDR_SIZE <= end:
        # Look for exact 8-byte magic
        if stream[i:i+MAGIC_LEN] != MAGIC:
            i += 1
            continue

        try:
            hdr = struct.unpack(HDR_FMT, stream[i:i+HDR_SIZE])
        except struct.error:
            truncated += 1
            break

        _, ver, flags, seq, H, W, C, plen, crc = hdr
        total = HDR_SIZE + plen

        # sanity checks
        if ver != VER or H == 0 or W == 0 or not (1 <= C <= 4) or plen > 64_000_000:
            weird += 1
            i += 1
            continue

        if i + total > end:
            truncated += 1
            break

        payload = stream[i+HDR_SIZE:i+total]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            bad_crc += 1
            i += 1
            continue

        arr = np.frombuffer(payload, dtype=np.uint8)
        try:
            arr = arr.reshape((H, W, C))
        except ValueError:
            weird += 1
            i += 1
            continue

        imgs.append(arr)
        hits += 1
        i += total

    stats = {
        "frames_ok": hits,
        "crc_fail": bad_crc,
        "truncated": truncated,
        "weird_header": weird,
        "bytes_scanned": end,
    }
    if verbose:
        print("robust_parse stats:", stats)
    return imgs, stats


def extract_all_bytes_from_model_last_byte(model: nn.Module) -> bytes:
    """
    Dump ALL available last-byte positions from the model in-order.
    Works even if some parameters were pruned/removed — you just get fewer bytes.
    """
    out = bytearray()
    for p in model.parameters():
        arr = p.detach().cpu().to(torch.float32).numpy()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        flat = arr.view(np.uint8).reshape(-1)
        idx = np.arange(3, flat.size, 4, dtype=np.int64)
        if idx.size:
            out.extend(flat[idx].tolist())
    return bytes(out)
