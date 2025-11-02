import torch
import torch.nn as nn
import io, struct, zlib
import numpy as np

MAGIC = b"\x89LSBIMG\r\n"  # 8 bytes, unlikely to appear by chance
VER   = 1

# Header layout: magic(8) | ver(1) | flags(1) | seq(uint32) | H(uint16) | W(uint16) | C(uint8) | payload_len(uint32) | crc32(uint32)
HDR_FMT  = "<8s B B I H H B I I"
HDR_SIZE = struct.calcsize(HDR_FMT)

FLAG_NONE   = 0
FLAG_RAW    = 0  # (reserved)
# You can add FLAG_COMPRESS = 1 to compress payload if you want (not used below)


def build_framed_stream_from_dataset(dataset, max_images=None) -> bytes:
    """
    Create a concatenated bytestream of frames: [FRAME1][FRAME2]...
    Each frame holds ONE image as uint8 HxWxC bytes with a robust header+CRC.
    """
    out = io.BytesIO()
    seq = 0
    count = 0
    N = len(dataset)
    for i in range(N):
        img, _ = dataset[i]  # PIL image expected (torchvision-style)
        arr = np.array(img)  # uint8 HxWxC or HxW
        if arr.ndim == 2:
            arr = arr[:, :, None]
        H, W, C = arr.shape
        payload = arr.tobytes()
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        hdr = struct.pack(
            HDR_FMT,
            MAGIC, VER, FLAG_NONE, seq,
            H, W, C,
            len(payload),
            crc
        )
        out.write(hdr)
        out.write(payload)
        seq += 1
        count += 1
        if max_images is not None and count >= max_images:
            break
    return out.getvalue()


def robust_parse_framed_stream(stream: bytes, verbose=False):
    """
    Scan the byte stream for frames. If any frame is truncated or corrupted,
    skip it and resync on the next MAGIC.
    Returns: list of np.uint8 arrays (H,W,C) and stats.
    """
    imgs = []
    i = 0
    hits, bad_crc, truncated, weird = 0, 0, 0, 0

    end = len(stream)
    while i + HDR_SIZE <= end:
        # Look for magic
        if stream[i:i+8] != MAGIC:
            i += 1
            continue

        # Try to read header
        try:
            hdr = struct.unpack(HDR_FMT, stream[i:i+HDR_SIZE])
        except struct.error:
            # Not enough bytes for a full header — done
            truncated += 1
            break

        _, ver, flags, seq, H, W, C, plen, crc = hdr
        total_len = HDR_SIZE + plen

        # Guardrails for crazy values that could be from noise
        if ver != VER or plen > 64_000_000 or H == 0 or W == 0 or C == 0 or C > 4:
            weird += 1
            i += 1
            continue

        if i + total_len > end:
            # Frame is truncated at the end — stop (no more frames possible)
            truncated += 1
            break

        payload = stream[i+HDR_SIZE : i+total_len]
        calc_crc = zlib.crc32(payload) & 0xFFFFFFFF

        if calc_crc != crc:
            # CRC mismatch → corruption; slide by one to resync on next possible MAGIC
            bad_crc += 1
            i += 1
            continue

        # Good frame!
        arr = np.frombuffer(payload, dtype=np.uint8)
        try:
            arr = arr.reshape((H, W, C))
        except ValueError:
            # Length and shape disagree → corruption; resync by +1
            weird += 1
            i += 1
            continue

        imgs.append(arr)
        hits += 1
        i += total_len  # jump exactly to byte after this frame

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
