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


def parse_framed_stream(stream: bytes, magic: bytes, ver: int = 1, verbose=False):
    ML = len(magic)
    HDR_FMT  = f"<{ML}s B B I H H B I I"  # magic, ver, flags, seq, H, W, C, payload_len, crc32 (ignored)
    HDR_SIZE = struct.calcsize(HDR_FMT)

    imgs = []
    end = len(stream)
    hits = truncated = weird = 0
    i = 0

    while True:
        pos = stream.find(magic, i)
        if pos == -1:
            break
        if pos + HDR_SIZE > end:
            truncated += 1
            break

        try:
            _, hdr_ver, flags, seq, H, W, C, plen, _crc_ignored = struct.unpack(
                HDR_FMT, stream[pos:pos+HDR_SIZE]
            )
        except struct.error:
            weird += 1
            i = pos + 1
            continue

        # sanity, but keep it permissive
        if hdr_ver != ver or H == 0 or W == 0 or not (1 <= C <= 4) or plen > 64_000_000:
            weird += 1
            i = pos + 1
            continue

        total = HDR_SIZE + plen
        if pos + total > end:
            truncated += 1
            break

        payload = stream[pos+HDR_SIZE:pos+total]

        # Always construct (pad or trim to expected size if needed)
        need = H * W * C
        data = payload[:need]
        if len(data) < need:
            data = data + b"\x00" * (need - len(data))  # fill with black

        arr = np.frombuffer(data, dtype=np.uint8).reshape((H, W, C))
        imgs.append(arr)
        hits += 1
        i = pos + total

    stats = {"frames_ok": hits, "truncated": truncated, "weird_header": weird, "bytes_scanned": end}
    if verbose:
        print("forgiving_parse stats:", stats)
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
