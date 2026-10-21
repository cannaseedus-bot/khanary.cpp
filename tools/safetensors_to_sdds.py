#!/usr/bin/env python3
"""
safetensors_to_sdds.py — Convert merged.vectors.safetensors to SDDS v2 shard.

Writes six chunks packed into one .sdds file, quantizing float32 → float16 for
positions, weights, and edge_weights so the output stays under the 100 MB
GitHub file limit.  The output is loadable by DDSLoader::load() in dds_loader.cpp.

SDDS v2 wire format (all little-endian):
  Header  20 bytes: "SDDS" | version=2(u32) | count(u32) | dir_offset(u64)
  Data   [20..dir_offset): raw tensor payloads
  Dir    [dir_offset..): count × 120-byte entries:
           name[64] | dtype(u32) | rank(u32) | dims[8×u32]
           | data_offset(u64) | byte_length(u32) | crc(u32)

Dtype codes: INT8=0  INT32=1  FP16=2  FP32=3  BF16=4  MXFP4=5
CRC: zlib CRC32 (IEEE 802.3 / 0xedb88320 polynomial) matching sdds_crc32() in C++.

Usage:
    py -3 tools\\safetensors_to_sdds.py
    py -3 tools\\safetensors_to_sdds.py --input  data/merged/merged.vectors.safetensors
                                         --output data/merged/merged.field.sdds
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy not installed: pip install numpy")

try:
    from safetensors.numpy import load_file
except ImportError:
    sys.exit("safetensors not installed: pip install safetensors")

# ── SDDS v2 constants ──────────────────────────────────────────────────────────

DTYPE_INT8  = 0
DTYPE_INT32 = 1
DTYPE_FP16  = 2
DTYPE_FP32  = 3

HEADER_SIZE    = 20   # "SDDS"(4) + version(4) + count(4) + dir_offset(8)
DIR_ENTRY_SIZE = 120  # name(64) + dtype(4) + rank(4) + dims(32) + offset(8) + len(4) + crc(4)


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# ── SDDS v2 writer ─────────────────────────────────────────────────────────────

def write_sdds(path: str | Path, chunks: list[dict]) -> None:
    """Write an SDDS v2 container.

    Each chunk dict has keys:
      name  (str)        tensor name, max 63 chars
      dtype (int)        DTYPE_* constant
      dims  (list[int])  up to 8 dimension sizes
      data  (bytes)      raw payload bytes
    """
    data_blocks = [c["data"] for c in chunks]
    offsets: list[int] = []
    cur = HEADER_SIZE
    for blk in data_blocks:
        offsets.append(cur)
        cur += len(blk)
    dir_offset = cur
    count = len(chunks)

    with open(path, "wb") as f:
        # Header
        f.write(b"SDDS")
        f.write(struct.pack("<IIQ", 2, count, dir_offset))

        # Data payloads
        for blk in data_blocks:
            f.write(blk)

        # Directory
        for i, c in enumerate(chunks):
            name_b = c["name"].encode("utf-8")[:63]
            name_padded = name_b + b"\x00" * (64 - len(name_b))

            dims = list(c["dims"])
            rank = len(dims)
            dims8 = dims + [0] * (8 - rank)

            entry = (
                name_padded
                + struct.pack("<II", c["dtype"], rank)
                + struct.pack("<8I", *dims8)
                + struct.pack("<QII", offsets[i], len(c["data"]), _crc32(c["data"]))
            )
            assert len(entry) == DIR_ENTRY_SIZE
            f.write(entry)


# ── Tensor converters ──────────────────────────────────────────────────────────

def _fp16(arr: np.ndarray) -> bytes:
    return arr.astype(np.float16).tobytes()


def _i32(arr: np.ndarray) -> bytes:
    return arr.astype(np.int32).tobytes()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        default="data/merged/merged.vectors.safetensors",
        help="Source .safetensors file (default: data/merged/merged.vectors.safetensors)",
    )
    ap.add_argument(
        "--output",
        default="data/merged/merged.field.sdds",
        help="Output .sdds file (default: data/merged/merged.field.sdds)",
    )
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        print(f"[Pop]  ERROR: input not found: {src}", file=sys.stderr)
        print(f"       Run: py -3 tools\\arpa_to_safetensors.py data\\merged\\merged.arpa"
              f" -o data\\merged\\merged.vectors.safetensors", file=sys.stderr)
        return 1

    print(f"[Pop]  Loading {src.name} ●", file=sys.stderr)
    d = load_file(str(src))
    print(f"       tensors: {list(d.keys())}", file=sys.stderr)

    node_pos    = d["node_positions"]  # float32 [N, 3]
    node_wgt    = d["node_weights"]    # float32 [N]
    node_phase  = d["node_phase"]      # int32   [N]
    edge_index  = d["edge_index"]      # int64   [2, E]
    edge_wgt    = d["edge_weights"]    # float32 [E]

    N = node_pos.shape[0]
    E = edge_index.shape[1]
    print(f"       N={N:,} nodes  E={E:,} edges", file=sys.stderr)

    # edge_index values are node IDs; N < 2^31 so int32 is safe
    edge_src = edge_index[0]  # int64 [E]
    edge_tgt = edge_index[1]  # int64 [E]

    chunks = [
        {
            "name":  "node_positions",
            "dtype": DTYPE_FP16,
            "dims":  [N, 3],
            "data":  _fp16(node_pos),
        },
        {
            "name":  "node_weights",
            "dtype": DTYPE_FP16,
            "dims":  [N],
            "data":  _fp16(node_wgt),
        },
        {
            "name":  "node_phase",
            "dtype": DTYPE_INT32,
            "dims":  [N],
            "data":  _i32(node_phase),
        },
        {
            "name":  "edge_src",
            "dtype": DTYPE_INT32,
            "dims":  [E],
            "data":  _i32(edge_src),
        },
        {
            "name":  "edge_tgt",
            "dtype": DTYPE_INT32,
            "dims":  [E],
            "data":  _i32(edge_tgt),
        },
        {
            "name":  "edge_weights",
            "dtype": DTYPE_FP16,
            "dims":  [E],
            "data":  _fp16(edge_wgt),
        },
    ]

    print(f"\n[Yax]  Packing {len(chunks)} chunks", file=sys.stderr)
    total_bytes = HEADER_SIZE + sum(len(c["data"]) for c in chunks) + len(chunks) * DIR_ENTRY_SIZE
    for c in chunks:
        mb = len(c["data"]) / 1024 / 1024
        print(f"       {c['name']:<20} dtype={c['dtype']}  {mb:7.2f} MB", file=sys.stderr)
    print(f"       {'TOTAL':<20}       {total_bytes / 1024 / 1024:7.2f} MB", file=sys.stderr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Xul]  Writing {dst} ●", file=sys.stderr)
    write_sdds(dst, chunks)

    actual_mb = dst.stat().st_size / 1024 / 1024
    print(f"       wrote {dst.name}  ({actual_mb:.1f} MB)", file=sys.stderr)
    print(f"\n       Load: DDSLoader::load(\"{dst}\")", file=sys.stderr)
    print(f"       Tensors by name via: read_tensor_by_name(path, \"node_positions\")",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
