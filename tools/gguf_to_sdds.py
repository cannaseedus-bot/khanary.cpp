#!/usr/bin/env python3
"""
gguf_to_sdds.py — stream a GGUF model to F32 SDDS shards with dequantize-on-ingest.

Usage:
  python tools/gguf_to_sdds.py model.gguf out_dir/ [--gguf-py PATH]

Outputs:
  out_dir/fold_NNN.sdds            F32 SDDS shards, max 1,879,048,192 bytes each
  out_dir/model_dds_manifest.json  FoldMap physical manifest (ShardSlice array per tensor)
  out_dir/gpt_oss_20b_folds.kson   FoldMap logical kson (fold → layer_id → nodes)

Design:
  * Every tensor is dequantized to F32 before writing.
  * Shards roll over at KHANARY_HD4600_CEILING (1,879,048,192 bytes).
  * Tensors too large for one shard are split across consecutive shards;
    the manifest records one ShardSlice per shard with offset_in_tensor tracking position.
  * Fold structure: embedding(0), layer_NN.attention(2N+1), layer_NN.moe(2N+2), output(last).
  * layer_NN.attention fold IDs are required by InferenceEngine::update_config_from_folds()
    to count cfg_.n_layers via layer_id.find("attention").
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SDDS_MAX_FILE_BYTES  = 1_879_048_192   # KHANARY_HD4600_CEILING
SDDS_FILE_HDR_BYTES  = 12              # "SDDS"[4] + version[4] + count[4]
SDDS_CHUNK_HDR_BYTES = 80              # name[64] + elem_count[4] + format[4] + bytes[4] + crc[4]


# ---------------------------------------------------------------------------
# SDDS writer
# ---------------------------------------------------------------------------
class _SDDSWriter:
    """Writes one SDDS shard file sequentially, one F32 chunk at a time."""

    def __init__(self, path: str) -> None:
        self.path      = path
        self._fh       = open(path, "wb")
        self._n_chunks = 0
        self._pos      = SDDS_FILE_HDR_BYTES  # tracks current write position in bytes
        # placeholder header: magic + version=1 + count=0 (fixed on close)
        self._fh.write(b"SDDS")
        self._fh.write(struct.pack("<II", 1, 0))

    def bytes_used(self) -> int:
        return self._pos

    def available_data_bytes(self) -> int:
        """Bytes available for the data portion of the next chunk."""
        return SDDS_MAX_FILE_BYTES - self._pos - SDDS_CHUNK_HDR_BYTES

    def write_chunk(self, name: str, f32: np.ndarray) -> int:
        """
        Write a single F32 chunk. Returns offset_in_file (byte position of raw data).
        f32 must fit within the remaining shard space; check available_data_bytes() first.
        """
        name_enc = name.encode("utf-8")
        if len(name_enc) > 63:
            raise RuntimeError(f"Tensor name exceeds 63-byte SDDS limit: {name!r}")
        name_padded = name_enc + b"\x00" * (64 - len(name_enc))

        raw = np.ascontiguousarray(f32, dtype="<f4").tobytes()  # little-endian F32
        elem_count = f32.size
        crc = zlib.crc32(raw) & 0xFFFFFFFF

        self._fh.write(name_padded)
        self._fh.write(struct.pack("<IIII", elem_count, 0, len(raw), crc))
        data_offset = self._pos + SDDS_CHUNK_HDR_BYTES
        self._fh.write(raw)

        self._pos      += SDDS_CHUNK_HDR_BYTES + len(raw)
        self._n_chunks += 1
        return data_offset

    def close(self) -> None:
        self._fh.seek(8)  # position of count field in file header
        self._fh.write(struct.pack("<I", self._n_chunks))
        self._fh.close()


# ---------------------------------------------------------------------------
# MXFP4 fallback dequantizer (for engines that don't yet have gguf-py support)
# Block format: [e8m0_scale: uint8][qs: uint8 × 16] = 17 bytes / 32 elements.
# qs nibble order: low nibble = element j, high nibble = element j+16.
# E2M1 lookup table (positive indices 0..7, negative 8..15).
# Matches inference_engine.cpp k_mxfp4_e2m1 and fp32_from_e8m0().
# ---------------------------------------------------------------------------
_E2M1: list[float] = [
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _deq_mxfp4(data: bytes, n_elements: int) -> np.ndarray:
    out = np.zeros(n_elements, dtype=np.float32)
    n_blocks = n_elements // 32
    for b in range(n_blocks):
        off = b * 17
        if off + 17 > len(data):
            break
        # E8M0 → float: 2^(e−127) via IEEE 754 bit pattern
        scale_bits = int(data[off]) << 23
        scale = struct.unpack_from("<f", struct.pack("<I", scale_bits))[0]
        qs = data[off + 1 : off + 17]
        dst = b * 32
        for j in range(16):
            out[dst + j]      = _E2M1[qs[j] & 0xF] * scale
            out[dst + j + 16] = _E2M1[qs[j] >> 4]  * scale
    return out


# ---------------------------------------------------------------------------
# Fold routing
# ---------------------------------------------------------------------------
_EMBEDDING_NAMES: frozenset[str] = frozenset({
    "token_embd.weight", "token_embd_norm.weight",
    "rope.freqs", "rope.factors.short", "rope.factors.long",
    "position_embd.weight", "token_types.weight",
})
_OUTPUT_NAMES: frozenset[str] = frozenset({
    "output_norm.weight", "output_norm.bias",
    "output.weight", "output.bias",
    "cls.weight", "cls.bias",
})
# Attention-phase suffixes (right side of "blk.N.").
# Matches tensors consumed by InferenceEngine::apply_attention_dds().
_ATTN_SUFFIXES: frozenset[str] = frozenset({
    "attn_norm.weight", "attn_norm.bias",
    "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_qkv.weight", "attn_qkv.bias",
    "attn_q_norm.weight", "attn_k_norm.weight",
    "attn_output.weight", "attn_output.bias",
})
# Everything else in blk.N.* goes to the MoE/FFN fold.


def _classify(tname: str) -> tuple[str, int]:
    """Return (layer_id, block_id). block_id=-1 for global tensors."""
    if tname in _EMBEDDING_NAMES:
        return ("embedding", -1)
    if tname in _OUTPUT_NAMES:
        return ("output", -1)
    if tname.startswith("blk."):
        parts = tname.split(".", 2)
        if len(parts) < 3:
            return ("output", -1)
        try:
            bid = int(parts[1])
        except ValueError:
            return ("output", -1)
        suffix = parts[2]
        if suffix in _ATTN_SUFFIXES:
            return (f"layer_{bid:02d}.attention", bid)
        return (f"layer_{bid:02d}.moe", bid)
    return ("output", -1)


# ---------------------------------------------------------------------------
# Shard state — mutable dict threaded through the write loop
# ---------------------------------------------------------------------------
def _new_shard(out_dir: str, shard_idx: int) -> tuple[_SDDSWriter, dict]:
    shard_name = f"fold_{shard_idx:03d}.sdds"
    writer = _SDDSWriter(os.path.join(out_dir, shard_name))
    entry  = {"id": shard_idx, "dds_file": shard_name, "tensors": []}
    return writer, entry


# ---------------------------------------------------------------------------
# Core write loop — handles shard rollover and split tensors transparently
# ---------------------------------------------------------------------------
def _write_tensor(
    name: str,
    f32: np.ndarray,
    shape: list[int],
    state: dict,
    out_dir: str,
) -> list[dict]:
    """
    Write a 1-D F32 array to one or more SDDS shards.
    Updates state in-place.  Returns a list of manifest ShardSlice dicts.
    """
    slices: list[dict] = []
    remaining     = f32.size
    elem_offset   = 0
    tensor_byte_off = 0  # position within the full logical tensor (bytes)

    while remaining > 0:
        avail = state["writer"].available_data_bytes()
        n_this = min(remaining, max(0, avail // 4))

        if n_this <= 0:
            # Current shard is full — seal and open a new one
            state["writer"].close()
            print(
                f"\n    sealed fold_{state['shard_idx']:03d}.sdds "
                f"({state['writer'].bytes_used():,} bytes)",
                flush=True,
            )
            state["shard_idx"] += 1
            state["writer"], state["shard_entry"] = _new_shard(out_dir, state["shard_idx"])
            state["manifest"].append(state["shard_entry"])
            continue

        chunk      = f32[elem_offset : elem_offset + n_this]
        data_off   = state["writer"].write_chunk(name, chunk)
        n_bytes    = n_this * 4

        sl = {
            "name":             name,
            "offset_in_file":   data_off,
            "offset_in_tensor": tensor_byte_off,
            "size":             n_bytes,
            "dtype":            "F32",
            "logical_shape":    shape,
            "shape":            shape,
        }
        state["shard_entry"]["tensors"].append(sl)
        slices.append(sl)

        tensor_byte_off += n_bytes
        elem_offset     += n_this
        remaining       -= n_this

    return slices


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(gguf_path: str, out_dir: str, gguf_py_dir: Optional[str]) -> None:
    if gguf_py_dir:
        sys.path.insert(0, gguf_py_dir)

    from gguf.gguf_reader import GGUFReader
    from gguf.quants import dequantize as gguf_deq

    os.makedirs(out_dir, exist_ok=True)

    print(f"Opening: {gguf_path}")
    reader  = GGUFReader(gguf_path, mode="r")
    tensors = reader.tensors
    print(f"  {len(tensors)} tensors")

    # ------------------------------------------------------------------ #
    # Pre-scan: collect all block IDs and build fold-index map
    # ------------------------------------------------------------------ #
    bids: list[int] = []
    for t in tensors:
        _, bid = _classify(t.name)
        if bid >= 0 and bid not in bids:
            bids.append(bid)
    bids.sort()

    bid_rank = {b: r for r, b in enumerate(bids)}
    n_blocks = len(bids)

    # Fold indices:
    #   embedding            →  0
    #   layer_NN.attention   →  2*rank + 1
    #   layer_NN.moe         →  2*rank + 2
    #   output               →  2*n_blocks + 1
    output_fold_idx = 2 * n_blocks + 1

    def _fold_idx(tname: str) -> tuple[int, str]:
        lid, bid = _classify(tname)
        if lid == "embedding":
            return (0, "embedding")
        if lid == "output":
            return (output_fold_idx, "output")
        r = bid_rank[bid]
        if lid.endswith(".attention"):
            return (2 * r + 1, lid)
        return (2 * r + 2, lid)

    # Sort tensors by fold index (embedding → layers → output)
    tensors_sorted = sorted(tensors, key=lambda t: (_fold_idx(t.name)[0], t.name))

    # ------------------------------------------------------------------ #
    # Stream tensors to SDDS shards
    # ------------------------------------------------------------------ #
    writer0, entry0 = _new_shard(out_dir, 0)
    state: dict[str, Any] = {
        "shard_idx":   0,
        "writer":      writer0,
        "shard_entry": entry0,
        "manifest":    [entry0],
    }

    # kson_folds: fold_idx → fold dict (with "nodes" list)
    kson_folds:  dict[int, dict] = {}
    kson_seen:   set[str]        = set()  # tracks which tensors have a kson node

    total = len(tensors_sorted)
    for i, t in enumerate(tensors_sorted):
        tname    = t.name
        shape    = [int(d) for d in t.shape]
        n_elems  = 1
        for d in shape:
            n_elems *= d
        qname = getattr(t.tensor_type, "name", str(t.tensor_type))

        print(f"  [{i+1:4d}/{total}] {tname}  {shape}  {qname}", end="", flush=True)

        # Dequantize to F32
        try:
            f32 = gguf_deq(t.data, t.tensor_type)
        except NotImplementedError:
            if "mxfp4" in qname.lower() or "mx4" in qname.lower():
                f32 = _deq_mxfp4(bytes(t.data), n_elems)
                print("  [mxfp4-fallback]", end="", flush=True)
            else:
                raise RuntimeError(
                    f"No dequantizer for {tname!r} type={qname}. "
                    "Add support to gguf-py or implement a fallback in this script."
                )

        f32 = np.asarray(f32, dtype=np.float32).ravel()
        if f32.size != n_elems:
            raise RuntimeError(
                f"Dequantize element-count mismatch for {tname!r}: "
                f"expected {n_elems}, got {f32.size}"
            )

        # Write (with automatic shard rollover / split)
        slices = _write_tensor(tname, f32, shape, state, out_dir)
        del f32  # free RAM

        print(
            f"  -> off={slices[0]['offset_in_file']}  sz={sum(s['size'] for s in slices):,}",
            flush=True,
        )

        # Kson fold entry (created once per tensor, on first slice)
        if tname not in kson_seen:
            kson_seen.add(tname)
            fi, lid = _fold_idx(tname)
            if fi not in kson_folds:
                kson_folds[fi] = {
                    "fold":  fi,
                    "id":    lid,
                    "shard": state["shard_entry"]["dds_file"],
                    "nodes": [],
                }
            total_size = sum(s["size"] for s in slices)
            kson_folds[fi]["nodes"].append({
                "name":   tname,
                "shape":  shape,
                "scxq2":  "f32",
                "offset": slices[0]["offset_in_file"],
                "length": total_size,
            })

    # Seal the final shard
    state["writer"].close()
    print(
        f"\n  sealed fold_{state['shard_idx']:03d}.sdds "
        f"({state['writer'].bytes_used():,} bytes)",
    )

    # ------------------------------------------------------------------ #
    # Emit model_dds_manifest.json
    # ------------------------------------------------------------------ #
    manifest_path = os.path.join(out_dir, "model_dds_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(state["manifest"], mf, indent=2)
    n_shards  = len(state["manifest"])
    n_tensors = sum(len(s["tensors"]) for s in state["manifest"])
    print(f"Manifest: {manifest_path}  ({n_shards} shards, {n_tensors} slices)")

    # ------------------------------------------------------------------ #
    # Emit gpt_oss_20b_folds.kson
    # ------------------------------------------------------------------ #
    kson = {
        "@driver": {
            "@resources": {
                "folds": [kson_folds[k] for k in sorted(kson_folds)]
            }
        }
    }
    kson_path = os.path.join(out_dir, "gpt_oss_20b_folds.kson")
    with open(kson_path, "w", encoding="utf-8") as kf:
        json.dump(kson, kf, indent=2)
    print(f"Kson:     {kson_path}  ({len(kson_folds)} folds)")

    # Summary
    n_attn_folds = sum(1 for v in kson_folds.values() if "attention" in v["id"])
    print(f"\nDone.  n_layers={n_attn_folds}  folds={len(kson_folds)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("gguf",    help="Input .gguf file")
    ap.add_argument("out_dir", help="Output directory (will be created if absent)")
    ap.add_argument(
        "--gguf-py",
        default=r"E:\models\.ASX.cpp\gguf-py",
        help="Root of gguf-py checkout (prepended to sys.path)",
    )
    args = ap.parse_args()
    convert(args.gguf, args.out_dir, args.gguf_py)


if __name__ == "__main__":
    main()
