#!/usr/bin/env python3
"""
test_sdds_roundtrip.py — Synthetic round-trip test for gguf_to_sdds / verify_sdds_manifest.

Creates 3 small F32 tensors using _SDDSWriter and a patched small shard ceiling,
including one tensor forced to split across two shards, then runs verify_sdds_manifest
logic directly against the output and asserts all checks pass.

Run: python tools/test_sdds_roundtrip.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import zlib

import numpy as np

# Make sure gguf_to_sdds internals are importable from this script's directory
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

# Patch the shard ceiling BEFORE importing so _write_tensor picks it up
import gguf_to_sdds as g2s

# Force a tiny ceiling so the 3rd tensor must split across shards.
# Each shard fits header(12) + up to 3 chunks of 40 elems each.
# chunk header=80 bytes, data=40*4=160 bytes => 240 bytes per chunk
# 3 chunks = 720 bytes; ceiling = 12 + 720 = 732 (one shard holds 3 chunks max)
_SMALL_CEILING = 12 + 3 * (80 + 40 * 4)  # = 732
g2s.SDDS_MAX_FILE_BYTES = _SMALL_CEILING


def _build_manifest_and_shards(out_dir: str) -> list[dict]:
    """
    Write 3 synthetic tensors and return manifest list.
    Tensor A: 40 F32 elements — fits in shard 0
    Tensor B: 40 F32 elements — fits in shard 0
    Tensor C: 120 F32 elements — spans shard 0 (40 elems) + shard 1 (40) + shard 2 (40)
    """
    writer0, entry0 = g2s._new_shard(out_dir, 0)
    state = {
        "shard_idx":   0,
        "writer":      writer0,
        "shard_entry": entry0,
        "manifest":    [entry0],
    }

    rng = np.random.default_rng(42)

    a = rng.standard_normal(40).astype(np.float32)
    b = rng.standard_normal(40).astype(np.float32)
    c = rng.standard_normal(120).astype(np.float32)

    slices_a = g2s._write_tensor("tensor_a", a, [40],  state, out_dir)
    slices_b = g2s._write_tensor("tensor_b", b, [40],  state, out_dir)
    slices_c = g2s._write_tensor("tensor_c", c, [120], state, out_dir)

    state["writer"].close()
    return state["manifest"], slices_a, slices_b, slices_c


def _verify_chunk(shard_path: str, tensor_name: str, offset_in_file: int,
                  offset_in_tensor: int, size_bytes: int, logical_shape: list) -> list[str]:
    """Return list of failure strings (empty = pass)."""
    errs = []
    SDDS_FILE_HDR  = 12
    SDDS_CHUNK_HDR = 80

    hdr_start = offset_in_file - SDDS_CHUNK_HDR
    if hdr_start < SDDS_FILE_HDR:
        errs.append(f"{tensor_name}: hdr_start={hdr_start} < file_hdr end={SDDS_FILE_HDR}")
        return errs

    with open(shard_path, "rb") as fh:
        # File header
        magic = fh.read(4)
        if magic != b"SDDS":
            errs.append(f"bad magic {magic!r}")

        # Chunk header
        fh.seek(hdr_start)
        raw_hdr = fh.read(SDDS_CHUNK_HDR)
        stored_name   = raw_hdr[:64].rstrip(b"\x00").decode()
        elem_count,  = struct.unpack_from("<I", raw_hdr, 64)
        fmt,         = struct.unpack_from("<I", raw_hdr, 68)
        stored_bytes,= struct.unpack_from("<I", raw_hdr, 72)
        stored_crc,  = struct.unpack_from("<I", raw_hdr, 76)

        if stored_name != tensor_name:
            errs.append(f"name mismatch: {stored_name!r} != {tensor_name!r}")
        if fmt != 0:
            errs.append(f"format={fmt} (expected 0)")
        if stored_bytes != size_bytes:
            errs.append(f"bytes: hdr={stored_bytes} manifest={size_bytes}")
        if elem_count != size_bytes // 4:
            errs.append(f"elem_count: {elem_count} != {size_bytes//4}")

        # CRC
        fh.seek(offset_in_file)
        data = fh.read(size_bytes)
        computed = zlib.crc32(data) & 0xFFFFFFFF
        if computed != stored_crc:
            errs.append(f"CRC: computed=0x{computed:08x} stored=0x{stored_crc:08x}")

    return errs


def run_test():
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        manifest, slices_a, slices_b, slices_c = _build_manifest_and_shards(tmp)

        # ── Single-shard tensors ─────────────────────────────────────────
        for sl in slices_a:
            errs = _verify_chunk(
                os.path.join(tmp, manifest[sl_shard_idx(manifest, sl)]["dds_file"]),
                sl["name"], sl["offset_in_file"], sl["offset_in_tensor"],
                sl["size"], sl["logical_shape"]
            )
            failures.extend(f"tensor_a: {e}" for e in errs)

        for sl in slices_b:
            errs = _verify_chunk(
                os.path.join(tmp, manifest[sl_shard_idx(manifest, sl)]["dds_file"]),
                sl["name"], sl["offset_in_file"], sl["offset_in_tensor"],
                sl["size"], sl["logical_shape"]
            )
            failures.extend(f"tensor_b: {e}" for e in errs)

        # ── Split tensor C ───────────────────────────────────────────────
        if len(slices_c) < 2:
            failures.append(
                f"tensor_c expected >=2 slices (split), got {len(slices_c)}; "
                f"ceiling={g2s.SDDS_MAX_FILE_BYTES}"
            )
        else:
            # offset_in_tensor chain
            if slices_c[0]["offset_in_tensor"] != 0:
                failures.append(f"tensor_c[0].offset_in_tensor={slices_c[0]['offset_in_tensor']} (expected 0)")
            for i in range(len(slices_c) - 1):
                expected = slices_c[i]["offset_in_tensor"] + slices_c[i]["size"]
                actual   = slices_c[i+1]["offset_in_tensor"]
                if actual != expected:
                    failures.append(
                        f"tensor_c chain break [{i}→{i+1}]: "
                        f"expected {expected}, got {actual}"
                    )
            # total size
            total = sum(s["size"] for s in slices_c)
            if total != 120 * 4:
                failures.append(f"tensor_c total bytes={total} (expected {120*4})")

            for sl in slices_c:
                shard_file = shard_file_for_slice(tmp, manifest, sl)
                errs = _verify_chunk(
                    shard_file, sl["name"], sl["offset_in_file"],
                    sl["offset_in_tensor"], sl["size"], sl["logical_shape"]
                )
                failures.extend(f"tensor_c: {e}" for e in errs)

        # ── File header counts ────────────────────────────────────────────
        for entry in manifest:
            shard_path = os.path.join(tmp, entry["dds_file"])
            if not os.path.exists(shard_path):
                failures.append(f"{entry['dds_file']} not found")
                continue
            with open(shard_path, "rb") as fh:
                hdr = fh.read(12)
            magic   = hdr[:4]
            version,= struct.unpack_from("<I", hdr, 4)
            count,  = struct.unpack_from("<I", hdr, 8)
            expected_count = len(entry["tensors"])
            if magic != b"SDDS":
                failures.append(f"{entry['dds_file']}: bad magic {magic!r}")
            if version != 1:
                failures.append(f"{entry['dds_file']}: version={version}")
            if count != expected_count:
                failures.append(
                    f"{entry['dds_file']}: header count={count} "
                    f"manifest tensors={expected_count}"
                )

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"PASS  ({len(slices_c)} slices for split tensor_c, ceiling={g2s.SDDS_MAX_FILE_BYTES})")


def sl_shard_idx(manifest, sl):
    """Return shard index that holds a slice (by matching dds_file)."""
    for i, entry in enumerate(manifest):
        for ts in entry["tensors"]:
            if ts is sl:
                return i
    raise KeyError(f"slice not found in manifest: {sl['name']}")


def shard_file_for_slice(tmp, manifest, sl):
    for entry in manifest:
        for ts in entry["tensors"]:
            if ts is sl:
                return os.path.join(tmp, entry["dds_file"])
    raise KeyError(f"slice not found: {sl['name']}")


if __name__ == "__main__":
    run_test()
