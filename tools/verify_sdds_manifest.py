#!/usr/bin/env python3
"""
verify_sdds_manifest.py — round-trip verifier for gguf_to_sdds.py output.

For every ShardSlice in model_dds_manifest.json this tool:
  1. Opens the SDDS shard file.
  2. Reads the 80-byte chunk header at offset_in_file − 80.
  3. Verifies name, elem_count, format, bytes fields match the manifest.
  4. Seeks to offset_in_file, streams 'size' bytes, and computes CRC-32.
  5. Compares the computed CRC against the stored CRC in the chunk header.

For multi-shard (split) tensors the tool also checks:
  • offset_in_tensor[0] == 0
  • offset_in_tensor[i+1] == offset_in_tensor[i] + slices[i].size  (no gap, no overlap)
  • sum(slice.size) == product(logical_shape) × 4  (complete F32 coverage)

Finally, the tool reads gpt_oss_20b_folds.kson and reports:
  • n_layers from count of folds whose id contains "attention"
  • Total fold count and shard count

Usage:
  python tools/verify_sdds_manifest.py out_dir/
  python tools/verify_sdds_manifest.py out_dir/ --verbose    # print each slice
  python tools/verify_sdds_manifest.py out_dir/ --no-crc     # skip data read (fast header-only)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from collections import defaultdict
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants — must match sdds_format.cpp and gguf_to_sdds.py
# ---------------------------------------------------------------------------
SDDS_FILE_HDR_BYTES  = 12   # "SDDS"[4] + version[4] + count[4]
SDDS_CHUNK_HDR_BYTES = 80   # name[64] + elem_count[4] + format[4] + bytes[4] + crc[4]
STREAM_CHUNK_BYTES   = 64 * 1024 * 1024  # 64 MB streaming window for CRC


# ---------------------------------------------------------------------------
# Streaming CRC-32 — zlib uses the same IEEE 802.3 polynomial as sdds_crc32()
# ---------------------------------------------------------------------------
def _stream_crc32(fh, offset: int, size: int) -> int:
    """Compute CRC-32 of bytes [offset, offset+size) in file fh, streaming."""
    fh.seek(offset)
    crc = 0
    remaining = size
    while remaining > 0:
        chunk = fh.read(min(remaining, STREAM_CHUNK_BYTES))
        if not chunk:
            break
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    return crc & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
class Failure(NamedTuple):
    tensor: str
    shard:  str
    msg:    str


# ---------------------------------------------------------------------------
# SDDS file-level header check
# ---------------------------------------------------------------------------
def _check_shard_file_header(
    path: str,
    expected_count: int,
    failures: list[Failure],
) -> bool:
    """Verify SDDS file magic, version, and chunk count.  Returns False on error."""
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(SDDS_FILE_HDR_BYTES)
        if len(hdr) < SDDS_FILE_HDR_BYTES:
            failures.append(Failure("(header)", os.path.basename(path),
                                    f"file too short ({len(hdr)} bytes)"))
            return False
        magic   = hdr[:4]
        version, = struct.unpack_from("<I", hdr, 4)
        count,   = struct.unpack_from("<I", hdr, 8)
        if magic != b"SDDS":
            failures.append(Failure("(header)", os.path.basename(path),
                                    f"bad magic {magic!r} (expected b'SDDS')"))
            return False
        if version != 1:
            failures.append(Failure("(header)", os.path.basename(path),
                                    f"unsupported version {version} (expected 1)"))
            return False
        if count != expected_count:
            failures.append(Failure("(header)", os.path.basename(path),
                                    f"chunk count mismatch: header={count} "
                                    f"manifest_expects={expected_count}"))
            return False
    except OSError as e:
        failures.append(Failure("(header)", os.path.basename(path), str(e)))
        return False
    return True


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def verify(out_dir: str, verbose: bool, no_crc: bool) -> int:
    manifest_path = os.path.join(out_dir, "model_dds_manifest.json")
    kson_path     = os.path.join(out_dir, "gpt_oss_20b_folds.kson")

    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    failures: list[Failure] = []

    # Collect slices per tensor name (preserving shard order for chain check)
    tensor_slices: dict[str, list[dict]] = defaultdict(list)
    n_slices_checked = 0

    # ── Per-shard verification ──────────────────────────────────────────────
    for shard_entry in manifest:
        shard_name  = shard_entry["dds_file"]
        shard_path  = os.path.join(out_dir, shard_name)
        shard_slices = shard_entry.get("tensors", [])
        n_chunks_expected = len(shard_slices)
        names_in_shard: set[str] = set()

        if not os.path.exists(shard_path):
            failures.append(Failure("(shard)", shard_name, "shard file not found"))
            continue

        _check_shard_file_header(shard_path, n_chunks_expected, failures)

        try:
            fh = open(shard_path, "rb")
        except OSError as e:
            failures.append(Failure("(shard)", shard_name, str(e)))
            continue

        with fh:
            for ts in shard_slices:
                n_slices_checked += 1
                tname           = ts["name"]
                off_in_file     = ts["offset_in_file"]
                off_in_tensor   = ts["offset_in_tensor"]
                size_bytes      = ts["size"]
                logical_shape   = ts.get("logical_shape", [])

                if tname in names_in_shard:
                    failures.append(Failure(
                        tname, shard_name,
                        "duplicate tensor name within one SDDS shard; split tensor "
                        "slices must use repeated names only across different shard files"
                    ))
                names_in_shard.add(tname)

                if verbose:
                    print(f"  {tname}  shard={shard_name}"
                          f"  off_file={off_in_file}  off_tensor={off_in_tensor}"
                          f"  sz={size_bytes:,}", flush=True)

                tensor_slices[tname].append({
                    "shard_file":       shard_path,
                    "shard_name":       shard_name,
                    "offset_in_file":   off_in_file,
                    "offset_in_tensor": off_in_tensor,
                    "size":             size_bytes,
                    "logical_shape":    logical_shape,
                })

                # ── Chunk header validation ───────────────────────────────
                hdr_start = off_in_file - SDDS_CHUNK_HDR_BYTES
                if hdr_start < SDDS_FILE_HDR_BYTES:
                    failures.append(Failure(
                        tname, shard_name,
                        f"offset_in_file={off_in_file} places chunk header "
                        f"({hdr_start}) before end of file header ({SDDS_FILE_HDR_BYTES})"
                    ))
                    continue

                fh.seek(hdr_start)
                raw_hdr = fh.read(SDDS_CHUNK_HDR_BYTES)
                if len(raw_hdr) != SDDS_CHUNK_HDR_BYTES:
                    failures.append(Failure(
                        tname, shard_name,
                        f"short read of chunk header at {hdr_start}: "
                        f"expected {SDDS_CHUNK_HDR_BYTES} bytes, got {len(raw_hdr)}"
                    ))
                    continue

                stored_name   = raw_hdr[:64].rstrip(b"\x00").decode("utf-8", errors="replace")
                elem_count,  = struct.unpack_from("<I", raw_hdr, 64)
                fmt,         = struct.unpack_from("<I", raw_hdr, 68)
                stored_bytes,= struct.unpack_from("<I", raw_hdr, 72)
                stored_crc,  = struct.unpack_from("<I", raw_hdr, 76)

                # Name
                if stored_name != tname:
                    failures.append(Failure(
                        tname, shard_name,
                        f"chunk name mismatch: manifest={tname!r} vs "
                        f"chunk_header={stored_name!r}"
                    ))

                # Format (must be 0 = F32)
                if fmt != 0:
                    failures.append(Failure(
                        tname, shard_name,
                        f"format={fmt} (expected 0 for F32)"
                    ))

                # Byte count
                if stored_bytes != size_bytes:
                    failures.append(Failure(
                        tname, shard_name,
                        f"byte count: manifest={size_bytes} chunk_header={stored_bytes}"
                    ))

                # elem_count must equal size_bytes // 4 for F32
                expected_elems = size_bytes // 4
                if elem_count != expected_elems:
                    failures.append(Failure(
                        tname, shard_name,
                        f"elem_count: chunk_header={elem_count} "
                        f"expected={expected_elems} (size_bytes/4)"
                    ))

                # ── CRC verification (may skip for --no-crc) ─────────────
                if no_crc:
                    continue

                computed_crc = _stream_crc32(fh, off_in_file, size_bytes)
                if computed_crc != stored_crc:
                    failures.append(Failure(
                        tname, shard_name,
                        f"CRC mismatch: stored=0x{stored_crc:08x} "
                        f"computed=0x{computed_crc:08x}"
                    ))
                elif verbose:
                    print(f"    CRC 0x{computed_crc:08x} OK")

    # ── Multi-shard offset-chain verification ──────────────────────────────
    for tname, slices in tensor_slices.items():
        if len(slices) < 2:
            continue
        # Sort by offset_in_tensor to handle non-insertion order
        slices.sort(key=lambda s: s["offset_in_tensor"])

        # First slice must start at byte 0 of the tensor
        if slices[0]["offset_in_tensor"] != 0:
            failures.append(Failure(
                tname, "(multi-shard)",
                f"first slice offset_in_tensor={slices[0]['offset_in_tensor']} "
                f"(expected 0)"
            ))

        # Contiguous, non-overlapping chain
        for i in range(len(slices) - 1):
            expected_next = slices[i]["offset_in_tensor"] + slices[i]["size"]
            actual_next   = slices[i + 1]["offset_in_tensor"]
            if actual_next != expected_next:
                failures.append(Failure(
                    tname, "(multi-shard)",
                    f"discontinuity between slices {i} and {i+1}: "
                    f"expected offset_in_tensor={expected_next}, "
                    f"got {actual_next} "
                    f"(shards: {slices[i]['shard_name']} -> "
                    f"{slices[i+1]['shard_name']})"
                ))

    # ── Logical-size verification (total slice bytes == full F32 tensor) ───
    for tname, slices in tensor_slices.items():
        total_bytes = sum(s["size"] for s in slices)
        shape = slices[0]["logical_shape"]
        if not shape:
            continue
        n_elems = 1
        for d in shape:
            n_elems *= d
        expected_bytes = n_elems * 4  # F32: 4 bytes per element
        if total_bytes != expected_bytes:
            failures.append(Failure(
                tname, "(logical-size)",
                f"total slice bytes={total_bytes:,} != "
                f"n_elems×4={expected_bytes:,}  (shape={shape})"
            ))

    # ── KSON fold summary ──────────────────────────────────────────────────
    if os.path.exists(kson_path):
        with open(kson_path, encoding="utf-8") as f:
            kson = json.load(f)
        folds = kson.get("@driver", {}).get("@resources", {}).get("folds", [])
        n_attn = sum(1 for fo in folds if "attention" in fo.get("id", ""))
        n_moe  = sum(1 for fo in folds if "moe"       in fo.get("id", ""))
        has_embed  = any(fo.get("id") == "embedding" for fo in folds)
        has_output = any(fo.get("id") == "output"    for fo in folds)
        print(
            f"\nKSON:  {len(folds)} folds"
            f"  attention={n_attn}  moe={n_moe}"
            f"  embedding={'yes' if has_embed else 'MISSING'}"
            f"  output={'yes' if has_output else 'MISSING'}"
            f"  -> n_layers={n_attn}"
        )
        if not has_embed:
            failures.append(Failure("(kson)", "embedding", "embedding fold missing"))
        if not has_output:
            failures.append(Failure("(kson)", "output", "output fold missing"))
        if n_attn == 0:
            failures.append(Failure(
                "(kson)", "n_layers",
                "no folds with 'attention' in id — "
                "InferenceEngine::update_config_from_folds() will set n_layers=0"
            ))
        # n_moe may be 0 for dense-FFN models (layer_NN.ffn instead of layer_NN.moe);
        # only flag a mismatch when both are present and disagree.
        if n_moe > 0 and n_attn != n_moe:
            failures.append(Failure(
                "(kson)", "fold-parity",
                f"attention folds={n_attn} != moe folds={n_moe} "
                f"(expected one of each per transformer block for MoE models)"
            ))
    else:
        print(f"\nWARNING: kson not found at {kson_path} — fold checks skipped")

    # ── Final summary ──────────────────────────────────────────────────────
    n_tensors = len(tensor_slices)
    n_split   = sum(1 for sl in tensor_slices.values() if len(sl) > 1)
    mode_str  = "HEADER-ONLY (--no-crc)" if no_crc else "FULL (header + CRC)"
    print(
        f"\nVerified [{mode_str}]:"
        f"  {n_slices_checked} slices"
        f"  {n_tensors} tensors"
        f"  {len(manifest)} shards"
        f"  {n_split} split tensors"
    )

    if failures:
        print(f"\n{'─'*68}")
        print(f"FAILED  {len(failures)} check(s):")
        for idx, fl in enumerate(failures, 1):
            print(f"  [{idx:3d}]  {fl.tensor}  [{fl.shard}]")
            print(f"           {fl.msg}")
        print(f"{'─'*68}")
        return 1

    print("All checks passed.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("out_dir",
                    help="Directory containing model_dds_manifest.json, "
                         "gpt_oss_20b_folds.kson, and *.sdds shards")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print each slice as it is verified")
    ap.add_argument("--no-crc", action="store_true",
                    help="Skip CRC read (header checks only, much faster)")
    args = ap.parse_args()
    sys.exit(verify(args.out_dir, args.verbose, args.no_crc))


if __name__ == "__main__":
    main()
