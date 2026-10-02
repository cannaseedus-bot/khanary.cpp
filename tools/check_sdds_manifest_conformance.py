#!/usr/bin/env python3
"""Synthetic SDDS/FoldMap conformance vectors.

This test keeps the GGUF -> SDDS -> FoldMap path independent of real model
weights. It proves the transport-level contracts that runtime loaders depend
on:

* offset_in_file points to the exact raw payload bytes.
* split tensors reconstruct from offset_in_tensor-ordered slices.
* repeated tensor names are valid across shard files, but invalid within one
  SDDS shard.
* FoldMap naming keeps semantic Fold authority separate from physical runtime
  grouping while preserving layer discovery through ".attention" fold IDs.

Suite name: sdds_manifest  (must match run_conformance.py required set)
"""
from __future__ import annotations

import json
import contextlib
import io
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, str(Path(__file__).parent))
from conformance_common import verdict
import verify_sdds_manifest


ROOT = Path(__file__).resolve().parents[1]
SDDS_FILE_HDR_BYTES = 12
SDDS_CHUNK_HDR_BYTES = 80


def _f32_payload(start: int, count: int) -> bytes:
    return b"".join(struct.pack("<f", float(start + i)) for i in range(count))


def _write_sdds(path: Path, chunks: list[tuple[str, bytes]]) -> list[dict]:
    """Write a minimal SDDS v1 shard and return chunk metadata."""
    offsets: list[dict] = []
    with path.open("wb") as f:
        f.write(b"SDDS")
        f.write(struct.pack("<II", 1, len(chunks)))
        pos = SDDS_FILE_HDR_BYTES
        for name, raw in chunks:
            name_b = name.encode("utf-8")
            if len(name_b) > 63:
                raise ValueError("name too long")
            f.write(name_b + b"\0" * (64 - len(name_b)))
            f.write(struct.pack("<IIII", len(raw) // 4, 0, len(raw), zlib.crc32(raw) & 0xFFFFFFFF))
            data_off = pos + SDDS_CHUNK_HDR_BYTES
            f.write(raw)
            offsets.append({"name": name, "offset_in_file": data_off, "size": len(raw)})
            pos += SDDS_CHUNK_HDR_BYTES + len(raw)
    return offsets


def _write_foldmap(out_dir: Path, stripped_attention: bool = False) -> None:
    if stripped_attention:
        folds = [
            {"fold": 0, "id": "embedding", "nodes": []},
            {"fold": 1, "id": "layer_00", "nodes": []},
            {"fold": 2, "id": "layer_00.moe", "nodes": []},
            {"fold": 3, "id": "layer_01", "nodes": []},
            {"fold": 4, "id": "layer_01.moe", "nodes": []},
            {"fold": 5, "id": "output", "nodes": []},
        ]
    else:
        folds = [
            {"fold": 0, "id": "embedding", "nodes": []},
            {"fold": 1, "id": "layer_00.attention", "nodes": []},
            {"fold": 2, "id": "layer_00.moe", "nodes": []},
            {"fold": 3, "id": "layer_01.attention", "nodes": []},
            {"fold": 4, "id": "layer_01.moe", "nodes": []},
            {"fold": 5, "id": "output", "nodes": []},
        ]
    (out_dir / "gpt_oss_20b_folds.kson").write_text(
        json.dumps({"@driver": {"@resources": {"folds": folds}}}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_good_fixture(out_dir: Path) -> bytes:
    """Create two shards with one split output.weight tensor."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _f32_payload(0, 8)
    first = payload[:16]
    second = payload[16:]
    off0 = _write_sdds(out_dir / "fold_000.sdds", [("output.weight", first)])[0]
    off1 = _write_sdds(out_dir / "fold_001.sdds", [("output.weight", second)])[0]
    manifest = [
        {
            "id": 0,
            "dds_file": "fold_000.sdds",
            "tensors": [{
                "name": "output.weight",
                "offset_in_file": off0["offset_in_file"],
                "offset_in_tensor": 0,
                "size": off0["size"],
                "dtype": "F32",
                "logical_shape": [8],
                "shape": [8],
            }],
        },
        {
            "id": 1,
            "dds_file": "fold_001.sdds",
            "tensors": [{
                "name": "output.weight",
                "offset_in_file": off1["offset_in_file"],
                "offset_in_tensor": len(first),
                "size": off1["size"],
                "dtype": "F32",
                "logical_shape": [8],
                "shape": [8],
            }],
        },
    ]
    (out_dir / "model_dds_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_foldmap(out_dir)
    return payload


def _read_slice(out_dir: Path, tensor_slice: dict) -> bytes:
    with (out_dir / tensor_slice["shard_name"]).open("rb") as f:
        f.seek(tensor_slice["offset_in_file"])
        return f.read(tensor_slice["size"])


def _manifest_slices(out_dir: Path) -> list[dict]:
    manifest = json.loads((out_dir / "model_dds_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for shard in manifest:
        for sl in shard["tensors"]:
            row = dict(sl)
            row["shard_name"] = shard["dds_file"]
            rows.append(row)
    return rows


def _reconstruct(out_dir: Path, tensor_name: str) -> bytes:
    slices = [s for s in _manifest_slices(out_dir) if s["name"] == tensor_name]
    slices.sort(key=lambda s: s["offset_in_tensor"])
    return b"".join(_read_slice(out_dir, s) for s in slices)


def _n_layers(out_dir: Path) -> int:
    kson = json.loads((out_dir / "gpt_oss_20b_folds.kson").read_text(encoding="utf-8"))
    folds = kson.get("@driver", {}).get("@resources", {}).get("folds", [])
    return sum(1 for fold in folds if "attention" in fold.get("id", ""))


def _verify(out_dir: Path) -> int:
    """Run the manifest verifier while keeping this suite's output stable."""
    with contextlib.redirect_stdout(io.StringIO()):
        return verify_sdds_manifest.verify(str(out_dir), verbose=False, no_crc=False)


def _write_duplicate_same_shard_fixture(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_a = _f32_payload(0, 2)
    payload_b = _f32_payload(2, 2)
    offsets = _write_sdds(out_dir / "fold_000.sdds", [
        ("output.weight", payload_a),
        ("output.weight", payload_b),
    ])
    manifest = [{
        "id": 0,
        "dds_file": "fold_000.sdds",
        "tensors": [
            {
                "name": "output.weight",
                "offset_in_file": offsets[0]["offset_in_file"],
                "offset_in_tensor": 0,
                "size": offsets[0]["size"],
                "dtype": "F32",
                "logical_shape": [4],
                "shape": [4],
            },
            {
                "name": "output.weight",
                "offset_in_file": offsets[1]["offset_in_file"],
                "offset_in_tensor": offsets[0]["size"],
                "size": offsets[1]["size"],
                "dtype": "F32",
                "logical_shape": [4],
                "shape": [4],
            },
        ],
    }]
    (out_dir / "model_dds_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_foldmap(out_dir)


def _copy_bad_foldmap_fixture(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("fold_000.sdds", "fold_001.sdds", "model_dds_manifest.json"):
        (dst / name).write_bytes((src / name).read_bytes())
    _write_foldmap(dst, stripped_attention=True)


def main() -> int:
    cases: list[tuple[str, bool]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="khanary_sdds_manifest_") as tmp:
            root = Path(tmp)
            good = root / "good"
            expected_payload = _write_good_fixture(good)

            verify_ok = _verify(good) == 0
            cases.append(("offset_integrity_direct_seek", verify_ok and _reconstruct(good, "output.weight") == expected_payload))

            rows = _manifest_slices(good)
            offsets_ok = sorted(s["offset_in_tensor"] for s in rows) == [0, 16]
            cases.append(("split_tensor_integrity", offsets_ok and _reconstruct(good, "output.weight") == expected_payload))

            shard_names = {s["shard_name"] for s in rows if s["name"] == "output.weight"}
            cases.append(("repeat_name_cross_shard_accepted", verify_ok and shard_names == {"fold_000.sdds", "fold_001.sdds"}))

            cases.append(("fold_naming_integrity", verify_ok and _n_layers(good) == 2))

            bad_dup = root / "bad_duplicate_same_shard"
            _write_duplicate_same_shard_fixture(bad_dup)
            cases.append(("duplicate_name_same_shard_rejected", _verify(bad_dup) != 0))

            bad_foldmap = root / "bad_foldmap"
            _copy_bad_foldmap_fixture(good, bad_foldmap)
            cases.append(("stripped_attention_breaks_layer_discovery", _n_layers(bad_foldmap) == 0 and _verify(bad_foldmap) != 0))

        return verdict("sdds_manifest", cases)
    except Exception as exc:
        return verdict("sdds_manifest", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
