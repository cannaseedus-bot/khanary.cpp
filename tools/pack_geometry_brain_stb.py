"""Pack khanary-geometry brain sidecars into the SVG-Tensor `.stb` container.

The geometry compiler emits tensor sidecars and an optional `.brain` bundle.
Historical `khanary_brain.stb` artifacts use a compact STB0 header followed by
the same sidecar payloads:

    nodes_time, nodes_freq, nodes_energy, edges, csr_index, csr_neighbors, experts

This script keeps that projection container reproducible without re-running the
WAV/spectrogram compiler.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


MAGIC_STB0 = 0x30425453
STB_VERSION = 1
TENSOR_COUNT = 7
HEADER_BYTES = 256


TENSORS = (
    ("nodes_time.bin", 0x00010000, "nodes", 0),
    ("nodes_freq.bin", 0x00010001, "nodes", 0),
    ("nodes_energy.bin", 0x00010002, "nodes", 0),
    ("edges.bin", 0x00020303, "edges", 2),
    ("csr_index.bin", 0x00010304, "csr_index", 0),
    ("csr_neighbors.bin", 0x00010305, "csr_entries", 0),
    ("experts.bin", 0x00010306, "nodes", 0),
)


def _read_header(run_dir: Path) -> tuple[int, int, int, int, int]:
    payload = (run_dir / "tensor_header.bin").read_bytes()
    if len(payload) != 20:
        raise ValueError(f"tensor_header.bin must be 20 bytes, got {len(payload)}")
    return struct.unpack("<IIIII", payload)


def pack_stb(run_dir: Path, output: Path) -> None:
    nodes, edges, csr_entries, _experts, _feature_count = _read_header(run_dir)
    counts = {
        "nodes": nodes,
        "edges": edges,
        "csr_index": nodes + 1,
        "csr_entries": csr_entries,
    }

    payloads = []
    for filename, _type_id, count_key, components in TENSORS:
        payload = (run_dir / filename).read_bytes()
        count = counts[count_key]
        expected_size = count * 4 * (components if components else 1)
        if len(payload) != expected_size:
            raise ValueError(
                f"{filename} has {len(payload)} bytes, expected {expected_size}"
            )
        payloads.append(payload)

    total_bytes = HEADER_BYTES + sum(len(payload) for payload in payloads)
    header = bytearray(HEADER_BYTES)
    struct.pack_into(
        "<IIIIIIII",
        header,
        0,
        MAGIC_STB0,
        (TENSOR_COUNT << 16) | STB_VERSION,
        0,
        0,
        HEADER_BYTES,
        0,
        total_bytes,
        0,
    )

    offset = HEADER_BYTES
    descriptor_offset = 32
    for (filename, type_id, count_key, components), payload in zip(TENSORS, payloads):
        del filename
        struct.pack_into(
            "<IIIIIIII",
            header,
            descriptor_offset,
            type_id,
            offset,
            0,
            len(payload),
            0,
            counts[count_key],
            components,
            0,
        )
        descriptor_offset += 32
        offset += len(payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        fh.write(header)
        for payload in payloads:
            fh.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output .stb path. Defaults to <run_dir>/khanary_brain.stb.",
    )
    args = parser.parse_args()
    output = args.output or (args.run_dir / "khanary_brain.stb")
    pack_stb(args.run_dir, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
