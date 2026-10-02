#!/usr/bin/env python3
"""Export khanary-geometry brain tensor sidecars to a KHANARY GML overlay.

Usage:
    python tools/export_brain_gml.py <run_dir> [output.gml]

If output.gml is omitted, writes to <run_dir>/brain.gml.

Sidecar contract (from pack_geometry_brain_stb.py):
    tensor_header.bin   20 bytes  struct "<IIIII"  nodes/edges/csr_entries/experts/feature_count
    nodes_time.bin      float32[N]   spectrogram time axis (seconds)
    nodes_freq.bin      float32[N]   spectrogram frequency axis (Hz)
    nodes_energy.bin    float32[N]   ridge energy (often zero; used as local_entropy proxy)
    experts.bin         uint32[N]    MoE expert assignment (0..experts-1)
    edges.bin           uint32[E*2]  Delaunay (source,target) index pairs
    csr_index.bin       uint32[N+1]  CSR row offsets (used to derive node degree)

GML field mapping:
    x            <- nodes_time  (normalised 0-1)
    y            <- nodes_freq  (normalised 0-1)
    z            <- 0.0
    local_entropy  <- nodes_energy (normalised 0-1)
    static_weight  <- nodes_energy (normalised 0-1)
    confidence     <- 1.0
    health         <- 1.0
    geodesic_entropy <- 0.0  (set by dispatch_geodesic_propagator() at runtime)
    provider_id    <- "brain4-expert-NN"  (from experts.bin)
    divergence     <- derived from node degree (hub > 0, endpoint < 0)
    singularity    <- "source" | "sink" | "ordinary"  (degree-based)
    verb           <- "@related_to"
    field_kind     <- "riemannian_flow"
    vx/vy          <- unit tangent (normalised-coord space)
    eh_x/eh_y      <- energy gradient projected onto edge direction
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def _read_floats(path: Path, count: int) -> list[float]:
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f"{path.name}: expected {count * 4} bytes, got {len(data)}")
    return list(struct.unpack(f"<{count}f", data))


def _read_uints(path: Path, count: int) -> list[int]:
    data = path.read_bytes()
    if len(data) != count * 4:
        raise ValueError(f"{path.name}: expected {count * 4} bytes, got {len(data)}")
    return list(struct.unpack(f"<{count}I", data))


def _normalise(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0.0:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def export_gml(run_dir: Path, output: Path) -> None:
    # ── header ────────────────────────────────────────────────────────────────
    hdr = (run_dir / "tensor_header.bin").read_bytes()
    if len(hdr) != 20:
        raise ValueError("tensor_header.bin must be exactly 20 bytes")
    n_nodes, n_edges, n_csr, n_experts, _ = struct.unpack("<IIIII", hdr)

    # ── per-node tensors ──────────────────────────────────────────────────────
    time_raw   = _read_floats(run_dir / "nodes_time.bin",   n_nodes)
    freq_raw   = _read_floats(run_dir / "nodes_freq.bin",   n_nodes)
    energy_raw = _read_floats(run_dir / "nodes_energy.bin", n_nodes)
    experts    = _read_uints (run_dir / "experts.bin",      n_nodes)

    # ── CSR row offsets → degree ──────────────────────────────────────────────
    csr_index = _read_uints(run_dir / "csr_index.bin", n_nodes + 1)
    assert csr_index[-1] == n_csr, (
        f"csr_index[-1]={csr_index[-1]} != n_csr={n_csr}"
    )
    degree = [csr_index[i + 1] - csr_index[i] for i in range(n_nodes)]

    # ── edge pairs ────────────────────────────────────────────────────────────
    edge_raw = _read_uints(run_dir / "edges.bin", n_edges * 2)
    edges = [(edge_raw[i * 2], edge_raw[i * 2 + 1]) for i in range(n_edges)]

    # sanity: all edge endpoints must be valid node indices
    bad = [(i, s, t) for i, (s, t) in enumerate(edges)
           if s >= n_nodes or t >= n_nodes]
    if bad:
        raise ValueError(f"{len(bad)} edges with out-of-range endpoints: {bad[:3]}")

    # sanity: all expert ids in range
    bad_exp = [v for v in experts if v >= n_experts]
    if bad_exp:
        raise ValueError(f"{len(bad_exp)} expert ids >= {n_experts}: {bad_exp[:3]}")

    # ── normalise coordinates and energy ─────────────────────────────────────
    x    = _normalise(time_raw)
    y    = _normalise(freq_raw)
    enrg = _normalise(energy_raw)

    mean_deg = sum(degree) / len(degree) if degree else 1.0
    min_deg  = min(degree)
    max_deg  = max(degree)
    deg_span = max(max_deg - min_deg, 1)

    def singularity_of(deg: int) -> str:
        if deg <= 1:
            return "sink"
        if deg >= 2 * mean_deg:
            return "source"
        return "ordinary"

    def divergence_of(deg: int) -> float:
        # +1 = maximum hub, -1 = endpoint; linear on [min_deg, max_deg]
        return 2.0 * (deg - min_deg) / deg_span - 1.0

    run_name = run_dir.name  # "brain4"

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        w = fh.write

        w("graph [\n")
        w("    directed 1\n")
        w(f'    comment "khanary-geometry {run_name}: {n_nodes} nodes '
          f'{n_edges} edges {n_experts} experts"\n')
        w(f'    label "{run_name}"\n')
        w("\n")

        for i in range(n_nodes):
            deg  = degree[i]
            xi   = x[i]
            yi   = y[i]
            ei   = enrg[i]
            div  = divergence_of(deg)
            sing = singularity_of(deg)
            exp_id = experts[i]

            w("    node [\n")
            w(f"        id {i}\n")
            w(f'        label "{run_name}:{i}"\n')
            w(f"        x {xi:.6f}\n")
            w(f"        y {yi:.6f}\n")
            w(f"        z 0.0\n")
            w(f"        local_entropy {ei:.6f}\n")
            w(f"        static_weight {ei:.6f}\n")
            w(f"        confidence 1.0\n")
            w(f"        health 1.0\n")
            w(f"        geodesic_entropy 0.0\n")
            w(f"        divergence {div:.6f}\n")
            w(f'        singularity "{sing}"\n')
            w(f'        provider_id "{run_name}-expert-{exp_id:02d}"\n')
            w("    ]\n")

        w("\n")

        for src, tgt in edges:
            dx = x[tgt] - x[src]
            dy = y[tgt] - y[src]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 1e-9:
                ux, uy = dx / length, dy / length
            else:
                ux, uy = 0.0, 0.0

            d_energy = enrg[tgt] - enrg[src]
            eh_x = d_energy * ux
            eh_y = d_energy * uy

            w("    edge [\n")
            w(f"        source {src}\n")
            w(f"        target {tgt}\n")
            w(f'        verb "@related_to"\n')
            w(f"        weight 1.0\n")
            w(f"        length {length:.6f}\n")
            w(f'        field_kind "riemannian_flow"\n')
            w(f"        vx {ux:.6f}\n")
            w(f"        vy {uy:.6f}\n")
            w(f"        vz 0.0\n")
            w(f"        eh_x {eh_x:.6f}\n")
            w(f"        eh_y {eh_y:.6f}\n")
            w(f"        eh_z 0.0\n")
            w("    ]\n")

        w("]\n")

    size_kb = output.stat().st_size // 1024
    print(f"Wrote {output}  ({n_nodes} nodes, {n_edges} edges, {size_kb} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export brain tensor sidecars to a KHANARY GML overlay."
    )
    ap.add_argument("run_dir", type=Path,
                    help="Brain run directory, e.g. models/khanary-geometry-v0.3.0/brain4")
    ap.add_argument("output", type=Path, nargs="?",
                    help="Output .gml path. Defaults to <run_dir>/brain.gml.")
    args = ap.parse_args()
    output = args.output or (args.run_dir / "brain.gml")
    export_gml(args.run_dir, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
