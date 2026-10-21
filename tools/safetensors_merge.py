#!/usr/bin/env python3
"""Merge multiple MX2LM vector field safetensors files.

Merge pipeline (matches the K'UHUL merge design):

  Pop   load source fields
  Yax   canonicalize node identity by label string
        build union topology (deduplicate nodes, remap edge endpoints)
        retain source membership mask
  Sek   resolve per-source conditional probabilities into union index space
        blend with chosen strategy
  Ch'en validate node identity, probability mass, edge references
  Xul   emit merged safetensors + source manifest

Alignment is by *label identity*.  Same surface string → same union slot
regardless of its array index in any source file.  This is required because
`field_a.node_weights[500]` and `field_b.node_weights[500]` may represent
completely different n-grams.

Three blend strategies:

  linear (default)
    P_merged(w|h) = Σ αᵢ Pᵢ(w|h)
    Valid conditional distribution when αᵢ sum to 1 and each Pᵢ is
    normalized.  Missing n-grams from a source contribute zero (equivalently,
    that source's P for the absent context is 0).  Cleanest to validate.

  log-linear
    log P_merged = Σ αᵢ log Pᵢ  (unnormalized geometric combination)
    Applies per-history normalization over the union vocabulary.  Suitable
    when all sources have dense coverage; breaks down with many zeros.

  slerp
    Spherical linear interpolation treating the aligned weight vector as a
    point on the unit hypersphere.  Appropriate for blending SEMANTIC VECTORS
    or model checkpoints, NOT for raw probability tables.  Sequential pairwise
    for 3+ sources.

Usage:
  py -3 tools\\safetensors_merge.py \\
      --inputs data/mx2lm.vectors.safetensors \\
               data/cfpc/wildcard-bigram.vectors.safetensors \\
               data/geometric-coder/coder.vectors.safetensors \\
      --alphas 0.4 0.35 0.25 \\
      --strategy linear \\
      --output data/merged/field.vectors.safetensors

Outputs:
  <output>                 merged safetensors
  <output>.labels.json     index -> surface string
  <output>.meta.json       counts, phase map, provenance, source alphas
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

# ── FNV-1a (must match arpa_to_safetensors.py) ───────────────────────────────

def _fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

PHASE_Z = {0: 0.000, 1: 0.167, 2: 0.333}

# ── Load a field ──────────────────────────────────────────────────────────────

def _base_name(path: Path) -> str:
    stem = path.name
    if stem.endswith(".safetensors"):
        return stem[: -len(".safetensors")]
    return path.stem

def _load_field(st_path: Path) -> dict:
    d      = load_file(str(st_path))
    base   = _base_name(st_path)
    lbl_p  = st_path.parent / f"{base}.labels.json"
    meta_p = st_path.parent / f"{base}.meta.json"
    if not lbl_p.exists():
        raise FileNotFoundError(f"Labels sidecar not found: {lbl_p}")
    labels = json.loads(lbl_p.read_text(encoding="utf-8"))
    meta   = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return {
        "pos":          d["node_positions"],
        "w":            d["node_weights"],
        "phase":        d["node_phase"],
        "ei":           d["edge_index"],
        "ew":           d["edge_weights"],
        "labels":       labels,
        "meta":         meta,
        "label_to_id":  {l: i for i, l in enumerate(labels)},
        "path":         st_path,
    }

# ── Yax: build union topology ─────────────────────────────────────────────────

def build_union(fields: list[dict]) -> dict:
    """
    Deduplicate nodes by label, build union index, compute source membership
    mask, map each source's node/edge indices into the union space.
    """
    print("  [Yax] Canonicalising node identity …", file=sys.stderr)

    # Union label set — sorted for reproducibility
    union_set: set[str] = set()
    for f in fields:
        union_set.update(f["labels"])
    union_labels = sorted(union_set)
    N = len(union_labels)
    lbl_to_uid = {l: i for i, l in enumerate(union_labels)}
    print(f"         Union vocabulary: {N:,} nodes "
          f"({sum(len(f['labels']) for f in fields):,} total before dedup)",
          file=sys.stderr)

    # Phase and position for each union node (from first source that has it)
    union_phase = np.zeros(N, dtype=np.int32)
    for f in fields:
        for lbl, sid in f["label_to_id"].items():
            uid = lbl_to_uid[lbl]
            union_phase[uid] = int(f["phase"][sid])

    union_pos = np.zeros((N, 3), dtype=np.float32)
    for i, lbl in enumerate(union_labels):
        ph = int(union_phase[i])
        h  = _fnv1a(lbl)
        union_pos[i, 0] = (h & 0xFFFF) / 65535.0
        union_pos[i, 1] = ((h >> 16) & 0xFFFF) / 65535.0
        union_pos[i, 2] = PHASE_Z.get(ph, 0.0)

    # Per-source mapping arrays: src_node_idx → union_idx
    src_maps = []
    for f in fields:
        m = np.array([lbl_to_uid[l] for l in f["labels"]], dtype=np.int64)
        src_maps.append(m)

    # Source membership bitmask: bit i set iff source i contains this node
    K = len(fields)
    node_source_mask = np.zeros(N, dtype=np.uint8)
    for fi, m in enumerate(src_maps):
        node_source_mask[m] |= np.uint8(1 << fi)

    # Remap each source's edges into union index space (vectorised)
    print("  [Yax] Remapping edge topology …", file=sys.stderr)
    remapped_edges = []   # list of (union_src [E], union_tgt [E], ew [E])
    for fi, f in enumerate(fields):
        ei  = f["ei"]           # [2, E] int64
        ew  = f["ew"]           # [E]    float32
        m   = src_maps[fi]
        u_src = m[ei[0]]        # vectorised remap
        u_tgt = m[ei[1]]
        remapped_edges.append((u_src, u_tgt, ew))

    return {
        "labels":           union_labels,
        "lbl_to_uid":       lbl_to_uid,
        "N":                N,
        "union_phase":      union_phase,
        "union_pos":        union_pos,
        "src_maps":         src_maps,
        "node_source_mask": node_source_mask,
        "remapped_edges":   remapped_edges,
    }

# ── Sek: weight alignment ─────────────────────────────────────────────────────

def align_weights(fields: list[dict], union: dict) -> tuple[list[np.ndarray],
                                                             list[np.ndarray]]:
    """
    For each source, scatter its node_weights and edge_weights into aligned
    arrays over the union vocabulary.  Missing entries stay 0.0.
    """
    N   = union["N"]
    src_maps = union["src_maps"]

    # Node weight columns: shape [N] per source
    node_cols: list[np.ndarray] = []
    for fi, f in enumerate(fields):
        col = np.zeros(N, dtype=np.float64)
        col[src_maps[fi]] = f["w"].astype(np.float64)
        node_cols.append(col)

    return node_cols

# ── Sek: edge deduplication and blending (vectorised) ────────────────────────

def merge_edges(remapped_edges: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                alphas: list[float],
                strategy: str,
                N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate all remapped edges, deduplicate by (u_src, u_tgt) key, blend.
    Returns (edge_index [2,E], edge_weights [E], edge_source_mask [E]).
    """
    print("  [Sek] Deduplicating and blending edges …", file=sys.stderr)

    # Encode each edge as a single int64 key: src * N + tgt
    # N_max = 8.9M → max key ≈ 7.9e13, fits in int64 (max 9.2e18) ✓
    all_keys   : list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_alphas : list[np.ndarray] = []
    all_srcbits: list[np.ndarray] = []

    N64 = np.int64(N)
    for fi, (u_src, u_tgt, ew) in enumerate(remapped_edges):
        keys = u_src.astype(np.int64) * N64 + u_tgt.astype(np.int64)
        alpha_col = np.full(len(keys), alphas[fi], dtype=np.float64)
        src_bit   = np.full(len(keys), np.uint8(1 << fi), dtype=np.uint8)
        all_keys.append(keys)
        all_weights.append(ew.astype(np.float64))
        all_alphas.append(alpha_col)
        all_srcbits.append(src_bit)

    keys   = np.concatenate(all_keys)
    weights = np.concatenate(all_weights)
    alphas_ = np.concatenate(all_alphas)
    srcbits = np.concatenate(all_srcbits)

    # Sort by key so identical edges are contiguous
    order  = np.argsort(keys, kind="stable")
    keys   = keys[order]
    weights = weights[order]
    alphas_ = alphas_[order]
    srcbits = srcbits[order]

    # Find group boundaries (where key changes)
    boundaries = np.concatenate([[0],
                                  np.where(np.diff(keys))[0] + 1,
                                  [len(keys)]])
    n_groups   = len(boundaries) - 1
    print(f"         {n_groups:,} unique edges from "
          f"{len(keys):,} total (with duplicates)", file=sys.stderr)

    merged_keys    = keys[boundaries[:-1]]          # one key per group
    merged_ew      = np.zeros(n_groups, dtype=np.float64)
    merged_srcmask = np.zeros(n_groups, dtype=np.uint8)

    if strategy == "linear":
        # P_merged(w|h) = Σ αᵢ Pᵢ(w|h)
        # For singleton edges: α_source × P_source (other sources contribute 0)
        for g in range(n_groups):
            s, e = boundaries[g], boundaries[g + 1]
            # Normalise alphas within present sources
            a = alphas_[s:e]
            w = weights[s:e]
            total_a = a.sum()
            merged_ew[g] = float((w * a).sum() / total_a) if total_a > 0 else 0.0
            for b in srcbits[s:e]:
                merged_srcmask[g] |= b

    elif strategy == "log-linear":
        # Unnormalized geometric combination; warn user to normalise downstream
        eps = 1e-30
        for g in range(n_groups):
            s, e = boundaries[g], boundaries[g + 1]
            a = alphas_[s:e]
            w = weights[s:e]
            total_a = a.sum()
            log_blend = float((a * np.log(np.maximum(w, eps))).sum() / total_a)
            merged_ew[g] = np.exp(log_blend)
            for b in srcbits[s:e]:
                merged_srcmask[g] |= b

    else:   # slerp
        # SLERP is defined on the full node_weight vector not per-edge;
        # here we apply scalar slerp = linear as a reasonable approximation.
        for g in range(n_groups):
            s, e = boundaries[g], boundaries[g + 1]
            a = alphas_[s:e]
            w = weights[s:e]
            total_a = a.sum()
            merged_ew[g] = float((w * a).sum() / total_a) if total_a > 0 else 0.0
            for b in srcbits[s:e]:
                merged_srcmask[g] |= b

    # Decode keys back to (src, tgt)
    u_src = (merged_keys // N64).astype(np.int64)
    u_tgt = (merged_keys  % N64).astype(np.int64)
    edge_index = np.array([u_src, u_tgt], dtype=np.int64)

    return edge_index, merged_ew.astype(np.float32), merged_srcmask

# ── Sek: node weight blending ─────────────────────────────────────────────────

def blend_node_weights(node_cols: list[np.ndarray],
                       alphas: list[float],
                       strategy: str) -> np.ndarray:
    N = node_cols[0].shape[0]

    if strategy == "linear":
        out = np.zeros(N, dtype=np.float64)
        for col, a in zip(node_cols, alphas):
            out += a * col
        return out.astype(np.float32)

    elif strategy == "log-linear":
        eps = 1e-30
        log_sum = np.zeros(N, dtype=np.float64)
        for col, a in zip(node_cols, alphas):
            log_sum += a * np.log(np.maximum(col, eps))
        return np.exp(log_sum).astype(np.float32)

    elif strategy == "slerp":
        # Full-vector SLERP on the aligned node_weight arrays
        result = node_cols[0].copy()
        cumulative = alphas[0]
        for col, a in zip(node_cols[1:], alphas[1:]):
            total = cumulative + a
            t = a / total if total > 0 else 0.5
            x, y = result, col
            n0, n1 = float(np.linalg.norm(x)), float(np.linalg.norm(y))
            if n0 < 1e-8 or n1 < 1e-8:
                result = (1 - t) * x + t * y
            else:
                u, v = x / n0, y / n1
                dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
                if abs(dot) > 0.9995:
                    z = (1 - t) * u + t * v
                else:
                    theta = np.arccos(dot)
                    z = (np.sin((1 - t) * theta) / np.sin(theta) * u
                         + np.sin(t * theta)      / np.sin(theta) * v)
                result = z * ((1 - t) * n0 + t * n1)
            cumulative = total
        return result.astype(np.float32)

    raise ValueError(f"Unknown strategy: {strategy!r}")

# ── Ch'en: validate ───────────────────────────────────────────────────────────

def validate(node_w: np.ndarray, edge_w: np.ndarray,
             edge_index: np.ndarray, N: int, strategy: str) -> None:
    print("  [Ch'en] Validating …", file=sys.stderr)
    errors: list[str] = []

    if np.any(node_w < 0):
        errors.append("negative node weights detected")
    if np.any(edge_w < 0):
        errors.append("negative edge weights detected")
    if np.any(edge_w > 1.0 + 1e-5):
        n_over = int((edge_w > 1.0 + 1e-5).sum())
        errors.append(f"{n_over:,} edge weights > 1.0 (conditional probabilities)")
    out_of_range = ((edge_index[0] < 0) | (edge_index[0] >= N) |
                    (edge_index[1] < 0) | (edge_index[1] >= N))
    if np.any(out_of_range):
        errors.append(f"{int(out_of_range.any(axis=0).sum()):,} edges reference invalid node indices")
    if np.any(np.isnan(node_w)) or np.any(np.isnan(edge_w)):
        errors.append("NaN values in merged tensors")

    if strategy == "log-linear":
        print("         [warn] log-linear weights are not renormalized; "
              "downstream consumers should apply per-history normalization.",
              file=sys.stderr)

    if errors:
        print("         [fail]", file=sys.stderr)
        for e in errors:
            print(f"           × {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("         [pass] node identity, edge references, weight bounds",
              file=sys.stderr)

# ── Xul: emit ────────────────────────────────────────────────────────────────

def emit(out_path: Path,
         tensors: dict[str, np.ndarray],
         labels: list[str],
         meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base      = _base_name(out_path)
    lbl_path  = out_path.parent / f"{base}.labels.json"
    meta_path = out_path.parent / f"{base}.meta.json"

    print(f"  [Xul]  Writing {out_path.name} …", file=sys.stderr)
    save_file(tensors, str(out_path))
    lbl_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    kb_st  = out_path.stat().st_size  // 1024
    kb_lbl = lbl_path.stat().st_size  // 1024
    print(f"\n  {out_path.name:<50s}  {kb_st:>8,} KB"
          f"  ({meta['n_nodes']:,} nodes, {meta['n_edges']:,} edges)",
          file=sys.stderr)
    print(f"  {lbl_path.name:<50s}  {kb_lbl:>8,} KB  (labels)",
          file=sys.stderr)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge MX2LM vector field safetensors files")
    ap.add_argument("--inputs",   nargs="+", required=True)
    ap.add_argument("--alphas",   nargs="+", type=float,
                    help="Per-source blend weights (default: equal)")
    ap.add_argument("--strategy", choices=["linear", "log-linear", "slerp"],
                    default="linear",
                    help="linear (default, normalized), log-linear, slerp")
    ap.add_argument("--output",   required=True)
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="Drop merged nodes below this weight")
    args = ap.parse_args()

    paths  = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            sys.exit(f"[error] Not found: {p}")

    alphas = args.alphas or [1.0 / len(paths)] * len(paths)
    if len(alphas) != len(paths):
        sys.exit(f"[error] --alphas count != --inputs count")
    total = sum(alphas)
    alphas = [a / total for a in alphas]

    t0 = time.time()
    print("[Pop]  Loading source fields …", file=sys.stderr)
    fields = []
    for p in paths:
        n = len(json.loads((p.parent / (_base_name(p) + ".labels.json")
                            ).read_text(encoding="utf-8")))
        print(f"       {p.name}  ({n:,} nodes)", file=sys.stderr)
        fields.append(_load_field(p))

    # Yax: topology
    union = build_union(fields)

    # Sek: weights
    print("  [Sek] Aligning node weights …", file=sys.stderr)
    node_cols = align_weights(fields, union)
    node_w    = blend_node_weights(node_cols, alphas, args.strategy)

    edge_index, edge_w, edge_src_mask = merge_edges(
        union["remapped_edges"], alphas, args.strategy, union["N"])

    # Optional min-prob filter
    N = union["N"]
    labels     = union["labels"]
    node_phase = union["union_phase"]
    node_pos   = union["union_pos"]
    node_srcmask = union["node_source_mask"]

    if args.min_prob > 0:
        keep   = np.where(node_w >= args.min_prob)[0]
        remap  = np.full(N, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep), dtype=np.int64)
        emask  = (remap[edge_index[0]] >= 0) & (remap[edge_index[1]] >= 0)
        edge_index   = np.array([remap[edge_index[0][emask]],
                                  remap[edge_index[1][emask]]], dtype=np.int64)
        edge_w       = edge_w[emask]
        edge_src_mask = edge_src_mask[emask]
        node_w       = node_w[keep]
        node_phase   = node_phase[keep]
        node_pos     = node_pos[keep]
        node_srcmask = node_srcmask[keep]
        labels       = [labels[i] for i in keep]
        N = len(labels)
        print(f"         After min-prob filter: {N:,} nodes "
              f"{edge_index.shape[1]:,} edges", file=sys.stderr)

    E = edge_index.shape[1]

    # Ch'en: validate
    validate(node_w, edge_w, edge_index, N, args.strategy)

    phase_counts = {str(ph): int((node_phase == ph).sum()) for ph in range(3)}
    print(f"         Phase counts: Pop={phase_counts['0']:,}  "
          f"Wo={phase_counts['1']:,}  Yax={phase_counts['2']:,}",
          file=sys.stderr)

    tensors = {
        "node_positions":  node_pos,
        "node_weights":    node_w,
        "node_phase":      node_phase,
        "node_source_mask": node_srcmask,
        "edge_index":      edge_index,
        "edge_weights":    edge_w,
        "edge_source_mask": edge_src_mask,
    }
    meta = {
        "schema":        "mx2lm.merged.vectors/1",
        "strategy":      args.strategy,
        "alphas":        alphas,
        "sources":       [str(p) for p in paths],
        "n_nodes":       N,
        "n_edges":       E,
        "phase_map":     {"Pop": 0, "Wo": 1, "Yax": 2},
        "phase_counts":  phase_counts,
        "source_bits":   {str(p.name): (1 << i) for i, p in enumerate(paths)},
    }

    # Xul: emit
    emit(Path(args.output), tensors, labels, meta)
    print(f"\nDone in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
