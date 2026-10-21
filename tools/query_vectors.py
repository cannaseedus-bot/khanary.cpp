#!/usr/bin/env python3
"""Interactive query demo for mx2lm.vectors.safetensors."""
import io
import json
import sys
import numpy as np
from safetensors.numpy import load_file
from pathlib import Path

# Force UTF-8 output so emoji / non-ASCII labels don't crash on cp1252 consoles.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

root   = Path(__file__).resolve().parents[1] / "data"
d      = load_file(str(root / "mx2lm.vectors.safetensors"))
labels = json.loads((root / "mx2lm.vectors.labels.json").read_text(encoding="utf-8"))
meta   = json.loads((root / "mx2lm.vectors.meta.json").read_text(encoding="utf-8"))

pos   = d["node_positions"]   # [N, 3]
w     = d["node_weights"]     # [N]
phase = d["node_phase"]       # [N]
ei    = d["edge_index"]       # [2, E]
ew    = d["edge_weights"]     # [E]

# ── Tensor shapes ──────────────────────────────────────────────────────────────
print("=== Tensor shapes ===")
for k, v in d.items():
    print(f"  {k:20s}  {str(v.shape):22s}  {v.dtype}")

# ── Layer counts ───────────────────────────────────────────────────────────────
print("\n=== Layer counts ===")
for pid, pname in meta["phase_map"].items():
    mask = phase == int(pid)
    n = mask.sum()
    if n == 0:
        continue
    wmin, wmax = float(w[mask].min()), float(w[mask].max())
    print(f"  phase {pid} ({pname:6s}): {n:>10,} nodes  "
          f"weight range [{wmin:.2e}, {wmax:.2e}]")

# ── Top-10 unigrams ────────────────────────────────────────────────────────────
print("\n=== Top-10 unigrams by P(w) ===")
pop_mask = phase == 0
scores   = np.where(pop_mask, w, 0.0)
for i in np.argsort(scores)[::-1][:10]:
    x, y, z = pos[i]
    print(f"  {labels[i]:20s}  P={w[i]:.4e}  pos=({x:.4f}, {y:.4f}, {z:.3f})")

# ── Top-10 bigrams ─────────────────────────────────────────────────────────────
print("\n=== Top-10 bigrams by P(w1,w2) ===")
wo_mask = phase == 1
scores  = np.where(wo_mask, w, 0.0)
for i in np.argsort(scores)[::-1][:10]:
    print(f"  {labels[i]:35s}  P={w[i]:.4e}")

# ── Top-10 trigrams ────────────────────────────────────────────────────────────
print("\n=== Top-10 trigrams by P(w1,w2,w3) ===")
yax_mask = phase == 2
scores   = np.where(yax_mask, w, 0.0)
for i in np.argsort(scores)[::-1][:10]:
    print(f"  {labels[i]:50s}  P={w[i]:.4e}")

# ── Nearest neighbors to "the" ─────────────────────────────────────────────────
print('\n=== Spatial nearest neighbors to "the" (Pop layer only) ===')
the_id  = labels.index("the")
the_pos = pos[the_id]
pop_idx = np.where(pop_mask)[0]
dists   = np.linalg.norm(pos[pop_idx] - the_pos, axis=1)
nn_order = np.argsort(dists)
phase_names = meta["phase_map"]
for rank, ii in enumerate(nn_order[:11]):
    nid = int(pop_idx[ii])
    if nid == the_id:
        continue
    print(f"  #{rank:02d}  {labels[nid]:25s}  dist={dists[ii]:.5f}  P={w[nid]:.3e}")
    if rank >= 10:
        break

# ── Highest conditional probability edges ─────────────────────────────────────
print("\n=== Highest conditional probability edges (top-10) ===")
for e in np.argsort(ew)[::-1][:10]:
    src, tgt = int(ei[0, e]), int(ei[1, e])
    print(f"  {labels[src]:28s} -> {labels[tgt]:40s}  P(tgt|src)={ew[e]:.6f}")

# ── Query: out-edges from a word ───────────────────────────────────────────────
def out_edges(word: str, top_k: int = 10):
    if word not in labels:
        print(f"  '{word}' not in vocabulary")
        return
    nid  = labels.index(word)
    mask = ei[0] == nid
    if mask.sum() == 0:
        print(f"  no outgoing edges from '{word}'")
        return
    idx = np.where(mask)[0]
    top = idx[np.argsort(ew[idx])[::-1][:top_k]]
    for e in top:
        tgt = int(ei[1, e])
        print(f"  {labels[nid]:20s} -> {labels[tgt]:35s}  P={ew[e]:.6f}")

print('\n=== Out-edges from "of" (top-10 by conditional probability) ===')
out_edges("of")

print('\n=== Out-edges from "the" ===')
out_edges("the")
