#!/usr/bin/env python3
"""Convert an ARPA language model into a SafeTensors graph dataset.

Same three-layer semantic structure as arpa_to_gml_overlay.py, but output is
compact binary tensors (~12x smaller than GML text) that are memory-mappable
and directly loadable into PyTorch / NumPy / HuggingFace pipelines.

Layout:

  Pop   (phase 0, z=0.000)  unigrams  — relevance mass
  Wo    (phase 1, z=0.167)  bigrams   — relativity / binding
  Yax   (phase 2, z=0.333)  trigrams  — sequence probability

Tensors written to <output>.safetensors:

  node_positions  float32 [N, 3]   (x, y, z)  FNV-1a hash → (x,y); phase → z
  node_weights    float32 [N]      P(ngram)    static_weight / WCR W component
  node_phase      int32   [N]      0=Pop 1=Wo 2=Yax 3=Sek 4=Ch'en 5=Xul
  edge_index      int64   [2, E]   (source_ids, target_ids) COO format
  edge_weights    float32 [E]      P(w_n | context) conditional probability

Sidecar files:

  <output>.labels.json   list of node label strings, index → label
  <output>.meta.json     counts, phase map, ARPA path, filter params

Dependencies:
  pip install numpy safetensors

Usage:
  py -3 tools\\arpa_to_safetensors.py tools\\mx2lm.arpa
  py -3 tools\\arpa_to_safetensors.py tools\\mx2lm.arpa -o data\\mx2lm.vectors.safetensors
  py -3 tools\\arpa_to_safetensors.py tools\\mx2lm.arpa --min-prob 1e-6

Loading in Python:
  from safetensors.numpy import load_file
  data = load_file("data/mx2lm.vectors.safetensors")
  positions = data["node_positions"]   # float32 [N, 3]
  weights   = data["node_weights"]     # float32 [N]

Loading in PyTorch:
  from safetensors.torch import load_file
  data = load_file("data/mx2lm.vectors.safetensors")
  # tensors are already on CPU as torch.Tensor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not found — pip install numpy", file=sys.stderr)
    raise SystemExit(1)

try:
    from safetensors.numpy import save_file as st_save
except ImportError:
    print("ERROR: safetensors not found — pip install safetensors", file=sys.stderr)
    raise SystemExit(1)


# ── Constants ─────────────────────────────────────────────────────────────────

PHASE_INT = {"Pop": 0, "Wo": 1, "Yax": 2, "Sek": 3, "Ch'en": 4, "Xul": 5}
PHASE_Z   = {0: 0.000, 1: 0.167, 2: 0.333, 3: 0.500, 4: 0.667, 5: 0.833}
SKIP_TOKENS = frozenset(["<unk>", "<s>", "</s>", "<UNK>"])


# ── Hash → 2D coordinates ─────────────────────────────────────────────────────

def _fnv1a32(s: str) -> int:
    h = 0x811C9DC5
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def key_to_xy(key: str) -> tuple[float, float]:
    h = _fnv1a32(key)
    return (h & 0xFFFF) / 65535.0, ((h >> 16) & 0xFFFF) / 65535.0


# ── ARPA parser ───────────────────────────────────────────────────────────────

class ArpaModel:
    def __init__(self):
        self.unigrams: dict[str, tuple[float, float]] = {}   # word → (log_prob, backoff)
        self.bigrams:  dict[tuple, tuple[float, float]] = {}
        self.trigrams: dict[tuple, float] = {}

    @staticmethod
    def _parse_line(line: str, order: int):
        parts = line.split()
        if not parts:
            return None
        try:
            log_prob = float(parts[0])
        except ValueError:
            return None
        if len(parts) < order + 1:
            return None
        words = tuple(parts[1:order + 1])
        if any(w in SKIP_TOKENS for w in words):
            return None
        backoff = 0.0
        if len(parts) > order + 1:
            try:
                backoff = float(parts[order + 1])
            except ValueError:
                pass
        return words, log_prob, backoff

    def load(self, path: Path, min_prob: float = 0.0, max_per_order: int = 0) -> None:
        current_order = 0
        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("\\data\\"):
                    current_order = -1
                    continue
                if line.startswith("ngram "):
                    continue
                if line.startswith("\\") and line.endswith("-grams:"):
                    try:
                        current_order = int(line[1:line.index("-")])
                    except ValueError:
                        current_order = 0
                    continue
                if line.startswith("\\end\\"):
                    break
                if current_order < 1 or current_order > 3:
                    continue
                parsed = self._parse_line(line, current_order)
                if parsed is None:
                    continue
                words, log_prob, backoff = parsed
                if min_prob > 0.0 and 10.0 ** log_prob < min_prob:
                    continue
                if max_per_order > 0:
                    if current_order == 1 and len(self.unigrams) >= max_per_order:
                        continue
                    if current_order == 2 and len(self.bigrams) >= max_per_order:
                        continue
                    if current_order == 3 and len(self.trigrams) >= max_per_order:
                        continue
                if current_order == 1:
                    self.unigrams[words[0]] = (log_prob, backoff)
                elif current_order == 2:
                    self.bigrams[words] = (log_prob, backoff)
                elif current_order == 3:
                    self.trigrams[words] = log_prob

        print(f"  loaded: {len(self.unigrams):,} unigrams  "
              f"{len(self.bigrams):,} bigrams  "
              f"{len(self.trigrams):,} trigrams", file=sys.stderr)


# ── Convert to tensors ────────────────────────────────────────────────────────

def _linear(log_prob: float) -> float:
    return max(1e-9, min(1.0, 10.0 ** log_prob))


def build_tensors(model: ArpaModel) -> tuple[dict[str, np.ndarray], list[str], dict]:
    """Build numpy arrays from the three ARPA layers.

    Returns:
        tensors  — dict ready for safetensors save_file()
        labels   — list of node label strings (index → string)
        meta     — metadata dict for the sidecar JSON
    """
    # ── Assign IDs ─────────────────────────────────────────────────────────
    node_id: dict[str, int] = {}

    for word in model.unigrams:
        node_id[word] = len(node_id)

    for (w1, w2) in model.bigrams:
        surface = w1 + " " + w2
        if surface not in node_id:
            node_id[surface] = len(node_id)

    for (w1, w2, w3) in model.trigrams:
        surface = w1 + " " + w2 + " " + w3
        if surface not in node_id:
            node_id[surface] = len(node_id)

    N = len(node_id)
    print(f"  {N:,} nodes total", file=sys.stderr)

    # ── Pre-allocate node arrays ────────────────────────────────────────────
    positions = np.zeros((N, 3), dtype=np.float32)
    weights   = np.zeros(N, dtype=np.float32)
    phases    = np.zeros(N, dtype=np.int32)

    # Pop layer
    for word, (log_prob, _) in model.unigrams.items():
        nid = node_id[word]
        x, y = key_to_xy(word)
        positions[nid] = (x, y, 0.000)
        weights[nid]   = _linear(log_prob)
        phases[nid]    = 0  # Pop

    # Wo layer — log_prob IS P(w2|w1); static_weight = joint P(w1,w2) via chain rule
    for (w1, w2), (log_prob, _) in model.bigrams.items():
        surface  = w1 + " " + w2
        nid      = node_id[surface]
        x, y     = key_to_xy(surface)
        p_cond   = _linear(log_prob)                                        # P(w2|w1)
        p_w1     = _linear(model.unigrams[w1][0]) if w1 in model.unigrams else p_cond
        positions[nid] = (x, y, 0.167)
        weights[nid]   = np.float32(min(p_w1 * p_cond, 1.0))               # P(w1,w2)
        phases[nid]    = 1  # Wo

    # Yax layer — log_prob IS P(w3|w1,w2); static_weight = joint P(w1,w2,w3) via chain rule
    for (w1, w2, w3), log_prob in model.trigrams.items():
        surface  = w1 + " " + w2 + " " + w3
        nid      = node_id[surface]
        x, y     = key_to_xy(surface)
        p_cond   = _linear(log_prob)                                        # P(w3|w1,w2)
        bg_entry = model.bigrams.get((w1, w2))
        p_bg     = _linear(bg_entry[0]) if bg_entry else p_cond             # P(w2|w1)
        p_w1     = _linear(model.unigrams[w1][0]) if w1 in model.unigrams else None
        p_joint  = min((p_w1 * p_bg if p_w1 is not None else p_bg) * p_cond, 1.0)
        positions[nid] = (x, y, 0.333)
        weights[nid]   = np.float32(p_joint)                                # P(w1,w2,w3)
        phases[nid]    = 2  # Yax

    # ── Build edge arrays ───────────────────────────────────────────────────
    # Pre-count edges to avoid repeated appends
    edge_count = 0
    for (w1, w2) in model.bigrams:
        if w1 in node_id:
            edge_count += 1
    for (w1, w2, w3) in model.trigrams:
        if (w1 + " " + w2) in node_id:
            edge_count += 1

    print(f"  {edge_count:,} edges total", file=sys.stderr)

    src_arr = np.zeros(edge_count, dtype=np.int64)
    tgt_arr = np.zeros(edge_count, dtype=np.int64)
    ew_arr  = np.zeros(edge_count, dtype=np.float32)
    ei = 0

    # Wo edges: w1 → "w1 w2",  weight = P(w2|w1) — ARPA value IS the conditional
    for (w1, w2), (log_prob, _) in model.bigrams.items():
        src_id = node_id.get(w1)
        if src_id is None:
            continue
        tgt_id      = node_id[w1 + " " + w2]
        src_arr[ei] = src_id
        tgt_arr[ei] = tgt_id
        ew_arr[ei]  = np.float32(_linear(log_prob))   # P(w2|w1) directly from ARPA
        ei += 1

    # Yax edges: "w1 w2" → "w1 w2 w3",  weight = P(w3|w1,w2) — ARPA value IS the conditional
    for (w1, w2, w3), log_prob in model.trigrams.items():
        bigram_key = w1 + " " + w2
        src_id = node_id.get(bigram_key)
        if src_id is None:
            continue
        tgt_id      = node_id[w1 + " " + w2 + " " + w3]
        src_arr[ei] = src_id
        tgt_arr[ei] = tgt_id
        ew_arr[ei]  = np.float32(_linear(log_prob))   # P(w3|w1,w2) directly from ARPA
        ei += 1

    # Stack into COO [2, E]
    edge_index = np.stack([src_arr[:ei], tgt_arr[:ei]], axis=0)

    # ── Labels sidecar ─────────────────────────────────────────────────────
    labels = [""] * N
    for label, nid in node_id.items():
        labels[nid] = label

    # ── Metadata ────────────────────────────────────────────────────────────
    meta = {
        "format":       "arpa-3layer-trigram",
        "node_count":   N,
        "edge_count":   int(ei),
        "phase_map":    {"0": "Pop", "1": "Wo", "2": "Yax",
                         "3": "Sek", "4": "Ch'en", "5": "Xul"},
        "z_map":        {"0": 0.000, "1": 0.167, "2": 0.333,
                         "3": 0.500, "4": 0.667, "5": 0.833},
        "layer_counts": {
            "Pop":  len(model.unigrams),
            "Wo":   len(model.bigrams),
            "Yax":  len(model.trigrams),
        },
        "tensors": {
            "node_positions": "float32 [N, 3]  (x, y, z) — FNV-1a hash → (x,y); phase → z",
            "node_weights":   "float32 [N]     P(ngram) — WCR static_weight W component",
            "node_phase":     "int32   [N]     0=Pop 1=Wo 2=Yax 3=Sek 4=Ch'en 5=Xul",
            "edge_index":     "int64   [2, E]  (source, target) COO sparse format",
            "edge_weights":   "float32 [E]     P(w_n | context) conditional probability",
        },
        "description": (
            "SRILM trigram LM three-layer overlay: "
            "Pop=relevance(P(w)), Wo=relativity(P(w1,w2)), Yax=probability(P(w1,w2,w3)). "
            "Edge weights = P(w_n | context). Coordinates are FNV-1a hash projections. "
            "Labels in companion *.labels.json, index aligned with tensor rows."
        ),
    }

    tensors = {
        "node_positions": positions,
        "node_weights":   weights,
        "node_phase":     phases,
        "edge_index":     edge_index,
        "edge_weights":   ew_arr[:ei],
    }

    return tensors, labels, meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arpa", type=Path,
                    help="Path to .arpa file (e.g. tools/mx2lm.arpa)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output .safetensors path (default: data/mx2lm.vectors.safetensors)")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="Skip ngrams with P < this threshold (e.g. 1e-6)")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="Cap nodes per order for dev/testing (0 = unlimited)")
    ap.add_argument("--no-labels", action="store_true",
                    help="Skip writing the labels sidecar (saves ~100 MB RAM peak)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out  = args.output or (root / "data" / "mx2lm.vectors.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.arpa.exists():
        print(f"ERROR: ARPA file not found: {args.arpa}", file=sys.stderr)
        print("       Run: tools\\train_srilm.bat", file=sys.stderr)
        return 1

    print(f"Loading {args.arpa} …", file=sys.stderr)
    model = ArpaModel()
    model.load(args.arpa, min_prob=args.min_prob, max_per_order=args.max_nodes)

    print("Building tensors …", file=sys.stderr)
    tensors, labels, meta = build_tensors(model)

    print(f"Writing {out} …", file=sys.stderr)
    st_save(tensors, str(out))

    labels_path = out.with_suffix(".labels.json")
    meta_path   = out.with_suffix(".meta.json")

    if not args.no_labels:
        labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    st_kb     = out.stat().st_size // 1024
    labels_kb = labels_path.stat().st_size // 1024 if not args.no_labels else 0

    N = meta["node_count"]
    E = meta["edge_count"]
    print(f"\n  {out.name}         {st_kb:>10,} KB  ({N:,} nodes, {E:,} edges)",
          file=sys.stderr)
    if not args.no_labels:
        print(f"  {labels_path.name}  {labels_kb:>10,} KB  (label strings)",
              file=sys.stderr)
    print(f"  {meta_path.name}    (schema / counts / phase map)", file=sys.stderr)

    print("\nLoading example:", file=sys.stderr)
    print("  from safetensors.numpy import load_file", file=sys.stderr)
    print(f'  d = load_file("{out.name}")', file=sys.stderr)
    print("  d['node_positions']  # float32 [N, 3]", file=sys.stderr)
    print("  d['node_weights']    # float32 [N]", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
