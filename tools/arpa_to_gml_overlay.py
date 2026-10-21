#!/usr/bin/env python3
"""Convert an ARPA language model into a three-layer KHANARY GML overlay.

Each ngram width lands in its canonical K'UHUL phase — carrying a different
semantic role that matches how the attention kernel (attention.hlsl) scores:

  Unigram  Pop  z=0.000  static_weight = P(w)          ← relevance mass
  Bigram   Wo   z=0.167  static_weight = P(w1 w2)      ← relativity + binding
                         edge weight   = P(w2|w1)
  Trigram  Yax  z=0.333  static_weight = P(w1 w2 w3)   ← sequence probability
                         edge weight   = P(w3|w1,w2)

The three layers preserve the SRILM conditional structure geometrically:
  - word → position (lookup by hash)
  - position → word (nearest-neighbor query in FieldDag)
  - P(sequence) → WCR W component (static_weight in each phase layer)

ARPA probabilities are log10; we convert to linear (P = 10^log_prob) and clamp
to [0, 1] so all weights satisfy the GML validator (W1-W4 rules).

Backoff weights from the ARPA header are stored as node attribute `backoff` for
future use (smoothing recall in missed lookups) but do not affect GML validity.

Usage:
  py -3 tools\\arpa_to_gml_overlay.py tools\\mx2lm.arpa
  py -3 tools\\arpa_to_gml_overlay.py tools\\mx2lm.arpa -o data\\mx2lm.vectors.gml
  py -3 tools\\arpa_to_gml_overlay.py tools\\mx2lm.arpa --min-prob 1e-5 --max-nodes 500000

Output:
  data\\mx2lm.vectors.gml  (default)  — loadable by FieldDag::load_gml()

After generation, register the GML in vectors.xml:
  <topology id="mx2lm_vectors" source="data/mx2lm.vectors.gml"
            graph="arpa-3layer-trigram" nodes="N" edges="E" directed="1"
            role="SRILM trigram LM: relevance(Pop) relativity(Wo) probability(Yax)" />
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

PHASE_Z   = {"Pop": 0.000, "Wo": 0.167, "Yax": 0.333, "Sek": 0.500}
SKIP_TOKENS = frozenset(["<unk>", "<s>", "</s>", "<UNK>"])
DEFAULT_CONF   = 0.5
DEFAULT_HEALTH = 1.0


# ── Hash → 2D coordinates ─────────────────────────────────────────────────────

def _fnv1a32(s: str) -> int:
    h = 0x811C9DC5
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def key_to_xy(key: str) -> tuple[float, float]:
    h = _fnv1a32(key)
    x = (h & 0xFFFF) / 65535.0
    y = ((h >> 16) & 0xFFFF) / 65535.0
    return x, y


# ── ARPA parser ───────────────────────────────────────────────────────────────

class ArpaModel:
    """Parsed ARPA language model.

    unigrams : {word: (log_prob, backoff)}
    bigrams  : {(w1, w2): (log_prob, backoff)}
    trigrams : {(w1, w2, w3): log_prob}
    """

    def __init__(self):
        self.unigrams: dict[str, tuple[float, float]] = {}
        self.bigrams:  dict[tuple, tuple[float, float]] = {}
        self.trigrams: dict[tuple, float] = {}

    @staticmethod
    def _parse_line(line: str, order: int):
        """Return (ngram_tuple, log_prob, backoff) or None on skip."""
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

    def load(self, path: Path,
             min_prob: float = 0.0,
             max_per_order: int = 0) -> None:
        """Parse an ARPA file into the three dicts."""
        current_order = 0
        counts: dict[int, int] = {}

        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")

                if line.startswith("\\data\\"):
                    current_order = -1
                    continue

                if line.startswith("ngram "):
                    try:
                        _, rest = line.split(None, 1)
                        o, n = rest.split("=")
                        counts[int(o.strip())] = int(n.strip())
                    except ValueError:
                        pass
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

                # min_prob filter
                if min_prob > 0.0 and 10.0 ** log_prob < min_prob:
                    continue

                # max_per_order cap (0 = unlimited)
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


# ── GML writer ────────────────────────────────────────────────────────────────

def _gml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

def _linear(log_prob: float) -> float:
    """Convert log10 probability to linear, clamped to [1e-9, 1]."""
    p = 10.0 ** log_prob
    return max(1e-9, min(1.0, p))


def build_and_write(model: ArpaModel, out_path: Path) -> tuple[int, int]:
    """Build GML from the three ARPA layers and write to out_path.

    Returns (node_count, edge_count).
    """
    # ── Assign IDs ─────────────────────────────────────────────────────────
    # Nodes: unigram words + bigram phrases + trigram phrases
    # Keyed by the surface string for dedup and edge lookup.

    node_id: dict[str, int] = {}
    next_id = 0

    # 1. Pop-layer unigram nodes
    for word in model.unigrams:
        node_id[word] = next_id
        next_id += 1

    # 2. Wo-layer bigram nodes (surface = "w1 w2")
    for (w1, w2) in model.bigrams:
        surface = w1 + " " + w2
        if surface not in node_id:
            node_id[surface] = next_id
            next_id += 1

    # 3. Yax-layer trigram nodes (surface = "w1 w2 w3")
    for (w1, w2, w3) in model.trigrams:
        surface = w1 + " " + w2 + " " + w3
        if surface not in node_id:
            node_id[surface] = next_id
            next_id += 1

    # ── Write GML ──────────────────────────────────────────────────────────
    edge_count = 0

    with out_path.open("w", encoding="utf-8") as fh:
        w = fh.write

        w("graph [\n")
        w('  comment "ARPA trigram LM three-layer overlay: '
          'Pop=relevance Wo=relativity Yax=probability"\n')
        w('  comment "static_weight = P(ngram); edge weight = P(w_n | context)"\n')
        w('  comment "generated by tools/arpa_to_gml_overlay.py"\n')
        w("  directed 1\n")
        w('  label "mx2lm_vectors"\n')
        w("\n")

        # ── Pop layer: unigrams ────────────────────────────────────────────
        for word, (log_prob, backoff) in model.unigrams.items():
            nid = node_id[word]
            x, y = key_to_xy(word)
            p    = _linear(log_prob)
            w("  node [\n")
            w(f"    id {nid}\n")
            w(f"    label {_gml_str(word)}\n")
            w(f'    provider_id "mx2lm-unigram"\n')
            w(f'    phase "Pop"\n')
            w(f"    x {x:.6f}  y {y:.6f}  z 0.000\n")
            w(f"    static_weight {p:.8f}\n")
            w(f"    confidence {DEFAULT_CONF:.1f}\n")
            w(f"    health {DEFAULT_HEALTH:.1f}\n")
            w(f"    local_entropy 0.0\n")
            w(f"    geodesic_entropy 0.0\n")
            w(f"    divergence 0.0\n")
            w(f'    singularity "ordinary"\n')
            if backoff != 0.0:
                w(f"    backoff {backoff:.6f}\n")
            w("  ]\n")

        # ── Wo layer: bigrams + conditional edges ─────────────────────────
        # ARPA log_prob IS P(w2|w1); static_weight = joint P(w1,w2) via chain rule
        for (w1, w2), (log_prob, backoff) in model.bigrams.items():
            surface  = w1 + " " + w2
            nid      = node_id[surface]
            x, y     = key_to_xy(surface)
            p_cond   = _linear(log_prob)                                    # P(w2|w1)
            p_w1     = _linear(model.unigrams[w1][0]) if w1 in model.unigrams else p_cond
            p_joint  = min(p_w1 * p_cond, 1.0)                             # P(w1,w2)

            w("  node [\n")
            w(f"    id {nid}\n")
            w(f"    label {_gml_str(surface)}\n")
            w(f'    provider_id "mx2lm-bigram"\n')
            w(f'    phase "Wo"\n')
            w(f"    x {x:.6f}  y {y:.6f}  z 0.167\n")
            w(f"    static_weight {p_joint:.8f}\n")
            w(f"    confidence {DEFAULT_CONF:.1f}\n")
            w(f"    health {DEFAULT_HEALTH:.1f}\n")
            w(f"    local_entropy 0.0\n")
            w(f"    geodesic_entropy 0.0\n")
            w(f"    divergence 0.0\n")
            w(f'    singularity "ordinary"\n')
            if backoff != 0.0:
                w(f"    backoff {backoff:.6f}\n")
            w("  ]\n")

            # Conditional edge: w1 → "w1 w2"  weight = P(w2|w1) directly from ARPA
            src_id = node_id.get(w1)
            if src_id is not None:
                w("  edge [\n")
                w(f"    source {src_id}  target {nid}\n")
                w(f'    verb "@ngram_context"\n')
                w(f"    weight {p_cond:.8f}\n")
                w("  ]\n")
                edge_count += 1

        # ── Yax layer: trigrams + conditional edges ───────────────────────
        # ARPA log_prob IS P(w3|w1,w2); static_weight = joint P(w1,w2,w3) via chain rule
        for (w1, w2, w3), log_prob in model.trigrams.items():
            surface    = w1 + " " + w2 + " " + w3
            bigram_key = w1 + " " + w2
            nid        = node_id[surface]
            x, y       = key_to_xy(surface)
            p_cond     = _linear(log_prob)                                  # P(w3|w1,w2)
            bg_entry   = model.bigrams.get((w1, w2))
            p_bg       = _linear(bg_entry[0]) if bg_entry else p_cond      # P(w2|w1)
            p_w1       = _linear(model.unigrams[w1][0]) if w1 in model.unigrams else None
            p_joint    = min((p_w1 * p_bg if p_w1 is not None else p_bg) * p_cond, 1.0)

            w("  node [\n")
            w(f"    id {nid}\n")
            w(f"    label {_gml_str(surface)}\n")
            w(f'    provider_id "mx2lm-trigram"\n')
            w(f'    phase "Yax"\n')
            w(f"    x {x:.6f}  y {y:.6f}  z 0.333\n")
            w(f"    static_weight {p_joint:.8f}\n")
            w(f"    confidence {DEFAULT_CONF:.1f}\n")
            w(f"    health {DEFAULT_HEALTH:.1f}\n")
            w(f"    local_entropy 0.0\n")
            w(f"    geodesic_entropy 0.0\n")
            w(f"    divergence 0.0\n")
            w(f'    singularity "ordinary"\n')
            w("  ]\n")

            # Conditional edge: "w1 w2" → "w1 w2 w3"  weight = P(w3|w1,w2) directly from ARPA
            src_id = node_id.get(bigram_key)
            if src_id is not None:
                w("  edge [\n")
                w(f"    source {src_id}  target {nid}\n")
                w(f'    verb "@ngram_context"\n')
                w(f"    weight {p_cond:.8f}\n")
                w("  ]\n")
                edge_count += 1

        w("]\n")

    return len(node_id), edge_count


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arpa", type=Path,
                    help="Path to .arpa file (e.g. tools/mx2lm.arpa)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output .gml path (default: data/mx2lm.vectors.gml)")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="Skip ngrams with P < this threshold (e.g. 1e-5)")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="Cap nodes per order for dev/testing (0 = unlimited)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out  = args.output or (root / "data" / "mx2lm.vectors.gml")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.arpa.exists():
        print(f"ERROR: ARPA file not found: {args.arpa}", file=sys.stderr)
        print("       Run: tools\\train_srilm.bat", file=sys.stderr)
        return 1

    print(f"Loading {args.arpa} …", file=sys.stderr)
    model = ArpaModel()
    model.load(args.arpa, min_prob=args.min_prob, max_per_order=args.max_nodes)

    print(f"Writing {out} …", file=sys.stderr)
    node_count, edge_count = build_and_write(model, out)

    size_kb = out.stat().st_size // 1024
    print(f"  wrote {out}  ({node_count:,} nodes, {edge_count:,} edges, {size_kb:,} KB)",
          file=sys.stderr)

    print(f"\nAdd to vectors.xml <gml_topology_corpus>:", file=sys.stderr)
    print(f'  <topology id="mx2lm_vectors" source="data/mx2lm.vectors.gml"', file=sys.stderr)
    print(f'            graph="arpa-3layer-trigram"', file=sys.stderr)
    print(f'            nodes="{node_count}" edges="{edge_count}" directed="1"', file=sys.stderr)
    print(f'            role="SRILM trigram LM: relevance(Pop) relativity(Wo) probability(Yax)" />', file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
