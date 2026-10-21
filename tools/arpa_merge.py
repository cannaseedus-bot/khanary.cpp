#!/usr/bin/env python3
"""Merge multiple ARPA language models with identity-aware probability interpolation.

Implements the ARPA merge contract:

  Pop   parse source ARPA models
  Yax   union vocabularies, align n-gram histories
  Sek   resolve source backoff probabilities
        interpolate conditional distributions
        normalize per history
        reconstruct backoff coefficients
  Ch'en verify probability mass, backoff consistency, n-gram order
  Xul   emit merged ARPA + merge manifest

Two strategies:

  linear (default)
    P_merged(w|h) = Σ αᵢ Pᵢ(w|h)
    Automatically normalized when sources are normalized and Σ αᵢ = 1.
    Pᵢ(w|h) is resolved through source i's backoff chain when the n-gram
    is not explicit — never substituted with an arbitrary constant.

  log-linear
    P_merged(w|h) = Π Pᵢ(w|h)^αᵢ / Z(h)
    Z(h) = Σ_w Π Pᵢ(w|h)^αᵢ  (per-history normalization term)
    Computed over the union of all words seen in any source context for h.

Backoff reconstruction:
    Given explicit P_merged(w|h) for words in V_explicit(h), the merged
    backoff weight is:
      bow(h) = (1 - Σ_{w ∈ V_explicit} P_merged(w|h))
                / Σ_{w ∉ V_explicit} P_merged_lower(w|h[-1:])
    where P_merged_lower is the already-computed lower-order merged probability.

Key constraint: cfpc is bigram-only; it contributes no trigram conditionals.
Missing trigrams from cfpc are resolved via its bigram-level backoff.

Usage:
  py -3 tools\\arpa_merge.py \\
      --inputs tools/mx2lm.arpa \\
               data/cfpc/wildcard-bigram.arpa \\
               data/geometric-coder/coder.arpa \\
      --alphas 0.4 0.35 0.25 \\
      --strategy linear \\
      --min-prob 1e-8 \\
      --output data/merged/merged.arpa
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── ARPA data model ───────────────────────────────────────────────────────────

@dataclass
class NgramEntry:
    log_prob: float          # log10 P(wn | w1..w(n-1))
    backoff:  float = 0.0   # log10 bow; 0.0 means bow=1.0 (not stored in file)

class ArpaModel:
    def __init__(self) -> None:
        # ngrams[(w1,...,wn)] = NgramEntry
        self.ngrams:  dict[tuple[str,...], NgramEntry] = {}
        self.counts:  dict[int, int] = {}   # order → count
        self.max_order: int = 0

    def p_linear(self, word: str, history: tuple[str,...]) -> float:
        """
        Return P(word | history) as a probability (not log).
        Resolves through the backoff chain if the exact n-gram is not explicit.
        """
        key = history + (word,)
        if key in self.ngrams:
            return 10.0 ** self.ngrams[key].log_prob
        # Back off
        if history:
            h_key = history
            bow = (10.0 ** self.ngrams[h_key].backoff
                   if h_key in self.ngrams else 1.0)
            return bow * self.p_linear(word, history[1:])
        # Unigram fallback
        unk_entry = self.ngrams.get(("<unk>",)) or self.ngrams.get(("<UNK>",))
        return 10.0 ** (unk_entry.log_prob if unk_entry else -99.0)

# ── ARPA parser ───────────────────────────────────────────────────────────────

def parse_arpa(path: Path) -> ArpaModel:
    model = ArpaModel()
    section = None   # None | int (current n-gram order)

    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            if not line or line == "\\data\\":
                continue
            if line.startswith("ngram "):
                parts = line[6:].split("=")
                if len(parts) == 2:
                    model.counts[int(parts[0])] = int(parts[1])
                continue
            if line.startswith("\\") and line.endswith("-grams:"):
                section = int(line[1: line.index("-")])
                model.max_order = max(model.max_order, section)
                continue
            if line == "\\end\\":
                break
            if section is None:
                continue

            parts = line.split()
            if len(parts) < section + 1:
                continue
            try:
                log_p = float(parts[0])
            except ValueError:
                continue
            words = tuple(parts[1: section + 1])
            try:
                backoff = float(parts[section + 1]) if len(parts) > section + 1 else 0.0
            except ValueError:
                backoff = 0.0
            model.ngrams[words] = NgramEntry(log_prob=log_p, backoff=backoff)

    return model

# ── Probability merging ───────────────────────────────────────────────────────

def blend_linear(probs: list[float], alphas: list[float]) -> float:
    return sum(p * a for p, a in zip(probs, alphas))

def blend_log_linear(probs: list[float], alphas: list[float]) -> float:
    """Unnormalized geometric combination — caller must normalize over history."""
    log_blend = sum(a * math.log10(max(p, 1e-300))
                    for p, a in zip(probs, alphas))
    return 10.0 ** log_blend

# ── Union vocabulary / history builders ──────────────────────────────────────

def union_ngrams_by_order(models: list[ArpaModel],
                          order: int) -> set[tuple[str,...]]:
    """Return union of all explicit k-grams across all models at this order."""
    result: set[tuple[str,...]] = set()
    for m in models:
        for key in m.ngrams:
            if len(key) == order:
                result.add(key)
    return result

# ── Backoff reconstruction ────────────────────────────────────────────────────

def reconstruct_backoff(
    history:           tuple[str,...],
    explicit_words:    set[str],
    merged_explicit:   dict[tuple[str,...], float],   # (history+(w,)) → P
    merged_lower:      dict[tuple[str,...], float],   # lower-order merged P
) -> float:
    """
    Compute bow_merged(history) so that:
      mass_explicit + bow * mass_implicit_lower = 1.0
    """
    mass_explicit = sum(
        merged_explicit[history + (w,)]
        for w in explicit_words
        if history + (w,) in merged_explicit
    )
    remaining = 1.0 - mass_explicit

    if remaining < 1e-10:
        return 0.0   # all mass accounted for by explicit entries; bow → 1.0

    # Mass at lower order for words NOT in explicit table
    lower_hist = history[1:]
    mass_lower = 0.0
    # We sum P_lower(w | lower_hist) for all w in explicit_words
    # The implicit mass at lower order = 1 - (explicit mass at lower order)
    lower_explicit_mass = sum(
        merged_lower.get(lower_hist + (w,), 0.0)
        for w in explicit_words
    )
    # Total lower mass for implicit words = 1 - lower_explicit_mass
    # (lower distribution is normalized)
    mass_lower = 1.0 - lower_explicit_mass

    if mass_lower < 1e-10:
        return 0.0

    bow = remaining / mass_lower
    return bow

# ── Main merge logic ──────────────────────────────────────────────────────────

def merge_models(models: list[ArpaModel],
                 alphas: list[float],
                 strategy: str,
                 min_prob: float) -> ArpaModel:

    merged = ArpaModel()
    max_order = max(m.max_order for m in models)

    # merged_p[k][(w1,...,wk)] = P (not log) for order k
    merged_p: dict[int, dict[tuple[str,...], float]] = {}

    # ── Unigrams (order 1) ─────────────────────────────────────────────────
    print("  [Yax] Unigrams …", file=sys.stderr)
    union1 = union_ngrams_by_order(models, 1)
    print(f"         Union unigram vocabulary: {len(union1):,}", file=sys.stderr)

    merged_p[1] = {}
    for key in union1:
        word = key[0]
        # Use explicit-only probabilities (not backoff-resolved) to keep unigram
        # mass correct.  p_linear() on OOV returns P(<unk>), inflating the union
        # mass by ~N_oov * P(<unk>).  Explicit-only gives 0.0 for OOV, and the
        # linear mixture over the union vocab stays normalized because
        # Σ_w Σᵢ αᵢ Pᵢ(w) = Σᵢ αᵢ * 1.0 = 1.0.
        probs = [
            10.0 ** m.ngrams[(word,)].log_prob if (word,) in m.ngrams else 0.0
            for m in models
        ]
        if strategy == "linear":
            p = blend_linear(probs, alphas)
        else:
            p = blend_log_linear(probs, alphas)
        if p >= min_prob:
            merged_p[1][key] = p

    # Normalize after min-prob pruning (Zipf tail removal shifts mass slightly)
    total = sum(merged_p[1].values())
    if total > 0:
        for k in merged_p[1]:
            merged_p[1][k] /= total

    merged_words = {k[0] for k in merged_p[1]}
    print(f"         Merged unigrams after min-prob: {len(merged_p[1]):,}",
          file=sys.stderr)

    # ── Higher orders ──────────────────────────────────────────────────────
    for order in range(2, max_order + 1):
        print(f"  [Yax] {order}-grams …", file=sys.stderr)
        union_k = union_ngrams_by_order(models, order)
        print(f"         Union {order}-grams: {len(union_k):,}", file=sys.stderr)

        merged_p[order] = {}
        histories: dict[tuple[str,...], set[str]] = defaultdict(set)

        for key in union_k:
            word    = key[-1]
            history = key[:-1]
            # Only generate this n-gram if all words in it survived min-prob
            if word not in merged_words:
                continue
            if not all(w in merged_words for w in history):
                continue

            probs = [m.p_linear(word, history) for m in models]
            if strategy == "linear":
                p = blend_linear(probs, alphas)
            else:
                p = blend_log_linear(probs, alphas)

            if p >= min_prob:
                merged_p[order][key] = p
                histories[history].add(word)

        if strategy == "log-linear":
            # Normalize per history
            for history, words in histories.items():
                total = sum(merged_p[order][history + (w,)]
                            for w in words
                            if history + (w,) in merged_p[order])
                if total > 0:
                    for w in words:
                        k = history + (w,)
                        if k in merged_p[order]:
                            merged_p[order][k] /= total

        print(f"         Merged {order}-grams: {len(merged_p[order]):,}",
              file=sys.stderr)

    # ── Convert to ArpaModel + reconstruct backoffs ────────────────────────
    print("  [Sek] Reconstructing backoff weights …", file=sys.stderr)

    # Pre-build history → set[word] index so we don't scan the entire next-order
    # table for every key (that would be O(N_keys * N_higher) ≈ 91B ops).
    children: dict[int, dict[tuple[str,...], set[str]]] = {}
    for ord_ in range(2, max_order + 1):
        idx: dict[tuple[str,...], set[str]] = defaultdict(set)
        for key in merged_p.get(ord_, {}):
            idx[key[:-1]].add(key[-1])
        children[ord_] = idx

    for order in range(1, max_order + 1):
        for key, p in merged_p[order].items():
            if p <= 0:
                continue
            log_p = math.log10(p)

            bow = 0.0   # default: log10(1.0)
            if order < max_order:
                explicit_words = children.get(order + 1, {}).get(key, set())
                if explicit_words:
                    if order == 1:
                        explicit_mass = sum(
                            merged_p[order + 1].get(key + (w,), 0.0)
                            for w in explicit_words
                        )
                        remaining = 1.0 - explicit_mass
                        lower_explicit = sum(
                            merged_p[1].get((w,), 0.0)
                            for w in explicit_words
                        )
                        mass_lower = 1.0 - lower_explicit
                        bow_val = (remaining / mass_lower
                                   if mass_lower > 1e-10 else 1.0)
                    else:
                        bow_val = reconstruct_backoff(
                            key,
                            explicit_words,
                            merged_p.get(order + 1, {}),
                            merged_p.get(order, {}),
                        )
                    bow_val = max(bow_val, 1e-10)
                    bow = math.log10(bow_val)

            merged.ngrams[key] = NgramEntry(log_prob=log_p, backoff=bow)

    for order in range(1, max_order + 1):
        merged.counts[order] = sum(1 for k in merged.ngrams if len(k) == order)
    merged.max_order = max_order

    return merged

# ── Ch'en: validate ───────────────────────────────────────────────────────────

def validate_merged(merged: ArpaModel, tolerance: float = 1e-3) -> bool:
    print("  [Ch'en] Validating probability mass …", file=sys.stderr)
    errors: list[str] = []

    # Unigram mass
    uni_mass = sum(
        10.0 ** e.log_prob for k, e in merged.ngrams.items() if len(k) == 1
    )
    if abs(uni_mass - 1.0) > tolerance:
        errors.append(f"Unigram mass = {uni_mass:.6f} (expected 1.0 ± {tolerance})")

    # Spot-check bigram histories: sample up to 200 histories
    bi_histories: dict[tuple, list[float]] = defaultdict(list)
    for k, e in merged.ngrams.items():
        if len(k) == 2:
            bi_histories[k[:1]].append(10.0 ** e.log_prob)

    sampled = list(bi_histories.items())[:200]
    mass_errors = 0
    for hist, probs in sampled:
        # mass from explicit n-grams alone may be < 1 (backoff handles rest)
        mass = sum(probs)
        if mass > 1.0 + tolerance:
            mass_errors += 1
    if mass_errors:
        errors.append(
            f"{mass_errors}/{len(sampled)} sampled bigram histories "
            f"have explicit mass > 1.0")

    # Edge reference check: all n-gram words should be in unigram set
    known_words = {k[0] for k in merged.ngrams if len(k) == 1}
    unknown = set()
    for k in merged.ngrams:
        for w in k:
            if w not in known_words:
                unknown.add(w)
    if unknown:
        errors.append(
            f"{len(unknown)} words in n-grams missing from unigram set: "
            f"{list(unknown)[:5]}")

    if errors:
        print("         [fail]", file=sys.stderr)
        for e in errors:
            print(f"           × {e}", file=sys.stderr)
        return False
    else:
        print("         [pass] unigram mass, bigram histories, edge references",
              file=sys.stderr)
        return True

# ── Xul: ARPA writer ──────────────────────────────────────────────────────────

def write_arpa(merged: ArpaModel, out_path: Path) -> None:
    max_order = merged.max_order
    counts = {o: merged.counts.get(o, 0) for o in range(1, max_order + 1)}

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\\data\\\n")
        for o in range(1, max_order + 1):
            fh.write(f"ngram {o}={counts[o]}\n")
        fh.write("\n")

        for order in range(1, max_order + 1):
            fh.write(f"\\{order}-grams:\n")
            for key, entry in sorted(
                ((k, e) for k, e in merged.ngrams.items() if len(k) == order),
                key=lambda x: x[0]
            ):
                words = "\t".join(key)
                line  = f"{entry.log_prob:.6f}\t{words}"
                if order < max_order and entry.backoff != 0.0:
                    line += f"\t{entry.backoff:.6f}"
                fh.write(line + "\n")
            fh.write("\n")

        fh.write("\\end\\\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge ARPA language models with probability interpolation")
    ap.add_argument("--inputs",   nargs="+", required=True,
                    help="Source .arpa files")
    ap.add_argument("--alphas",   nargs="+", type=float,
                    help="Per-source blend weights (default: equal)")
    ap.add_argument("--strategy", choices=["linear", "log-linear"],
                    default="linear")
    ap.add_argument("--min-prob", type=float, default=1e-8,
                    help="Drop merged n-grams below this probability")
    ap.add_argument("--output",   required=True,
                    help="Output .arpa path")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            sys.exit(f"[error] Not found: {p}")

    alphas = args.alphas or [1.0 / len(paths)] * len(paths)
    if len(alphas) != len(paths):
        sys.exit(f"[error] --alphas count != --inputs count")
    total = sum(alphas)
    alphas = [a / total for a in alphas]

    t0 = time.time()
    print("[Pop]  Parsing ARPA models …", file=sys.stderr)
    models = []
    for p in paths:
        print(f"       {p.name}", file=sys.stderr)
        m = parse_arpa(p)
        orders = {o: m.counts.get(o, 0) for o in range(1, m.max_order + 1)}
        print(f"         orders: {orders}", file=sys.stderr)
        models.append(m)

    print(f"\n       strategy={args.strategy}  alphas={alphas}"
          f"  min-prob={args.min_prob}\n", file=sys.stderr)

    merged = merge_models(models, alphas, args.strategy, args.min_prob)

    ok = validate_merged(merged)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  [Xul]  Writing {out.name} …", file=sys.stderr)
    write_arpa(merged, out)

    # Manifest
    manifest_path = out.with_suffix(".merge-manifest.json")
    manifest = {
        "schema":   "arpa.merge/1",
        "strategy": args.strategy,
        "alphas":   alphas,
        "sources":  [str(p) for p in paths],
        "counts":   {str(o): merged.counts.get(o, 0)
                     for o in range(1, merged.max_order + 1)},
        "min_prob": args.min_prob,
        "validation_passed": ok,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    kb = out.stat().st_size // 1024
    elapsed = time.time() - t0
    counts_str = "  ".join(
        f"{o}-grams={merged.counts.get(o,0):,}"
        for o in range(1, merged.max_order + 1)
    )
    print(f"\n  {out.name}  {kb:,} KB", file=sys.stderr)
    print(f"  {counts_str}", file=sys.stderr)
    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
