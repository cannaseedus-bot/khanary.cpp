#!/usr/bin/env python3
"""XQuery-style field query engine over mx2lm.vectors.safetensors.

Query syntax  (Pop / Wo / Yax are the three phase axes):
  Pop::the                               single node — "the" in Pop layer
  Pop::*[@weight > 0.01]                 high-frequency unigrams
  Pop::the / Wo::*                       all bigrams starting with "the"
  Pop::the / Wo::* / Yax::*             trigrams starting with "the"
  Pop::the / Wo::*[@cond > 0.4]         high-probability bigram continuations
  Pop::the / Wo::*[@cond > 0.4] / Yax::*[@cond > 0.5]
  Wo::*[@label starts-with "of "]       bigrams prefixed by "of"
  Yax::*[@label contains "kuhul"]       trigrams containing "kuhul"
  Pop::*[@weight > 0.005] limit 5       explicit row cap (default 20)

Phase-axis map:
  Pop  = unigrams  — static_weight = P(w)
  Wo   = bigrams   — static_weight = P(w1,w2)   edge = P(w2|w1)
  Yax  = trigrams  — static_weight = P(w1,w2,w3) edge = P(w3|w1,w2)

Predicates:
  @weight     P(ngram) — node joint probability
  @cond       P(wn|context) — incoming edge conditional weight
  @label      node surface label (string ops: contains, starts-with, = !=)
  @x @y @z    geometric position

REPL commands:
  :help       show this help
  :stats      field statistics
  :quit       exit
"""
from __future__ import annotations

import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

def _ensure_utf8() -> None:
    try:
        enc = getattr(sys.stdout, "encoding", "") or ""
        if hasattr(sys.stdout, "buffer") and enc.lower().replace("-", "") != "utf8":
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_ensure_utf8()

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[1] / "data"
_ST   = _ROOT / "mx2lm.vectors.safetensors"
_LBL  = _ROOT / "mx2lm.vectors.labels.json"
_META = _ROOT / "mx2lm.vectors.meta.json"

PHASE_NAME = {0: "Pop", 1: "Wo", 2: "Yax", 3: "Sek", 4: "Ch'en", 5: "Xul"}
PHASE_ID   = {v: k for k, v in PHASE_NAME.items()}

_MAX_INTERMEDIATE = 50_000   # hard cap on mid-path result set

# ── Field index ───────────────────────────────────────────────────────────────

class FieldIndex:
    """Loaded safetensors field with CSR adjacency for fast path traversal."""

    def __init__(
        self,
        vectors: Path = _ST,
        labels: Path = _LBL,
        metadata: Path = _META,
    ) -> None:
        print("Loading field …", file=sys.stderr)
        d          = load_file(str(vectors))
        self.pos   = d["node_positions"]   # float32 [N, 3]
        self.w     = d["node_weights"]     # float32 [N]
        self.phase = d["node_phase"]       # int32   [N]
        ei         = d["edge_index"]       # int64   [2, E]
        ew         = d["edge_weights"]     # float32 [E]
        N          = self.pos.shape[0]
        E          = ew.shape[0]

        self.labels: list[str] = json.loads(labels.read_text(encoding="utf-8"))
        self.meta              = json.loads(metadata.read_text(encoding="utf-8"))

        print(f"  {N:,} nodes  {E:,} edges", file=sys.stderr)
        print("Building CSR adjacency …", file=sys.stderr)

        self.label_to_id: dict[str, int] = {l: i for i, l in enumerate(self.labels)}

        order        = np.argsort(ei[0], kind="stable")
        self.csr_tgt = ei[1][order]
        self.csr_ew  = ew[order]
        indptr       = np.zeros(N + 1, dtype=np.int64)
        np.add.at(indptr, ei[0][order] + 1, 1)
        np.cumsum(indptr, out=indptr)
        self.indptr  = indptr

        print("  ready.\n", file=sys.stderr)

    def out_neighbors(self, nid: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = int(self.indptr[nid]), int(self.indptr[nid + 1])
        return self.csr_tgt[s:e], self.csr_ew[s:e]

    def stats(self) -> None:
        N = self.pos.shape[0]
        E = self.csr_tgt.shape[0]
        print(f"  nodes   : {N:,}")
        print(f"  edges   : {E:,}")
        for pid, pname in PHASE_NAME.items():
            mask = self.phase == pid
            n = int(mask.sum())
            if n == 0:
                continue
            wvals = self.w[mask]
            print(f"  {pname:8s}: {n:>10,} nodes  "
                  f"P range [{float(wvals.min()):.2e}, {float(wvals.max()):.2e}]")


# ── Query AST ─────────────────────────────────────────────────────────────────

@dataclass
class Predicate:
    attr:  str          # 'weight', 'cond', 'label', 'x', 'y', 'z', 'phase'
    op:    str          # '>', '<', '>=', '<=', '=', '!=', 'contains', 'starts-with'
    value: float | str

@dataclass
class Step:
    phase:      int | None    # None = any
    selector:   str | None    # None = wildcard *
    predicates: list[Predicate] = field(default_factory=list)

@dataclass
class Query:
    steps: list[Step]
    limit: int = 20

# ── Tokeniser ─────────────────────────────────────────────────────────────────

_TOK_RE = re.compile(r"""
    (?P<DCOLON>  ::)
  | (?P<CMP>     >=|<=|!=|=|>|<)
  | (?P<FLOAT>   -?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<STRING>  "(?:[^"\\]|\\.)*")
  | (?P<SLASH>   /)
  | (?P<LBRACK>  \[)
  | (?P<RBRACK>  \])
  | (?P<ATTR>    @(?:weight|cond|label|phase|x|y|z)\b)
  | (?P<IDENT>   [A-Za-z'][A-Za-z0-9'_.*-]*)
  | (?P<STAR>    \*)
  | (?P<WS>      \s+)
""", re.VERBOSE)

def tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    while pos < len(text):
        m = _TOK_RE.match(text, pos)
        if m is None:
            raise ValueError(f"Unexpected character at {pos}: {text[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append((kind, m.group()))
    return tokens

# ── Parser ────────────────────────────────────────────────────────────────────

class Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos    = 0

    def peek(self, offset: int = 0) -> tuple[str, str] | None:
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else None

    def consume(self, kind: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of query")
        if kind and tok[0] != kind:
            raise ValueError(f"Expected {kind}, got {tok[0]}={tok[1]!r}")
        self.pos += 1
        return tok

    def parse_query(self) -> Query:
        steps = [self.parse_step()]
        while self.peek() and self.peek()[0] == "SLASH":
            self.consume("SLASH")
            steps.append(self.parse_step())
        limit = 20
        tok = self.peek()
        if tok and tok[1].lower() == "limit":
            self.consume()
            _, v = self.consume("FLOAT")
            limit = max(1, int(float(v)))
        return Query(steps=steps, limit=limit)

    def parse_step(self) -> Step:
        tok = self.consume()
        kind, raw = tok

        # Peek ahead for ::
        if self.peek() and self.peek()[0] == "DCOLON":
            self.consume("DCOLON")
            # raw = phase name (or *)
            phase = None if raw == "*" else PHASE_ID.get(raw)
            sel_tok = self.peek()
            if sel_tok and sel_tok[0] == "STAR":
                self.consume()
                selector = None
            elif sel_tok and sel_tok[0] in ("IDENT", "FLOAT"):
                self.consume()
                selector = sel_tok[1]
            elif sel_tok and sel_tok[0] == "STRING":
                self.consume()
                selector = sel_tok[1][1:-1]   # strip quotes
            else:
                selector = None
        else:
            # Bare selector without Phase:: prefix
            phase = None
            if kind == "STAR":
                selector = None
            elif kind == "STRING":
                selector = raw[1:-1]
            else:
                selector = raw

        preds: list[Predicate] = []
        while self.peek() and self.peek()[0] == "LBRACK":
            preds.append(self.parse_predicate())

        return Step(phase=phase, selector=selector, predicates=preds)

    def parse_predicate(self) -> Predicate:
        self.consume("LBRACK")
        _, raw_attr = self.consume("ATTR")
        attr = raw_attr[1:]   # strip @

        nxt = self.peek()
        if nxt and nxt[1] in ("starts-with", "contains", "starts_with"):
            op = self.consume()[1].replace("_", "-")
            val_tok = self.consume()
            val: float | str = (val_tok[1][1:-1]
                                if val_tok[0] == "STRING" else val_tok[1])
        else:
            op = self.consume("CMP")[1]
            val_tok = self.consume()
            val = (val_tok[1][1:-1] if val_tok[0] == "STRING"
                   else float(val_tok[1]))

        self.consume("RBRACK")
        return Predicate(attr=attr, op=op, value=val)

# ── Predicate evaluation ──────────────────────────────────────────────────────

def _cmp(a: float, op: str, b: float) -> bool:
    return (a > b  if op == ">"  else
            a < b  if op == "<"  else
            a >= b if op == ">=" else
            a <= b if op == "<=" else
            a == b if op == "="  else
            a != b if op == "!=" else True)

def _str_cmp(s: str, op: str, pat: str) -> bool:
    return (s == pat          if op in ("=", "==")      else
            s != pat          if op == "!="             else
            pat in s          if op == "contains"       else
            s.startswith(pat) if op == "starts-with"    else True)

def _node_ok(nid: int, node_w: float, idx: FieldIndex,
             preds: list[Predicate]) -> bool:
    for p in preds:
        if p.attr == "cond":
            continue
        if p.attr == "weight":
            if not _cmp(node_w, p.op, float(p.value)):
                return False
        elif p.attr == "label":
            if not _str_cmp(idx.labels[nid], p.op, str(p.value)):
                return False
        elif p.attr in ("x", "y", "z"):
            coord = ("x", "y", "z").index(p.attr)
            if not _cmp(float(idx.pos[nid, coord]), p.op, float(p.value)):
                return False
        elif p.attr == "phase":
            if not _cmp(float(idx.phase[nid]), p.op, float(p.value)):
                return False
    return True

def _cond_ok(ew: float, preds: list[Predicate]) -> bool:
    return all(_cmp(ew, p.op, float(p.value))
               for p in preds if p.attr == "cond")

# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    path:        list[str]
    score:       float       # product of traversed edge weights
    nid:         int
    node_weight: float

# ── Execution ─────────────────────────────────────────────────────────────────

def execute(query: Query, idx: FieldIndex) -> list[Result]:
    current: list[Result] = []

    for step_idx, step in enumerate(query.steps):

        if step_idx == 0:
            # ── Seed step: select nodes ───────────────────────────────────
            if step.selector is not None:
                nid = idx.label_to_id.get(step.selector)
                if nid is None:
                    return []
                if step.phase is not None and int(idx.phase[nid]) != step.phase:
                    return []
                nw = float(idx.w[nid])
                if not _node_ok(nid, nw, idx, step.predicates):
                    return []
                current = [Result([step.selector], 1.0, nid, nw)]

            else:
                # Wildcard: collect all matching nodes
                if step.phase is not None:
                    candidate_ids = np.where(idx.phase == step.phase)[0]
                else:
                    candidate_ids = np.arange(len(idx.labels), dtype=np.int64)

                # 1. Vectorised weight filter (fast numpy path)
                for p in step.predicates:
                    if p.attr != "weight":
                        continue
                    wvals = idx.w[candidate_ids]
                    v = float(p.value)
                    mask = (wvals > v  if p.op == ">"  else
                            wvals < v  if p.op == "<"  else
                            wvals >= v if p.op == ">=" else
                            wvals <= v if p.op == "<=" else
                            wvals == v if p.op == "="  else
                            wvals != v)
                    candidate_ids = candidate_ids[mask]

                # 2. Label filter (Python loop — must run before the cap)
                has_label_pred = any(p.attr == "label" for p in step.predicates)
                if has_label_pred:
                    lbl_mask = np.ones(len(candidate_ids), dtype=bool)
                    for p in step.predicates:
                        if p.attr != "label":
                            continue
                        pat = str(p.value)
                        for i, n in enumerate(candidate_ids):
                            if not _str_cmp(idx.labels[int(n)], p.op, pat):
                                lbl_mask[i] = False
                    candidate_ids = candidate_ids[lbl_mask]

                # 3. Cap after all filtering
                if len(candidate_ids) > _MAX_INTERMEDIATE:
                    print(f"  warning: trimming to {_MAX_INTERMEDIATE:,} "
                          f"(add @weight or @label predicates to narrow)",
                          file=sys.stderr)
                    candidate_ids = candidate_ids[:_MAX_INTERMEDIATE]

                current = [
                    Result([idx.labels[int(n)]], 1.0, int(n), float(idx.w[n]))
                    for n in candidate_ids
                    if _node_ok(int(n), float(idx.w[n]), idx, step.predicates)
                ]

        else:
            # ── Traversal step: follow out-edges ──────────────────────────
            next_results: list[Result] = []
            for r in current:
                tgts, ews = idx.out_neighbors(r.nid)
                for t, ew in zip(tgts, ews):
                    t  = int(t)
                    ew = float(ew)
                    if step.phase is not None and int(idx.phase[t]) != step.phase:
                        continue
                    if step.selector is not None and idx.labels[t] != step.selector:
                        continue
                    if not _cond_ok(ew, step.predicates):
                        continue
                    nw = float(idx.w[t])
                    if not _node_ok(t, nw, idx, step.predicates):
                        continue
                    next_results.append(Result(
                        r.path + [idx.labels[t]],
                        r.score * ew,
                        t, nw,
                    ))
                    if len(next_results) >= _MAX_INTERMEDIATE:
                        break
                if len(next_results) >= _MAX_INTERMEDIATE:
                    break
            current = next_results

    current.sort(key=lambda r: -r.score)
    return current[: query.limit]

# ── Output formatting ─────────────────────────────────────────────────────────

def _fmt_path(r: Result) -> str:
    return " -> ".join(r.path)

def print_results(results: list[Result], query: Query) -> None:
    if not results:
        print("  (no results)")
        return
    multi_step = len(query.steps) > 1
    for r in results:
        label = _fmt_path(r) if multi_step else r.path[-1]
        print(f"  {label:<60s}  score={r.score:.6f}  P(node)={r.node_weight:.2e}")
    print(f"  — {len(results)} result(s)")

# ── REPL ──────────────────────────────────────────────────────────────────────

_EXAMPLES = [
    ('Pop::the / Wo::*',
     'bigrams starting with "the"'),
    ('Pop::the / Wo::*[@cond > 0.3] / Yax::*[@cond > 0.4]',
     'high-probability trigrams starting with "the"'),
    ('Pop::*[@weight > 0.005]',
     'common unigrams'),
    ('Wo::*[@label starts-with "of "]',
     'bigrams prefixed by "of "'),
]

def _help() -> None:
    print(__doc__)
    print("Examples:")
    for q, desc in _EXAMPLES:
        print(f"  {q}")
        print(f"    → {desc}")

def run_one(line: str, idx: FieldIndex) -> None:
    toks    = tokenize(line)
    query   = Parser(toks).parse_query()
    results = execute(query, idx)
    print_results(results, query)

def repl(idx: FieldIndex) -> None:
    print("Field Query REPL — type :help for syntax, :stats for index info, :quit to exit")
    print('Try: Pop::the / Wo::*[@cond > 0.3]\n')

    try:
        import readline as _rl  # type: ignore
        _rl.set_history_length(500)
    except ImportError:
        pass

    while True:
        try:
            line = input("fq> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q", "quit", "exit"):
            break
        if line in (":help", ":h", "help"):
            _help()
            continue
        if line in (":stats", ":s"):
            idx.stats()
            continue
        try:
            run_one(line, idx)
        except Exception as e:
            print(f"  error: {e}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    _ensure_utf8()
    import argparse

    ap = argparse.ArgumentParser(
        description="XQuery-style field query over mx2lm.vectors.safetensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("query", nargs="?", help="Run a single query then exit")
    ap.add_argument("--vectors", type=Path, default=_ST,
                    help=f"SafeTensors field path (default: {_ST})")
    ap.add_argument("--labels", type=Path, default=None,
                    help="Label sidecar (default: <vectors stem>.labels.json)")
    ap.add_argument("--meta", type=Path, default=None,
                    help="Metadata sidecar (default: <vectors stem>.meta.json)")
    args = ap.parse_args()

    labels = args.labels or args.vectors.with_suffix(".labels.json")
    metadata = args.meta or args.vectors.with_suffix(".meta.json")

    for p in (args.vectors, labels, metadata):
        if not p.exists():
            print(f"ERROR: {p} not found. Run:", file=sys.stderr)
            print(f"  py -3 tools\\arpa_to_safetensors.py tools\\mx2lm.arpa --min-prob 1e-6",
                  file=sys.stderr)
            return 1

    idx = FieldIndex(args.vectors, labels, metadata)

    if args.query:
        try:
            run_one(args.query, idx)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        repl(idx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
