#!/usr/bin/env python3
"""Question-to-Fold compiler.

Translates a natural language question into a compiled question record
(schema kh.question-fold/1) that drives field_query.py path execution
with XCFE-classified intent and WCR-weighted ranking.

Five question operators (XCFE mapping):
  DEFINE   — "What is X?"      CONDITION + VERIFICATION  Wo / Ch'en
  EXPLAIN  — "How does X?"     CAUSAL                    Wo
  COMPARE  — "Compare X and Y" SELECTION                 Yax
  WHY      — "Why is X?"       CAUSAL                    Wo
  DISCOVER — "What connects X and Y?" / "Find X"         EXECUTION  Sek

Compiled record (kh.question-fold/1):
  {
    "schema": "kh.question-fold/1",
    "question": "...",
    "intent": "DEFINE",
    "target": "fold",
    "xcfe": {"class": ..., "verification_class": ...,
             "phase": ..., "verification_phase": ...},
    "field": {"path": "Pop::fold / Wo::*[@cond > 0.3] / Yax::*",
              "threshold": 0.3, "limit": 100},
    "ranking": {"prior": "safetensors", "learned": "gram_store",
                "relevance": "definition", "combine": "multiply"},
    "learning": {"read_phase": "Pop", "write_phase": "Xul",
                 "write_requires": "evaluated_outcome"}
  }

Usage:
  py -3 tools\\question_to_fold.py "What is a fold?"
  py -3 tools\\question_to_fold.py --run "What is a fold?"   # compile + execute
  py -3 tools\\question_to_fold.py                           # REPL
"""
from __future__ import annotations

import io
import json
import re
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

# ── stdout UTF-8 guard ────────────────────────────────────────────────────────

def _ensure_utf8() -> None:
    try:
        enc = getattr(sys.stdout, "encoding", "") or ""
        if hasattr(sys.stdout, "buffer") and enc.lower().replace("-", "") != "utf8":
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

_ensure_utf8()

# ── Intent definitions ────────────────────────────────────────────────────────

@dataclass
class IntentSpec:
    name:               str
    xcfe_class:         str
    verification_class: Optional[str]
    phase:              str
    verification_phase: Optional[str]
    relevance:          str    # used to select R-weighting in WCR
    threshold:          float
    path_template:      str    # {target} will be substituted

_INTENTS: list[IntentSpec] = [
    IntentSpec(
        name="DEFINE",
        xcfe_class="CONDITION",
        verification_class="VERIFICATION",
        phase="Wo",
        verification_phase="Ch'en",
        relevance="definition",
        threshold=0.3,
        path_template="Pop::{target} / Wo::*[@cond > {thr}] / Yax::*",
    ),
    IntentSpec(
        name="EXPLAIN",
        xcfe_class="CAUSAL",
        verification_class=None,
        phase="Wo",
        verification_phase=None,
        relevance="causal_chain",
        threshold=0.2,
        path_template="Pop::{target} / Wo::*[@cond > {thr}] / Yax::*[@cond > {thr}]",
    ),
    IntentSpec(
        name="COMPARE",
        xcfe_class="SELECTION",
        verification_class=None,
        phase="Yax",
        verification_phase=None,
        relevance="contrast",
        threshold=0.15,
        path_template="Pop::{target} / Wo::* / Yax::*[@cond > {thr}]",
    ),
    IntentSpec(
        name="WHY",
        xcfe_class="CAUSAL",
        verification_class=None,
        phase="Wo",
        verification_phase=None,
        relevance="causal_necessity",
        threshold=0.25,
        path_template="Pop::{target} / Wo::*[@cond > {thr}] / Yax::*",
    ),
    IntentSpec(
        name="DISCOVER",
        xcfe_class="EXECUTION",
        verification_class=None,
        phase="Sek",
        verification_phase=None,
        relevance="structural_connection",
        threshold=0.1,
        path_template="Pop::{target} / Wo::* / Yax::*[@cond > {thr}]",
    ),
]

_INTENT_MAP: dict[str, IntentSpec] = {s.name: s for s in _INTENTS}

# ── Intent detection grammar ──────────────────────────────────────────────────
# Each rule: (pattern, intent_name)
# Patterns are matched case-insensitively against the full question.

_INTENT_RULES: list[tuple[re.Pattern[str], str]] = [
    # DEFINE — "What is", "Define", "Describe", "Explain what"
    (re.compile(r"\bwhat\s+is\b|\bwhat\s+are\b|\bdefine\b|\bdescribe\b", re.I), "DEFINE"),
    # WHY — "Why is", "Why are", "Why does", "Why do"
    (re.compile(r"\bwhy\b", re.I), "WHY"),
    # EXPLAIN — "How does", "How do", "How is", "Explain how"
    (re.compile(r"\bhow\s+does\b|\bhow\s+do\b|\bhow\s+is\b|\bhow\s+are\b"
                r"|\bexplain\b", re.I), "EXPLAIN"),
    # COMPARE — "Compare", "Difference between", "Contrast", "vs"
    (re.compile(r"\bcompare\b|\bdifference\b|\bcontrast\b|\bversus\b|\bvs\.?\b", re.I),
     "COMPARE"),
    # DISCOVER — "What connects", "Find", "Show", "List", "How are X and Y related"
    (re.compile(r"\bconnects?\b|\bfind\b|\bshow\b|\blist\b|\brelated\b|\blinks?\b",
                re.I), "DISCOVER"),
]

def detect_intent(question: str) -> str:
    for pat, intent in _INTENT_RULES:
        if pat.search(question):
            return intent
    return "DEFINE"   # conservative fallback

# ── Target extraction ─────────────────────────────────────────────────────────
# Strips question scaffolding and returns the core noun phrase.
# For COMPARE/DISCOVER we want the first operand; caller handles secondary.

_STRIP_PREFIXES = re.compile(
    r"^(?:"
    r"what\s+is\s+(?:a|an|the)?\s*"
    r"|what\s+are\s+(?:the)?\s*"
    r"|what\s+connects\s+(?:the)?\s*"
    r"|how\s+does\s+(?:a|an|the)?\s*"
    r"|how\s+do\s+(?:the)?\s*"
    r"|how\s+is\s+(?:a|an|the)?\s*"
    r"|why\s+(?:is|are)\s+(?:the)?\s*"
    r"|why\s+does\s+(?:a|an|the)?\s*"
    r"|explain\s+(?:what\s+)?(?:a|an|the)?\s*"
    r"|describe\s+(?:a|an|the)?\s*"
    r"|define\s+(?:a|an|the)?\s*"
    r"|compare\s+(?:a|an|the)?\s*"
    r"|find\s+(?:the)?\s*"
    r"|show\s+(?:me\s+)?(?:the)?\s*"
    r"|list\s+(?:the)?\s*"
    r")", re.I
)

_STRIP_PUNCTUATION = re.compile(r"\s*[?!.]+$")
_STRIP_SUFFIXES = re.compile(
    r"\s+(?:work|works|necessary|needed|important|used|useful|mean|means)$",
    re.I
)

# For COMPARE: strip "and Y" / "vs Y" from the end to get primary target
_COMPARE_SPLIT = re.compile(
    r"\s+(?:and|vs\.?|versus|with)\s+(.+)$", re.I
)

def extract_target(question: str, intent: str) -> tuple[str, Optional[str]]:
    """Returns (primary_target, secondary_target_or_None)."""
    q = question.strip()
    q = _STRIP_PREFIXES.sub("", q)
    q = _STRIP_PUNCTUATION.sub("", q).strip()   # punctuation first
    q = _STRIP_SUFFIXES.sub("", q).strip()       # then trailing verbs/adjectives

    secondary = None
    if intent in ("COMPARE", "DISCOVER"):
        m = _COMPARE_SPLIT.search(q)
        if m:
            secondary = _normalise_target(m.group(1))
            q = q[:m.start()]

    primary = _normalise_target(q)
    if not primary:
        primary = "fold"   # safe default — always present in the corpus
    return primary, secondary


def _normalise_target(s: str) -> str:
    """Lowercase, collapse spaces, strip punctuation."""
    s = s.lower().strip(" ?!.,;:")
    s = re.sub(r"\s+", " ", s)
    return s

# ── Field path builder ────────────────────────────────────────────────────────

def build_path(spec: IntentSpec, target: str, secondary: Optional[str]) -> str:
    thr = f"{spec.threshold:.2f}".rstrip("0").rstrip(".")
    path = spec.path_template.format(target=target, thr=thr)
    if secondary and spec.name in ("COMPARE", "DISCOVER"):
        # Append a second branch for the secondary target
        second = spec.path_template.format(target=secondary, thr=thr)
        path = f"{path} | {second}"
    return path

# ── WCR relevance functions (corpus-level) ────────────────────────────────────
# These are symbolic policy descriptors; the actual numeric R computation
# runs inside field_query.py's executor using edge-conditional weights.

_RELEVANCE_DESC: dict[str, str] = {
    "definition":          "Rank by definitional co-occurrence depth (Wo+Yax cond product)",
    "causal_chain":        "Rank by longest high-cond chain through Wo and Yax",
    "contrast":            "Rank by divergence between primary and secondary neighborhoods",
    "causal_necessity":    "Rank by frequency of necessity-marker bigrams in neighborhood",
    "structural_connection": "Rank by shortest high-cond bridge path between targets",
}

# ── Compiled record builder ───────────────────────────────────────────────────

def compile_question(question: str) -> dict:
    intent_name             = detect_intent(question)
    spec                    = _INTENT_MAP[intent_name]
    primary, secondary      = extract_target(question, intent_name)
    path                    = build_path(spec, primary, secondary)

    xcfe: dict = {
        "class":              spec.xcfe_class,
        "phase":              spec.phase,
    }
    if spec.verification_class:
        xcfe["verification_class"]  = spec.verification_class
        xcfe["verification_phase"]  = spec.verification_phase

    record: dict = {
        "schema":   "kh.question-fold/1",
        "question": question,
        "intent":   intent_name,
        "target":   primary,
        "xcfe":     xcfe,
        "field": {
            "path":      path,
            "threshold": spec.threshold,
            "limit":     100,
        },
        "ranking": {
            "prior":     "safetensors",
            "learned":   "gram_store",
            "relevance": spec.relevance,
            "relevance_description": _RELEVANCE_DESC[spec.relevance],
            "combine":   "multiply",
        },
        "learning": {
            "read_phase":    "Pop",
            "write_phase":   "Xul",
            "write_requires": "evaluated_outcome",
        },
    }
    if secondary:
        record["secondary_target"] = secondary

    return record

# ── Field execution integration ───────────────────────────────────────────────
# Lazy import so question_to_fold.py can be used without loading 153 MB
# when only the compile step is wanted.

_fq_module = None    # cached module reference
_fq_index  = None    # cached FieldIndex (153 MB, load once)

def _load_field_query():
    global _fq_module, _fq_index
    if _fq_module is not None:
        return _fq_module, _fq_index
    tools_dir = str(Path(__file__).parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import importlib
    fq = importlib.import_module("field_query")
    idx = fq.FieldIndex()
    _fq_module = fq
    _fq_index  = idx
    return fq, idx


def run_field_query(record: dict) -> list[dict]:
    """Execute the compiled record's field path via field_query internals."""
    try:
        fq, idx = _load_field_query()
    except Exception as e:
        print(f"[error] Cannot load field_query: {e}", file=sys.stderr)
        return []

    path  = record["field"]["path"]
    limit = record["field"].get("limit", 20)

    # Handle compound paths (COMPARE/DISCOVER with |)
    branches = [p.strip() for p in path.split("|")]
    all_results = []

    for branch in branches:
        branch_with_limit = f"{branch} limit {limit}"
        try:
            toks    = fq.tokenize(branch_with_limit)
            query   = fq.Parser(toks).parse_query()
            results = fq.execute(query, idx)
            all_results.extend(results)
        except Exception as e:
            print(f"[warn] branch failed: {branch!r}: {e}", file=sys.stderr)

    # De-duplicate by path string, keep highest score
    seen: dict[str, object] = {}
    for r in all_results:
        key = " -> ".join(r.path)
        if key not in seen or r.score > seen[key].score:
            seen[key] = r

    ranked = sorted(seen.values(), key=lambda r: -r.score)
    return [{"path": r.path, "score": r.score, "node_weight": r.node_weight}
            for r in ranked[:limit]]


def _fmt_results(rows: list[dict], record: dict, limit: int = 20) -> None:
    intent = record["intent"]
    target = record["target"]
    path   = record["field"]["path"]
    print(f"\n--- {intent} :: {target!r}  ({len(rows)} results) ---")
    print(f"    path: {path}")
    print()
    for i, r in enumerate(rows[:limit]):
        score = r.get("score", 0.0)
        pstr  = " -> ".join(r.get("path", []))
        print(f"  {i+1:3d}.  {score:.4f}  {pstr}")
    if len(rows) > limit:
        print(f"  ... ({len(rows) - limit} more)")
    print()

# ── GramStore stub (Pop read / Xul write) ─────────────────────────────────────
# Full integration with gram_store.cpp is over JSON-RPC (port 8787).
# This module provides the protocol scaffolding; actual I/O requires KHANARY.exe.

def gramstore_read(target: str) -> dict:
    """Pop phase: attempt to read learned evidence for target from gram_store."""
    key = f"q_pref:question_fold:{target}"
    # Stub: returns neutral W_learned = 1.0 (no evidence yet)
    return {"key": key, "w_learned": 1.0, "evidence_count": 0, "status": "miss"}


def gramstore_write(target: str, intent: str, evaluated_delta: float) -> dict:
    """Xul phase: record evaluated outcome — only call after verified answer."""
    key = f"q_pref:question_fold:{target}"
    # Stub: in production this POSTs to JSON-RPC port 8787
    return {
        "key": key,
        "intent": intent,
        "delta": evaluated_delta,
        "write_requires": "evaluated_outcome",
        "status": "stub_not_written",
    }

# ── CLI / REPL ────────────────────────────────────────────────────────────────

_BANNER = """
Question-to-Fold compiler  (kh.question-fold/1)
  Type a question  —  "What is a fold?"
  --run <question> to compile + execute field query
  :compile <question>  to show record only
  :help  :quit
"""

def _print_record(record: dict) -> None:
    print(json.dumps(record, indent=2, ensure_ascii=False))


def main() -> None:
    args = sys.argv[1:]
    run_mode  = False
    questions: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--run" and i + 1 < len(args):
            run_mode = True
            questions.append(args[i + 1])
            i += 2
        elif args[i] == "--compile" and i + 1 < len(args):
            questions.append(args[i + 1])
            i += 2
        elif not args[i].startswith("--"):
            questions.append(args[i])
            i += 1
        else:
            i += 1

    if questions:
        for q in questions:
            record = compile_question(q)
            _print_record(record)
            if run_mode:
                rows = run_field_query(record)
                _fmt_results(rows, record)
        return

    # REPL
    print(_BANNER)
    while True:
        try:
            line = input("q2f> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q", "quit", "exit"):
            break
        if line == ":help":
            print(__doc__)
            continue

        execute = False
        if line.startswith(":compile "):
            line = line[len(":compile "):].strip()
        elif line.startswith(":run "):
            line = line[len(":run "):].strip()
            execute = True

        try:
            record = compile_question(line)
            _print_record(record)
            if execute:
                rows = run_field_query(record)
                _fmt_results(rows, record)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
