#!/usr/bin/env python3
"""
Build grammar-driven code-lane payloads from source text or <code>...</code> blocks.

This is a dedicated code lane (separate from Mx2LM natural-language n-gram lanes):
- preserves atomic code blocks
- parses structure with .code-grammar.json rules
- emits typed nodes/edges (+ optional fold fingerprints)
- preserves source shape via line/col/span metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def trim_copy(value: str) -> str:
    return value.strip()


def lower_copy(value: str) -> str:
    return value.lower()


def read_inputs(text_values: list[str], input_files: list[Path]) -> tuple[str, list[str]]:
    chunks: list[str] = []
    source_labels: list[str] = []

    for text in text_values:
        t = trim_copy(text)
        if not t:
            continue
        chunks.append(t)
        source_labels.append("inline")

    for p in input_files:
        data = p.read_text(encoding="utf-8", errors="replace")
        if trim_copy(data):
            chunks.append(data)
            source_labels.append(str(p))

    if not chunks and not sys.stdin.isatty():
        data = sys.stdin.read()
        if trim_copy(data):
            chunks.append(data)
            source_labels.append("stdin")

    return "\n".join(chunks), source_labels


def detect_lang_hint(attrs: str) -> str:
    if not attrs:
        return ""
    for pat in (
        r'lang\s*=\s*"([^"]+)"',
        r"lang\s*=\s*'([^']+)'",
        r'language\s*=\s*"([^"]+)"',
        r"language\s*=\s*'([^']+)'",
    ):
        m = re.search(pat, attrs, re.IGNORECASE)
        if m:
            return trim_copy(m.group(1))
    return ""


def extract_code_blocks(source_text: str, strict: bool) -> tuple[list[dict[str, Any]], bool]:
    pattern = re.compile(
        r"<code(?P<attrs>[^>]*)>(?P<body>.*?)</code>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(source_text))
    if not matches:
        if strict:
            raise ValueError("no <code>...</code> blocks found while --strict-code-tags is enabled")
        return [{"index": 0, "lang_hint": "", "text": source_text}], False

    blocks: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        body = m.group("body")
        attrs = m.group("attrs") or ""
        blocks.append(
            {
                "index": i,
                "lang_hint": detect_lang_hint(attrs),
                "text": body,
            }
        )
    return blocks, True


def load_grammar(grammar_file: str, repo_root: Path) -> tuple[dict[str, Any], str]:
    candidates = []
    gpath = Path(grammar_file)
    if gpath.is_file():
        candidates.append(gpath)
    else:
        candidates.append(repo_root / grammar_file)
        candidates.append(repo_root / "data" / "grammar" / grammar_file)

    for p in candidates:
        if not p.is_file():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj, str(p)

    raise FileNotFoundError(f"grammar file not found or invalid JSON: {grammar_file}")


def line_starts_for(source: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def line_col_at(starts: list[int], pos: int) -> tuple[int, int]:
    if pos < 0:
        return 1, 0
    idx = bisect_right(starts, pos) - 1
    if idx < 0:
        return 1, 0
    line = idx + 1
    col = pos - starts[idx]
    return line, col


@dataclass
class CodeNode:
    id: str
    type: str
    text: str
    line: int
    col: int
    start: int
    end: int
    block_index: int
    captures: dict[str, str]


@dataclass
class CodeEdge:
    from_id: str
    to_id: str
    kind: str
    weight: float = 1.0


def parse_with_grammar(
    grammar_json: dict[str, Any],
    source: str,
    block_index: int,
    id_prefix: str,
) -> tuple[list[CodeNode], list[CodeEdge]]:
    nodes: list[CodeNode] = []
    edges: list[CodeEdge] = []
    tokens_val = grammar_json.get("tokens")
    edges_val = grammar_json.get("edges")
    if not isinstance(tokens_val, list):
        return nodes, edges

    starts = line_starts_for(source)
    node_counter = 0

    for tok_spec in tokens_val:
        if not isinstance(tok_spec, dict):
            continue
        tok_type = tok_spec.get("type")
        raw_pattern = tok_spec.get("pattern")
        captures = tok_spec.get("captures", [])
        if not isinstance(tok_type, str) or not tok_type:
            continue
        if not isinstance(raw_pattern, str) or not raw_pattern:
            continue
        if not isinstance(captures, list):
            captures = []
        cap_names = [c for c in captures if isinstance(c, str)]

        try:
            rx = re.compile(raw_pattern, re.MULTILINE)
        except re.error:
            continue

        for m in rx.finditer(source):
            st, en = m.start(0), m.end(0)
            line, col = line_col_at(starts, st)
            cap_map: dict[str, str] = {}
            for gi, cname in enumerate(cap_names, start=1):
                if gi <= (m.lastindex or 0) and m.group(gi) is not None:
                    cap_map[cname] = m.group(gi)
            nodes.append(
                CodeNode(
                    id=f"{id_prefix}n{node_counter}",
                    type=tok_type,
                    text=m.group(0),
                    line=line,
                    col=col,
                    start=st,
                    end=en,
                    block_index=block_index,
                    captures=cap_map,
                )
            )
            node_counter += 1

    nodes.sort(key=lambda n: (n.line, n.col, n.start, n.end))
    by_type: dict[str, list[int]] = {}
    for i, node in enumerate(nodes):
        by_type.setdefault(node.type, []).append(i)

    if not isinstance(edges_val, list):
        return nodes, edges

    for edge_rule in edges_val:
        if not isinstance(edge_rule, dict):
            continue
        from_type = edge_rule.get("from_type")
        to_type = edge_rule.get("to_type")
        kind = edge_rule.get("kind")
        condition = edge_rule.get("condition", "sequence")
        if not isinstance(from_type, str) or not isinstance(to_type, str) or not isinstance(kind, str):
            continue
        if not isinstance(condition, str):
            condition = "sequence"

        from_nodes = by_type.get(from_type, [])
        to_nodes = by_type.get(to_type, [])

        def after(a: CodeNode, b: CodeNode) -> bool:
            return b.line > a.line or (b.line == a.line and b.col > a.col)

        if from_type == to_type:
            if condition == "sequence":
                for i in range(len(from_nodes) - 1):
                    a = nodes[from_nodes[i]]
                    b = nodes[from_nodes[i + 1]]
                    edges.append(CodeEdge(a.id, b.id, kind, 1.0))
            continue

        if condition == "sequence":
            for fi in from_nodes:
                fn = nodes[fi]
                for ti in to_nodes:
                    tn = nodes[ti]
                    if after(fn, tn):
                        edges.append(CodeEdge(fn.id, tn.id, kind, 1.0))
                        break
        elif condition == "precedes":
            for fi in from_nodes:
                fn = nodes[fi]
                for ti in to_nodes:
                    tn = nodes[ti]
                    if tn.line == fn.line + 1 or (tn.line == fn.line and tn.col > fn.col):
                        edges.append(CodeEdge(fn.id, tn.id, kind, 1.0))
        else:
            for fi in from_nodes:
                for ti in to_nodes:
                    if nodes[fi].id == nodes[ti].id:
                        continue
                    edges.append(CodeEdge(nodes[fi].id, nodes[ti].id, kind, 1.0))

    return nodes, edges


def fold_fingerprints(nodes: list[CodeNode], edges: list[CodeEdge]) -> list[dict[str, Any]]:
    if not nodes:
        return []

    node_idx = {node.id: i for i, node in enumerate(nodes)}
    columns: list[dict[str, Any]] = []
    col_idx: dict[str, int] = {}
    for ed in edges:
        key = f"{ed.kind}:{ed.from_id}:{ed.to_id}"
        if key not in col_idx:
            col_idx[key] = len(columns)
            columns.append({"kind": ed.kind, "from": ed.from_id, "to": ed.to_id, "members": []})
        for nid in (ed.from_id, ed.to_id):
            if nid in node_idx:
                columns[col_idx[key]]["members"].append(node_idx[nid])

    m = len(columns)
    out: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        contexts: list[str] = []
        active = 0
        for col in columns:
            members = col.get("members", [])
            if i in members:
                active += 1
                contexts.append(f"{col['kind']}:{col['to']}")
        out.append(
            {
                "node": node.id,
                "type": node.type,
                "text": node.text,
                "active": active,
                "density": (float(active) / float(m)) if m else 0.0,
                "contexts": contexts,
            }
        )
    return out


def build_payload(
    source_text: str,
    *,
    grammar_json: dict[str, Any],
    grammar_file: str,
    extract_code: bool,
    strict_code_tags: bool,
    include_fold: bool,
) -> dict[str, Any]:
    blocks, used_code_tags = extract_code_blocks(source_text, strict=strict_code_tags) if extract_code else (
        [{"index": 0, "lang_hint": "", "text": source_text}],
        False,
    )

    all_nodes: list[CodeNode] = []
    all_edges: list[CodeEdge] = []
    code_blocks_meta: list[dict[str, Any]] = []

    language = ""
    if isinstance(grammar_json.get("language"), str):
        language = str(grammar_json.get("language", ""))

    for block in blocks:
        idx = int(block["index"])
        code = str(block.get("text", ""))
        lang_hint = str(block.get("lang_hint", ""))
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        lines = code.count("\n") + (1 if code else 0)
        nodes, edges = parse_with_grammar(grammar_json, code, idx, id_prefix=f"b{idx}_")
        all_nodes.extend(nodes)
        all_edges.extend(edges)
        code_blocks_meta.append(
            {
                "index": idx,
                "lang_hint": lang_hint,
                "chars": len(code),
                "lines": lines,
                "sha256": code_hash,
                "preview": trim_copy(code.replace("\n", " "))[:120],
            }
        )

    payload: dict[str, Any] = {
        "ok": True,
        "source": "code-gram-builder",
        "invoke": "@code_gram",
        "response": "<code>{graph}</code>",
        "lane": "code",
        "grammar_file": grammar_file,
        "language": language,
        "code_tag_mode": "explicit_code_blocks" if used_code_tags else "raw_input_block",
        "block_count": len(code_blocks_meta),
        "node_count": len(all_nodes),
        "edge_count": len(all_edges),
        "code_blocks": code_blocks_meta,
        "nodes": [
            {
                "id": n.id,
                "type": n.type,
                "text": n.text,
                "line": n.line,
                "col": n.col,
                "span": {"start": n.start, "end": n.end},
                "block_index": n.block_index,
                "captures": n.captures,
            }
            for n in all_nodes
        ],
        "edges": [
            {
                "from": e.from_id,
                "to": e.to_id,
                "kind": e.kind,
                "weight": e.weight,
            }
            for e in all_edges
        ],
        "structured_composition": {
            "atomic_unit": "code_block",
            "shape_preservation": "line_col_span",
            "analysis_lane": "grammar_graph",
            "semantic_authority": "definitions_and_graph_relations",
            "association_authority": "pattern_strength_only",
        },
    }

    if include_fold:
        payload["fold_fingerprints"] = fold_fingerprints(all_nodes, all_edges)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grammar-driven code-lane graph JSON.")
    parser.add_argument("--text", action="append", default=[], help="Inline source text (repeatable).")
    parser.add_argument("--input", action="append", default=[], metavar="PATH", help="Source file path (repeatable).")
    parser.add_argument(
        "--grammar-file",
        default="cpp.code-grammar.json",
        help='Grammar file path or name under data/grammar (default: "cpp.code-grammar.json").',
    )
    parser.add_argument(
        "--extract-code",
        action="store_true",
        help="Extract <code>...</code> blocks; without this, the whole input is one atomic block.",
    )
    parser.add_argument(
        "--strict-code-tags",
        action="store_true",
        help="With --extract-code, fail if no <code>...</code> blocks are found.",
    )
    parser.add_argument(
        "--no-fold",
        action="store_true",
        help="Disable fold_fingerprints output.",
    )
    parser.add_argument("--output", default="", metavar="PATH", help="Optional output file path; defaults to stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        source_text, source_labels = read_inputs(
            [str(v) for v in args.text],
            [Path(p) for p in args.input],
        )
    except Exception as exc:
        print(f"ERROR: failed reading inputs: {exc}", file=sys.stderr)
        return 1

    if not trim_copy(source_text):
        print("ERROR: no input source provided (--text/--input or stdin).", file=sys.stderr)
        return 1

    try:
        repo_root = Path(__file__).resolve().parent
        grammar_json, grammar_path = load_grammar(str(args.grammar_file), repo_root)
    except Exception as exc:
        print(f"ERROR: failed loading grammar: {exc}", file=sys.stderr)
        return 1

    try:
        out = build_payload(
            source_text,
            grammar_json=grammar_json,
            grammar_file=grammar_path,
            extract_code=bool(args.extract_code),
            strict_code_tags=bool(args.strict_code_tags),
            include_fold=not bool(args.no_fold),
        )
    except Exception as exc:
        print(f"ERROR: failed building code graph payload: {exc}", file=sys.stderr)
        return 1

    out["sources"] = source_labels
    text = json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None)

    if args.output:
        try:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"ERROR: failed writing output: {exc}", file=sys.stderr)
            return 1
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

