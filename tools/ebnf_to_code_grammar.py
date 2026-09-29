#!/usr/bin/env python3
"""
Generate .code-grammar.json from an EBNF-like source.

Supported EBNF subset:
- rule assignment with "::=" or "="
- alternation:      a | b
- concatenation:    a b  (or a, b)
- grouping:         ( ... )
- optional:         [ ... ]
- repetition:       { ... }, postfix *, +, ?
- string terminals: "literal" or 'literal'
- regex terminals:  /.../

Rule format:
  token_type[capture1,capture2] ::= /regex with capture groups/ ;

Directives (line starts with "@"):
  @language <name>
  @version <value>
  @emit <rule> [more rules...]
  @helper <rule> [more rules...]
  @token <rule> [type=<token_type>] [edge_kind=<kind>]
  @edge <from> <to> <kind> [condition]
  @edge {"from_type":"...","to_type":"...","kind":"...","condition":"sequence"}
  @auto_edges <true|false|on|off|1|0|yes|no>
  @edge_default_kind <kind>
  @edge_default_condition <sequence|precedes|file>
  @builtin <NAME> <regex>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_BUILTINS: dict[str, str] = {
    "WS": r"\s+",
    "S": r"\s*",
    "IDENT": r"[A-Za-z_]\w*",
    "QUAL_IDENT": r"[A-Za-z_][\w:]*",
    "NUMBER": r"\d+(?:\.\d+)?",
    "INT": r"\d+",
    "PATH": r"[^>\"]+",
    "NEWLINE": r"\r?\n",
}


@dataclass
class DirectiveState:
    language: str = ""
    version: str = ""
    emit_rules: set[str] = field(default_factory=set)
    helper_rules: set[str] = field(default_factory=set)
    token_meta: dict[str, dict[str, str]] = field(default_factory=dict)
    edges: list[dict[str, str]] = field(default_factory=list)
    auto_edges: bool = True
    edge_default_kind: str = "contains"
    edge_default_condition: str = "sequence"
    builtins: dict[str, str] = field(default_factory=dict)


@dataclass
class LexToken:
    kind: str
    value: str
    pos: int


@dataclass
class ExprNode:
    kind: str
    value: str = ""
    items: list["ExprNode"] = field(default_factory=list)


@dataclass
class RuleSpec:
    name: str
    captures: list[str]
    expr: ExprNode


class ParseError(RuntimeError):
    pass


def trim_copy(value: str) -> str:
    return value.strip()


def lower_copy(value: str) -> str:
    return value.lower()


def read_inputs(text_values: list[str], input_files: list[Path]) -> str:
    chunks: list[str] = []
    for text in text_values:
        t = trim_copy(text)
        if t:
            chunks.append(t)

    for p in input_files:
        data = p.read_text(encoding="utf-8", errors="replace")
        if trim_copy(data):
            chunks.append(data)

    if not chunks and not sys.stdin.isatty():
        data = sys.stdin.read()
        if trim_copy(data):
            chunks.append(data)

    return "\n".join(chunks)


def parse_bool_text(raw: str) -> bool:
    value = lower_copy(trim_copy(raw))
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {raw}")


def parse_edge_spec(raw: str) -> dict[str, str]:
    text = trim_copy(raw)
    if not text:
        raise ValueError("empty edge directive")

    if text.startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("edge JSON must be an object")
        from_type = trim_copy(str(obj.get("from_type", "")))
        to_type = trim_copy(str(obj.get("to_type", "")))
        kind = trim_copy(str(obj.get("kind", "")))
        condition = trim_copy(str(obj.get("condition", "sequence"))) or "sequence"
        if not from_type or not to_type or not kind:
            raise ValueError("edge JSON requires from_type, to_type, and kind")
        return {
            "from_type": from_type,
            "to_type": to_type,
            "kind": kind,
            "condition": condition,
        }

    normalized = (
        text.replace("->", " ")
        .replace(":", " ")
        .replace(",", " ")
    )
    parts = [p for p in normalized.split() if p]
    if len(parts) < 3:
        raise ValueError("edge requires at least: <from> <to> <kind>")
    condition = parts[3] if len(parts) >= 4 else "sequence"
    return {
        "from_type": parts[0],
        "to_type": parts[1],
        "kind": parts[2],
        "condition": condition,
    }


def parse_rule_with_capture_suffix(raw: str) -> tuple[str, list[str]]:
    text = trim_copy(raw)
    if not text:
        raise ValueError("empty rule name")

    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*)(?:\[(.*?)\])?", text)
    if not m:
        raise ValueError(f"invalid rule name syntax: {raw}")
    name = m.group(1)
    captures_blob = m.group(2)
    captures: list[str] = []
    if captures_blob:
        for part in captures_blob.split(","):
            cap = trim_copy(part)
            if cap:
                captures.append(cap)
    return name, captures


def parse_directives(source_text: str) -> tuple[DirectiveState, str]:
    state = DirectiveState()
    non_directive_lines: list[str] = []

    for line in source_text.splitlines():
        stripped = trim_copy(line)
        if not stripped.startswith("@"):
            non_directive_lines.append(line)
            continue

        body = stripped[1:]
        if not body:
            continue
        keyword, _, rest = body.partition(" ")
        key = lower_copy(trim_copy(keyword))
        payload = trim_copy(rest)

        if key == "language":
            state.language = payload
            continue
        if key == "version":
            state.version = payload
            continue
        if key == "emit":
            for token in payload.split():
                state.emit_rules.add(token)
            continue
        if key == "helper":
            for token in payload.split():
                state.helper_rules.add(token)
            continue
        if key == "token":
            fields = payload.split()
            if not fields:
                raise ValueError("@token requires a rule name")
            rule_name = fields[0]
            state.emit_rules.add(rule_name)
            meta = state.token_meta.setdefault(rule_name, {})
            for opt in fields[1:]:
                if "=" not in opt:
                    continue
                k, v = opt.split("=", 1)
                meta[lower_copy(trim_copy(k))] = trim_copy(v)
            continue
        if key == "edge":
            state.edges.append(parse_edge_spec(payload))
            continue
        if key == "auto_edges":
            state.auto_edges = parse_bool_text(payload)
            continue
        if key == "edge_default_kind":
            state.edge_default_kind = payload or state.edge_default_kind
            continue
        if key == "edge_default_condition":
            state.edge_default_condition = payload or state.edge_default_condition
            continue
        if key == "builtin":
            fields = payload.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError("@builtin requires: <NAME> <regex>")
            name = fields[0]
            regex_blob = trim_copy(fields[1])
            if regex_blob.startswith("/") and regex_blob.endswith("/") and len(regex_blob) >= 2:
                regex_blob = regex_blob[1:-1]
            state.builtins[name] = regex_blob
            continue

        raise ValueError(f"unknown directive '@{keyword}'")

    return state, "\n".join(non_directive_lines)


def tokenize_ebnf(source_text: str) -> list[LexToken]:
    out: list[LexToken] = []
    i = 0
    n = len(source_text)

    while i < n:
        ch = source_text[i]

        if ch.isspace():
            i += 1
            continue

        if source_text.startswith("//", i):
            nl = source_text.find("\n", i + 2)
            i = n if nl < 0 else nl + 1
            continue
        if source_text.startswith("#", i):
            nl = source_text.find("\n", i + 1)
            i = n if nl < 0 else nl + 1
            continue
        if source_text.startswith("/*", i):
            end = source_text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if source_text.startswith("(*", i):
            end = source_text.find("*)", i + 2)
            i = n if end < 0 else end + 2
            continue

        if source_text.startswith("::=", i):
            out.append(LexToken("ASSIGN", "::=", i))
            i += 3
            continue
        if ch == "=":
            out.append(LexToken("ASSIGN", "=", i))
            i += 1
            continue

        if ch in {"|", "(", ")", "[", "]", "{", "}", ",", ";", "*", "+", "?"}:
            kind_map = {
                "|": "PIPE",
                "(": "LPAREN",
                ")": "RPAREN",
                "[": "LBRACK",
                "]": "RBRACK",
                "{": "LBRACE",
                "}": "RBRACE",
                ",": "COMMA",
                ";": "SEMI",
                "*": "STAR",
                "+": "PLUS",
                "?": "QMARK",
            }
            out.append(LexToken(kind_map[ch], ch, i))
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            j = i + 1
            buf: list[str] = []
            escaped = False
            while j < n:
                cj = source_text[j]
                if escaped:
                    buf.append(cj)
                    escaped = False
                    j += 1
                    continue
                if cj == "\\":
                    escaped = True
                    j += 1
                    continue
                if cj == quote:
                    break
                buf.append(cj)
                j += 1
            if j >= n or source_text[j] != quote:
                raise ParseError(f"unterminated string literal at position {i}")
            out.append(LexToken("STRING", "".join(buf), i))
            i = j + 1
            continue

        if ch == "/" and not source_text.startswith("//", i) and not source_text.startswith("/*", i):
            j = i + 1
            buf: list[str] = []
            escaped = False
            in_class = False
            while j < n:
                cj = source_text[j]
                if escaped:
                    buf.append(cj)
                    escaped = False
                    j += 1
                    continue
                if cj == "\\":
                    escaped = True
                    buf.append(cj)
                    j += 1
                    continue
                if cj == "[":
                    in_class = True
                    buf.append(cj)
                    j += 1
                    continue
                if cj == "]":
                    in_class = False
                    buf.append(cj)
                    j += 1
                    continue
                if cj == "/" and not in_class:
                    break
                buf.append(cj)
                j += 1
            if j >= n or source_text[j] != "/":
                raise ParseError(f"unterminated regex literal at position {i}")
            out.append(LexToken("REGEX", "".join(buf), i))
            i = j + 1
            continue

        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (source_text[j].isalnum() or source_text[j] in {"_", "-"}):
                j += 1
            out.append(LexToken("IDENT", source_text[i:j], i))
            i = j
            continue

        raise ParseError(f"unexpected character '{ch}' at position {i}")

    out.append(LexToken("EOF", "", n))
    return out


class EBNFParser:
    def __init__(self, tokens: list[LexToken]) -> None:
        self.tokens = tokens
        self.idx = 0

    def peek(self) -> LexToken:
        return self.tokens[self.idx]

    def advance(self) -> LexToken:
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok

    def accept(self, kind: str) -> LexToken | None:
        if self.peek().kind == kind:
            return self.advance()
        return None

    def expect(self, kind: str) -> LexToken:
        tok = self.peek()
        if tok.kind != kind:
            raise ParseError(f"expected {kind}, got {tok.kind} at position {tok.pos}")
        return self.advance()

    def parse_rules(self) -> tuple[list[RuleSpec], dict[str, RuleSpec]]:
        ordered: list[RuleSpec] = []
        by_name: dict[str, RuleSpec] = {}

        while self.peek().kind != "EOF":
            if self.accept("SEMI"):
                continue

            lhs_parts: list[LexToken] = [self.expect("IDENT")]
            if self.accept("LBRACK"):
                lhs_parts.append(LexToken("LBRACK", "[", self.peek().pos))
                cap_tokens: list[str] = []
                while self.peek().kind not in {"RBRACK", "EOF"}:
                    t = self.advance()
                    cap_tokens.append(t.value if t.kind != "COMMA" else ",")
                self.expect("RBRACK")
                lhs_parts.append(LexToken("RAW", "".join(cap_tokens), self.peek().pos))
                lhs_parts.append(LexToken("RBRACK", "]", self.peek().pos))

            lhs_text = lhs_parts[0].value
            if len(lhs_parts) > 1:
                lhs_text = f"{lhs_parts[0].value}[{lhs_parts[2].value}]"

            name, captures = parse_rule_with_capture_suffix(lhs_text)
            self.expect("ASSIGN")
            expr = self.parse_alt(stop={"SEMI", "EOF"})
            self.accept("SEMI")

            spec = RuleSpec(name=name, captures=captures, expr=expr)
            ordered.append(spec)
            by_name[name] = spec

        return ordered, by_name

    def parse_alt(self, *, stop: set[str]) -> ExprNode:
        parts = [self.parse_seq(stop=stop | {"PIPE"})]
        while self.accept("PIPE"):
            parts.append(self.parse_seq(stop=stop | {"PIPE"}))
        if len(parts) == 1:
            return parts[0]
        return ExprNode(kind="alt", items=parts)

    def parse_seq(self, *, stop: set[str]) -> ExprNode:
        items: list[ExprNode] = []
        while True:
            tok = self.peek()
            if tok.kind in stop:
                break
            if tok.kind == "COMMA":
                self.advance()
                continue
            items.append(self.parse_postfix(stop=stop))

        if not items:
            return ExprNode(kind="empty")
        if len(items) == 1:
            return items[0]
        return ExprNode(kind="seq", items=items)

    def parse_postfix(self, *, stop: set[str]) -> ExprNode:
        node = self.parse_atom(stop=stop)
        while True:
            if self.accept("STAR"):
                node = ExprNode(kind="rep0", items=[node])
                continue
            if self.accept("PLUS"):
                node = ExprNode(kind="rep1", items=[node])
                continue
            if self.accept("QMARK"):
                node = ExprNode(kind="opt", items=[node])
                continue
            break
        return node

    def parse_atom(self, *, stop: set[str]) -> ExprNode:
        tok = self.peek()
        if tok.kind == "STRING":
            self.advance()
            return ExprNode(kind="lit", value=tok.value)
        if tok.kind == "REGEX":
            self.advance()
            return ExprNode(kind="regex", value=tok.value)
        if tok.kind == "IDENT":
            self.advance()
            return ExprNode(kind="ref", value=tok.value)
        if tok.kind == "LPAREN":
            self.advance()
            inner = self.parse_alt(stop={"RPAREN", "EOF"})
            self.expect("RPAREN")
            return ExprNode(kind="group", items=[inner])
        if tok.kind == "LBRACK":
            self.advance()
            inner = self.parse_alt(stop={"RBRACK", "EOF"})
            self.expect("RBRACK")
            return ExprNode(kind="opt", items=[inner])
        if tok.kind == "LBRACE":
            self.advance()
            inner = self.parse_alt(stop={"RBRACE", "EOF"})
            self.expect("RBRACE")
            return ExprNode(kind="rep0", items=[inner])

        if tok.kind in stop:
            return ExprNode(kind="empty")
        raise ParseError(f"unexpected token {tok.kind} at position {tok.pos}")


def collect_refs(expr: ExprNode, out: set[str]) -> None:
    if expr.kind == "ref":
        out.add(expr.value)
        return
    for item in expr.items:
        collect_refs(item, out)


def compile_expr(
    expr: ExprNode,
    *,
    rules: dict[str, RuleSpec],
    builtins: dict[str, str],
    cache: dict[str, str],
    visiting: set[str],
    strict_refs: bool,
) -> str:
    if expr.kind == "empty":
        return ""
    if expr.kind == "lit":
        return re.escape(expr.value)
    if expr.kind == "regex":
        return expr.value
    if expr.kind == "ref":
        name = expr.value
        if name in builtins:
            return builtins[name]
        if name in rules:
            return f"(?:{compile_rule(name, rules=rules, builtins=builtins, cache=cache, visiting=visiting, strict_refs=strict_refs)})"
        if strict_refs:
            raise ValueError(f"unknown reference '{name}'")
        return re.escape(name)
    if expr.kind == "seq":
        return "".join(
            compile_expr(
                item,
                rules=rules,
                builtins=builtins,
                cache=cache,
                visiting=visiting,
                strict_refs=strict_refs,
            )
            for item in expr.items
        )
    if expr.kind == "alt":
        parts = [
            compile_expr(
                item,
                rules=rules,
                builtins=builtins,
                cache=cache,
                visiting=visiting,
                strict_refs=strict_refs,
            )
            for item in expr.items
        ]
        return f"(?:{'|'.join(parts)})"
    if expr.kind == "group":
        inner = compile_expr(
            expr.items[0],
            rules=rules,
            builtins=builtins,
            cache=cache,
            visiting=visiting,
            strict_refs=strict_refs,
        )
        return f"(?:{inner})"
    if expr.kind == "opt":
        inner = compile_expr(
            expr.items[0],
            rules=rules,
            builtins=builtins,
            cache=cache,
            visiting=visiting,
            strict_refs=strict_refs,
        )
        return f"(?:{inner})?"
    if expr.kind == "rep0":
        inner = compile_expr(
            expr.items[0],
            rules=rules,
            builtins=builtins,
            cache=cache,
            visiting=visiting,
            strict_refs=strict_refs,
        )
        return f"(?:{inner})*"
    if expr.kind == "rep1":
        inner = compile_expr(
            expr.items[0],
            rules=rules,
            builtins=builtins,
            cache=cache,
            visiting=visiting,
            strict_refs=strict_refs,
        )
        return f"(?:{inner})+"
    raise ValueError(f"unsupported expression node kind: {expr.kind}")


def compile_rule(
    name: str,
    *,
    rules: dict[str, RuleSpec],
    builtins: dict[str, str],
    cache: dict[str, str],
    visiting: set[str],
    strict_refs: bool,
) -> str:
    if name in cache:
        return cache[name]
    if name in builtins:
        return builtins[name]
    if name not in rules:
        if strict_refs:
            raise ValueError(f"unknown rule: {name}")
        return re.escape(name)

    if name in visiting:
        raise ValueError(f"recursive rule cycle detected at '{name}'")
    visiting.add(name)
    compiled = compile_expr(
        rules[name].expr,
        rules=rules,
        builtins=builtins,
        cache=cache,
        visiting=visiting,
        strict_refs=strict_refs,
    )
    visiting.remove(name)
    cache[name] = compiled
    return compiled


def split_list_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            item = trim_copy(part)
            if item:
                out.append(item)
    return out


def parse_builtin_cli(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--builtin expects NAME=REGEX, got: {raw}")
        name, regex_blob = raw.split("=", 1)
        n = trim_copy(name)
        r = trim_copy(regex_blob)
        if not n or not r:
            raise ValueError(f"--builtin expects NAME=REGEX, got: {raw}")
        if r.startswith("/") and r.endswith("/") and len(r) >= 2:
            r = r[1:-1]
        out[n] = r
    return out


def dedupe_edges(edges: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, str]] = []
    for edge in edges:
        key = (
            edge.get("from_type", ""),
            edge.get("to_type", ""),
            edge.get("kind", ""),
            edge.get("condition", "sequence"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "from_type": key[0],
                "to_type": key[1],
                "kind": key[2],
                "condition": key[3],
            }
        )
    return out


def build_code_grammar(
    source_text: str,
    *,
    language_override: str,
    version_override: str,
    emit_overrides: list[str],
    helper_overrides: list[str],
    edge_overrides: list[str],
    no_auto_edges: bool,
    strict_refs: bool,
    builtin_overrides: dict[str, str],
) -> dict[str, Any]:
    directives, ebnf_body = parse_directives(source_text)
    tokens = tokenize_ebnf(ebnf_body)
    parser = EBNFParser(tokens)
    ordered_rules, rules_by_name = parser.parse_rules()

    if not ordered_rules:
        raise ValueError("no rules parsed from EBNF input")

    emit_set = set(directives.emit_rules)
    emit_set.update(split_list_args(emit_overrides))

    helper_set = set(directives.helper_rules)
    helper_set.update(split_list_args(helper_overrides))

    if emit_set:
        emit_rules = [r.name for r in ordered_rules if r.name in emit_set and r.name not in helper_set]
    else:
        emit_rules = [r.name for r in ordered_rules if r.name not in helper_set and not r.name.startswith("_")]

    if not emit_rules:
        raise ValueError("no token rules selected for emission; check @emit/@helper or CLI overrides")

    builtins = dict(DEFAULT_BUILTINS)
    for key, val in list(DEFAULT_BUILTINS.items()):
        builtins[lower_copy(key)] = val
    builtins.update(directives.builtins)
    builtins.update(builtin_overrides)

    cache: dict[str, str] = {}
    visiting: set[str] = set()
    token_meta = directives.token_meta

    tokens_out: list[dict[str, Any]] = []
    emit_type_by_rule: dict[str, str] = {}

    for rule_name in emit_rules:
        if rule_name not in rules_by_name:
            raise ValueError(f"emit rule not found: {rule_name}")
        spec = rules_by_name[rule_name]
        compiled = compile_rule(
            rule_name,
            rules=rules_by_name,
            builtins=builtins,
            cache=cache,
            visiting=visiting,
            strict_refs=strict_refs,
        )

        try:
            rx = re.compile(compiled, re.MULTILINE)
        except re.error as exc:
            raise ValueError(f"invalid regex for rule '{rule_name}': {exc}") from exc

        if len(spec.captures) > rx.groups:
            raise ValueError(
                f"rule '{rule_name}' defines {len(spec.captures)} captures but pattern has {rx.groups} groups"
            )

        meta = token_meta.get(rule_name, {})
        token_type = meta.get("type", rule_name)
        emit_type_by_rule[rule_name] = token_type
        tok_obj: dict[str, Any] = {
            "type": token_type,
            "pattern": compiled,
            "captures": list(spec.captures),
        }
        edge_kind = meta.get("edge_kind")
        if edge_kind:
            tok_obj["edge_kind"] = edge_kind
        tokens_out.append(tok_obj)

    auto_edges_enabled = directives.auto_edges and (not no_auto_edges)
    edge_default_kind = directives.edge_default_kind
    edge_default_condition = directives.edge_default_condition

    edges_out: list[dict[str, str]] = []
    if auto_edges_enabled:
        emit_rule_set = set(emit_rules)
        for rule_name in emit_rules:
            refs: set[str] = set()
            collect_refs(rules_by_name[rule_name].expr, refs)
            refs = {r for r in refs if r in emit_rule_set}
            for ref in sorted(refs):
                from_type = emit_type_by_rule[rule_name]
                to_type = emit_type_by_rule[ref]
                kind = token_meta.get(rule_name, {}).get("edge_kind", edge_default_kind)
                edges_out.append(
                    {
                        "from_type": from_type,
                        "to_type": to_type,
                        "kind": kind,
                        "condition": edge_default_condition,
                    }
                )

    for edge in directives.edges:
        e = dict(edge)
        if e["from_type"] in emit_type_by_rule:
            e["from_type"] = emit_type_by_rule[e["from_type"]]
        if e["to_type"] in emit_type_by_rule:
            e["to_type"] = emit_type_by_rule[e["to_type"]]
        if not e.get("condition"):
            e["condition"] = "sequence"
        edges_out.append(
            {
                "from_type": e["from_type"],
                "to_type": e["to_type"],
                "kind": e["kind"],
                "condition": e["condition"],
            }
        )

    for raw in edge_overrides:
        e = parse_edge_spec(raw)
        if e["from_type"] in emit_type_by_rule:
            e["from_type"] = emit_type_by_rule[e["from_type"]]
        if e["to_type"] in emit_type_by_rule:
            e["to_type"] = emit_type_by_rule[e["to_type"]]
        edges_out.append(e)

    grammar: dict[str, Any] = {
        "language": language_override or directives.language or "code",
        "version": version_override or directives.version or "1.0",
        "tokens": tokens_out,
        "edges": dedupe_edges(edges_out),
    }
    return grammar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate .code-grammar.json from EBNF input.")
    parser.add_argument("--text", action="append", default=[], help="Inline EBNF text (repeatable).")
    parser.add_argument("--input", action="append", default=[], metavar="PATH", help="EBNF input path (repeatable).")
    parser.add_argument("--language", default="", help="Override output language field.")
    parser.add_argument("--version", default="", help="Override output version field.")
    parser.add_argument("--emit", action="append", default=[], help="Comma- or space-list of token rules to emit.")
    parser.add_argument("--helper", action="append", default=[], help="Comma- or space-list of helper-only rules.")
    parser.add_argument(
        "--edge",
        action="append",
        default=[],
        help='Manual edge (repeatable), e.g. "function_def function_call calls sequence".',
    )
    parser.add_argument("--builtin", action="append", default=[], help="Builtin mapping NAME=REGEX (repeatable).")
    parser.add_argument("--no-auto-edges", action="store_true", help="Disable auto edges from rule references.")
    parser.add_argument(
        "--strict-refs",
        action="store_true",
        help="Fail if EBNF references undefined rules/builtins.",
    )
    parser.add_argument("--output", default="", metavar="PATH", help="Output path; defaults to stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        source = read_inputs(
            [str(v) for v in args.text],
            [Path(p) for p in args.input],
        )
    except Exception as exc:
        print(f"ERROR: failed reading input: {exc}", file=sys.stderr)
        return 1

    if not trim_copy(source):
        print("ERROR: no EBNF input provided (--text/--input or stdin).", file=sys.stderr)
        return 1

    try:
        builtin_map = parse_builtin_cli([str(v) for v in args.builtin])
        grammar = build_code_grammar(
            source,
            language_override=str(args.language),
            version_override=str(args.version),
            emit_overrides=[str(v) for v in args.emit],
            helper_overrides=[str(v) for v in args.helper],
            edge_overrides=[str(v) for v in args.edge],
            no_auto_edges=bool(args.no_auto_edges),
            strict_refs=bool(args.strict_refs),
            builtin_overrides=builtin_map,
        )
    except Exception as exc:
        print(f"ERROR: failed generating code grammar: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(grammar, ensure_ascii=False, indent=2 if args.pretty else None)
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
