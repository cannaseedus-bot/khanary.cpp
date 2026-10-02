"""
lfm_factory.py — LFM distillation dataset factory.

Feeds prompts to LFM (CHAT-µ, port 17889), parses the responses, and writes
KHANARY data files: greeting banks, key-grammar JSONL, semantic track stubs,
capability-gap records, and raw JSONL logs.

Usage:
    python tools/lfm_factory.py run specs/greeting.jsonl
    python tools/lfm_factory.py run specs/greeting.jsonl --out data/grammar --dry-run
    python tools/lfm_factory.py ask "list 5 greetings for Pop phase" --format greeting_bank
    python tools/lfm_factory.py formats        # list available output formats

Spec file: one JSON object per line, fields:
    id          string  unique identifier for this record (required)
    format      string  output format: greeting_bank | key_grammar | track |
                        capability_gap | aiml | control_flow | code_grammar |
                        scfml_pattern_learning | raw_jsonl
    prompt      string  question/instruction sent to LFM (if absent, auto-built
                        from 'format' + other fields)
    out         string  output filename relative to --out dir (optional)
    phase       string  K'UHUL phase (Pop/Wo/Yax/Sek/Ch'en/Xul) for context
    kind        string  greeting kind (canonical/learned/associative/etc.)
    domain      string  domain hint injected into auto-prompts
    count       int     number of items to request (default: 3)
    namespace   string  gram key namespace prefix (default: derived from format+id)
    class_name  string  class label for key-grammar records (default: uppercase id)
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

LFM_PORT    = 17889
LFM_BASE    = f"http://127.0.0.1:{LFM_PORT}"
DATA_DIR    = Path(__file__).parent.parent / "data"
SPECS_DIR   = Path(__file__).parent.parent / "specs"
DEFAULT_OUT = DATA_DIR / "grammar"

_PORTS_FILE = DATA_DIR / "ports.json"
_resolved_ports: dict | None = None

def _load_ports() -> dict:
    """Read data/ports.json written by KHANARY.bat or start-lfm.bat. Cached."""
    global _resolved_ports
    if _resolved_ports is None:
        try:
            _resolved_ports = json.loads(_PORTS_FILE.read_text()) if _PORTS_FILE.exists() else {}
        except Exception:
            _resolved_ports = {}
    return _resolved_ports

def _chat_mu_port() -> int:
    """Return the live CHAT-µ port, falling back to LFM_PORT if ports.json absent."""
    p = _load_ports().get("chat_mu")
    return int(p) if p else LFM_PORT

# ── LFM client ─────────────────────────────────────────────────────────────────

def lfm_complete(
    prompt: str,
    system: str = "",
    port: int | None = None,
    max_tokens: int = 512,
    temperature: float = 0.4,
    tools: list | None = None,
) -> tuple[str, list[dict]]:
    """Returns (content_text, tool_calls_list).
    tool_calls_list entries: {"name": str, "arguments": dict}
    """
    if port is None:
        port = _chat_mu_port()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict = {
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        msg = data["choices"][0]["message"]

        # Collect tool calls from structured response
        calls: list[dict] = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append({"name": fn.get("name", ""), "arguments": args})

        content = msg.get("content") or ""
        return content.strip() if isinstance(content, str) else "", calls
    except Exception as exc:
        raise RuntimeError(f"LFM at port {port} unreachable: {exc}") from exc


def lfm_text(prompt: str, system: str = "", port: int | None = None,
             max_tokens: int = 512, temperature: float = 0.4) -> str:
    """Convenience wrapper — returns only the text content."""
    text, _ = lfm_complete(prompt, system=system, port=port,
                           max_tokens=max_tokens, temperature=temperature)
    return text


# ── Response parsers ────────────────────────────────────────────────────────────

def _strip_fence(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_json_response(text: str) -> Any:
    """Try to extract and parse any JSON structure from LFM output."""
    s = _strip_fence(text)
    # Direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Find first JSON array or object
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pattern, s)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


def parse_string_list(text: str) -> list[str]:
    """Parse a numbered or bulleted list of strings from LFM output."""
    parsed = parse_json_response(text)
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]

    lines = text.splitlines()
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip list markers: "1.", "-", "*", "•"
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        line = line.strip('"').strip("'").strip()
        if line and len(line) > 3:
            results.append(line)
    return results


# ── Ngram extraction ────────────────────────────────────────────────────────────

_PHASE_ROLES = {
    1: ("unigram",  "semantic_address",    "Pop"),
    2: ("bigram",   "context_association", "Wo"),
    3: ("trigram",  "discourse_frame",     "Yax"),
    4: ("quadgram", "fold_sequence",       "Sek"),
}

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[a-z']+|[0-9]+", text.lower())]


def build_gram_channels(template: str) -> dict[str, list[dict]]:
    tokens = _tokenize(template)
    channels: dict[str, list[dict]] = {}

    for n, (channel, role, phase) in _PHASE_ROLES.items():
        if len(tokens) < n:
            continue
        seen: dict[str, dict] = {}
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            if gram in seen:
                seen[gram]["count"] += 1
            else:
                seen[gram] = {
                    "gram": gram,
                    "count": 1,
                    "width": n,
                    "role": role,
                    "position": i,
                    "phase": phase,
                    "provenance": "lfm_factory",
                }
        channels[channel] = sorted(seen.values(), key=lambda x: x["position"])

    return channels


def build_key_grammar_record(
    template: str,
    key: str,
    phase: str,
    lexicon_source: str,
    class_name: str = "GENERATED",
    word: str = "",
) -> dict:
    return {
        "word":           word or key.lower().replace(":", "-"),
        "key":            key,
        "unit":           "unigram",
        "class":          class_name,
        "book":           f"Book({class_name})",
        "definition":     template,
        "lexicon_source": lexicon_source,
        "gram_channels":  build_gram_channels(template),
    }


_SCFML_PATTERN_KINDS = {
    "aiml_pattern",
    "aiml_template",
    "semantic_branch",
    "key_grammar",
    "route_pattern",
}
_SCFML_EPISTEMIC_STATES = {"SUPPORTED", "CONTRADICTED", "UNKNOWN", "MISS"}


def _scfml_text(value: Any, fallback: str = "") -> str:
    text = str(value if value is not None else fallback).strip()
    return text if text else fallback


def _scfml_safe_ns(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return cleaned or "scfml"


def _scfml_epistemic_state(entry: dict, spec: dict) -> str:
    state = _scfml_text(
        entry.get("epistemic_state", entry.get("state", "UNKNOWN")),
        "UNKNOWN",
    ).upper()
    if state not in _SCFML_EPISTEMIC_STATES:
        return "UNKNOWN"
    verified = bool(spec.get("user_verified") or entry.get("user_verified"))
    if state in {"SUPPORTED", "CONTRADICTED"} and not verified:
        return "UNKNOWN"
    return state


def build_scfml_branch_record(entry: dict, spec: dict, index: int, namespace: str) -> dict | None:
    condition = _scfml_text(
        entry.get("condition", entry.get("if", entry.get("pattern", "")))
    )
    then = _scfml_text(
        entry.get("then", entry.get("template", entry.get("response", entry.get("action", ""))))
    )
    if not condition or not then:
        return None

    branch_key = _scfml_text(
        entry.get("key"),
        f"scfml:branch:{namespace}:{index}",
    )
    if not branch_key.startswith(("scfml:branch:", "semantic_branch:")):
        branch_key = f"scfml:branch:{branch_key}"

    row = {
        "schema": "scfml.branch.v1",
        "key": branch_key,
        "condition": condition,
        "then": then,
        "else": _scfml_text(entry.get("else")) or None,
    }
    track = _scfml_text(entry.get("track", spec.get("track_id", "")))
    if track:
        row["track"] = track
    row["epistemic_state"] = _scfml_epistemic_state(entry, spec)
    unknown = _scfml_text(entry.get("unknown"))
    if unknown:
        row["unknown"] = unknown
    row["source_surface"] = _scfml_text(entry.get("source_surface"), "json")
    row["pattern_learning"] = True
    return row


def build_scfml_pattern_learning_record(
    entry: dict,
    spec: dict,
    index: int,
    namespace: str,
    branch_key: str = "",
) -> dict:
    candidate_kind = _scfml_text(
        entry.get("candidate_kind", entry.get("kind", spec.get("candidate_kind", ""))),
        "semantic_branch" if branch_key else "key_grammar",
    )
    if candidate_kind not in _SCFML_PATTERN_KINDS:
        candidate_kind = "semantic_branch" if branch_key else "key_grammar"

    source_event = _scfml_text(
        entry.get("source_event", entry.get("source", spec.get("source_event", ""))),
        f"prompt:{spec.get('id', 'lfm-factory')}:{index}",
    )
    target = _scfml_text(
        entry.get("target", branch_key),
        f"scfml:candidate:{namespace}:{index}",
    )
    return {
        "schema": "scfml.pattern_learning.v1",
        "source_event": source_event,
        "candidate_kind": candidate_kind,
        "requires": "SUPPORTED_or_user_verified",
        "target": target,
    }


def build_template_slot_record(
    slot_name: str,
    slot_def: dict,
    kit_id: str = "*",
) -> dict:
    """Build a template_slot_row gram record from a slot_definitions entry."""
    return {
        "schema":      "key.grammar.template_slot.v1",
        "key":         f"template:slot:{slot_name}",
        "name":        slot_name,
        "type":        slot_def.get("type", "string"),
        "phase":       slot_def.get("phase", "Yax"),
        "edge_kind":   slot_def.get("edge_kind", "@provides"),
        "default":     str(slot_def.get("default", "")),
        "unit":        slot_def.get("unit", "content_token"),
        "kit_id":      kit_id,
        "description": slot_def.get("description", ""),
        "css_vars":    slot_def.get("css_vars", []),
    }


def write_template_slots(out_dir: Path, dry_run: bool = False) -> Path:
    """Emit template:slot:* key-grammar rows for all slot_definitions in registry.json."""
    registry = _load_registry()
    slot_defs = registry.get("slot_definitions", {})
    if not slot_defs:
        print("[factory] no slot_definitions in registry — skipping")
        return out_dir

    records = [build_template_slot_record(name, defn) for name, defn in slot_defs.items()]
    out_file = out_dir / "template-slots.key-grammar.jsonl"
    print(f"[factory] template_slots -> {out_file} ({len(records)} slots)")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return out_file


# ── Format-specific prompts ─────────────────────────────────────────────────────

_FORMAT_SYSTEMS = {
    "greeting_bank": (
        "You generate greeting templates for a conversational AI system. "
        "Output a JSON array of strings only — no explanation, no markdown, "
        "just the JSON array. Each string is one complete greeting template."
    ),
    "key_grammar": (
        "You generate short domain phrases or sentences for a semantic grammar system. "
        "Output a JSON array of strings. Each string is one phrase or sentence."
    ),
    "capability_gap": (
        "You identify missing capabilities or knowledge gaps in an AI system. "
        "Output a JSON array of objects with fields: "
        "{\"gap\": \"short name\", \"description\": \"one sentence\", \"priority\": \"high|medium|low\"}."
    ),
    "track": (
        "You define micronaut track configurations for an AI routing system. "
        "Output a single JSON object matching this schema: "
        "{\"id\": str, \"description\": str, \"phase\": str, \"category\": str, "
        "\"governs\": [str], \"capabilities\": [str]}."
    ),
    "aiml": (
        "You generate AIML pattern/response pairs. "
        "Output a JSON array of objects: [{\"pattern\": str, \"response\": str}]. "
        "Patterns use uppercase, may contain wildcards *. Keep responses short."
    ),
    "control_flow": (
        "You generate control-flow operator entries for a semantic grammar system. "
        "Output a JSON array of objects. Each object must have these exact fields: "
        "{\"word\": str, \"key\": \"control:<word>\", \"class\": str, \"operator\": str, "
        "\"role\": str, \"sugar\": str, \"xcfe_op\": str, \"arity\": int, "
        "\"block_role\": \"open|mid|close|none\", \"phase\": str, "
        "\"authority_phase\": str, \"flow_phase\": str, \"emit_phase\": str, "
        "\"book\": str, \"requires\": [str], \"lexicon_source\": \"lfm_factory\"}. "
        "xcfe_op values: @if @else @then @for @and @or @not @cause @where @node @from @as @direct. "
        "phases from: Pop Wo Yax Sek Ch'en Xul."
    ),
    "scfml_pattern_learning": (
        "You generate SCFML semantic branch and pattern-learning candidates. "
        "Output a JSON array of objects only. Each object may contain: "
        "{\"condition\": str, \"then\": str, \"else\": str|null, \"unknown\": str, "
        "\"track\": str, \"candidate_kind\": \"aiml_pattern|aiml_template|semantic_branch|key_grammar|route_pattern\", "
        "\"target\": str, \"source_event\": str, \"epistemic_state\": \"UNKNOWN|MISS|SUPPORTED|CONTRADICTED\"}. "
        "UNKNOWN is the default. Never claim SUPPORTED or CONTRADICTED unless the caller marks the candidate user_verified."
    ),
    "code_grammar": (
        "You generate grammar token and edge definitions for a code parser. "
        "Output a single JSON object with two arrays: "
        "\"tokens\": [{\"type\": str, \"pattern\": str, \"captures\": [str]}], "
        "\"edges\": [{\"from_type\": str, \"to_type\": str, \"kind\": str, \"condition\": str}]. "
        "No explanation, no markdown — just the JSON object."
    ),
    "template_fill": (
        "You are filling template slot values for a project scaffold. "
        "Answer only the value for the specific slot asked — no explanation."
    ),
    "scaffold": (
        "You are a project scaffolding agent. Output every file using the exact format:\n"
        "### FILE: path/to/file\n"
        "```\n"
        "file content\n"
        "```\n"
        "No explanations, no other text — only FILE blocks. "
        "Each block must have a complete, working file. "
        "If write_file tools are available, use them instead of text blocks."
    ),
    "raw_jsonl": (
        "Output a JSON object or array. It will be written as-is to a JSONL log."
    ),
}

# Tools exposed to LFM for scaffold mode
_SCAFFOLD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file to the project with given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root (e.g. src/main.tsx)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content"
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to an existing file (or create it).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_file",
            "description": "Record a short description for a file without writing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["path", "description"],
            },
        },
    },
]


def build_auto_prompt(spec: dict) -> str:
    fmt    = spec.get("format", "raw_jsonl")
    phase  = spec.get("phase", "Pop")
    kind   = spec.get("kind", "canonical")
    domain = spec.get("domain", "technical assistant AI")
    count  = int(spec.get("count", 3))

    if fmt == "greeting_bank":
        return (
            f"Generate {count} greeting templates for a {domain}. "
            f"Phase: {phase} ({kind}). "
            "They should feel natural and match the phase character. "
            "Use placeholders like {cue}, {greeting}, {last}, {turns} where appropriate for learned/reactive phases."
        )
    if fmt == "key_grammar":
        ns = spec.get("namespace", spec.get("id", "domain"))
        return (
            f"Generate {count} short phrases or sentences for the '{ns}' domain "
            f"in a {phase}-phase semantic grammar. "
            "These will be used as vocabulary seeds for a routing system."
        )
    if fmt == "capability_gap":
        cap = spec.get("capability", spec.get("id", "general"))
        return (
            f"List {count} capability gaps for a '{cap}' subsystem in an AI routing runtime. "
            "For each gap: name, one-sentence description, priority (high/medium/low)."
        )
    if fmt == "track":
        track_id = spec.get("track_id", spec.get("id", "NEW-µ"))
        return (
            f"Define a micronaut track '{track_id}' for an AI routing system. "
            f"Phase: {phase}. Domain: {domain}. "
            "Include: id, description, category, governs (list of keywords), capabilities (list)."
        )
    if fmt == "aiml":
        topic = spec.get("topic", spec.get("id", "general"))
        return (
            f"Generate {count} AIML pattern/response pairs for topic '{topic}'. "
            "Patterns are uppercase with optional * wildcards. Responses are short."
        )
    if fmt == "control_flow":
        domain = spec.get("domain", "control flow")
        return (
            f"Generate {count} control-flow operator entries for the '{domain}' domain. "
            "Each entry describes one semantic operator (like if/else/for/when/because). "
            "Include: word, class, operator, role description, sugar syntax, xcfe_op code, "
            "arity (1 or 2), block_role (open/mid/close/none), and phase assignments."
        )
    if fmt == "scfml_pattern_learning":
        topic = spec.get("topic", spec.get("domain", spec.get("id", "semantic branching")))
        track = spec.get("track_id", spec.get("track", "PLAN-µ"))
        return (
            f"Generate {count} SCFML pattern-learning candidates for '{topic}' in track '{track}'. "
            "Each candidate should describe an observable pattern and a safe branch: "
            "condition -> then action, optional else only for a true contradicted complement, "
            "and unknown as preserve/retrieve/defer when evidence is missing. "
            "Prefer candidate_kind semantic_branch when a branch is present; otherwise use key_grammar, "
            "aiml_pattern, aiml_template, or route_pattern. Do not mark candidates SUPPORTED."
        )
    if fmt == "code_grammar":
        lang = spec.get("language", spec.get("id", "generic"))
        return (
            f"Generate grammar tokens and edges for the '{lang}' language. "
            f"Focus on {domain}. Include {count} tokens and enough edges to connect them. "
            "Each token needs: type (e.g. KEYWORD, IDENTIFIER, OPERATOR), pattern (regex), "
            "optional captures. Each edge: from_type, to_type, kind (e.g. follows/contains), "
            "optional condition."
        )

    if fmt == "scaffold":
        template  = spec.get("template", spec.get("domain", "vite react typescript"))
        name      = spec.get("name", spec.get("id", "app"))
        extra     = spec.get("features", "")
        feat_hint = f" Include: {extra}." if extra else ""
        return (
            f"Scaffold a {template} project named '{name}'.{feat_hint} "
            "Output each file using this exact format — no other text:\n\n"
            "### FILE: relative/path/to/file\n"
            "```\n"
            "file content here\n"
            "```\n\n"
            "Include package.json, index.html, src/main entry point, and at least one component. "
            "Write complete working file contents."
        )

    return spec.get("prompt", f"Generate content for: {spec.get('id', 'unknown')}")


# ── Output writers ──────────────────────────────────────────────────────────────

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_greeting_bank(spec: dict, templates: list[str], out_dir: Path, dry_run: bool) -> dict:
    phase     = spec.get("phase", "Pop")
    kind      = spec.get("kind", "canonical")
    bank_path = out_dir / spec.get("out", "greeting-bank.json")

    existing: dict = {}
    if bank_path.exists():
        try:
            existing = json.loads(bank_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    greetings: list = existing.get("greetings", [])
    # Find or create phase entry
    entry = next((g for g in greetings if g.get("phase") == phase and g.get("kind") == kind), None)
    if entry is None:
        entry = {"phase": phase, "kind": kind, "templates": []}
        greetings.append(entry)
    for t in templates:
        if t not in entry["templates"]:
            entry["templates"].append(t)

    result = {
        "schema":      existing.get("schema", "key.grammar.greeting_bank.v1"),
        "description": existing.get("description", "LFM-generated greeting bank."),
        "greetings":   greetings,
    }

    if not dry_run:
        _ensure_parent(bank_path)
        bank_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"file": str(bank_path), "phase": phase, "kind": kind, "added": len(templates)}


def write_key_grammar(spec: dict, templates: list[str], out_dir: Path, dry_run: bool) -> dict:
    ns         = spec.get("namespace", spec.get("id", "GEN").upper())
    phase      = spec.get("phase", "Pop")
    class_name = spec.get("class_name", ns)
    out_name   = spec.get("out", f"{ns.lower()}.key-grammar.jsonl")
    out_path   = out_dir / out_name
    lex_source = str(out_path)

    records = []
    for i, template in enumerate(templates, start=1):
        key = f"{ns}:{phase.upper()}:{i}"
        rec = build_key_grammar_record(
            template   = template,
            key        = key,
            phase      = phase,
            lexicon_source = lex_source,
            class_name = class_name,
            word       = f"{ns.lower()}-{i}",
        )
        records.append(rec)

    if not dry_run:
        _ensure_parent(out_path)
        with open(out_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"file": str(out_path), "records": len(records)}


def write_capability_gap(spec: dict, gaps: list[dict], out_dir: Path, dry_run: bool) -> dict:
    out_name = spec.get("out", "control-capability-gaps.jsonl")
    out_path = out_dir / out_name
    ts       = datetime.now().isoformat(timespec="seconds")

    rows = []
    for gap in gaps:
        row = {
            "id":          f"{spec['id']}.{gap.get('gap', 'unknown').replace(' ', '_')}",
            "source_spec": spec["id"],
            "generated":   ts,
            **gap,
        }
        rows.append(row)

    if not dry_run:
        _ensure_parent(out_path)
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"file": str(out_path), "records": len(rows)}


def write_track(spec: dict, track_obj: dict, out_dir: Path, dry_run: bool) -> dict:
    track_id = track_obj.get("id", spec.get("track_id", spec["id"]))
    out_name = spec.get("out", f"{track_id}.semantic-tracks.v1.json")
    out_path = (out_dir.parent / "tracks" / out_name)

    merged = {
        "@no-eval":   True,
        "generated":  datetime.now().isoformat(timespec="seconds"),
        **track_obj,
    }
    merged.setdefault("id",          track_id)
    merged.setdefault("phase",       spec.get("phase", "Pop"))
    merged.setdefault("category",    spec.get("domain", "general"))
    merged.setdefault("confidence",  0.7)
    merged.setdefault("governs",     [])
    merged.setdefault("capabilities", [])

    if not dry_run:
        _ensure_parent(out_path)
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"file": str(out_path), "id": track_id}


def write_aiml(spec: dict, pairs: list[dict], out_dir: Path, dry_run: bool) -> dict:
    out_name = spec.get("out", f"{spec['id']}.aiml")
    out_path = (out_dir.parent / "alice" / out_name)

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<aiml version="2.0">']
    for pair in pairs:
        pattern  = str(pair.get("pattern", "")).upper().strip()
        response = str(pair.get("response", "")).strip()
        if not pattern or not response:
            continue
        lines.append(f"  <category>")
        lines.append(f"    <pattern>{pattern}</pattern>")
        lines.append(f"    <template>{response}</template>")
        lines.append(f"  </category>")
    lines.append("</aiml>")

    if not dry_run:
        _ensure_parent(out_path)
        out_path.write_text("\n".join(lines), encoding="utf-8")

    return {"file": str(out_path), "pairs": len(pairs)}


def write_control_flow(spec: dict, entries: list[dict], out_dir: Path, dry_run: bool) -> dict:
    ns       = spec.get("namespace", "control")
    out_name = spec.get("out", f"{ns}.key-grammar.jsonl")
    out_path = out_dir / out_name

    rows = []
    for entry in entries:
        word = str(entry.get("word", "")).strip().lower()
        if not word:
            continue
        row = {
            "word":             word,
            "key":              entry.get("key", f"{ns}:{word}"),
            "class":            entry.get("class", "CONTROL"),
            "operator":         entry.get("operator", "SEQUENCE"),
            "role":             entry.get("role", word),
            "sugar":            entry.get("sugar", f"{word} <expression>"),
            "xcfe_op":          entry.get("xcfe_op", "@if"),
            "arity":            int(entry.get("arity", 1)),
            "block_role":       entry.get("block_role", "none"),
            "phase":            entry.get("phase", spec.get("phase", "Pop")),
            "authority_phase":  entry.get("authority_phase", "Pop"),
            "flow_phase":       entry.get("flow_phase", "Pop"),
            "emit_phase":       entry.get("emit_phase", "Wo"),
            "book":             entry.get("book", f"ELIZA-1:{entry.get('operator', 'SEQUENCE')}"),
            "requires":         entry.get("requires", []),
            "lexicon_source":   entry.get("lexicon_source", "lfm_factory"),
        }
        rows.append(row)

    if not dry_run:
        _ensure_parent(out_path)
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"file": str(out_path), "records": len(rows)}


def write_scfml_pattern_learning(spec: dict, entries: list[dict], out_dir: Path, dry_run: bool) -> dict:
    namespace = _scfml_safe_ns(str(spec.get("namespace", spec.get("id", "scfml"))))
    out_name = spec.get("out", f"{namespace}.scfml-pattern-learning.key-grammar.jsonl")
    out_path = out_dir / out_name

    rows: list[dict] = []
    branch_count = 0
    for i, raw_entry in enumerate(entries, start=1):
        entry = raw_entry if isinstance(raw_entry, dict) else {
            "candidate_kind": "aiml_pattern",
            "target": str(raw_entry),
        }
        branch = build_scfml_branch_record(entry, spec, i, namespace)
        branch_key = ""
        if branch is not None:
            branch_key = branch["key"]
            rows.append(branch)
            branch_count += 1
        rows.append(build_scfml_pattern_learning_record(entry, spec, i, namespace, branch_key))

    if not dry_run:
        _ensure_parent(out_path)
        with open(out_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "file": str(out_path),
        "records": len(rows),
        "branch_records": branch_count,
        "pattern_records": len(rows) - branch_count,
    }


def write_code_grammar(spec: dict, grammar: dict, out_dir: Path, dry_run: bool) -> dict:
    lang     = spec.get("language", spec.get("id", "generic"))
    out_name = spec.get("out", f"{lang}.code-grammar.json")
    out_path = out_dir / out_name

    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    tokens = existing.get("tokens", []) + grammar.get("tokens", [])
    edges  = existing.get("edges",  []) + grammar.get("edges",  [])

    result = {
        "language": existing.get("language", lang),
        "version":  existing.get("version", "lfm_factory.1"),
        "tokens":   tokens,
        "edges":    edges,
        "adapter_layer":                    "AST",
        "ast_name":                         "Abstract Syntax Transpiler",
        "downstream_execution_authority":   "K'UHUL",
        "generated":                        datetime.now().isoformat(timespec="seconds"),
    }

    if not dry_run:
        _ensure_parent(out_path)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"file": str(out_path), "tokens": len(tokens), "edges": len(edges)}


def parse_file_blocks(text: str) -> list[dict]:
    """Parse ### FILE: path / ``` content ``` blocks from LFM plain-text scaffold output.

    Also handles common variants:
        ## File: path
        **path:**
        // === path ===
    """
    results = []
    # Primary: ### FILE: path\n```[lang]\ncontent\n```
    primary = re.finditer(
        r"#{1,4}\s*[Ff][Ii][Ll][Ee]:\s*([^\n]+)\n```[^\n]*\n(.*?)```",
        text, re.DOTALL,
    )
    for m in primary:
        path = m.group(1).strip()
        content = m.group(2)
        results.append({"path": path, "content": content})

    if results:
        return results

    # Fallback: bold **path:** then fenced block
    fallback = re.finditer(
        r"\*\*([^*\n]+)\*\*:?\s*\n```[^\n]*\n(.*?)```",
        text, re.DOTALL,
    )
    for m in fallback:
        path = m.group(1).strip()
        content = m.group(2)
        if "/" in path or "." in path:
            results.append({"path": path, "content": content})

    if results:
        return results

    # Last resort: any fenced block preceded by a line that looks like a filename
    last = re.finditer(
        r"(?:^|\n)([^\n`#*]{1,80}(?:\.[a-z]{1,5}|\/[^\n]{1,40}))\s*\n```[^\n]*\n(.*?)```",
        text, re.DOTALL,
    )
    for m in last:
        path = m.group(1).strip().lstrip("-").strip()
        content = m.group(2)
        results.append({"path": path, "content": content})

    return results


def execute_scaffold(
    spec: dict,
    tool_calls: list[dict],
    content_text: str,
    root_dir: Path,
    dry_run: bool,
) -> dict:
    """Execute write_file / append_file calls from tool calls or parsed text blocks."""
    written: list[str] = []
    described: list[str] = []
    manifest: list[dict] = []
    _written_set: set[str] = set()

    def _write(rel_path: str, file_content: str, action: str = "write_file") -> None:
        if action == "write_file" and rel_path in _written_set:
            return  # skip duplicate — first write wins
        _written_set.add(rel_path)
        target = root_dir / rel_path
        if not dry_run:
            _ensure_parent(target)
            if action == "append_file" and target.exists():
                with open(target, "a", encoding="utf-8") as f:
                    f.write(file_content)
            else:
                target.write_text(file_content, encoding="utf-8")
        written.append(rel_path)
        manifest.append({"path": rel_path, "action": action,
                         "bytes": len(file_content.encode())})

    # 1. Structured tool calls (preferred)
    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        rel_path = args.get("path", "").lstrip("/\\")
        if not rel_path:
            continue
        if name in ("write_file", "append_file"):
            _write(rel_path, args.get("content", ""), name)
        elif name == "describe_file":
            desc = args.get("description", "")
            described.append(rel_path)
            manifest.append({"path": rel_path, "action": "describe", "description": desc})

    # 2. Parse ### FILE: blocks from plain text
    if not tool_calls and content_text:
        blocks = parse_file_blocks(content_text)
        if blocks:
            for block in blocks:
                rel = block["path"].lstrip("/\\")
                _write(rel, block["content"])
        else:
            # Nothing parseable — save raw text so output isn't lost
            notes = root_dir / "SCAFFOLD_NOTES.md"
            if not dry_run:
                _ensure_parent(notes)
                notes.write_text(content_text, encoding="utf-8")
            written.append("SCAFFOLD_NOTES.md")
            manifest.append({"path": "SCAFFOLD_NOTES.md", "action": "write_file",
                              "bytes": len(content_text.encode()),
                              "note": "LFM did not use tool calls or FILE blocks"})

    manifest_path = root_dir / ".scaffold_manifest.json"
    if not dry_run and manifest:
        _ensure_parent(manifest_path)
        manifest_path.write_text(
            json.dumps({"spec_id": spec.get("id"),
                        "generated": datetime.now().isoformat(timespec="seconds"),
                        "files": manifest}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "file": str(root_dir),
        "files_written": len(written),
        "files_described": len(described),
        "manifest": str(manifest_path),
    }


TEMPLATES_DIR  = DATA_DIR / "templates"
REGISTRY_PATH  = TEMPLATES_DIR / "registry.json"
REPO_ROOT      = DATA_DIR.parent


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"kits": []}


def _kit_by_id(kit_id: str) -> dict | None:
    reg = _load_registry()
    return next((k for k in reg.get("kits", []) if k["id"] == kit_id), None)


def _resolve_source(source: str) -> Path:
    """Resolve a kit source path — absolute, or relative to repo root."""
    p = Path(source)
    if p.is_absolute():
        return p
    return (REPO_ROOT / source).resolve()


def _apply_slots(text: str, slots: dict[str, str]) -> str:
    """Replace {{slot}} occurrences with slot values."""
    for k, v in slots.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def _expand_lfm_markers(text: str, context: str, port: int = LFM_PORT) -> str:
    """Find {{lfm:expand — instruction}} markers and call LFM to fill them."""
    def _replace(m: re.Match) -> str:
        instruction = m.group(1).strip()
        try:
            result = lfm_text(
                f"{instruction}\n\nContext: {context}\nOutput only the replacement content.",
                max_tokens=256, temperature=0.4,
            )
            return result.strip()
        except Exception:
            return f"<!-- lfm:expand failed: {instruction} -->"
    return re.sub(r"\{\{lfm:([^}]+)\}\}", _replace, text)


def _inject_html_slots(html: str, slots: dict[str, str]) -> str:
    """Inject brand tokens into an ASX HTML kit.

    Strategy:
    1. Replace {{slot}} markers (if any were hand-placed).
    2. Inject a :root override block right before </style> or </head>
       so it overrides the original token values without touching source vars.
    3. Swap <title> content.
    """
    html = _apply_slots(html, slots)

    # Build CSS override block for the house token set
    accent    = slots.get("brand_accent",    "#16f2aa")
    secondary = slots.get("brand_secondary", "#38bdf8")
    bg        = slots.get("brand_bg",        "#020617")
    surface   = slots.get("brand_surface",   "#0f172a")
    text_col  = slots.get("brand_text",      "#e2e8f0")

    override = (
        "\n/* ── TEMPLATE-µ brand override ── */\n"
        ":root {\n"
        f"  --accent:          {accent};\n"
        f"  --asx-primary:     {accent};\n"
        f"  --atomic-border:   {accent};\n"
        f"  --kuhul-color:     {accent};\n"
        f"  --accent-teal:     {accent};\n"
        f"  --alpha-green:     {accent};\n"
        f"  --asx-secondary:   {secondary};\n"
        f"  --accent-blue:     {secondary};\n"
        f"  --bg:              {bg};\n"
        f"  --bg-primary:      {bg};\n"
        f"  --asx-bg:          {bg};\n"
        f"  --dark-pool:       {bg};\n"
        f"  --asx-surface:     {surface};\n"
        f"  --asx-color:       {text_col};\n"
        "}\n"
    )

    # Inject before </style> if present, else before </head>
    if "</style>" in html:
        html = html.replace("</style>", override + "</style>", 1)
    elif "</head>" in html:
        html = html.replace("</head>", f"<style>{override}</style>\n</head>", 1)

    # Swap <title>
    app_name = slots.get("app_name", "")
    if app_name:
        html = re.sub(r"(<title>)[^<]*(</title>)", rf"\g<1>{app_name}\g<2>", html, count=1)

    # Inject brand + chat JS before </body>
    scripts = (
        '\n<!-- KHANARY WebUI injections -->\n'
        '<script src="/khanary-brand.js"></script>\n'
        '<script src="/khanary-chat.js"></script>\n'
    )
    if "</body>" in html:
        html = html.replace("</body>", scripts + "</body>", 1)
    else:
        html += scripts

    return html


def apply_template_kit(
    spec: dict,
    out_dir: Path,
    dry_run: bool,
    verbose: bool = False,
) -> dict:
    """Fill a template kit. Dispatches on source_type: html | vite."""
    kit_id = spec.get("template", spec.get("id", "vite-react-ts"))

    # Look up kit in registry first, fall back to local kit.json
    kit_reg = _kit_by_id(kit_id)
    source_type = (kit_reg or {}).get("source_type", "vite")

    # Build slot values from spec (registry slot names are informational only)
    reg_house = _load_registry().get("house_tokens", {})
    slots: dict[str, str] = {**reg_house}           # start with house defaults
    for k, v in spec.items():
        if not k.startswith("_") and isinstance(v, (str, int, float, bool)):
            slots[k] = str(v)

    feats = spec.get("features", "")
    if isinstance(feats, list):
        feats = ", ".join(feats)
    slots["features"] = feats

    # Resolve output root
    raw_out = spec.get("out")
    if raw_out:
        root_dir = Path(raw_out).expanduser().resolve()
    elif str(out_dir) != str(DEFAULT_OUT):
        root_dir = out_dir
    else:
        root_dir = out_dir.parent / "scaffold" / kit_id

    expand_lfm  = spec.get("expand_lfm", True)
    context_hint = f"App: {slots.get('app_name','app')}, features: {feats or 'none'}"

    # ── HTML kit path ──────────────────────────────────────────────────────────
    if source_type == "html":
        src_path = _resolve_source((kit_reg or {}).get("source", ""))
        if not src_path.exists():
            return {"ok": False, "error": f"HTML source not found: {src_path}"}

        html = src_path.read_text(encoding="utf-8")
        html = _inject_html_slots(html, slots)
        if expand_lfm and "{{lfm:" in html:
            html = _expand_lfm_markers(html, context_hint)

        app_name = slots.get("app_name", kit_id)
        out_name = spec.get("out_file", "index.html")
        target   = root_dir / out_name
        if not dry_run:
            _ensure_parent(target)
            target.write_text(html, encoding="utf-8")
            # Copy JS assets and registry into the webui dir
            webui_assets = REPO_ROOT / "data" / "webui"
            for asset in ("khanary-brand.js", "khanary-chat.js"):
                src = webui_assets / asset
                if src.exists():
                    import shutil
                    shutil.copy2(src, root_dir / asset)
            # Write registry for the template switcher
            reg_dest = root_dir / "khanary-registry.json"
            if REGISTRY_PATH.exists():
                shutil.copy2(REGISTRY_PATH, reg_dest)
        if verbose:
            print(f"  [kit] {out_name} ({len(html)} chars) + brand.js + chat.js + registry")
        return {
            "ok": True,
            "file": str(target),
            "kit": kit_id,
            "files_written": 1,
            "slots": {k: v for k, v in slots.items() if k in (kit_reg or {}).get("slots", list(slots.keys()))},
        }

    # ── Vite multi-file kit path ───────────────────────────────────────────────
    kit_dir       = TEMPLATES_DIR / kit_id
    kit_meta_path = kit_dir / "kit.json"
    if not kit_dir.exists():
        return {"ok": False, "error": f"kit not found: {kit_dir}"}
    if not kit_meta_path.exists():
        return {"ok": False, "error": f"kit.json missing in {kit_dir}"}

    kit_meta  = json.loads(kit_meta_path.read_text(encoding="utf-8"))
    for slot_name, slot_def in kit_meta.get("slots", {}).items():
        slots.setdefault(slot_name, str(slot_def.get("default", "")))

    file_list = kit_meta.get("files", [])
    written: list[str] = []

    for rel in file_list:
        src_path = kit_dir / rel
        if not src_path.exists():
            candidates = list(kit_dir.rglob(Path(rel).name))
            src_path = candidates[0] if candidates else None
        if src_path is None or not src_path.exists():
            if verbose:
                print(f"  [kit] SKIP {rel}")
            continue

        content = src_path.read_text(encoding="utf-8")
        content = _apply_slots(content, slots)
        if expand_lfm and "{{lfm:" in content:
            content = _expand_lfm_markers(content, context_hint)

        target = root_dir / rel
        if not dry_run:
            _ensure_parent(target)
            target.write_text(content, encoding="utf-8")
        written.append(rel)
        if verbose:
            print(f"  [kit] {rel} ({len(content)} chars)")

    return {
        "ok": True,
        "file": str(root_dir),
        "kit": kit_id,
        "files_written": len(written),
        "slots": {k: slots[k] for k in kit_meta.get("slots", {}) if k in slots},
    }


def write_raw_jsonl(spec: dict, parsed: Any, out_dir: Path, dry_run: bool) -> dict:
    out_name = spec.get("out", f"{spec['id']}.jsonl")
    out_path = out_dir / out_name
    row = {
        "id": spec["id"],
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data": parsed,
    }
    if not dry_run:
        _ensure_parent(out_path)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"file": str(out_path)}


# ── Spec processor ──────────────────────────────────────────────────────────────

def process_spec(spec: dict, out_dir: Path, dry_run: bool = False, verbose: bool = False) -> dict:
    spec_id = spec.get("id", "?")
    fmt     = spec.get("format", "raw_jsonl")

    prompt  = spec.get("prompt") or build_auto_prompt(spec)
    system  = _FORMAT_SYSTEMS.get(fmt, "")
    count   = int(spec.get("count", 3))

    if verbose:
        print(f"  [{spec_id}] sending to LFM ({fmt})...")

    # Template slots — no LFM needed; reads slot_definitions from registry
    if fmt == "template_slot":
        out_file = write_template_slots(out_dir, dry_run)
        return {"id": spec_id, "ok": True, "format": "template_slot", "file": str(out_file)}

    # Template fill — kit-backed, no file list needed
    if fmt == "template_fill":
        result = apply_template_kit(spec, out_dir, dry_run, verbose=verbose)
        if not result.get("ok"):
            return {"id": spec_id, "ok": False, **result}
        return {"id": spec_id, "ok": True, "format": "template_fill", **result}

    # Scaffold — multi-turn: list files then generate each one
    if fmt == "scaffold":
        raw_out = spec.get("out")
        if raw_out:
            root_dir = Path(raw_out).expanduser().resolve()
        elif str(out_dir) != str(DEFAULT_OUT):
            root_dir = out_dir
        else:
            root_dir = out_dir.parent / "scaffold" / spec_id

        template = spec.get("template", "vite react typescript")
        name     = spec.get("name", spec_id)
        features = spec.get("features", "")
        feat_hint = f" with {features}" if features else ""
        max_tok  = int(spec.get("max_tokens", 512))

        # Turn 1: get file list
        list_prompt = (
            f"List all source files for a {template} project named '{name}'{feat_hint}. "
            "Use relative paths from the project root — e.g. src/main.tsx, index.html, package.json. "
            "One path per line. No node_modules, no dist, no lock files, no explanations."
        )
        try:
            file_list_raw = lfm_text(list_prompt, max_tokens=256)
        except RuntimeError as exc:
            return {"id": spec_id, "ok": False, "error": str(exc)}

        _BAD_PREFIXES = ("node_modules", "dist/", "build/", ".git", "package-lock", "yarn.lock", "pnpm-lock")
        seen: set[str] = set()
        file_paths = []
        for ln in file_list_raw.splitlines():
            p = ln.strip().lstrip("-*0123456789.) ").strip()
            if not p or not ("/" in p or "." in p):
                continue
            if p.startswith("#"):
                continue
            if any(p.startswith(bad) or p == bad for bad in _BAD_PREFIXES):
                continue
            if p in seen:
                continue
            seen.add(p)
            file_paths.append(p)

        if verbose:
            print(f"  [{spec_id}] file list ({len(file_paths)}): {file_paths[:8]}")

        if not file_paths:
            return {"id": spec_id, "ok": False,
                    "error": "LFM returned no file paths", "raw": file_list_raw}

        # Turn 2+: generate each file
        all_calls: list[dict] = []
        for fpath in file_paths:
            file_prompt = (
                f"Write the complete content of '{fpath}' for a {template} project "
                f"named '{name}'{feat_hint}. "
                "Output only the file content — no explanation, no markdown fence."
            )
            try:
                content = lfm_text(file_prompt, max_tokens=max_tok, temperature=0.3)
            except RuntimeError as exc:
                if verbose:
                    print(f"  [{spec_id}] SKIP {fpath}: {exc}")
                continue
            all_calls.append({"name": "write_file",
                               "arguments": {"path": fpath, "content": content}})
            if verbose:
                print(f"  [{spec_id}] {fpath} ({len(content)} chars)")

        try:
            result = execute_scaffold(spec, all_calls, "", root_dir, dry_run)
        except Exception as exc:
            return {"id": spec_id, "ok": False, "error": str(exc)}
        return {"id": spec_id, "ok": True, "format": "scaffold", **result}

    try:
        raw = lfm_text(prompt, system=system, max_tokens=max(256, count * 120))
    except RuntimeError as exc:
        return {"id": spec_id, "ok": False, "error": str(exc)}

    if verbose:
        preview = raw[:120].replace("\n", " ")
        print(f"  [{spec_id}] response: {preview}...")

    # Dispatch
    try:
        if fmt == "greeting_bank":
            templates = parse_string_list(raw)[:count * 2]
            if not templates:
                return {"id": spec_id, "ok": False, "error": "no templates parsed", "raw": raw}
            result = write_greeting_bank(spec, templates, out_dir, dry_run)
            # Also write key-grammar for each template
            kg_spec = dict(spec, format="key_grammar",
                           namespace=spec.get("namespace", "GREETING"),
                           out=spec.get("kg_out", "greeting-bank.key-grammar.jsonl"))
            write_key_grammar(kg_spec, templates, out_dir, dry_run)

        elif fmt == "key_grammar":
            templates = parse_string_list(raw)[:count * 2]
            if not templates:
                return {"id": spec_id, "ok": False, "error": "no phrases parsed", "raw": raw}
            result = write_key_grammar(spec, templates, out_dir, dry_run)

        elif fmt == "capability_gap":
            parsed = parse_json_response(raw)
            if isinstance(parsed, list):
                gaps = [g if isinstance(g, dict) else {"gap": str(g), "description": str(g), "priority": "medium"} for g in parsed]
            else:
                lines = parse_string_list(raw)
                gaps = [{"gap": ln, "description": ln, "priority": "medium"} for ln in lines]
            result = write_capability_gap(spec, gaps, out_dir, dry_run)

        elif fmt == "track":
            parsed = parse_json_response(raw)
            if not isinstance(parsed, dict):
                return {"id": spec_id, "ok": False, "error": "track response was not a JSON object", "raw": raw}
            result = write_track(spec, parsed, out_dir, dry_run)

        elif fmt == "aiml":
            parsed = parse_json_response(raw)
            if isinstance(parsed, list):
                pairs = [p if isinstance(p, dict) else {} for p in parsed]
            else:
                pairs = []
            result = write_aiml(spec, pairs, out_dir, dry_run)

        elif fmt == "control_flow":
            parsed = parse_json_response(raw)
            if isinstance(parsed, list):
                entries = [e if isinstance(e, dict) else {} for e in parsed]
            else:
                return {"id": spec_id, "ok": False, "error": "control_flow response was not a JSON array", "raw": raw}
            result = write_control_flow(spec, entries, out_dir, dry_run)

        elif fmt == "scfml_pattern_learning":
            parsed = parse_json_response(raw)
            if isinstance(parsed, dict):
                entries = [parsed]
            elif isinstance(parsed, list):
                entries = [e if isinstance(e, dict) else {"target": str(e)} for e in parsed]
            else:
                entries = [
                    {"candidate_kind": "aiml_pattern", "target": line}
                    for line in parse_string_list(raw)
                ]
            if not entries:
                return {"id": spec_id, "ok": False, "error": "no SCFML candidates parsed", "raw": raw}
            result = write_scfml_pattern_learning(spec, entries, out_dir, dry_run)

        elif fmt == "code_grammar":
            parsed = parse_json_response(raw)
            if not isinstance(parsed, dict):
                return {"id": spec_id, "ok": False, "error": "code_grammar response was not a JSON object", "raw": raw}
            result = write_code_grammar(spec, parsed, out_dir, dry_run)

        else:  # raw_jsonl
            parsed = parse_json_response(raw) or raw
            result = write_raw_jsonl(spec, parsed, out_dir, dry_run)

    except Exception as exc:
        return {"id": spec_id, "ok": False, "error": str(exc), "raw": raw}

    return {"id": spec_id, "ok": True, "format": fmt, **result}


# ── CLI ─────────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser().resolve()
    specs_file = Path(args.specs).expanduser().resolve()

    if not specs_file.exists():
        print(f"[error] specs file not found: {specs_file}")
        sys.exit(1)

    specs = []
    with open(specs_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                specs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping malformed spec line: {exc}")

    print(f"[factory] {len(specs)} spec(s), out={out_dir}, dry_run={args.dry_run}")
    ok_count = 0
    try:
        for spec in specs:
            result = process_spec(spec, out_dir, dry_run=args.dry_run, verbose=True)
            status = "ok" if result.get("ok") else "FAIL"
            print(f"  [{result.get('id','?')}] {status}", end="")
            if result.get("ok"):
                file_ = result.get("file", "")
                extra = {k: v for k, v in result.items() if k not in ("id", "ok", "format", "file")}
                if file_:
                    print(f" → {Path(file_).name}", end="")
                if extra:
                    print(f" {extra}", end="")
                ok_count += 1
            else:
                print(f" {result.get('error', '')} | raw: {str(result.get('raw',''))[:80]}", end="")
            print()
            if not args.no_delay:
                time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n[factory] interrupted")

    print(f"[factory] done: {ok_count}/{len(specs)} ok")


def cmd_ask(args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser().resolve()
    spec: dict = {
        "id":     args.id or "cli-ask",
        "format": args.format,
        "prompt": args.prompt,
        "count":  args.count,
        "phase":  args.phase,
    }
    # Fold in named optional fields if provided
    for field in ("language", "namespace", "kind", "domain",
                  "track_id", "capability", "topic", "class_name",
                  "template", "name", "features", "max_tokens"):
        val = getattr(args, field, None)
        if val is not None:
            spec[field] = val
    # --set key=value overrides everything
    for kv in (args.set or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            spec[k.strip()] = v.strip()
    result = process_spec(spec, out_dir, dry_run=args.dry_run, verbose=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_formats(_args: argparse.Namespace) -> None:
    rows = [
        ("greeting_bank",  "greeting-bank.json + key-grammar JSONL",
         "phase, kind, count, domain"),
        ("key_grammar",    "*.key-grammar.jsonl",
         "namespace, phase, class_name, count"),
        ("control_flow",   "*.key-grammar.jsonl (control schema)",
         "namespace, domain, count"),
        ("scfml_pattern_learning", "*.scfml-pattern-learning.key-grammar.jsonl",
         "namespace, track_id, topic, candidate_kind, count"),
        ("code_grammar",   "data/grammar/*.code-grammar.json",
         "language, domain, count"),
        ("capability_gap", "control-capability-gaps.jsonl",
         "capability, count"),
        ("track",          "data/tracks/*.semantic-tracks.v1.json",
         "track_id, phase, domain"),
        ("aiml",           "data/alice/*.aiml",
         "topic, count"),
        ("template_slot",  "data/grammar/template-slots.key-grammar.jsonl (no LFM)",
         "(no args — reads slot_definitions from registry.json)"),
        ("template_fill",  "any directory (kit slots filled, LFM expands markers)",
         "template, app_name, brand_color, brand_dark, features, rpc_port, out"),
        ("scaffold",       "any directory (LFM multi-turn file-by-file)",
         "template, name, features, out, max_tokens"),
        ("raw_jsonl",      "*.jsonl (append)",
         "out, count"),
    ]
    print(f"{'Format':<16}  {'Output':<42}  Spec fields")
    print("-" * 80)
    for fmt, out, fields in rows:
        print(f"  {fmt:<14}  {out:<42}  {fields}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LFM distillation dataset factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Process a JSONL spec file")
    run_p.add_argument("specs",             help="JSONL spec file")
    run_p.add_argument("--out",             default=str(DEFAULT_OUT), help="Output directory")
    run_p.add_argument("--dry-run",         action="store_true", help="Parse and plan without writing files")
    run_p.add_argument("--no-delay",        action="store_true", help="Skip inter-request delay")

    ask_p = sub.add_parser("ask", help="One-shot generation from CLI")
    ask_p.add_argument("prompt",            help="Prompt to send to LFM")
    ask_p.add_argument("--format",          default="raw_jsonl",  help="Output format")
    ask_p.add_argument("--out",             default=str(DEFAULT_OUT))
    ask_p.add_argument("--count",           type=int, default=3)
    ask_p.add_argument("--phase",           default="Pop")
    ask_p.add_argument("--dry-run",         action="store_true")
    ask_p.add_argument("--id",              default=None,         help="Record id (default: cli-ask)")
    ask_p.add_argument("--language",        default=None,         help="Language hint (code_grammar)")
    ask_p.add_argument("--namespace",       default=None,         help="Namespace prefix (key_grammar / control_flow)")
    ask_p.add_argument("--kind",            default=None,         help="Greeting kind")
    ask_p.add_argument("--domain",          default=None,         help="Domain hint")
    ask_p.add_argument("--track-id",        dest="track_id",      default=None, help="Track id (track format)")
    ask_p.add_argument("--capability",      default=None,         help="Capability name (capability_gap)")
    ask_p.add_argument("--topic",           default=None,         help="Topic (aiml)")
    ask_p.add_argument("--class-name",      dest="class_name",    default=None, help="Class label (key_grammar)")
    ask_p.add_argument("--template",        default=None,         help="Project template (scaffold): 'vite react typescript'")
    ask_p.add_argument("--name",            default=None,         help="Project name (scaffold)")
    ask_p.add_argument("--features",        default=None,         help="Extra features to include (scaffold)")
    ask_p.add_argument("--max-tokens",      dest="max_tokens",    type=int, default=None, help="Override max tokens")
    ask_p.add_argument("--set",             action="append",      metavar="KEY=VAL",
                       help="Set any spec field: --set language=klsl --set count=5")

    sub.add_parser("formats", help="List available output formats and their spec fields")

    args = parser.parse_args()
    if   args.cmd == "run":     cmd_run(args)
    elif args.cmd == "ask":     cmd_ask(args)
    elif args.cmd == "formats": cmd_formats(args)
    else:                       parser.print_help()


if __name__ == "__main__":
    main()
