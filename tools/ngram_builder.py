#!/usr/bin/env python3
"""
Build Mx2LM n-gram payload/checkpoint JSON from text or files.

This mirrors the runtime Mx2LM surface used by key_query:
- bigrams/trigrams
- coarse n-gram structural priors
- control bindings
- semantic nodes + relations
- taxonomy relations (hyponym/hypernym/co-hyponym/synonym/meronym patterns)
- latent relation templates (e.g., capital of -> is_capital_of)
- key_references / ngram_references (manifest-backed)
- structured_composition
- optional binary snapshot output (snapshot-only association priors)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCX_MAX_DEPTH = 64
MX2LM_NGRAM_BIN_MAGIC = b"MX2NGRAM"
MX2LM_NGRAM_BIN_VERSION = 1
MX2LM_NGRAM_SECTION_IDS: dict[str, int] = {
    "bigrams": 1,
    "trigrams": 2,
    "coarse_bigrams": 3,
    "coarse_trigrams": 4,
}


def trim_copy(value: str) -> str:
    return value.strip()


def lower_copy(value: str) -> str:
    return value.lower()


def contains_phrase_ci(text_lower: str, phrase_lower: str) -> bool:
    return phrase_lower in text_lower


def infer_micronaut_from_text(text: str) -> str:
    low = lower_copy(text)
    if (
        low.startswith("tools:")
        or contains_phrase_ci(low, "@tools")
        or low.startswith("status:")
        or contains_phrase_ci(low, "@status")
        or low.startswith("cube:")
        or contains_phrase_ci(low, "@cube")
        or low.startswith("world:")
        or contains_phrase_ci(low, "@world")
        or low.startswith("field:")
        or contains_phrase_ci(low, "@field")
        or low.startswith("fieldsvg:")
        or contains_phrase_ci(low, "@fieldsvg")
        or low.startswith("relgraph:")
        or contains_phrase_ci(low, "@relgraph")
        or low.startswith("fold:")
        or contains_phrase_ci(low, "@fold")
        or low.startswith("keyquery:")
        or contains_phrase_ci(low, "@keyquery")
        or low.startswith("dispatch:")
        or contains_phrase_ci(low, "@dispatch")
        or low.startswith("manifest:")
        or contains_phrase_ci(low, "@manifest_query")
        or low.startswith("confidence:")
        or contains_phrase_ci(low, "@confidence")
        or low.startswith("replay:")
        or contains_phrase_ci(low, "@replay")
        or low.startswith("sandbox:")
        or contains_phrase_ci(low, "@sandbox")
        or low.startswith("infer:")
        or contains_phrase_ci(low, "@infer")
        or low.startswith("listkeys:")
        or contains_phrase_ci(low, "@listkeys")
        or contains_phrase_ci(low, "tool calls")
    ):
        return "TOOL-\u00b5"
    if (
        low.startswith("code:")
        or contains_phrase_ci(low, "@code")
        or low.startswith("debug:")
        or contains_phrase_ci(low, "@debug")
    ):
        return "coder"
    if (
        low.startswith("summarize:")
        or contains_phrase_ci(low, "@summarize")
        or low.startswith("explain:")
        or contains_phrase_ci(low, "@explain")
        or low.startswith("meeting:")
        or contains_phrase_ci(low, "@meeting")
    ):
        return "reasoner"
    if (
        low.startswith("rewrite:")
        or contains_phrase_ci(low, "@rewrite")
        or low.startswith("translate:")
        or contains_phrase_ci(low, "@translate")
        or low.startswith("brainstorm:")
        or contains_phrase_ci(low, "@brainstorm")
        or low.startswith("email:")
        or contains_phrase_ci(low, "@email")
    ):
        return "CREATE-\u00b5"
    if low.startswith("plan:") or contains_phrase_ci(low, "@plan"):
        return "Coordinator"
    if contains_phrase_ci(low, "mx2lm") or contains_phrase_ci(low, "@mx2lm"):
        return "Mx2LM"
    if (
        contains_phrase_ci(low, "app-wizard")
        or contains_phrase_ci(low, "app wizard")
        or contains_phrase_ci(low, "@app")
        or contains_phrase_ci(low, "app:")
    ):
        return "APP-WIZARD-\u00b5"
    if (
        contains_phrase_ci(low, "@blog")
        or contains_phrase_ci(low, "blog:")
        or contains_phrase_ci(low, "blog")
    ):
        return "BLOG-\u00b5"
    if (
        contains_phrase_ci(low, "search for")
        or contains_phrase_ci(low, "web research")
        or contains_phrase_ci(low, "@search")
        or low.startswith("search:")
        or contains_phrase_ci(low, "@research")
        or contains_phrase_ci(low, "find information")
        or contains_phrase_ci(low, "look up ")
        or contains_phrase_ci(low, "lookup ")
        or contains_phrase_ci(low, "citation")
        or contains_phrase_ci(low, "no fact found")
        or contains_phrase_ci(low, "fact miss")
        or contains_phrase_ci(low, "research")
    ):
        return "WEB-RESEARCH-\u00b5"
    if contains_phrase_ci(low, "librarian"):
        return "Librarian"
    if contains_phrase_ci(low, "coordinator"):
        return "Coordinator"
    if contains_phrase_ci(low, "orch") or contains_phrase_ci(low, "orchestrator"):
        return "ORCH-\u00b5"
    if (
        contains_phrase_ci(low, "concept map")
        or contains_phrase_ci(low, "conceptual map")
        or contains_phrase_ci(low, "hyponym")
        or contains_phrase_ci(low, "hypernym")
        or contains_phrase_ci(low, "taxonomy")
        or contains_phrase_ci(low, "@concept")
        or low.startswith("concept:")
    ):
        return "CONCEPT-\u00b5"
    if contains_phrase_ci(low, "reasoner"):
        return "reasoner"
    return ""


class ManifestRegistry:
    def __init__(self) -> None:
        self._map: dict[str, list[str]] = {}
        self._cache: dict[str, list[str]] = {}

    def load(self, path: Path) -> None:
        root = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(root, dict):
            raise ValueError(f"manifest root must be object: {path}")
        for k, v in root.items():
            arr: list[str] = []
            if isinstance(v, str):
                arr.append(v)
            elif isinstance(v, list):
                for elem in v:
                    if isinstance(elem, str):
                        arr.append(elem)
                    else:
                        arr.append(json.dumps(elem, ensure_ascii=False))
            self._map[k] = arr
            self._cache.pop(k, None)

    def has(self, key: str) -> bool:
        return key in self._map

    def resolve(self, key: str) -> list[str]:
        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)
        result = self._resolve_inner(key, visited=[], depth=0)
        if key in self._map:
            self._cache[key] = list(result)
        return result

    def _resolve_inner(self, key: str, visited: list[str], depth: int) -> list[str]:
        if depth >= SCX_MAX_DEPTH:
            return ["[SCX_DEPTH_LIMIT]"]
        if key in visited:
            return [f"[SCX_CYCLE:{key}]"]
        arr = self._map.get(key)
        if arr is None:
            return [key]

        out: list[str] = []
        visited.append(key)
        for elem in arr:
            if len(elem) > 1 and elem[0] == "@":
                if elem.startswith("@node:"):
                    out.extend(self._resolve_inner(elem[6:], visited, depth + 1))
                else:
                    out.append(elem)
            else:
                if elem in self._map:
                    out.extend(self._resolve_inner(elem, visited, depth + 1))
                else:
                    out.append(elem)
        visited.pop()
        return out


def load_all_manifests(manifest_dir: Path) -> ManifestRegistry:
    reg = ManifestRegistry()
    for name in (
        "css.manifest.json",
        "kuhul.manifest.json",
        "words.manifest.json",
        "micronaut.manifest.json",
    ):
        p = manifest_dir / name
        if p.exists():
            reg.load(p)
    return reg


def resolve_manifest_key(raw_key: str, manifest_reg: ManifestRegistry) -> str:
    key = trim_copy(raw_key)
    if not key:
        return ""
    if manifest_reg.has(key):
        return key

    candidates: list[str] = [lower_copy(key)]
    if not key.startswith("@"):
        candidates.append(f"@{key}")
        candidates.append(f"@{lower_copy(key)}")
    if not key.endswith(":"):
        candidates.append(f"{key}:")
        candidates.append(f"{lower_copy(key)}:")
        candidates.append(f"{lower_copy(key).upper()}:")

    for candidate in candidates:
        if manifest_reg.has(candidate):
            return candidate
    return ""


def is_token_char(ch: str) -> bool:
    return ch.isalnum() or ch in {"'", "_"}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if is_token_char(ch):
            current.append(ch.lower())
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def count_ngrams(tokens: list[str], width: int) -> Counter[str]:
    out: Counter[str] = Counter()
    if width <= 0 or len(tokens) < width:
        return out
    for i in range(len(tokens) - width + 1):
        out[" ".join(tokens[i : i + width])] += 1
    return out


def top_ngrams(freq: Counter[str], top_k: int) -> list[tuple[str, int]]:
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]


def content_function_profile(tokens: list[str], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    content_tokens: list[str] = []
    function_tokens: list[str] = []
    classes: list[str] = []

    for tok in tokens:
        if not tok:
            continue
        if is_function_word(tok):
            function_tokens.append(tok)
            classes.append("f")
        else:
            content_tokens.append(tok)
            classes.append("c")

    total = len(classes)
    content_count = len(content_tokens)
    function_count = len(function_tokens)

    transitions = 0
    if total > 1:
        for i in range(1, total):
            if classes[i] != classes[i - 1]:
                transitions += 1

    function_with_content_neighbor = 0
    if function_count > 0:
        for i, cl in enumerate(classes):
            if cl != "f":
                continue
            left_is_content = i > 0 and classes[i - 1] == "c"
            right_is_content = (i + 1) < total and classes[i + 1] == "c"
            if left_is_content or right_is_content:
                function_with_content_neighbor += 1

    top_content = [
        {"word": word, "count": int(count)}
        for word, count in top_ngrams(Counter(content_tokens), top_k)
    ]
    top_function = [
        {"word": word, "count": int(count)}
        for word, count in top_ngrams(Counter(function_tokens), top_k)
    ]

    profile: dict[str, Any] = {
        "content_token_count": int(content_count),
        "function_token_count": int(function_count),
        "content_ratio": (float(content_count) / float(total)) if total else 0.0,
        "function_ratio": (float(function_count) / float(total)) if total else 0.0,
        "interleaving_ratio": (float(transitions) / float(total - 1)) if total > 1 else 0.0,
        "function_with_content_neighbor_ratio": (
            float(function_with_content_neighbor) / float(function_count)
            if function_count
            else 0.0
        ),
        "classification_policy": "closed_class_function_lexicon_else_content_open_class",
        "function_lexicon_size": int(len(FUNCTION_WORDS)),
    }
    return top_content, top_function, profile


def _normalize_freq_entries(freq: Any) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    if not isinstance(freq, dict):
        return entries
    for key, value in freq.items():
        if not isinstance(key, str):
            continue
        try:
            count = int(value)
        except Exception:
            continue
        entries.append((key, max(0, count)))
    entries.sort(key=lambda item: item[0])
    return entries


def _pack_ngram_section(section_id: int, entries: list[tuple[str, int]]) -> bytes:
    body = bytearray()
    for gram, count in entries:
        enc = gram.encode("utf-8")
        if len(enc) > 0xFFFF:
            raise ValueError(f"ngram key too long for binary format: {gram[:64]!r}")
        body.extend(struct.pack("<H", len(enc)))
        body.extend(enc)
        body.extend(struct.pack("<I", int(count)))
    header = struct.pack("<BBII", int(section_id), 0, len(entries), len(body))
    return header + bytes(body)


def write_mx2lm_ngram_binary(
    out_path: Path,
    *,
    bigram_map: dict[str, int],
    trigram_map: dict[str, int],
    coarse_bigram_map: dict[str, int],
    coarse_trigram_map: dict[str, int],
) -> dict[str, Any]:
    sections: list[tuple[str, bytes, int]] = []
    for name, freq in (
        ("bigrams", bigram_map),
        ("trigrams", trigram_map),
        ("coarse_bigrams", coarse_bigram_map),
        ("coarse_trigrams", coarse_trigram_map),
    ):
        entries = _normalize_freq_entries(freq)
        section_bytes = _pack_ngram_section(MX2LM_NGRAM_SECTION_IDS[name], entries)
        sections.append((name, section_bytes, len(entries)))

    payload = b"".join(blob for _, blob, _ in sections)
    payload_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(
        "<8sHHII",
        MX2LM_NGRAM_BIN_MAGIC,
        MX2LM_NGRAM_BIN_VERSION,
        len(sections),
        len(payload),
        payload_crc32,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + payload)

    return {
        "path": str(out_path),
        "bytes": int(len(header) + len(payload)),
        "magic": MX2LM_NGRAM_BIN_MAGIC.decode("ascii"),
        "version": int(MX2LM_NGRAM_BIN_VERSION),
        "section_count": int(len(sections)),
        "payload_bytes": int(len(payload)),
        "payload_crc32": f"{payload_crc32:08x}",
        "sections": {name: count for name, _, count in sections},
        "mode": "snapshot_only",
        "semantic_authority": "definitions_and_relations",
    }


FUNCTION_WORDS: set[str] = {
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "in", "on", "at", "by", "for", "from", "to", "of", "with", "without", "under", "over",
    "above", "below", "before", "after", "between", "among", "into", "onto", "through",
    "across", "around", "about", "towards", "toward", "as", "if", "than", "then",
    "and", "or", "but", "nor", "so", "yet", "because", "while", "although", "though",
    "unless", "until", "since", "however", "thus",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought",
    "not", "no", "yes", "ok", "okay",
    "up", "down", "out", "off",
    "oh", "ah", "eh", "there", "here",
}

COMPOSITION_BRIDGES: set[str] = {
    "in", "of", "for", "with", "to", "from", "on", "into", "within", "about", "between", "through", "across",
}

COMPOSITIONAL_ALIASES: dict[str, list[str]] = {
    "organic_life_in_space": ["astrobiology", "exobiology", "biosignature", "habitability"],
    "life_in_space": ["astrobiology", "exobiology", "biosignature"],
    "search_for_life": ["biosignature", "habitability", "astrobiology"],
    "signs_of_life": ["biosignature", "bio_marker", "habitability"],
}

RELATION_PHRASE_TEMPLATES: dict[tuple[str, str], str] = {
    ("capital", "of"): "is_capital_of",
    ("part", "of"): "is_part_of",
    ("type", "of"): "is_type_of",
    ("kind", "of"): "is_kind_of",
    ("located", "in"): "is_located_in",
    ("born", "in"): "born_in",
    ("depends", "on"): "depends_on",
    ("made", "of"): "made_of",
}

TAXONOMY_ARTICLES: set[str] = {"a", "an", "the"}
TAXONOMY_CONNECTORS: set[str] = {"and", "or"}

WSD_LEXEME_SPECS: dict[str, list[tuple[str, list[str]]]] = {
    "simulation": [
        ("modeling_runtime", ["model", "runtime", "compute", "training", "inference", "synthetic", "data"]),
        ("pretend_imitation", ["pretend", "imitation", "acting", "theater", "game", "fiction"]),
        ("medical_training", ["medical", "clinical", "patient", "surgery", "hospital", "procedure"]),
    ],
    "play": [
        ("theater_drama", ["theater", "stage", "actor", "drama", "script", "scene"]),
        ("sports_tactic", ["coach", "team", "offense", "defense", "strategy", "formation"]),
        ("recreation_activity", ["children", "park", "fun", "toy", "recess", "outside"]),
    ],
    "model": [
        ("ai_architecture", ["ai", "neural", "weights", "inference", "training", "runtime"]),
        ("representation_schema", ["schema", "representation", "semantic", "graph", "abstraction"]),
        ("person_object", ["person", "fashion", "pose", "photo", "camera"]),
    ],
    "field": [
        ("domain_area", ["domain", "discipline", "subject", "context", "topic"]),
        ("spatial_region", ["space", "region", "area", "map", "world", "inside"]),
        ("physical_vector_field", ["vector", "physics", "force", "gradient", "flow"]),
    ],
}


def is_function_word(token: str) -> bool:
    return token in FUNCTION_WORDS


def extract_coarse_ngrams(tokens: list[str], width: int) -> Counter[str]:
    coarse_tokens: list[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if is_function_word(tok):
            coarse_tokens.append(f"f:{tok}")
        else:
            coarse_tokens.append("c")
    return count_ngrams(coarse_tokens, width)


def extract_compositional_signatures(tokens: list[str]) -> list[str]:
    signatures: list[str] = []
    seen: set[str] = set()
    if not tokens:
        return signatures

    for i, tok_i in enumerate(tokens):
        if len(tok_i) < 2 or is_function_word(tok_i):
            continue
        sig = tok_i
        content_count = 1
        for j in range(i + 1, min(len(tokens), i + 5)):
            tj = tokens[j]
            if len(tj) < 2:
                break
            if is_function_word(tj):
                if tj not in COMPOSITION_BRIDGES:
                    break
                sig += f"_{tj}"
                continue
            sig += f"_{tj}"
            content_count += 1
            if content_count >= 2 and sig not in seen:
                seen.add(sig)
                signatures.append(sig)

    for sig in list(signatures):
        for alias in COMPOSITIONAL_ALIASES.get(sig, []):
            if alias in seen:
                continue
            seen.add(alias)
            signatures.append(alias)
    return signatures


def infer_relation_templates(tokens: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dedup: set[tuple[str, str, str, str]] = set()

    def push_template(
        relation: str,
        from_node: str,
        to_node: str,
        pattern: str,
        evidence: str,
        template_kind: str,
        association_weight: float,
    ) -> None:
        key = (relation, from_node, to_node, pattern)
        if key in dedup:
            return
        dedup.add(key)
        out.append(
            {
                "relation": relation,
                "from": from_node,
                "to": to_node,
                "pattern": pattern,
                "evidence": evidence,
                "template_kind": template_kind,
                "association_weight": float(max(0.0, min(1.0, association_weight))),
                "weight_kind": "pattern_strength",
                "semantic_authority": "definition_or_graph_relation_required",
            }
        )

    for i in range(len(tokens) - 3):
        left = tokens[i]
        p0 = tokens[i + 1]
        p1 = tokens[i + 2]
        right = tokens[i + 3]
        relation = RELATION_PHRASE_TEMPLATES.get((p0, p1))
        if not relation:
            continue
        pattern = f"{p0} {p1}"
        evidence = " ".join(tokens[i : i + 4])
        if not is_function_word(left) and not is_function_word(right):
            push_template(relation, left, right, pattern, evidence, "grounded_instance", 0.9)

    for i in range(len(tokens) - 4):
        left = tokens[i]
        cop = tokens[i + 1]
        p0 = tokens[i + 2]
        p1 = tokens[i + 3]
        right = tokens[i + 4]
        if cop not in {"is", "are", "was", "were"}:
            continue
        relation = RELATION_PHRASE_TEMPLATES.get((p0, p1))
        if not relation:
            continue
        if is_function_word(left) or is_function_word(right):
            continue
        pattern = f"{cop} {p0} {p1}"
        evidence = " ".join(tokens[i : i + 5])
        push_template(relation, left, right, pattern, evidence, "grounded_instance", 0.95)

    for i in range(len(tokens) - 2):
        p0 = tokens[i]
        p1 = tokens[i + 1]
        right = tokens[i + 2]
        relation = RELATION_PHRASE_TEMPLATES.get((p0, p1))
        if not relation or is_function_word(right):
            continue
        pattern = f"{p0} {p1}"
        evidence = " ".join(tokens[i : i + 3])
        push_template(relation, "$X", right, pattern, evidence, "latent_template", 0.6)

    return out


def infer_taxonomy_relations(tokens: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dedup: set[tuple[str, str, str, str]] = set()
    hypernym_to_hyponyms: dict[str, set[str]] = {}

    def is_content_token(tok: str) -> bool:
        return bool(tok) and len(tok) >= 2 and (not is_function_word(tok))

    def push_relation(
        frm: str,
        rel: str,
        to: str,
        evidence: str,
        weight: float,
        *,
        relation_family: str = "taxonomy",
    ) -> None:
        clean_from = trim_copy(frm)
        clean_rel = trim_copy(rel)
        clean_to = trim_copy(to)
        clean_evidence = trim_copy(evidence)
        if not clean_from or not clean_rel or not clean_to:
            return
        key = (clean_from, clean_rel, clean_to, clean_evidence)
        if key in dedup:
            return
        dedup.add(key)
        out.append(
            {
                "from": clean_from,
                "relation": clean_rel,
                "to": clean_to,
                "relation_family": relation_family,
                "evidence": clean_evidence,
                "association_weight": float(max(0.0, min(1.0, weight))),
                "weight_kind": "pattern_strength",
                "semantic_authority": "definition_or_graph_relation_required",
            }
        )

    # Pattern: X is/are/was/were (a|an|the)? Y
    for i in range(len(tokens) - 2):
        left = tokens[i]
        cop = tokens[i + 1]
        if cop not in {"is", "are", "was", "were"}:
            continue
        j = i + 2
        if j < len(tokens) and tokens[j] in TAXONOMY_ARTICLES:
            j += 1
        if j >= len(tokens):
            continue
        right = tokens[j]
        if not is_content_token(left) or not is_content_token(right):
            continue
        evidence = " ".join(tokens[i : j + 1])
        push_relation(left, "hyponym_of", right, evidence, 0.9)
        push_relation(right, "hypernym_of", left, evidence, 0.9)
        hypernym_to_hyponyms.setdefault(right, set()).add(left)

    # Pattern: H such as x and y ...
    for i in range(1, len(tokens) - 2):
        if tokens[i] != "such" or tokens[i + 1] != "as":
            continue
        hypernym = tokens[i - 1]
        if not is_content_token(hypernym):
            continue

        hyponyms: list[str] = []
        for j in range(i + 2, min(len(tokens), i + 10)):
            tok = tokens[j]
            if tok in TAXONOMY_CONNECTORS or tok in TAXONOMY_ARTICLES:
                continue
            if is_function_word(tok):
                if hyponyms:
                    break
                continue
            if len(tok) < 2:
                continue
            if tok not in hyponyms:
                hyponyms.append(tok)

        if not hyponyms:
            continue

        evidence = " ".join(tokens[i - 1 : min(len(tokens), i + 10)])
        for hypo in hyponyms:
            push_relation(hypo, "hyponym_of", hypernym, evidence, 0.88)
            push_relation(hypernym, "hypernym_of", hypo, evidence, 0.88)
            hypernym_to_hyponyms.setdefault(hypernym, set()).add(hypo)

    # Pattern: X part of Y / X is part of Y
    for i in range(len(tokens) - 3):
        left = tokens[i]
        if tokens[i + 1] != "part" or tokens[i + 2] != "of":
            continue
        right = tokens[i + 3]
        if not is_content_token(left) or not is_content_token(right):
            continue
        evidence = " ".join(tokens[i : i + 4])
        push_relation(left, "meronym_of", right, evidence, 0.74, relation_family="mereology")
        push_relation(right, "holonym_of", left, evidence, 0.74, relation_family="mereology")

    for i in range(len(tokens) - 4):
        left = tokens[i]
        cop = tokens[i + 1]
        if cop not in {"is", "are", "was", "were"}:
            continue
        if tokens[i + 2] != "part" or tokens[i + 3] != "of":
            continue
        right = tokens[i + 4]
        if not is_content_token(left) or not is_content_token(right):
            continue
        evidence = " ".join(tokens[i : i + 5])
        push_relation(left, "meronym_of", right, evidence, 0.8, relation_family="mereology")
        push_relation(right, "holonym_of", left, evidence, 0.8, relation_family="mereology")

    # Pattern: X aka Y / X alias Y
    for i in range(len(tokens) - 2):
        left = tokens[i]
        marker = tokens[i + 1]
        right = tokens[i + 2]
        if marker not in {"aka", "alias"}:
            continue
        if not is_content_token(left) or not is_content_token(right):
            continue
        evidence = " ".join(tokens[i : i + 3])
        push_relation(left, "synonym_of", right, evidence, 0.85, relation_family="synonymy")
        push_relation(right, "synonym_of", left, evidence, 0.85, relation_family="synonymy")

    # Pattern: X also known as Y
    for i in range(len(tokens) - 4):
        left = tokens[i]
        if tokens[i + 1] != "also" or tokens[i + 2] != "known" or tokens[i + 3] != "as":
            continue
        right = tokens[i + 4]
        if not is_content_token(left) or not is_content_token(right):
            continue
        evidence = " ".join(tokens[i : i + 5])
        push_relation(left, "synonym_of", right, evidence, 0.9, relation_family="synonymy")
        push_relation(right, "synonym_of", left, evidence, 0.9, relation_family="synonymy")

    # Co-hyponymy from shared hypernyms
    for hypernym, hypos_set in hypernym_to_hyponyms.items():
        hypos = sorted(hypos_set)
        if len(hypos) < 2:
            continue
        evidence = f"shared_hypernym:{hypernym}"
        for i in range(len(hypos)):
            for j in range(i + 1, len(hypos)):
                a, b = hypos[i], hypos[j]
                push_relation(a, "co_hyponym_with", b, evidence, 0.72)
                push_relation(b, "co_hyponym_with", a, evidence, 0.72)

    return out


def build_semantic_folding_projection(taxonomy_relations: list[dict[str, Any]]) -> dict[str, Any]:
    relation_counts: dict[str, int] = {}
    for rel in taxonomy_relations:
        r = str(rel.get("relation", ""))
        if not r:
            continue
        relation_counts[r] = relation_counts.get(r, 0) + 1

    return {
        "model": "lexical_ontology_sparse_geometry",
        "role": "similarity_kernel",
        "topology": {
            "hypernym_region": "broad_spread_overlap",
            "hyponym_region": "localized_nested_subregion",
            "co_hyponym_region": "adjacent_distinct_subregions",
        },
        "hybrid_layers": {
            "word_sense_layer": "external_contextual_disambiguation",
            "coarse_ngram_layer": "external_local_order_surrogate",
            "boolean_ops_layer": "fingerprint_set_operations",
        },
        "boolean_ops_scope": "vector_bitset_overlap_only_not_control_flow_logic",
        "captures": {
            "hyponymy": relation_counts.get("hyponym_of", 0) > 0,
            "synonymy": relation_counts.get("synonym_of", 0) > 0,
            "co_hyponymy": relation_counts.get("co_hyponym_with", 0) > 0,
            "meronymy": relation_counts.get("meronym_of", 0) > 0,
        },
        "capture_notes": {
            "meronymy": "context_dependent_partial_capture",
        },
        "limitations": {
            "antonymy": "not_structurally_distinguished",
            "word_order_syntax": "not_encoded",
            "negation": "not_reliably_encoded",
            "predication": "requires_sequence_model",
            "control_flow": "requires_explicit_program_or_sequence_layer",
        },
        "relation_counts": relation_counts,
    }


def resolve_word_senses(tokens: list[str]) -> list[dict[str, Any]]:
    token_set = set(tokens)
    out: list[dict[str, Any]] = []
    for term, options in WSD_LEXEME_SPECS.items():
        if term not in token_set:
            continue
        chosen = options[0]
        chosen_hits = 0
        for sense_id, cues in options:
            hits = sum(1 for cue in cues if cue in token_set)
            if hits > chosen_hits:
                chosen = (sense_id, cues)
                chosen_hits = hits
        sense_id, cues = chosen
        cue_hits = [cue for cue in cues if cue in token_set]
        cue_cov = (len(cue_hits) / len(cues)) if cues else 0.0
        confidence = 0.45 + 0.55 * cue_cov if cue_hits else 0.35
        out.append(
            {
                "term": term,
                "sense_id": sense_id,
                "cue_hits": cue_hits,
                "cue_count": len(cues),
                "confidence": round(float(max(0.0, min(1.0, confidence))), 6),
            }
        )
    return out


CONTROL_RULES: dict[str, tuple[str, str]] = {
    "if": ("DEPENDENCY", "conditional-branch"),
    "else": ("ALTERNATIVE", "branch-alternative"),
    "when": ("SEQUENCE", "temporal-condition"),
    "where": ("RELATION", "location-binding"),
    "how": ("DIRECT", "method-execution"),
    "is": ("STATE", "state-assertion"),
    "are": ("STATE", "state-assertion"),
    "equal": ("SUBSTITUTE", "comparison-equivalence"),
    "for": ("SEQUENCE", "iteration-or-scope"),
    "and": ("RELATION", "conjunctive-join"),
    "or": ("ALTERNATIVE", "disjunctive-branch"),
    "not": ("CONFLICT", "negation-constraint"),
    "because": ("CAUSE", "causal-link"),
    "then": ("SEQUENCE", "ordered-transition"),
    "requires": ("DEPENDENCY", "precondition-binding"),
    "depends": ("DEPENDENCY", "precondition-binding"),
    "evidence": ("EVIDENCE", "evidence-binding"),
    "shows": ("EVIDENCE", "evidence-binding"),
    "proves": ("EVIDENCE", "evidence-binding"),
    "verify": ("VERIFY", "verification-gate"),
    "check": ("TEST", "test-gate"),
    "assert": ("TEST", "test-gate"),
}


def resolve_control_bindings(tokens: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for tok in tokens:
        rule = CONTROL_RULES.get(tok)
        if not rule:
            continue
        op_name, semantic_role = rule
        out.append(
            {
                "token": tok,
                "operator": op_name,
                "book": f"ELIZA-1:{op_name}",
                "detect_phase": "Pop",
                "emit_phase": "Sek",
                "semantic_role": semantic_role,
            }
        )
    return out


def is_control_token(token: str) -> bool:
    return token in CONTROL_RULES


def collapse_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    index: dict[tuple[str, str, str], int] = {}
    for rel in relations:
        key = (rel["from"], rel["relation"], rel["to"])
        if key not in index:
            index[key] = len(out)
            out.append(dict(rel))
            continue
        pos = index[key]
        out[pos]["association_weight"] = float(out[pos]["association_weight"]) + float(rel["association_weight"])

    for rel in out:
        rel["association_weight"] = max(0.0, min(1.0, float(rel["association_weight"])))
    return out


def infer_relations(tokens: list[str]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if not tokens:
        return edges

    for template in infer_relation_templates(tokens):
        if template["template_kind"] != "grounded_instance":
            continue
        if template["from"] == "$X" or template["to"] == "$Y":
            continue
        edges.append(
            {
                "from": template["from"],
                "relation": template["relation"],
                "to": template["to"],
                "association_weight": template["association_weight"],
                "weight_kind": template["weight_kind"],
            }
        )

    for i in range(1, len(tokens) - 1):
        left, control, right = tokens[i - 1], tokens[i], tokens[i + 1]
        if not is_control_token(control):
            continue
        if is_control_token(left) or is_control_token(right):
            continue
        edges.append(
            {
                "from": left,
                "relation": control,
                "to": right,
                "association_weight": 1.0,
                "weight_kind": "pattern_strength",
            }
        )

    for i in range(len(tokens) - 1):
        left, right = tokens[i], tokens[i + 1]
        if is_control_token(left) or is_control_token(right):
            continue
        edges.append(
            {
                "from": left,
                "relation": "assoc",
                "to": right,
                "association_weight": 0.5,
                "weight_kind": "pattern_strength",
            }
        )

    return collapse_relations(edges)


@dataclass
class StructuredProjection:
    nodes: list[dict[str, str]]
    control_bindings: list[dict[str, str]]
    relations: list[dict[str, Any]]
    clause: str
    sentence: str


def compose(tokens: list[str]) -> StructuredProjection:
    control_bindings = resolve_control_bindings(tokens)
    relations = infer_relations(tokens)

    control_by_token: dict[str, dict[str, str]] = {}
    for cb in control_bindings:
        control_by_token.setdefault(cb["token"], cb)

    seen: set[str] = set()
    nodes: list[dict[str, str]] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        cb = control_by_token.get(tok)
        if cb:
            nodes.append(
                {
                    "id": tok,
                    "kind": "control_node",
                    "meaning_key": cb["semantic_role"],
                    "operator": cb["operator"],
                    "book": cb["book"],
                }
            )
        else:
            nodes.append({"id": tok, "kind": "semantic_node", "meaning_key": tok})

    clause = " ".join(tokens)
    sentence = clause
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
        if sentence[-1] not in ".!?":
            sentence += "."
    return StructuredProjection(nodes, control_bindings, relations, clause, sentence)


def extract_mx2lm_payload(text: str) -> str:
    t = trim_copy(text)
    low = lower_copy(t)

    open_pos = low.find("<mx2lm>")
    if open_pos != -1:
        start = open_pos + len("<mx2lm>")
        close = low.find("</mx2lm>", start)
        if close == -1:
            close = low.find("<mx2lm>", start)
        if close == -1:
            close = len(t)
        return trim_copy(t[start:close])

    if low.startswith("mx2lm:"):
        return trim_copy(t[len("mx2lm:") :])
    if low.startswith("@mx2lm"):
        after = trim_copy(t[len("@mx2lm") :])
        if after.startswith(":"):
            after = trim_copy(after[1:])
        return after

    pos = low.find("mx2lm:")
    if pos != -1:
        return trim_copy(t[pos + len("mx2lm:") :])
    pos = low.find("@mx2lm")
    if pos != -1:
        after = trim_copy(t[pos + len("@mx2lm") :])
        if after.startswith(":"):
            after = trim_copy(after[1:])
        return after
    return ""


def dedupe_keep_order(values: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def build_key_references(
    payload: str,
    tokens: list[str],
    top_bi: list[tuple[str, int]],
    top_tri: list[tuple[str, int]],
    top_coarse_bi: list[tuple[str, int]],
    top_coarse_tri: list[tuple[str, int]],
    projection: StructuredProjection,
    relation_templates: list[dict[str, Any]],
    taxonomy_relations: list[dict[str, Any]],
    compositional_signatures: list[str],
    manifest_reg: ManifestRegistry,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    dedup: set[tuple[str, str, str]] = set()

    def push_reference(key: str, source_kind: str, source_value: str) -> None:
        clean_key = trim_copy(key)
        clean_kind = trim_copy(source_kind)
        clean_value = trim_copy(source_value)
        if not clean_key:
            return
        uniq = (clean_key, clean_kind, clean_value)
        if uniq in dedup:
            return
        dedup.add(uniq)

        resolved_key = resolve_manifest_key(clean_key, manifest_reg)
        row: dict[str, Any] = {
            "key": clean_key,
            "source_kind": clean_kind,
            "source_value": clean_value,
            "resolved": bool(resolved_key),
        }
        if resolved_key and resolved_key != clean_key:
            row["resolved_key"] = resolved_key
        if resolved_key:
            row["expansion"] = dedupe_keep_order(manifest_reg.resolve(resolved_key), 8)
        else:
            row["expansion"] = []
        refs.append(row)

    seen_tokens: set[str] = set()
    for tok in tokens:
        if tok in seen_tokens:
            continue
        seen_tokens.add(tok)
        push_reference(tok, "token", tok)

    for gram, _ in top_bi:
        push_reference(gram, "ngram", gram)
    for gram, _ in top_tri:
        push_reference(gram, "ngram", gram)
    for gram, _ in top_coarse_bi:
        push_reference(gram, "coarse_ngram", gram)
    for gram, _ in top_coarse_tri:
        push_reference(gram, "coarse_ngram", gram)

    for binding in projection.control_bindings:
        push_reference(binding["book"], "control_book", binding["token"])
        push_reference(binding["semantic_role"], "semantic_role", binding["token"])
        push_reference(f"@phase:{binding['detect_phase']}", "phase_anchor", binding["token"])
        push_reference(f"@phase:{binding['emit_phase']}", "phase_anchor", binding["token"])

    for node in projection.nodes:
        push_reference(node["meaning_key"], "meaning_key", node["id"])

    for sig in compositional_signatures:
        push_reference(sig, "compositional_signature", payload)

    for template in relation_templates:
        relation_name = trim_copy(str(template.get("relation", "")))
        pattern = trim_copy(str(template.get("pattern", "")))
        frm = trim_copy(str(template.get("from", "")))
        to = trim_copy(str(template.get("to", "")))
        if relation_name:
            push_reference(relation_name, "relation_template", pattern or payload)
        if frm and not frm.startswith("$"):
            push_reference(frm, "relation_entity", relation_name or pattern)
        if to and not to.startswith("$"):
            push_reference(to, "relation_entity", relation_name or pattern)

    for tax_rel in taxonomy_relations:
        relation_name = trim_copy(str(tax_rel.get("relation", "")))
        evidence = trim_copy(str(tax_rel.get("evidence", "")))
        frm = trim_copy(str(tax_rel.get("from", "")))
        to = trim_copy(str(tax_rel.get("to", "")))
        if relation_name:
            push_reference(relation_name, "taxonomy_relation", evidence or payload)
        if frm and not frm.startswith("$"):
            push_reference(frm, "taxonomy_entity", relation_name or evidence)
        if to and not to.startswith("$"):
            push_reference(to, "taxonomy_entity", relation_name or evidence)

    inferred_track = infer_micronaut_from_text(payload)
    if inferred_track:
        push_reference(inferred_track, "inferred_track", payload)
        push_reference(f"@node:{inferred_track}", "node_anchor", payload)

    return refs


def build_ngram_references(
    top_bi: list[tuple[str, int]],
    top_tri: list[tuple[str, int]],
    manifest_reg: ManifestRegistry,
    channel: str = "lexical",
) -> list[dict[str, Any]]:
    def split_gram_tokens(gram: str) -> list[str]:
        # Preserve coarse labels like "f:is" as single tokens.
        return [tok for tok in gram.split(" ") if tok]

    ranked_grams: list[dict[str, Any]] = []
    for gram, count in top_bi:
        ranked_grams.append({"gram": gram, "count": int(count), "width": 2, "tokens": split_gram_tokens(gram)})
    for gram, count in top_tri:
        ranked_grams.append({"gram": gram, "count": int(count), "width": 3, "tokens": split_gram_tokens(gram)})

    out: list[dict[str, Any]] = []
    for i, g in enumerate(ranked_grams):
        row: dict[str, Any] = {
            "gram": g["gram"],
            "count": g["count"],
            "width": g["width"],
            "tokens": list(g["tokens"]),
            "channel": channel,
        }

        definitions: list[dict[str, Any]] = []
        resolved_gram = resolve_manifest_key(g["gram"], manifest_reg)
        if resolved_gram:
            entry: dict[str, Any] = {"key": resolved_gram, "kind": "ngram_key"}
            if resolved_gram != g["gram"]:
                entry["source_value"] = g["gram"]
            entry["expansion"] = dedupe_keep_order(manifest_reg.resolve(resolved_gram), 8)
            definitions.append(entry)

        for token in g["tokens"]:
            resolved_token = resolve_manifest_key(token, manifest_reg)
            if not resolved_token:
                continue
            entry = {"key": resolved_token, "kind": "token_key"}
            if resolved_token != token:
                entry["source_value"] = token
            entry["expansion"] = dedupe_keep_order(manifest_reg.resolve(resolved_token), 6)
            definitions.append(entry)

        related: list[dict[str, Any]] = []
        for j, other in enumerate(ranked_grams):
            if i == j:
                continue
            if not set(g["tokens"]).intersection(other["tokens"]):
                continue
            related.append(
                {"gram": other["gram"], "count": other["count"], "width": other["width"]}
            )
            if len(related) >= 4:
                break

        row["definitions"] = definitions
        row["related_ngrams"] = related
        out.append(row)
    return out


def build_payload(
    payload: str,
    source: str,
    top_k: int,
    manifest_reg: ManifestRegistry,
) -> dict[str, Any]:
    tokens = tokenize(payload)
    bigram_map = count_ngrams(tokens, 2)
    trigram_map = count_ngrams(tokens, 3)
    coarse_bigram_map = extract_coarse_ngrams(tokens, 2)
    coarse_trigram_map = extract_coarse_ngrams(tokens, 3)
    top_bi = top_ngrams(bigram_map, top_k)
    top_tri = top_ngrams(trigram_map, top_k)
    top_coarse_bi = top_ngrams(coarse_bigram_map, top_k)
    top_coarse_tri = top_ngrams(coarse_trigram_map, top_k)
    projection = compose(tokens)
    relation_templates = infer_relation_templates(tokens)
    taxonomy_relations = infer_taxonomy_relations(tokens)
    semantic_folding_projection = build_semantic_folding_projection(taxonomy_relations)
    compositional_signatures = extract_compositional_signatures(tokens)
    word_senses = resolve_word_senses(tokens)
    top_content_words, top_function_words, content_function = content_function_profile(tokens, top_k)

    relations: list[dict[str, Any]] = []
    for relation in projection.relations:
        relations.append(
            {
                "from": relation["from"],
                "relation": relation["relation"],
                "to": relation["to"],
                "association_weight": relation["association_weight"],
                "weight_kind": relation["weight_kind"],
                "weight": relation["association_weight"],
            }
        )

    existing = {(r["from"], r["relation"], r["to"]) for r in relations}
    for template in relation_templates:
        if template.get("template_kind") != "grounded_instance":
            continue
        frm = str(template.get("from", ""))
        rel = str(template.get("relation", ""))
        to = str(template.get("to", ""))
        if not frm or not rel or not to:
            continue
        key = (frm, rel, to)
        if key in existing:
            continue
        existing.add(key)
        relations.append(
            {
                "from": frm,
                "relation": rel,
                "to": to,
                "association_weight": float(template.get("association_weight", 0.8)),
                "weight_kind": "pattern_strength",
                "weight": float(template.get("association_weight", 0.8)),
            }
        )

    key_references = build_key_references(
        payload,
        tokens,
        top_bi,
        top_tri,
        top_coarse_bi,
        top_coarse_tri,
        projection,
        relation_templates,
        taxonomy_relations,
        compositional_signatures,
        manifest_reg,
    )
    ngram_references = build_ngram_references(top_bi, top_tri, manifest_reg, channel="lexical")
    coarse_ngram_references = build_ngram_references(
        top_coarse_bi, top_coarse_tri, manifest_reg, channel="coarse"
    )

    return {
        "ok": True,
        "source": source,
        "invoke": "@mx2lm",
        "response": "<mx2lm>{ngrams}</mx2lm>",
        "payload": payload,
        "tokens": len(tokens),
        "bigrams": [{"gram": gram, "count": count} for gram, count in top_bi],
        "trigrams": [{"gram": gram, "count": count} for gram, count in top_tri],
        "coarse_bigrams": [{"gram": gram, "count": count} for gram, count in top_coarse_bi],
        "coarse_trigrams": [{"gram": gram, "count": count} for gram, count in top_coarse_tri],
        "content_words": top_content_words,
        "function_words": top_function_words,
        "content_function_profile": content_function,
        "mx2lm_bigram_map": {k: int(v) for k, v in sorted(bigram_map.items())},
        "mx2lm_trigram_map": {k: int(v) for k, v in sorted(trigram_map.items())},
        "mx2lm_coarse_bigram_map": {k: int(v) for k, v in sorted(coarse_bigram_map.items())},
        "mx2lm_coarse_trigram_map": {k: int(v) for k, v in sorted(coarse_trigram_map.items())},
        "mx2lm_checkpoint": {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "token_count": len(tokens),
            "bigram_count": len(bigram_map),
            "trigram_count": len(trigram_map),
            "coarse_bigram_count": len(coarse_bigram_map),
            "coarse_trigram_count": len(coarse_trigram_map),
            "source": source,
        },
        "control_bindings": projection.control_bindings,
        "nodes": projection.nodes,
        "relations": relations,
        "relation_templates": relation_templates,
        "taxonomy_relations": taxonomy_relations,
        "semantic_folding_projection": semantic_folding_projection,
        "compositional_signatures": compositional_signatures,
        "word_sense_candidates": word_senses,
        "key_references": key_references,
        "ngram_references": ngram_references,
        "coarse_ngram_references": coarse_ngram_references,
        "structured_composition": {
            "hierarchy": ["atom", "gram", "relation", "clause", "sentence", "paragraph"],
            "control_authority": "XCFE/K'UHUL",
            "node_semantics_authority": "nodes",
            "weight_semantics": "pattern_strength_only",
            "weight_scale": "[0,1]",
            "content_word_role": "semantic_payload_primary",
            "function_word_role": "grammatical_structure_primary",
            "coarse_ngram_role": "association_prior_only",
            "template_role": "relation_generalization_prior",
            "semantic_folding_axis": "taxonomic_subsumption_geometry",
            "predication_axis": "sequence_required_outside_folding",
            "syntax_and_negation_support": "limited_without_sequence_model",
            "boolean_ops_scope": "fingerprint_overlap_ops_not_symbolic_logic",
            "control_token_role": "surface_relational_cues_not_executable_control_flow",
            "content_function_binding_ratio": content_function.get("function_with_content_neighbor_ratio", 0.0),
            "clause": projection.clause,
            "sentence": projection.sentence,
            "paragraph": projection.sentence,
        },
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mx2LM n-gram payload/checkpoint JSON.")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Inline text to process (repeatable).",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="PATH",
        help="Text file path to process (repeatable).",
    )
    parser.add_argument(
        "--extract-payload",
        action="store_true",
        help="Extract payload from Mx2LM wrappers (Mx2LM:, @mx2lm, <mx2lm>...</mx2lm>).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Data directory containing manifest/*.manifest.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of top bigrams/trigrams to emit (default: 6).",
    )
    parser.add_argument(
        "--output",
        default="",
        metavar="PATH",
        help="Optional output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--binary-out",
        default="",
        metavar="PATH",
        help="Optional binary snapshot output path (mx2lm.ngram.bin format).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload, source_labels = read_inputs(
            [str(v) for v in args.text],
            [Path(p) for p in args.input],
        )
    except Exception as exc:
        print(f"ERROR: failed reading inputs: {exc}", file=sys.stderr)
        return 1

    if not trim_copy(payload):
        print("ERROR: no input text provided (--text/--input or stdin).", file=sys.stderr)
        return 1

    if args.extract_payload:
        extracted = extract_mx2lm_payload(payload)
        if extracted:
            payload = extracted

    if args.top_k <= 0:
        print("ERROR: --top-k must be >= 1", file=sys.stderr)
        return 1

    try:
        manifest_dir = Path(args.data_dir) / "manifest"
        manifest_reg = load_all_manifests(manifest_dir)
    except Exception as exc:
        print(f"ERROR: failed loading manifests: {exc}", file=sys.stderr)
        return 1

    out = build_payload(payload, source="mx2lm-ngram-builder", top_k=args.top_k, manifest_reg=manifest_reg)
    out["sources"] = source_labels

    if args.binary_out:
        try:
            out["mx2lm_binary_checkpoint"] = write_mx2lm_ngram_binary(
                Path(args.binary_out),
                bigram_map=out.get("mx2lm_bigram_map", {}),
                trigram_map=out.get("mx2lm_trigram_map", {}),
                coarse_bigram_map=out.get("mx2lm_coarse_bigram_map", {}),
                coarse_trigram_map=out.get("mx2lm_coarse_trigram_map", {}),
            )
        except Exception as exc:
            print(f"ERROR: failed writing binary snapshot: {exc}", file=sys.stderr)
            return 1

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
