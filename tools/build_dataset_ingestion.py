#!/usr/bin/env python3
"""Compile declared datasets into KHANARY gram, candidate key-book, and vector artifacts.

This is a streaming bootstrap compiler.  It reads external corpora from a source
manifest, extracts text, and emits derived evidence artifacts under data/.  Raw
dataset rows are never copied into the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from ngram_builder import (  # noqa: E402
    build_mx2lm_gram_container,
    build_mx2lm_pattern_container,
    count_ngrams,
    extract_coarse_ngrams,
    extract_function_grams,
    extract_tool_grams,
    fnv1a64,
    signed_component,
    tokenize,
    top_ngrams,
    vector_key_safe,
    write_mx2lm_ngram_binary,
)


PREFERRED_TEXT_KEYS = (
    "text",
    "body",
    "title",
    "joke",
    "selftext",
    "prompt",
    "response",
    "completion",
    "instruction",
    "input",
    "output",
    "question",
    "answer",
    "chosen",
    "rejected",
    "reasoning",
    "cot",
    "content",
    "messages",
    "conversations",
)

SKIP_TEXT_KEYS = {
    "id",
    "_id",
    "uuid",
    "hash",
    "sha256",
    "url",
    "path",
    "source",
    "created",
    "created_at",
    "timestamp",
    "score",
    "ups",
    "downs",
    "author",
}

REQUIRED_CHANNELS = (
    "unigram",
    "bigram",
    "trigram",
    "quadgram",
    "coarse_gram",
    "function_gram",
    "tool_gram",
)

BINARY_ONLY_COUNTERS = ("coarse_bigram", "coarse_trigram")


def stable_hash(text: str, size: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]


def resolve_path(raw: str) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return ROOT / path


def clean_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value).replace("\x00", " ").split())
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    return text.strip()


def collect_strings(value: Any, *, max_items: int = 64) -> list[str]:
    out: list[str] = []

    def visit(node: Any, key_hint: str = "") -> None:
        if len(out) >= max_items:
            return
        if isinstance(node, str):
            if key_hint.lower() not in SKIP_TEXT_KEYS:
                out.append(node)
            return
        if isinstance(node, list):
            for item in node:
                visit(item, key_hint)
                if len(out) >= max_items:
                    break
            return
        if isinstance(node, dict):
            for key in PREFERRED_TEXT_KEYS:
                if key in node:
                    visit(node[key], key)
                    if len(out) >= max_items:
                        return
            for key, item in node.items():
                if str(key).lower() in SKIP_TEXT_KEYS or key in PREFERRED_TEXT_KEYS:
                    continue
                visit(item, str(key))
                if len(out) >= max_items:
                    return

    visit(value)
    return out


def json_record_text(record: Any, max_chars: int) -> str:
    parts = collect_strings(record)
    return clean_text(" ".join(parts), max_chars)


def csv_record_text(row: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    for key in PREFERRED_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    if not parts:
        for key, value in row.items():
            if str(key).lower() in SKIP_TEXT_KEYS:
                continue
            if isinstance(value, str) and value.strip():
                parts.append(value)
    return clean_text(" ".join(parts), max_chars)


def iter_jsonl(path: Path, max_rows: int, max_chars: int) -> Iterable[tuple[str, bool]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle, start=1):
            if max_rows > 0 and index > max_rows:
                break
            raw = line.strip()
            if not raw:
                yield "", True
                continue
            try:
                record = json.loads(raw)
                yield json_record_text(record, max_chars), True
            except json.JSONDecodeError:
                yield clean_text(raw, max_chars), False


def iter_csv(path: Path, max_rows: int, max_chars: int) -> Iterable[tuple[str, bool]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if max_rows > 0 and index > max_rows:
                break
            yield csv_record_text(row, max_chars), True


def iter_source_texts(
    source: dict[str, Any],
    *,
    max_rows: int,
    max_chars: int,
) -> Iterable[tuple[str, bool]]:
    path = resolve_path(str(source.get("path", "")))
    fmt = str(source.get("format", "")).lower()
    if fmt == "jsonl":
        yield from iter_jsonl(path, max_rows, max_chars)
    elif fmt == "csv":
        yield from iter_csv(path, max_rows, max_chars)
    elif fmt == "json":
        text = json_record_text(json.loads(path.read_text(encoding="utf-8")), max_chars)
        yield text, True
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        yield clean_text(text, max_chars), True


def limited_counter(counter: Counter[str], top_k: int) -> Counter[str]:
    if top_k <= 0:
        return Counter(counter)
    return Counter(dict(top_ngrams(counter, top_k)))


def track_slug(track: str) -> str:
    """Filesystem-safe slug for a track name: 'ADVISOR-µ' → 'advisor_mu'."""
    import re
    s = track.replace("µ", "mu").replace("-", "_")
    s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    return s.lower() or "unknown"


def make_track_counters() -> dict[str, Counter[str]]:
    return {name: Counter() for name in tuple(REQUIRED_CHANNELS) + BINARY_ONLY_COUNTERS}


def update_counters(key_words: list[str], counters: dict[str, Counter[str]]) -> None:
    counters["unigram"].update(key_words)
    counters["bigram"].update(count_ngrams(key_words, 2))
    counters["trigram"].update(count_ngrams(key_words, 3))
    counters["quadgram"].update(count_ngrams(key_words, 4))
    coarse_bigram = extract_coarse_ngrams(key_words, 2)
    coarse_trigram = extract_coarse_ngrams(key_words, 3)
    counters["coarse_bigram"].update(coarse_bigram)
    counters["coarse_trigram"].update(coarse_trigram)
    counters["coarse_gram"].update(coarse_bigram)
    counters["coarse_gram"].update(coarse_trigram)
    counters["function_gram"].update(extract_function_grams(key_words))
    counters["tool_gram"].update(extract_tool_grams(key_words))


def candidate_key_rows(gram_container: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    channels = gram_container.get("channels", {})
    if not isinstance(channels, dict):
        return rows

    for channel_name in REQUIRED_CHANNELS:
        channel_rows = channels.get(channel_name, [])
        if not isinstance(channel_rows, list):
            continue
        for row in channel_rows:
            if not isinstance(row, dict):
                continue
            gram = str(row.get("gram", "")).strip()
            if not gram:
                continue
            role = str(row.get("role", "pattern_evidence"))
            phase = str(row.get("phase", "Wo"))
            key_words = [str(token) for token in row.get("tokens", []) if str(token)]
            count = int(row.get("count", 0) or 0)
            key = f"dataset:{channel_name}:{stable_hash(channel_name + ':' + gram)}"
            expansion = [
                f"gram:{channel_name}",
                f"role:{role}",
                f"phase:{phase}",
                f"count:{count}",
            ]
            expansion.extend(f"key-word:{key_word}" for key_word in key_words[:8])
            rows.append(
                {
                    "@type": "khanary.key_grammar.candidate.v1",
                    "key": key,
                    "key_word": gram,
                    "key_book": expansion,
                    "expansion": expansion,
                    "gram": gram,
                    "gram_class": channel_name,
                    "role": role,
                    "phase": phase,
                    "count": count,
                    "width": int(row.get("width", len(key_words) or 1) or 1),
                    "key_words": key_words,
                    "source": "dataset_ingest",
                    "target_container": "mx2lm:gram_container:v1",
                    "admission": "candidate",
                    "semantic_authority": False,
                }
            )
    return rows


def projection_coords(seed: str, weight: float) -> tuple[float, float, float]:
    h0 = fnv1a64(seed)
    h1 = fnv1a64(f"y:{seed}")
    h2 = fnv1a64(f"z:{seed}")
    coords = [
        signed_component(h0, 0) * weight,
        signed_component(h1, 8) * weight,
        signed_component(h2, 16) * weight,
    ]
    norm = math.sqrt(sum(component * component for component in coords))
    if not math.isfinite(norm) or norm <= 1e-9:
        return 1.0, 0.0, 0.0
    return coords[0] / norm, coords[1] / norm, coords[2] / norm


def vector_projection_container(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    projections: list[dict[str, Any]] = []
    for row in candidate_rows:
        gram = str(row.get("gram", ""))
        gram_class = str(row.get("gram_class", "gram"))
        phase = str(row.get("phase", "Wo"))
        role = str(row.get("role", "pattern_evidence"))
        count = max(1.0, float(row.get("count", 1) or 1))
        width = max(1.0, float(row.get("width", 1) or 1))
        weight = min(1.0, math.log1p(count) / (1.0 + width))
        x, y, z = projection_coords(f"{gram_class}:{phase}:{role}:{gram}", weight)
        projections.append(
            {
                "key": f"vector:dataset:{vector_key_safe(gram_class)}:{stable_hash(gram)}",
                "source_key": str(row.get("key", "")),
                "source_track": "Mx2LM",
                "source_phase": phase,
                "projection_track": "VECTOR-\u00b5",
                "basis": "deterministic_dataset_gram_hash_projection",
                "relation_model": "relationisa_conditioned_geometry",
                "x": x,
                "y": y,
                "z": z,
                "norm": 1.0,
                "semantic_authority": False,
            }
        )

    return {
        "@type": "vector.projection_container.v1",
        "key": "vector:dataset_projection_container:v1",
        "namespace": "VECTOR-\u00b5",
        "authority": "vector:projection-only",
        "phase": "Wo",
        "projection_id": "vector:dataset:bootstrap:v1:g0",
        "generation": 0,
        "learner_version": "dataset_ingest.deterministic_hash.v1",
        "topology_version": "dataset_ingest_candidates",
        "relation_model": "relationisa_conditioned_geometry",
        "alignment": "bootstrap_no_cross_generation_alignment",
        "similarity_scope": "semantic_similarity_subset_of_relatedness",
        "dimensions": 3,
        "projection_count": len(projections),
        "semantic_authority": False,
        "responsibility": "derived_geometry_projection_only",
        "projections": projections,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "data/ingest/khanary_dataset_sources.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/mx2lm")
    parser.add_argument("--grammar-out", type=Path, default=ROOT / "data/grammar/dataset.key-grammar.candidates.jsonl")
    parser.add_argument("--manifest-out", type=Path, default=ROOT / "data/ingest/dataset_ingestion_manifest.json")
    parser.add_argument("--max-rows", type=int, default=500, help="Rows per source; 0 means full stream.")
    parser.add_argument("--max-chars-per-row", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=512, help="Top grams per channel; 0 keeps all.")
    parser.add_argument(
        "--binary-top-k",
        type=int,
        default=50000,
        help=(
            "Top ngrams per binary section, independent of --top-k. "
            "Use 0 only when the corpus is small enough to hold all binary counters in memory."
        ),
    )
    parser.add_argument("--include-authority-sources", action="store_true")
    parser.add_argument("--binary-out", type=Path, default=ROOT / "data/mx2lm/dataset.ngram.bin")
    return parser.parse_args()


def emit_track_artifacts(
    track: str,
    counters: dict[str, Counter[str]],
    track_totals: dict[str, int],
    source_manifest_path: Path,
    out_dir: Path,
    grammar_out_dir: Path,
    binary_out_base: Path,
    top_k: int,
    binary_top_k: int,
) -> dict[str, Any]:
    """Emit gram/pattern/vector/binary artifacts for one track. Returns artifact summary."""
    slug = track_slug(track)

    observed_channel_counts = {name: int(len(counters[name])) for name in REQUIRED_CHANNELS}
    observed_binary_channel_counts = {name: int(len(counters[name])) for name in BINARY_ONLY_COUNTERS}

    limited = {
        name: limited_counter(counters[name], top_k)
        for name in tuple(REQUIRED_CHANNELS) + BINARY_ONLY_COUNTERS
    }

    gram_container = build_mx2lm_gram_container(
        unigram_map=limited["unigram"],
        bigram_map=limited["bigram"],
        trigram_map=limited["trigram"],
        quadgram_map=limited["quadgram"],
        coarse_gram_map=limited["coarse_gram"],
        function_gram_map=limited["function_gram"],
        tool_gram_map=limited["tool_gram"],
    )
    gram_container["key"] = f"mx2lm:dataset_gram_container:{slug}:v1"
    gram_container["track"] = track
    gram_container["authority"] = "dataset_ingest:candidate-evidence"
    gram_container["semantic_authority"] = False
    gram_container["admission"] = "candidate"
    gram_container["source_manifest"] = str(
        source_manifest_path.relative_to(ROOT) if source_manifest_path.is_relative_to(ROOT) else source_manifest_path
    )

    pattern_container = build_mx2lm_pattern_container(
        surface=f"dataset_ingest:{slug}:{stable_hash(json.dumps(track_totals, sort_keys=True), 24)}",
        gram_container=gram_container,
        source_track=track,
        source_phase="Wo",
        outcome_kind="dataset_pattern_observed",
        card=f"{track} dataset ingestion bootstrap",
        accepted=True,
    )
    pattern_container["key"] = f"mx2lm:dataset_pattern_container:{slug}:v1"
    pattern_container["track"] = track
    pattern_container["semantic_authority"] = False
    pattern_container["admission"] = "candidate"
    pattern_container["event_count"] = len(pattern_container.get("events", []))

    candidates = candidate_key_rows(gram_container)
    vector_container = vector_projection_container(candidates)

    gram_path = out_dir / f"{slug}_gram_container.json"
    pattern_path = out_dir / f"{slug}_pattern_container.json"
    vector_path = out_dir / f"{slug}_vector_projection.json"
    summary_path = out_dir / f"{slug}_ngram_summary.json"
    grammar_path = grammar_out_dir / f"dataset.{slug}.key-grammar.candidates.jsonl"
    binary_path = binary_out_base.parent / f"{binary_out_base.stem}.{slug}.bin"

    binary_limited = {
        name: limited_counter(counters[name], binary_top_k)
        for name in ("bigram", "trigram", "coarse_bigram", "coarse_trigram")
    }
    binary_summary = write_mx2lm_ngram_binary(
        binary_path,
        bigram_map=dict(binary_limited["bigram"]),
        trigram_map=dict(binary_limited["trigram"]),
        coarse_bigram_map=dict(binary_limited["coarse_bigram"]),
        coarse_trigram_map=dict(binary_limited["coarse_trigram"]),
    )
    binary_summary["top_k_per_section"] = binary_top_k
    binary_summary["observed_section_counts"] = {
        "bigrams": int(len(counters["bigram"])),
        "trigrams": int(len(counters["trigram"])),
        "coarse_bigrams": int(len(counters["coarse_bigram"])),
        "coarse_trigrams": int(len(counters["coarse_trigram"])),
    }
    binary_summary["emitted_section_counts"] = {
        "bigrams": int(len(binary_limited["bigram"])),
        "trigrams": int(len(binary_limited["trigram"])),
        "coarse_bigrams": int(len(binary_limited["coarse_bigram"])),
        "coarse_trigrams": int(len(binary_limited["coarse_trigram"])),
    }

    ngram_summary = {
        "@type": "mx2lm.dataset_ngram_summary.v1",
        "track": track,
        "source_manifest": str(
            source_manifest_path.relative_to(ROOT) if source_manifest_path.is_relative_to(ROOT) else source_manifest_path
        ),
        "top_k_per_channel": top_k,
        "binary_top_k_per_section": binary_top_k,
        "selection_policy": (
            "emit_all_observed_grams" if top_k <= 0
            else "emit_top_k_per_channel; observed_unique_channel_counts records the pre-cap population"
        ),
        "totals": track_totals,
        "observed_unique_channel_counts": observed_channel_counts,
        "observed_binary_channel_counts": observed_binary_channel_counts,
        "observed_total_unique_grams": int(sum(observed_channel_counts.values())),
        "emitted_channel_counts": gram_container.get("channel_counts", {}),
        "total_grams": gram_container.get("total_grams", 0),
        "semantic_authority": False,
        "admission": "candidate",
        "top": {
            name: [{"gram": gram, "count": int(count)} for gram, count in top_ngrams(limited[name], 32)]
            for name in REQUIRED_CHANNELS
        },
    }

    write_json(gram_path, gram_container)
    write_json(pattern_path, pattern_container)
    write_json(vector_path, vector_container)
    write_json(summary_path, ngram_summary)
    write_jsonl(grammar_path, candidates)

    def rel(p: Path) -> str:
        return str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)

    return {
        "track": track,
        "slug": slug,
        "candidate_rows": len(candidates),
        "projection_count": vector_container["projection_count"],
        "totals": track_totals,
        "observed_unique_channel_counts": observed_channel_counts,
        "emitted_channel_counts": gram_container.get("channel_counts", {}),
        "artifacts": {
            "gram_container": rel(gram_path),
            "pattern_container": rel(pattern_path),
            "vector_projection_container": rel(vector_path),
            "ngram_summary": rel(summary_path),
            "candidate_key_grammar": rel(grammar_path),
            "binary_snapshot": rel(binary_path),
        },
        "binary_snapshot": binary_summary,
    }


def main() -> int:
    args = parse_args()
    source_manifest_path = args.sources if args.sources.is_absolute() else ROOT / args.sources
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    sources = source_manifest.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("source manifest must contain a sources array")

    # One counter set per track — keeps phase-domain vocabularies separate.
    track_counters: dict[str, dict[str, Counter[str]]] = {}
    track_totals: dict[str, dict[str, int]] = {}
    source_summaries: list[dict[str, Any]] = []
    totals = {
        "sources_declared": len(sources),
        "sources_read": 0,
        "authority_sources_referenced": 0,
        "rows_seen": 0,
        "rows_used": 0,
        "json_decode_fallback_rows": 0,
        "lexical_units": 0,
    }

    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id", "source"))
        path = resolve_path(str(source.get("path", "")))
        track = str(source.get("track", "Mx2LM"))
        semantic_authority = bool(source.get("semantic_authority", False))
        summary: dict[str, Any] = {
            "id": source_id,
            "path": str(source.get("path", "")),
            "format": str(source.get("format", "")),
            "domain": str(source.get("domain", "")),
            "track": track,
            "role": str(source.get("role", "dataset_evidence")),
            "semantic_authority": semantic_authority,
            "exists": path.exists(),
            "rows_seen": 0,
            "rows_used": 0,
            "lexical_units": 0,
        }
        if semantic_authority and not args.include_authority_sources:
            totals["authority_sources_referenced"] += 1
            summary["read_mode"] = "authority_reference_only"
            source_summaries.append(summary)
            continue
        if not path.exists():
            summary["read_mode"] = "missing"
            source_summaries.append(summary)
            continue

        if track not in track_counters:
            track_counters[track] = make_track_counters()
            track_totals[track] = {"rows_seen": 0, "rows_used": 0, "lexical_units": 0}

        totals["sources_read"] += 1
        summary["read_mode"] = "streamed"
        for text, parsed_cleanly in iter_source_texts(
            source,
            max_rows=max(0, int(args.max_rows)),
            max_chars=max(0, int(args.max_chars_per_row)),
        ):
            summary["rows_seen"] += 1
            totals["rows_seen"] += 1
            track_totals[track]["rows_seen"] += 1
            if not parsed_cleanly:
                totals["json_decode_fallback_rows"] += 1
            if not text:
                continue
            key_words = tokenize(text)
            if not key_words:
                continue
            update_counters(key_words, track_counters[track])
            summary["rows_used"] += 1
            summary["lexical_units"] += len(key_words)
            totals["rows_used"] += 1
            totals["lexical_units"] += len(key_words)
            track_totals[track]["rows_used"] += 1
            track_totals[track]["lexical_units"] += len(key_words)
        source_summaries.append(summary)

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    grammar_base = args.grammar_out if args.grammar_out.is_absolute() else ROOT / args.grammar_out
    grammar_out_dir = grammar_base.parent
    manifest_out = args.manifest_out if args.manifest_out.is_absolute() else ROOT / args.manifest_out
    binary_out = args.binary_out if args.binary_out.is_absolute() else ROOT / args.binary_out
    binary_top_k = int(args.binary_top_k)
    top_k = int(args.top_k)

    track_results: list[dict[str, Any]] = []
    for track, counters in track_counters.items():
        result = emit_track_artifacts(
            track=track,
            counters=counters,
            track_totals=track_totals[track],
            source_manifest_path=source_manifest_path,
            out_dir=out_dir,
            grammar_out_dir=grammar_out_dir,
            binary_out_base=binary_out,
            top_k=top_k,
            binary_top_k=binary_top_k,
        )
        track_results.append(result)

    def rel(p: Path) -> str:
        return str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)

    manifest = {
        "@type": "khanary.dataset_ingestion_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "tools/build_dataset_ingestion.py",
        "source_manifest": rel(source_manifest_path),
        "policy": {
            "raw_dataset_copy": False,
            "external_sources_emit": "candidate_evidence",
            "dictionary_sources_emit": "authority_reference",
            "semantic_authority": False,
            "vector_projection": "derived_rebuildable",
            "gram_pools": "per_track",
        },
        "parameters": {
            "max_rows_per_source": int(args.max_rows),
            "max_chars_per_row": int(args.max_chars_per_row),
            "top_k_per_channel": top_k,
            "binary_top_k_per_section": binary_top_k,
            "include_authority_sources": bool(args.include_authority_sources),
        },
        "totals": totals,
        "tracks": track_results,
        "source_summaries": source_summaries,
    }
    write_json(manifest_out, manifest)

    registry = {
        "@type": "khanary.gram_pool_registry.v1",
        "generated_at_utc": manifest["generated_at_utc"],
        "generated_by": "tools/build_dataset_ingestion.py",
        "tracks": {
            r["track"]: {
                "slug": r["slug"],
                "artifacts": r["artifacts"],
                "rows_used": r["totals"]["rows_used"],
                "lexical_units": r["totals"]["lexical_units"],
            }
            for r in track_results
        },
    }
    write_json(out_dir / "gram_pool_registry.json", registry)

    total_candidates = sum(r["candidate_rows"] for r in track_results)
    total_projections = sum(r["projection_count"] for r in track_results)
    print(
        json.dumps(
            {
                "@type": "khanary.dataset_ingestion.run.v1",
                "verdict": "PASS",
                "tracks_emitted": len(track_results),
                "track_slugs": [r["slug"] for r in track_results],
                "sources_read": totals["sources_read"],
                "rows_used": totals["rows_used"],
                "lexical_units": totals["lexical_units"],
                "candidate_rows_total": total_candidates,
                "projection_count_total": total_projections,
                "manifest": rel(manifest_out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
