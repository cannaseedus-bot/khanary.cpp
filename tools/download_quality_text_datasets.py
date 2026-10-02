"""
download_quality_text_datasets.py
Download high-signal text corpora and normalize them to top-level data/*.jsonl.

This is meant for Mx2LM semantic/ngram training prep (not numeric spam benchmarks).

Usage:
    python tools\\download_quality_text_datasets.py
    python tools\\download_quality_text_datasets.py --max-rows 50000
    python tools\\download_quality_text_datasets.py --only ag_news,wikitext2,squad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets not installed — run: pip install datasets")


TransformFn = Callable[[dict[str, Any], str], dict[str, Any] | None]


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _ag_news_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    text = _clean_text(str(row.get("text", "")))
    if not text:
        return None
    return {
        "dataset": "ag_news",
        "split": split,
        "task": "classification",
        "label": int(row.get("label", -1)),
        "text": text,
    }


def _wikitext_row(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    text = _clean_text(str(row.get("text", "")))
    if not text:
        return None
    # Skip markup-like section headers to keep usable natural text.
    if text.startswith("=") and text.endswith("="):
        return None
    return {
        "dataset": "wikitext2",
        "split": split,
        "task": "language_modeling",
        "text": text,
    }


def _squad_row(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    context = _clean_text(str(row.get("context", "")))
    question = _clean_text(str(row.get("question", "")))
    answers = row.get("answers", {}) or {}
    answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
    answer = _clean_text(str(answer_texts[0])) if answer_texts else ""
    if not context or not question:
        return None
    text = f"question: {question}\ncontext: {context}"
    if answer:
        text += f"\nanswer: {answer}"
    return {
        "dataset": "squad",
        "split": split,
        "task": "qa",
        "id": str(row.get("id", "")),
        "title": _clean_text(str(row.get("title", ""))),
        "question": question,
        "context": context,
        "answer": answer,
        "text": text,
    }


DATASET_SPECS: list[dict[str, Any]] = [
    {
        "name": "ag_news",
        "hf_id": "ag_news",
        "hf_config": None,
        "split": "train",
        "output_name": "ag_news_train.jsonl",
        "transform": _ag_news_row,
    },
    {
        "name": "wikitext2",
        "hf_id": "wikitext",
        "hf_config": "wikitext-2-raw-v1",
        "split": "train",
        "output_name": "wikitext2_train.jsonl",
        "transform": _wikitext_row,
    },
    {
        "name": "squad",
        "hf_id": "squad",
        "hf_config": None,
        "split": "train",
        "output_name": "squad_train.jsonl",
        "transform": _squad_row,
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="data",
        help="Target directory for output JSONL files (default: data)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Per-dataset max rows to write (0 = all rows).",
    )
    ap.add_argument(
        "--only",
        default="",
        help="Comma-separated dataset names to download (ag_news,wikitext2,squad).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return ap.parse_args()


def selected_specs(only: str) -> list[dict[str, Any]]:
    if not only.strip():
        return DATASET_SPECS
    wanted = {part.strip() for part in only.split(",") if part.strip()}
    return [spec for spec in DATASET_SPECS if spec["name"] in wanted]


def write_dataset(
    spec: dict[str, Any],
    out_path: Path,
    max_rows: int,
    overwrite: bool,
) -> int:
    if out_path.exists() and not overwrite:
        print(f"SKIP {spec['name']}: {out_path} already exists (use --overwrite)", file=sys.stderr)
        return -1

    hf_id = spec["hf_id"]
    hf_config = spec["hf_config"]
    split = spec["split"]
    transform: TransformFn = spec["transform"]

    if hf_config:
        ds = load_dataset(hf_id, hf_config, split=split)
    else:
        ds = load_dataset(hf_id, split=split)

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in ds:
            rec = transform(row, split)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if max_rows > 0 and written >= max_rows:
                break
    return written


def main() -> int:
    args = parse_args()
    out_dir = Path(args.data_dir)
    specs = selected_specs(args.only)
    if not specs:
        print("ERROR: no datasets selected", file=sys.stderr)
        return 2

    total_written = 0
    for spec in specs:
        out_path = out_dir / spec["output_name"]
        try:
            written = write_dataset(spec, out_path, args.max_rows, args.overwrite)
        except Exception as exc:
            print(f"FAIL {spec['name']}: {exc}", file=sys.stderr)
            return 1
        if written >= 0:
            print(f"WROTE {spec['name']} -> {out_path} rows={written}", file=sys.stderr)
            total_written += written

    print(f"DONE total_rows={total_written}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
