r"""
download_wikipedia_article_jsonl.py
Download a Wikipedia article via the public API and normalize it to data/*.jsonl.

Usage:
    python tools\download_wikipedia_article_jsonl.py --url https://en.wikipedia.org/wiki/Philosophy_of_mind
    python tools\download_wikipedia_article_jsonl.py --title Philosophy_of_mind --out data\wikipedia_philosophy_of_mind.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://en.wikipedia.org/w/api.php"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="Wikipedia article URL (e.g., https://en.wikipedia.org/wiki/Philosophy_of_mind)")
    ap.add_argument("--title", default="", help="Wikipedia article title or slug (e.g., Philosophy_of_mind)")
    ap.add_argument("--data-dir", default="data", help="Output directory (default: data)")
    ap.add_argument("--out", default="", help="Output JSONL path (default: data/wikipedia_<slug>.jsonl)")
    ap.add_argument("--min-chars", type=int, default=64, help="Minimum cleaned paragraph length to keep")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    return ap.parse_args()


def title_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    if not parsed.netloc or "wikipedia.org" not in parsed.netloc.lower():
        raise ValueError("URL must be a wikipedia.org article URL")
    path = parsed.path or ""
    if "/wiki/" not in path:
        raise ValueError("URL path must contain /wiki/<title>")
    slug = path.split("/wiki/", 1)[1].strip("/")
    if not slug:
        raise ValueError("Missing article title in URL")
    return urllib.parse.unquote(slug)


def normalize_title(raw_title: str, raw_url: str) -> str:
    if raw_title.strip():
        return raw_title.strip().replace(" ", "_")
    if raw_url.strip():
        return title_from_url(raw_url)
    raise ValueError("Provide either --title or --url")


def fetch_extract(title: str) -> tuple[str, str]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "titles": title,
    }
    query = urllib.parse.urlencode(params)
    req_url = f"{API_URL}?{query}"
    req = urllib.request.Request(req_url, headers={"User-Agent": "KHANARY.CPP dataset tool"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))

    pages = (((payload or {}).get("query") or {}).get("pages") or [])
    if not pages:
        raise RuntimeError("No pages returned by Wikipedia API")
    page = pages[0]
    page_title = str(page.get("title") or title)
    extract = str(page.get("extract") or "")
    if not extract.strip():
        raise RuntimeError(f"Empty extract for title: {title}")
    return page_title, extract


def clean_paragraph(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_paragraphs(extract: str, min_chars: int) -> list[str]:
    chunks = re.split(r"\n\s*\n+", extract)
    out: list[str] = []
    for chunk in chunks:
        clean = clean_paragraph(chunk)
        if len(clean) < min_chars:
            continue
        # Skip heading-like lines.
        if re.fullmatch(r"[A-Za-z0-9 ,.'’\-:()]+", clean) and clean.count(" ") <= 6 and not clean.endswith("."):
            continue
        out.append(clean)
    return out


def default_out_path(data_dir: Path, title: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", title.replace(" ", "_")).strip("_").lower()
    return data_dir / f"wikipedia_{slug}.jsonl"


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (use --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    try:
        title = normalize_title(args.title, args.url)
        page_title, extract = fetch_extract(title)
        paragraphs = split_paragraphs(extract, min_chars=max(1, int(args.min_chars)))
        if not paragraphs:
            print("ERROR: no paragraphs after filtering", file=sys.stderr)
            return 2

        data_dir = Path(args.data_dir)
        out_path = Path(args.out) if args.out.strip() else default_out_path(data_dir, page_title)
        source_url = args.url.strip() or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

        rows: list[dict[str, Any]] = []
        for i, paragraph in enumerate(paragraphs, start=1):
            rows.append(
                {
                    "dataset": "wikipedia",
                    "task": "knowledge",
                    "source": "wikipedia",
                    "source_url": source_url,
                    "title": page_title,
                    "title_key": title,
                    "paragraph_id": i,
                    "text": paragraph,
                }
            )

        write_jsonl(out_path, rows, overwrite=args.overwrite)
        print(f"WROTE {out_path} rows={len(rows)}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
