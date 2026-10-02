"""
extract_dictionary.py
Extract entries from the Sheridan English dictionary PDF into JSONL.

Output: data/gram/english-dictionary.jsonl
Each line: {"word": "...", "pos": "...", "pronunciation": "...", "definition": "..."}

Usage:
    python tools/extract_dictionary.py
    python tools/extract_dictionary.py --pdf C:/path/to/dict.pdf --out data/gram/out.jsonl
"""

import re
import sys
import json
import argparse
from pathlib import Path

try:
    import pypdf
except ImportError:
    sys.exit("pypdf not installed — run: pip install pypdf")

# Part-of-speech markers found in the dictionary
POS_TOKENS = ["v. a.", "v. n.", "v. def.", "v.", "f.", "a.", "ad.", "prep.",
              "prp.", "conj.", "interj.", "pron.", "part.", "r.", "s."]

# Map abbreviated POS to readable label
POS_MAP = {
    "v. a.": "verb_transitive",
    "v. n.": "verb_intransitive",
    "v. def.": "verb_defective",
    "v.":     "verb",
    "f.":     "noun",
    "r.":     "noun",
    "s.":     "noun",
    "a.":     "adjective",
    "ad.":    "adverb",
    "prep.":  "preposition",
    "prp.":   "preposition",
    "conj.":  "conjunction",
    "interj.":"interjection",
    "pron.":  "pronoun",
    "part.":  "participle",
}

# Normalise long-s and common OCR artefacts in definitions
_LONG_S_RE = re.compile(r'(?<=[a-z])f(?=[a-z])')  # interior 'f' as long-s (conservative)

def clean(text: str) -> str:
    # collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', ' ', text).strip()
    # long-s: 'f' between two lowercase letters → 's'  (conservative heuristic)
    text = _LONG_S_RE.sub('s', text)
    # common OCR noise
    text = text.replace('&', 'and').replace('Ihe', 'the').replace('ihe', 'the')
    return text

def extract_entries(pdf_path: str, start_page: int = 9) -> list[dict]:
    """
    Read all pages from start_page onward, concatenate text,
    then split on ALLCAPS headword lines.
    Returns list of entry dicts.
    """
    reader = pypdf.PdfReader(pdf_path)
    total = len(reader.pages)
    print(f"PDF: {total} pages — extracting from page {start_page + 1}", file=sys.stderr)

    # Gather raw text
    chunks = []
    for i in range(start_page, total):
        txt = reader.pages[i].extract_text() or ""
        # strip running-header lines (3 short ALL-CAPS words at start of page)
        lines = txt.splitlines()
        if lines and re.match(r'^[A-Z]{2,6}(\s+[A-Z]{2,6}){1,3}\s*$', lines[0]):
            lines = lines[1:]
        chunks.append("\n".join(lines))
        if (i - start_page) % 50 == 0:
            print(f"  read page {i + 1}/{total}", file=sys.stderr)

    full_text = "\n".join(chunks)

    # Split on headword pattern: line starting with 2+ uppercase letters
    # Headwords may contain hyphens/apostrophes: ANTHONY'S FIRE
    headword_re = re.compile(
        r'\n([A-Z][A-Z\'\-\s]{1,40}?),\s*([^\n.]{0,50}?)\.\s*(v\. [an]\.|v\.|f\.|r\.|s\.|a\.|ad\.|prep\.|prp\.|conj\.|interj\.|pron\.|part\.)\s*',
    )

    entries = []
    last_end = 0
    last_entry = None

    for m in headword_re.finditer(full_text):
        if last_entry is not None:
            # definition = text between previous match end and this match start
            defn_raw = full_text[last_end:m.start()]
            last_entry["definition"] = clean(defn_raw)
            entries.append(last_entry)

        word = m.group(1).strip()
        pron = m.group(2).strip()
        pos_raw = m.group(3).strip()
        pos = POS_MAP.get(pos_raw, pos_raw)

        last_entry = {
            "word": word,
            "pronunciation": pron,
            "pos": pos,
            "definition": "",
        }
        last_end = m.end()

    # flush last entry
    if last_entry is not None:
        last_entry["definition"] = clean(full_text[last_end:last_end + 300])
        entries.append(last_entry)

    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",  default=r"C:/Users/canna/Downloads/english-distionary.pdf")
    ap.add_argument("--out",  default="data/gram/english-dictionary.jsonl")
    ap.add_argument("--start-page", type=int, default=9,
                    help="0-indexed page to start extraction (skip front matter)")
    args = ap.parse_args()

    entries = extract_entries(args.pdf, start_page=args.start_page)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            if len(e["word"]) < 2 or len(e["definition"]) < 3:
                continue
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} entries → {out_path}", file=sys.stderr)

    # Also write a flat word-list for synonym_collapse / ELIZA registry
    words_path = out_path.with_name("english-wordlist.txt")
    with open(words_path, "w", encoding="utf-8") as f:
        seen = set()
        for e in entries:
            w = e["word"].strip()
            if w and w not in seen:
                f.write(w + "\n")
                seen.add(w)
    print(f"Wrote {len(seen)} headwords → {words_path}", file=sys.stderr)

    # Write a triples JSONL for SemanticGraph (T-REx style)
    triples_path = out_path.with_name("english-triples.jsonl")
    with open(triples_path, "w", encoding="utf-8") as f:
        for e in entries:
            if not e["definition"]:
                continue
            triple = {
                "subject": e["word"],
                "relation": "means",
                "object": e["definition"][:200],
                "pos": e["pos"],
                "pronunciation": e["pronunciation"],
            }
            f.write(json.dumps(triple, ensure_ascii=False) + "\n")
    print(f"Wrote triples → {triples_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
