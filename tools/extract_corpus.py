"""Extract plain text sentences from JSONL/CSV/Parquet corpus files for SRILM ngram-count.

Handles schemas found in the KHANARY/K'UHUL corpus:
  1. {"instruction"/"INSTRUCTION": "...", "output"/"OUTPUT": "..."} -- qa-corpus, G-Code
  2. {"query"/"prompt": "...", "response": "..."}     -- instruct_merged, help_merged, smgm16
  3. {"question": "...", "answer"/"reference": "..."}  -- deepseek, qwen, baichuan, KodCode
  4. {"mapping": {id: {message: {content: {parts}}}}}  -- conversations-gpt (ChatGPT export)
  5. {"source":"doc", "content":"[<JSON-encoded export>]"} -- G-Code wrapper
  6. {"record": {"content": "..."}}                   -- mistral-chat
  7. {"messages": [{"role": "...", "content": "..."}]} -- OpenAI messages / coder_outputs chunks
  8. {"input": "...", "output": "..."}                -- smgm16 coder_outputs (system_user_assistant)
  9. {"coarse_gram": "...", "id": "...", ...}          -- semantic-json (AST/math gold mine)
 10. {"text": "..."}                                   -- plain-text fallback
 11. CSV with "selftext"/"title" columns               -- Reddit jokes
 12. Parquet: KodCode question/solution/test columns   -- requires pyarrow

Accepts files and/or directories (directories are scanned recursively for *.jsonl).
Output: one sentence per line, lowercased, punctuation stripped at ends.

Usage:
  py -3 tools\\extract_corpus.py E:/data/coder_outputs/ E:/data/KodCode/train.parquet -o corpus.txt
  py -3 tools\\extract_corpus.py E:/data/semantic-json E:/data/math_data/merged.jsonl -o corpus.txt
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean(text: str) -> list[str]:
    """Split text into non-empty sentences; lowercase; strip leading/trailing punct."""
    lines = []
    for line in re.split(r'[.!?\n]+', text):
        line = line.strip().lower()
        if len(line) < 8 or line.startswith('#') or line.startswith('```'):
            continue
        if ' ' not in line:
            continue
        lines.append(line)
    return lines


def clean_code(text: str) -> list[str]:
    """Extract natural-language tokens from code: comments, docstrings, identifiers."""
    lines = []
    # Single-line comments
    for m in re.finditer(r'(?:#|//)\s*(.+)', text):
        lines.extend(clean(m.group(1)))
    # Docstrings / triple-quoted strings
    for m in re.finditer(r'"""(.*?)"""', text, re.DOTALL):
        lines.extend(clean(m.group(1)))
    for m in re.finditer(r"'''(.*?)'''", text, re.DOTALL):
        lines.extend(clean(m.group(1)))
    # Function / class names → space-separated words via snake/camel splitting
    for name in re.findall(r'\b(?:def|class|function)\s+([A-Za-z_]\w*)', text):
        words = re.sub(r'([A-Z])', r' \1', name).replace('_', ' ').strip().lower()
        if ' ' in words and len(words) > 6:
            lines.append(words)
    return lines


def dot_to_words(s: str) -> list[str]:
    """Convert dot-notation like 'math.constant.pi' → 'math constant pi' if multi-word."""
    words = s.replace('.', ' ').replace('_', ' ').replace('-', ' ').strip().lower()
    if ' ' in words and len(words) > 6:
        return [words]
    return []


# ── Record extractors ──────────────────────────────────────────────────────────

def extract_flat(rec: dict) -> list[str]:
    """instruction/output, query/response, question/answer, prompt/response, input/output."""
    parts = []
    for key in ('instruction', 'INSTRUCTION', 'query', 'prompt', 'question', 'input'):
        if key in rec and isinstance(rec[key], str):
            parts.extend(clean(rec[key]))
            break
    for key in ('output', 'OUTPUT', 'response', 'answer', 'reference'):
        if key in rec and isinstance(rec[key], str):
            val = rec[key]
            # coder_outputs: output may be code — extract NL from it
            if any(kw in val for kw in ('def ', 'class ', 'function ', '//', '/*', '"""')):
                parts.extend(clean_code(val))
            else:
                parts.extend(clean(val))
            break
    if not parts and 'text' in rec and isinstance(rec['text'], str):
        parts.extend(clean(rec['text']))
    return parts


def extract_messages(rec: dict) -> list[str]:
    """OpenAI messages format: [{"role": "user"/"assistant", "content": "..."}]."""
    lines = []
    for msg in rec.get('messages', []):
        if not isinstance(msg, dict):
            continue
        if msg.get('role') not in ('user', 'assistant'):
            continue
        content = msg.get('content', '')
        if not isinstance(content, str) or not content.strip():
            continue
        if any(kw in content for kw in ('def ', 'class ', 'function ', '//', '/*', '"""')):
            lines.extend(clean_code(content))
        else:
            lines.extend(clean(content))
    return lines


def walk_chatgpt_tree(mapping: dict) -> list[str]:
    """Walk ChatGPT export mapping tree."""
    lines = []
    for node in mapping.values():
        msg = node.get('message')
        if not msg:
            continue
        role = (msg.get('author') or {}).get('role', '')
        if role not in ('user', 'assistant'):
            continue
        for part in (msg.get('content') or {}).get('parts', []):
            if isinstance(part, str) and part.strip():
                lines.extend(clean(part))
    return lines


def extract_semantic_json(rec: dict) -> list[str]:
    """semantic-json records: coarse_gram dot-paths, id labels, and any NL string values.

    These have no user/assistant tags but carry rich semantic vocabulary:
    coarse_gram 'math.constant.pi' → 'math constant pi'
    id 'builtin.pi.mathml' → 'builtin pi mathml'
    String fields > 20 chars that don't look like XML/JSON → extracted as sentences.
    """
    lines = []

    # coarse_gram and id as space-separated vocabulary
    for key in ('coarse_gram', 'id', 'kind', 'schema'):
        val = rec.get(key)
        if isinstance(val, str):
            lines.extend(dot_to_words(val))

    # Walk all string values for NL content
    def walk(obj, depth=0):
        if depth > 4:
            return
        if isinstance(obj, str):
            # Skip XML/MathML, short tokens, pure numbers
            if len(obj) > 20 and '<' not in obj and '{' not in obj and obj.count(' ') >= 2:
                lines.extend(clean(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:8]:  # cap recursion width
                walk(item, depth + 1)

    # Skip the heavy nested fields that are pure numeric/binary data
    skip_keys = {'fibonacci', 'geodesic', 'payload_contract', 'absorb', 'mathml'}
    for k, v in rec.items():
        if k not in skip_keys:
            walk(v)

    return lines


def extract_record(rec: dict) -> list[str]:
    # ChatGPT export
    if 'mapping' in rec and isinstance(rec['mapping'], dict):
        return walk_chatgpt_tree(rec['mapping'])
    # G-Code doc wrapper
    if rec.get('source') == 'doc' and isinstance(rec.get('content'), str):
        try:
            inner = json.loads(rec['content'])
            if isinstance(inner, list):
                lines = []
                for conv in inner:
                    if isinstance(conv, dict) and 'mapping' in conv:
                        lines.extend(walk_chatgpt_tree(conv['mapping']))
                return lines
        except json.JSONDecodeError:
            pass
    # Mistral-chat
    record_obj = rec.get('record')
    if isinstance(record_obj, dict) and isinstance(record_obj.get('content'), str):
        return clean(record_obj['content'])
    # semantic-json: identified by coarse_gram or phase_trajectory
    if 'coarse_gram' in rec or 'phase_trajectory' in rec or 'geodesic' in rec:
        return extract_semantic_json(rec)
    # OpenAI messages
    if 'messages' in rec and isinstance(rec['messages'], list):
        lines = extract_messages(rec)
        # smgm16: also has input/output/question at top level — append those
        for key in ('input', 'question'):
            if key in rec and isinstance(rec[key], str):
                lines.extend(clean(rec[key]))
        return lines
    return extract_flat(rec)


# ── File processors ────────────────────────────────────────────────────────────

def process_csv(path: Path, out) -> int:
    count = 0
    with path.open(encoding='utf-8', errors='replace', newline='') as f:
        for row in csv.DictReader(f):
            for col in ('selftext', 'title'):
                val = row.get(col, '')
                if val and val not in ('[deleted]', '[removed]'):
                    for sentence in clean(val):
                        out.write(sentence + '\n')
                        count += 1
    return count


def process_parquet(path: Path, out) -> int:
    """Extract text from Parquet files.  Requires: pip install pyarrow"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f'  [skip] {path.name}: pyarrow not installed — pip install pyarrow', file=sys.stderr)
        return 0

    tbl = pq.read_table(path)
    cols = tbl.schema.names
    count = 0

    # KodCode: question (NL) + solution (code) + test (code)
    nl_cols   = [c for c in ('question', 'instruction', 'prompt', 'query', 'text') if c in cols]
    code_cols = [c for c in ('solution', 'output', 'answer', 'code') if c in cols]
    test_cols = [c for c in ('test',) if c in cols]

    rows = tbl.to_pydict()
    n = len(tbl)
    for i in range(n):
        for col in nl_cols:
            val = rows[col][i]
            if isinstance(val, str) and val.strip():
                for s in clean(val):
                    out.write(s + '\n')
                    count += 1
        for col in code_cols + test_cols:
            val = rows[col][i]
            if isinstance(val, str) and val.strip():
                for s in clean_code(val):
                    out.write(s + '\n')
                    count += 1
    return count


def process_jsonl(path: Path, out) -> int:
    count = 0
    # Try utf-8-sig first (handles BOM from coder_outputs chunks), fall back to utf-8
    for enc in ('utf-8-sig', 'utf-8'):
        try:
            with path.open(encoding=enc, errors='replace') as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for sentence in extract_record(rec):
                        out.write(sentence + '\n')
                        count += 1
            return count
        except UnicodeDecodeError:
            count = 0
            continue
    return count


def process_file(path: Path, out) -> int:
    ext = path.suffix.lower()
    if ext == '.csv':
        return process_csv(path, out)
    if ext == '.parquet':
        return process_parquet(path, out)
    return process_jsonl(path, out)


def collect_paths(inputs: list[Path]) -> list[Path]:
    """Expand directories to supported corpus files; keep explicit files as-is."""
    result: list[Path] = []
    for p in inputs:
        if p.is_dir():
            found = (
                sorted(p.rglob('*.jsonl'))
                + sorted(p.rglob('*.csv'))
                + sorted(p.rglob('*.parquet'))
            )
            if not found:
                print(f'[warn] no .jsonl/.csv/.parquet in {p}', file=sys.stderr)
            result.extend(found)
        elif p.exists():
            result.append(p)
        else:
            print(f'[warn] not found: {p}', file=sys.stderr)
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', type=Path,
                    help='JSONL/CSV/Parquet files or directories (dirs scanned for *.jsonl)')
    ap.add_argument('-o', '--output', type=Path, default=Path(r'E:\models\LM-SDDS.v1.0\corpus.txt'),
                    help='output plain-text file (default: E:\\models\\LM-SDDS.v1.0\\corpus.txt)')
    ap.add_argument('--append', action='store_true',
                    help='append to existing output instead of overwriting')
    args = ap.parse_args()

    paths = collect_paths(args.files)
    if not paths:
        print('ERROR: no input files found', file=sys.stderr)
        sys.exit(1)

    mode = 'a' if args.append else 'w'
    total = 0
    with args.output.open(mode, encoding='utf-8') as out:
        for path in paths:
            n = process_file(path, out)
            print(f'  {path.name}: {n:,} sentences')
            total += n

    existing = ''
    if args.append:
        existing = ' (appended)'
    print(f'\nTotal: {total:,} sentences → {args.output}{existing}')
    print()
    print('Next: build the LM with SRILM:')
    print(f'  tools\\train_srilm.bat')


if __name__ == '__main__':
    main()
