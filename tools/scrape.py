"""
scrape.py — local file corpus builder for KHANARY CODEGEN expert micronauts

Walks C:/ and E:/ collecting source files, extracts code-meaningful bigrams,
and writes JSONL entries tagged with the domain expert micronaut they belong to.

Supported domains and their expert micronauts:
  kuhul / klsl / khl / kxml  →  CODEGEN-KUHUL-µ / CODEGEN-KLSL-µ / CODEGEN-KHL-µ
  json                        →  CODEGEN-JSON-µ
  py                          →  CODEGEN-PY-µ
  js / ts                     →  CODEGEN-JS-µ
  cpp / hlsl                  →  CODEGEN-CPP-µ
  cs                          →  CODEGEN-CSHARP-µ
  css                         →  CODEGEN-CSS-µ
  html                        →  CODEGEN-HTML-µ

Usage:
    # Full corpus (all domains, single file)
    py -3 tools\\scrape.py

    # K'UHUL expert only (filtered single file)
    py -3 tools\\scrape.py --domains kuhul --out data/gram/codegen-kuhul.jsonl

    # Split into per-domain files under data/gram/  (one file per expert)
    py -3 tools\\scrape.py --split

    # Specific drives, split mode
    py -3 tools\\scrape.py --drives C E --split --out-dir data/gram

    # Dry-run: count files without writing
    py -3 tools\\scrape.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import IO

# Windows cp1252 console can't print some Unicode — force UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────

EXTENSIONS = {
    ".kuhul", ".klsl", ".khl", ".kxml",   # K'UHUL family
    ".json",                               # manifests / schemas / contracts
    ".py",                                 # Python
    ".js", ".ts",                          # JavaScript / TypeScript
    ".cpp", ".h", ".hlsl",                 # C++ / HLSL shaders
    ".cs",                                 # C#
    ".css",                                # CSS
    ".html", ".htm",                       # HTML
}

# Domain → expert micronaut name
DOMAIN_EXPERT: dict[str, str] = {
    "kuhul": "CODEGEN-KUHUL-µ",
    "klsl":  "CODEGEN-KLSL-µ",
    "khl":   "CODEGEN-KHL-µ",
    "kxml":  "CODEGEN-KUHUL-µ",   # KXML is K'UHUL binary tape format
    "json":  "CODEGEN-JSON-µ",
    "py":    "CODEGEN-PY-µ",
    "js":    "CODEGEN-JS-µ",
    "ts":    "CODEGEN-JS-µ",
    "cpp":   "CODEGEN-CPP-µ",
    "h":     "CODEGEN-CPP-µ",
    "hlsl":  "CODEGEN-CPP-µ",     # HLSL lives alongside C++ GPU code
    "cs":    "CODEGEN-CSHARP-µ",
    "css":   "CODEGEN-CSS-µ",
    "html":  "CODEGEN-HTML-µ",
    "htm":   "CODEGEN-HTML-µ",
}

# Preset domain groups (expand --domains <preset>)
DOMAIN_PRESETS: dict[str, list[str]] = {
    "kuhul":   ["kuhul", "klsl", "khl", "kxml", "json"],
    "cpp":     ["cpp", "h", "hlsl"],
    "web":     ["js", "ts", "css", "html", "htm"],
    "dotnet":  ["cs", "json"],
    "all":     list(DOMAIN_EXPERT.keys()),
}

# Directories to prune entirely (case-insensitive name match)
SKIP_DIRS = {
    "windows", "program files", "program files (x86)",
    "$recycle.bin", "system volume information",
    "appdata", "programdata",
    "node_modules", ".git", "__pycache__",
    ".vs", ".vscode", "obj", "dist", "build", ".next",
    "site-packages", "venv", ".venv", "env",
}

# Skip files larger than these thresholds (bytes)
MAX_SIZE: dict[str, int] = {
    ".json":  400_000,
    ".cpp":   200_000,
    ".h":     200_000,
    ".hlsl":  200_000,
    ".py":    200_000,
    ".js":    200_000,
    ".ts":    200_000,
    ".cs":    200_000,
    ".css":   200_000,
    ".html":  200_000,
    ".htm":   200_000,
    ".kuhul": 500_000,
    ".klsl":  500_000,
    ".khl":   500_000,
    ".kxml":  500_000,
}
DEFAULT_MAX = 200_000

# ── Token extraction ──────────────────────────────────────────────────────────

# Python
_PY_DEF    = re.compile(r'^\s*(?:def|class|async def)\s+([A-Za-z_]\w*)', re.M)

# JavaScript / TypeScript
_JS_DEF    = re.compile(r'(?:function\s+([A-Za-z_$]\w*)|(?:const|let|var|class)\s+([A-Za-z_$]\w*))', re.M)
_TS_IFACE  = re.compile(r'(?:interface|type|class|enum)\s+([A-Za-z_]\w*)', re.M)

# C++ / HLSL  (HLSL shares C-like syntax)
_CPP_DECL  = re.compile(
    r'(?:class|struct|namespace|enum)\s+([A-Za-z_]\w*)'
    r'|(?:^|\s)([A-Za-z_]\w+)\s*\([^)]*\)\s*(?:const\s*)?\{',
    re.M)
_HLSL_SEM  = re.compile(r'\b(SV_\w+|register\s*\(\w+\d+\)|cbuffer\s+\w+|numthreads)', re.M)

# C#
_CS_DEF    = re.compile(
    r'(?:class|interface|struct|enum|delegate|record)\s+([A-Za-z_]\w*)'
    r'|(?:public|private|protected|internal|static)\s+\w[\w<>[\],\s]*\s+([A-Za-z_]\w+)\s*\(',
    re.M)

# CSS: property names and selector keywords
_CSS_PROP  = re.compile(r'^\s{0,4}([a-z][a-z0-9-]{2,})\s*:', re.M)
_CSS_SEL   = re.compile(r'(?:^|\s)([.#]?[a-z][a-z0-9_-]{2,})(?:\s*[{,>+~])', re.M)

# HTML: tag names, id/class attribute values
_HTML_TAG  = re.compile(r'<([a-z][a-z0-9-]{1,})', re.M)
_HTML_ATTR = re.compile(r'(?:class|id|data-\w+)="([^"]{3,60})"', re.M)

# K'UHUL / KLSL / KHL / KXML: @directives, ⟁KEYs, bracket phase ops, #pragmas
_KUHUL_TOK = re.compile(r'(@[a-z_]+|⟁[A-Z_]+|\[[A-Z][a-z\']+\]|#[A-Za-z_]+)', re.M)

# JSON: top-level string keys (manifests / schemas / contracts)
_JSON_KEY  = re.compile(r'"([A-Za-z_$⟁µ@][^"]{1,60})"(?:\s*:)')

# Generic identifiers (≥3 chars, alnum+_) — fallback for any type
_IDENT     = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b')

# Noise words to drop from bigram candidates
_STOPWORDS = {
    "the","and","for","not","this","with","that","from","have","are",
    "was","will","can","but","its","then","else","true","false","null",
    "undefined","void","return","import","export","const","let","var",
    "def","class","self","new","int","str","bool","float","double",
    "None","else","pass","break","continue","try","catch","throw",
    "public","private","protected","static","final","override",
    "include","pragma","ifdef","ifndef","endif","using","namespace",
    "get","set","add","remove","value","result","output","input",
}


def extract_tokens(path: Path, text: str) -> list[str]:
    ext = path.suffix.lower()
    toks: list[str] = []

    if ext in {".kuhul", ".klsl", ".khl", ".kxml"}:
        toks = _KUHUL_TOK.findall(text)
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext == ".py":
        toks = _PY_DEF.findall(text)
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext in {".js", ".ts"}:
        raw = _JS_DEF.findall(text) + _TS_IFACE.findall(text)
        toks = [x for group in raw for x in (group if isinstance(group, tuple) else [group]) if x]
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext in {".cpp", ".h"}:
        raw = _CPP_DECL.findall(text)
        toks = [x for group in raw for x in group if x]
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext == ".hlsl":
        raw = _CPP_DECL.findall(text)
        toks = [x for group in raw for x in group if x]
        toks += _HLSL_SEM.findall(text)
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext == ".cs":
        raw = _CS_DEF.findall(text)
        toks = [x for group in raw for x in group if x]
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext == ".css":
        toks = _CSS_PROP.findall(text) + _CSS_SEL.findall(text)
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext in {".html", ".htm"}:
        toks = _HTML_TAG.findall(text)
        # class/id attrs can be space-separated; split them
        for attr in _HTML_ATTR.findall(text):
            toks.extend(attr.split())
        toks += [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    elif ext == ".json":
        toks = _JSON_KEY.findall(text)

    else:
        toks = [t for t in _IDENT.findall(text) if t not in _STOPWORDS]

    # Deduplicate preserving order, keep ≥ 3 chars
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        t = t.strip()
        if len(t) >= 3 and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def make_bigrams(toks: list[str]) -> list[tuple[str, str]]:
    return [(toks[i], toks[i + 1]) for i in range(len(toks) - 1)]


def snippet_for(text: str, tok: str, window: int = 120) -> str:
    pos = text.find(tok)
    if pos < 0:
        return text[:window].replace("\n", " ").replace("\t", " ")
    start = max(0, pos - window // 2)
    raw = text[start: start + window]
    return raw.replace("\n", " ").replace("\t", " ").replace('"', "'")


# ── Drive walker ──────────────────────────────────────────────────────────────

def should_skip_dir(name: str) -> bool:
    return name.lower() in SKIP_DIRS or name.startswith(".")


def walk_drive(root: str):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in EXTENSIONS:
                yield Path(dirpath) / fname


# ── Corpus writer ─────────────────────────────────────────────────────────────

def process_file(fpath: Path, out_fh: IO[str], domain_override: str | None = None) -> int:
    ext = fpath.suffix.lower()
    max_sz = MAX_SIZE.get(ext, DEFAULT_MAX)
    try:
        sz = fpath.stat().st_size
        if sz > max_sz or sz < 20:
            return 0
        text = fpath.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return 0

    domain   = domain_override or ext.lstrip(".")
    micronaut = DOMAIN_EXPERT.get(domain, "CODEGEN-µ")
    toks = extract_tokens(fpath, text)
    if len(toks) < 2:
        return 0

    bigrams = make_bigrams(toks)
    written = 0
    seen_slugs: set[str] = set()

    for a, b in bigrams:
        slug = f"{a}_{b}"
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        snip = snippet_for(text, a)
        entry = {
            "key":       slug,
            "text":      snip,
            "surfaces":  [slug, a, b, domain],
            "phase":     "Wo",
            "domain":    domain,
            "micronaut": micronaut,
            "source":    str(fpath),
        }
        out_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        written += 1

    return written


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="KHANARY local code corpus scraper — builds per-expert CODEGEN-µ corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--drives",   nargs="+", default=["C", "E"],
                    help="Drive letters to scan (default: C E)")
    ap.add_argument("--out",      default="data/gram/codegen-ngrams.jsonl",
                    help="Output JSONL path for single-file mode (default: data/gram/codegen-ngrams.jsonl)")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Count files without writing corpus")
    ap.add_argument("--limit",    type=int, default=0,
                    help="Max files per drive (0=unlimited, useful for testing)")
    ap.add_argument("--domains",  nargs="+", default=[],
                    metavar="DOMAIN",
                    help="Emit only entries for these domains or presets "
                         "(presets: kuhul cpp web dotnet all). "
                         "Default: all domains.")
    ap.add_argument("--split",    action="store_true",
                    help="Write one codegen-{domain}.jsonl per domain instead of a single file")
    ap.add_argument("--out-dir",  default="data/gram",
                    help="Output directory for --split mode (default: data/gram)")
    ap.add_argument("--paths",    nargs="+", default=[],
                    metavar="PATH",
                    help="Restrict scanning to these directory prefixes (any drive). "
                         "Useful to limit noisy domains: "
                         "--domains json --paths E:\\\\models\\\\_khanary_inspect\\\\KHANARY.CPP")
    args = ap.parse_args()

    # Expand domain presets
    domain_filter: set[str] = set()
    for d in args.domains:
        if d in DOMAIN_PRESETS:
            domain_filter.update(DOMAIN_PRESETS[d])
        else:
            domain_filter.add(d)

    # Normalise --paths to resolved absolute strings for prefix-matching
    path_filter: list[str] = [str(Path(p).resolve()) for p in args.paths]

    if domain_filter:
        print(f"[scrape] domain filter: {sorted(domain_filter)}")
    if path_filter:
        print(f"[scrape] path filter:   {path_filter}")

    if args.split:
        print(f"[scrape] split mode → {args.out_dir}/codegen-{{domain}}.jsonl")
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        os.makedirs(Path(args.out).parent, exist_ok=True)

    total_files   = 0
    total_entries = 0
    t0 = time.time()

    # In split mode we keep one open file handle per domain
    split_fhs: dict[str, IO[str]] = {}

    def get_split_fh(domain: str) -> IO[str]:
        if domain not in split_fhs:
            p = os.path.join(args.out_dir, f"codegen-{domain}.jsonl")
            split_fhs[domain] = open(p, "w", encoding="utf-8")
        return split_fhs[domain]

    devnull_fh = open(os.devnull, "w") if args.dry_run else None
    single_fh  = (open(args.out, "w", encoding="utf-8")
                  if (not args.split and not args.dry_run) else None)

    try:
        for drive in args.drives:
            root = f"{drive}:/"
            if not os.path.exists(root):
                print(f"[scrape] drive {root} not found, skipping")
                continue

            print(f"\n[scrape] scanning {root} for {', '.join(sorted(EXTENSIONS))}")
            drive_files   = 0
            drive_entries = 0

            for fpath in walk_drive(root):
                if args.limit and drive_files >= args.limit:
                    break

                ext_domain = fpath.suffix.lower().lstrip(".")
                if domain_filter and ext_domain not in domain_filter:
                    continue
                if path_filter:
                    fpath_str = str(fpath.resolve())
                    if not any(fpath_str.startswith(pf) for pf in path_filter):
                        continue

                drive_files += 1

                if drive_files % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"  {root}  files={drive_files}  entries={drive_entries}  {elapsed:.0f}s",
                          end="\r", flush=True)

                if not args.dry_run:
                    if args.split:
                        fh = get_split_fh(ext_domain)
                    else:
                        fh = single_fh  # type: ignore[assignment]
                    n = process_file(fpath, fh)
                    drive_entries += n

            print(f"\n  {root}  files={drive_files}  entries={drive_entries}  done")
            total_files   += drive_files
            total_entries += drive_entries

    finally:
        for fh in split_fhs.values():
            fh.close()
        if single_fh:
            single_fh.close()
        if devnull_fh:
            devnull_fh.close()

    elapsed = time.time() - t0
    if args.split:
        domains_written = sorted(split_fhs.keys())
        print(f"\n[scrape] total files={total_files}  entries={total_entries}  {elapsed:.1f}s")
        print(f"         domains written: {domains_written}")
        for d in domains_written:
            p = os.path.join(args.out_dir, f"codegen-{d}.jsonl")
            mb = Path(p).stat().st_size / 1024 / 1024
            expert = DOMAIN_EXPERT.get(d, "CODEGEN-µ")
            print(f"         {p}  ({mb:.1f} MB)  [{expert}]")
    else:
        print(f"\n[scrape] total files={total_files}  entries={total_entries}  {elapsed:.1f}s -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
