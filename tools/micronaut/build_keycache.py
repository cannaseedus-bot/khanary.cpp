#!/usr/bin/env python3
"""
Build KHANARY key-store cache files without launching the full runtime.

Outputs:
  - data/factory_keys.bin
  - data/factory_keys.cachemeta.json
  - data/factory_keys.json (optional compatibility snapshot)
"""

from __future__ import annotations

import argparse
import html
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


MAGIC = b"KFACBIN1"
BIN_VERSION = 1
DEFAULT_TFIDF_ACCEPT = 0.42


def lower_ascii(text: str) -> str:
    return text.lower()


def normalize_hash_token(text: str) -> str:
    token = lower_ascii((text or "").strip())
    if token.startswith("sha256:"):
        token = token[7:]
    if token.startswith("hash:"):
        token = token[5:]
    if token.startswith("0x"):
        token = token[2:]
    if not token:
        return ""
    if any(ch not in "0123456789abcdef" for ch in token):
        return ""
    return token


def fnv1a64_hex(text: str) -> str:
    h = 14695981039346656037
    for b in text.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def derive_hash_token(lower_key: str, hash_hint: str) -> str:
    hint = normalize_hash_token(hash_hint)
    return hint if hint else fnv1a64_hex(lower_key)


@dataclass
class CacheFingerprint:
    latest_write_s: int = 0
    total_bytes: int = 0
    file_count: int = 0


def fingerprint_file(path: Path, fp: CacheFingerprint) -> None:
    if not path.exists() or not path.is_file():
        return
    st = path.stat()
    fp.latest_write_s = max(fp.latest_write_s, int(st.st_mtime))
    fp.total_bytes += int(st.st_size)
    fp.file_count += 1


def fingerprint_tree(root: Path, fp: CacheFingerprint) -> None:
    if not root.exists() or not root.is_dir():
        return
    for p in root.rglob("*"):
        if p.is_file():
            fingerprint_file(p, fp)


def compute_fingerprint(data_dir: Path, repo_root: Path) -> CacheFingerprint:
    fp = CacheFingerprint()

    if data_dir.exists():
        for p in data_dir.glob("*.jsonl"):
            fingerprint_file(p, fp)

    fingerprint_file(data_dir / "gram" / "english-dictionary.jsonl", fp)
    fingerprint_tree(repo_root / "docs", fp)
    fingerprint_tree(data_dir / "schema", fp)
    fingerprint_tree(repo_root / "models", fp)
    return fp


class KeyStoreBuilder:
    def __init__(self, tfidf_accept: float = DEFAULT_TFIDF_ACCEPT) -> None:
        self.tfidf_accept = float(tfidf_accept)
        self._entries_by_lower: Dict[str, Dict[str, object]] = {}

    def record_key(self, key: str, response: str, hash_hint: str = "") -> None:
        if not key or not response:
            return
        kl = lower_ascii(key)
        self._entries_by_lower[kl] = {
            "key": key,
            "response": response,
            "hash": derive_hash_token(kl, hash_hint),
            "hits": 0,
            "confidence": 0.5,
        }

    def load_from_json_state(self, json_path: Path) -> bool:
        if not json_path.exists():
            return False
        try:
            root = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        entries = root.get("entries")
        if not isinstance(entries, list):
            return False

        tfidf = root.get("tfidf_accept")
        if isinstance(tfidf, (int, float)):
            self.tfidf_accept = float(tfidf)

        loaded = 0
        for item in entries:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            response = item.get("response")
            if not isinstance(key, str) or not isinstance(response, str):
                continue
            kl = lower_ascii(key)
            hash_token = derive_hash_token(kl, str(item.get("hash", "")))
            hits_raw = item.get("hits", 0)
            conf_raw = item.get("confidence", 0.5)
            try:
                hits = max(0, int(hits_raw))
            except Exception:
                hits = 0
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.5
            if conf < 0.0:
                conf = 0.0
            elif conf > 1.0:
                conf = 1.0
            self._entries_by_lower[kl] = {
                "key": key,
                "response": response,
                "hash": hash_token,
                "hits": hits,
                "confidence": conf,
            }
            loaded += 1
        return loaded > 0

    def entries_sorted(self) -> list[Dict[str, object]]:
        return [self._entries_by_lower[k] for k in sorted(self._entries_by_lower.keys())]

    def save_binary(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.entries_sorted()
        with out_path.open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", BIN_VERSION))
            f.write(struct.pack("<f", float(self.tfidf_accept)))
            f.write(struct.pack("<Q", len(entries)))
            for e in entries:
                self._write_blob(f, str(e["key"]))
                self._write_blob(f, str(e["response"]))
                self._write_blob(f, str(e["hash"]))
                f.write(struct.pack("<Q", int(e["hits"])))
                f.write(struct.pack("<f", float(e["confidence"])))

    def save_json(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for e in self.entries_sorted():
            entries.append(
                {
                    "key": e["key"],
                    "response": e["response"],
                    "hash": e["hash"],
                    "hits": int(e["hits"]),
                    "confidence": float(e["confidence"]),
                }
            )
        root = {
            "version": 1.0,
            "tfidf_accept": float(self.tfidf_accept),
            "entries": entries,
        }
        out_path.write_text(json.dumps(root, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _write_blob(stream, value: str) -> None:
        payload = value.encode("utf-8")
        stream.write(struct.pack("<I", len(payload)))
        stream.write(payload)


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def seed_default_keys(builder: KeyStoreBuilder) -> None:
    builder.record_key("Mx2LM:", "<mx2lm>{ngrams}</mx2lm>")
    builder.record_key("@mx2lm", "<mx2lm>{ngrams}</mx2lm>")
    builder.record_key("APP:", "<app>{intent}</app>")
    builder.record_key("@app", "<app>{intent}</app>")
    builder.record_key("memory:", "<memory> @memory")
    builder.record_key("@memory", "<memory> @memory")


def load_dictionary(builder: KeyStoreBuilder, data_dir: Path) -> int:
    path = data_dir / "gram" / "english-dictionary.jsonl"
    loaded = 0
    for obj in iter_jsonl(path):
        word = obj.get("word")
        definition = obj.get("definition")
        if isinstance(word, str) and isinstance(definition, str) and word and definition:
            hash_hint = ""
            for hk in ("hash", "hash_hex", "hash64"):
                hv = obj.get(hk)
                if isinstance(hv, str) and hv:
                    hash_hint = hv
                    break
            builder.record_key(word, definition, hash_hint)
        loaded += 1
    return loaded


def load_ag_news(builder: KeyStoreBuilder, data_dir: Path) -> int:
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    loaded = 0
    for obj in iter_jsonl(data_dir / "ag_news_train.jsonl"):
        text = obj.get("text")
        if not isinstance(text, str) or not text:
            continue
        label = obj.get("label")
        try:
            li = int(label)
        except Exception:
            li = -1
        cat = labels[li] if 0 <= li < 4 else "News"
        key = text.split("\n", 1)[0][:80]
        val = f"[{cat}] {text}"
        builder.record_key(key, val)
        loaded += 1
    return loaded


def load_squad(builder: KeyStoreBuilder, data_dir: Path) -> int:
    loaded = 0
    for obj in iter_jsonl(data_dir / "squad_train.jsonl"):
        question = obj.get("question")
        context = obj.get("context")
        if (
            isinstance(question, str)
            and isinstance(context, str)
            and len(question) >= 8
            and context
        ):
            builder.record_key(question, context)
            loaded += 1
    return loaded


def load_wiki2(builder: KeyStoreBuilder, data_dir: Path) -> int:
    loaded = 0
    for obj in iter_jsonl(data_dir / "wikitext2_train.jsonl"):
        text = obj.get("text")
        if not isinstance(text, str) or len(text) < 40:
            continue
        key = text[:60]
        if len(key) < 8:
            continue
        builder.record_key(key, text)
        loaded += 1
    return loaded


def load_theory(builder: KeyStoreBuilder, data_dir: Path) -> int:
    skip = {
        "ag_news_train",
        "squad_train",
        "wikitext2_train",
        "english-dictionary",
        "english-triples",
    }
    loaded = 0
    for file_path in data_dir.glob("*.jsonl"):
        if file_path.stem in skip:
            continue
        for obj in iter_jsonl(file_path):
            task = obj.get("task")
            text = obj.get("text")
            if task != "knowledge" or not isinstance(text, str) or len(text) < 20:
                continue

            pkey = text[:60]
            if len(pkey) >= 8:
                builder.record_key(pkey, text)
                loaded += 1

            pid_raw = obj.get("paragraph_id", 0)
            try:
                pid = int(pid_raw)
            except Exception:
                pid = 0
            if pid != 1:
                continue

            title = obj.get("title")
            if isinstance(title, str) and title:
                builder.record_key(title, text)

            title_key = obj.get("title_key")
            if isinstance(title_key, str) and title_key:
                builder.record_key(title_key, text)
                tk_spaces = title_key.replace("_", " ")
                if tk_spaces != title_key:
                    builder.record_key(tk_spaces, text)
    return loaded


def load_docs(builder: KeyStoreBuilder, repo_root: Path) -> int:
    docs_dir = repo_root / "docs"
    if not docs_dir.exists():
        return 0
    sections = 0
    for entry in docs_dir.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext == ".md":
            sections += _load_markdown_sections(builder, entry)
        elif ext == ".html":
            sections += _load_html_sections(builder, entry)
    return sections


def _load_markdown_sections(builder: KeyStoreBuilder, path: Path) -> int:
    count = 0
    current_header = ""
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal count, current_header, body_lines
        body = "\n".join(body_lines).strip()
        if len(current_header) >= 4 and body:
            builder.record_key(current_header, body)
            count += 1

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("### "):
                flush()
                current_header = line[4:].rstrip(" \r\n")
                body_lines = []
            elif line.startswith("## "):
                flush()
                current_header = line[3:].rstrip(" \r\n")
                body_lines = []
            elif current_header:
                body_lines.append(line.rstrip("\n"))
    flush()
    return count


def _remove_html_blocks(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}\b.*?</{tag}>", re.IGNORECASE | re.DOTALL)
    return pattern.sub("", text)


def _strip_tags(text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", no_tags).strip()


def _load_html_sections(builder: KeyStoreBuilder, path: Path) -> int:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _remove_html_blocks(raw, "style")
    text = _remove_html_blocks(text, "script")
    text = _remove_html_blocks(text, "nav")

    heading_matches = list(
        re.finditer(r"<h([23])\b[^>]*>(.*?)</h\1>", text, re.IGNORECASE | re.DOTALL)
    )
    if not heading_matches:
        return 0

    sections = 0
    for idx, m in enumerate(heading_matches):
        header = html.unescape(_strip_tags(m.group(2)))
        body_start = m.end()
        body_end = heading_matches[idx + 1].end() if idx + 1 < len(heading_matches) else len(text)
        if body_end <= body_start:
            continue
        body = html.unescape(_strip_tags(text[body_start:body_end]))
        if len(header) >= 4 and len(body) >= 8:
            builder.record_key(header, body)
            sections += 1
    return sections


def load_schema_and_models(builder: KeyStoreBuilder, data_dir: Path, repo_root: Path) -> int:
    schema_loaded = 0
    sg_path = data_dir / "schema" / "internal-schema-graph.json"
    if sg_path.exists():
        try:
            sg = json.loads(sg_path.read_text(encoding="utf-8"))
        except Exception:
            sg = {}
        mods = sg.get("modules", [])
        if isinstance(mods, list):
            for mod in mods:
                if not isinstance(mod, dict):
                    continue
                mname = mod.get("name")
                schema_id = mod.get("schema_id")
                schema_file = mod.get("schema_file")
                if not isinstance(schema_file, str) or not schema_file:
                    continue
                schema_path = repo_root / schema_file
                if not schema_path.exists():
                    continue
                try:
                    schema = json.loads(schema_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                title = schema.get("title")
                desc = schema.get("description")
                val = ""
                if isinstance(title, str) and title:
                    val = title
                if isinstance(desc, str) and desc:
                    val = f"{val}: {desc}" if val else desc
                if not val and isinstance(mname, str):
                    val = mname
                if not val:
                    continue

                if isinstance(mname, str) and mname:
                    builder.record_key(mname, val)
                if isinstance(schema_id, str) and schema_id:
                    builder.record_key(schema_id, val)
                if isinstance(title, str) and title and (not isinstance(mname, str) or title != mname):
                    builder.record_key(title, val)
                schema_loaded += 1

                defs = schema.get("$defs")
                if isinstance(defs, dict):
                    for def_name, def_obj in defs.items():
                        if not isinstance(def_name, str) or not isinstance(def_obj, dict):
                            continue
                        def_val = ""
                        def_desc = def_obj.get("description")
                        if isinstance(def_desc, str) and def_desc:
                            def_val = def_desc

                        enum_v = def_obj.get("enum")
                        if isinstance(enum_v, list):
                            values = [str(v) for v in enum_v if isinstance(v, str)]
                            if values:
                                ev = f"{def_name} valid values: {', '.join(values)}"
                                def_val = f"{def_val} — {ev}" if def_val else ev

                        props_v = def_obj.get("properties")
                        if isinstance(props_v, dict) and props_v:
                            pv = f"{def_name} properties: {', '.join(props_v.keys())}"
                            def_val = f"{def_val} — {pv}" if def_val else pv

                        if def_val:
                            builder.record_key(def_name, def_val)
                            schema_loaded += 1

    models_dir = repo_root / "models"
    if models_dir.exists():
        for entry in models_dir.iterdir():
            if not entry.is_dir():
                continue
            model_path = entry / "MODEL.json"
            if not model_path.exists():
                continue
            try:
                mj = json.loads(model_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            mname = mj.get("name")
            if not isinstance(mname, str) or not mname:
                continue
            version = mj.get("version")
            profile = mj.get("knu_profile")

            mval = mname
            if isinstance(version, str) and version:
                mval += f" v{version}"
            if isinstance(profile, str) and profile:
                mval += f": {profile}"

            ops_v = mj.get("ops")
            if isinstance(ops_v, dict) and ops_v:
                mval += " | ops: " + ", ".join(ops_v.keys())
            be_v = mj.get("backends")
            if isinstance(be_v, dict) and be_v:
                mval += " | backends: " + ", ".join(be_v.keys())

            builder.record_key(mname, mval)
            builder.record_key(f"{mname} model", mval)
            if isinstance(version, str) and version:
                builder.record_key(f"{mname} v{version}", mval)

            mname_sp = mname.replace("-", " ")
            if mname_sp != mname:
                builder.record_key(mname_sp, mval)
                builder.record_key(f"{mname_sp} model", mval)
            schema_loaded += 1

            data_obj = mj.get("data")
            if isinstance(data_obj, dict):
                sid = data_obj.get("schema_id")
                stid = data_obj.get("schema_types_id")
                if isinstance(sid, str) and sid:
                    builder.record_key(sid, mval)
                if isinstance(stid, str) and stid:
                    builder.record_key(stid, mval)
    return schema_loaded


def save_cache_meta(path: Path, fp: CacheFingerprint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = {
        "version": 1,
        "latest_write_s": int(fp.latest_write_s),
        "total_bytes": int(fp.total_bytes),
        "file_count": int(fp.file_count),
    }
    path.write_text(json.dumps(root, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build KHANARY binary key cache from source datasets.")
    parser.add_argument("--data-dir", default="data", help="Path to data directory.")
    parser.add_argument("--repo-root", default=".", help="Repository root path.")
    parser.add_argument("--out-bin", default="", help="Output binary cache path.")
    parser.add_argument("--out-json", default="", help="Output JSON state path.")
    parser.add_argument("--out-meta", default="", help="Output cache metadata path.")
    parser.add_argument("--skip-json", action="store_true", help="Do not write factory_keys.json.")
    parser.add_argument(
        "--use-existing-json",
        action="store_true",
        help="If factory_keys.json exists, reuse it to pack binary instead of rebuilding sources.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_bin = Path(args.out_bin).resolve() if args.out_bin else (data_dir / "factory_keys.bin")
    out_json = Path(args.out_json).resolve() if args.out_json else (data_dir / "factory_keys.json")
    out_meta = Path(args.out_meta).resolve() if args.out_meta else (data_dir / "factory_keys.cachemeta.json")

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    builder = KeyStoreBuilder()
    reused_json = False
    if args.use_existing_json and out_json.exists():
        reused_json = builder.load_from_json_state(out_json)

    if not reused_json:
        seed_default_keys(builder)
        dict_loaded = load_dictionary(builder, data_dir)
        ag_loaded = load_ag_news(builder, data_dir)
        squad_loaded = load_squad(builder, data_dir)
        wiki2_loaded = load_wiki2(builder, data_dir)
        theory_loaded = load_theory(builder, data_dir)
        docs_loaded = load_docs(builder, repo_root)
        schema_loaded = load_schema_and_models(builder, data_dir, repo_root)

        print(f"Dict: {dict_loaded} rows parsed")
        print(f"AGNews: {ag_loaded} rows loaded")
        print(f"SQuAD: {squad_loaded} rows loaded")
        print(f"Wiki2: {wiki2_loaded} rows loaded")
        print(f"Theory: {theory_loaded} rows loaded")
        print(f"Docs: {docs_loaded} sections loaded")
        print(f"Schemas: {schema_loaded} schema/model entries loaded")
    else:
        print(f"Reused JSON state from: {out_json}")

    fp = compute_fingerprint(data_dir, repo_root)
    builder.save_binary(out_bin)
    save_cache_meta(out_meta, fp)
    if not args.skip_json:
        builder.save_json(out_json)

    print(f"Binary cache: {out_bin}")
    print(f"Cache meta:   {out_meta}")
    if not args.skip_json:
        print(f"JSON state:   {out_json}")
    print(f"Entries:      {len(builder.entries_sorted())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
