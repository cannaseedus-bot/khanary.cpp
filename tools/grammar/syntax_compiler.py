"""SyntaxCompiler — CSS/HTML and graph surfaces → ⟁KEY token encoder.

Implements the SyntaxCompiler.cs pipeline from css.manifest.json in Python:
  SyntaxParser → KeyResolver → BookExpander → RelationBinder
  → SyntaxGraphBuilder → FoldBuilder → SyntaxProjector

Primary use: given a CSS property block or HTML element binding, find the set
of ⟁KEY tokens from the CSS manifest that best decompose the input.

CLI examples:
    python tools/grammar/syntax_compiler.py "background:#020b12"
    python tools/grammar/syntax_compiler.py --key PANEL
    python tools/grammar/syntax_compiler.py --relational-graph data/grammar/kuhul-relational-graph.xml
    python tools/grammar/syntax_compiler.py --selftest
"""
import argparse
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifest" / "css.manifest.json"
SCX_MAX_DEPTH = 64
RELATIONAL_NS = "https://kuhul.io/relational"

_META = frozenset({
    "⟁KEYGRAMMAR", "⟁GRAMMARSYNTAXGRAPH", "⟁AST",
    "⟁SYNTAXCOMPILER", "⟁SYNTAXUNITKINDS",
    "⟁HTMLBIND", "⟁HTMLARTICLE", "⟁HTMLHEADING1",
    "⟁HTMLPARAGRAPH", "⟁HTMLCLASSATTR",
})


# ── BookExpander / SCX ────────────────────────────────────────────────────────

def _expand(manifest, word, visited, depth):
    """Recursively expand a manifest key word to its leaf CSS declarations."""
    if depth > SCX_MAX_DEPTH or word in visited:
        return []
    full_key = f"⟁{word}"
    if full_key not in manifest:
        return []
    visited = visited | {word}
    result = []
    for item in manifest[full_key]:
        if item.startswith("⟁"):
            result.extend(_expand(manifest, item[1:], visited, depth + 1))
        elif (":" in item and not item.startswith("kind:")
              and not item.startswith("language:")
              and not item.startswith("role:")):
            result.append(item)
    return result


def build_key_index(manifest):
    """Pre-expand every CSS key to its full leaf CSS declaration set.

    Returns {word: {"leaf_css": [...], "direct_css": [...], "key_refs": [...]}}.
    """
    index = {}
    for mkey, vals in manifest.items():
        if not mkey.startswith("⟁") or mkey in _META or mkey.startswith("⟁TYPE:"):
            continue
        word = mkey[1:]
        leaf = list(dict.fromkeys(_expand(manifest, word, frozenset(), 0)))
        direct_css = [v for v in vals if (":" in v and not v.startswith("⟁")
                      and not v.startswith("kind:") and not v.startswith("language:"))]
        key_refs = [v[1:] for v in vals if v.startswith("⟁")]
        index[word] = {"leaf_css": leaf, "direct_css": direct_css, "key_refs": key_refs}
    return index


# ── SyntaxParser ──────────────────────────────────────────────────────────────

def parse_css(css_input):
    """Parse a CSS declaration block string into {property: value} dict.

    Accepts: inline style string, braced block, or dict (passthrough).
    Splits on ';' but not inside parentheses (handles rgba(...) values).
    """
    if isinstance(css_input, dict):
        return {k.strip(): v.strip() for k, v in css_input.items()}
    text = str(css_input).strip().lstrip("{").rstrip("}")
    result = {}
    for decl in re.split(r";(?![^()]*\))", text):
        decl = decl.strip()
        if not decl or ":" not in decl:
            continue
        colon = decl.index(":")
        prop, val = decl[:colon].strip(), decl[colon + 1:].strip()
        if prop:
            result[prop] = val
    return result


# ── KeyResolver — greedy decomposition ───────────────────────────────────────

def _to_pair_set(declarations):
    """Normalise 'property:value' strings to a frozenset of (prop, val) tuples."""
    out = set()
    for item in declarations:
        if ":" in item:
            prop, _, val = item.partition(":")
            out.add((prop.strip(), val.strip()))
    return out


def resolve_keys(css_input, manifest, key_index):
    """Greedy decomposition of css_input into the best-fit ⟁KEY token sequence.

    Tie-breaking: prefer keys with fewer total leaf properties (more specific
    match), so PANEL wins over CARD for the same 4-property hit.

    Returns:
      matches   — [{"word", "key", "coverage", "covered"}]
      remaining — set of (property, value) tuples not covered by any key
    """
    parsed = parse_css(css_input) if not isinstance(css_input, set) else css_input
    remaining = {(k.strip(), v.strip()) for k, v in parsed.items()}
    matches = []

    while remaining:
        best_word = None
        best_hit: frozenset = frozenset()
        best_leaf_size = float("inf")
        for word, info in key_index.items():
            leaf_set = _to_pair_set(info["leaf_css"])
            hit = remaining & leaf_set
            if not hit:
                continue
            if len(hit) > len(best_hit) or (
                len(hit) == len(best_hit) and len(leaf_set) < best_leaf_size
            ):
                best_hit = frozenset(hit)
                best_word = word
                best_leaf_size = len(leaf_set)
        if not best_word:
            break
        remaining -= best_hit
        matches.append({
            "word": best_word,
            "key": f"CSS:{best_word.upper()}",
            "coverage": len(best_hit),
            "covered": [f"{p}:{v}" for p, v in sorted(best_hit)],
        })

    return matches, remaining


# ── RelationBinder ────────────────────────────────────────────────────────────

def bind_html_element(element_name, manifest, key_index):
    """Map an HTML element name to its ⟁KEY bindings via ⟁HTMLBIND entries."""
    html_key = f"⟁HTML{element_name.upper()}"
    if html_key not in manifest:
        return []
    refs = [v[1:] for v in manifest[html_key]
            if v.startswith("⟁") and v != "⟁HTMLBIND"]
    return [{"word": ref, "key": f"CSS:{ref.upper()}", "via_html": element_name}
            for ref in refs if ref in key_index]


# ── SyntaxProjector ───────────────────────────────────────────────────────────

def project(matches, remaining):
    """Format the compiler result as a structured dict."""
    return {
        "keys": [{"key": m["key"], "word": m["word"], "coverage": m["coverage"]}
                 for m in matches],
        "unmatched_properties": [f"{p}:{v}" for p, v in sorted(remaining)],
        "total_coverage": sum(m["coverage"] for m in matches),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def compile_css(css_input, manifest_path=None):
    """Full SyntaxCompiler pipeline: CSS string/dict → key token list.

    Returns the SyntaxProjector output dict.
    """
    mp = Path(manifest_path) if manifest_path else MANIFEST_PATH
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    key_index = build_key_index(manifest)
    matches, remaining = resolve_keys(css_input, manifest, key_index)
    return project(matches, remaining)


def expand_key(word, manifest_path=None):
    """Expand a single ⟁KEY word to its full leaf CSS declaration list."""
    mp = Path(manifest_path) if manifest_path else MANIFEST_PATH
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    return list(dict.fromkeys(_expand(manifest, word, frozenset(), 0)))


# ── XSLT/SDM relational graph lowering ───────────────────────────────────────

def _xml_local_name(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children_named(element, name):
    return [child for child in list(element) if _xml_local_name(child.tag) == name]


def _coerce_xml_text(xml_input):
    if hasattr(xml_input, "read"):
        return xml_input.read()
    if isinstance(xml_input, Path):
        return xml_input.read_text(encoding="utf-8")
    text = str(xml_input)
    maybe_path = Path(text)
    if "<" not in text[:16] and maybe_path.exists():
        return maybe_path.read_text(encoding="utf-8")
    return text


def _edge_directed(edge, graph_default):
    directed_attr = edge.attrib.get("directed")
    if directed_attr is not None:
        return directed_attr.lower() != "false"
    return str(graph_default).lower() != "undirected"


def _edge_verb(edge, directed):
    explicit = edge.attrib.get("verb")
    if explicit:
        return explicit
    return "@relate" if not directed else "@call"


def _find_graph_root(root):
    if _xml_local_name(root.tag) == "graph":
        return root
    for child in root.iter():
        if _xml_local_name(child.tag) == "graph":
            return child
    raise ValueError("relational graph XML does not contain a <graph> root")


def compile_relational_graph(xml_input):
    """Lower a GraphML-style XML graph to a K'UHUL relational micronaut model.

    This is the dependency-free mirror of the checked-in XSLT contract:
    graph → relational network, node → micronaut instance, edge → relational
    verb, nested graph → sub-micronaut cluster. The result is a projection
    surface for execution planning and rendering; it does not assert semantic
    truth or promote a micronaut entry by itself.
    """
    root = ET.fromstring(_coerce_xml_text(xml_input))
    graph_root = _find_graph_root(root)
    nodes = []
    edges = []
    clusters = []
    node_occurrences = {}

    def walk_graph(graph_el, parent=None, depth=0):
        graph_default = graph_el.attrib.get("edgedefault", "directed")

        for node in _children_named(graph_el, "node"):
            node_id = node.attrib.get("id")
            if not node_id:
                raise ValueError("relational <node> missing required id")
            nested_graphs = _children_named(node, "graph")
            occurrence_index = node_occurrences.get(node_id, 0)
            node_occurrences[node_id] = occurrence_index + 1
            occurrence = {
                "id": node_id,
                "occurrence_id": f"{parent or 'root'}::{node_id}::{occurrence_index}",
                "type": "micronaut",
                "category": node.attrib.get("category", "uncategorized"),
                "parent": parent,
                "depth": depth,
                "cluster": bool(nested_graphs),
            }
            nodes.append(occurrence)
            if nested_graphs:
                clusters.append({
                    "id": node_id,
                    "parent": parent,
                    "depth": depth,
                    "kind": "sub-micronaut-cluster",
                })
                for nested in nested_graphs:
                    walk_graph(nested, parent=node_id, depth=depth + 1)

        for edge in _children_named(graph_el, "edge"):
            source = edge.attrib.get("source")
            target = edge.attrib.get("target")
            if not source or not target:
                raise ValueError("relational <edge> missing source or target")
            directed = _edge_directed(edge, graph_default)
            edges.append({
                "from": source,
                "to": target,
                "verb": _edge_verb(edge, directed),
                "directed": directed,
                "parent": parent,
                "relation_type": "directed" if directed else "undirected",
                "isLink": True,
            })

    walk_graph(graph_root)
    return {
        "@type": "kuhul.relational_sdm.v1",
        "namespace": RELATIONAL_NS,
        "layout": "Tree",
        "source_root": "graph",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "cluster_count": len(clusters),
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
        "authority_boundary": (
            "relational graph rendering is an execution-plan projection; "
            "Ch'en/Xul remain responsible for verification and persistence"
        ),
    }


def _xml_attr(name, value):
    return f'{name}="{html.escape(str(value), quote=True)}"'


def render_relational_sdm_xml(model):
    """Render a compiled relational graph as SDM XML."""
    nodes_by_parent = {}
    edges_by_parent = {}
    for node in model["nodes"]:
        nodes_by_parent.setdefault(node.get("parent"), []).append(node)
    for edge in model["edges"]:
        edges_by_parent.setdefault(edge.get("parent"), []).append(edge)

    lines = [f'<sdm xmlns:k="{RELATIONAL_NS}">']

    def render_graph(parent, indent):
        pad = " " * indent
        for node in nodes_by_parent.get(parent, []):
            attrs = [
                _xml_attr("id", node["id"]),
                _xml_attr("type", "micronaut"),
                _xml_attr("category", node.get("category", "uncategorized")),
            ]
            has_nested = node["id"] in nodes_by_parent or node["id"] in edges_by_parent
            if has_nested:
                lines.append(f"{pad}<node {' '.join(attrs)}>")
                lines.append(f"{pad}  <graph>")
                render_graph(node["id"], indent + 4)
                lines.append(f"{pad}  </graph>")
                lines.append(f"{pad}</node>")
            else:
                lines.append(f"{pad}<node {' '.join(attrs)}/>")

        for edge in edges_by_parent.get(parent, []):
            attrs = [
                _xml_attr("from", edge["from"]),
                _xml_attr("to", edge["to"]),
                _xml_attr("verb", edge["verb"]),
                _xml_attr("isLink", "true"),
            ]
            if not edge.get("directed", True):
                attrs.append(_xml_attr("directed", "false"))
            lines.append(f"{pad}<edge {' '.join(attrs)}/>")

    render_graph(None, 2)
    lines.append("</sdm>")
    return "\n".join(lines) + "\n"


# ── Self-test ─────────────────────────────────────────────────────────────────

_SELFTEST_CASES = [
    # leaf-only input → single key
    ("background:#020b12", ["BGpanel"]),
    # PANEL's full expansion (4 leaf properties) → PANEL (not CARD which adds box-shadow)
    ("background:#020b12; border-radius:1.25rem; padding:20px;"
     " border:1px solid rgba(148,163,184,0.5)", ["PANEL"]),
    # single layout primitive
    ("display:flex", ["FLEX"]),
]


def selftest(manifest):
    key_index = build_key_index(manifest)
    passed = 0
    for css_str, expected_words in _SELFTEST_CASES:
        matches, _ = resolve_keys(css_str, manifest, key_index)
        got = [m["word"] for m in matches]
        ok = got == expected_words
        print(f"  {'PASS' if ok else 'FAIL'} | {css_str[:60]!r} -> {got}")
        if ok:
            passed += 1
    total = len(_SELFTEST_CASES)
    print(f"{'PASS' if passed == total else 'FAIL'} | selftest: {passed}/{total}")
    return passed == total


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("css", nargs="?",
                        help="CSS declaration block string to encode")
    parser.add_argument("--input-file", metavar="FILE",
                        help="read CSS from a file instead")
    parser.add_argument("--key", metavar="WORD",
                        help="expand a single ⟁KEY to its leaf CSS")
    parser.add_argument("--selftest", action="store_true",
                        help="run built-in test cases")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), metavar="FILE")
    parser.add_argument("--relational-graph", metavar="FILE",
                        help="compile GraphML-style XML into a K'UHUL relational model")
    parser.add_argument("--sdm-xml", action="store_true",
                        help="with --relational-graph, emit SDM XML instead of JSON")
    args = parser.parse_args()

    if args.relational_graph:
        model = compile_relational_graph(Path(args.relational_graph))
        if args.sdm_xml:
            print(render_relational_sdm_xml(model), end="")
        else:
            print(json.dumps(model, ensure_ascii=False, indent=2))
        return 0

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    if args.selftest:
        return 0 if selftest(manifest) else 1

    if args.key:
        leaf = list(dict.fromkeys(_expand(manifest, args.key, frozenset(), 0)))
        print(json.dumps({"key": args.key, "leaf_css": leaf}, ensure_ascii=False, indent=2))
        return 0

    if args.input_file:
        css_str = Path(args.input_file).read_text(encoding="utf-8")
    elif args.css:
        css_str = args.css
    else:
        parser.print_help()
        return 1

    key_index = build_key_index(manifest)
    matches, remaining = resolve_keys(css_str, manifest, key_index)
    result = project(matches, remaining)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
