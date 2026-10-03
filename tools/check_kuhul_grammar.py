#!/usr/bin/env python3
"""Validate the K'UHUL v8 standalone/runtime grammar surface."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "kuhul-grammar.ebnf"
RUNTIME = ROOT / "data/grammar/kuhul.runtime-grammar.ebnf"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SEMANTIC_FOLDS = ROOT / "docs/SEMANTIC_FOLDS.md"

STANDALONE_TOKENS = [
    "K'UHUL v8.0.0 Standalone EBNF Grammar",
    "Hyperbolic Event Coordinates",
    "Einstein Contractions",
    "Holographic Mesh Boundaries",
    "CDATA GPU Kernel Dispatches",
    "program",
    "cso_decl",
    "cdata_block",
    "hlsl_kernel_source",
    "event_decl",
    "minkowski_four_vector",
    "interval_signature",
    "mesh_inequality",
    "raychaudhuri_metric",
    "contractionExpr",
    "⊓",
    "⊗",
    "cso_sha256_id",
    "cso://",
    "phase_expression",
    "Yax",
    "Ch'en",
    "Fold namespace boundary",
    "KLSL/KXC/KHLC folds",
]

RUNTIME_TOKENS = [
    "K'UHULSemanticFold != KLSLFold",
    "K'UHULSemanticFold != KXCFold",
    "K'UHULSemanticFold != KHLCFold",
    "event_definition",
    "event_decl",
    "cso_definition",
    "cso_decl",
    "cdata_block",
    "hlsl_kernel_source",
    "minkowski_four_vector",
    "interval_signature",
    "mesh_inequality",
    "holographic_boundary",
    "raychaudhuri_metric",
    "einstein_contraction",
    "⊓",
    "semantic authority",
    "projection and",
]

FOLD_NAMESPACE_TOKENS = [
    "K'UHUL semantic fold",
    "KLSL fold",
    "KXC fold",
    "KHLC fold",
    "N(k) --[α, θ]--> B --Xul--> k'",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def main() -> int:
    standalone = read(STANDALONE) if STANDALONE.exists() else ""
    runtime = read(RUNTIME) if RUNTIME.exists() else ""
    glossary = read(GLOSSARY) if GLOSSARY.exists() else ""
    semantic_folds = read(SEMANTIC_FOLDS) if SEMANTIC_FOLDS.exists() else ""
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    check("standalone_file_present", STANDALONE.exists())
    check("runtime_file_present", RUNTIME.exists())
    check("glossary_file_present", GLOSSARY.exists())
    check("semantic_folds_file_present", SEMANTIC_FOLDS.exists())
    for token in STANDALONE_TOKENS:
        check(f"standalone/has/{token}", token in standalone)
    for token in RUNTIME_TOKENS:
        check(f"runtime/has/{token}", token in runtime)
    for token in FOLD_NAMESPACE_TOKENS:
        check(f"glossary/fold_namespace/{token}", token in glossary)
        check(f"semantic_folds/fold_namespace/{token}", token in semantic_folds)

    check(
        "standalone/program_admits_cso",
        "program         = { statement | bundle_decl | manifold_decl | universe_decl | cso_decl } ;"
        in standalone,
    )
    check(
        "standalone/statement_admits_event",
        "phaseDecl | bundle_decl | manifold_decl | cso_decl | event_decl"
        in standalone,
    )
    check(
        "runtime/program_admits_event_and_cso",
        "event_definition*" in runtime and "cso_definition*" in runtime,
    )
    check(
        "runtime/phase_statement_admits_event_and_cso",
        "| event_definition" in runtime and "| cso_definition" in runtime,
    )
    check(
        "runtime/expression_admits_einstein_contraction",
        "| einstein_contraction" in runtime and 'einstein_contraction ::= expression "⊓" expression'
        in runtime,
    )
    check(
        "cso_uses_cdata_envelope",
        '"<![CDATA["' in standalone and '"<![CDATA[" hlsl_kernel_source "]]>"' in runtime,
    )
    check(
        "events_carry_interval_classes",
        all(label in standalone and label in runtime for label in ("timelike", "spacelike", "lightlike")),
    )
    check(
        "backend_folds_are_not_semantic_authority",
        "do not define it" in glossary and "does not become the semantic fold itself" in semantic_folds,
    )

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.kuhul_grammar.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "standalone": str(STANDALONE.relative_to(ROOT)),
        "runtime": str(RUNTIME.relative_to(ROOT)),
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
