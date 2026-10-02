#!/usr/bin/env python3
"""Validate the Micronaut runtime grammar/profile/JROM projection contract."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "data/grammar/micronaut.runtime-grammar.ebnf"
TRACK_SCHEMA = ROOT / "data/schema/micronaut-track.schema.json"
TRACK_TYPES = ROOT / "data/schema/micronaut-track-types.schema.json"
JROM_H = ROOT / "src/io/jrom.h"
JROM_CPP = ROOT / "src/io/jrom.cpp"
MAIN_CPP = ROOT / "src/main.cpp"
MICRONAUT_MANIFEST = ROOT / "data/manifest/micronaut.manifest.json"
SCHEMA_MANIFEST = ROOT / "data/manifest/schema.manifest.json"

PROFILE_FIELDS = ["brain_profile", "personality", "capability_grams", "progression"]
AUTHORITY_FIELDS = [
    "semantic_authority",
    "topology_authority",
    "verification_authority",
    "commit_authority",
]
EPISTEMIC_TYPES = [
    "observation",
    "preference",
    "plan",
    "generated_output",
    "verified_fact",
    "failed_retrieval",
    "capability_metric",
    "execution_trace",
]
GRAMMAR_SURFACE_TOKENS = [
    "epistemic_state",
    "scfml_surface",
    "pattern_learning",
    "aiml_pattern",
    "aiml_template",
    "key_grammar",
]
GRAMMAR_TOKENS = [
    "micronaut_decl",
    "micronaut_profile_decl",
    "tensor_slot_section",
    "route_contract_decl",
    "aiml_pattern_decl",
    "aiml_template_decl",
    "scfml_pattern_decl",
    "scfml_template_decl",
    "key_grammar_decl",
    "pattern_ref",
    "template_ref",
    "key_grammar_ref",
    "semantic_envelope_decl",
    "projection_only_clause",
    "bounded_metric_object",
    "progression_object",
    "authority_boundary_section",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def manifest_has(manifest: dict, needle: str) -> bool:
    for key, value in manifest.items():
        if needle in key:
            return True
        if isinstance(value, list) and any(needle in str(item) for item in value):
            return True
    return False


def main() -> int:
    cases: list[tuple[str, bool]] = []
    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    grammar = read_text(GRAMMAR) if GRAMMAR.exists() else ""
    track_schema = json.loads(read_text(TRACK_SCHEMA)) if TRACK_SCHEMA.exists() else {}
    track_types = json.loads(read_text(TRACK_TYPES)) if TRACK_TYPES.exists() else {}
    jrom_h = read_text(JROM_H) if JROM_H.exists() else ""
    jrom_cpp = read_text(JROM_CPP) if JROM_CPP.exists() else ""
    main_cpp = read_text(MAIN_CPP) if MAIN_CPP.exists() else ""
    micronaut_manifest = json.loads(read_text(MICRONAUT_MANIFEST)) if MICRONAUT_MANIFEST.exists() else {}
    schema_manifest = json.loads(read_text(SCHEMA_MANIFEST)) if SCHEMA_MANIFEST.exists() else {}

    check("grammar_file_present", GRAMMAR.exists())
    for token in GRAMMAR_TOKENS:
        check(f"grammar/has/{token}", token in grammar)
    for token in GRAMMAR_SURFACE_TOKENS:
        check(f"grammar/surface/{token}", token in grammar)
    for field in PROFILE_FIELDS:
        check(f"grammar/profile_field/{field}", field in grammar)
        check(f"track_schema/profile_field/{field}", field in track_schema.get("properties", {}))
    for field in AUTHORITY_FIELDS:
        check(f"grammar/authority_boundary/{field}", field in grammar)
    for epistemic in EPISTEMIC_TYPES:
        check(f"grammar/epistemic_type/{epistemic}", f'"{epistemic}"' in grammar)
        check(f"jrom/epistemic_type/{epistemic}", f'"{epistemic}"' in jrom_cpp)
        check(f"rpc/epistemic_type/{epistemic}", f'"{epistemic}"' in main_cpp)

    defs = track_types.get("$defs", {})
    projection = defs.get("ProjectionOnlyObject", {})
    projection_props = projection.get("properties", {})
    check("track_types/projection_only_const", projection_props.get("projection_only", {}).get("const") is True)
    check("track_types/authority_boundary", "authority_boundary" in projection_props)
    check("jrom_event_has_epistemic_type", "epistemic_type" in jrom_h and "epistemic_type" in jrom_cpp)
    check("jrom_has_valid_epistemic_gate", "is_valid_epistemic_type" in jrom_h and "is_valid_epistemic_type" in jrom_cpp)
    check("semantic_envelope_logs_epistemic_type", "jrom.log(\"semantic_envelope\"" in main_cpp and "safe_epistemic_type" in main_cpp)
    check("micronaut_manifest_references_grammar", manifest_has(micronaut_manifest, "micronaut.runtime-grammar.ebnf"))
    check("schema_manifest_references_grammar", manifest_has(schema_manifest, "micronaut.runtime-grammar.ebnf"))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    result = {
        "@kind": "khanary.micronaut_grammar.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "grammar": str(GRAMMAR.relative_to(ROOT)),
        "verdict": verdict,
    }
    print(json.dumps(result, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
