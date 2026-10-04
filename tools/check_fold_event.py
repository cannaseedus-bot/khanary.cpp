#!/usr/bin/env python3
"""Validate Fold Event JSON Schema, contracts, and conformance fixtures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "data/schema/fold-event.schema.json"
CONTRACTS = ROOT / "data/grammar/fold-event.contracts.json"
FIXTURES = ROOT / "data/conformance/fold_event.json"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SEMANTIC_FOLDS = ROOT / "docs/SEMANTIC_FOLDS.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"
CMAKE = ROOT / "CMakeLists.txt"
SCHEMA_MANIFEST = ROOT / "data/manifest/schema.manifest.json"

DOC_TOKENS = [
    "Fold Theory",
    "Fold Field",
    "Fold Event",
    "FrakturF = (F_K, M_F, E_F, delta_Pi, B, Exe)",
    "M_F(h) = bigoplus_tau B_tau[h,*] tensor W_tau",
    "p_next = delta_XCFE(p_t, e_t, Pi)",
    "EVENT_SHIFT_TRACK",
    "EVENT_CYCLE_PHASE",
    "EVENT_FOLD_STATE",
    "EVENT_COMPETENCE_CANDIDATE",
    "EVENT_TOPOLOGY_COMMIT",
    "gradient_Q(M_F) is candidate navigation pressure",
]

SCHEMA_TOKENS = [
    "urn:khanary:schema:fold-event:1",
    "khanary.fold_event.v1",
    "EVENT_SHIFT_TRACK",
    "EVENT_CYCLE_PHASE",
    "EVENT_FOLD_STATE",
    "EVENT_OSCILLATE_DENSITY",
    "EVENT_SUBSTRATE_HANDSHAKE",
    "EVENT_COMPETENCE_CANDIDATE",
    "EVENT_COMPETENCE_UPGRADE",
    "EVENT_FIELD_GRADIENT",
    "EVENT_TOPOLOGY_CANDIDATE",
    "EVENT_TOPOLOGY_COMMIT",
    "p_next = delta_XCFE(p_t, e_t, Pi)",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def load_json(path: Path) -> Any:
    return json.loads(read(path))


def is_object(value: Any) -> bool:
    return isinstance(value, dict)


def is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def next_phase(phase: str, contracts: dict[str, Any]) -> str:
    phases = contracts.get("phase_cycle", [])
    return phases[(phases.index(phase) + 1) % len(phases)]


def schema_validator(schema: dict[str, Any]):
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_event(event: Any, contracts: dict[str, Any], validator: Any) -> str:
    errors = list(validator.iter_errors(event))
    if errors:
        return "SCHEMA_INVALID"
    kind = event["@event_kind"]
    axis = event["@axis"]
    if contracts.get("axis_by_event", {}).get(kind) != axis:
        return "EVENT_AXIS_MISMATCH"
    before = event["@before"]
    after = event["@proposed_after"]
    xcfe = event["@xcfe"]
    delta = event["@field_delta"]
    projection = event.get("@projection")
    if projection is not None and projection.get("@semantic_authority") is not False:
        return "PROJECTION_CLAIMS_AUTHORITY"

    if kind == "EVENT_SHIFT_TRACK":
        if before["@phase"] != after["@phase"]:
            return "SHIFT_TRACK_CHANGED_PHASE"
        if before["@track"] == after["@track"]:
            return "SHIFT_TRACK_UNCHANGED"
    elif kind == "EVENT_CYCLE_PHASE":
        if before["@track"] != after["@track"]:
            return "CYCLE_PHASE_CHANGED_TRACK"
        if after["@phase"] != next_phase(before["@phase"], contracts):
            return "CYCLE_PHASE_WRONG_STEP"
    elif kind in {"EVENT_FOLD_STATE", "EVENT_OSCILLATE_DENSITY"}:
        if before["@phase"] != after["@phase"] or before["@track"] != after["@track"]:
            return "FOLD_STATE_CHANGED_PHASE_OR_TRACK"
        if before["@fold_state"] == after["@fold_state"]:
            return "FOLD_STATE_NOT_TOGGLED"
    elif kind == "EVENT_COMPETENCE_CANDIDATE":
        if before.get("@competence") == after.get("@competence"):
            return "COMPETENCE_UNCHANGED"
        if xcfe.get("@authorized") is True or xcfe.get("@status") == "authorized":
            return "COMPETENCE_CANDIDATE_AUTHORIZED"
    elif kind == "EVENT_COMPETENCE_UPGRADE":
        if before.get("@competence") == after.get("@competence"):
            return "COMPETENCE_UNCHANGED"
        if xcfe.get("@authorized") is not True or xcfe.get("@status") != "authorized":
            return "COMPETENCE_UPGRADE_NOT_AUTHORIZED"
    elif kind == "EVENT_FIELD_GRADIENT":
        if delta.get("@kind") != "field_projection" or delta.get("@delta_B") is not False:
            return "FIELD_GRADIENT_MUST_NOT_COMMIT"
        if delta.get("@gradient_role") != "candidate_navigation_pressure":
            return "FIELD_GRADIENT_BAD_ROLE"
    elif kind == "EVENT_TOPOLOGY_CANDIDATE":
        if delta.get("@kind") == "topology_commit" or delta.get("@xul") is True:
            return "TOPOLOGY_CANDIDATE_COMMITTED"
    elif kind == "EVENT_TOPOLOGY_COMMIT":
        if before.get("@phase") != "Xul" or after.get("@phase") != "Xul":
            return "TOPOLOGY_COMMIT_REQUIRES_XUL_PHASE"
        required = [delta.get("@delta_B"), delta.get("@verified"), delta.get("@authorized"), delta.get("@fresh"), delta.get("@xul"), xcfe.get("@authorized")]
        if delta.get("@kind") != "topology_commit" or not all(value is True for value in required):
            return "TOPOLOGY_COMMIT_REQUIRES_VERIFIED_AUTHORIZED_FRESH_XUL"
    return ""


def main() -> int:
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    try:
        schema = load_json(SCHEMA) if SCHEMA.exists() else {}
        contracts = load_json(CONTRACTS) if CONTRACTS.exists() else {}
        fixtures = load_json(FIXTURES) if FIXTURES.exists() else {}
        validator = schema_validator(schema)
        docs = "\n".join(read(path) for path in (GLOSSARY, SEMANTIC_FOLDS, SPEC))
        schema_text = read(SCHEMA)
        manifest = load_json(SCHEMA_MANIFEST) if SCHEMA_MANIFEST.exists() else {}

        check("schema_file_present", SCHEMA.exists())
        check("contracts_file_present", CONTRACTS.exists())
        check("fixtures_file_present", FIXTURES.exists())
        check("schema_id", schema.get("$id") == "urn:khanary:schema:fold-event:1")
        check("contracts_type", contracts.get("@type") == "khanary.fold_event_contracts.v1")
        check("semantic_authority_false", contracts.get("semantic_authority") is False)
        check("json_schema_ref", contracts.get("json_schema") == "data/schema/fold-event.schema.json")
        check("fold_space_contract_ref", contracts.get("fold_space_contract") == "data/grammar/fold-space.contracts.json")
        check("transition_law", contracts.get("event_law", {}).get("transition") == "p_next = delta_XCFE(p_t, e_t, Pi)")
        check("fold_theory_formal", contracts.get("fold_theory", {}).get("formal") == "FrakturF = (F_K, M_F, E_F, delta_Pi, B, Exe)")
        check("fold_field_formula", contracts.get("fold_field", {}).get("formula") == "M_F(h) = bigoplus_tau B_tau[h,*] tensor W_tau")
        check("gradient_law", "candidate navigation pressure" in str(contracts.get("fold_field", {}).get("gradient_law", "")))
        check("axis_map_complete", set(contracts.get("axis_by_event", {})) == set(contracts.get("event_kinds", [])))
        check("schema_manifest_references_fold_event", any("fold-event.schema.json" in item for item in manifest.get("schema:fold_event", [])))
        for token in SCHEMA_TOKENS:
            check(f"schema/has/{token}", token in schema_text)
        for token in DOC_TOKENS:
            check(f"docs/has/{token}", token in docs)
        for invariant in [f"FE{i}" for i in range(1, 13)]:
            check(f"invariant_declared/{invariant}", invariant in contracts.get("invariants", {}))
        check("spec_lists_fold_event_schema", "fold-event.schema.json" in read(SPEC))
        check("spec_lists_fold_event_checker", "check_fold_event.py" in read(SPEC))
        check("cmake_registers_fold_event_test", "KHANARY_FOLD_EVENT_TEST" in read(CMAKE))

        fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
        check("fixtures_have_cases", isinstance(fixture_cases, list) and len(fixture_cases) >= 20)
        pass_seen = 0
        fail_seen = 0
        for row in fixture_cases:
            if not is_object(row):
                check("fixture_case_is_object", False)
                continue
            case_id = str(row.get("id", "unnamed"))
            error = validate_event(row.get("event"), contracts, validator)
            actual = "PASS" if not error else "FAIL"
            if actual == "PASS":
                pass_seen += 1
            else:
                fail_seen += 1
            check(f"fixture/{case_id}/expect", actual == row.get("expect"))
            if row.get("expect") == "FAIL":
                check(f"fixture/{case_id}/error", error == row.get("error"))
        check("fixtures_include_valid_events", pass_seen >= 9)
        check("fixtures_include_negative_events", fail_seen >= 17)

    except Exception as exc:
        cases.append(("exception/" + str(exc), False))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.fold_event.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "schema": str(SCHEMA.relative_to(ROOT)),
        "contracts": str(CONTRACTS.relative_to(ROOT)),
        "fixtures": str(FIXTURES.relative_to(ROOT)),
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
