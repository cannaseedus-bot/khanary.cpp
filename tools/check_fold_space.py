#!/usr/bin/env python3
"""Validate Fold Space navigation contracts and fixtures."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "data/grammar/fold-space.ebnf"
CONTRACTS = ROOT / "data/grammar/fold-space.contracts.json"
FIXTURES = ROOT / "data/conformance/fold_space.json"
LATTICE_CONTRACTS = ROOT / "data/grammar/fold-lattice.contracts.json"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SEMANTIC_FOLDS = ROOT / "docs/SEMANTIC_FOLDS.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"
CMAKE = ROOT / "CMakeLists.txt"

ROOT_KEYS = {"@type", "@coordinate", "@movements", "@book", "@distances", "@pipeline", "@authority", "@lattice_ref"}
POSITION_KEYS = {"@h", "@phase", "@track", "@fold_state", "@competence", "@runtime_state"}
RUNTIME_KEYS = {"@mutable", "@state_id", "@digest"}
MOVEMENT_KEYS = {"@type", "@from", "@to"}
BOOK_KEYS = {"@handle", "@neighborhood", "@bounded_by", "@open", "@close"}
DISTANCE_KEYS = {"@type", "@formula", "@typed"}
AUTHORITY_KEYS = {"@c6_authority", "@s1_projection_only", "@semantic_authority", "@route_authority", "@proof_boundary"}
AT_KEY_RE = re.compile(r"^@[A-Za-z][A-Za-z0-9_]*$")

GRAMMAR_TOKENS = [
    "Fold Space Grammar",
    "F_K = K'UHUL Fold Space",
    "p_t = (h, phi, T, s, X_t)",
    "p_t = (h, phi, T, s, sigma, X_t)",
    "K = (theta, sigma) is a slice",
    "C6 is the authoritative execution",
    "S1/angle geometry is a projection",
    "Fold Space != Fold Lattice != Fold Kernel != XCFE Policy != Execution",
    "Horizontal      ::=",
    "Vertical        ::=",
    "Radial          ::=",
    "BookADT         ::= \"Book\"",
    "UnfoldLaw       ::= \"Unfold\"",
    "Pipeline        ::= \"LOCATE\"",
    "FS10 ::= C6 is authoritative execution coordinate; S1 angle is projection only",
    "FS11 ::= Book(h) = N(h) and Unfold(h,tau) is bounded by B_tau[h,e] != 0",
    "FS12 ::= distance is typed: semantic, phase, track, or behavior; no universal metric is implied",
    "FS13 ::= navigation order is LOCATE -> UNFOLD -> NAVIGATE -> OPERATE -> VERIFY -> FOLD -> XUL",
    "FS14 ::= Fold != Xul",
    "FS15 ::= Sek proposes, Ch'en proves, XCFE authorizes, Xul commits",
    "TypeJudgment     ::= \"⊢\" FoldSpaceADT \":\" \"FoldSpace\"",
]

DOC_TOKENS = [
    "Fold Space",
    "F_K = K'UHUL Fold Space",
    "p_t = (h, phi, T, s, X_t)",
    "p_t = (h, phi, T, s, sigma, X_t)",
    "Horizontal movement changes semantic track",
    "Vertical movement advances phase",
    "Radial movement toggles fold/unfold",
    "C6 is the authoritative execution coordinate; S1 is a projection",
    "Book(h) == N(h)",
    "LOCATE -> UNFOLD -> NAVIGATE -> OPERATE -> VERIFY -> FOLD -> XUL",
    "Fold Lattice is the K = (theta, sigma) slice",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def is_object(value: Any) -> bool:
    return isinstance(value, dict)


def is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def has_only_at_extension_keys(obj: dict[str, Any], known: set[str]) -> bool:
    for key in obj:
        if key in known:
            continue
        if not AT_KEY_RE.match(key):
            return False
    return True


def validate_runtime_state(value: Any) -> str:
    if not is_object(value) or not has_only_at_extension_keys(value, RUNTIME_KEYS):
        return "INVALID_RUNTIME_STATE"
    if not isinstance(value.get("@mutable"), bool):
        return "INVALID_RUNTIME_STATE"
    if not is_string(value.get("@state_id")):
        return "INVALID_RUNTIME_STATE"
    if "@digest" in value and not is_string(value["@digest"]):
        return "INVALID_RUNTIME_STATE"
    return ""


def validate_position(value: Any, contracts: dict[str, Any]) -> str:
    if not is_object(value) or not has_only_at_extension_keys(value, POSITION_KEYS):
        return "INVALID_POSITION"
    if not is_string(value.get("@h")):
        return "INVALID_HANDLE"
    if value.get("@phase") not in contracts.get("phase_cycle", []):
        return "INVALID_PHASE"
    if not is_string(value.get("@track")):
        return "INVALID_TRACK"
    if value.get("@fold_state") not in contracts.get("fold_states", []):
        return "INVALID_FOLD_STATE"
    if "@competence" in value and value.get("@competence") not in contracts.get("competence_axis", []):
        return "INVALID_COMPETENCE"
    return validate_runtime_state(value.get("@runtime_state"))


def next_phase(phase: str, contracts: dict[str, Any]) -> str:
    phases = contracts.get("phase_cycle", [])
    return phases[(phases.index(phase) + 1) % len(phases)]


def validate_movement(value: Any, contracts: dict[str, Any]) -> str:
    if not is_object(value) or not has_only_at_extension_keys(value, MOVEMENT_KEYS):
        return "INVALID_MOVEMENT"
    movement_type = value.get("@type")
    if movement_type not in contracts.get("movement_types", []):
        return "INVALID_MOVEMENT_TYPE"
    source = value.get("@from")
    target = value.get("@to")
    error = validate_position(source, contracts)
    if error:
        return error
    error = validate_position(target, contracts)
    if error:
        return error
    if movement_type == "horizontal":
        if source.get("@phase") != target.get("@phase"):
            return "HORIZONTAL_PHASE_CHANGED"
        if source.get("@track") == target.get("@track"):
            return "HORIZONTAL_TRACK_UNCHANGED"
    elif movement_type == "vertical":
        if source.get("@track") != target.get("@track"):
            return "VERTICAL_TRACK_CHANGED"
        if target.get("@phase") != next_phase(source.get("@phase"), contracts):
            return "VERTICAL_WRONG_PHASE_STEP"
    elif movement_type == "radial":
        if source.get("@phase") != target.get("@phase") or source.get("@track") != target.get("@track"):
            return "RADIAL_PHASE_OR_TRACK_CHANGED"
        if source.get("@fold_state") == target.get("@fold_state"):
            return "RADIAL_FOLD_STATE_NOT_TOGGLED"
    return ""


def validate_book(book: Any, coordinate: dict[str, Any]) -> str:
    if not is_object(book) or not has_only_at_extension_keys(book, BOOK_KEYS):
        return "INVALID_BOOK"
    if not is_string(book.get("@handle")) or not is_string(book.get("@neighborhood")):
        return "INVALID_BOOK"
    if book.get("@handle") != coordinate.get("@h"):
        return "BOOK_HANDLE_MISMATCH"
    if book.get("@bounded_by") != "B_tau[h,e] != 0":
        return "INVALID_BOOK_BOUNDARY"
    if book.get("@open") != "UNFOLD" or book.get("@close") != "FOLD":
        return "INVALID_BOOK_OPERATION"
    return ""


def validate_distances(distances: Any, contracts: dict[str, Any]) -> str:
    if not isinstance(distances, list) or not distances:
        return "INVALID_DISTANCES"
    seen: set[str] = set()
    allowed = set(contracts.get("distance_metrics", {}).keys())
    for distance in distances:
        if not is_object(distance) or not has_only_at_extension_keys(distance, DISTANCE_KEYS):
            return "INVALID_DISTANCE"
        metric = distance.get("@type")
        if metric not in allowed:
            return "INVALID_DISTANCE_METRIC"
        if distance.get("@typed") is not True:
            return "DISTANCE_NOT_TYPED"
        if not is_string(distance.get("@formula")):
            return "INVALID_DISTANCE_FORMULA"
        seen.add(metric)
    if seen != allowed:
        return "MISSING_DISTANCE_METRIC"
    return ""


def validate_authority(authority: Any, contracts: dict[str, Any]) -> str:
    if not is_object(authority) or not has_only_at_extension_keys(authority, AUTHORITY_KEYS):
        return "INVALID_AUTHORITY"
    required = contracts.get("authority_required", {})
    if authority.get("@c6_authority") is not required.get("c6_authority"):
        return "C6_NOT_AUTHORITY"
    if authority.get("@s1_projection_only") is not required.get("s1_projection_only"):
        return "S1_NOT_PROJECTION_ONLY"
    if authority.get("@semantic_authority") is not required.get("semantic_authority"):
        return "SEMANTIC_AUTHORITY_CLAIMED"
    if authority.get("@route_authority") != required.get("route_authority"):
        return "INVALID_ROUTE_AUTHORITY"
    if authority.get("@proof_boundary") != required.get("proof_boundary"):
        return "INVALID_PROOF_BOUNDARY"
    return ""


def validate_space(space: Any, contracts: dict[str, Any]) -> str:
    if not is_object(space) or not has_only_at_extension_keys(space, ROOT_KEYS):
        return "INVALID_SPACE"
    if space.get("@type") != "khanary.fold_space.v1":
        return "INVALID_TYPE"
    coordinate = space.get("@coordinate")
    error = validate_position(coordinate, contracts)
    if error:
        return error
    movements = space.get("@movements")
    if not isinstance(movements, list) or not movements:
        return "INVALID_MOVEMENTS"
    movement_types = set()
    for movement in movements:
        error = validate_movement(movement, contracts)
        if error:
            return error
        movement_types.add(movement.get("@type"))
    if not movement_types.issubset(set(contracts.get("movement_types", []))):
        return "INVALID_MOVEMENT_TYPE"
    error = validate_book(space.get("@book"), coordinate)
    if error:
        return error
    error = validate_distances(space.get("@distances"), contracts)
    if error:
        return error
    if space.get("@pipeline") != contracts.get("pipeline"):
        return "INVALID_PIPELINE"
    error = validate_authority(space.get("@authority"), contracts)
    if error:
        return error
    if space.get("@lattice_ref") != contracts.get("lattice_contract"):
        return "INVALID_LATTICE_REF"
    return ""


def main() -> int:
    contracts = load_json(CONTRACTS) if CONTRACTS.exists() else {}
    fixtures = load_json(FIXTURES) if FIXTURES.exists() else {}
    lattice_contracts = load_json(LATTICE_CONTRACTS) if LATTICE_CONTRACTS.exists() else {}
    grammar = read(GRAMMAR)
    docs = "\n".join(read(path) for path in (GLOSSARY, SEMANTIC_FOLDS, SPEC))
    cmake = read(CMAKE)
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    check("grammar_file_present", GRAMMAR.exists())
    check("contracts_file_present", CONTRACTS.exists())
    check("fixtures_file_present", FIXTURES.exists())
    check("contracts_type", contracts.get("@type") == "khanary.fold_space_contracts.v1")
    check("semantic_authority_false", contracts.get("semantic_authority") is False)
    law = contracts.get("law", {})
    check("law_name", law.get("name") == "F_K = K'UHUL Fold Space")
    check("law_coordinate", law.get("coordinate") == "p_t = (h, phi, T, s, X_t)")
    check("law_learning_coordinate", law.get("learning_coordinate") == "p_t = (h, phi, T, s, sigma, X_t)")
    check("law_lattice_slice", "Fold Lattice K = (theta, sigma)" in str(law.get("lattice_slice", "")))
    check("phase_cycle", contracts.get("phase_cycle") == ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"])
    check("fold_states", contracts.get("fold_states") == ["folded", "unfolded"])
    check("movement_types", contracts.get("movement_types") == ["horizontal", "vertical", "radial"])
    check("distance_metrics", set(contracts.get("distance_metrics", {}).keys()) == {"semantic", "phase", "track", "behavior"})
    check("pipeline_declared", contracts.get("pipeline") == ["LOCATE", "UNFOLD", "NAVIGATE", "OPERATE", "VERIFY", "FOLD", "XUL"])
    check("lattice_contract_ref", contracts.get("lattice_contract") == "data/grammar/fold-lattice.contracts.json")
    check("lattice_points_back_to_fold_space", lattice_contracts.get("fold_space_contract") == "data/grammar/fold-space.contracts.json")

    for token in GRAMMAR_TOKENS:
        check(f"grammar/has/{token}", token in grammar)
    for token in DOC_TOKENS:
        check(f"docs/has/{token}", token in docs)
    for invariant in [f"FS{i}" for i in range(1, 16)]:
        check(f"invariant_declared/{invariant}", invariant in contracts.get("invariants", {}))
    check("spec_lists_fold_space_grammar", "fold-space.ebnf" in read(SPEC))
    check("spec_lists_fold_space_checker", "check_fold_space.py" in read(SPEC))
    check("cmake_registers_fold_space_test", "KHANARY_FOLD_SPACE_TEST" in cmake)

    fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
    check("fixtures_have_cases", isinstance(fixture_cases, list) and len(fixture_cases) >= 20)
    pass_seen = 0
    fail_seen = 0
    for case in fixture_cases:
        if not is_object(case):
            check("fixture_case_is_object", False)
            continue
        case_id = str(case.get("id", "unnamed"))
        error = validate_space(case.get("space"), contracts)
        actual = "PASS" if not error else "FAIL"
        if actual == "PASS":
            pass_seen += 1
        else:
            fail_seen += 1
        check(f"fixture/{case_id}/expect", actual == case.get("expect"))
        if case.get("expect") == "FAIL":
            check(f"fixture/{case_id}/error", error == case.get("error"))
    check("fixtures_include_valid_spaces", pass_seen >= 2)
    check("fixtures_include_negative_spaces", fail_seen >= 18)

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.fold_space.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "grammar": str(GRAMMAR.relative_to(ROOT)),
        "contracts": str(CONTRACTS.relative_to(ROOT)),
        "fixtures": str(FIXTURES.relative_to(ROOT)),
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
