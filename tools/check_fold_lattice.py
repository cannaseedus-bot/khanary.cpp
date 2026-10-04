#!/usr/bin/env python3
"""Validate Fold Lattice product-graph contracts and fixtures."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "data/grammar/fold-lattice.ebnf"
CONTRACTS = ROOT / "data/grammar/fold-lattice.contracts.json"
FIXTURES = ROOT / "data/conformance/fold_lattice.json"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SEMANTIC_FOLDS = ROOT / "docs/SEMANTIC_FOLDS.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"
CMAKE = ROOT / "CMakeLists.txt"

ROOT_KEYS = {"@type", "@fold", "@generation", "@track", "@fold_state", "@runtime_state", "@axes", "@cells", "@metrics", "@policy_ref"}
RUNTIME_KEYS = {"@mutable", "@state_id", "@digest"}
AXIS_KEYS = {"@theta", "@sigma"}
CELL_KEYS = {"@phase", "@competence", "@substrate", "@callback", "@route_hint", "@policy_gate"}
METRIC_KEYS = {
    "@observations",
    "@support_rate",
    "@exception_rate",
    "@contract_pass_rate",
    "@teacher_distance",
    "@policy_agreement",
}
AT_KEY_RE = re.compile(r"^@[A-Za-z][A-Za-z0-9_]*$")

GRAMMAR_TOKENS = [
    "Fold Lattice Grammar",
    "K_t = (F_h, phi, T, s, sigma, X_t)",
    "K_t = (F_h, phi, T, s, sigma, X_t)",
    "K = (theta, sigma)",
    "G_fold_lattice = G_phase □ G_competence",
    "Fold Lattice != Fold Kernel != XCFE Policy != Execution",
    "FoldLattice     ::=",
    "Cell            ::=",
    "L5 ::= sigma < autonomous implies callback != none",
    "L7 ::= theta = Xul and sigma = autonomous implies policy_gate = true",
    "L9 ::= evolutionary transition recommendation never bypasses XCFE Policy",
    "L11 ::= fold_state is independent of phase; Fold != Xul",
    "L12 ::= Sek proposes, Ch'en proves, XCFE authorizes, Xul commits",
    "L13 ::= K = (theta, sigma) is a slice of K_t = (F_h, phi, T, s, sigma, X_t)",
    "TypeJudgment     ::= \"⊢\" FoldLatticeADT \":\" \"FoldLattice\"",
]

DOC_TOKENS = [
    "Fold Lattice",
    "K_t = (F_h, phi, T, s, sigma, X_t)",
    "K_t = (F_h, phi, T, s, sigma, X_t)",
    "K = (theta, sigma)",
    "G_fold_lattice = G_phase □ G_competence",
    "phase-granular",
    "metrics recommend competence movement; XCFE Policy authorizes state/route movement",
    "Fold != Xul",
    "Sek proposes, Ch'en proves, XCFE authorizes, Xul commits",
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


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_probability(value: Any) -> bool:
    return is_number(value) and 0.0 <= float(value) <= 1.0


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


def validate_axes(value: Any, contracts: dict[str, Any]) -> str:
    if not is_object(value) or not has_only_at_extension_keys(value, AXIS_KEYS):
        return "INVALID_AXES"
    if value.get("@theta") != contracts.get("theta_axis"):
        return "THETA_AXIS_MISMATCH"
    if value.get("@sigma") != contracts.get("sigma_axis"):
        return "SIGMA_AXIS_MISMATCH"
    return ""


def sigma_less_than_autonomous(sigma: str, contracts: dict[str, Any]) -> bool:
    order = contracts.get("sigma_order", {})
    return int(order.get(sigma, -1)) < int(order.get("autonomous", 5))


def validate_cell(cell: Any, contracts: dict[str, Any]) -> str:
    if not is_object(cell) or not has_only_at_extension_keys(cell, CELL_KEYS):
        return "INVALID_CELL"
    phase = cell.get("@phase")
    sigma = cell.get("@competence")
    if phase not in contracts.get("theta_axis", []):
        return "INVALID_PHASE"
    if sigma not in contracts.get("sigma_axis", []):
        return "INVALID_COMPETENCE"
    if cell.get("@substrate") not in contracts.get("substrates", []):
        return "INVALID_SUBSTRATE"
    if cell.get("@callback") not in contracts.get("callbacks", []):
        return "INVALID_CALLBACK"
    if cell.get("@route_hint") not in contracts.get("route_hints", []):
        return "INVALID_ROUTE_HINT"
    if not isinstance(cell.get("@policy_gate"), bool):
        return "INVALID_POLICY_GATE"
    if sigma_less_than_autonomous(sigma, contracts) and cell.get("@callback") == "none":
        return "SUBAUTONOMOUS_REQUIRES_CALLBACK"
    if sigma == "autonomous" and cell.get("@callback") != "none":
        return "AUTONOMOUS_CALLBACK_NOT_NONE"
    if phase == "Xul" and sigma == "autonomous" and cell.get("@policy_gate") is not True:
        return "XUL_AUTONOMOUS_REQUIRES_POLICY_GATE"
    return ""


def validate_metrics(metrics: Any) -> str:
    if not is_object(metrics) or not has_only_at_extension_keys(metrics, METRIC_KEYS):
        return "INVALID_METRICS"
    if not is_int(metrics.get("@observations")) or int(metrics["@observations"]) < 0:
        return "METRIC_NEGATIVE"
    for key in ("@support_rate", "@exception_rate", "@contract_pass_rate", "@policy_agreement"):
        if not is_probability(metrics.get(key)):
            return "METRIC_RATE_OUT_OF_RANGE"
    if not is_number(metrics.get("@teacher_distance")) or float(metrics["@teacher_distance"]) < 0:
        return "METRIC_NEGATIVE"
    return ""


def recommendations(metrics: dict[str, Any]) -> list[str]:
    observations = int(metrics.get("@observations", 0))
    support_rate = float(metrics.get("@support_rate", 0.0))
    exception_rate = float(metrics.get("@exception_rate", 1.0))
    contract_pass_rate = float(metrics.get("@contract_pass_rate", 0.0))
    teacher_distance = float(metrics.get("@teacher_distance", math.inf))
    policy_agreement = float(metrics.get("@policy_agreement", 0.0))
    out: list[str] = []
    if observations >= 8 and support_rate >= 0.60:
        out.append("observed_to_learned")
    if teacher_distance <= 0.25 and support_rate >= 0.70:
        out.append("learned_to_bound")
    if contract_pass_rate >= 0.90 and exception_rate <= 0.10:
        out.append("bound_to_validated")
    if policy_agreement >= 0.95 and contract_pass_rate >= 0.95:
        out.append("validated_to_autonomous")
    return out


def validate_lattice(lattice: Any, contracts: dict[str, Any]) -> str:
    if not is_object(lattice):
        return "LATTICE_NOT_OBJECT"
    if not has_only_at_extension_keys(lattice, ROOT_KEYS):
        return "INVALID_MEMBER_KEY"
    if lattice.get("@type") != "khanary.fold_lattice.v1":
        return "INVALID_TYPE"
    if not is_string(lattice.get("@fold")):
        return "INVALID_FOLD"
    if "@generation" in lattice and (not is_int(lattice["@generation"]) or int(lattice["@generation"]) < 0):
        return "INVALID_GENERATION"
    if not is_string(lattice.get("@track")):
        return "INVALID_TRACK"
    if lattice.get("@fold_state") not in contracts.get("fold_states", []):
        return "INVALID_FOLD_STATE"
    runtime_error = validate_runtime_state(lattice.get("@runtime_state"))
    if runtime_error:
        return runtime_error
    if not is_string(lattice.get("@policy_ref")):
        return "MISSING_POLICY_REF"
    axes_error = validate_axes(lattice.get("@axes"), contracts)
    if axes_error:
        return axes_error
    cells = lattice.get("@cells")
    if not isinstance(cells, list) or not cells:
        return "MISSING_CELLS"
    seen: set[tuple[str, str]] = set()
    phases: list[str] = []
    for cell in cells:
        error = validate_cell(cell, contracts)
        if error:
            return error
        coord = (cell["@phase"], cell["@competence"])
        if coord in seen:
            return "DUPLICATE_COORDINATE"
        seen.add(coord)
        phases.append(cell["@phase"])
    if len(cells) > 1:
        if sorted(phases) != sorted(contracts.get("theta_axis", [])):
            return "FULL_LATTICE_REQUIRES_ONE_CELL_PER_PHASE"
    metrics_error = validate_metrics(lattice.get("@metrics"))
    if metrics_error:
        return metrics_error
    return ""


def main() -> int:
    contracts = load_json(CONTRACTS) if CONTRACTS.exists() else {}
    fixtures = load_json(FIXTURES) if FIXTURES.exists() else {}
    grammar = read(GRAMMAR)
    docs = "\n".join(read(path) for path in (GLOSSARY, SEMANTIC_FOLDS, SPEC))
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    check("grammar_file_present", GRAMMAR.exists())
    check("contracts_file_present", CONTRACTS.exists())
    check("fixtures_file_present", FIXTURES.exists())
    check("contracts_type", contracts.get("@type") == "khanary.fold_lattice_contracts.v1")
    check("semantic_authority_false", contracts.get("semantic_authority") is False)
    check("boundary_declared", "Fold Lattice != Fold Kernel != XCFE Policy != Execution" in str(contracts.get("law", {})))
    check("theta_axis_declared", contracts.get("theta_axis") == ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"])
    check("sigma_axis_declared", contracts.get("sigma_axis") == ["unbound", "observed", "learned", "bound", "validated", "autonomous"])
    check("product_has_36_cells", len(contracts.get("theta_axis", [])) * len(contracts.get("sigma_axis", [])) == 36)
    law = contracts.get("law", {})
    check("full_coordinate_declared", law.get("full_coordinate") == "K_t = (F_h, phi, T, s, sigma, X_t)")
    check("slice_coordinate_declared", "phase/competence slice" in str(law.get("slice_coordinate", "")))
    check("fold_not_xul_declared", "Fold != Xul" in str(law.get("fold_boundary", "")))
    check("proof_boundary_declared", "Sek proposes" in str(law.get("proof_boundary", "")))
    check("fold_states_declared", contracts.get("fold_states") == ["folded", "unfolded"])
    for token in GRAMMAR_TOKENS:
        check(f"grammar/has/{token}", token in grammar)
    for token in DOC_TOKENS:
        check(f"docs/has/{token}", token in docs)
    for invariant in [f"L{i}" for i in range(1, 14)]:
        check(f"invariant_declared/{invariant}", invariant in contracts.get("invariants", {}))
    for name in ["observed_to_learned", "learned_to_bound", "bound_to_validated", "validated_to_autonomous"]:
        check(f"transition_metric/{name}", name in contracts.get("transition_metrics", {}))

    fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
    check("fixtures_have_cases", isinstance(fixture_cases, list) and len(fixture_cases) >= 15)
    pass_seen = 0
    fail_seen = 0
    recommendation_seen = 0
    for case in fixture_cases:
        if not is_object(case):
            check("fixture_case_is_object", False)
            continue
        case_id = str(case.get("id", "unnamed"))
        error = validate_lattice(case.get("lattice"), contracts)
        actual = "PASS" if not error else "FAIL"
        if actual == "PASS":
            pass_seen += 1
        else:
            fail_seen += 1
        check(f"fixture/{case_id}/expect", actual == case.get("expect"))
        if case.get("expect") == "FAIL":
            check(f"fixture/{case_id}/error", error == case.get("error"))
        if actual == "PASS" and "expect_recommendations" in case:
            recommendation_seen += 1
            got = recommendations(case["lattice"].get("@metrics", {}))
            check(f"fixture/{case_id}/recommendations", got == case.get("expect_recommendations"))
    check("fixtures_include_valid_lattice_cases", pass_seen >= 3)
    check("fixtures_include_negative_cases", fail_seen >= 13)
    check("fixtures_include_transition_recommendations", recommendation_seen >= 1)
    check("spec_lists_fold_lattice_grammar", "fold-lattice.ebnf" in read(SPEC))
    check("spec_lists_fold_lattice_checker", "check_fold_lattice.py" in read(SPEC))
    check("cmake_registers_fold_lattice_test", "KHANARY_FOLD_LATTICE_TEST" in read(CMAKE))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.fold_lattice.verdict.v1",
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
