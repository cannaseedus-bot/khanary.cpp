#!/usr/bin/env python3
"""Validate Fold Kernel grammar, lifecycle contracts, and conformance fixtures."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from check_xcfe_policy import evaluate_confidence as evaluate_xcfe_confidence
from check_xcfe_policy import validate_policy_record

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "data/grammar/fold-kernel.ebnf"
CONTRACTS = ROOT / "data/grammar/fold-kernel.contracts.json"
BINDINGS = ROOT / "data/grammar/fold-bindings.contracts.json"
XCFE_POLICY_CONTRACTS = ROOT / "data/grammar/xcfe-policy.contracts.json"
FIXTURES = ROOT / "data/conformance/fold_kernel.json"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SEMANTIC_FOLDS = ROOT / "docs/SEMANTIC_FOLDS.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"

ROOT_KEYS = {
    "@fold",
    "@phase",
    "@semantic",
    "@state",
    "@learned",
    "@binding",
    "@evidence",
    "@policy",
    "@route",
    "@ui",
}

LEARNED_KEYS = {"@pattern", "@ngrams", "@signature"}
BINDING_KEYS = {"@backend", "@entry", "@dispatch", "@contract"}
EVIDENCE_KEYS = {"@observations", "@support", "@exceptions", "@validated", "@failures", "@confidence"}
ROUTE_KEYS = {"@authority", "@target", "@executor", "@reason"}
UI_KEYS = {"@display", "@color", "@icon"}

AT_KEY_RE = re.compile(r"^@[A-Za-z][A-Za-z0-9_]*$")

GRAMMAR_TOKENS = [
    "FoldKernel",
    "Concrete JSON syntax",
    "Abstract semantic structure",
    "State-dependent JSON record",
    "Fold Kernel != Fold Binding",
    "Fold Binding binds (alpha, theta) to a backend entrypoint",
    "State           ::= '\"unbound\"'",
    "Evidence        ::= \"{\" EvidenceMember",
    "Policy          ::= XCFEPolicy",
    "I1 ::= evidence.observations = evidence.support + evidence.exceptions",
    "S5 ::= state in { validated, autonomous } implies route != null",
    "A2 ::= evidence.confidence = XCFEPolicy(evidence)",
    "TypeJudgment ::= \"⊢\" FoldKernelADT \":\" \"FoldKernel\"",
]

DOC_TOKENS = [
    "Fold Kernel",
    "N-Gram Burner lifecycle",
    "@learned",
    "@binding",
    "@evidence",
    "@policy",
    "@route",
    "XCFE owns the route",
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


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_nonnegative_int(value: Any) -> bool:
    return is_int(value) and value >= 0


def is_probability(value: Any) -> bool:
    return is_number(value) and 0.0 <= float(value) <= 1.0


def has_only_at_extension_keys(obj: dict[str, Any], known: set[str]) -> bool:
    for key in obj:
        if key in known:
            continue
        if not AT_KEY_RE.match(key):
            return False
    return True


def first_missing(obj: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key not in obj or obj[key] is None:
            return key
    return ""


def dispatch_shape(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(is_int(part) and part > 0 for part in value)
    )


def contract_shape(value: Any) -> bool:
    return (
        is_object(value)
        and set(value.keys()) == {"pre", "post"}
        and is_string(value.get("pre"))
        and is_string(value.get("post"))
    )


def validate_learned(value: Any) -> str:
    if not is_object(value):
        return "INVALID_LEARNED"
    if not has_only_at_extension_keys(value, LEARNED_KEYS):
        return "INVALID_MEMBER_KEY"
    if not is_string(value.get("@pattern")):
        return "INVALID_LEARNED_PATTERN"
    if "@ngrams" in value:
        ngrams = value["@ngrams"]
        if not isinstance(ngrams, list) or not all(is_string(item) for item in ngrams):
            return "INVALID_LEARNED_NGRAMS"
    if "@signature" in value and not is_string(value["@signature"]):
        return "INVALID_LEARNED_SIGNATURE"
    return ""


def validate_binding(value: Any) -> str:
    if not is_object(value):
        return "INVALID_BINDING"
    if not has_only_at_extension_keys(value, BINDING_KEYS):
        return "INVALID_MEMBER_KEY"
    if not is_string(value.get("@backend")):
        return "INVALID_BINDING_BACKEND"
    if not is_string(value.get("@entry")):
        return "INVALID_BINDING_ENTRY"
    if "@dispatch" in value and not dispatch_shape(value["@dispatch"]):
        return "DISPATCH_SHAPE_INVALID"
    if not contract_shape(value.get("@contract")):
        return "INVALID_BINDING_CONTRACT"
    return ""


def validate_evidence(value: Any) -> str:
    if not is_object(value):
        return "INVALID_EVIDENCE"
    if not has_only_at_extension_keys(value, EVIDENCE_KEYS):
        return "INVALID_MEMBER_KEY"
    for key in ("@observations", "@support", "@exceptions", "@validated", "@failures"):
        if not is_nonnegative_int(value.get(key)):
            return "INVALID_EVIDENCE_FIELD"
    if value["@observations"] != value["@support"] + value["@exceptions"]:
        return "EVIDENCE_OBSERVATIONS_MISMATCH"
    if value["@validated"] < value["@failures"]:
        return "EVIDENCE_VALIDATION_MISMATCH"
    if "@confidence" in value and not is_probability(value["@confidence"]):
        return "CONFIDENCE_OUT_OF_RANGE"
    return ""



def policy_evidence_from_fold(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "observations": evidence.get("@observations", 0),
        "support": evidence.get("@support", 0),
        "exceptions": evidence.get("@exceptions", 0),
        "validated": evidence.get("@validated", 0),
        "failures": evidence.get("@failures", 0),
    }

def validate_route_target(value: Any, allowed_targets: list[str]) -> str:
    if value not in allowed_targets:
        return "INVALID_ROUTE_TARGET"
    return ""


def validate_route(value: Any, contracts: dict[str, Any]) -> str:
    if not is_object(value):
        return "INVALID_ROUTE"
    if not has_only_at_extension_keys(value, ROUTE_KEYS):
        return "INVALID_MEMBER_KEY"
    if value.get("@authority") != "XCFE":
        return "ROUTE_AUTHORITY_NOT_XCFE"
    route_error = validate_route_target(value.get("@target"), contracts.get("allowed_route_targets", []))
    if route_error:
        return route_error
    if "@executor" in value and not is_string(value["@executor"]):
        return "INVALID_ROUTE_EXECUTOR"
    if "@reason" in value and not is_string(value["@reason"]):
        return "INVALID_ROUTE_REASON"
    return ""


def validate_ui(value: Any) -> str:
    if not is_object(value):
        return "INVALID_UI"
    if not has_only_at_extension_keys(value, UI_KEYS):
        return "INVALID_MEMBER_KEY"
    if not is_string(value.get("@display")):
        return "INVALID_UI_DISPLAY"
    if "@color" in value and not is_string(value["@color"]):
        return "INVALID_UI_COLOR"
    if "@icon" in value and not is_string(value["@icon"]):
        return "INVALID_UI_ICON"
    return ""


def gamma_confidence(evidence: dict[str, Any], policy: dict[str, Any], zero: float) -> float:
    weights = policy["@weights"]
    observations = float(evidence["@observations"])
    validated = float(evidence["@validated"])
    support_rate = float(evidence["@support"]) / observations if observations else zero
    validation_rate = (float(evidence["@validated"]) - float(evidence["@failures"])) / validated if validated else zero
    non_exception_rate = support_rate
    non_failure_rate = validation_rate
    return (
        float(weights["@alpha"]) * support_rate
        + float(weights["@beta"]) * validation_rate
        + float(weights["@gamma"]) * non_exception_rate
        + float(weights["@delta"]) * non_failure_rate
    )


def matches_binding_registry(record: dict[str, Any], binding_registry: dict[str, Any]) -> bool:
    binding = record.get("@binding")
    if not is_object(binding):
        return True
    if binding.get("@backend") != "hlsl":
        return True
    contract = binding.get("@contract")
    for candidate in binding_registry.get("bindings", []):
        if not is_object(candidate):
            continue
        if candidate.get("@fold") != record.get("@fold"):
            continue
        if candidate.get("@phase") != record.get("@phase"):
            continue
        if candidate.get("@semantic") != record.get("@semantic"):
            continue
        if candidate.get("@backend") != binding.get("@backend"):
            continue
        if candidate.get("@entry") != binding.get("@entry"):
            continue
        if candidate.get("@contract") != contract:
            continue
        if "@dispatch" in binding and candidate.get("@dispatch") != binding.get("@dispatch"):
            continue
        return True
    return False


def validate_record(record: Any, contracts: dict[str, Any], binding_registry: dict[str, Any], policy_contracts: dict[str, Any]) -> str:
    if not is_object(record):
        return "RECORD_NOT_OBJECT"
    if not has_only_at_extension_keys(record, ROOT_KEYS):
        return "INVALID_MEMBER_KEY"
    missing = first_missing(record, contracts.get("required_core_keys", []))
    if missing:
        return "MISSING_CORE_KEY"
    if not is_string(record.get("@fold")):
        return "INVALID_FOLD"
    if record.get("@phase") not in contracts.get("phase_cycle", []):
        return "INVALID_PHASE"
    if record.get("@state") not in contracts.get("states", []):
        return "INVALID_STATE"

    semantic_tags = contracts.get("semantic_tags_by_phase", {}).get(record["@phase"], [])
    if record.get("@semantic") not in semantic_tags:
        return "SEMANTIC_PHASE_MISMATCH"

    requirements = contracts.get("state_requirements", {}).get(record["@state"], [])
    for required_key in requirements:
        if required_key not in record or record[required_key] is None:
            if required_key == "@learned":
                return "STATE_REQUIRES_LEARNED"
            if required_key == "@binding":
                return "STATE_REQUIRES_BINDING"
            if required_key == "@evidence":
                return "STATE_REQUIRES_EVIDENCE"
            if required_key == "@policy":
                return "STATE_REQUIRES_POLICY"
            if required_key == "@route":
                return "STATE_REQUIRES_ROUTE"
            return "STATE_REQUIREMENT_MISSING"

    if "@learned" in record and record["@learned"] is not None:
        error = validate_learned(record["@learned"])
        if error:
            return error
    if "@binding" in record and record["@binding"] is not None:
        error = validate_binding(record["@binding"])
        if error:
            return error
        if not matches_binding_registry(record, binding_registry):
            return "BINDING_REGISTRY_MISMATCH"
    if "@evidence" in record and record["@evidence"] is not None:
        error = validate_evidence(record["@evidence"])
        if error:
            return error
    if "@policy" in record and record["@policy"] is not None:
        error = validate_policy_record(record["@policy"], policy_contracts)
        if error:
            return error
    if "@route" in record and record["@route"] is not None:
        error = validate_route(record["@route"], contracts)
        if error:
            return error
    if "@ui" in record and record["@ui"] is not None:
        error = validate_ui(record["@ui"])
        if error:
            return error

    evidence = record.get("@evidence")
    policy = record.get("@policy")
    if is_object(evidence) and "@confidence" in evidence:
        if not is_object(policy):
            return "CONFIDENCE_REQUIRES_POLICY"
        expected = evaluate_xcfe_confidence(
            policy,
            policy_evidence_from_fold(evidence),
            policy_contracts,
        )
        confidence_policy = contracts.get("confidence_policy", {})
        tolerance = float(confidence_policy.get("tolerance", 1e-6))
        if abs(float(evidence["@confidence"]) - expected) > tolerance:
            return "CONFIDENCE_DERIVATION_MISMATCH"

    return ""


def main() -> int:
    contracts = load_json(CONTRACTS) if CONTRACTS.exists() else {}
    binding_registry = load_json(BINDINGS) if BINDINGS.exists() else {}
    policy_contracts = load_json(XCFE_POLICY_CONTRACTS) if XCFE_POLICY_CONTRACTS.exists() else {}
    fixtures = load_json(FIXTURES) if FIXTURES.exists() else {}
    grammar = read(GRAMMAR)
    docs = "\n".join(read(path) for path in (GLOSSARY, SEMANTIC_FOLDS, SPEC))
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    check("grammar_file_present", GRAMMAR.exists())
    check("contracts_file_present", CONTRACTS.exists())
    check("fixtures_file_present", FIXTURES.exists())
    check("binding_registry_file_present", BINDINGS.exists())
    check("xcfe_policy_contracts_file_present", XCFE_POLICY_CONTRACTS.exists())
    check("xcfe_policy_contracts_type", policy_contracts.get("@type") == "khanary.xcfe_policy_contracts.v1")
    check("contracts_type", contracts.get("@type") == "khanary.fold_kernel_contracts.v1")
    check("semantic_authority_false", contracts.get("semantic_authority") is False)
    check("authority_boundary", "not_semantic_authority" in str(contracts.get("authority", "")))

    for token in GRAMMAR_TOKENS:
        check(f"grammar/has/{token}", token in grammar)
    for token in DOC_TOKENS:
        check(f"docs/has/{token}", token in docs)

    check("phase_cycle_has_six", contracts.get("phase_cycle") == ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"])
    check(
        "state_order_declared",
        contracts.get("states") == ["unbound", "observed", "learned", "bound", "validated", "autonomous"],
    )
    state_requirements = contracts.get("state_requirements", {})
    check("learned_requires_learned", state_requirements.get("learned") == ["@learned"])
    check("bound_requires_learned_binding", state_requirements.get("bound") == ["@learned", "@binding"])
    check(
        "validated_requires_evidence_route",
        state_requirements.get("validated") == ["@learned", "@binding", "@evidence", "@route"],
    )
    check(
        "autonomous_requires_policy_route",
        state_requirements.get("autonomous") == ["@learned", "@binding", "@evidence", "@policy", "@route"],
    )

    invariants = contracts.get("invariants", {})
    for invariant in ["I1", "I2", "I3", "I4", "I5", "S1", "S2", "S3", "S4", "S5", "A1", "A2", "A3", "P1", "P2"]:
        check(f"invariant_declared/{invariant}", invariant in invariants)

    for phase in contracts.get("phase_cycle", []):
        tags = contracts.get("semantic_tags_by_phase", {}).get(phase, [])
        check(f"semantic_tags/{phase}", isinstance(tags, list) and len(tags) >= 2)

    fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
    check("fixtures_have_cases", isinstance(fixture_cases, list) and len(fixture_cases) >= 20)
    pass_seen = 0
    fail_seen = 0
    for case in fixture_cases:
        if not is_object(case):
            check("fixture_case_is_object", False)
            continue
        case_id = str(case.get("id", "unnamed"))
        expected = case.get("expect")
        error = validate_record(case.get("record"), contracts, binding_registry, policy_contracts)
        actual = "PASS" if not error else "FAIL"
        if actual == "PASS":
            pass_seen += 1
        else:
            fail_seen += 1
        check(f"fixture/{case_id}/expect", actual == expected)
        if expected == "FAIL":
            check(f"fixture/{case_id}/error", error == case.get("error"))
    check("fixtures_include_valid_lifecycle_cases", pass_seen >= 6)
    check("fixtures_include_negative_cases", fail_seen >= 15)

    check("spec_lists_fold_kernel_grammar", "fold-kernel.ebnf" in read(SPEC))
    check("spec_lists_fold_kernel_checker", "check_fold_kernel.py" in read(SPEC))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.fold_kernel.verdict.v1",
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
