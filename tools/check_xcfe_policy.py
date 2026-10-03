#!/usr/bin/env python3
"""Validate XCFE policy grammar, expression language, and control semantics."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "data/grammar/xcfe-policy.ebnf"
CONTRACTS = ROOT / "data/grammar/xcfe-policy.contracts.json"
FIXTURES = ROOT / "data/conformance/xcfe_policy.json"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SEMANTIC_FOLDS = ROOT / "docs/SEMANTIC_FOLDS.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"

POLICY_KEYS = {"@ref", "@evidence", "@weights", "@thresholds", "@routes", "@guard"}
EVIDENCE_SPEC_KEYS = {"@inputs", "@derive"}
DERIVE_KEYS = {"@confidence"}
WEIGHTS_KEYS = {"@alpha", "@beta", "@gamma", "@delta", "@epsilon"}
THRESHOLD_KEYS = {"@autonomy", "@fallback", "@failure_limit", "@hysteresis"}
ROUTES_KEYS = {"@autonomous", "@validated", "@default"}
GUARD_KEYS = {"@when", "@require", "@reject"}

EVIDENCE_FIELDS = {"observations", "support", "exceptions", "validated", "failures"}
WEIGHT_VARS = {"alpha", "beta", "gamma", "delta", "epsilon"}
GUARD_EXTRA_VARS = {"confidence"}
EVENT_NAMES = {"contract_failed", "authorize", "uncertainty"}
FUNCTIONS = {"clamp", "min", "max", "sqrt", "log", "exp"}

AT_KEY_RE = re.compile(r"^@[A-Za-z][A-Za-z0-9_]*$")
POLICY_REF_RE = re.compile(r"^xcfe://[A-Za-z0-9._:-]+/[A-Za-z0-9._:-]+(@[A-Za-z0-9._:+-]+)?$")
TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|(?P<op><=|>=|[+\-*/(),<>=]))"
)

GRAMMAR_TOKENS = [
    "XCFE Policy Grammar",
    "Policy != Control != Execution",
    "Policy          ::= \"{\" PolicyMember",
    "EvidenceKey     ::= '\"@evidence\"'",
    "Expr            ::= Term",
    "Function        ::= \"clamp\" | \"min\" | \"max\" | \"sqrt\" | \"log\" | \"exp\"",
    "W1  ::= alpha + beta + gamma + delta = 1",
    "W10 ::= vars(guards) subset_of inputs union { confidence }",
    "ControlFunction ::= \"Π\"",
    "TypeJudgment ::= \"⊢\" PolicyADT \":\" \"Policy\"",
]

DOC_TOKENS = [
    "XCFE Policy",
    "Policy != Control != Execution",
    "Pi(Evidence, State, Event) -> (State, Route)",
    "policy is data",
    "XCFE is the evaluator",
]


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


class ExprError(ValueError):
    pass


class Expr:
    def variables(self) -> set[str]:
        raise NotImplementedError

    def eval(self, env: dict[str, float]) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class NumberExpr(Expr):
    value: float

    def variables(self) -> set[str]:
        return set()

    def eval(self, env: dict[str, float]) -> float:
        return self.value


@dataclass(frozen=True)
class VarExpr(Expr):
    name: str

    def variables(self) -> set[str]:
        return {self.name}

    def eval(self, env: dict[str, float]) -> float:
        return float(env.get(self.name, 0.0))


@dataclass(frozen=True)
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr

    def variables(self) -> set[str]:
        return self.left.variables() | self.right.variables()

    def eval(self, env: dict[str, float]) -> float:
        lhs = safe_number(self.left.eval(env))
        rhs = safe_number(self.right.eval(env))
        if self.op == "+":
            return lhs + rhs
        if self.op == "-":
            return lhs - rhs
        if self.op == "*":
            return lhs * rhs
        if self.op == "/":
            if rhs == 0:
                if lhs == 0:
                    return 0.0
                return math.copysign(math.inf, lhs)
            return lhs / rhs
        raise ExprError(f"unknown binary op {self.op}")


@dataclass(frozen=True)
class UnaryExpr(Expr):
    op: str
    expr: Expr

    def variables(self) -> set[str]:
        return self.expr.variables()

    def eval(self, env: dict[str, float]) -> float:
        value = safe_number(self.expr.eval(env))
        if self.op == "+":
            return value
        if self.op == "-":
            return -value
        raise ExprError(f"unknown unary op {self.op}")


@dataclass(frozen=True)
class FuncExpr(Expr):
    name: str
    args: tuple[Expr, ...]

    def variables(self) -> set[str]:
        out: set[str] = set()
        for arg in self.args:
            out |= arg.variables()
        return out

    def eval(self, env: dict[str, float]) -> float:
        values = [safe_number(arg.eval(env)) for arg in self.args]
        if self.name == "clamp":
            if len(values) != 3:
                raise ExprError("clamp requires 3 args")
            value, lo, hi = values
            if math.isnan(value):
                value = 0.0
            return max(lo, min(hi, value))
        if self.name == "min":
            if len(values) != 2:
                raise ExprError("min requires 2 args")
            return min(values)
        if self.name == "max":
            if len(values) != 2:
                raise ExprError("max requires 2 args")
            return max(values)
        if self.name == "sqrt":
            if len(values) != 1:
                raise ExprError("sqrt requires 1 arg")
            return math.sqrt(values[0]) if values[0] >= 0 else 0.0
        if self.name == "log":
            if len(values) != 1:
                raise ExprError("log requires 1 arg")
            return math.log(values[0]) if values[0] > 0 else 0.0
        if self.name == "exp":
            if len(values) != 1:
                raise ExprError("exp requires 1 arg")
            try:
                return math.exp(values[0])
            except OverflowError:
                return math.inf
        raise ExprError(f"unknown function {self.name}")


def safe_number(value: float) -> float:
    value = float(value)
    if math.isnan(value):
        return 0.0
    return value


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        match = TOKEN_RE.match(text, pos)
        if not match:
            raise ExprError(f"invalid token near {text[pos:pos + 16]!r}")
        pos = match.end()
        if match.group("num") is not None:
            tokens.append(Token("num", match.group("num")))
        elif match.group("ident") is not None:
            tokens.append(Token("ident", match.group("ident")))
        else:
            tokens.append(Token("op", match.group("op")))
    tokens.append(Token("eof", ""))
    return tokens


class Parser:
    def __init__(self, text: str):
        self.tokens = tokenize(text)
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def eat(self, value: str | None = None, kind: str | None = None) -> Token:
        token = self.current()
        if value is not None and token.value != value:
            raise ExprError(f"expected {value!r}, got {token.value!r}")
        if kind is not None and token.kind != kind:
            raise ExprError(f"expected kind {kind!r}, got {token.kind!r}")
        self.pos += 1
        return token

    def parse_expr(self) -> Expr:
        expr = self.parse_term()
        while self.current().value in {"+", "-"}:
            op = self.eat().value
            expr = BinaryExpr(op, expr, self.parse_term())
        return expr

    def parse_term(self) -> Expr:
        expr = self.parse_factor()
        while self.current().value in {"*", "/"}:
            op = self.eat().value
            expr = BinaryExpr(op, expr, self.parse_factor())
        return expr

    def parse_factor(self) -> Expr:
        token = self.current()
        if token.value in {"+", "-"}:
            return UnaryExpr(self.eat().value, self.parse_factor())
        if token.kind == "num":
            return NumberExpr(float(self.eat(kind="num").value))
        if token.kind == "ident":
            name = self.eat(kind="ident").value
            if self.current().value == "(":
                self.eat("(")
                args: list[Expr] = []
                if self.current().value != ")":
                    args.append(self.parse_expr())
                    while self.current().value == ",":
                        self.eat(",")
                        args.append(self.parse_expr())
                self.eat(")")
                if name not in FUNCTIONS:
                    raise ExprError(f"unknown function {name}")
                return FuncExpr(name, tuple(args))
            return VarExpr(name)
        if token.value == "(":
            self.eat("(")
            expr = self.parse_expr()
            self.eat(")")
            return expr
        raise ExprError(f"unexpected token {token.value!r}")


def parse_expr(text: str) -> Expr:
    parser = Parser(text)
    expr = parser.parse_expr()
    if parser.current().kind != "eof":
        raise ExprError(f"unexpected trailing token {parser.current().value!r}")
    return expr


def parse_condition(text: str) -> tuple[Expr, str, Expr]:
    depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0:
            for op in ("<=", ">=", "<", ">", "="):
                if text.startswith(op, i):
                    lhs = parse_expr(text[:i].strip())
                    rhs = parse_expr(text[i + len(op):].strip())
                    return lhs, op, rhs
    raise ExprError("condition missing comparison operator")


def eval_condition(text: str, env: dict[str, float]) -> bool:
    lhs, op, rhs = parse_condition(text)
    a = lhs.eval(env)
    b = rhs.eval(env)
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == "=":
        return abs(a - b) <= 1e-9
    if op == ">=":
        return a >= b
    if op == ">":
        return a > b
    raise ExprError(f"unknown comparison {op}")


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


def is_probability(value: Any) -> bool:
    return is_number(value) and 0.0 <= float(value) <= 1.0


def has_only_at_extension_keys(obj: dict[str, Any], known: set[str]) -> bool:
    for key in obj:
        if key in known:
            continue
        if not AT_KEY_RE.match(key):
            return False
    return True


def normalize_evidence(raw: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in EVIDENCE_FIELDS:
        value = raw.get(field, raw.get("@" + field, 0))
        out[field] = float(value)
    return out


def env_for_policy(policy: dict[str, Any], evidence: dict[str, Any], confidence: float | None = None) -> dict[str, float]:
    env = normalize_evidence(evidence)
    weights = policy.get("@weights")
    if is_object(weights):
        env.update({
            "alpha": float(weights.get("@alpha", 0.0)),
            "beta": float(weights.get("@beta", 0.0)),
            "gamma": float(weights.get("@gamma", 0.0)),
            "delta": float(weights.get("@delta", 0.0)),
            "epsilon": float(weights.get("@epsilon", 0.0)),
        })
    if confidence is not None:
        env["confidence"] = float(confidence)
    return env


def derive_expression(policy: dict[str, Any], contracts: dict[str, Any]) -> str:
    evidence_spec = policy.get("@evidence", {})
    if is_object(evidence_spec):
        derive = evidence_spec.get("@derive")
        if is_object(derive) and is_string(derive.get("@confidence")):
            return str(derive["@confidence"])
    return str(contracts.get("default_derive", ""))


def evaluate_confidence(policy: dict[str, Any], evidence: dict[str, Any], contracts: dict[str, Any] | None = None) -> float:
    if contracts is None:
        contracts = load_json(CONTRACTS)
    expr = parse_expr(derive_expression(policy, contracts))
    value = safe_number(expr.eval(env_for_policy(policy, evidence)))
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    if math.isnan(value):
        return 0.0
    return value


def expression_variables(text: str) -> set[str]:
    return parse_expr(text).variables()


def validate_policy_record(policy: Any, contracts: dict[str, Any] | None = None) -> str:
    if contracts is None:
        contracts = load_json(CONTRACTS)
    if not is_object(policy):
        return "POLICY_NOT_OBJECT"
    if not has_only_at_extension_keys(policy, POLICY_KEYS):
        return "INVALID_MEMBER_KEY"
    if not is_string(policy.get("@ref")):
        return "MISSING_POLICY_REF"
    if not POLICY_REF_RE.match(policy["@ref"]):
        return "INVALID_POLICY_REF"

    evidence_spec = policy.get("@evidence")
    if not is_object(evidence_spec):
        return "MISSING_EVIDENCE_SPEC"
    if not has_only_at_extension_keys(evidence_spec, EVIDENCE_SPEC_KEYS):
        return "INVALID_EVIDENCE_SPEC_KEY"
    inputs = evidence_spec.get("@inputs")
    if not isinstance(inputs, list) or not inputs:
        return "MISSING_EVIDENCE_INPUTS"
    if any(field not in EVIDENCE_FIELDS for field in inputs):
        return "INVALID_EVIDENCE_INPUT"
    input_set = set(inputs)

    derive = evidence_spec.get("@derive")
    derive_text = ""
    if derive is not None:
        if not is_object(derive) or set(derive.keys()) != DERIVE_KEYS or not is_string(derive.get("@confidence")):
            return "INVALID_DERIVE_SPEC"
        derive_text = derive["@confidence"]
        try:
            vars_seen = expression_variables(derive_text)
        except ExprError:
            return "INVALID_DERIVE_EXPR"
        evidence_vars = vars_seen & EVIDENCE_FIELDS
        if not evidence_vars.issubset(input_set):
            return "DERIVE_REFERENCES_UNDECLARED_INPUT"
        if "confidence" in vars_seen:
            return "DERIVE_REFERENCES_CONFIDENCE"
        unknown = vars_seen - EVIDENCE_FIELDS - WEIGHT_VARS
        if unknown:
            return "INVALID_DERIVE_VAR"
    else:
        derive_text = str(contracts.get("default_derive", ""))

    weights = policy.get("@weights")
    weight_vars_needed = bool((expression_variables(derive_text) & WEIGHT_VARS) if derive_text else False)
    if weights is None and weight_vars_needed:
        return "WEIGHTS_REQUIRED"
    if weights is not None:
        if not is_object(weights) or set(weights.keys()) != WEIGHTS_KEYS:
            return "INVALID_WEIGHTS"
        for key in ("@alpha", "@beta", "@gamma", "@delta"):
            if not is_number(weights.get(key)) or float(weights[key]) < 0:
                return "NEGATIVE_WEIGHT"
        total = sum(float(weights[key]) for key in ("@alpha", "@beta", "@gamma", "@delta"))
        if abs(total - 1.0) > 1e-6:
            return "WEIGHTS_NOT_NORMALIZED"
        if not is_number(weights.get("@epsilon")) or float(weights["@epsilon"]) <= 0:
            return "EPSILON_NONPOSITIVE"

    thresholds = policy.get("@thresholds")
    if not is_object(thresholds) or set(thresholds.keys()) != THRESHOLD_KEYS:
        return "INVALID_THRESHOLDS"
    if not is_probability(thresholds.get("@autonomy")) or not is_probability(thresholds.get("@fallback")):
        return "INVALID_THRESHOLDS"
    if not is_int(thresholds.get("@failure_limit")) or int(thresholds["@failure_limit"]) < 0:
        return "INVALID_FAILURE_LIMIT"
    if not is_probability(thresholds.get("@hysteresis")):
        return "INVALID_HYSTERESIS"
    if float(thresholds["@fallback"]) > float(thresholds["@autonomy"]):
        return "THRESHOLD_ORDER"
    if float(thresholds["@hysteresis"]) > (float(thresholds["@autonomy"]) - float(thresholds["@fallback"])):
        return "HYSTERESIS_BAND"

    routes = policy.get("@routes")
    if not is_object(routes) or set(routes.keys()) != ROUTES_KEYS:
        return "ROUTES_NOT_TOTAL"
    if any(target not in contracts.get("route_targets", []) for target in routes.values()):
        return "INVALID_ROUTE_TARGET"

    guards = policy.get("@guard", [])
    if guards is None:
        guards = []
    if not isinstance(guards, list):
        return "INVALID_GUARD"
    for guard in guards:
        if not is_object(guard) or not has_only_at_extension_keys(guard, GUARD_KEYS):
            return "INVALID_GUARD"
        if not any(key in guard for key in GUARD_KEYS):
            return "INVALID_GUARD"
        for key, value in guard.items():
            if key == "@reject" and isinstance(value, str) and value in EVENT_NAMES:
                continue
            if not is_string(value):
                return "INVALID_GUARD_CONDITION"
            try:
                lhs, _, rhs = parse_condition(value)
                vars_seen = lhs.variables() | rhs.variables()
            except ExprError:
                return "INVALID_GUARD_CONDITION"
            allowed = input_set | GUARD_EXTRA_VARS
            if not vars_seen.issubset(allowed):
                return "GUARD_REFERENCES_UNDECLARED_VAR"

    return ""


def transition(policy: dict[str, Any], evidence: dict[str, Any], state: str, event: str,
               contracts: dict[str, Any] | None = None) -> tuple[str, str, float]:
    if contracts is None:
        contracts = load_json(CONTRACTS)
    confidence = evaluate_confidence(policy, evidence, contracts)
    thresholds = policy["@thresholds"]
    routes = policy["@routes"]
    failures = int(evidence.get("failures", evidence.get("@failures", 0)))
    env = env_for_policy(policy, evidence, confidence)

    for guard in policy.get("@guard", []) or []:
        if "@reject" in guard:
            reject = guard["@reject"]
            if reject == event:
                return "learned", routes["@default"], confidence
            if isinstance(reject, str) and reject not in EVENT_NAMES and eval_condition(reject, env):
                return "learned", routes["@default"], confidence
        if "@require" in guard and not eval_condition(guard["@require"], env):
            return "learned", routes["@default"], confidence

    autonomy = float(thresholds["@autonomy"])
    fallback = float(thresholds["@fallback"])
    failure_limit = int(thresholds["@failure_limit"])

    if event == "contract_failed":
        return "learned", routes["@default"], confidence
    if state == "validated" and confidence >= autonomy and failures <= failure_limit:
        return "autonomous", routes["@autonomous"], confidence
    if state == "autonomous" and confidence >= fallback and failures <= failure_limit:
        return "autonomous", routes["@autonomous"], confidence
    if state == "autonomous" and (confidence < fallback or failures > failure_limit):
        return "bound", routes["@default"], confidence
    if state == "validated" and confidence >= fallback:
        return "validated", routes["@validated"], confidence
    return state, routes["@default"], confidence


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
    check("contracts_type", contracts.get("@type") == "khanary.xcfe_policy_contracts.v1")
    check("semantic_authority_false", contracts.get("semantic_authority") is False)
    check("policy_is_data", contracts.get("policy_is_data") is True)
    check("evaluator_is_xcfe", contracts.get("evaluator") == "XCFE")
    for token in GRAMMAR_TOKENS:
        check(f"grammar/has/{token}", token in grammar)
    for token in DOC_TOKENS:
        check(f"docs/has/{token}", token in docs)
    for invariant in [f"W{i}" for i in range(1, 11)]:
        check(f"invariant_declared/{invariant}", invariant in contracts.get("invariants", {}))
    check("default_derive_declared", "clamp(" in contracts.get("default_derive", ""))
    check("transition_contract_declared", "validated_to_autonomous" in contracts.get("transition", {}))

    fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
    check("fixtures_have_cases", isinstance(fixture_cases, list) and len(fixture_cases) >= 18)
    pass_seen = 0
    fail_seen = 0
    eval_seen = 0
    for case in fixture_cases:
        if not is_object(case):
            check("fixture_case_is_object", False)
            continue
        case_id = str(case.get("id", "unnamed"))
        expected = case.get("expect")
        policy = case.get("policy")
        error = validate_policy_record(policy, contracts)
        actual = "PASS" if not error else "FAIL"
        if actual == "PASS":
            pass_seen += 1
        else:
            fail_seen += 1
        check(f"fixture/{case_id}/expect", actual == expected)
        if expected == "FAIL":
            check(f"fixture/{case_id}/error", error == case.get("error"))
        if expected == "PASS" and "eval" in case:
            eval_seen += 1
            result_state, result_route, result_confidence = transition(
                policy,
                case["eval"]["evidence"],
                case["eval"]["state"],
                case["eval"]["event"],
                contracts,
            )
            expected_eval = case["eval"]["expect"]
            check(f"fixture/{case_id}/eval_state", result_state == expected_eval.get("state"))
            check(f"fixture/{case_id}/eval_route", result_route == expected_eval.get("route"))
            if "confidence" in expected_eval:
                check(
                    f"fixture/{case_id}/eval_confidence",
                    abs(result_confidence - float(expected_eval["confidence"])) <= 1e-6,
                )
    check("fixtures_include_valid_policies", pass_seen >= 5)
    check("fixtures_include_negative_policies", fail_seen >= 12)
    check("fixtures_include_eval_cases", eval_seen >= 4)
    check("spec_lists_xcfe_policy_grammar", "xcfe-policy.ebnf" in read(SPEC))
    check("spec_lists_xcfe_policy_checker", "check_xcfe_policy.py" in read(SPEC))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.xcfe_policy.verdict.v1",
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
