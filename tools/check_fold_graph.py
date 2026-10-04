#!/usr/bin/env python3
"""Validate Fold Graph recoverable contraction contracts and fixtures."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "data/grammar/fold-graph.contracts.json"
FIXTURES = ROOT / "data/conformance/fold_graph.json"
FOLD_GRAPH_DOC = ROOT / "docs/FOLD_GRAPH.md"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"
MANIFEST = ROOT / "data/manifest/schema.manifest.json"
CMAKE = ROOT / "CMakeLists.txt"

DOC_TOKENS = [
    "Fold Graph",
    "Unfold(h, τ) = { e | B_τ[h, e] ≠ 0 }",
    "𝒢_F(h, τ, Q)",
    "Handle -> Unfold(T) -> Neighborhood -> Attention -> Fold(T) -> C_T",
    "𝒢_F(h) = (V_h, E→_h, B_h, W_h, τ_h, Θ_h)",
    "W_E = strength of relationship",
    "O_E = permitted direction of traversal",
    "semantic edge orientation",
    "C6 phase-forward",
    "Fold = topology-preserving contraction with recoverable local structure",
    "Handle is not merely a summary value",
    "Unfold(Fold(𝒢_F)) ≃ 𝒢_F",
    "semantic/topological equivalence under the Fold contract",
    "Local Fold Validity + Boundary Compatibility => Composable Fold Validity",
    "Demaine",
    "Abel",
    "not a claimed equivalence",
    "Unfold does not create semantic relationships.",
    "Fold does not destroy the graph.",
    "Only the verified/authorized Xul path can persist a topology delta.",
]

GLOSSARY_TOKENS = [
    "Fold Graph",
    "𝒢_F(h)",
    "Active Fold Graph",
    "Fold contracts it back into a",
    "recoverable Handle",
]

CONTRACT_TOKENS = [
    "Fold Graph recoverable contraction",
    "Unfold(h,tau) = { e | B_tau[h,e] != 0 }",
    "G_F(h) = (V_h, E→_h, B_h, W_h, tau_h, Theta_h)",
    "O_E is permitted traversal direction; W_E is only relationship strength",
    "runtime paths satisfy semantic edge orientation plus C6 phase-forward law",
    "Unfold(Fold(G_F)) ~= G_F",
    "Local Fold Validity + Boundary Compatibility => Composable Fold Validity",
]

REQUIRED_KEYS = {
    "@type",
    "@handle",
    "@nodes",
    "@edges",
    "@incidence",
    "@weights",
    "@tau",
    "@projection",
    "@query_conditioned",
    "@active",
    "@recoverability",
    "@authority",
    "@boundary",
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def load_json(path: Path) -> Any:
    return json.loads(read(path))


def is_object(value: Any) -> bool:
    return isinstance(value, dict)


def is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def is_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def validate_graph(graph: Any, contracts: dict[str, Any]) -> str:
    if not is_object(graph):
        return "GRAPH_NOT_OBJECT"
    if graph.get("@type") != "khanary.fold_graph.v1":
        return "INVALID_TYPE"
    missing = REQUIRED_KEYS.difference(graph.keys())
    if missing:
        return "MISSING_REQUIRED_KEY"
    if not is_string(graph.get("@handle")):
        return "INVALID_HANDLE"
    nodes = graph.get("@nodes")
    if not isinstance(nodes, list) or not nodes or not all(is_string(node) for node in nodes):
        return "INVALID_NODES"
    node_set = set(nodes)
    if len(node_set) != len(nodes):
        return "DUPLICATE_NODE"
    edges = graph.get("@edges")
    if not isinstance(edges, list) or not edges:
        return "INVALID_EDGES"
    edge_ids: set[str] = set()
    for edge in edges:
        if not is_object(edge):
            return "INVALID_EDGE"
        edge_id = edge.get("@id")
        if not is_string(edge_id):
            return "INVALID_EDGE_ID"
        if edge_id in edge_ids:
            return "DUPLICATE_EDGE_ID"
        edge_ids.add(edge_id)
        if edge.get("@typed") is not True:
            return "EDGE_NOT_TYPED"
        if edge.get("@from") not in node_set or edge.get("@to") not in node_set:
            return "EDGE_ENDPOINT_UNKNOWN"
        if not is_string(edge.get("@relation")) or not str(edge.get("@relation")).startswith("@"):
            return "INVALID_RELATION"
        orientation = edge.get("@orientation")
        if not is_string(orientation):
            return "EDGE_ORIENTATION_MISSING"
        if orientation not in contracts.get("orientation_modes", []):
            return "EDGE_ORIENTATION_INVALID"
    incidence = graph.get("@incidence")
    if not isinstance(incidence, list) or not incidence:
        return "MISSING_INCIDENCE"
    for pair in incidence:
        if not isinstance(pair, list) or len(pair) != 2:
            return "INVALID_INCIDENCE"
        if pair[0] not in node_set or pair[1] not in edge_ids:
            return "INCIDENCE_UNKNOWN_REFERENCE"
    weights = graph.get("@weights")
    if not is_object(weights) or not is_object(weights.get("@pattern_strength")):
        return "INVALID_WEIGHTS"
    for edge_id in edge_ids:
        if edge_id not in weights["@pattern_strength"]:
            return "MISSING_WEIGHT"
        if not is_probability(weights["@pattern_strength"][edge_id]):
            return "WEIGHT_OUT_OF_RANGE"
    if not is_string(graph.get("@tau")):
        return "INVALID_TAU"
    if graph.get("@projection") not in contracts.get("projection_modes", []):
        return "INVALID_PROJECTION"
    if not isinstance(graph.get("@query_conditioned"), bool) or not isinstance(graph.get("@active"), bool):
        return "INVALID_ACTIVE_FLAGS"
    if graph.get("@query_conditioned") is True or graph.get("@active") is True:
        gate = graph.get("@query_gate")
        if not is_object(gate) or gate.get("@operator") != "Gate_Q" or not is_object(gate.get("@mask")):
            return "INVALID_ACTIVE_GATE"
    recoverability = graph.get("@recoverability")
    if not is_object(recoverability):
        return "INVALID_RECOVERABILITY"
    if recoverability.get("@equivalent") is not True or recoverability.get("@lossless_canonical") is not True:
        return "RECOVERABILITY_BROKEN"
    if "Handle(" not in str(recoverability.get("@fold", "")) or "G_F(" not in str(recoverability.get("@unfold", "")):
        return "RECOVERABILITY_BROKEN"
    authority = graph.get("@authority")
    required_authority = contracts.get("authority_required", {})
    if not is_object(authority):
        return "INVALID_AUTHORITY"
    if authority.get("@semantic_authority") is not required_authority.get("semantic_authority"):
        return "SEMANTIC_AUTHORITY_CLAIMED"
    if authority.get("@xul_required_for_delta_B") is not required_authority.get("xul_required_for_delta_B"):
        return "XUL_BOUNDARY_MISSING"
    boundary = graph.get("@boundary")
    if not is_object(boundary):
        return "INVALID_BOUNDARY"
    if boundary.get("@local_valid") is not True:
        return "LOCAL_VALIDITY_FAILED"
    if boundary.get("@boundary_compatible") is not True:
        return "BOUNDARY_INCOMPATIBLE"
    if boundary.get("@delta_B") is True:
        return "DELTA_B_REQUIRES_XUL_EVENT"
    return ""


def main() -> int:
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    try:
        contracts = load_json(CONTRACTS) if CONTRACTS.exists() else {}
        fixtures = load_json(FIXTURES) if FIXTURES.exists() else {}
        doc = read(FOLD_GRAPH_DOC)
        glossary = read(GLOSSARY)
        spec = read(SPEC)
        manifest = load_json(MANIFEST) if MANIFEST.exists() else {}
        cmake = read(CMAKE)

        check("contracts_file_present", CONTRACTS.exists())
        check("fixtures_file_present", FIXTURES.exists())
        check("docs_fold_graph_present", FOLD_GRAPH_DOC.exists())
        check("contracts_type", contracts.get("@type") == "khanary.fold_graph_contracts.v1")
        check("semantic_authority_false", contracts.get("semantic_authority") is False)
        check("source_topology", contracts.get("source_topology") == "data/grammar/fold_graph.xml")
        check("fold_space_contract_ref", contracts.get("fold_space_contract") == "data/grammar/fold-space.contracts.json")
        check("fold_event_contract_ref", contracts.get("fold_event_contract") == "data/grammar/fold-event.contracts.json")
        law = contracts.get("law", {})
        for token in CONTRACT_TOKENS:
            check(f"contracts/has/{token}", token in json.dumps(contracts, ensure_ascii=False))
        check("law_unfold_no_create", law.get("unfold_does_not_create_relationships") is True)
        check("law_fold_no_destroy", law.get("fold_does_not_destroy_graph") is True)
        check("shape_fields", contracts.get("shape_fields") == ["V_h", "E→_h", "B_h", "W_h", "tau_h", "Theta_h"])
        check("orientation_modes", contracts.get("orientation_modes") == ["forward", "reverse", "bidirectional", "blocked"])
        check("required_record_keys", set(contracts.get("required_record_keys", [])) == REQUIRED_KEYS)
        check("authority_required", contracts.get("authority_required", {}).get("semantic_authority") is False)
        check("xul_required", contracts.get("authority_required", {}).get("xul_required_for_delta_B") is True)
        for invariant in [f"FG{i}" for i in range(1, 15)]:
            check(f"invariant_declared/{invariant}", invariant in contracts.get("invariants", {}))
        for token in DOC_TOKENS:
            check(f"docs/has/{token}", token in doc)
        for token in GLOSSARY_TOKENS:
            check(f"glossary/has/{token}", token in glossary)
        check("spec_lists_contract", "fold-graph.contracts.json" in spec)
        check("spec_lists_fixture", "fold_graph.json" in spec)
        check("spec_lists_checker", "check_fold_graph.py" in spec)
        check("manifest_lists_contract", any("fold-graph.contracts.json" in item for item in manifest.get("grammar:fold_graph_contract", [])))
        check("manifest_lists_checker", any("check_fold_graph.py" in item for item in manifest.get("grammar:fold_graph_contract", [])))
        check("cmake_registers_test", "KHANARY_FOLD_GRAPH_TEST" in cmake)

        fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
        check("fixtures_have_cases", isinstance(fixture_cases, list) and len(fixture_cases) >= 12)
        pass_seen = 0
        fail_seen = 0
        for row in fixture_cases:
            if not is_object(row):
                check("fixture_case_is_object", False)
                continue
            case_id = str(row.get("id", "unnamed"))
            error = validate_graph(row.get("graph"), contracts)
            actual = "PASS" if not error else "FAIL"
            if actual == "PASS":
                pass_seen += 1
            else:
                fail_seen += 1
            check(f"fixture/{case_id}/expect", actual == row.get("expect"))
            if row.get("expect") == "FAIL":
                check(f"fixture/{case_id}/error", error == row.get("error"))
        check("fixtures_include_valid_graphs", pass_seen >= 2)
        check("fixtures_include_negative_graphs", fail_seen >= 10)
    except Exception as exc:
        cases.append(("exception/" + str(exc), False))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.fold_graph.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "contracts": str(CONTRACTS.relative_to(ROOT)),
        "fixtures": str(FIXTURES.relative_to(ROOT)),
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
