#!/usr/bin/env python3
"""Validate the KHANARY public API gateway contract and implementation surface."""
from __future__ import annotations

import copy
import json
import py_compile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "data/grammar/api-gateway.contracts.json"
FIXTURES = ROOT / "data/conformance/api_gateway.json"
DOC = ROOT / "docs/API_GATEWAY.md"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
KQL_DOC = ROOT / "docs/KQL.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"
CMAKE = ROOT / "CMakeLists.txt"
MANIFEST = ROOT / "data/manifest/schema.manifest.json"
GATEWAY = ROOT / "tools/khanary_completion_api.py"
LAUNCHER = ROOT / "KHANARY.bat"
PORTS = ROOT / "data/ports.json"

REQUIRED_ENDPOINTS = {
    ("GET", "/"),
    ("GET", "/v1"),
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/code/modify"),
    ("POST", "/v1/images/generations"),
    ("POST", "/v1/learn/events"),
    ("POST", "/rpc"),
}

REQUIRED_RPC_METHODS = {
    "key_query",
    "resolve_manifest",
    "fold_query",
    "fold_graph_parse",
    "micronaut_dispatch",
    "micronaut_transpile",
    "journal_semantic_envelope",
    "journal_hyperbolic_event",
    "tensor_register",
    "tensor_info",
    "tensor_list",
    "tensor_stats",
    "tensor_gemm",
    "list_keys",
    "record_key",
    "infer",
}

DOC_TOKENS = [
    "https://api.kuhul.dev:443",
    "http://127.0.0.1:17892",
    "text completions",
    "code-modification proposal",
    "image-generation request",
    "learning_commenced",
    "user_scoped_mesh_kql",
    "Ch'en/XCFE/Xul",
    "semantic_authority:false",
    "topology_mutation:false",
    "commit_authorized:false",
]

GATEWAY_TOKENS = [
    'PUBLIC_API_BASE_URL = "https://api.kuhul.dev"',
    "PUBLIC_API_PORT = 443",
    "public_port",
    "/v1/chat/completions",
    "/v1/code/modify",
    "/v1/images/generations",
    "/v1/learn/events",
    "/rpc",
    "LEARNING_COMMENCED_STATUS",
    "code.modification.proposal",
    "image.generation.request",
    "learning.event",
    "semantic_authority",
    "topology_mutation",
    "commit_authorized",
    "Ch'en/XCFE/Xul",
]

LAUNCHER_TOKENS = [
    "KHANARY_PUBLIC_API_BASE_URL=https://api.kuhul.dev",
    "KHANARY_PUBLIC_API_PORT=443",
    "--public-base-url",
    "--public-port",
    "public_api",
    "public edge",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def load_json(path: Path) -> Any:
    return json.loads(read(path))


def is_object(value: Any) -> bool:
    return isinstance(value, dict)


def endpoint_key(endpoint: dict[str, Any]) -> tuple[str, str]:
    return (str(endpoint.get("method", "")).upper(), str(endpoint.get("path", "")))


def endpoint_map(contracts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for endpoint in contracts.get("endpoints", []):
        if is_object(endpoint):
            out[str(endpoint.get("id", ""))] = endpoint
    return out


def merged_contract(contracts: dict[str, Any], patch: Any) -> dict[str, Any]:
    result = copy.deepcopy(contracts)
    if is_object(patch):
        result.update(patch)
    return result


def check_contract_root(contracts: dict[str, Any]) -> str:
    if contracts.get("@type") != "khanary.api_gateway_contracts.v1":
        return "CONTRACT_TYPE_INVALID"
    if contracts.get("semantic_authority") is not False:
        return "CONTRACT_CLAIMS_AUTHORITY"
    if contracts.get("public_origin") != "https://api.kuhul.dev":
        return "PUBLIC_ORIGIN_INVALID"
    if contracts.get("public_scheme") != "https":
        return "PUBLIC_SCHEME_INVALID"
    if contracts.get("public_host") != "api.kuhul.dev":
        return "PUBLIC_HOST_INVALID"
    if contracts.get("public_port") != 443:
        return "PUBLIC_PORT_NOT_443"
    if contracts.get("public_endpoint") != "https://api.kuhul.dev:443":
        return "PUBLIC_ENDPOINT_INVALID"
    if contracts.get("launcher") != "KHANARY.bat":
        return "LAUNCHER_INVALID"
    if contracts.get("ports_registry") != "data/ports.json":
        return "PORTS_REGISTRY_INVALID"
    if contracts.get("local_gateway") != "tools/khanary_completion_api.py":
        return "LOCAL_GATEWAY_INVALID"
    if contracts.get("local_default_port") != 17892:
        return "LOCAL_PORT_INVALID"
    if contracts.get("learning_status") != "learning_commenced":
        return "LEARNING_STATUS_INVALID"
    flow = contracts.get("flow")
    if not isinstance(flow, list) or flow[:2] != ["api.kuhul.dev:443", "session/auth"]:
        return "FLOW_INVALID"
    if "user_scoped_mesh_kql" not in flow:
        return "FLOW_MISSING_USER_SCOPED_KQL"
    if flow[-1] != "Ch'en/XCFE/Xul":
        return "FLOW_GATE_INVALID"
    return ""


def check_endpoint(endpoint: Any, contracts: dict[str, Any]) -> str:
    root_error = check_contract_root(contracts)
    if root_error:
        return root_error
    if not is_object(endpoint):
        return "ENDPOINT_NOT_OBJECT"
    endpoints = endpoint_map(contracts)
    endpoint_id = str(endpoint.get("id", ""))
    declared = endpoints.get(endpoint_id)
    if not declared:
        return "ENDPOINT_NOT_DECLARED"
    if endpoint_key(endpoint) != endpoint_key(declared):
        return "ENDPOINT_SHAPE_MISMATCH"
    if endpoint.get("surface") != declared.get("surface"):
        return "ENDPOINT_SURFACE_MISMATCH"
    if endpoint.get("semantic_authority") is not False:
        return "ENDPOINT_CLAIMS_AUTHORITY"
    public_post = endpoint.get("method") == "POST"
    if public_post and endpoint.get("requires_session") is not True:
        return "PUBLIC_ENDPOINT_REQUIRES_SESSION"
    if endpoint.get("topology_mutation") is True:
        return "ENDPOINT_MUTATES_TOPOLOGY"
    if endpoint.get("commit_authorized") is True:
        return "ENDPOINT_AUTHORIZES_COMMIT"
    if public_post and endpoint.get("next_gate") != "Ch'en/XCFE/Xul":
        return "NEXT_GATE_INVALID"

    if endpoint_id == "text_completions":
        if endpoint.get("operation") != "key_query":
            return "TEXT_COMPLETION_OPERATION_INVALID"
        if endpoint.get("emits_learning_event") is not True:
            return "LEARNING_EVENT_NOT_EMITTED"
    if endpoint_id == "code_modification":
        if endpoint.get("object") != "code.modification.proposal":
            return "CODE_ENDPOINT_OBJECT_INVALID"
        if endpoint.get("applies_patch") is not False:
            return "CODE_ENDPOINT_APPLIES_PATCH"
        if endpoint.get("emits_learning_event") is not True:
            return "LEARNING_EVENT_NOT_EMITTED"
    if endpoint_id == "image_generation":
        if endpoint.get("object") != "image.generation.request":
            return "IMAGE_ENDPOINT_OBJECT_INVALID"
        if endpoint.get("emits_learning_event") is not True:
            return "LEARNING_EVENT_NOT_EMITTED"
    if endpoint_id == "learning_events":
        if endpoint.get("object") != "learning.event":
            return "LEARNING_ENDPOINT_OBJECT_INVALID"
        if endpoint.get("status") != "learning_commenced":
            return "LEARNING_STATUS_INVALID"
    if endpoint_id == "rpc_bridge":
        if endpoint.get("allowlisted_only") is not True:
            return "RPC_BRIDGE_NOT_ALLOWLISTED"
    return ""


def check_rpc_method(method: Any, contracts: dict[str, Any]) -> str:
    root_error = check_contract_root(contracts)
    if root_error:
        return root_error
    allowlist = set(contracts.get("rpc_bridge", {}).get("allowlist", []))
    if not isinstance(method, str) or not method:
        return "RPC_METHOD_INVALID"
    if method not in allowlist:
        return "RPC_METHOD_NOT_ALLOWLISTED"
    return ""


def main() -> int:
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    try:
        contracts = load_json(CONTRACTS) if CONTRACTS.exists() else {}
        fixtures = load_json(FIXTURES) if FIXTURES.exists() else {}
        manifest = load_json(MANIFEST) if MANIFEST.exists() else {}
        docs = "\n".join(read(path) for path in (DOC, GLOSSARY, KQL_DOC, SPEC))
        gateway_src = read(GATEWAY)
        launcher_src = read(LAUNCHER)
        ports = load_json(PORTS) if PORTS.exists() else {}
        spec = read(SPEC)
        cmake = read(CMAKE)

        check("contracts_file_present", CONTRACTS.exists())
        check("fixtures_file_present", FIXTURES.exists())
        check("docs_file_present", DOC.exists())
        check("gateway_file_present", GATEWAY.exists())
        check("launcher_file_present", LAUNCHER.exists())
        check("ports_file_present", PORTS.exists())
        check("contract_root_valid", check_contract_root(contracts) == "")
        check("contract_public_port_443", contracts.get("public_port") == 443)
        check("contract_public_endpoint_443", contracts.get("public_endpoint") == "https://api.kuhul.dev:443")
        check("contract_local_port_17892", contracts.get("local_default_port") == 17892)
        check("learning_status_declared", contracts.get("learning_status") == "learning_commenced")

        endpoints = contracts.get("endpoints", [])
        endpoint_keys = {endpoint_key(endpoint) for endpoint in endpoints if is_object(endpoint)}
        check("required_endpoint_set", endpoint_keys == REQUIRED_ENDPOINTS)
        check("all_endpoint_authority_false", all(endpoint.get("semantic_authority") is False for endpoint in endpoints if is_object(endpoint)))
        check("all_post_endpoints_session_scoped", all(endpoint.get("requires_session") is True for endpoint in endpoints if is_object(endpoint) and endpoint.get("method") == "POST"))
        check("all_post_endpoints_next_gate", all(endpoint.get("next_gate") == "Ch'en/XCFE/Xul" for endpoint in endpoints if is_object(endpoint) and endpoint.get("method") == "POST"))
        check("all_post_endpoints_no_commit", all(endpoint.get("commit_authorized") is False for endpoint in endpoints if is_object(endpoint) and endpoint.get("method") == "POST"))
        check("all_post_endpoints_no_topology_mutation", all(endpoint.get("topology_mutation") is False for endpoint in endpoints if is_object(endpoint) and endpoint.get("method") == "POST"))
        check("rpc_allowlist_complete", set(contracts.get("rpc_bridge", {}).get("allowlist", [])) == REQUIRED_RPC_METHODS)

        invariants = contracts.get("invariants", {})
        for invariant in [f"API{i}" for i in range(1, 13)]:
            check(f"invariant_declared/{invariant}", invariant in invariants)

        for token in DOC_TOKENS:
            check(f"docs/has/{token}", token in docs)
        for token in GATEWAY_TOKENS:
            check(f"gateway/has/{token}", token in gateway_src)
        for token in LAUNCHER_TOKENS:
            check(f"launcher/has/{token}", token in launcher_src)

        public_api = ports.get("public_api") if is_object(ports) else {}
        abstract_api = ports.get("abstract_ports", {}).get("api") if is_object(ports.get("abstract_ports")) else {}
        check("ports_public_api_base", is_object(public_api) and public_api.get("base_url") == "https://api.kuhul.dev")
        check("ports_public_api_port", is_object(public_api) and public_api.get("port") == 443)
        check("ports_public_api_non_authority", is_object(public_api) and "not semantic authority" in str(public_api.get("authority", "")))
        check("ports_abstract_api_points_to_completion", is_object(abstract_api) and abstract_api.get("port_ref") == "completion")
        check("ports_abstract_api_public_ref", is_object(abstract_api) and abstract_api.get("public_ref") == "public_api")

        check("spec_lists_contract", "api-gateway.contracts.json" in spec)
        check("spec_lists_fixture", "api_gateway.json" in spec)
        check("spec_lists_checker", "check_api_gateway.py" in spec)
        check("cmake_registers_test", "KHANARY_API_GATEWAY_TEST" in cmake)
        check("manifest_lists_contract", any("api-gateway.contracts.json" in item for item in manifest.get("grammar:api_gateway_contract", [])))
        check("manifest_lists_checker", any("check_api_gateway.py" in item for item in manifest.get("grammar:api_gateway_contract", [])))
        check("manifest_lists_launcher", any("KHANARY.bat" in item for item in manifest.get("grammar:api_gateway_contract", [])))
        check("manifest_lists_ports", any("data/ports.json" in item for item in manifest.get("grammar:api_gateway_contract", [])))

        try:
            py_compile.compile(str(GATEWAY), doraise=True)
            check("gateway_py_compile", True)
        except py_compile.PyCompileError:
            check("gateway_py_compile", False)

        fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
        check("fixtures_have_cases", isinstance(fixture_cases, list) and len(fixture_cases) >= 16)
        pass_seen = 0
        fail_seen = 0
        for row in fixture_cases:
            if not is_object(row):
                check("fixture_case_is_object", False)
                continue
            case_id = str(row.get("id", "unnamed"))
            case_contracts = merged_contract(contracts, row.get("contract_patch"))
            error = ""
            if "rpc_method" in row:
                error = check_rpc_method(row.get("rpc_method"), case_contracts)
                if not error:
                    error = check_endpoint(row.get("endpoint"), case_contracts)
            else:
                error = check_endpoint(row.get("endpoint"), case_contracts)
            actual = "PASS" if not error else "FAIL"
            if actual == "PASS":
                pass_seen += 1
            else:
                fail_seen += 1
            check(f"fixture/{case_id}/expect", actual == row.get("expect"))
            if row.get("expect") == "FAIL":
                check(f"fixture/{case_id}/error", error == row.get("error"))
        check("fixtures_include_valid_endpoints", pass_seen >= 8)
        check("fixtures_include_negative_cases", fail_seen >= 8)

    except Exception as exc:
        cases.append(("exception/" + str(exc), False))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.api_gateway.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "contracts": str(CONTRACTS.relative_to(ROOT)),
        "fixtures": str(FIXTURES.relative_to(ROOT)),
        "gateway": str(GATEWAY.relative_to(ROOT)),
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
