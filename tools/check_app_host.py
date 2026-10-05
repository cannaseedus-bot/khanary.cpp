#!/usr/bin/env python3
"""Validate the KHANARY Apps host vs MCP boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "data/grammar/app-host.contracts.json"
FIXTURES = ROOT / "data/conformance/app_host.json"
DOC = ROOT / "docs/HOST_LAYER.md"
README = ROOT / "README.md"
HOST_README = ROOT / "src/host/README.md"
CMAKE = ROOT / "CMakeLists.txt"
CSS_MANIFEST = ROOT / "data/manifest/css.manifest.json"

DOC_TOKENS = [
    "Apps host lane",
    "MCP lane",
    "src/host",
    "src/io/mcp_server.cpp",
    "app_user_sessions",
    "user_scoped_kql_tables",
    "gas_drive_sheet_bridges",
    "Atomic DOM",
    "atomic_dom",
    "atomic_dom_widgets",
    "ASX Blocks",
    "Atomic CSS key grammar",
    "data/manifest/css.manifest.json",
    "semantic_authority:true",
    "bin/",
]

CSS_REQUIRED_KEYS = [
    "⟁KEYGRAMMAR",
    "⟁SYNTAXUNITKINDS",
    "⟁TYPE:WIDGET",
    "⟁TYPE:WIDGETCARD",
    "⟁WIDGET",
    "⟁WIDGETCARD",
    "⟁HTMLBIND",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def load_json(path: Path) -> Any:
    return json.loads(read(path))


def is_object(value: Any) -> bool:
    return isinstance(value, dict)


def deep_merge(base: Any, patch: Any) -> Any:
    if not is_object(base) or not is_object(patch):
        return copy.deepcopy(patch)
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if key in out and is_object(out[key]) and is_object(value):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def as_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def check_contract(contract: dict[str, Any]) -> str:
    if contract.get("@type") != "khanary.app_host_contracts.v1":
        return "CONTRACT_TYPE_INVALID"
    if contract.get("semantic_authority") is not False:
        return "ROOT_CLAIMS_AUTHORITY"
    if contract.get("topology_mutation") is not False:
        return "ROOT_MUTATES_TOPOLOGY"
    if contract.get("commit_authorized") is not False:
        return "ROOT_AUTHORIZES_COMMIT"
    if contract.get("apps_are_mcp") is not False:
        return "APPS_AND_MCP_MERGED"
    if contract.get("mcp_is_apps") is not False:
        return "MCP_AND_APPS_MERGED"
    if contract.get("canonical_output_dir") != "bin":
        return "APP_HOST_OUTPUT_NOT_BIN"

    host = contract.get("host_layer")
    mcp = contract.get("mcp_layer")
    if not is_object(host) or not is_object(mcp):
        return "LAYER_NOT_OBJECT"
    if host.get("path") != "src/host":
        return "APP_HOST_PATH_INVALID"
    if mcp.get("path") != "src/io/mcp_server.cpp":
        return "MCP_PATH_INVALID"
    if host.get("path") == mcp.get("path"):
        return "LAYER_PATH_COLLISION"
    if host.get("ui_substrate") != "atomic_dom":
        return "APP_UI_SUBSTRATE_INVALID"
    if host.get("widget_substrate") != "atomic_dom":
        return "APP_WIDGET_SUBSTRATE_INVALID"
    widget_surfaces = as_set(host.get("widget_surfaces"))
    if "atomic_dom_widgets" not in widget_surfaces:
        return "APP_WIDGET_SURFACE_MISSING"
    if not {"typescript_mu_playground", "kuhul_mu_playground"}.issubset(widget_surfaces):
        return "APP_PLAYGROUND_WIDGET_MISSING"
    atomic_dom = host.get("atomic_dom")
    if not is_object(atomic_dom):
        return "APP_ATOMIC_DOM_MISSING"
    if atomic_dom.get("semantic_authority") is not False:
        return "APP_ATOMIC_DOM_CLAIMS_AUTHORITY"
    if atomic_dom.get("topology_mutation") is not False:
        return "APP_ATOMIC_DOM_MUTATES_TOPOLOGY"
    if atomic_dom.get("commit_authorized") is not False:
        return "APP_ATOMIC_DOM_AUTHORIZES_COMMIT"
    if not {"asx_blocks", "atomic_css_key_grammar"}.issubset(as_set(atomic_dom.get("declares"))):
        return "APP_ATOMIC_DOM_DECLARATION_INVALID"
    if atomic_dom.get("css_key_grammar_manifest") != "data/manifest/css.manifest.json":
        return "APP_ATOMIC_CSS_MANIFEST_INVALID"
    if atomic_dom.get("key_grammar_anchor") != "⟁KEYGRAMMAR":
        return "APP_ATOMIC_KEY_GRAMMAR_ANCHOR_INVALID"
    if atomic_dom.get("unit_kinds_key") != "⟁SYNTAXUNITKINDS":
        return "APP_ATOMIC_UNIT_KINDS_INVALID"
    if not {"Block", "Component"}.issubset(as_set(atomic_dom.get("required_unit_kinds"))):
        return "APP_ASX_BLOCK_KIND_MISSING"
    if not {"⟁WIDGET", "⟁WIDGETCARD", "⟁HTMLBIND"}.issubset(as_set(atomic_dom.get("required_widget_keys"))):
        return "APP_ATOMIC_WIDGET_KEY_MISSING"

    host_protocols = as_set(host.get("protocols"))
    mcp_protocols = as_set(mcp.get("protocols"))
    if "mcp" in host_protocols:
        return "APP_HOST_USES_MCP_PROTOCOL"
    if mcp_protocols != {"mcp"}:
        return "MCP_PROTOCOL_INVALID"

    host_owns = as_set(host.get("owns"))
    mcp_owns = as_set(mcp.get("owns"))
    if "app_user_sessions" not in host_owns:
        return "APP_HOST_SESSION_MISSING"
    if "user_scoped_kql_tables" not in host_owns:
        return "APP_HOST_KQL_SCOPE_MISSING"
    if "gas_drive_sheet_bridges" not in host_owns:
        return "APP_HOST_GAS_BRIDGE_MISSING"
    if mcp_owns & {"app_user_sessions", "ui_routes", "user_scoped_kql_tables", "gas_drive_sheet_bridges"}:
        return "MCP_OWNS_APP_STATE"
    if host_owns & mcp_owns:
        return "APP_MCP_OWNERSHIP_COLLISION"

    host_must_not = as_set(host.get("must_not_expose"))
    mcp_must_not = as_set(mcp.get("must_not_own"))
    if not {"raw_mcp_tool_call", "mcp_resource_template", "mcp_prompt_surface"}.issubset(host_must_not):
        return "APP_HOST_EXPOSES_MCP"
    if not {"app_user_sessions", "ui_routes", "user_scoped_kql_tables", "gas_drive_sheet_bridges"}.issubset(mcp_must_not):
        return "MCP_APP_BOUNDARY_MISSING"

    host_may_call = as_set(host.get("may_call"))
    if "json_rpc" not in host_may_call or "completion_gateway" not in host_may_call or "kql" not in host_may_call:
        return "APP_HOST_DOWNSTREAM_INVALID"
    if "mcp" in host_may_call:
        return "APP_HOST_CALLS_MCP_DIRECTLY"

    for layer_name, layer in (("APP_HOST", host), ("MCP", mcp)):
        authority = layer.get("authority")
        if not is_object(authority):
            return f"{layer_name}_AUTHORITY_MISSING"
        if authority.get("semantic_authority") is not False:
            return f"{layer_name}_CLAIMS_AUTHORITY"
        if authority.get("topology_mutation") is not False:
            return f"{layer_name}_MUTATES_TOPOLOGY"
        if authority.get("commit_authorized") is not False:
            return f"{layer_name}_AUTHORIZES_COMMIT"
        if authority.get("next_gate") != "Ch'en/XCFE/Xul":
            return f"{layer_name}_GATE_INVALID"

    invariants = contract.get("invariants")
    if not is_object(invariants):
        return "INVARIANTS_MISSING"
    for key in [f"APP{i}" for i in range(1, 9)]:
        if key not in invariants:
            return "INVARIANT_MISSING"

    return ""


def check_css_manifest(manifest: dict[str, Any]) -> str:
    if not is_object(manifest):
        return "CSS_MANIFEST_NOT_OBJECT"
    for key in CSS_REQUIRED_KEYS:
        if key not in manifest:
            return "CSS_MANIFEST_KEY_MISSING"

    key_grammar = manifest.get("⟁KEYGRAMMAR")
    if not isinstance(key_grammar, list):
        return "CSS_KEY_GRAMMAR_INVALID"
    key_grammar_tokens = as_set(key_grammar)
    if not {
        "key:address(book)",
        "book:declared-neighborhood",
        "grammar:constrained-book-traversal",
    }.issubset(key_grammar_tokens):
        return "CSS_KEY_GRAMMAR_INVALID"

    unit_kinds = as_set(manifest.get("⟁SYNTAXUNITKINDS"))
    if not {"Block", "Component"}.issubset(unit_kinds):
        return "CSS_ASX_BLOCK_KIND_MISSING"

    widget_type = as_set(manifest.get("⟁TYPE:WIDGET"))
    widget_card_type = as_set(manifest.get("⟁TYPE:WIDGETCARD"))
    if "kind:component" not in widget_type or "kind:component" not in widget_card_type:
        return "CSS_WIDGET_TYPE_INVALID"
    if "⟁WIDGETCARD" not in as_set(manifest.get("⟁WIDGET")):
        return "CSS_WIDGET_EXPANSION_INVALID"
    if "projection:style-binding" not in as_set(manifest.get("⟁HTMLBIND")):
        return "CSS_HTML_BIND_INVALID"

    return ""


def main() -> int:
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    try:
        contract = load_json(CONTRACTS) if CONTRACTS.exists() else {}
        fixtures = load_json(FIXTURES) if FIXTURES.exists() else {}
        css_manifest = load_json(CSS_MANIFEST) if CSS_MANIFEST.exists() else {}
        docs = "\n".join(read(path) for path in (DOC, README, HOST_README))
        cmake = read(CMAKE)

        check("contracts_file_present", CONTRACTS.exists())
        check("fixtures_file_present", FIXTURES.exists())
        check("css_manifest_file_present", CSS_MANIFEST.exists())
        check("css_manifest_valid", check_css_manifest(css_manifest) == "")
        check("docs_file_present", DOC.exists())
        check("host_readme_present", HOST_README.exists())
        check("contract_root_valid", check_contract(contract) == "")
        check("cmake_registers_test", "KHANARY_APP_HOST_TEST" in cmake)
        for token in DOC_TOKENS:
            check(f"docs/has/{token}", token in docs)

        fixture_cases = fixtures.get("cases", []) if is_object(fixtures) else []
        check("fixtures_have_cases", len(fixture_cases) >= 11)
        pass_seen = 0
        fail_seen = 0
        for row in fixture_cases:
            if not is_object(row):
                check("fixture_case_is_object", False)
                continue
            case_id = str(row.get("id", "unnamed"))
            candidate = deep_merge(contract, row.get("contract_patch", {}))
            error = check_contract(candidate)
            actual = "PASS" if not error else "FAIL"
            if actual == "PASS":
                pass_seen += 1
            else:
                fail_seen += 1
            check(f"fixture/{case_id}/expect", actual == row.get("expect"))
            if row.get("expect") == "FAIL":
                check(f"fixture/{case_id}/error", error == row.get("error"))
        check("fixtures_include_valid_case", pass_seen >= 1)
        check("fixtures_include_negative_cases", fail_seen >= 10)
    except Exception as exc:
        cases.append(("exception/" + str(exc), False))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.app_host.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "contracts": str(CONTRACTS.relative_to(ROOT)),
        "fixtures": str(FIXTURES.relative_to(ROOT)),
        "docs": str(DOC.relative_to(ROOT)),
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
