#!/usr/bin/env python3
"""Validate K'UHUL fold binding contracts and backend registry drift."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/grammar/fold-bindings.contracts.json"
STANDALONE = ROOT / "kuhul-grammar.ebnf"
RUNTIME = ROOT / "data/grammar/kuhul.runtime-grammar.ebnf"
GLOSSARY = ROOT / "docs/GLOSSARY.md"
SEMANTIC_FOLDS = ROOT / "docs/SEMANTIC_FOLDS.md"
SPEC = ROOT / "docs/KHANARY-CPP-SPEC.md"

REQUIRED_PHASES = ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"]

EXPECTED_HLSL_ENTRIES = {
    "Pop": "PopKernel",
    "Wo": "ConstraintKernel",
    "Yax": "CandidateKernel",
    "Sek": "OperationKernel",
    "Ch'en": "ObservationKernel",
    "Xul": "CollapseKernel",
}

EXPECTED_CONTRACTS = {
    "Pop": {
        "pre": "input is |k>",
        "post": "output is a distribution over N(k)",
    },
    "Wo": {
        "pre": "input is B_P",
        "post": "output is B_W subset_of B_P plus rejection outcome",
    },
    "Yax": {
        "pre": "input is B_W",
        "post": "output is candidate set B_Y",
    },
    "Sek": {
        "pre": "input is B_Y and policy Pi",
        "post": "output is B_S = Pi_op(B_Y)",
    },
    "Ch'en": {
        "pre": "input is B_S",
        "post": "output is observable B_C",
    },
    "Xul": {
        "pre": "input is B_C and policy Pi",
        "post": "output is k_prime = Pi(B_C)",
    },
}

REQUIRED_GRAMMAR_TOKENS = [
    "fold_binding_decl",
    "fold_binding_definition",
    "Fold Binding = (alpha, theta, backend, entrypoint, contract)",
    "Fold -> Phase Binding -> Backend Binding -> Executor",
    "K'UHUL Fold -> Fold Binding -> KXC -> HLSL -> GPU",
    "Bind_backend(alpha_theta) = K_theta^backend",
    "K_theta^backend models Contract(alpha_theta)",
    "backend_target",
    "hoare_contract",
]

REQUIRED_DOC_TOKENS = [
    "Fold Binding = (α, θ, backend, entrypoint, contract)",
    "K'UHUL owns the fold and phase",
    "backend supplies the entrypoint",
    "K_θ^backend ⊨ Contract(α_θ)",
    "Fold → Phase Binding → Backend Binding → Executor",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def is_dispatch_shape(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(x, int) and x > 0 for x in value)
    )


def main() -> int:
    registry = load_json(REGISTRY)
    standalone = read(STANDALONE)
    runtime = read(RUNTIME)
    docs = "\n".join(read(p) for p in (GLOSSARY, SEMANTIC_FOLDS, SPEC))
    cases: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        cases.append((name, bool(ok)))

    check("registry_file_present", REGISTRY.exists())
    check("registry_type", registry.get("@type") == "khanary.fold_binding_registry.v1")
    check("semantic_owner_is_kuhul", registry.get("semantic_owner") == "K'UHUL")
    check("backend_owner_is_binding_layer", registry.get("backend_owner") == "binding_layer")
    check("semantic_authority_false", registry.get("semantic_authority") is False)
    check("authority_boundary_explicit", "not_semantic_authority" in str(registry.get("authority", "")))

    law = registry.get("law", {})
    if not isinstance(law, dict):
        law = {}
    check("law_tuple", law.get("tuple") == "FoldBinding = (alpha, theta, backend, entrypoint, contract)")
    check("law_relation", law.get("relation") == "Bind_backend(alpha_theta) = K_theta^backend")
    check("law_validity", law.get("validity") == "K_theta^backend models Contract(alpha_theta)")
    check(
        "law_chain",
        law.get("chain") == ["K'UHUL Fold", "Fold Binding", "KXC", "HLSL", "GPU"],
    )
    check(
        "law_control_chain",
        law.get("control_chain") == ["Fold", "Phase Binding", "Backend Binding", "Executor"],
    )
    check("law_boundary", "K'UHUL owns alpha and theta" in str(law.get("boundary", "")))

    phase_contracts = registry.get("phase_contracts", {})
    if not isinstance(phase_contracts, dict):
        phase_contracts = {}
    check("phase_contract_keys_exact", sorted(phase_contracts) == sorted(REQUIRED_PHASES))
    for phase in REQUIRED_PHASES:
        phase_contract = phase_contracts.get(phase, {})
        expected = EXPECTED_CONTRACTS[phase]
        check(f"phase/{phase}/pre", phase_contract.get("pre") == expected["pre"])
        check(f"phase/{phase}/post", phase_contract.get("post") == expected["post"])
        check(f"phase/{phase}/alpha_declared", bool(phase_contract.get("alpha")))
        check(f"phase/{phase}/contract_type_declared", bool(phase_contract.get("contract_type")))

    bindings = registry.get("bindings", [])
    if not isinstance(bindings, list):
        bindings = []
    check("binding_count", len(bindings) == len(REQUIRED_PHASES))
    seen_phases: set[str] = set()
    for idx, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            check(f"binding/{idx}/is_object", False)
            continue
        phase = binding.get("@phase")
        seen_phases.add(str(phase))
        check(f"binding/{idx}/fold", isinstance(binding.get("@fold"), str) and bool(binding["@fold"]))
        check(f"binding/{idx}/phase_known", phase in REQUIRED_PHASES)
        check(f"binding/{idx}/semantic_tag", isinstance(binding.get("@semantic"), str) and binding["@semantic"].startswith("XCFE:"))
        check(f"binding/{idx}/backend_hlsl", binding.get("@backend") == "hlsl")
        check(f"binding/{idx}/entry_expected", binding.get("@entry") == EXPECTED_HLSL_ENTRIES.get(str(phase)))
        check(f"binding/{idx}/dispatch_shape", is_dispatch_shape(binding.get("@dispatch")))
        contract = binding.get("@contract", {})
        if not isinstance(contract, dict):
            contract = {}
        expected = EXPECTED_CONTRACTS.get(str(phase), {})
        check(f"binding/{idx}/contract_pre", contract.get("pre") == expected.get("pre"))
        check(f"binding/{idx}/contract_post", contract.get("post") == expected.get("post"))
        check(
            f"binding/{idx}/matches_phase_contract",
            contract.get("pre") == phase_contracts.get(str(phase), {}).get("pre")
            and contract.get("post") == phase_contracts.get(str(phase), {}).get("post"),
        )
    check("bindings_cover_required_phases", sorted(seen_phases) == sorted(REQUIRED_PHASES))

    grammar_text = standalone + "\n" + runtime
    for token in REQUIRED_GRAMMAR_TOKENS:
        check(f"grammar/has/{token}", token in grammar_text)
    for token in REQUIRED_DOC_TOKENS:
        check(f"docs/has/{token}", token in docs)
    check("spec_lists_registry", "fold-bindings.contracts.json" in read(SPEC))
    check("spec_lists_checker", "check_fold_bindings.py" in read(SPEC))

    rows = [{"id": name, "verdict": "PASS" if ok else "FAIL"} for name, ok in sorted(cases)]
    verdict = "PASS" if all(ok for _, ok in cases) else "FAIL"
    print(json.dumps({
        "@kind": "khanary.fold_binding_contracts.verdict.v1",
        "case_count": len(cases),
        "cases": rows,
        "registry": str(REGISTRY.relative_to(ROOT)),
        "verdict": verdict,
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
