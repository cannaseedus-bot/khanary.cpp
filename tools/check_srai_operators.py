#!/usr/bin/env python3
"""
check_srai_operators.py — Conformance checker for SRAI operator dispatch records.

Validates:
  - Binding frame fields: operator, executor, phase, input_roles, output_role, effects
  - effects must be a subset of the permitted set
  - Operators with propose_delta must declare commit_requires (non-null, non-empty)
  - TRAIN must declare W_target = W_learned
  - COLLAPSE output_role must be collapsed_candidate
  - VERIFY output_role must be verified_outcome
  - Pipeline sequences: TRAIN must be preceded by VERIFY

Usage:
  py -3 tools/check_srai_operators.py
  py -3 tools/check_srai_operators.py --contracts data/grammar/srai-operators.contracts.json
  py -3 tools/check_srai_operators.py --fixtures data/conformance/srai_operators.json
"""

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent

PERMITTED_EFFECTS = {
    "read", "measure", "order", "analyze", "gate",
    "project", "emit", "collapse", "propose_delta"
}

REQUIRED_FRAME_FIELDS = ["operator", "executor", "phase", "input_roles", "output_role", "effects"]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_frame(frame: dict) -> list[str]:
    """Validate a single binding frame. Returns list of violation strings."""
    violations = []

    for field in REQUIRED_FRAME_FIELDS:
        if field not in frame:
            violations.append(f"Missing required field: {field}")

    effects = frame.get("effects", [])
    bad_effects = [e for e in effects if e not in PERMITTED_EFFECTS]
    if bad_effects:
        violations.append(f"Unknown effects: {bad_effects}; permitted: {sorted(PERMITTED_EFFECTS)}")

    if "propose_delta" in effects:
        cr = frame.get("commit_requires")
        if not cr:
            violations.append("effects contains propose_delta but commit_requires is null or missing")

    op = frame.get("operator", "")
    if op == "TRAIN":
        if frame.get("W_target") != "W_learned":
            violations.append("TRAIN must declare W_target='W_learned'")
        input_roles = frame.get("input_roles", [])
        if "verified_feedback" not in input_roles:
            violations.append("TRAIN input_roles must include verified_feedback")

    if op == "COLLAPSE":
        out = frame.get("output_role", "")
        if out not in ("collapsed_candidate", ""):
            violations.append(f"COLLAPSE output_role must be collapsed_candidate, got '{out}'")

    if op == "VERIFY":
        out = frame.get("output_role", "")
        if out not in ("verified_outcome", ""):
            violations.append(f"VERIFY output_role must be verified_outcome, got '{out}'")

    return violations


def validate_pipeline_sequence(seq: list[str]) -> list[str]:
    """Validate a pipeline sequence. TRAIN must be preceded by VERIFY."""
    violations = []
    if "TRAIN" in seq:
        idx_train = seq.index("TRAIN")
        if "VERIFY" not in seq[:idx_train]:
            violations.append(f"TRAIN at position {idx_train} not preceded by VERIFY in sequence {seq}")
    return violations


def check_fixtures(fixtures_path: Path) -> tuple[int, int]:
    data = load_json(fixtures_path)
    passed = 0
    failed = 0

    for fixture in data.get("positive", []):
        fid = fixture["id"]
        frame = fixture.get("frame")
        seq = fixture.get("pipeline_sequence")
        if frame:
            violations = validate_frame(frame)
        elif seq:
            violations = validate_pipeline_sequence(seq)
        else:
            violations = []

        if violations:
            print(f"FAIL [positive/{fid}]: expected valid but got violations: {violations}")
            failed += 1
        else:
            passed += 1

    for fixture in data.get("negative", []):
        fid = fixture["id"]
        frame = fixture.get("frame")
        seq = fixture.get("pipeline_sequence")
        if frame:
            violations = validate_frame(frame)
        elif seq:
            violations = validate_pipeline_sequence(seq)
        else:
            violations = []

        if violations:
            passed += 1
        else:
            print(f"FAIL [negative/{fid}]: expected invalid but no violations found (expected: {fixture.get('violation')})")
            failed += 1

    return passed, failed


def check_contracts(contracts_path: Path) -> tuple[int, int]:
    """Validate the contracts registry itself."""
    data = load_json(contracts_path)
    passed = 0
    failed = 0

    for op in data.get("operators", []):
        violations = validate_frame(op)
        name = op.get("operator", "?")
        if violations:
            print(f"FAIL [contracts/{name}]: {violations}")
            failed += 1
        else:
            passed += 1

    pipeline_ops = data.get("pipeline", {}).get("canonical_order", [])
    if "TRAIN" in pipeline_ops:
        idx = pipeline_ops.index("TRAIN")
        if "VERIFY" not in pipeline_ops[:idx]:
            print("FAIL [contracts/pipeline]: TRAIN not preceded by VERIFY in canonical_order")
            failed += 1
        else:
            passed += 1

    invariants = data.get("invariants", [])
    if len(invariants) < 7:
        print(f"WARN [contracts/invariants]: only {len(invariants)} invariants declared, expected >= 7")

    return passed, failed


def main():
    ap = argparse.ArgumentParser(description="SRAI operators conformance checker")
    ap.add_argument("--contracts", default=str(ROOT / "data/grammar/srai-operators.contracts.json"))
    ap.add_argument("--fixtures", default=str(ROOT / "data/conformance/srai_operators.json"))
    args = ap.parse_args()

    total_pass = 0
    total_fail = 0

    contracts_path = Path(args.contracts)
    if contracts_path.exists():
        p, f = check_contracts(contracts_path)
        total_pass += p
        total_fail += f
    else:
        print(f"WARN: contracts file not found: {contracts_path}")

    fixtures_path = Path(args.fixtures)
    if fixtures_path.exists():
        p, f = check_fixtures(fixtures_path)
        total_pass += p
        total_fail += f
    else:
        print(f"WARN: fixtures file not found: {fixtures_path}")

    total = total_pass + total_fail
    print(f"\nSRAI operators: {total_pass}/{total} passed, {total_fail} failed")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
