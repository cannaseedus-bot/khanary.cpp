"""Reproduce the packaged geometry model checks from the repository root.

This checkout ships the versioned geometry model as assets plus KNU/kernel
bindings. The original GPU generator is upstream provenance, so the local
reproduction command verifies the packaged artifact contract.
"""
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "tools" / "check_geometry_model.py")]
    proc = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    try:
        verdict = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.returncode or 2
    if verdict.get("@kind") != "khanary.contract.verdict.v1":
        return 2
    return int(verdict.get("exit_code", proc.returncode))


if __name__ == "__main__":
    raise SystemExit(main())
