import json
import subprocess
import sys
from pathlib import Path


def test_geometry_model_assets():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "tools" / "check_geometry_model.py")],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    verdict = json.loads(proc.stdout)
    assert verdict["@kind"] == "khanary.contract.verdict.v1"
    assert verdict["suite"] == "geometry_assets"
    assert verdict["verdict"] == "PASS"
