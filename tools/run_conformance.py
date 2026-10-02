"""Build an isolated native tree and aggregate CTest's stable verdict records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/conformance")
    parser.add_argument("--config", choices=["Debug", "Release"], default="Release")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which("cmake") or "C:/Program Files/CMake/bin/cmake.exe"
    ctest = str(Path(cmake).with_name("ctest.exe" if os.name == "nt" else "ctest"))
    # Windows environment keys are case-insensitive; inherited Path/PATH pairs
    # otherwise cause MSBuild's CL task to fail before invoking the compiler.
    env = {key.upper(): value for key, value in os.environ.items()} if os.name == "nt" else dict(os.environ)
    if args.skip_build:
        cache = build / "CMakeCache.txt"
        if not cache.exists():
            raise RuntimeError(
                f"--skip-build: build tree not configured at {build} — run without --skip-build first")
        cmake_lists = ROOT / "CMakeLists.txt"
        freshness_markers = [cache, build / "CMakeFiles" / "generate.stamp",
                             build / "CMakeFiles" / "cmake.check_cache"]
        configured_mtime = max(marker.stat().st_mtime for marker in freshness_markers if marker.exists())
        if cmake_lists.stat().st_mtime > configured_mtime:
            raise RuntimeError(
                f"--skip-build: build tree is stale (CMakeLists.txt newer than configure stamp) — "
                "re-run without --skip-build to reconfigure")
    else:
        subprocess.run([cmake, "-S", str(ROOT), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64"],
                       env=env, cwd=ROOT, check=True)
        subprocess.run([cmake, "--build", str(build), "--config", args.config], env=env, cwd=ROOT, check=True)
    result = subprocess.run([ctest, "--test-dir", str(build), "-C", args.config,
                             "--no-tests=error", "-V"], env=env, cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    (build / "conformance.log").write_text(result.stdout, encoding="utf-8")
    suites = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\d+: (\{.*\})$", line)
        if not match:
            continue
        row = json.loads(match.group(1))
        if row.get("@kind") != "khanary.contract.verdict.v1":
            continue
        seeds = "\n".join(f"{row['suite']}|{case['id']}|{case['verdict']}" for case in row["cases"])
        expected = "sha256:" + hashlib.sha256(seeds.encode()).hexdigest()
        if expected != row["summary_hash"]:
            raise RuntimeError("verdict hash mismatch: " + row["suite"])
        suites.append(row)
    required = {
        "projection",
        "aiml",
        "attention",
        "router",
        "contracts",
        "canonical",
        "sdds",
        "sdds_manifest",
        "schema",
        "geometry_assets",
        "klsl",
        "gml_validator",
        "episode",
        "field_vectors",
        "vectors_xml",
        "dataset_ingestion",
        "metric_graph",
        "stb_tensor_brains",
        "micronaut_tensor_slots",
        "runtime_schema_stack",
        "boundary_split",
        "gpu-device",
        "wgsl-codegen",
    }
    found = {suite["suite"] for suite in suites}
    if not required.issubset(found) or len(found) != len(suites):
        raise RuntimeError("missing or duplicate verdict records: " + ",".join(sorted(required - found)))
    suites.sort(key=lambda s: s["suite"])
    passed = result.returncode == 0 and all(s["verdict"] == "PASS" for s in suites)
    seeds = "\n".join(f"{s['suite']}|{s['verdict']}|{s['summary_hash']}" for s in suites)

    suite_index = {suite["suite"]: suite for suite in suites}

    def lane_state(suite_name: str) -> str:
        row = suite_index.get(suite_name)
        if row is None:
            return "not_run"
        return "pass" if row.get("verdict") == "PASS" else "fail"

    summary = {"@kind": "khanary.conformance.summary.v1", "verdict": "PASS" if passed else "FAIL",
               "counts": {"suites": len(suites), "cases": sum(len(s['cases']) for s in suites)},
               "summary_hash": "sha256:" + hashlib.sha256(seeds.encode()).hexdigest(), "suites": suites,
               "scope": {"gpu_execution": lane_state("gpu-device"),
                         "wgsl_execution": lane_state("wgsl-codegen"),
                         "scxq2": "not_root_conformant"}}
    (build / "conformance-summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "suites"}, sort_keys=True))
    if not passed:
        print(result.stdout, file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"@kind": "khanary.conformance.summary.v1", "verdict": "ERROR", "error": str(exc)}))
        raise SystemExit(2)
