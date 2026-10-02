"""Validate STB tensor brain bindings for Micronaut projection containers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conformance_common import verdict

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/world/stb_tensor_brains.json"
MODEL = ROOT / "models/khanary-geometry-v0.3.0/MODEL.json"
GRAM_POOLS = ROOT / "data/mx2lm/gram_pool_registry.json"

EXPECTED_BINDINGS = {
    "TENSOR-µ",
    "VECTOR-µ",
    "METRIC-µ",
    "ADVISOR-µ",
    "JOKE-µ",
    "Mx2LM",
    "PLAN-µ",
    "POLITICS-µ",
}
TRACK_LOCAL = {"ADVISOR-µ", "JOKE-µ", "Mx2LM", "PLAN-µ", "POLITICS-µ"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_file(cases: list[tuple[str, bool]], prefix: str, record: dict | None) -> None:
    cases.append((prefix + "/record_present", isinstance(record, dict)))
    if not isinstance(record, dict):
        return
    rel = record.get("path", "")
    path = ROOT / rel
    cases.append((prefix + "/exists", bool(record.get("exists")) and path.exists()))
    if not path.exists():
        return
    cases.append((prefix + "/bytes", path.stat().st_size == record.get("bytes")))
    cases.append((prefix + "/sha256", _sha256(path) == record.get("sha256")))


def main() -> int:
    cases: list[tuple[str, bool]] = []
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        model = json.loads(MODEL.read_text(encoding="utf-8"))
        gram_pools = json.loads(GRAM_POOLS.read_text(encoding="utf-8"))

        cases.append(("registry/type", registry.get("@type") == "khanary.stb_tensor_brain_registry.v1"))
        cases.append(("registry/projection_boundary",
                      "do not verify semantic truth" in registry.get("authority_boundary", "") and
                      "mutate FoldGraph B" in registry.get("authority_boundary", "")))

        runs = {run["id"]: run for run in model.get("local_artifacts", {}).get("brain_runs", [])}
        canonical = registry.get("canonical_geometry_brain", {})
        canonical_id = canonical.get("run_id")
        cases.append(("canonical/run_declared", canonical_id in runs))
        if canonical_id in runs:
            run = runs[canonical_id]
            cases.append(("canonical/stb_matches_model", canonical.get("stb_source", {}).get("sha256") == run.get("stb_sha256")))
            cases.append(("canonical/brain_matches_model", canonical.get("brain_source", {}).get("sha256") == run.get("brain_sha256")))
        _check_file(cases, "canonical/stb", canonical.get("stb_source"))
        _check_file(cases, "canonical/brain", canonical.get("brain_source"))
        _check_file(cases, "canonical/proof", canonical.get("proof_source"))

        bindings = registry.get("micronaut_brains", [])
        ids = [binding.get("micronaut_id") for binding in bindings]
        cases.append(("coverage/count", registry.get("coverage", {}).get("bound_count") == len(bindings)))
        cases.append(("coverage/expected_bindings", set(ids) == EXPECTED_BINDINGS))
        cases.append(("coverage/no_duplicates", len(ids) == len(set(ids))))

        pool_tracks = gram_pools.get("tracks", {})
        for binding in bindings:
            mid = binding.get("micronaut_id", "<missing>")
            prefix = "binding/" + mid
            cases.append((prefix + "/type", binding.get("@type") == "khanary.stb_tensor_brain_binding.v1"))
            cases.append((prefix + "/projection_only", binding.get("projection_only") is True))
            cases.append((prefix + "/no_semantic_authority", binding.get("semantic_authority") is False))
            cases.append((prefix + "/no_topology_authority", binding.get("topology_authority") is False))
            cases.append((prefix + "/brain_run", binding.get("brain_run") in runs))
            if binding.get("brain_run") in runs:
                run = runs[binding["brain_run"]]
                header = binding.get("tensor_header", {})
                cases.append((prefix + "/tensor_header",
                              header.get("nodes") == run.get("nodes") and
                              header.get("edges") == run.get("edges") and
                              header.get("csr_entries") == run.get("csr_entries") and
                              header.get("experts") == run.get("experts") and
                              header.get("feature_count") == run.get("feature_count")))
                cases.append((prefix + "/stb_model_sha", binding.get("stb_source", {}).get("sha256") == run.get("stb_sha256")))
            _check_file(cases, prefix + "/stb", binding.get("stb_source"))
            _check_file(cases, prefix + "/brain", binding.get("brain_source"))

            if mid in TRACK_LOCAL:
                cases.append((prefix + "/track_pool_exists", mid in pool_tracks))
                cases.append((prefix + "/track_artifacts_present", isinstance(binding.get("track_artifacts"), dict)))
                for name, record in sorted((binding.get("track_artifacts") or {}).items()):
                    _check_file(cases, prefix + "/track_artifact/" + name, record)
            else:
                cases.append((prefix + "/not_track_local", binding.get("binding_kind") != "track_local_gram_tensor_brain"))

        return verdict("stb_tensor_brains", cases)
    except Exception as exc:  # pragma: no cover - conformance error path
        return verdict("stb_tensor_brains", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())