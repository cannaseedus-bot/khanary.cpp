"""Validate runtime metadata stack and hash-bound phase-driver artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from conformance_common import verdict


ROOT = Path(__file__).resolve().parents[1]


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_hash(value: str) -> str:
    value = value.strip().lower()
    return value if value.startswith("sha256:") else f"sha256:{value}"


def main() -> int:
    cases: list[tuple[str, bool]] = []
    try:
        schema_dir = ROOT / "data" / "schema"
        fold_runtime_path = schema_dir / "fold.runtime.json"
        runtime_meta_path = schema_dir / "kuhul.fold-runtime.json"
        cycle_spec_path = schema_dir / "cycle-hash.spec.json"

        cases.append(("fold_runtime_present", fold_runtime_path.exists()))
        cases.append(("runtime_meta_present", runtime_meta_path.exists()))
        cases.append(("cycle_spec_present", cycle_spec_path.exists()))

        fold_runtime = json.loads(fold_runtime_path.read_text(encoding="utf-8"))
        runtime_meta = json.loads(runtime_meta_path.read_text(encoding="utf-8"))
        cases.append(("fold_runtime_parse", isinstance(fold_runtime, dict)))
        cases.append(("runtime_meta_parse", isinstance(runtime_meta, dict)))

        lock = fold_runtime.get("phase_driver_lock", {})
        cases.append(("phase_driver_lock_object", isinstance(lock, dict)))
        cases.append(("phase_driver_enabled", lock.get("enabled") is True))
        cases.append(("phase_driver_mode_enforce", lock.get("mode") == "enforce"))
        phases = lock.get("phases", [])
        cases.append(("phase_list_present", isinstance(phases, list) and len(phases) == 6))

        expected_phases = {"Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"}
        declared_phases = {row.get("phase") for row in phases if isinstance(row, dict)}
        cases.append(("all_phase_rows_declared", declared_phases == expected_phases))

        phase_hashes = runtime_meta.get("phase_driver_sha256", {})
        cases.append(("phase_driver_hash_map_object", isinstance(phase_hashes, dict)))
        cases.append(("phase_driver_hash_algo_sha256", phase_hashes.get("algorithm") == "sha256"))

        phase_root = ROOT / lock.get("dll_root", "data/drivers/phases")
        for row in phases:
            if not isinstance(row, dict):
                continue
            phase = row.get("phase", "unknown")
            artifact_name = row.get("dll")
            if not artifact_name:
                cases.append((f"phase_{phase}_artifact_declared", False))
                continue
            cases.append((f"phase_{phase}_artifact_declared", True))
            artifact_path = (phase_root / artifact_name).resolve()
            exists = artifact_path.exists()
            cases.append((f"phase_{phase}_artifact_exists", exists))

            expected_hash = phase_hashes.get(artifact_name, "")
            hash_declared = isinstance(expected_hash, str) and "__PENDING__" not in expected_hash and bool(expected_hash)
            cases.append((f"phase_{phase}_hash_declared", hash_declared))
            if exists and hash_declared:
                actual_hash = f"sha256:{sha256_hex(artifact_path)}"
                cases.append((f"phase_{phase}_hash_match", normalize_hash(expected_hash) == actual_hash))

        stack_hashes = runtime_meta.get("sha256", {})
        cases.append(("runtime_hash_map_object", isinstance(stack_hashes, dict)))
        cases.append(("runtime_hash_algo_sha256", stack_hashes.get("algorithm") == "sha256"))
        for key, path in (("fold_runtime", fold_runtime_path), ("cycle_hash_spec", cycle_spec_path)):
            expected_hash = stack_hashes.get(key, "")
            declared = isinstance(expected_hash, str) and expected_hash.startswith("sha256:")
            cases.append((f"{key}_hash_declared", declared))
            if declared:
                actual_hash = f"sha256:{sha256_hex(path)}"
                cases.append((f"{key}_hash_match", normalize_hash(expected_hash) == actual_hash))

        return verdict("runtime_schema_stack", cases)
    except Exception as exc:
        return verdict("runtime_schema_stack", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
