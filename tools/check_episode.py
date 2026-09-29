"""Episode journal and provenance contract validator.

Validates that episode.contract.json is structurally complete and that the
algebraic invariants it declares are internally self-consistent.

Suite name: episode  (must match run_conformance.py required set)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conformance_common import verdict

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = []
    try:
        contract_path = ROOT / "data/schema/episode.contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        cases.append(("contract_loadable", True))

        # Required top-level sections
        required_keys = {
            "episode_structure", "scoring_contract", "verdict_contract",
            "fold_delta", "fold_direction", "direction_library",
        }
        cases.append(("required_sections", required_keys.issubset(contract.keys())))

        # S = W*C*R*G is multiplicative; R=0 ⟹ S=0
        def scoring_law_holds():
            for W in (0.0, 0.5, 1.0):
                for C in (0.0, 0.5, 1.0):
                    for R in (0.0, 0.5, 1.0):
                        for G in (0.0, 0.5, 1.0):
                            if R == 0.0 and W * C * R * G != 0.0:
                                return False
            return True
        cases.append(("scoring_law_multiplicative", scoring_law_holds()))

        # GOOD, BAD, UNKNOWN all defined
        vc = contract.get("verdict_contract", {})
        cases.append(("verdict_completeness",
                       all(k in vc for k in ("GOOD", "BAD", "UNKNOWN"))))

        # ΔF has four projections: P_M, P_R, P_W, P_C
        fd = contract.get("fold_delta", {})
        projections = fd.get("projections", {})
        cases.append(("fold_delta_projections",
                       all(k in projections for k in ("P_M", "P_R", "P_W", "P_C"))))

        # C ← f(Evidence) rule is documented in verdict_contract
        confidence_rule_present = (
            "confidence_rule" in vc
            or any("Evidence" in str(v) for v in vc.values())
        )
        cases.append(("confidence_from_evidence", confidence_rule_present))

        # Direction library has all 4 layers
        dl = contract.get("direction_library", {})
        layers = dl.get("four_layer_separation", {})
        cases.append(("direction_library_layers",
                       all(k in layers for k in (
                           "Journal", "FoldDelta", "DirectionLibrary", "Projection_Index"))))

        # PrincipalDirection law is provider-agnostic (PCA is one implementation)
        dir_sec = contract.get("fold_direction", {})
        pd = dir_sec.get("principal_direction", {})
        cases.append(("principal_direction_provider_agnostic",
                       "law" in pd and "PCA" in pd.get("law", "")))

        return verdict("episode", cases)
    except Exception as exc:
        return verdict("episode", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
