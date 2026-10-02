"""Build or validate per-Micronaut tensor slot bindings.

The slot registry answers the question "does each Micronaut have a tensor?"
without pretending every Micronaut already has a private STB brain. A slot is
the per-Micronaut tensor/fiber address. STB tensor brains are backing evidence
only when the existing registry proves one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from conformance_common import verdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/world/micronaut_tensor_slots.json"
MICRONAUT_MANIFEST = ROOT / "data/manifest/micronaut.manifest.json"
RELATIONAL_MANIFEST = ROOT / "data/manifest/relational.manifest.json"
RELATIONAL_GRAPH = ROOT / "data/grammar/kuhul-relational-graph.xml"
STB_REGISTRY = ROOT / "data/world/stb_tensor_brains.json"

GRAMMAR_TOOLS = ROOT / "tools/grammar"
if str(GRAMMAR_TOOLS) not in sys.path:
    sys.path.insert(0, str(GRAMMAR_TOOLS))
from syntax_compiler import compile_relational_graph  # noqa: E402

MICRONAUT_RE = re.compile(r"\b[A-Z0-9]+(?:-[A-Z0-9]+)*-µ\b|\bMx2LM\b")

PHASE_ORDER = ["Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"]
CATEGORY_PHASE = {
    "executive": "Wo",
    "cognitive": "Sek",
    "creative": "Yax",
    "storage": "Xul",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_id(micronaut_id: str) -> str:
    return (
        micronaut_id
        .replace("µ", "u")
        .replace("'", "")
        .replace(":", "")
        .replace(" ", "_")
    )


def _manifest_phase(manifest: dict, micronaut_id: str) -> str | None:
    hits: list[str] = []
    for key, values in manifest.items():
        haystack = [key] + list(values if isinstance(values, list) else [])
        if micronaut_id not in haystack:
            continue
        for item in haystack:
            if isinstance(item, str) and item.startswith("⟁"):
                phase = item[1:]
                if phase in PHASE_ORDER:
                    hits.append(phase)
    if hits:
        return min(hits, key=PHASE_ORDER.index)
    return None


def _known_micronauts(micronaut_manifest: dict, relational_manifest: dict, stb_registry: dict) -> set[str]:
    ids: set[str] = set()
    for payload in (micronaut_manifest, relational_manifest, stb_registry):
        ids.update(MICRONAUT_RE.findall(json.dumps(payload, ensure_ascii=False)))
    ids.update(relational_manifest.get("relational:micronaut-network", []))
    ids.update(binding.get("micronaut_id") for binding in stb_registry.get("micronaut_brains", []))
    return {mid for mid in ids if mid}


def _relational_categories() -> dict[str, str]:
    if not RELATIONAL_GRAPH.exists():
        return {}
    model = compile_relational_graph(RELATIONAL_GRAPH)
    categories: dict[str, str] = {}
    for node in model.get("nodes", []):
        categories.setdefault(node["id"], node.get("category", "uncategorized"))
    return categories


def build_registry() -> dict:
    micronaut_manifest = _load_json(MICRONAUT_MANIFEST)
    relational_manifest = _load_json(RELATIONAL_MANIFEST)
    stb_registry = _load_json(STB_REGISTRY)

    stb_bindings = {
        binding["micronaut_id"]: binding
        for binding in stb_registry.get("micronaut_brains", [])
    }
    categories = _relational_categories()
    canonical = stb_registry.get("canonical_geometry_brain", {})
    canonical_run = canonical.get("run_id", "brain4")
    known = sorted(_known_micronauts(micronaut_manifest, relational_manifest, stb_registry))

    slots = []
    for micronaut_id in known:
        binding = stb_bindings.get(micronaut_id)
        category = categories.get(micronaut_id, "manifest")
        phase = (
            (binding or {}).get("phase")
            or CATEGORY_PHASE.get(category)
            or _manifest_phase(micronaut_manifest, micronaut_id)
            or "Yax"
        )

        if binding:
            if binding.get("binding_kind") == "track_local_gram_tensor_brain":
                materialization = "track_local_evidence_backed"
                rank_note = "gram_tensor_rank_dynamic"
            elif binding.get("micronaut_id") == "VECTOR-µ":
                materialization = "provider_backed_rank1"
                rank_note = "rank_(1,0)_provider"
            elif binding.get("micronaut_id") == "METRIC-µ":
                materialization = "provider_backed_metric"
                rank_note = "rank_(0,2)_metric_provider"
            else:
                materialization = "provider_backed_tensor"
                rank_note = "type_(p,q)_provider"
            backing = {
                "kind": "stb_tensor_brain_binding",
                "registry": "data/world/stb_tensor_brains.json",
                "micronaut_id": micronaut_id,
                "binding_kind": binding.get("binding_kind"),
                "brain_run": binding.get("brain_run"),
                "stb_source": binding.get("stb_source", {}).get("path"),
                "track_artifacts": sorted((binding.get("track_artifacts") or {}).keys()),
            }
        else:
            materialization = "shared_virtual_tensor_slot"
            rank_note = "type_(p,q)_slot_until_private_evidence"
            backing = {
                "kind": "shared_tensor_provider",
                "provider": "TENSOR-µ",
                "metric_provider": "METRIC-µ",
                "rank1_provider": "VECTOR-µ",
                "registry": "data/world/stb_tensor_brains.json",
                "brain_run": canonical_run,
                "promotion_requires": [
                    "track_local_gram_pool_evidence",
                    "validator_PASS",
                    "Ch'en verification",
                    "Xul commit",
                ],
            }

        slots.append({
            "@type": "khanary.micronaut_tensor_slot.v1",
            "micronaut_id": micronaut_id,
            "slot_id": f"tensor_slot:{_safe_id(micronaut_id)}",
            "phase": phase,
            "category": category,
            "materialization": materialization,
            "rank_contract": {
                "notation": "TensorBrain_mu = <Track_mu, BaseCoord_mu, Fiber_mu, Rank(p,q), Generation, Backing>",
                "rank_note": rank_note,
                "fiber_owner": micronaut_id,
                "base_coordinate": f"@node:{micronaut_id}",
            },
            "backing": backing,
            "provider_stack": ["TENSOR-µ", "METRIC-µ", "VECTOR-µ"],
            "projection_only": True,
            "semantic_authority": False,
            "topology_authority": False,
        })

    private_count = sum(1 for slot in slots if slot["backing"]["kind"] == "stb_tensor_brain_binding")
    return {
        "@type": "khanary.micronaut_tensor_slot_registry.v1",
        "version": "1.0.0",
        "generated_by": "tools/check_micronaut_tensor_slots.py --write",
        "authority_boundary": (
            "Every Micronaut may own a tensor slot, but tensor slots are projection/fiber state. "
            "They cannot verify semantic truth, mutate FoldGraph B, rewrite RelationISA, or commit memory."
        ),
        "source_surfaces": {
            "micronaut_manifest": "data/manifest/micronaut.manifest.json",
            "relational_manifest": "data/manifest/relational.manifest.json",
            "relational_graph": "data/grammar/kuhul-relational-graph.xml",
            "stb_tensor_brains": "data/world/stb_tensor_brains.json",
        },
        "slot_law": {
            "every_known_micronaut_has_slot": True,
            "private_stb_brain_requires_evidence": True,
            "shared_provider_slot_is_valid": True,
            "formula": "MicronautTensor_mu = <base_coord_mu, fiber_mu, rank_contract, backing>",
            "promotion_rule": "shared_virtual_tensor_slot -> track_local_evidence_backed only after evidence + Ch'en + Xul",
        },
        "coverage": {
            "known_micronaut_count": len(slots),
            "private_or_provider_stb_backed_count": private_count,
            "shared_virtual_slot_count": len(slots) - private_count,
        },
        "slots": slots,
    }


def _render(registry: dict) -> str:
    return json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def check_registry(write: bool = False) -> int:
    cases: list[tuple[str, bool]] = []
    try:
        expected = build_registry()
        if write:
            OUT.write_text(_render(expected), encoding="utf-8", newline="\n")

        cases.append(("registry/exists", OUT.exists()))
        if not OUT.exists():
            return verdict("micronaut_tensor_slots", cases)

        actual_text = OUT.read_text(encoding="utf-8")
        actual = json.loads(actual_text)
        cases.append(("registry/type", actual.get("@type") == "khanary.micronaut_tensor_slot_registry.v1"))
        cases.append(("registry/current", actual_text == _render(expected)))
        cases.append(("authority/no_truth_claim",
                      "cannot verify semantic truth" in actual.get("authority_boundary", "")))

        slots = actual.get("slots", [])
        slot_ids = [slot.get("micronaut_id") for slot in slots]
        cases.append(("coverage/count", actual.get("coverage", {}).get("known_micronaut_count") == len(slots)))
        cases.append(("coverage/no_duplicate_slots", len(slot_ids) == len(set(slot_ids))))
        cases.append(("coverage/all_known", set(slot_ids) == {slot["micronaut_id"] for slot in expected["slots"]}))

        stb_registry = _load_json(STB_REGISTRY)
        stb_ids = {binding["micronaut_id"] for binding in stb_registry.get("micronaut_brains", [])}
        for slot in slots:
            mid = slot.get("micronaut_id", "<missing>")
            prefix = "slot/" + mid
            cases.append((prefix + "/type", slot.get("@type") == "khanary.micronaut_tensor_slot.v1"))
            cases.append((prefix + "/projection_only", slot.get("projection_only") is True))
            cases.append((prefix + "/no_semantic_authority", slot.get("semantic_authority") is False))
            cases.append((prefix + "/no_topology_authority", slot.get("topology_authority") is False))
            cases.append((prefix + "/provider_stack", "TENSOR-µ" in slot.get("provider_stack", [])))
            backing = slot.get("backing", {})
            if mid in stb_ids:
                cases.append((prefix + "/stb_backed", backing.get("kind") == "stb_tensor_brain_binding"))
            else:
                cases.append((prefix + "/shared_provider_backed", backing.get("kind") == "shared_tensor_provider"))
                cases.append((prefix + "/promotion_rule", "promotion_requires" in backing))

        return verdict("micronaut_tensor_slots", cases)
    except Exception as exc:  # pragma: no cover - conformance error path
        return verdict("micronaut_tensor_slots", cases, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the generated registry before checking")
    args = parser.parse_args()
    return check_registry(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
