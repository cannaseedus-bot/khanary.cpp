"""Field-vector metadata validator for data/world/field.xml.

Checks coordinate validity (x,y,z), vector_source metadata, tensor section
presence, and QKV metadata state. Coords and source are hard-checked (FAIL
on error); tensor and QKV are known partial — they warn to stderr but PASS.

Suite name: field_vectors  (must match run_conformance.py required set)
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conformance_common import verdict

ROOT = Path(__file__).resolve().parents[1]

VALID_SOURCES = frozenset(["gml", "ontology_static", "phase_default"])


def main():
    cases = []
    try:
        field_path = ROOT / "data/world/field.xml"
        tree = ET.parse(str(field_path))
        root_elem = tree.getroot()
        nodes = list(root_elem.iter("node"))

        # ── C1: coordinate validity — nodes with any x/y/z must have all 3 ────
        coord_errors = []
        for node in nodes:
            nid = node.attrib.get("id", "<unknown>")
            has_any = any(node.attrib.get(c) is not None for c in ("x", "y", "z"))
            if not has_any:
                continue
            for coord in ("x", "y", "z"):
                val = node.attrib.get(coord)
                if val is None:
                    coord_errors.append(f"partial_coord@{nid}")
                    break
                try:
                    fval = float(val)
                    if not math.isfinite(fval):
                        coord_errors.append(f"non_finite@{nid}")
                except ValueError:
                    coord_errors.append(f"non_numeric@{nid}")
        cases.append(("coord_xyz_valid", len(coord_errors) == 0))

        # ── C2: source validity — recognised vector_source values only ─────────
        source_errors = [
            node.attrib.get("vector_source")
            for node in nodes
            if node.attrib.get("vector_source") is not None
            and node.attrib["vector_source"] not in VALID_SOURCES
        ]
        cases.append(("source_valid", len(source_errors) == 0))

        # ── C3: tensor section present (partial: empty is expected) ───────────
        tensors = root_elem.find(".//tensors")
        tensor_count = len(list(tensors)) if tensors is not None else -1
        cases.append(("tensor_section_present", tensors is not None))
        if tensor_count == 0:
            print("warn | tensors section present but empty — partial, not yet populated",
                  file=sys.stderr)

        # ── C4: K coordinates present; Q/V coords absent (all-K partial) ──────
        k_count = sum(1 for n in nodes if n.attrib.get("x") is not None)
        q_count = sum(1 for n in nodes if n.attrib.get("qx") is not None)
        v_count = sum(1 for n in nodes if n.attrib.get("vx") is not None)
        # K coords present is the positive assertion; Q/V absence is a warn
        cases.append(("qkv_k_present", k_count > 0))
        if q_count == 0 or v_count == 0:
            print(f"warn | QKV partial: K={k_count}, Q={q_count}, V={v_count}"
                  " — all-K is the current state; Q/V not yet populated",
                  file=sys.stderr)

        return verdict("field_vectors", cases)
    except Exception as exc:
        return verdict("field_vectors", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
