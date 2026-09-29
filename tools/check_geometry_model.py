"""Verify packaged geometry integrity and declared KNU bindings; no GPU claim."""
import hashlib
import json
from pathlib import Path
from conformance_common import verdict


def main():
    cases = []
    try:
        root = Path(__file__).resolve().parents[1] / "models/khanary-geometry-v0.3.0"
        model = json.loads((root / "MODEL.json").read_text(encoding="utf-8"))
        mesh = (root / model["data"]["file"]).read_bytes()
        cases.append(("mesh_bytes", len(mesh) == model["data"]["bytes"]))
        cases.append(("mesh_sha256", hashlib.sha256(mesh).hexdigest() == model["data"]["sha256"]))
        for op, spec in sorted(model["ops"].items()):
            program = json.loads((root / spec["knu"]).read_text(encoding="utf-8"))
            cases.append((op + "/op_binding", program["op"] == op and bool(program["stream"])))
            for index, instruction in enumerate(program["stream"]):
                word = instruction["word"]
                decoded = instruction["decoded"]
                cases.append((f"{op}/word_{index}", int(instruction["word_hex"], 16) == word and
                              (word >> 28) == model["knu_ver_field"] == decoded["ver"] and
                              ((word >> 20) & 255) == model["glyphs"][spec["glyph"]] == decoded["glyph_id"] and
                              instruction["glyph"] == spec["glyph"]))
            for kernel in spec["kernels"]:
                source = (root / kernel).read_text(encoding="utf-8")
                cases.append((op + "/" + Path(kernel).suffix[1:] + "_entrypoint",
                              "main(" in source and ("numthreads" in source or "@workgroup_size" in source)))
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
        registry = Registry()
        for field in ("schema", "schema_types"):
            path = root / model["data"][field]
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            resource = Resource.from_contents(schema)
            for uri in (path.name, path.as_uri(), schema["$id"]):
                registry = registry.with_resource(uri, resource)
        schema = json.loads((root / model["data"]["schema"]).read_text(encoding="utf-8"))
        dataset = json.loads((root / model["data"]["dataset"]).read_text(encoding="utf-8"))
        cases.append(("birdsong_schema", Draft202012Validator(schema, registry=registry).is_valid(dataset)))
        return verdict("geometry_assets", cases)
    except Exception as exc:
        return verdict("geometry_assets", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
