"""Verify packaged geometry integrity and declared KNU bindings; no GPU claim."""
import hashlib
import json
import struct
from pathlib import Path
from conformance_common import verdict


STB_MAGIC = 0x30425453
STB_TENSORS = (
    ("nodes_time.bin", 0x00010000, "nodes", 0),
    ("nodes_freq.bin", 0x00010001, "nodes", 0),
    ("nodes_energy.bin", 0x00010002, "nodes", 0),
    ("edges.bin", 0x00020303, "edges", 2),
    ("csr_index.bin", 0x00010304, "csr_index", 0),
    ("csr_neighbors.bin", 0x00010305, "csr_entries", 0),
    ("experts.bin", 0x00010306, "nodes", 0),
)


def check_stb_payload(cases, prefix, run_dir, stb, nodes, edges, csr_entries):
    counts = {
        "nodes": nodes,
        "edges": edges,
        "csr_index": nodes + 1,
        "csr_entries": csr_entries,
    }
    magic, version_count, _r0, _r1, data_offset, _r2, total_bytes, _r3 = struct.unpack(
        "<IIIIIIII", stb[:32]
    )
    cases.append((prefix + "/stb_magic", magic == STB_MAGIC))
    cases.append((prefix + "/stb_version", (version_count & 0xffff) == 1))
    cases.append((prefix + "/stb_tensor_count", (version_count >> 16) == len(STB_TENSORS)))
    cases.append((prefix + "/stb_total_bytes", total_bytes == len(stb)))
    cases.append((prefix + "/stb_data_offset", data_offset == 256))
    for index, (filename, tensor_type, count_key, components) in enumerate(STB_TENSORS):
        descriptor = 32 + index * 32
        got_type, offset, _pad0, byte_count, _pad1, item_count, got_components, _pad2 = struct.unpack(
            "<IIIIIIII", stb[descriptor:descriptor + 32]
        )
        payload = (run_dir / filename).read_bytes()
        cases.append((f"{prefix}/stb_tensor_{index}_type", got_type == tensor_type))
        cases.append((f"{prefix}/stb_tensor_{index}_count", item_count == counts[count_key]))
        cases.append((f"{prefix}/stb_tensor_{index}_components", got_components == components))
        cases.append((f"{prefix}/stb_tensor_{index}_bytes", byte_count == len(payload)))
        cases.append((f"{prefix}/stb_tensor_{index}_payload", stb[offset:offset + byte_count] == payload))


def main():
    cases = []
    try:
        root = Path(__file__).resolve().parents[1] / "models/khanary-geometry-v0.3.0"
        model = json.loads((root / "MODEL.json").read_text(encoding="utf-8"))
        mesh = (root / model["data"]["file"]).read_bytes()
        cases.append(("mesh_bytes", len(mesh) == model["data"]["bytes"]))
        cases.append(("mesh_sha256", hashlib.sha256(mesh).hexdigest() == model["data"]["sha256"]))
        local = model.get("local_artifacts", {})
        cases.append(("local_artifacts_projection_only",
                      "projection evidence" in local.get("authority_boundary", "") and
                      "do not verify semantic truth" in local.get("authority_boundary", "")))
        for artifact_id in ("compiler", "source_audio", "stb_packer"):
            artifact = local.get(artifact_id, {})
            payload = (root / artifact.get("file", "__missing__")).read_bytes()
            cases.append((artifact_id + "/bytes", len(payload) == artifact.get("bytes")))
            cases.append((artifact_id + "/sha256",
                          hashlib.sha256(payload).hexdigest() == artifact.get("sha256")))
        for run in local.get("brain_runs", []):
            prefix = "brain_run/" + run["id"]
            run_dir = root / run["dir"]
            header = (run_dir / "tensor_header.bin").read_bytes()
            nodes, edges, csr_entries, experts, feature_count = struct.unpack("IIIII", header)
            cases.append((prefix + "/tensor_header",
                          nodes == run["nodes"] and
                          edges == run["edges"] and
                          csr_entries == run["csr_entries"] and
                          experts == run["experts"] and
                          feature_count == run["feature_count"]))
            cases.append((prefix + "/node_feature_sizes",
                          (run_dir / "nodes_time.bin").stat().st_size == nodes * 4 and
                          (run_dir / "nodes_freq.bin").stat().st_size == nodes * 4 and
                          (run_dir / "nodes_energy.bin").stat().st_size == nodes * 4 and
                          (run_dir / "experts.bin").stat().st_size == nodes * 4))
            cases.append((prefix + "/edge_csr_sizes",
                          (run_dir / "edges.bin").stat().st_size == edges * 2 * 4 and
                          (run_dir / "csr_index.bin").stat().st_size == (nodes + 1) * 4 and
                          (run_dir / "csr_neighbors.bin").stat().st_size == csr_entries * 4))
            stb = (root / run["stb_file"]).read_bytes()
            cases.append((prefix + "/stb_bytes", len(stb) == run["stb_bytes"]))
            cases.append((prefix + "/stb_sha256",
                          hashlib.sha256(stb).hexdigest() == run["stb_sha256"]))
            check_stb_payload(cases, prefix, run_dir, stb, nodes, edges, csr_entries)
            if "brain_file" in run:
                brain = (root / run["brain_file"]).read_bytes()
                cases.append((prefix + "/brain_bytes", len(brain) == run["brain_bytes"]))
                cases.append((prefix + "/brain_sha256",
                              hashlib.sha256(brain).hexdigest() == run["brain_sha256"]))
                magic, version, b_nodes, b_edges, b_csr, b_experts = struct.unpack("IIIIII", brain[:24])
                cases.append((prefix + "/brain_header",
                              magic == 0x42524E31 and version == 1 and
                              b_nodes == nodes and b_edges == edges and
                              b_csr == csr_entries and b_experts == experts))
            if "proof_file" in run:
                proof = json.loads((root / run["proof_file"]).read_text(encoding="utf-8"))
                cases.append((prefix + "/proof_identity",
                              proof.get("id") == run["id"] and
                              proof.get("artifacts", {}).get("stb_sha256") == run["stb_sha256"]))
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
