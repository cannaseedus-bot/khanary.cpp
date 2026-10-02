"""Validate the semantic attention shader projection contract.

The runtime has two copies of the shader contract:

* data/shaders/attention.hlsl is the source asset.
* src/gpu/gram_gpu_scorer.cpp embeds the shader for D3DCompile.

This check keeps the Micronaut tensor-slot fields, cbuffer mix lane, and CPU
record stride aligned.  The shader remains a projection backend: tensor slots
may bias Yax admission, but they do not verify semantic truth.

The hot SemanticRecord path stays 64 bytes.  Continuous geodesic/entropy/mesh
state is carried in an optional 36-byte STBGeodesicRecord side buffer at t1.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTENTION_HLSL = ROOT / "data" / "shaders" / "attention.hlsl"
GRAM_GPU_SCORER = ROOT / "src" / "gpu" / "gram_gpu_scorer.cpp"

EXPECTED_RECORD_FIELDS = [
    "W",
    "C",
    "R",
    "relation_weight",
    "fold_weight",
    "phase_weight",
    "policy_ok",
    "capable",
    "xcfe_ok",
    "tensor_mass",
    "tensor_alignment",
    "tensor_health",
    "tensor_enabled",
    "reserved0",
    "reserved1",
    "reserved2",
]

EXPECTED_CBUFFER_FIELDS = [
    "record_count",
    "admission_threshold",
    "relation_mix",
    "tensor_mix",
    "geodesic_mix",
    "beta_diffusion",
    "reserved_c0",
    "reserved_c1",
]

EXPECTED_LORENTZ_CBUFFER_FIELDS = [
    "boost_row0",
    "boost_row1",
    "boost_row2",
    "boost_row3",
    "lorentz_enabled",
    "boost_alignment_mix",
    "boost_mass_scale",
    "boost_reserved",
]

EXPECTED_GEODESIC_FIELDS = [
    "x",
    "y",
    "z",
    "local_entropy",
    "geodesic_entropy",
    "unruh_temp",
    "divergence",
    "circumradius_R",
    "inradius_r",
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL | attention-shader-contract: {message}")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def strip_line_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def extract_block(text: str, header: str) -> str:
    pattern = re.compile(rf"{re.escape(header)}\s*\{{(?P<body>.*?)\}};", re.DOTALL)
    match = pattern.search(text)
    if not match:
        fail(f"missing block: {header}")
    return match.group("body")


def extract_float_fields(block: str) -> list[str]:
    fields: list[str] = []
    for line in strip_line_comments(block).splitlines():
        match = re.search(r"\bfloat\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", line)
        if match:
            fields.append(match.group(1))
    return fields


def extract_uint_float_cbuffer_fields(block: str) -> list[str]:
    fields: list[str] = []
    for line in strip_line_comments(block).splitlines():
        match = re.search(r"\b(?:uint|float|float4)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", line)
        if match:
            fields.append(match.group(1))
    return fields


def extract_embedded_shader(cpp: str) -> str:
    match = re.search(
        r'static\s+const\s+char\*\s+k_attention_hlsl\s*=\s*R"HLSL\((?P<body>.*?)\)HLSL";',
        cpp,
        re.DOTALL,
    )
    if not match:
        fail("embedded k_attention_hlsl raw string not found")
    return match.group("body")


def check_shader(label: str, shader: str) -> None:
    cbuffer = extract_block(shader, "cbuffer SemanticAttentionConstants : register(b0)")
    cbuffer_fields = extract_uint_float_cbuffer_fields(cbuffer)
    if cbuffer_fields != EXPECTED_CBUFFER_FIELDS:
        fail(f"{label} cbuffer fields {cbuffer_fields!r} != {EXPECTED_CBUFFER_FIELDS!r}")

    lorentz_cbuffer = extract_block(shader, "cbuffer LorentzTransitionTensor : register(b1)")
    lorentz_cbuffer_fields = extract_uint_float_cbuffer_fields(lorentz_cbuffer)
    if lorentz_cbuffer_fields != EXPECTED_LORENTZ_CBUFFER_FIELDS:
        fail(f"{label} Lorentz cbuffer fields {lorentz_cbuffer_fields!r} != {EXPECTED_LORENTZ_CBUFFER_FIELDS!r}")

    record = extract_block(shader, "struct SemanticRecord")
    record_fields = extract_float_fields(record)
    if record_fields != EXPECTED_RECORD_FIELDS:
        fail(f"{label} SemanticRecord fields {record_fields!r} != {EXPECTED_RECORD_FIELDS!r}")

    geodesic = extract_block(shader, "struct GeodesicMetricRecord")
    geodesic_fields = extract_float_fields(geodesic)
    if geodesic_fields != EXPECTED_GEODESIC_FIELDS:
        fail(f"{label} GeodesicMetricRecord fields {geodesic_fields!r} != {EXPECTED_GEODESIC_FIELDS!r}")

    required_fragments = [
        "float tensor_slot_modifier(SemanticRecord rec)",
        "if (rec.tensor_enabled < 0.5f)",
        "float4 tensor_four_vector(SemanticRecord rec)",
        "float4 lorentz_contract(float4 v)",
        "u.x = dot(boost_row0, v);",
        "u.y = dot(boost_row1, v);",
        "u.z = dot(boost_row2, v);",
        "u.w = dot(boost_row3, v);",
        "float interval = -u.x * u.x + spatial2;",
        "saturate(rec.tensor_mass)",
        "saturate(rec.tensor_alignment)",
        "saturate(rec.tensor_health)",
        "StructuredBuffer<GeodesicMetricRecord> GeodesicMetrics : register(t1);",
        "float geodesic_sidecar_modifier(uint idx)",
        "if (geodesic_mix < 0.5f)",
        "geo.circumradius_R >= 2.0f * geo.inradius_r",
        "BaseScore[idx] = s;",
        "rec.policy_ok >= 0.5f && rec.capable >= 0.5f && rec.xcfe_ok >= 0.5f",
    ]
    for fragment in required_fragments:
        if fragment not in shader:
            fail(f"{label} missing shader fragment: {fragment}")

    if "lerp(1.0f,tensor_modifier,saturate(tensor_mix))" not in compact(shader):
        fail(f"{label} missing tensor_mix blend")
    if "lerp(1.0f,geodesic_modifier,saturate(geodesic_mix))" not in compact(shader):
        fail(f"{label} missing geodesic_mix blend")


def main() -> int:
    shader = ATTENTION_HLSL.read_text(encoding="utf-8")
    cpp = GRAM_GPU_SCORER.read_text(encoding="utf-8")
    embedded = extract_embedded_shader(cpp)

    check_shader("source", shader)
    check_shader("embedded", embedded)

    cpu_required = [
        "float xcfe_ok, tensor_mass, tensor_alignment, tensor_health;",
        "float tensor_enabled, r0, r1, r2;",
        "static_assert(sizeof(SemanticRecordCPU) == 64",
        "struct LorentzCB {",
        "static_assert(sizeof(LorentzCB) == 80",
        "LorentzCB make_lorentz_cb(const LorentzBoostRows* boost)",
        "lorentz_cbuf_",
        '"gram_records",  TensorFormat::F32, N, 64',
        '"gram_geodesic", TensorFormat::F32, N, 36',
        "std::vector<STBGeodesicRecord> neutral_geodesics;",
        "const std::vector<STBGeodesicRecord>* geodesics = geodesic_metrics;",
        "float geodesic_mix;",
        "float beta_diffusion;",
        "const LorentzBoostRows* lorentz_boost",
        "const std::unordered_map<std::string, std::array<float, 3>>* vector_overrides",
        "ctx->CSSetConstantBuffers(0, 2, raw_cbs);",
    ]
    for fragment in cpu_required:
        if fragment not in cpp:
            fail(f"gram_gpu_scorer.cpp missing CPU contract fragment: {fragment}")

    if not re.search(
        r"float\s+admission_threshold,\s*float\s+relation_mix,\s*float\s+tensor_mix,\s*const\s+std::vector<STBGeodesicRecord>\*\s+geodesic_metrics,\s*float\s+geodesic_mix,\s*float\s+beta_diffusion,\s*const\s+LorentzBoostRows\*\s+lorentz_boost,\s*const\s+std::unordered_map<std::string,\s*std::array<float,\s*3>>\*\s+vector_overrides\)",
        cpp,
        re.DOTALL,
    ):
        fail("gram_gpu_scorer.cpp score_batch signature missing geodesic/Lorentz/vector override parameters")

    print("PASS | attention-shader-contract: source+embedded tensor-slot+geodesic+Lorentz layout aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
