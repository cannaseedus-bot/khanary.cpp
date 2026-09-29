#include "svg_compute_runtime.h"
#include "tensor_buffer.h"
#include "../klsl/d3dcompile_bridge.h"
#include "../kuhul/phase_engine.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace khanary::gpu {

namespace {

struct SvgComputeConstants {
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t phase_count;
    uint32_t pad0;

    float screen_center[2];
    float world_scale;
    float pad1;

    float z_to_x;
    float z_to_y;
    float bar_min;
    float bar_max;

    float edge_len_min;
    float edge_len_max;
    float pad2[2];
};

struct GpuSvgTensorNode {
    float    position[3];
    float    activation;
    float    confidence;
    float    relevance;
    float    pattern_strength;
    float    score;
    uint32_t phase;
    uint32_t node_kind;
    float    block0123[4];
    float    block456x[4];
};

struct GpuSvgTensorEdge {
    uint32_t from_index;
    uint32_t to_index;
    float    association_weight;
    float    relation_confidence;
};

struct Float4 {
    float x;
    float y;
    float z;
    float w;
};

static_assert(sizeof(Float4) == 16, "Float4 must be 16 bytes");

uint32_t phase_index_from_name(const std::string& phase_name) {
    kuhul::Phase p = kuhul::Phase::Pop;
    if (kuhul::phase_from_name(phase_name, p))
        return static_cast<uint32_t>(p);
    return 0u;
}

uint32_t node_kind_index(const std::string& kind) {
    if (kind == "tensor") return 1u;
    if (kind == "pipeline_stage") return 2u;
    return 0u; // track / default
}

float safe_confidence(const world::FieldNode& n) {
    if (n.block_intensities[4] > 0.0f)
        return std::clamp(n.block_intensities[4], 0.0f, 1.0f);
    return std::clamp(n.confidence, 0.0f, 1.0f);
}

float edge_relation_confidence(const world::FieldEdge& e) {
    float c = 0.50f;
    if (e.type == "delegate") c = 0.85f;
    else if (e.type == "horizontal_fold") c = 0.65f;
    else if (e.type == "vertical_fold") c = 0.55f;
    if (!e.xcfe_required) c += 0.05f;
    return std::clamp(c, 0.0f, 1.0f);
}

} // namespace

SvgComputeRuntime::SvgComputeRuntime(D3D11Device& dev)
    : dev_(dev), node_dispatch_(dev), edge_dispatch_(dev) {}

bool SvgComputeRuntime::compile_shader_entry(
    const std::string& source,
    const char* entry,
    ComputeDispatch& dispatch,
    std::string* error_out)
{
    klsl::D3DCompileBridge compiler;
    auto r = compiler.compile(source, entry, "cs_5_0");
    if (!r.ok || !r.bytecode) {
        if (error_out) {
            *error_out = r.error_msg.empty()
                ? ("D3DCompile failed for entry: " + std::string(entry))
                : r.error_msg;
        }
        return false;
    }
    if (!dispatch.set_shader(r.bytecode.Get())) {
        if (error_out)
            *error_out = "CreateComputeShader failed for entry: " + std::string(entry);
        return false;
    }
    return true;
}

bool SvgComputeRuntime::init(const std::string& shader_path) {
    shader_path_ = shader_path;
    ready_ = false;
    last_error_.clear();

    if (!dev_.is_valid()) {
        last_error_ = "D3D11 device unavailable";
        return false;
    }

    std::ifstream in(shader_path_, std::ios::binary);
    if (!in) {
        last_error_ = "Cannot open shader file: " + shader_path_;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string source = ss.str();
    if (source.empty()) {
        last_error_ = "Shader file is empty: " + shader_path_;
        return false;
    }

    if (!compile_shader_entry(source, "CSProjectNodes", node_dispatch_, &last_error_))
        return false;
    if (!compile_shader_entry(source, "CSProjectEdges", edge_dispatch_, &last_error_))
        return false;

    ready_ = true;
    return true;
}

bool SvgComputeRuntime::project(
    const std::vector<world::FieldNode>& nodes,
    const std::vector<world::FieldEdge>& edges,
    world::FieldSvgProjection& out_projection)
{
    out_projection.nodes.clear();
    out_projection.gpu_projected = false;

    if (!ready_) return false;
    if (nodes.empty()) return false;
    if (nodes.size() > static_cast<size_t>(UINT32_MAX)) return false;
    if (edges.size() > static_cast<size_t>(UINT32_MAX)) return false;

    std::unordered_map<std::string, uint32_t> node_index;
    node_index.reserve(nodes.size());

    std::vector<GpuSvgTensorNode> gpu_nodes(nodes.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(nodes.size()); ++i) {
        const auto& n = nodes[i];
        node_index[n.track_id] = i;

        auto& gn = gpu_nodes[i];
        gn.position[0] = n.x;
        gn.position[1] = n.y;
        gn.position[2] = n.z;
        gn.activation = std::clamp(n.block_intensities[0], 0.0f, 1.0f);
        gn.confidence = safe_confidence(n);
        gn.relevance = std::clamp(
            std::max(n.block_intensities[3], n.wcr_prior), 0.0f, 1.0f);
        gn.pattern_strength = std::clamp(
            std::max(n.block_intensities[2], n.static_weight), 0.0f, 1.0f);
        gn.score = std::clamp(n.static_weight * n.health * gn.confidence, 0.0f, 1.0f);
        gn.phase = phase_index_from_name(n.phase_name);
        gn.node_kind = node_kind_index(n.kind);
        gn.block0123[0] = std::clamp(n.block_intensities[0], 0.0f, 1.0f);
        gn.block0123[1] = std::clamp(n.block_intensities[1], 0.0f, 1.0f);
        gn.block0123[2] = std::clamp(n.block_intensities[2], 0.0f, 1.0f);
        gn.block0123[3] = std::clamp(n.block_intensities[3], 0.0f, 1.0f);
        gn.block456x[0] = std::clamp(n.block_intensities[4], 0.0f, 1.0f);
        gn.block456x[1] = std::clamp(n.block_intensities[5], 0.0f, 1.0f);
        gn.block456x[2] = std::clamp(n.block_intensities[6], 0.0f, 1.0f);
        gn.block456x[3] = 0.0f;
    }

    std::vector<GpuSvgTensorEdge> gpu_edges(edges.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(edges.size()); ++i) {
        const auto& e = edges[i];
        auto& ge = gpu_edges[i];
        auto it_from = node_index.find(e.from);
        auto it_to = node_index.find(e.to);
        if (it_from == node_index.end() || it_to == node_index.end()) {
            ge.from_index = 0u;
            ge.to_index = 0u;
            ge.association_weight = 0.0f;
            ge.relation_confidence = 0.0f;
            continue;
        }
        ge.from_index = it_from->second;
        ge.to_index = it_to->second;
        ge.association_weight = std::clamp((e.weight + 1.0f) * 0.5f, 0.0f, 1.0f);
        ge.relation_confidence = edge_relation_confidence(e);
    }

    const uint32_t node_count = static_cast<uint32_t>(gpu_nodes.size());
    const uint32_t edge_count = static_cast<uint32_t>(gpu_edges.size());

    TensorDesc d_nodes;
    d_nodes.name = "svg_nodes";
    d_nodes.format = TensorFormat::F32;
    d_nodes.elem_count = node_count;
    d_nodes.stride_bytes = static_cast<uint32_t>(sizeof(GpuSvgTensorNode));
    TensorBuffer b_nodes(dev_, d_nodes);
    if (!b_nodes.is_valid()) return false;
    if (!b_nodes.upload(gpu_nodes.data(), sizeof(GpuSvgTensorNode) * gpu_nodes.size()))
        return false;

    TensorDesc d_node_proj;
    d_node_proj.name = "svg_node_proj";
    d_node_proj.format = TensorFormat::F32;
    d_node_proj.elem_count = node_count;
    d_node_proj.stride_bytes = static_cast<uint32_t>(sizeof(Float4));
    TensorBuffer b_node_proj(dev_, d_node_proj);
    TensorBuffer b_node_bars0(dev_, d_node_proj);
    TensorBuffer b_node_bars1(dev_, d_node_proj);
    TensorBuffer b_node_state(dev_, d_node_proj);
    if (!b_node_proj.is_valid() || !b_node_bars0.is_valid() ||
        !b_node_bars1.is_valid() || !b_node_state.is_valid())
        return false;

    TensorBuffer b_edges;
    TensorBuffer b_edge_proj;
    TensorBuffer b_edge_state;
    if (edge_count > 0) {
        TensorDesc d_edges;
        d_edges.name = "svg_edges";
        d_edges.format = TensorFormat::F32;
        d_edges.elem_count = edge_count;
        d_edges.stride_bytes = static_cast<uint32_t>(sizeof(GpuSvgTensorEdge));
        b_edges = TensorBuffer(dev_, d_edges);
        if (!b_edges.is_valid()) return false;
        if (!b_edges.upload(gpu_edges.data(), sizeof(GpuSvgTensorEdge) * gpu_edges.size()))
            return false;

        d_node_proj.elem_count = edge_count;
        b_edge_proj = TensorBuffer(dev_, d_node_proj);
        b_edge_state = TensorBuffer(dev_, d_node_proj);
        if (!b_edge_proj.is_valid() || !b_edge_state.is_valid())
            return false;
    }

    SvgComputeConstants c{};
    c.node_count = node_count;
    c.edge_count = edge_count;
    c.phase_count = 6u;
    c.pad0 = 0u;
    c.screen_center[0] = 512.0f;
    c.screen_center[1] = 400.0f;
    c.world_scale = 150.0f;
    c.pad1 = 0.0f;
    c.z_to_x = 22.0f;
    c.z_to_y = 12.0f;
    c.bar_min = 3.0f;
    c.bar_max = 28.0f;
    c.edge_len_min = 30.0f;
    c.edge_len_max = 220.0f;
    c.pad2[0] = 0.0f;
    c.pad2[1] = 0.0f;

    node_dispatch_.set_constants(&c, sizeof(c));
    {
        std::vector<ID3D11UnorderedAccessView*> uavs = {
            b_node_proj.uav(), b_node_bars0.uav(), b_node_bars1.uav(), b_node_state.uav()
        };
        std::vector<ID3D11ShaderResourceView*> srvs = {
            b_nodes.srv()
        };
        node_dispatch_.run_for_elements(uavs, srvs, node_count, 64u);
    }

    if (edge_count > 0) {
        edge_dispatch_.set_constants(&c, sizeof(c));
        std::vector<ID3D11UnorderedAccessView*> uavs(6, nullptr);
        uavs[4] = b_edge_proj.uav();
        uavs[5] = b_edge_state.uav();
        std::vector<ID3D11ShaderResourceView*> srvs = {
            b_nodes.srv(), b_edges.srv()
        };
        edge_dispatch_.run_for_elements(uavs, srvs, edge_count, 64u);
    }

    std::vector<Float4> node_proj(node_count);
    std::vector<Float4> bars0(node_count);
    std::vector<Float4> bars1(node_count);
    if (!b_node_proj.readback(node_proj.data(), node_proj.size() * sizeof(Float4)))
        return false;
    if (!b_node_bars0.readback(bars0.data(), bars0.size() * sizeof(Float4)))
        return false;
    if (!b_node_bars1.readback(bars1.data(), bars1.size() * sizeof(Float4)))
        return false;

    out_projection.nodes.resize(node_count);
    for (uint32_t i = 0; i < node_count; ++i) {
        auto& pn = out_projection.nodes[i];
        pn.x = node_proj[i].x;
        pn.y = node_proj[i].y;
        pn.z = node_proj[i].z;
        pn.phase_angle = node_proj[i].w;
        pn.bar_heights[0] = bars0[i].x;
        pn.bar_heights[1] = bars0[i].y;
        pn.bar_heights[2] = bars0[i].z;
        pn.bar_heights[3] = bars0[i].w;
        pn.bar_heights[4] = bars1[i].x;
        pn.bar_heights[5] = bars1[i].y;
        pn.bar_heights[6] = bars1[i].z;
        pn.bar_heights_valid = true;
    }

    out_projection.gpu_projected = true;
    return true;
}

} // namespace khanary::gpu
