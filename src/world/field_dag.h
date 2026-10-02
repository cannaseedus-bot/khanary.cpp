#pragma once
#include "world_state.h"
#include "tensor_substrate.h"
#include "vector_projection.h"
#include "../micronaut/micronaut_registry.h"
#include "../gpu/d3d11_device.h"
#include "../gpu/compute_dispatch.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <cstdint>

namespace khanary::world {

// One entry in a tensor or pipeline-stage fold_map.
struct FoldEntry {
    std::string phase_name;
    std::string hook;     // "on_read" | "on_write" | "on_gc" | ""
    std::string monitor;  // condition expression e.g. "gradient_norm > 10.0", or ""
    std::string action;   // "warn_clip" | "promote_checkpoint" | ""
};

struct FieldNode {
    // "track" = micronaut (one dominant phase, circle in SVG)
    // "tensor" = living tensor (multi-phase arc in SVG)
    // "pipeline_stage" = C6 pipeline stage (rect in SVG)
    // "ontology" = top-level ontology class (rect in SVG)
    std::string kind = "track";

    std::string track_id;
    std::string phase_name;  // dominant / primary phase
    std::string face_name;
    std::string block_name;
    std::string category;    // "executive"|"cognitive"|"storage"|"creative"|""
    float       confidence;
    float       static_weight = 1.0f;
    float       health = 1.0f;
    float       wcr_prior = 1.0f;
    float       x, y, z;
    std::string vector_source = "world_position";
    bool        coordinator;
    std::vector<std::string> effects;
    // field geometry — populated from GML overlay; defaults safe for non-GML nodes
    float       local_entropy    = 0.0f;  // H(x): point-wise Shannon uncertainty
    float       geodesic_entropy = 0.0f;  // H_G:  updated by geodesic_propagator dispatch
    float       unruh_temp       = 0.0f;  // T:    holographic boundary temperature
    float       divergence       = 0.0f;  // div V: negative = gravity well pull
    float       circumradius_R   = 0.0f;  // R:    triangle circumcircle horizon
    float       inradius_r       = 0.0f;  // r:    triangle incircle compression boundary
    uint32_t    singularity_type = 0u;    // 0=ordinary 1=source 2=sink 3=saddle
    std::string provider_id;             // from GML overlay; empty for non-GML nodes
    std::string parameter_id;            // GML parameter_id if present
    // SVG-3D weight layer: intensity per AtomicDOMBlock (Perceive..Evolve)
    std::array<float, 7> block_intensities{};
    // Multi-phase presence declaration (tensors + pipeline stages)
    std::vector<FoldEntry> fold_map;
};

// Directed (or undirected) arc in the semantic field graph
struct FieldEdge {
    std::string from;
    std::string to;
    // "delegate" | "horizontal_fold" | "vertical_fold"
    std::string type;
    // Relation token when built from relations data (canonical + compatibility aliases).
    // Canonical examples: @is_a, @part_of, @supports, @requires, @governs, @delegates_to.
    std::string verb;
    // Canonical phase hint for verb relation family (Pop/Wo/Sek/Ch'en).
    std::string relation_phase;
    float       weight;       // cos(Δθ): 1.0 / 0.5 / -1.0 / ...
    float       delta_theta;  // degrees
    bool        xcfe_required;
    bool        directed = true;  // false → horizontal relation (Δθ=0)
    // field geometry — populated from GML overlay
    float       length = 0.0f;              // L_e: metric interval distance
    float       vx = 0.0f, vy = 0.0f, vz = 0.0f;           // tangent velocity
    float       eh_x = 0.0f, eh_y = 0.0f, eh_z = 0.0f;     // entropy gradient
};

struct FieldSvgProjectionNode {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float phase_angle = 0.0f;
    std::array<float, 7> bar_heights{};
    bool  bar_heights_valid = false;
};

struct FieldSvgProjection {
    std::vector<FieldSvgProjectionNode> nodes;
    bool gpu_projected = false;
};

// Result of a GPU geodesic propagator dispatch.
struct PropagatorDispatchResult {
    std::vector<float> geodesic_entropy; // per-node, indexed by nodes() order
    bool gpu_ran = false;
};

struct HyperbolicAttentionGpuBuffer {
    Microsoft::WRL::ComPtr<ID3D11Buffer> structured_buffer;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    uint32_t record_count = 0;
    uint32_t capacity_records = 0;
    uint32_t stride_bytes = sizeof(HyperbolicAttentionRecord64);
};

// Builds the 2-D semantic field DAG from WorldState + MicronautRegistry.
// Nodes = tracks; edges = declared delegate arcs + C6 fold arcs.
// Serializes to XML-DAG format suitable for PRIMEOS WebView2 rendering.
class FieldDag {
public:
    FieldDag(const WorldState& ws, const micronaut::MicronautRegistry& reg);

    void build();
    // Overlay GML graph nodes/edges onto the DAG. Must be called AFTER build().
    void load_gml(const std::string& path);
    std::string to_xml() const;
    // SVG output: field substrate + tensor arcs (kind=tensor) + track circles + SVG-3D blocks
    std::string to_svg(const FieldSvgProjection* projection = nullptr) const;

    // Register a living tensor node with its fold_map phases.
    // Adds the node and its @live / @self_aware entries; no edges auto-added.
    void add_tensor(const std::string& id,
                    const std::string& shape,
                    std::vector<FoldEntry> entries);

    const std::vector<FieldNode>& nodes() const { return nodes_; }
    const std::vector<FieldEdge>& edges() const { return edges_; }
    const std::unordered_map<std::string, TensorSubstrate>& substrates() const
        { return substrates_; }

    // ── GPU geodesic propagator ───────────────────────────────────────────────
    // Bind NodeIn/EdgeIn structured buffers, dispatch geodesic_propagator.hlsl,
    // read back updated geodesic_entropy onto FieldNodes.
    // cso: pre-compiled CSO blob for geodesic_propagator.hlsl.
    // Returns empty result if dev is invalid or cso is null.
    PropagatorDispatchResult dispatch_geodesic_propagator(
        gpu::D3D11Device& dev,
        ID3DBlob*         cso,
        float             beta_inv,
        float             kb_boltzmann = 1.0f);

    // ── Ch'en Raychaudhuri gate ───────────────────────────────────────────────
    // Integrates the discrete geodesic entropy rate along `path` (track_ids).
    // Returns true = admissible; false = Ch'en drops the branch.
    bool check_raychaudhuri_gate(
        const std::vector<std::string>& path,
        float hg_max = 4.0f) const;

    // ── DirectionStore → cbuffer feed ─────────────────────────────────────────
    // Translates a co-occurrence delta dM into an updated beta_inv for
    // PropagatorConstants.  Called by the Factory after mesh re-triangulation.
    static float compute_beta_inv_from_delta_m(
        float    delta_m,
        float    base_beta_inv,
        uint32_t step);

    // ── Hyperbolic key-grammar event → attention SRV bridge ────────────────
    // Packs future-sheet HyperbolicEventVector rows into the existing
    // attention.hlsl 64-byte SemanticRecord slot.  This is a projection-only
    // GPU upload surface; Ch'en/Xul still own verification and persistence.
    static bool upload_hyperbolic_attention_records(
        gpu::D3D11Device& dev,
        const std::vector<HyperbolicAttentionRecord64>& records,
        HyperbolicAttentionGpuBuffer& out,
        uint32_t extra_capacity_records = 32);

    static bool upload_hyperbolic_trajectories(
        gpu::D3D11Device& dev,
        const std::vector<HyperbolicEventVector>& events,
        const std::unordered_map<std::string, float>& wcr_w_map,
        const std::unordered_map<std::string, float>& wcr_c_map,
        const std::unordered_map<std::string, float>& wcr_r_map,
        HyperbolicAttentionGpuBuffer& out,
        uint32_t extra_capacity_records = 32);

private:
    const WorldState& ws_;
    const micronaut::MicronautRegistry& reg_;
    std::vector<FieldNode> nodes_;
    std::vector<FieldEdge> edges_;
    std::unordered_map<std::string, TensorSubstrate> substrates_;
    static uint64_t next_substrate_id_;

    static std::string xml_escape(std::string_view s);

    // SVG helpers
    static constexpr float SVG_CX    = 512.0f;
    static constexpr float SVG_CY    = 400.0f;
    static constexpr float SVG_SCALE = 150.0f;

    struct SVGPos { float x, y; };
    // Track position (includes z-slot diagonal offset)
    static SVGPos to_svg_pos(const FieldNode& n);
    // Raw C₆ phase position on circle, no z-offset (for tensor arcs)
    static SVGPos phase_svg_pos(kuhul::Phase p);
    static std::string svg_edge_path(const SVGPos& a, const SVGPos& b);
    std::string svg_tensor_block(
        const FieldNode& n,
        const SVGPos& p,
        const FieldSvgProjectionNode* projected) const;
    std::string svg_tensor_arc(const FieldNode& n) const;
};

} // namespace khanary::world
