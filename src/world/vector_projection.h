#pragma once
#include "../micronaut/mx2lm-ngram.h"
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace khanary::world {

// Future-sheet event coordinate used by key-grammar hyperbolic projections.
// This is projection evidence only; it does not carry semantic authority.
struct HyperbolicEventVector {
    std::string key;
    std::string track;
    std::string phase;
    std::string event_role;
    std::array<float, 4> coords{1.0f, 0.0f, 0.0f, 0.0f}; // x0,x1,x2,x3
    std::array<float, 3> direction{0.0f, 0.0f, 0.0f};
    float rapidity_rho = 0.0f;
    float minkowski_interval = 0.0f;
    std::string interval_class;
    bool semantic_authority = false;
    bool policy_ok = true;
    bool capable = true;
    bool xcfe_ok = true;
    float tensor_health = 1.0f;
    bool tensor_enabled = true;
};

// CPU mirror of attention.hlsl SemanticRecord.  The first 13 fields are
// consumed by the active attention shader.  reserved0..2 carry raw x1/x2/x3
// hyperbolic coordinates for later Lorentz-boost shader work without widening
// the current 64-byte stride.
struct alignas(16) HyperbolicAttentionRecord64 {
    float W, C, R, relation_weight;
    float fold_weight, phase_weight, policy_ok, capable;
    float xcfe_ok, tensor_mass, tensor_alignment, tensor_health;
    float tensor_enabled, reserved0, reserved1, reserved2;
};
static_assert(sizeof(HyperbolicAttentionRecord64) == 64,
              "HyperbolicAttentionRecord64 must match attention.hlsl SemanticRecord");
static_assert(alignof(HyperbolicAttentionRecord64) == 16,
              "HyperbolicAttentionRecord64 must keep a 16-byte alignment boundary");

HyperbolicAttentionRecord64 make_hyperbolic_attention_record(
    const HyperbolicEventVector& event,
    float w,
    float c,
    float r);

std::vector<HyperbolicAttentionRecord64> pack_hyperbolic_attention_records(
    const std::vector<HyperbolicEventVector>& events,
    const std::unordered_map<std::string, float>& wcr_w_map,
    const std::unordered_map<std::string, float>& wcr_c_map,
    const std::unordered_map<std::string, float>& wcr_r_map);

std::vector<HyperbolicAttentionRecord64> pack_hyperbolic_attention_records(
    const std::vector<HyperbolicEventVector>& events);

// Parse GML topology files and append per-node vector entries to `container`.
// Overlay ID: @node:gml:{lowercase_file_stem}:{gml_node_id}
// x/y/z   = raw GML node coordinates (not normalised to unit sphere)
// norm     = sqrt(x²+y²+z²) = fold_weight per fold_graph.xml geometry contract
// Nodes with zero-magnitude coordinates are skipped.
void load_gml_topology_vectors(
    const std::vector<std::string>& gml_paths,
    micronaut::Mx2LmVectorProjectionContainer& container);

} // namespace khanary::world
