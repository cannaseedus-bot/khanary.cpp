#include "vector_projection.h"
#include "gml_parser.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>

namespace khanary::world {

namespace fs = std::filesystem;

namespace {

float finite_or(float v, float fallback) {
    return std::isfinite(v) ? v : fallback;
}

float clamp01(float v) {
    return std::clamp(finite_or(v, 0.0f), 0.0f, 1.0f);
}

float lookup_or(const std::unordered_map<std::string, float>& map,
                const std::string& key,
                float fallback) {
    const auto it = map.find(key);
    return (it == map.end()) ? fallback : it->second;
}

float causal_alignment(const HyperbolicEventVector& event) {
    const std::string& cls = event.interval_class;
    if (cls.find("spacelike") != std::string::npos) return 0.0f;
    if (cls.find("lightlike") != std::string::npos) return 0.5f;
    if (cls.find("timelike") != std::string::npos) return 1.0f;

    const float interval = finite_or(event.minkowski_interval, 0.0f);
    if (interval < 0.0f) return 1.0f;
    if (interval == 0.0f) return 0.5f;
    return 0.0f;
}

} // namespace

HyperbolicAttentionRecord64 make_hyperbolic_attention_record(
    const HyperbolicEventVector& event,
    float w,
    float c,
    float r)
{
    const float x0 = finite_or(event.coords[0], 0.0f);
    const float x1 = finite_or(event.coords[1], 0.0f);
    const float x2 = finite_or(event.coords[2], 0.0f);
    const float x3 = finite_or(event.coords[3], 0.0f);
    const float space_mag = std::sqrt(std::max(0.0f, x1*x1 + x2*x2 + x3*x3));

    HyperbolicAttentionRecord64 rec{};
    rec.W = clamp01(w);
    rec.C = clamp01(c);
    rec.R = clamp01(r);
    rec.relation_weight = (space_mag > 1e-6f)
        ? clamp01(std::sqrt(std::max(0.0f, x1*x1 + x2*x2)) / space_mag)
        : 0.0f;

    rec.fold_weight = finite_or(space_mag, 0.0f);
    rec.phase_weight = (space_mag > 1e-6f)
        ? clamp01(0.5f + 0.5f * (x3 / space_mag))
        : 0.5f;
    rec.policy_ok = event.policy_ok ? 1.0f : 0.0f;
    rec.capable = event.capable ? 1.0f : 0.0f;

    rec.xcfe_ok = event.xcfe_ok ? 1.0f : 0.0f;
    rec.tensor_mass = std::max(0.0f, finite_or(x0, 0.0f));
    rec.tensor_alignment = clamp01(causal_alignment(event));
    rec.tensor_health = clamp01(event.tensor_health);

    rec.tensor_enabled = event.tensor_enabled ? 1.0f : 0.0f;
    rec.reserved0 = x1;
    rec.reserved1 = x2;
    rec.reserved2 = x3;
    return rec;
}

std::vector<HyperbolicAttentionRecord64> pack_hyperbolic_attention_records(
    const std::vector<HyperbolicEventVector>& events,
    const std::unordered_map<std::string, float>& wcr_w_map,
    const std::unordered_map<std::string, float>& wcr_c_map,
    const std::unordered_map<std::string, float>& wcr_r_map)
{
    std::vector<HyperbolicAttentionRecord64> records;
    records.reserve(events.size());
    for (const auto& event : events) {
        records.push_back(make_hyperbolic_attention_record(
            event,
            lookup_or(wcr_w_map, event.key, 1.0f),
            lookup_or(wcr_c_map, event.key, 1.0f),
            lookup_or(wcr_r_map, event.key, 0.5f)));
    }
    return records;
}

std::vector<HyperbolicAttentionRecord64> pack_hyperbolic_attention_records(
    const std::vector<HyperbolicEventVector>& events)
{
    const std::unordered_map<std::string, float> empty;
    return pack_hyperbolic_attention_records(events, empty, empty, empty);
}

void load_gml_topology_vectors(
    const std::vector<std::string>& gml_paths,
    micronaut::Mx2LmVectorProjectionContainer& container)
{
    for (const auto& path : gml_paths) {
        auto graph = parse_gml_file(path);
        if (!graph) continue;

        std::string stem = fs::path(path).stem().string();
        std::transform(stem.begin(), stem.end(), stem.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        for (const auto& node : graph->nodes) {
            const float x   = node.has_x ? static_cast<float>(node.x) : 0.0f;
            const float y   = node.has_y ? static_cast<float>(node.y) : 0.0f;
            const float z   = node.has_z ? static_cast<float>(node.z) : 0.0f;
            const float mag = std::sqrt(x*x + y*y + z*z);
            if (mag < 1e-6f) continue;

            micronaut::Mx2LmVectorProjection entry;
            entry.key          = "@node:gml:" + stem + ":" + std::to_string(node.id);
            entry.source_key   = "gml:topology_contract";
            entry.source_track = stem;
            entry.source_phase = "Wo";
            entry.basis        = "gml_coordinate_geometry";
            entry.x            = x;
            entry.y            = y;
            entry.z            = z;
            entry.norm         = mag;
            entry.semantic_authority = false;
            container.projections.push_back(std::move(entry));
        }
    }
}

} // namespace khanary::world
