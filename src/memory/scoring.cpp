#include "scoring.h"
#include <cmath>
#include <functional>
#include <algorithm>

namespace khanary::memory {

MicronautScorer::MicronautScorer(
    PerceptionStore& perception,
    ConfidenceStore& confidence,
    SemanticGraph& graph)
    : perception_(perception), confidence_(confidence), graph_(graph)
{}

float MicronautScorer::relevance(const std::string& id) const {
    const auto& field = perception_.field();
    if (field.empty()) return 0.5f;
    // Compute a pseudo field-vector for the micronaut from its id hash
    std::hash<std::string> h;
    size_t hv = h(id);
    uint32_t dim = (uint32_t)field.size();
    float dot = 0.0f, norm_f = 0.0f, norm_m = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float m = std::sin((float)((hv >> (i % 64)) & 0xFF) * 0.1f + (float)i);
        dot    += field[i] * m;
        norm_f += field[i] * field[i];
        norm_m += m * m;
    }
    float denom = std::sqrt(norm_f) * std::sqrt(norm_m);
    return denom > 1e-6f ? (dot / denom + 1.0f) * 0.5f : 0.5f;
}

ScoreComponents MicronautScorer::score(const std::string& id) const {
    ScoreComponents sc{};
    // W_µ: graph-reachability weight
    // Use average delta_z from graph nodes to this id
    auto neighbors = graph_.neighbors(id);
    if (!neighbors.empty()) {
        float sum = 0.0f;
        for (const auto& n : neighbors) sum += graph_.delta_z(id, n);
        sc.w_mu = sum / neighbors.size();
    } else {
        sc.w_mu = graph_.has_node(id) ? 0.5f : 0.1f;
    }

    sc.c_mu = confidence_.get(id);
    sc.r_mu = relevance(id);
    sc.total = sc.w_mu * sc.c_mu * sc.r_mu;

    // ΔZ: change in combined score
    sc.delta_z = sc.w_mu * sc.c_mu; // simplified: without prior
    return sc;
}

std::unordered_map<std::string, ScoreComponents> MicronautScorer::score_all(
    const std::vector<std::string>& ids) const
{
    std::unordered_map<std::string, ScoreComponents> out;
    for (const auto& id : ids) out[id] = score(id);
    return out;
}

std::string MicronautScorer::best(
    const std::vector<std::string>& ids,
    float threshold) const
{
    std::string best_id;
    float best_score = threshold;
    for (const auto& id : ids) {
        float s = score(id).total;
        if (s > best_score) { best_score = s; best_id = id; }
    }
    return best_id;
}

float MicronautScorer::compute_delta_z(const std::string& from, const std::string& to) const {
    return graph_.delta_z(from, to);
}

} // namespace khanary::memory
