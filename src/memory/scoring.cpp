#include "scoring.h"
#include <cmath>
#include <functional>
#include <algorithm>

namespace khanary::memory {

namespace {

static float clamp01(float v, float fallback = 0.0f) {
    if (!std::isfinite(v)) return fallback;
    return std::clamp(v, 0.0f, 1.0f);
}

} // namespace

MicronautScorer::MicronautScorer(
    PerceptionStore& perception,
    ConfidenceStore& confidence,
    SemanticGraph& graph)
    : perception_(perception), confidence_(confidence), graph_(graph)
{}

float MicronautScorer::relevance(const std::string& id) const {
    if (id.empty()) return 0.0f;

    const auto& field = perception_.field();
    if (field.empty()) return 0.5f;

    // Compute a pseudo field-vector for the micronaut from its id hash
    std::hash<std::string> h;
    const size_t hv = h(id);
    const uint32_t dim = static_cast<uint32_t>(field.size());
    constexpr uint32_t kHashBits = static_cast<uint32_t>(sizeof(size_t) * 8u);
    float dot = 0.0f, norm_f = 0.0f, norm_m = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float m = std::sin(
            static_cast<float>((hv >> (i % kHashBits)) & 0xFFu) * 0.1f + static_cast<float>(i));
        dot    += field[i] * m;
        norm_f += field[i] * field[i];
        norm_m += m * m;
    }
    const float denom = std::sqrt(norm_f) * std::sqrt(norm_m);
    if (!(denom > 1e-6f)) return 0.5f;
    return clamp01((dot / denom + 1.0f) * 0.5f, 0.5f);
}

ScoreComponents MicronautScorer::score(const std::string& id) const {
    ScoreComponents sc{};
    if (id.empty()) return sc;

    // W_µ: graph-reachability weight
    // Use weighted neighborhood blend: arc weight + topological proximity.
    const auto neighborhood = graph_.book(id);
    if (!neighborhood.empty()) {
        float sum = 0.0f;
        size_t count = 0;
        for (const auto& arc : neighborhood) {
            if (arc.to.empty()) continue;
            const float arc_w = clamp01(arc.weight, 0.0f);
            const float dz = clamp01(graph_.delta_z(id, arc.to), 0.0f);
            sum += 0.65f * arc_w + 0.35f * dz;
            ++count;
        }
        sc.w_mu = count ? (sum / static_cast<float>(count)) : 0.5f;
    } else {
        sc.w_mu = graph_.has_node(id) ? 0.5f : 0.1f;
    }
    sc.w_mu = clamp01(sc.w_mu, 0.1f);

    sc.c_mu = clamp01(confidence_.get(id), 0.5f);
    sc.r_mu = clamp01(relevance(id), 0.5f);
    sc.total = clamp01(sc.w_mu * sc.c_mu * sc.r_mu, 0.0f);

    // ΔZ: change in combined score
    sc.delta_z = clamp01(sc.w_mu * sc.c_mu, 0.0f); // simplified: without prior
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
    const float min_score = std::isfinite(threshold) ? threshold : 0.01f;
    float best_score = min_score;
    constexpr float kEps = 1e-6f;
    for (const auto& id : ids) {
        if (id.empty()) continue;
        const float s = clamp01(score(id).total, 0.0f);
        if (s > best_score + kEps ||
            (std::fabs(s - best_score) <= kEps && !best_id.empty() && id < best_id)) {
            best_score = s;
            best_id = id;
        }
    }
    return best_id;
}

float MicronautScorer::compute_delta_z(const std::string& from, const std::string& to) const {
    return clamp01(graph_.delta_z(from, to), 0.0f);
}

} // namespace khanary::memory
