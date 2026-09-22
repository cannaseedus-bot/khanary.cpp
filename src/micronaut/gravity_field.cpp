#include "gravity_field.h"
#include <algorithm>
#include <cmath>

namespace khanary::micronaut {

void GravityField::register_nodes(
    const std::string& micronaut_id,
    const std::unordered_set<std::string>& nodes)
{
    registry_[micronaut_id] = {micronaut_id, nodes};
}

float GravityField::min_distance(
    const std::string& token,
    const std::unordered_set<std::string>& nodes) const
{
    // Direct membership: distance = 0
    if (nodes.count(token)) return 0.0f;
    // Prefix/substring: distance = 1
    for (const auto& n : nodes) {
        if (n.find(token) != std::string::npos || token.find(n) != std::string::npos)
            return 1.0f;
    }
    // Levenshtein-lite: first char match → distance = 2
    if (!token.empty()) {
        for (const auto& n : nodes) {
            if (!n.empty() && n[0] == token[0]) return 2.0f;
        }
    }
    return 4.0f; // far
}

std::unordered_map<std::string, float> GravityField::score(
    const std::vector<std::string>& tokens,
    kuhul::Phase phase) const
{
    std::unordered_map<std::string, float> scores;
    for (const auto& [id, entry] : registry_) {
        float g = 0.0f;
        uint32_t hits = 0;
        for (const auto& tok : tokens) {
            float d = min_distance(tok, entry.nodes);
            if (d < 4.0f) {
                g += (float)entry.nodes.size() / ((d + 1.0f) * (d + 1.0f));
                ++hits;
            }
        }
        // Phase arc weight modulates the score
        double w = kuhul::arc_weight(entry.phase, phase);
        g *= (float)std::max(0.0, w);
        scores[id] = g;
    }
    return scores;
}

std::string GravityField::best_match(
    const std::vector<std::string>& tokens,
    kuhul::Phase phase,
    float threshold) const
{
    auto scores = score(tokens, phase);
    std::string best_id;
    float best_score = threshold;
    for (const auto& [id, s] : scores) {
        if (s > best_score) { best_score = s; best_id = id; }
    }
    return best_id;
}

std::vector<std::string> GravityField::merge_phase(
    const std::string& micronaut_a,
    const std::string& micronaut_b,
    kuhul::Phase /*phase*/) const
{
    // Horizontal fold: union of node sets at same phase
    std::vector<std::string> merged;
    auto it_a = registry_.find(micronaut_a);
    auto it_b = registry_.find(micronaut_b);
    if (it_a != registry_.end())
        for (const auto& n : it_a->second.nodes) merged.push_back(n);
    if (it_b != registry_.end())
        for (const auto& n : it_b->second.nodes) merged.push_back(n);
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    return merged;
}

} // namespace khanary::micronaut
