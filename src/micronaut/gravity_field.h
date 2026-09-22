#pragma once
#include "../kuhul/phase_engine.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace khanary::micronaut {

// Gravity well: score = established_nodes / distance²
// Low gravity → route to Factory for new micronaut creation
class GravityField {
public:
    // Register that a micronaut owns a set of topic nodes
    void register_nodes(const std::string& micronaut_id,
                        const std::unordered_set<std::string>& nodes);

    // Score all registered micronauts for a query token set + phase
    // Returns map: micronaut_id → gravity score
    std::unordered_map<std::string, float> score(
        const std::vector<std::string>& tokens,
        kuhul::Phase phase) const;

    // Best match above threshold (returns "" if none)
    std::string best_match(
        const std::vector<std::string>& tokens,
        kuhul::Phase phase,
        float threshold = 0.1f) const;

    // Merge token sets (horizontal fold: same phase, different nodes)
    std::vector<std::string> merge_phase(
        const std::string& micronaut_a,
        const std::string& micronaut_b,
        kuhul::Phase phase) const;

private:
    struct MicronautNodes {
        std::string                   id;
        std::unordered_set<std::string> nodes;
        kuhul::Phase                  phase = kuhul::Phase::Pop;
    };
    std::unordered_map<std::string, MicronautNodes> registry_;

    // BFS distance from token to nearest registered node
    float min_distance(const std::string& token,
                       const std::unordered_set<std::string>& nodes) const;
};

} // namespace khanary::micronaut
