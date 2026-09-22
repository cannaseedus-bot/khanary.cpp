#pragma once
#include "perception.h"
#include "confidence.h"
#include "semantic_graph.h"
#include <string>
#include <unordered_map>

namespace khanary::memory {

// S_µ(Q) = W_µ · C_µ · R_µ(Q)
// W_µ: semantic graph weight  (delta_z from nearest node)
// C_µ: confidence score       (from ConfidenceStore)
// R_µ(Q): relevance from F_Q  (cosine similarity between field and micronaut vector)
// ΔZ = ΔG_semantic ⊕ ΔC_earned
struct ScoreComponents {
    float w_mu; // graph weight
    float c_mu; // confidence
    float r_mu; // relevance
    float total;
    float delta_z;
};

class MicronautScorer {
public:
    MicronautScorer(PerceptionStore& perception,
                    ConfidenceStore& confidence,
                    SemanticGraph& graph);

    // Compute S_µ(Q) for a named micronaut
    ScoreComponents score(const std::string& micronaut_id) const;

    // Score all known micronauts (pass id list)
    std::unordered_map<std::string, ScoreComponents> score_all(
        const std::vector<std::string>& ids) const;

    // Best scoring micronaut
    std::string best(const std::vector<std::string>& ids,
                     float threshold = 0.01f) const;

    // ΔZ: decomposed change after a cycle
    float compute_delta_z(const std::string& from_node,
                          const std::string& to_node) const;

private:
    PerceptionStore& perception_;
    ConfidenceStore& confidence_;
    SemanticGraph&   graph_;

    float relevance(const std::string& id) const;
};

} // namespace khanary::memory
