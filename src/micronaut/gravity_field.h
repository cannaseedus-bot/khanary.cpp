#pragma once
#include "../kuhul/phase_engine.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace khanary::micronaut {

struct MicronautAttraction {
    std::string micronaut_id;
    float semantic_mass = 0.0f;       // G(M_mu): govern/token basin mass
    float geometric_alignment = 0.0f; // cos(Q,K_mu) mapped to [0,1]
    float phase_relevance = 0.0f;     // R_mu phase/provider relevance
    float static_weight = 1.0f;       // W_mu
    float confidence = 1.0f;          // C_mu
    float health = 1.0f;              // runtime reliability/liveness
    float wcr_prior = 0.0f;           // W_mu*C_mu*R_mu, health-damped
    float entropy_diffusion = 0.0f;   // exploratory uncertainty pressure
    float phi = 0.0f;                 // semantic attraction force
    float capture_score = 0.0f;       // phi - entropy_diffusion when legal
    bool policy_ok = true;
    bool capable = true;
    bool xcfe_ok = true;
    bool legal = true;
    bool captured = false;
};

// Gravity well: score = established_nodes / distance²
// Low gravity → route to Factory for new micronaut creation
class GravityField {
public:
    // Register that a micronaut owns a set of topic nodes
    void register_nodes(const std::string& micronaut_id,
                        const std::unordered_set<std::string>& nodes,
                        kuhul::Phase phase = kuhul::Phase::Pop,
                        float vector_x = 0.0f,
                        float vector_y = 0.0f,
                        float vector_z = 0.0f,
                        bool vector_declared = false,
                        float static_weight = 1.0f,
                        float confidence = 1.0f,
                        float health = 1.0f,
                        float entropy_diffusion = 0.0f,
                        bool policy_ok = true,
                        bool capable = true,
                        bool xcfe_ok = true);

    // Remove a micronaut's gravity registration.
    bool unregister(const std::string& micronaut_id);

    // Score all registered micronauts for a query token set + phase
    // Returns map: micronaut_id → gravity score
    std::unordered_map<std::string, float> score(
        const std::vector<std::string>& tokens,
        kuhul::Phase phase) const;

    // Full semantic-attraction capture evaluation:
    // Phi_mu(Q)=Mass_mu*cos(Q,K_mu)*(W_mu*C_mu*R_mu).
    // Capture is candidate admission only; it never verifies truth or commits B.
    std::vector<MicronautAttraction> capture(
        const std::vector<std::string>& tokens,
        kuhul::Phase phase,
        float threshold = 0.1f) const;

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
        float                         vector_x = 0.0f;
        float                         vector_y = 0.0f;
        float                         vector_z = 0.0f;
        bool                          vector_declared = false;
        float                         static_weight = 1.0f;
        float                         confidence = 1.0f;
        float                         health = 1.0f;
        float                         entropy_diffusion = 0.0f;
        bool                          policy_ok = true;
        bool                          capable = true;
        bool                          xcfe_ok = true;
    };
    mutable std::mutex mtx_;
    std::unordered_map<std::string, MicronautNodes> registry_;

    // BFS distance from token to nearest registered node
    float min_distance(const std::string& token,
                       const std::unordered_set<std::string>& nodes) const;
};

} // namespace khanary::micronaut
