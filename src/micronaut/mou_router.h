#pragma once
// MouRouter — Mixture-of-Micronauts (Moµ) two-tier routing.
//
// Tier 1 (tensor):   MoERouter  — dot(W_µ, F_Q) + C_µ bias → softmax → top-k
// Tier 2 (semantic): phase gate — restrict output to micronauts at requested phase
//
// Expert weights:   TrackVectorMeta {x,y,z} per micronaut (hidden_dim = 3)
// Routing bias:     ConfidenceStore C_µ per micronaut
// Query vector:     3D projection of tokens (sin-hash, matches PerceptionStore)
//
// routing_variance() is an entropy proxy:
//   high → one µ dominates (crystallized, low H_norm)
//   low  → flat distribution (high H_norm → consider delegating to ELIZA)
//
// Call sync_index() after bulk track load.
// Call sync_bias() after any ConfidenceStore update.
//
#include "micronaut_registry.h"
#include "../inference/moe_router.h"
#include "../memory/confidence.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <array>
#include <cstdint>

namespace khanary::micronaut {

// One routing allocation — string identity + tensor index + blend weight.
struct MouAlloc {
    std::string  id;      // micronaut string ID
    uint32_t     index;   // expert index in [0, n_micronauts)
    float        weight;  // renormalized blend weight (all weights in result sum to 1.0)
};

class MouRouter {
public:
    explicit MouRouter(MicronautRegistry&      reg,
                       memory::ConfidenceStore& confidence);

    // Rebuild expert index + weight matrix from current registry state.
    // Must be called after bulk track load (TrackLoader::load_dir) or
    // after any MicronautRegistry::add / remove call.
    void sync_index();

    // Reload routing bias from ConfidenceStore snapshot.
    // Call after any confidence update (Ch'en→Xul gate).
    void sync_bias();

    // Two-tier route:
    //   1. Derive 3D F_Q from tokens → MoERouter dot product + softmax → top-k * 3
    //   2. Phase filter → renormalize → return top_k
    // Fallback to unfiltered if phase filter empties the result.
    // Returns at most top_k allocations with weights summing to 1.0.
    std::vector<MouAlloc> route(
        kuhul::Phase                    phase,
        const std::vector<std::string>& tokens,
        uint32_t                        top_k = 4) const;

    // Sample variance of top-expert weight from recent route() calls.
    // High variance → crystallized (Jyggalag); low → flat (Sheogorath → ELIZA delegate).
    float routing_variance(uint32_t n_recent = 16) const;

    // Record a dispatch outcome for load-balancing bias decay.
    void record_dispatch(const std::string& id);

    uint32_t n_micronauts() const;

private:
    MicronautRegistry&           reg_;
    memory::ConfidenceStore&     confidence_;
    inference::MoERouter         moe_;

    mutable std::mutex                       idx_mtx_;
    std::vector<std::string>                 index_to_id_;
    std::unordered_map<std::string,uint32_t> id_to_index_;

    // Derive a 3D unit query vector from token list.
    // Uses sin-hash projection matching PerceptionStore::embed() semantics.
    static std::array<float,3> query_vec(const std::vector<std::string>& tokens);
    static float                norm3(const std::array<float,3>& v);
    static std::array<float,3>  normalize3(const std::array<float,3>& v);
};

} // namespace khanary::micronaut
