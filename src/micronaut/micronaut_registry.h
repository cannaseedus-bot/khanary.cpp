#pragma once
#include "../kuhul/phase_engine.h"
#include "gravity_field.h"
#include "sandbox.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <mutex>

namespace khanary::micronaut {

struct MicronautEntry {
    std::string                   id;
    std::string                   description;
    kuhul::Phase                  primary_phase;
    std::vector<std::string>      capabilities;
    std::vector<std::string>      sub_micronauts;  // declared delegation targets
    std::unordered_set<std::string> governs;    // token domain
    float                         confidence;   // L_t score [0, 1]
    bool                          active;
    TrackSandbox                  sandbox;      // declared capability bounds
};

using DispatchFn = std::function<std::string(const std::string& query, const MicronautEntry&)>;

class MicronautRegistry {
public:
    MicronautRegistry();

    // Register a micronaut
    void add(const MicronautEntry& entry);
    void add(MicronautEntry&& entry);

    // Remove a micronaut (only if confidence >= threshold)
    bool remove(const std::string& id, float confidence_floor = 0.7f);

    bool has(const std::string& id) const;
    const MicronautEntry* get(const std::string& id) const;
    MicronautEntry* get_mut(const std::string& id);

    // Hard dispatch: single best match (kept for legacy call sites)
    std::string dispatch(kuhul::Phase phase,
                         const std::vector<std::string>& tokens,
                         float gravity_threshold = 0.1f) const;

    // LBS soft dispatch: top-k tracks with normalized weights summing to 1.0.
    // Equivalent to 4-bone blend weights in skeletal animation.
    std::vector<std::pair<std::string, float>> dispatch_blend(
        kuhul::Phase phase,
        const std::vector<std::string>& tokens,
        float gravity_threshold = 0.1f,
        size_t top_k = 4) const;

    // List all micronauts
    std::vector<std::string> ids() const;

    GravityField& gravity() { return gravity_; }

    void set_dispatch_fn(DispatchFn fn) { dispatch_fn_ = std::move(fn); }

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, MicronautEntry> entries_;
    GravityField gravity_;
    DispatchFn   dispatch_fn_;
};

} // namespace khanary::micronaut
