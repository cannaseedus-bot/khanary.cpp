#pragma once
#include "../kuhul/phase_engine.h"
#include "../util/json.h"
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

// One directed (or undirected) relation edge between two micronauts.
// Relation ISA is closed and typed (semantic/causal/epistemic/dependency/context/control/runtime),
// with compatibility aliases normalized at load time.
// directed=false → horizontal fold (Δθ=0), no XCFE gate required.
struct RelationEntry {
    std::string target_id;
    std::string verb;      // canonical relation token (e.g. @is_a, @supports, @delegates_to)
    bool        directed = true;
};

// Track fold declaration entry.
// This is a declarative capability/weight surface for phase drivers, not a fixed execution script.
struct FoldSpec {
    kuhul::Phase phase = kuhul::Phase::Pop;
    std::string  op;
    std::string  key;
    std::string  scope;
    std::string  delegate;
    std::string  note;
    std::string  gate;
    float        weight = 1.0f; // static fold preference [0,1]
};

struct TrackVectorMeta {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    bool declared = false;
    std::string source = "phase_default";
};

// Scheme-bound callable surface for a micronaut endpoint.
// Example schemes: math, time, date, plan, mx2lm, eliza, tools, memory.
struct RouteContract {
    std::vector<std::string> schemes;
    std::vector<std::string> verbs;
    std::vector<std::string> aiml_patterns;
    std::vector<std::string> aiml_templates;
    std::vector<std::string> key_patterns;
    std::vector<std::string> key_templates;
    std::vector<std::string> key_grammar;
    std::vector<std::string> tokens;
};

// Map a relation token to its dominant K'uhul phase.
kuhul::Phase verb_to_phase(const std::string& verb);

struct MicronautEntry {
    std::string                   id;
    std::string                   description;
    kuhul::Phase                  primary_phase;
    std::string                   category;        // "executive"|"cognitive"|"storage"|"creative"|""
    std::vector<std::string>      capabilities;
    std::vector<std::string>      sub_micronauts;  // plain delegation targets (legacy / no verb)
    std::vector<RelationEntry>    relations;        // typed relational verb edges
    std::vector<FoldSpec>         folds;            // phase-typed fold capability declarations
    std::unordered_set<std::string> governs;        // token domain
    RouteContract               route_contract;   // scheme/verb/pattern/template/token callable surface
    util::JsonPtr                 profile_projection; // optional brain/personality/progression projection; non-authoritative
    TrackVectorMeta               vector_meta;      // semantic Key vector metadata (K in W·C·cos(Q,K))
    float                         static_weight = 1.0f; // deterministic routing prior [0,1]
    float                         health = 1.0f;        // liveness/availability signal [0,1]
    float                         confidence;       // L_t score [0, 1]
    bool                          no_eval = true;   // hard policy: tracks are declarative, not eval-enabled scripts
    bool                          active;
    TrackSandbox                  sandbox;          // declared capability bounds
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

    // Relational dispatch: return all micronauts reachable from `from_id` via `verb`.
    // Uses the declared relations table — does not score by gravity.
    std::vector<std::string> relational_dispatch(
        const std::string& from_id,
        const std::string& verb) const;

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
