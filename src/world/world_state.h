#pragma once
#include "atomic_dom.h"
#include "../micronaut/micronaut_registry.h"
#include "../micronaut/sandbox_inspector.h"
#include "../util/json.h"
#include <array>
#include <string>
#include <vector>

namespace khanary::world {

// Per-track 3D node in the K-CUBE world.
// Position is derived from the track's declared primary phase (angle on C₆ circle, r=2).
// Tracks sharing a phase are offset on the z-axis (0.5 per track).
struct TrackWorldNode {
    std::string    track_id;
    float          x, y, z;         // world-space position
    kuhul::Phase   primary_phase;
    AtomicDOMBlock active_block;     // current/declared primary block
    float          confidence;       // L_t score from registry entry
    float          static_weight;    // deterministic routing prior
    float          health;           // liveness/availability
    uint32_t       gram_count;       // grams in namespace (0 until GramStore integration)
    bool           sandbox_violation;// true if any violation recorded for this track
    std::string    vector_source;    // phase_default | track.vector | ...

    // Per-block glow intensity [0..1].
    // Populated at rebuild() time: 1.0 for the primary block, 0.0 for others.
    // [Evolve] defaults to 0.0 — must be explicitly activated when ⟁EVOLVE⟁ fires.
    std::array<float, static_cast<size_t>(AtomicDOMBlock::COUNT)> block_intensities;
};

// WorldState: the K-CUBE product graph G_runtime = G_semantic □ G_phase
// projected as a live 3D scene.
//
// Rebuild from MicronautRegistry + SandboxInspector each time world_state is queried.
// Extension points:
//   - gram_count: wire GramStore.count_ns(gram_ns)
//   - sandbox_violation: check violations_json() per track
//   - active_block: update from KuhulVM current phase per track
class WorldState {
public:
    WorldState(micronaut::MicronautRegistry& reg,
               micronaut::SandboxInspector& inspector);

    // Rebuild snapshot from registry + sandbox inspector.
    void rebuild();

    // JSON for the world_state MCP tool.
    // Returns:
    //   { "faces": [...], "tracks": [...] }
    util::JsonPtr to_json() const;

    const std::vector<TrackWorldNode>& nodes() const { return nodes_; }

private:
    micronaut::MicronautRegistry& reg_;
    micronaut::SandboxInspector&  inspector_;
    std::vector<TrackWorldNode>   nodes_;

    // Position on the C₆ circle at radius r, phase p
    static float pos_x(kuhul::Phase p, float r = 2.0f);
    static float pos_y(kuhul::Phase p, float r = 2.0f);

    // Build the faces JSON array (static — derived from KCUBE_FACES table)
    static util::JsonPtr faces_json();
};

} // namespace khanary::world
