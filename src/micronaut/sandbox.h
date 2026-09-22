#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace khanary::micronaut {

// Declared sandbox for one track.
// Loaded from the "sandbox" block in .semantic-tracks.v1.json.
// Enforced at fold-op boundaries: gramRead/Write namespacing, effect checks, port checks.
struct TrackSandbox {
    std::string              gram_ns;          // gram key prefix (e.g. "ELIZA" → "ELIZA::key")
    std::vector<std::string> manifest_access;  // which manifests this track may resolve
    std::vector<std::string> effects;          // allowed effect strings (see below)
    std::vector<uint16_t>    ports;            // allowed outbound TCP ports (empty = no network)
    bool                     coordinator;      // true → may read across all gram namespaces

    // Effect vocabulary:
    //   gramRead             — read from own gram_ns
    //   gramWrite            — write to own gram_ns
    //   semantic_graph:write — mutate G_semantic (requires ⟁EVOLVE⟁ gate)
    //   micronaut:create     — create new micronauts (Factory)
    //   micronaut:delete     — delete micronauts (Factory, confidence-gated)
    //   route:delegate       — invoke sub-micronauts (Coordinator only)
    //   manifest:write       — write to manifest files (KNOWLED-1, EVOLVE-gated)

    bool allows_effect(const std::string& eff) const {
        for (const auto& e : effects) if (e == eff) return true;
        return false;
    }

    bool allows_port(uint16_t port) const {
        for (uint16_t p : ports) if (p == port) return true;
        return false;
    }

    bool allows_manifest(const std::string& name) const {
        for (const auto& m : manifest_access) if (m == name) return true;
        return false;
    }

    // Namespace a gram key for this track (coordinator sees keys as-is)
    std::string ns_key(const std::string& key) const {
        if (coordinator || gram_ns.empty()) return key;
        if (key.find("::") != std::string::npos) return key;  // already namespaced
        return gram_ns + "::" + key;
    }
};

// Default (open) sandbox — used when no "sandbox" block is present in the track file.
// Allows everything, no namespacing. Exists only for backwards compatibility during
// the transition period; all tracks should declare an explicit sandbox.
inline TrackSandbox open_sandbox() {
    return TrackSandbox{
        .gram_ns = "",
        .manifest_access = {"words", "micronaut", "kuhul", "css"},
        .effects = {"gramRead", "gramWrite", "semantic_graph:write",
                    "micronaut:create", "micronaut:delete", "route:delegate", "manifest:write"},
        .ports = {9000, 8764, 17480, 8787},
        .coordinator = true,
    };
}

} // namespace khanary::micronaut
