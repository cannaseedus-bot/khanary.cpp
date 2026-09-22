#pragma once
#include "sandbox.h"
#include "../util/json.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>

namespace khanary::micronaut {

// A sandbox violation: track attempted an operation outside its declared sandbox.
struct SandboxViolation {
    std::string track_id;
    std::string kind;         // "effect", "port", "manifest", "gram_ns"
    std::string detail;       // what was attempted
    uint64_t    seq;          // monotone index
};

// SandboxInspector: runtime sandbox enforcement + JSON object exposure.
//
// Enforcement:
//   Call check_effect(), check_port(), check_manifest() at fold-op dispatch time.
//   Each returns true if allowed, false + records violation if denied.
//
// Inspection (JSON REST surface):
//   sandbox_json(track_id) → JSON object for one track
//   all_json()             → JSON object keyed by track_id
//   violations_json(n)     → last N violations as JSON array
class SandboxInspector {
public:
    // Register a track's declared sandbox
    void register_track(const std::string& id, TrackSandbox sb);

    // Enforcement checks — return true if allowed
    bool check_effect  (const std::string& track_id, const std::string& eff);
    bool check_port    (const std::string& track_id, uint16_t port);
    bool check_manifest(const std::string& track_id, const std::string& manifest_name);

    // Namespace a gram key through the track's sandbox
    std::string ns_key(const std::string& track_id, const std::string& raw_key) const;

    // JSON object for one track's declared sandbox
    util::JsonPtr sandbox_json(const std::string& track_id) const;

    // JSON object keyed by track_id → sandbox descriptor
    util::JsonPtr all_json() const;

    // Last N violations as JSON array (default: all)
    util::JsonPtr violations_json(size_t n = 0) const;

    bool has(const std::string& track_id) const;

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, TrackSandbox> sandboxes_;
    std::vector<SandboxViolation> violations_;
    uint64_t seq_ = 0;

    void record_violation(const std::string& track_id,
                          const std::string& kind,
                          const std::string& detail);

    static util::JsonPtr sandbox_to_json(const std::string& id, const TrackSandbox& sb);
};

} // namespace khanary::micronaut
