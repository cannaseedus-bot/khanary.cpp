#pragma once
#include "jrom.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace khanary::io {

// JROMTrace: the input to project_v() — a slice of the JROM journal for one request.
// Obtain via jrom.read_all() or jrom.replay() on the request's trace window.
using JROMTrace = std::vector<JROMEvent>;

struct ProjectionRequest {
    std::string abi_version = "⟁PROJECTION_ABI_v1";
    std::string target      = "chat";    // "chat" | "code" | "patch" | "kuhul"
    std::string thinking    = "medium";  // "low" | "medium" | "high"
    std::string anchor;  // optional: "⟁VERIFY" etc. — listed first in projection text
};

struct ProjectionResult {
    std::string abi_version;
    std::string target;
    std::string thinking_level;
    std::string anchor;
    std::string text;             // serialized authorized view — this is what DDS reads
    std::string projection_hash;  // sha256_hex(text) — identical traces → identical hashes
    std::string semantic_attention_contract; // Q=query vector, K=key vector meta, V=Book payload
    std::string semantic_attention_formula;  // S = W*C*cos(Q,K)
    bool        anchor_satisfied; // false when anchor dim had no trace material to emphasize
    bool        ok;               // false on ABI version mismatch or bad request fields
    std::string error;
};

// ProjectionEngine: the narrow waist of Khanary.
// Loads a ⟁PROJECTION_ABI_vN Book file at construction and exposes project_v().
//
// project_v() is select-only — it implements the Book without reasoning:
//   select authorized events → filter by thinking/target policy → serialize deterministically
//
// Determinism guarantee: the same sequence of authorized event types and payloads,
// at any absolute seq numbers, produces byte-identical text and projection_hash.
// This follows from (a) within-dim sort by seq, (b) verbatim payload_json
// (never re-parsed, so unordered_map key ordering can't introduce variance).
class ProjectionEngine {
public:
    explicit ProjectionEngine(const std::string& abi_book_path);

    ProjectionResult project_v(
        const JROMTrace&         trace,
        const ProjectionRequest& request) const;

    bool        has_abi(const std::string& version) const;
    std::string loaded_version()                    const { return book_.version; }
    const std::unordered_map<std::string, std::string>& event_to_dim() const { return book_.event_to_dim; }

private:
    struct AbiBook {
        std::string                                                    version;
        std::vector<std::string>                                       canonical_dims;
        std::unordered_map<std::string, std::string>                   event_to_dim;
        std::unordered_map<std::string, std::unordered_set<std::string>> thinking_dims;
        std::unordered_map<std::string, std::unordered_set<std::string>> target_dims;
    };

    AbiBook book_;

    static AbiBook      load_book(const std::string& path);
    static std::string  serialize_view(
        const std::vector<std::string>&                                          dim_order,
        const std::unordered_map<std::string, std::vector<const JROMEvent*>>&   dim_events);
};

} // namespace khanary::io
