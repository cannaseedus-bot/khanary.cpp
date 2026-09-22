#pragma once
#include "../micronaut/micronaut_registry.h"
#include "../memory/scoring.h"
#include "../kuhul/phase_engine.h"
#include "../util/json.h"
#include <string>
#include <functional>

namespace khanary::io {

struct RouteRequest {
    std::string              query;
    std::vector<std::string> tokens;
    kuhul::Phase             phase;
    util::JsonPtr            payload;
};

struct RouteResult {
    std::string   micronaut_id;  // primary bone (highest-weight track)
    kuhul::Phase  dispatched_phase;
    float         score;
    // LBS bone weights: top-k tracks with normalized weights (Σw_k = 1.0).
    // blend[0] is the primary bone; blend[1..] are secondary influences.
    std::vector<std::pair<std::string, float>> blend;
    util::JsonPtr response;
};

using ResponseFn = std::function<util::JsonPtr(
    const std::string& micronaut_id,
    const RouteRequest& req)>;

class RequestRouter {
public:
    RequestRouter(
        micronaut::MicronautRegistry& mu_reg,
        memory::MicronautScorer& scorer);

    // Route a request through gravity scoring → best micronaut → response
    RouteResult route(const RouteRequest& req);

    // Register a response handler for a specific micronaut
    void register_handler(const std::string& micronaut_id, ResponseFn fn);

    // Default handler (for unmatched or factory-bound requests)
    void set_default_handler(ResponseFn fn);

private:
    micronaut::MicronautRegistry& mu_reg_;
    memory::MicronautScorer&      scorer_;
    std::unordered_map<std::string, ResponseFn> handlers_;
    ResponseFn default_handler_;

    std::vector<std::string> tokenize_query(const std::string& query) const;
};

} // namespace khanary::io
