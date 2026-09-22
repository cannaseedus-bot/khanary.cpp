#include "request_router.h"
#include <sstream>
#include <cctype>

namespace khanary::io {

RequestRouter::RequestRouter(
    micronaut::MicronautRegistry& mu_reg,
    memory::MicronautScorer& scorer)
    : mu_reg_(mu_reg), scorer_(scorer)
{}

void RequestRouter::register_handler(const std::string& id, ResponseFn fn) {
    handlers_[id] = std::move(fn);
}

void RequestRouter::set_default_handler(ResponseFn fn) {
    default_handler_ = std::move(fn);
}

std::vector<std::string> RequestRouter::tokenize_query(const std::string& query) const {
    std::vector<std::string> tokens;
    std::istringstream ss(query);
    std::string tok;
    while (ss >> tok) {
        std::string lc;
        for (char c : tok) lc += (char)tolower((unsigned char)c);
        tokens.push_back(lc);
    }
    return tokens;
}

RouteResult RequestRouter::route(const RouteRequest& req) {
    RouteResult result;
    result.dispatched_phase = req.phase;

    auto tokens = req.tokens.empty() ? tokenize_query(req.query) : req.tokens;

    // Confidence-based scorer selects primary track
    auto ids = mu_reg_.ids();
    result.micronaut_id = scorer_.best(ids, 0.01f);

    // LBS soft dispatch: always populate blend (bone weights for this query)
    result.blend = mu_reg_.dispatch_blend(req.phase, tokens);

    // If scorer gave nothing, take highest-weight gravity bone
    if (result.micronaut_id.empty() && !result.blend.empty())
        result.micronaut_id = result.blend[0].first;

    if (!result.micronaut_id.empty())
        result.score = scorer_.score(result.micronaut_id).total;

    // Handler chain: walk blend in weight order, first registered handler wins
    util::JsonPtr response;
    for (const auto& [track_id, weight] : result.blend) {
        auto it = handlers_.find(track_id);
        if (it != handlers_.end()) {
            response = it->second(track_id, req);
            if (response) break;
        }
    }
    // Fallback: primary track handler (covers scorer-selected track not in blend)
    if (!response && !result.micronaut_id.empty()) {
        auto it = handlers_.find(result.micronaut_id);
        if (it != handlers_.end())
            response = it->second(result.micronaut_id, req);
    }
    if (!response && default_handler_)
        response = default_handler_(result.micronaut_id, req);
    if (!response)
        response = util::json_string("no handler registered");

    result.response = std::move(response);
    return result;
}

} // namespace khanary::io
