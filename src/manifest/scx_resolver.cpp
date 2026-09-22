#include "scx_resolver.h"
#include "../config.h"

namespace khanary::manifest {

SCXResolver::SCXResolver(const ManifestRegistry& reg) : reg_(reg) {}

bool SCXResolver::is_operator(const std::string& s) {
    return !s.empty() && s[0] == '@';
}

bool SCXResolver::is_node_ref(const std::string& s) {
    return s.rfind("@node:", 0) == 0;
}

std::string SCXResolver::node_ref_name(const std::string& s) {
    if (is_node_ref(s)) return s.substr(6);
    return s;
}

std::vector<std::string> SCXResolver::resolve(
    const std::string& key,
    const ResolveContext& ctx) const
{
    std::unordered_set<std::string> visited;
    return expand(key, visited, 0, ctx);
}

std::vector<std::string> SCXResolver::expand(
    const std::string& key,
    std::unordered_set<std::string>& visited,
    int depth,
    const ResolveContext& ctx) const
{
    if (depth >= (int)SCX_MAX_DEPTH)
        return {"[SCX_DEPTH_LIMIT]"};
    if (visited.count(key))
        return {"[SCX_CYCLE:" + key + "]"};

    // @node:Name — resolve the referenced name
    if (is_node_ref(key)) {
        std::string name = node_ref_name(key);
        return expand(name, visited, depth + 1, ctx);
    }

    // @-operator (not @node:) — emit as operative token
    if (is_operator(key)) {
        return {key};
    }

    // Not in registry → raw string
    if (!reg_.has(key)) return {key};

    const auto* arr = reg_.raw(key);
    if (!arr) return {key};

    visited.insert(key);
    std::vector<std::string> result;
    for (const auto& elem : *arr) {
        if (is_node_ref(elem)) {
            auto sub = expand(node_ref_name(elem), visited, depth + 1, ctx);
            result.insert(result.end(), sub.begin(), sub.end());
        } else if (is_operator(elem)) {
            result.push_back(elem); // pass operative tokens through
        } else if (reg_.has(elem)) {
            auto sub = expand(elem, visited, depth + 1, ctx);
            result.insert(result.end(), sub.begin(), sub.end());
        } else {
            result.push_back(elem); // raw string
        }
    }
    visited.erase(key);
    return result;
}

} // namespace khanary::manifest
