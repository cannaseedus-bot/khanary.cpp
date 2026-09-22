#pragma once
#include "manifest_registry.h"
#include <string>
#include <vector>
#include <unordered_set>

namespace khanary::manifest {

// SCXResolver: context-aware manifest expansion
// Follows SCX compression law: key → neighborhood (adjacency list)
// Handles: plain keys, @node:Name, raw strings, @-operator pass-through
struct ResolveContext {
    std::string instance;   // current instance key (innermost scope)
    std::string package;    // package-level scope
    std::string workflow;   // workflow-level scope
};

class SCXResolver {
public:
    explicit SCXResolver(const ManifestRegistry& reg);

    // Full chain: instance → local → package → workflow → runtime → system
    // Returns leaf strings (raw values + operator tokens)
    std::vector<std::string> resolve(const std::string& key, const ResolveContext& ctx = {}) const;

    // Check if elem is an @-operator (not to be expanded)
    static bool is_operator(const std::string& elem);

    // Check if elem is an @node: reference
    static bool is_node_ref(const std::string& elem);

    // Extract the name from @node:Name
    static std::string node_ref_name(const std::string& elem);

private:
    const ManifestRegistry& reg_;

    std::vector<std::string> expand(
        const std::string& key,
        std::unordered_set<std::string>& visited,
        int depth,
        const ResolveContext& ctx) const;
};

} // namespace khanary::manifest
