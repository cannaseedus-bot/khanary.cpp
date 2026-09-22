#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <optional>
#include <queue>
#include <functional>

namespace khanary::memory {

// EVOLVE gate: only this token authorizes G_semantic mutation
constexpr const char* EVOLVE_GATE = "⟁EVOLVE⟁";

struct Arc {
    std::string to;
    float       weight;
};

// G_semantic: semantic graph topology
// AddNode/AddArc only via EVOLVE gate
// Book(id) ≡ adjacency list ≡ manifest neighborhood
class SemanticGraph {
public:
    SemanticGraph() = default;

    // Gate-protected mutation
    bool add_node(const std::string& id, const std::string& evolve_token = EVOLVE_GATE);
    bool add_arc(const std::string& from, const std::string& to,
                 float weight = 1.0f, const std::string& evolve_token = EVOLVE_GATE);

    // Read-only operations (no gate)
    bool has_node(const std::string& id) const;
    bool has_arc(const std::string& from, const std::string& to) const;

    // Book(v) = adjacency list (neighborhood)
    std::vector<Arc> book(const std::string& id) const;

    // Neighbors (just ids)
    std::vector<std::string> neighbors(const std::string& id) const;

    // BFS distance (-1 if unreachable)
    int distance(const std::string& a, const std::string& b) const;

    // ΔZ decomposition: semantic topology change
    float delta_z(const std::string& from, const std::string& to) const;

    size_t node_count() const;
    std::vector<std::string> nodes() const;

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::vector<Arc>> adj_;

    static bool check_gate(const std::string& token);
};

} // namespace khanary::memory
