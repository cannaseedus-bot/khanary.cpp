#include "semantic_graph.h"
#include <algorithm>
#include <stdexcept>

namespace khanary::memory {

bool SemanticGraph::check_gate(const std::string& token) {
    return token == EVOLVE_GATE;
}

bool SemanticGraph::add_node(const std::string& id, const std::string& token) {
    if (!check_gate(token)) return false;
    std::lock_guard lk(mtx_);
    if (!adj_.count(id)) adj_[id] = {};
    return true;
}

bool SemanticGraph::add_arc(const std::string& from, const std::string& to,
                             float weight, const std::string& token)
{
    if (!check_gate(token)) return false;
    std::lock_guard lk(mtx_);
    // Ensure both nodes exist
    if (!adj_.count(from)) adj_[from] = {};
    if (!adj_.count(to))   adj_[to]   = {};
    // Check for duplicate
    auto& arcs = adj_[from];
    for (auto& a : arcs) {
        if (a.to == to) { a.weight = weight; return true; } // update weight
    }
    arcs.push_back({to, weight});
    return true;
}

bool SemanticGraph::has_node(const std::string& id) const {
    std::lock_guard lk(mtx_);
    return adj_.count(id) > 0;
}

bool SemanticGraph::has_arc(const std::string& from, const std::string& to) const {
    std::lock_guard lk(mtx_);
    auto it = adj_.find(from);
    if (it == adj_.end()) return false;
    for (const auto& a : it->second) if (a.to == to) return true;
    return false;
}

std::vector<Arc> SemanticGraph::book(const std::string& id) const {
    std::lock_guard lk(mtx_);
    auto it = adj_.find(id);
    if (it == adj_.end()) return {};
    return it->second;
}

std::vector<std::string> SemanticGraph::neighbors(const std::string& id) const {
    auto arcs = book(id);
    std::vector<std::string> out;
    for (const auto& a : arcs) out.push_back(a.to);
    return out;
}

int SemanticGraph::distance(const std::string& a, const std::string& b) const {
    if (a == b) return 0;
    std::lock_guard lk(mtx_);
    // BFS
    std::unordered_map<std::string, int> visited;
    std::queue<std::string> q;
    q.push(a);
    visited[a] = 0;
    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        auto it = adj_.find(cur);
        if (it == adj_.end()) continue;
        for (const auto& arc : it->second) {
            if (!visited.count(arc.to)) {
                visited[arc.to] = visited[cur] + 1;
                if (arc.to == b) return visited[arc.to];
                q.push(arc.to);
            }
        }
    }
    return -1;
}

float SemanticGraph::delta_z(const std::string& from, const std::string& to) const {
    int d = distance(from, to);
    if (d < 0) return 0.0f;
    return 1.0f / (1.0f + (float)d);
}

size_t SemanticGraph::node_count() const {
    std::lock_guard lk(mtx_);
    return adj_.size();
}

std::vector<std::string> SemanticGraph::nodes() const {
    std::lock_guard lk(mtx_);
    std::vector<std::string> out;
    for (const auto& [k, _] : adj_) out.push_back(k);
    return out;
}

} // namespace khanary::memory
