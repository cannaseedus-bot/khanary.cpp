#include "semantic_graph.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace khanary::memory {

namespace {

static bool valid_node_id(const std::string& id) {
    return !id.empty();
}

static float normalize_arc_weight(float weight) {
    if (!std::isfinite(weight)) return 0.0f;
    return std::clamp(weight, 0.0f, 1.0f);
}

static uint64_t fnv1a64(std::string_view text) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : text) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace

bool SemanticGraph::check_gate(const std::string& token) {
    return token == EVOLVE_GATE;
}

bool SemanticGraph::add_node(const std::string& id, const std::string& token) {
    if (!check_gate(token) || !valid_node_id(id)) return false;
    std::lock_guard lk(mtx_);
    if (!adj_.count(id)) adj_[id] = {};
    return true;
}

bool SemanticGraph::add_arc(const std::string& from, const std::string& to,
                             float weight, const std::string& token)
{
    if (!check_gate(token) || !valid_node_id(from) || !valid_node_id(to)) return false;
    const float w = normalize_arc_weight(weight);

    std::lock_guard lk(mtx_);
    // Ensure both nodes exist
    if (!adj_.count(from)) adj_[from] = {};
    if (!adj_.count(to))   adj_[to]   = {};
    // Check for duplicate
    auto& arcs = adj_[from];
    for (auto& a : arcs) {
        if (a.to == to) { a.weight = w; return true; } // update weight
    }
    arcs.push_back({to, w});
    return true;
}

bool SemanticGraph::has_node(const std::string& id) const {
    if (!valid_node_id(id)) return false;
    std::lock_guard lk(mtx_);
    return adj_.count(id) > 0;
}

bool SemanticGraph::has_arc(const std::string& from, const std::string& to) const {
    if (!valid_node_id(from) || !valid_node_id(to)) return false;
    std::lock_guard lk(mtx_);
    auto it = adj_.find(from);
    if (it == adj_.end()) return false;
    for (const auto& a : it->second) if (a.to == to) return true;
    return false;
}

std::array<float, 3> SemanticGraph::derive_seed_vector(const std::string& seed) {
    const uint64_t h = fnv1a64(seed);
    const float ax = static_cast<float>(h & 0xffffull) * 0.0011f;
    const float ay = static_cast<float>((h >> 16) & 0xffffull) * 0.0013f;
    const float az = static_cast<float>((h >> 32) & 0xffffull) * 0.0017f;
    float x = std::sin(ax);
    float y = std::cos(ay);
    float z = std::sin(az);
    const float n = std::sqrt(x * x + y * y + z * z);
    if (n > 1e-6f) {
        x /= n;
        y /= n;
        z /= n;
    }
    return {x, y, z};
}

float SemanticGraph::cosine01(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    const float an = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const float bn = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    if (!(an > 1e-6f) || !(bn > 1e-6f)) return 0.5f;
    const float dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (an * bn);
    return std::clamp(0.5f * (dot + 1.0f), 0.0f, 1.0f);
}

bool SemanticGraph::set_node_vector(const std::string& id, float x, float y, float z, const std::string& token) {
    if (!check_gate(token) || !valid_node_id(id)) return false;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    const float n = std::sqrt(x * x + y * y + z * z);
    if (!(n > 1e-6f)) return false;

    std::lock_guard lk(mtx_);
    if (!adj_.count(id)) adj_[id] = {};
    vectors_[id] = {x / n, y / n, z / n};
    return true;
}

std::optional<std::array<float, 3>> SemanticGraph::node_vector(const std::string& id) const {
    if (!valid_node_id(id)) return std::nullopt;
    std::lock_guard lk(mtx_);
    auto it = vectors_.find(id);
    if (it == vectors_.end()) return std::nullopt;
    return it->second;
}

float SemanticGraph::vector_relevance(const std::string& query_seed, const std::string& node_id) const {
    if (!valid_node_id(node_id)) return 0.0f;
    const auto qv = derive_seed_vector(query_seed.empty() ? std::string{"query"} : query_seed);

    std::array<float, 3> kv{};
    {
        std::lock_guard lk(mtx_);
        if (!adj_.count(node_id)) return 0.0f;
        auto it = vectors_.find(node_id);
        kv = (it != vectors_.end()) ? it->second : derive_seed_vector(node_id);
    }
    return cosine01(qv, kv);
}

std::vector<Arc> SemanticGraph::book(const std::string& id) const {
    if (!valid_node_id(id)) return {};
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
    if (!valid_node_id(a) || !valid_node_id(b)) return -1;

    if (a == b) {
        std::lock_guard lk(mtx_);
        return adj_.count(a) ? 0 : -1;
    }

    std::lock_guard lk(mtx_);
    if (!adj_.count(a) || !adj_.count(b)) return -1;

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
    const float score = 1.0f / (1.0f + static_cast<float>(d));
    return normalize_arc_weight(score);
}

size_t SemanticGraph::node_count() const {
    std::lock_guard lk(mtx_);
    return adj_.size();
}

std::vector<std::string> SemanticGraph::nodes() const {
    std::lock_guard lk(mtx_);
    std::vector<std::string> out;
    for (const auto& [k, _] : adj_) out.push_back(k);
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace khanary::memory
