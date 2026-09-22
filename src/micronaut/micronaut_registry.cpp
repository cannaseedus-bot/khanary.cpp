#include "micronaut_registry.h"
#include <algorithm>

namespace khanary::micronaut {

MicronautRegistry::MicronautRegistry() {}

void MicronautRegistry::add(const MicronautEntry& entry) {
    std::lock_guard lk(mtx_);
    entries_[entry.id] = entry;
    gravity_.register_nodes(entry.id, entry.governs);
}

void MicronautRegistry::add(MicronautEntry&& entry) {
    std::lock_guard lk(mtx_);
    std::string id = entry.id;
    auto governs = entry.governs;
    entries_[id] = std::move(entry);
    gravity_.register_nodes(id, governs);
}

bool MicronautRegistry::remove(const std::string& id, float confidence_floor) {
    std::lock_guard lk(mtx_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    if (it->second.confidence < confidence_floor) return false;
    entries_.erase(it);
    return true;
}

bool MicronautRegistry::has(const std::string& id) const {
    std::lock_guard lk(mtx_);
    return entries_.count(id) > 0;
}

const MicronautEntry* MicronautRegistry::get(const std::string& id) const {
    std::lock_guard lk(mtx_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

MicronautEntry* MicronautRegistry::get_mut(const std::string& id) {
    std::lock_guard lk(mtx_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

std::string MicronautRegistry::dispatch(
    kuhul::Phase phase,
    const std::vector<std::string>& tokens,
    float gravity_threshold) const
{
    // Score by gravity
    auto scores = gravity_.score(tokens, phase);

    // Filter to active micronauts, sort by score desc
    std::vector<std::pair<float, std::string>> ranked;
    {
        std::lock_guard lk(mtx_);
        for (const auto& [id, s] : scores) {
            auto it = entries_.find(id);
            if (it != entries_.end() && it->second.active)
                ranked.push_back({s, id});
        }
    }
    if (ranked.empty()) return "";
    std::sort(ranked.rbegin(), ranked.rend());
    if (ranked[0].first < gravity_threshold) return "";
    return ranked[0].second;
}

std::vector<std::pair<std::string, float>> MicronautRegistry::dispatch_blend(
    kuhul::Phase phase,
    const std::vector<std::string>& tokens,
    float gravity_threshold,
    size_t top_k) const
{
    auto scores = gravity_.score(tokens, phase);

    std::vector<std::pair<float, std::string>> ranked;
    {
        std::lock_guard lk(mtx_);
        for (const auto& [id, s] : scores) {
            auto it = entries_.find(id);
            if (it != entries_.end() && it->second.active && s >= gravity_threshold)
                ranked.push_back({s, id});
        }
    }
    if (ranked.empty()) return {};

    std::sort(ranked.rbegin(), ranked.rend());
    if (ranked.size() > top_k) ranked.resize(top_k);

    // Normalize: weights sum to 1.0 (LBS invariant: Σw_k = 1)
    float total = 0.0f;
    for (const auto& [s, _] : ranked) total += s;
    float inv = (total > 0.0f) ? 1.0f / total : 1.0f / (float)ranked.size();

    std::vector<std::pair<std::string, float>> blend;
    blend.reserve(ranked.size());
    for (const auto& [s, id] : ranked)
        blend.push_back({id, s * inv});
    return blend;
}

std::vector<std::string> MicronautRegistry::ids() const {
    std::lock_guard lk(mtx_);
    std::vector<std::string> out;
    for (const auto& [id, _] : entries_) out.push_back(id);
    return out;
}

} // namespace khanary::micronaut
