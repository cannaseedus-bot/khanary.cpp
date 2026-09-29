#include "confidence.h"
#include "../util/json.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace khanary::memory {

using namespace khanary::util;

namespace {

static float clamp_score(float value, float fallback = 0.5f) {
    if (!std::isfinite(value)) return fallback;
    return std::clamp(value, 0.0f, 1.0f);
}

static float clamp_delta(float value) {
    if (!std::isfinite(value)) return 0.0f;
    return std::clamp(value, -1.0f, 1.0f);
}

static std::string resolve_path(const std::string& requested, const std::string& persisted) {
    return requested.empty() ? persisted : requested;
}

} // namespace

ConfidenceStore::ConfidenceStore() {}

ConfidenceStore::ConfidenceStore(const std::string& path)
    : persist_path_(path)
{
    load(path);
}

bool ConfidenceStore::load(const std::string& path) {
    const std::string resolved = resolve_path(path, persist_path_);
    if (resolved.empty()) return false;

    std::ifstream f(resolved);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    auto root = parse(ss.str());
    if (!root || !root->is_object()) return false;

    std::unordered_map<std::string, float> parsed;
    parsed.reserve(root->as_object().size());
    for (const auto& [k, v] : root->as_object()) {
        if (k.empty() || !v || !v->is_number()) continue;
        parsed[k] = clamp_score(static_cast<float>(v->as_number()));
    }

    std::lock_guard lk(mtx_);
    scores_.swap(parsed);
    persist_path_ = resolved;
    return true;
}

bool ConfidenceStore::save(const std::string& path) const {
    std::string resolved_path;
    std::unordered_map<std::string, float> snapshot;
    {
        std::lock_guard lk(mtx_);
        resolved_path = resolve_path(path, persist_path_);
        if (resolved_path.empty()) return false;
        snapshot = scores_;
    }

    JsonObject obj;
    for (const auto& [k, v] : snapshot)
        obj[k] = json_number(clamp_score(v));

    std::ofstream f(resolved_path);
    if (!f) return false;
    f << stringify(json_object(std::move(obj)), true);
    return true;
}

float ConfidenceStore::get(const std::string& id) const {
    if (id.empty()) return 0.5f;
    std::lock_guard lk(mtx_);
    auto it = scores_.find(id);
    return it != scores_.end() ? clamp_score(it->second) : 0.5f;
}

void ConfidenceStore::update(const std::string& id, float delta) {
    if (id.empty()) return;
    const float d = clamp_delta(delta);
    if (d == 0.0f) return;

    std::lock_guard lk(mtx_);
    auto [it, inserted] = scores_.emplace(id, 0.5f);
    float& v = it->second;
    v = clamp_score(v + d);
}

void ConfidenceStore::set(const std::string& id, float value) {
    if (id.empty()) return;
    std::lock_guard lk(mtx_);
    scores_[id] = clamp_score(value);
}

std::vector<float> ConfidenceStore::as_vector(const std::vector<std::string>& ids) const {
    std::lock_guard lk(mtx_);
    std::vector<float> v;
    v.reserve(ids.size());
    for (const auto& id : ids) {
        if (id.empty()) {
            v.push_back(0.5f);
            continue;
        }
        auto it = scores_.find(id);
        v.push_back(it != scores_.end() ? clamp_score(it->second) : 0.5f);
    }
    return v;
}

std::unordered_map<std::string, float> ConfidenceStore::snapshot() const {
    std::lock_guard lk(mtx_);
    return scores_;
}

} // namespace khanary::memory
