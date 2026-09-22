#include "confidence.h"
#include "../util/json.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace khanary::memory {

using namespace khanary::util;

ConfidenceStore::ConfidenceStore() {}

ConfidenceStore::ConfidenceStore(const std::string& path)
    : persist_path_(path)
{
    load(path);
}

bool ConfidenceStore::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    auto root = parse(ss.str());
    if (!root || !root->is_object()) return false;
    std::lock_guard lk(mtx_);
    for (const auto& [k, v] : root->as_object()) {
        if (v->is_number())
            scores_[k] = std::max(0.0f, std::min(1.0f, (float)v->as_number()));
    }
    return true;
}

bool ConfidenceStore::save(const std::string& path) const {
    std::lock_guard lk(mtx_);
    JsonObject obj;
    for (const auto& [k, v] : scores_)
        obj[k] = json_number(v);
    std::ofstream f(path);
    if (!f) return false;
    f << stringify(json_object(std::move(obj)), true);
    return true;
}

float ConfidenceStore::get(const std::string& id) const {
    std::lock_guard lk(mtx_);
    auto it = scores_.find(id);
    return it != scores_.end() ? it->second : 0.5f;
}

void ConfidenceStore::update(const std::string& id, float delta) {
    std::lock_guard lk(mtx_);
    float& v = scores_[id];
    v = std::max(0.0f, std::min(1.0f, v + delta));
}

void ConfidenceStore::set(const std::string& id, float value) {
    std::lock_guard lk(mtx_);
    scores_[id] = std::max(0.0f, std::min(1.0f, value));
}

std::vector<float> ConfidenceStore::as_vector(const std::vector<std::string>& ids) const {
    std::lock_guard lk(mtx_);
    std::vector<float> v;
    v.reserve(ids.size());
    for (const auto& id : ids) {
        auto it = scores_.find(id);
        v.push_back(it != scores_.end() ? it->second : 0.5f);
    }
    return v;
}

std::unordered_map<std::string, float> ConfidenceStore::snapshot() const {
    std::lock_guard lk(mtx_);
    return scores_;
}

} // namespace khanary::memory
