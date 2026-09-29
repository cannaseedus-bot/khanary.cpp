#include "manifest_registry.h"
#include "../util/json.h"
#include "../util/schema.h"
#include "../config.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace khanary::manifest {

using namespace khanary::util;

void ManifestRegistry::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("ManifestRegistry: cannot open " + path);
    std::ostringstream ss; ss << f.rdbuf();
    auto root = parse(ss.str());
    if (!root || !root->is_object())
        throw std::runtime_error("ManifestRegistry: root must be object in " + path);

    std::string error;
    const auto schema = std::filesystem::path(path).parent_path().parent_path() / "schema/manifest-map.schema.json";
    if (!validate_schema_file(*root, schema, &error))
        throw std::runtime_error("ManifestRegistry: " + error);

    // Validate the entire map before committing any entries. Objects have the
    // typed XCFE conditional shape and are serialized as declarative tokens.
    std::unique_lock lk(mtx_);
    for (const auto& [k, v] : root->as_object()) {
        Array arr;
        if (v->is_string()) {
            arr.push_back(v->as_string());
        } else if (v->is_array()) {
            for (const auto& elem : v->as_array()) {
                if (elem->is_string()) arr.push_back(elem->as_string());
                else arr.push_back(stringify(elem));
            }
        }
        map_[k].raw = std::move(arr);
        map_[k].cache.reset();
    }
}

void ManifestRegistry::insert(const std::string& key, Array values) {
    std::unique_lock lk(mtx_);
    map_[key].raw = std::move(values);
    map_[key].cache.reset();
}

bool ManifestRegistry::has(const std::string& key) const {
    std::shared_lock lk(mtx_);
    return map_.count(key) > 0;
}

const ManifestRegistry::Array* ManifestRegistry::raw(const std::string& key) const {
    std::shared_lock lk(mtx_);
    auto it = map_.find(key);
    if (it == map_.end()) return nullptr;
    return &it->second.raw;
}

std::vector<std::string> ManifestRegistry::keys() const {
    std::shared_lock lk(mtx_);
    std::vector<std::string> out;
    out.reserve(map_.size());
    for (const auto& [k, _] : map_) out.push_back(k);
    return out;
}

std::vector<std::string> ManifestRegistry::resolve(const std::string& key) const {
    {
        std::shared_lock lk(mtx_);
        auto it = map_.find(key);
        if (it != map_.end() && it->second.cache)
            return *it->second.cache;
    }
    std::vector<std::string> visited;
    auto result = resolve_inner(key, visited, 0);
    {
        std::unique_lock lk(mtx_);
        auto it = map_.find(key);
        if (it != map_.end()) it->second.cache = result;
    }
    return result;
}

std::vector<std::string> ManifestRegistry::resolve_inner(
    const std::string& key,
    std::vector<std::string>& visited,
    int depth) const
{
    if (depth >= (int)SCX_MAX_DEPTH) return {"[SCX_DEPTH_LIMIT]"};
    if (std::find(visited.begin(), visited.end(), key) != visited.end())
        return {"[SCX_CYCLE:" + key + "]"};

    std::shared_lock lk(mtx_);
    auto it = map_.find(key);
    if (it == map_.end()) return {key}; // not a key → treat as raw string

    const auto& arr = it->second.raw;
    lk.unlock();

    std::vector<std::string> result;
    visited.push_back(key);
    for (const auto& elem : arr) {
        if (elem.size() > 1 && elem[0] == '@') {
            // @node:Name → inline expansion
            if (elem.rfind("@node:", 0) == 0) {
                auto sub = resolve_inner(elem.substr(6), visited, depth + 1);
                result.insert(result.end(), sub.begin(), sub.end());
            } else {
                result.push_back(elem); // declarative operator — pass through
            }
        } else {
            std::shared_lock lk2(mtx_);
            bool is_key = map_.count(elem) > 0;
            lk2.unlock();
            if (is_key) {
                auto sub = resolve_inner(elem, visited, depth + 1);
                result.insert(result.end(), sub.begin(), sub.end());
            } else {
                result.push_back(elem);
            }
        }
    }
    visited.pop_back();
    return result;
}

size_t ManifestRegistry::size() const {
    std::shared_lock lk(mtx_);
    return map_.size();
}

} // namespace khanary::manifest
