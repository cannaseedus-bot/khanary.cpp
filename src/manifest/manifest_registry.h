#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <optional>

namespace khanary::manifest {

// Manifest registry: key → array of strings (expansion keys or raw strings)
// O(1) lookup; lazy recursive expansion via SCXResolver.
class ManifestRegistry {
public:
    using Array = std::vector<std::string>;

    // Load (and merge) a manifest JSON file into this registry
    void load(const std::string& path);

    // Insert a key→array directly
    void insert(const std::string& key, Array values);

    bool has(const std::string& key) const;

    // Raw array (unexpanded)
    const Array* raw(const std::string& key) const;

    // All keys
    std::vector<std::string> keys() const;

    // Resolved (recursively expanded) array — cached after first resolution
    // Returns all leaf strings (cycles are broken at depth limit)
    std::vector<std::string> resolve(const std::string& key) const;

    size_t size() const;

private:
    struct Entry {
        Array              raw;
        mutable std::optional<std::vector<std::string>> cache;
    };

    mutable std::shared_mutex mtx_;
    std::unordered_map<std::string, Entry> map_;

    std::vector<std::string> resolve_inner(
        const std::string& key,
        std::vector<std::string>& visited,
        int depth) const;
};

} // namespace khanary::manifest
