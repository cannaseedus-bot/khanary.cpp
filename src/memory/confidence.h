#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace khanary::memory {

// L_t: persistent confidence store
// Updated only via Ch'en→Xul gate (not during Pop–Sek phases)
class ConfidenceStore {
public:
    ConfidenceStore();
    explicit ConfidenceStore(const std::string& path);

    // Load from confidence.json
    bool load(const std::string& path);

    // Save to confidence.json
    bool save(const std::string& path) const;

    // Get confidence for a micronaut [0, 1], defaults to 0.5
    float get(const std::string& micronaut_id) const;

    // Update confidence (only called from Xul commit gate)
    // delta ∈ [-1, +1]
    void update(const std::string& micronaut_id, float delta);

    // Direct set (factory patch)
    void set(const std::string& micronaut_id, float value);

    // Return all entries as float vector (for perception embedding)
    std::vector<float> as_vector(const std::vector<std::string>& ordered_ids) const;

    std::unordered_map<std::string, float> snapshot() const;

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, float> scores_;
    std::string persist_path_;
};

} // namespace khanary::memory
