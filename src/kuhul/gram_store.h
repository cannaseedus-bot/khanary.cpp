#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <mutex>
#include <fstream>

namespace khanary::kuhul {

// KXML gram node. Each gram lives inside a fold.
struct Gram {
    std::string key;
    std::string value;
    uint64_t    seq;        // monotone insertion order
    std::string phase;      // which phase wrote this gram
};

// GramStore: Pop=gramRead, Xul=gramWrite, K'ayab'→Kumk'u=gramCheckpoint
// In-memory with optional JSONL append log
class GramStore {
public:
    GramStore();
    explicit GramStore(const std::string& journal_path);

    // Pop phase: read a gram by key
    std::optional<Gram> gramRead(const std::string& key) const;

    // Xul phase: write a gram (creates or updates)
    void gramWrite(const std::string& key, const std::string& value, const std::string& phase = "Xul");

    // K'ayab' → Kumk'u: checkpoint all grams to journal
    void gramCheckpoint();

    // Replay a previously saved journal into this store
    void replay(const std::string& path);

    bool has(const std::string& key) const;
    size_t size() const;

    // List all gram keys
    std::vector<std::string> keys() const;

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, Gram> grams_;
    uint64_t seq_ = 0;
    std::string journal_path_;
    std::ofstream journal_;

    void append_journal(const Gram& g);
};

} // namespace khanary::kuhul
