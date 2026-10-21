#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <mutex>
#include <fstream>
#include <array>

namespace khanary::kuhul {

// KXML gram node. Each gram lives inside a fold.
struct Gram {
    std::string key;
    std::string value;
    uint64_t    seq;        // monotone insertion order
    std::string phase;      // which phase wrote this gram
    std::string channel;    // semantic channel (e.g. similarity/control/predication)
    std::string authority;  // semantics authority for this gram
};

// GramStore: Pop=gramRead, Xul=gramWrite, K'ayab'→Kumk'u=gramCheckpoint
// In-memory with optional JSONL append log
class GramStore {
public:
    GramStore();
    explicit GramStore(const std::string& journal_path);

    // Pop phase: read a gram by key
    std::optional<Gram> gramRead(const std::string& key) const;

    // Xul phase: write a gram (creates or updates).
    // channel/authority keep control-flow semantics explicit and separate from similarity lanes.
    void gramWrite(
        const std::string& key,
        const std::string& value,
        const std::string& phase = "Xul",
        const std::string& channel = "general",
        const std::string& authority = "definitions_and_relations");

    // Convenience surface for vector metadata records.
    void gramWriteVector(
        const std::string& key,
        float x, float y, float z,
        const std::string& phase = "Xul",
        const std::string& channel = "taxonomy_geometry",
        const std::string& authority = "vector_meta");
    std::optional<std::array<float, 3>> gramVector(const std::string& key) const;

    // K'ayab' → Kumk'u: checkpoint all grams to journal
    void gramCheckpoint();

    // Replay a previously saved journal into this store
    void replay(const std::string& path);

    // Load a scraper-generated code corpus (key/text/phase/domain JSONL format
    // produced by tools/scrape.py). Skips keys already in the store so live
    // runtime entries always win. Does not append to the journal.
    // key_prefix: if non-empty, all keys are stored as "{key_prefix}:{key}" so
    //   per-expert corpora don't collide (e.g. "cpp", "kuhul", "py").
    // Returns the number of entries loaded.
    size_t load_corpus(const std::string& path, const std::string& key_prefix = "");

    bool has(const std::string& key) const;
    size_t size() const;
    size_t replay_error_count() const;
    std::optional<std::string> last_replay_error() const;

    // List all gram keys
    std::vector<std::string> keys() const;

    // Write a live snapshot of mem:* and memory:event:* entries to
    // data_dir/drivers/micronauts/semantic/memory-state.json so that
    // find_best_semantic_meta() and load_pattern_meta_for_track() can inject
    // current memory state as LFM sys context for MEMORY-µ dispatch.
    void flush_memory_driver(const std::string& data_dir) const;

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, Gram> grams_;
    uint64_t seq_ = 0;
    std::string journal_path_;
    std::ofstream journal_;
    size_t replay_errors_ = 0;
    std::optional<std::string> last_replay_error_;

    void append_journal(const Gram& g);
    void register_replay_error_unlocked(const std::string& message);
};

} // namespace khanary::kuhul
