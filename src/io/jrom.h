#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <vector>
#include <cstdint>
#include <functional>

namespace khanary::io {

struct JROMEvent {
    std::string type;
    std::string payload_json;
    uint64_t    seq;
};

// JROM: append-only JSONL audit journal
// Logs tool events, phase transitions, fold executions
class JROM {
public:
    JROM();
    explicit JROM(const std::string& path);

    bool open(const std::string& path);
    void close();

    // Append a log entry
    uint64_t log(const std::string& event_type, const std::string& payload_json);

    // Replay from offset; calls cb(event) for each entry
    void replay(uint64_t from_seq,
                std::function<void(const JROMEvent&)> cb) const;

    // Read all events into vector
    std::vector<JROMEvent> read_all() const;

    uint64_t last_seq() const;
    bool is_open() const;

private:
    mutable std::mutex mtx_;
    std::ofstream out_;
    std::string   path_;
    uint64_t      seq_ = 0;
};

} // namespace khanary::io
