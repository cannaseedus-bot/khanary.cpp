#include "jrom.h"
#include <fstream>
#include <sstream>
#include <filesystem>

namespace khanary::io {

namespace fs = std::filesystem;

JROM::JROM() {}

JROM::JROM(const std::string& path) { open(path); }

bool JROM::open(const std::string& path) {
    path_ = path;
    fs::path p(path);
    if (p.has_parent_path())
        fs::create_directories(p.parent_path());

    // Count existing entries to get starting seq
    {
        std::ifstream f(path);
        if (f) {
            std::string line;
            while (std::getline(f, line)) ++seq_;
        }
    }
    out_.open(path, std::ios::app);
    return out_.is_open();
}

void JROM::close() {
    out_.close();
}

uint64_t JROM::log(const std::string& event_type, const std::string& payload_json) {
    std::lock_guard lk(mtx_);
    uint64_t s = ++seq_;
    // {"seq":N,"type":"...","payload":{...}}
    std::string entry = "{\"seq\":" + std::to_string(s) +
                        ",\"type\":\"" + event_type + "\"" +
                        ",\"payload\":" + payload_json + "}\n";
    if (out_.is_open()) {
        out_ << entry;
        out_.flush();
    }
    return s;
}

void JROM::replay(uint64_t from_seq,
                  std::function<void(const JROMEvent&)> cb) const
{
    if (path_.empty()) return;
    std::ifstream f(path_);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Quick parse: extract seq
        size_t seq_pos = line.find("\"seq\":");
        if (seq_pos == std::string::npos) continue;
        uint64_t s = std::stoull(line.substr(seq_pos + 6));
        if (s < from_seq) continue;
        // Extract type
        size_t type_pos = line.find("\"type\":\"");
        std::string type_val;
        if (type_pos != std::string::npos) {
            size_t start = type_pos + 8;
            size_t end = line.find('"', start);
            type_val = line.substr(start, end - start);
        }
        // payload is the rest
        size_t payload_pos = line.find("\"payload\":");
        std::string payload;
        if (payload_pos != std::string::npos) {
            payload = line.substr(payload_pos + 10);
            if (!payload.empty() && payload.back() == '}') payload.pop_back();
        }
        cb({type_val, payload, s});
    }
}

std::vector<JROMEvent> JROM::read_all() const {
    std::vector<JROMEvent> out;
    replay(0, [&](const JROMEvent& e){ out.push_back(e); });
    return out;
}

uint64_t JROM::last_seq() const {
    std::lock_guard lk(mtx_);
    return seq_;
}

bool JROM::is_open() const {
    return out_.is_open();
}

} // namespace khanary::io
