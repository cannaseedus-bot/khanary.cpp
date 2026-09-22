#include "gram_store.h"
#include "../util/json.h"
#include <sstream>
#include <fstream>
#include <stdexcept>

namespace khanary::kuhul {

using namespace khanary::util;

GramStore::GramStore() {}

GramStore::GramStore(const std::string& journal_path)
    : journal_path_(journal_path)
{
    journal_.open(journal_path, std::ios::app);
}

std::optional<Gram> GramStore::gramRead(const std::string& key) const {
    std::lock_guard lk(mtx_);
    auto it = grams_.find(key);
    if (it == grams_.end()) return std::nullopt;
    return it->second;
}

void GramStore::gramWrite(const std::string& key, const std::string& value, const std::string& phase) {
    std::lock_guard lk(mtx_);
    Gram g{key, value, ++seq_, phase};
    grams_[key] = g;
    if (journal_.is_open()) append_journal(g);
}

void GramStore::gramCheckpoint() {
    std::lock_guard lk(mtx_);
    if (!journal_.is_open() && !journal_path_.empty()) {
        journal_.open(journal_path_, std::ios::app);
    }
    if (!journal_.is_open()) return;
    for (const auto& [k, g] : grams_) append_journal(g);
    journal_.flush();
}

void GramStore::append_journal(const Gram& g) {
    // {"key":"...","value":"...","seq":N,"phase":"..."}
    JsonObject obj;
    obj["key"]   = json_string(g.key);
    obj["value"] = json_string(g.value);
    obj["seq"]   = json_number((double)g.seq);
    obj["phase"] = json_string(g.phase);
    journal_ << stringify(json_object(std::move(obj))) << '\n';
}

void GramStore::replay(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    std::lock_guard lk(mtx_);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            auto v = parse(line);
            if (!v || !v->is_object()) continue;
            Gram g;
            g.key   = v->get_string("key").value_or("");
            g.value = v->get_string("value").value_or("");
            g.seq   = (uint64_t)v->get_number("seq").value_or(0.0);
            g.phase = v->get_string("phase").value_or("Xul");
            if (g.key.empty()) continue;
            grams_[g.key] = g;
            if (g.seq > seq_) seq_ = g.seq;
        } catch (...) {}
    }
}

bool GramStore::has(const std::string& key) const {
    std::lock_guard lk(mtx_);
    return grams_.count(key) > 0;
}

size_t GramStore::size() const {
    std::lock_guard lk(mtx_);
    return grams_.size();
}

std::vector<std::string> GramStore::keys() const {
    std::lock_guard lk(mtx_);
    std::vector<std::string> out;
    out.reserve(grams_.size());
    for (const auto& [k, _] : grams_) out.push_back(k);
    return out;
}

} // namespace khanary::kuhul
