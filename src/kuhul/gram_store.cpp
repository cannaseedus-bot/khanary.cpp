#include "gram_store.h"
#include "../util/json.h"
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <array>
#include <cmath>

namespace khanary::kuhul {

using namespace khanary::util;

namespace {

static std::string trim_copy(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string normalize_channel(const std::string& raw) {
    const std::string v = lower_copy(trim_copy(raw));
    if (v.empty()) return "general";
    if (v == "similarity" || v == "similarity_kernel" || v == "semantic_fold")
        return "similarity_kernel";
    if (v == "control" || v == "control_flow" || v == "xcfe_control")
        return "xcfe_control_program";
    if (v == "predication" || v == "sequence" || v == "syntax")
        return "predication_sequence";
    if (v == "taxonomy" || v == "concept")
        return "taxonomy_geometry";
    return v;
}

static std::string normalize_authority(const std::string& raw, const std::string& channel) {
    const std::string v = lower_copy(trim_copy(raw));
    if (!v.empty()) return v;
    if (channel == "xcfe_control_program") return "xcfe_control_program";
    return "definitions_and_relations";
}

} // namespace

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

void GramStore::gramWrite(
    const std::string& key,
    const std::string& value,
    const std::string& phase,
    const std::string& channel,
    const std::string& authority)
{
    std::lock_guard lk(mtx_);
    const std::string norm_channel = normalize_channel(channel);
    const std::string norm_authority = normalize_authority(authority, norm_channel);
    Gram g{key, value, ++seq_, phase, norm_channel, norm_authority};
    grams_[key] = g;
    if (journal_.is_open()) {
        append_journal(g);
        journal_.flush();
    }
}

void GramStore::gramWriteVector(
    const std::string& key,
    float x, float y, float z,
    const std::string& phase,
    const std::string& channel,
    const std::string& authority)
{
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return;
    JsonObject vec;
    vec["x"] = json_number(static_cast<double>(x));
    vec["y"] = json_number(static_cast<double>(y));
    vec["z"] = json_number(static_cast<double>(z));
    gramWrite(key,
              stringify(json_object(std::move(vec))),
              phase,
              channel,
              authority);
}

std::optional<std::array<float, 3>> GramStore::gramVector(const std::string& key) const {
    auto g = gramRead(key);
    if (!g || g->value.empty()) return std::nullopt;
    auto parsed = parse(g->value);
    if (!parsed || !parsed->is_object()) return std::nullopt;
    const auto x = parsed->get_number("x");
    const auto y = parsed->get_number("y");
    const auto z = parsed->get_number("z");
    if (!x || !y || !z) return std::nullopt;
    if (!std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) return std::nullopt;
    return std::array<float, 3>{
        static_cast<float>(*x),
        static_cast<float>(*y),
        static_cast<float>(*z)
    };
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
    // {"key":"...","value":"...","seq":N,"phase":"...","channel":"...","authority":"..."}
    JsonObject obj;
    obj["key"]   = json_string(g.key);
    obj["value"] = json_string(g.value);
    obj["seq"]   = json_number((double)g.seq);
    obj["phase"] = json_string(g.phase);
    obj["channel"] = json_string(g.channel);
    obj["authority"] = json_string(g.authority);
    journal_ << stringify(json_object(std::move(obj))) << '\n';
}

void GramStore::register_replay_error_unlocked(const std::string& message) {
    ++replay_errors_;
    last_replay_error_ = message;
}

void GramStore::replay(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    std::lock_guard lk(mtx_);
    replay_errors_ = 0;
    last_replay_error_.reset();
    size_t line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        if (line.empty()) continue;
        try {
            auto v = parse(line);
            if (!v || !v->is_object()) {
                register_replay_error_unlocked("invalid journal JSON at line " + std::to_string(line_no));
                continue;
            }
            Gram g;
            g.key   = v->get_string("key").value_or("");
            g.value = v->get_string("value").value_or("");
            g.seq   = (uint64_t)v->get_number("seq").value_or(0.0);
            g.phase = v->get_string("phase").value_or("Xul");
            g.channel = normalize_channel(v->get_string("channel").value_or("general"));
            g.authority = normalize_authority(v->get_string("authority").value_or(""), g.channel);
            if (g.key.empty()) {
                register_replay_error_unlocked("journal record missing key at line " + std::to_string(line_no));
                continue;
            }
            grams_[g.key] = g;
            if (g.seq > seq_) seq_ = g.seq;
        } catch (const std::exception& e) {
            register_replay_error_unlocked(
                "journal replay error at line " + std::to_string(line_no) + ": " + e.what());
            continue;
        }
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

size_t GramStore::replay_error_count() const {
    std::lock_guard lk(mtx_);
    return replay_errors_;
}

std::optional<std::string> GramStore::last_replay_error() const {
    std::lock_guard lk(mtx_);
    return last_replay_error_;
}

std::vector<std::string> GramStore::keys() const {
    std::lock_guard lk(mtx_);
    std::vector<std::string> out;
    out.reserve(grams_.size());
    for (const auto& [k, _] : grams_) out.push_back(k);
    return out;
}

} // namespace khanary::kuhul
