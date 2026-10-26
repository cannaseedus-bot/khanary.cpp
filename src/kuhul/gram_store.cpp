#include "gram_store.h"
#include "../util/json.h"
#include <sstream>
#include <fstream>
#include <filesystem>
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
    gramWriteProvenance(key, value, phase, channel, authority,
                        "system_corpus", "global", "evidence_only");
}

void GramStore::gramWriteProvenance(
    const std::string& key,
    const std::string& value,
    const std::string& phase,
    const std::string& channel,
    const std::string& authority,
    const std::string& source_class,
    const std::string& scope_id,
    const std::string& learning_state)
{
    std::lock_guard lk(mtx_);
    const std::string norm_channel = normalize_channel(channel);
    const std::string norm_authority = normalize_authority(authority, norm_channel);
    Gram g{key, value, ++seq_, phase, norm_channel, norm_authority,
           source_class, scope_id, learning_state};
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
    // Gram values are free-form text; only rows written by gramWriteVector
    // carry JSON payloads. Non-JSON values are simply not vector rows, so a
    // Pop-phase read must not let util::parse throw through (the scorer calls
    // this for every candidate when no explicit overrides are supplied).
    JsonPtr parsed;
    try {
        parsed = parse(g->value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
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
    obj["source_class"] = json_string(g.source_class);
    obj["scope_id"] = json_string(g.scope_id);
    obj["learning_state"] = json_string(g.learning_state);
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
            g.source_class = v->get_string("source_class").value_or("system_corpus");
            g.scope_id = v->get_string("scope_id").value_or("global");
            g.learning_state = v->get_string("learning_state").value_or("evidence_only");
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

size_t GramStore::load_corpus(const std::string& path, const std::string& key_prefix) {
    std::ifstream f(path);
    if (!f) return 0;

    // Map scraper domain strings to GramStore channels.
    auto domain_to_channel = [](const std::string& domain) -> std::string {
        if (domain == "kuhul" || domain == "klsl" || domain == "khl")
            return "xcfe_control_program";
        if (domain == "json")
            return "taxonomy_geometry";
        return "predication_sequence"; // py, js, ts, cpp, default
    };

    // K'UHUL-syntax tokens start with @, ⟁ (UTF-8 E2 9F 81), or [.
    // These get surface-alias grams so models can look them up by token.
    auto is_kuhul_token = [](const std::string& s) {
        return !s.empty() && (s[0] == '@' || (unsigned char)s[0] == 0xe2 || s[0] == '[');
    };

    // Hard cap to prevent accidental full-load of the 11M-entry corpus.
    // The filtered codegen-kuhul.jsonl is expected to be well under this.
    constexpr size_t kMaxEntries = 500'000;

    std::string line;
    size_t loaded = 0;
    std::lock_guard lk(mtx_);
    while (std::getline(f, line) && loaded < kMaxEntries) {
        if (line.empty()) continue;
        try {
            auto v = parse(line);
            if (!v || !v->is_object()) continue;
            const std::string key    = v->get_string("key").value_or("");
            const std::string text   = v->get_string("text").value_or("");
            const std::string phase  = v->get_string("phase").value_or("Wo");
            const std::string domain = v->get_string("domain").value_or("py");
            const std::string source = v->get_string("source").value_or("");
            if (key.empty()) continue;

            // Rich JSON value: snippet + domain + source for CODEGEN-µ context.
            JsonObject val;
            val["text"]   = json_string(text);
            val["domain"] = json_string(domain);
            if (!source.empty()) val["source"] = json_string(source);
            const std::string value_str = stringify(json_object(std::move(val)));

            const std::string channel   = normalize_channel(domain_to_channel(domain));
            const std::string authority = normalize_authority("codegen_corpus", channel);

            // Namespace the key when a prefix is supplied so per-expert corpora
            // don't collide: "cpp:fold_engine_phase", "kuhul:@if_@then", etc.
            const std::string store_key = key_prefix.empty() ? key : (key_prefix + ":" + key);
            if (!grams_.count(store_key)) {
                grams_[store_key] = Gram{store_key, value_str, ++seq_, phase, channel, authority};
                ++loaded;
            }

            // For K'UHUL/KLSL/KHL domain entries, write surface-alias grams for
            // @-ops, ⟁-keys, and [Phase] tokens so models can look up K'UHUL
            // syntax directly (e.g. "kuhul:codegen:@if" -> bigram context snippet).
            if ((domain == "kuhul" || domain == "klsl" || domain == "khl") && loaded < kMaxEntries) {
                const JsonValue* sarr = v->get("surfaces");
                if (sarr && sarr->is_array()) {
                    for (const auto& sp : sarr->as_array()) {
                        if (!sp || !sp->is_string()) continue;
                        const std::string surf = sp->as_string();
                        if (surf == key || !is_kuhul_token(surf)) continue;
                        const std::string alias_base = "codegen:" + surf;
                        const std::string alias = key_prefix.empty()
                            ? alias_base : (key_prefix + ":" + alias_base);
                        if (!grams_.count(alias)) {
                            grams_[alias] = Gram{alias, value_str, ++seq_, phase, channel, authority};
                            ++loaded;
                        }
                    }
                }
            }
        } catch (...) {
            continue;
        }
    }
    return loaded;
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

void GramStore::flush_memory_driver(const std::string& data_dir) const {
    namespace fs = std::filesystem;

    // Collect mem:* and memory:event:* grams, sorted by seq (oldest first).
    std::vector<const Gram*> entries;
    {
        std::lock_guard lk(mtx_);
        for (const auto& [k, g] : grams_) {
            if (k.rfind("mem:", 0) == 0 || k.rfind("memory:event:", 0) == 0)
                entries.push_back(&g);
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const Gram* a, const Gram* b) { return a->seq < b->seq; });

    // Build the response text.
    std::ostringstream body;
    body << "MEMORY-\xc2\xb5 operation vocabulary:\n"
         << "  op=save   phase=Xul  — gramWrite to mem:{category}:{key}\n"
         << "  op=recall phase=Pop  — gramRead from mem:{category}:{key}\n"
         << "  op=list   phase=Pop  — enumerate keys by category\n"
         << "  op=forget phase=Xul  — clear key\n"
         << "  categories: preference | session | note | micronaut_data\n"
         << "  event keys: memory:event:{kind}:{epoch}-{seq}\n\n";

    if (entries.empty()) {
        body << "No entries in gram store yet.\n";
    } else {
        body << "Current gram store entries (" << entries.size() << "):\n";
        // Emit at most 30 entries to keep the driver compact.
        const size_t limit = std::min(entries.size(), size_t{30});
        for (size_t i = entries.size() > limit ? entries.size() - limit : 0;
             i < entries.size(); ++i) {
            const Gram* g = entries[i];
            body << "  [" << g->phase << "] " << g->key << " = ";
            // Truncate long values.
            if (g->value.size() > 80)
                body << g->value.substr(0, 80) << "…";
            else
                body << g->value;
            body << "\n";
        }
    }

    // Write driver JSON.
    const fs::path dir = fs::path(data_dir) / "drivers" / "micronauts" / "semantic";
    if (!fs::exists(dir)) {
        try { fs::create_directories(dir); } catch (...) { return; }
    }
    const fs::path out_path = dir / "memory-state.json";

    // Read existing to preserve metadata fields (loadCount, etc.) if present.
    std::string load_count_str = "0";
    {
        std::ifstream in(out_path);
        if (in) {
            std::ostringstream ss; ss << in.rdbuf();
            try {
                auto root = util::parse(ss.str());
                if (root && root->is_object())
                    if (auto lc = root->get_string("loadCount"))
                        load_count_str = *lc;
            } catch (...) {}
        }
    }

    // Escape body for JSON string.
    std::string resp = body.str();
    std::string escaped;
    escaped.reserve(resp.size() + 32);
    for (unsigned char c : resp) {
        if      (c == '"')  escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else                escaped += static_cast<char>(c);
    }

    std::ofstream f(out_path, std::ios::trunc);
    if (!f) return;
    f << "{\n"
      << "  \"id\": \"memory-state\",\n"
      << "  \"subject\": \"MEMORY-\\u00b5 live gram store snapshot\",\n"
      << "  \"capability\": \"memory\",\n"
      << "  \"fold\": \"Pop\",\n"
      << "  \"loadCount\": " << load_count_str << ",\n"
      << "  \"response\": \"" << escaped << "\",\n"
      << "  \"related\": [\"user-state\"],\n"
      << "  \"sources\": [],\n"
      << "  \"tags\": {\n"
      << "    \"source\": \"gram_store_flush\",\n"
      << "    \"keywords\": \"memory gram store recall save preference session note\"\n"
      << "  },\n"
      << "  \"executeCount\": 0,\n"
      << "  \"contributionCount\": 0\n"
      << "}\n";
}

} // namespace khanary::kuhul
