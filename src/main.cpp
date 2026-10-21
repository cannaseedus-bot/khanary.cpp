#include "config.h"
#include "manifest/manifest_registry.h"
#include "manifest/manifest_loader.h"
#include "kuhul/phase_engine.h"
#include "kuhul/fold_delta.h"
#include "kuhul/fold_direction.h"
#include "kuhul/fold_engine.h"
#include "kuhul/xcfe_executor.h"
#include "kuhul/gram_store.h"
#include "kuhul/phase_driver_lock.h"
#include "kuhul/semantic_accumulator.h"
#include "kuhul/kuhul_vm.h"
#include "gpu/d3d11_device.h"
#include "gpu/dml_ops.h"
#include "gpu/unified_tensor.h"
#include "gpu/dds_loader.h"
#include "gpu/hot_swap.h"
#include "gpu/svg_compute_runtime.h"
#include "gpu/fold_score_dispatch.h"
#include "inference/tokenizer.h"
#include "inference/inference_engine.h"
#include "inference/semantic_attention.h"
#include "decision/decision_mu.h"
#include "micronaut/micronaut_registry.h"
#include "micronaut/track_loader.h"
#include "micronaut/aiml_engine.h"
#include "micronaut/mx2lm-ngram.h"
#include "micronaut/micronaut_coder.h"
#include "micronaut/micronaut_transpiler.h"
#include "micronaut/factory.h"
#include "micronaut/greeting_style.h"
#include "micronaut/control_flow.h"
#include "micronaut/relation_isa.h"
#include "micronaut/sandbox_inspector.h"
#include "micronaut/neighborhood_selector.h"
#include "world/world_state.h"
#include "world/field_dag.h"
#include "memory/perception.h"
#include "memory/confidence.h"
#include "memory/semantic_graph.h"
#include "memory/scoring.h"
#include "io/json_rpc_server.h"
#include "io/chat_completion.h"
#include "io/mcp_server.h"
#include "io/code_parser.h"
#include "io/jrom.h"
#include "io/projection.h"
#include "io/request_router.h"
#include "io/runtime_attention.h"
#include "kuhul/xul_gate.h"
#include "util/json.h"
#include "util/sha256.h"
#include "util/canonical_hash.h"
#include "world/gyroscope.h"
#include "world/vector_projection.h"
#include "llm/llama_infer.h"
#include "llm/model_swapper.h"
#include "io/semantic_driver.h"
#include "io/aiml_meta.h"

#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <filesystem>
#include <cstring>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <algorithm>
#include <numeric>
#include <array>
#include <cctype>
#include <chrono>
#include <ctime>
#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <limits>
#include <exception>
#include <vector>

namespace fs = std::filesystem;
using namespace khanary;
// Pure greeting-bank / salutation-policy surface (unit-tested separately).
namespace greeting = khanary::micronaut::greeting;

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int) { g_shutdown = true; }

static std::string trim_copy(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static std::string sanitize_user_manifest_segment(std::string s) {
    s = lower_copy(trim_copy(std::move(s)));
    std::string out;
    out.reserve(s.size());
    bool prev_sep = false;
    for (unsigned char uc : s) {
        if (std::isalnum(uc)) {
            out.push_back(static_cast<char>(uc));
            prev_sep = false;
        } else {
            if (!out.empty() && !prev_sep) {
                out.push_back('_');
                prev_sep = true;
            }
        }
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    if (out.empty()) return "global";
    return out;
}

static bool preload_file_to_ram(
    const fs::path& path,
    std::vector<uint8_t>* out,
    std::string* err)
{
    if (!out) {
        if (err) *err = "preload_file_to_ram: output buffer is null";
        return false;
    }

    std::error_code ec;
    const auto file_bytes = fs::file_size(path, ec);
    if (ec) {
        if (err) *err = "unable to stat file: " + path.string();
        return false;
    }
    if (file_bytes > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        if (err) *err = "file is too large to address in process memory: " + path.string();
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "unable to open file: " + path.string();
        return false;
    }

    std::vector<uint8_t> buf(static_cast<size_t>(file_bytes), 0u);
    constexpr size_t READ_CHUNK = 64ULL * 1024ULL * 1024ULL;
    size_t offset = 0;
    while (offset < buf.size()) {
        const size_t take = std::min(READ_CHUNK, buf.size() - offset);
        in.read(reinterpret_cast<char*>(buf.data() + offset), static_cast<std::streamsize>(take));
        const std::streamsize got = in.gcount();
        if (got != static_cast<std::streamsize>(take)) {
            if (err) *err = "short read while preloading GGUF: " + path.string();
            return false;
        }
        offset += take;
    }

    *out = std::move(buf);
    return true;
}

static std::string normalize_model_tag(std::string s) {
    s = lower_copy(trim_copy(s));
    auto strip_suffix = [&](const std::string& suffix) -> bool {
        if (s.size() < suffix.size()) return false;
        if (s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
        s.resize(s.size() - suffix.size());
        return true;
    };

    // Path to basename
    std::replace(s.begin(), s.end(), '\\', '/');
    const auto slash = s.find_last_of('/');
    if (slash != std::string::npos) s = s.substr(slash + 1);

    // Trim common extensions/suffixes in descending specificity.
    strip_suffix(".tensor_map.json");
    strip_suffix("_tensor_map.json");
    strip_suffix(".model_dds_manifest.json");
    strip_suffix(".dds_manifest.json");
    strip_suffix(".fold_manifest.json");
    strip_suffix(".json");
    strip_suffix(".kson");
    strip_suffix(".gguf");
    strip_suffix(".safetensors");
    strip_suffix(".tensor_map");
    strip_suffix("_tensor_map");
    strip_suffix("_folds");
    strip_suffix(".folds");

    return trim_copy(s);
}

static bool read_text_file(const fs::path& path, std::string* out, std::string* err) {
    if (!out) {
        if (err) *err = "read_text_file: output is null";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "unable to open file: " + path.string();
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return true;
}

static std::string strip_sha256_prefix(std::string value) {
    value = lower_copy(trim_copy(value));
    constexpr const char* kPrefix = "sha256:";
    if (value.rfind(kPrefix, 0) == 0) value.erase(0, 7);
    return value;
}

static bool is_pending_hash(const std::string& value) {
    const std::string low = lower_copy(value);
    return low.empty() || low.find("__pending__") != std::string::npos;
}

static bool load_json_object(const fs::path& path, util::JsonPtr* out, std::string* err) {
    if (!out) {
        if (err) *err = "load_json_object: output is null";
        return false;
    }

    std::string text;
    if (!read_text_file(path, &text, err)) return false;
    util::JsonPtr root;
    try {
        root = util::parse(text);
    } catch (const std::exception& ex) {
        if (err) *err = "parse failed for " + path.string() + ": " + ex.what();
        return false;
    }
    if (!root || !root->is_object()) {
        if (err) *err = "JSON root is not an object: " + path.string();
        return false;
    }
    *out = std::move(root);
    return true;
}

static bool compute_file_sha256_prefixed(const fs::path& path, std::string* out, std::string* err) {
    if (!out) {
        if (err) *err = "compute_file_sha256_prefixed: output is null";
        return false;
    }
    std::string bytes;
    if (!read_text_file(path, &bytes, err)) return false;
    *out = "sha256:" + lower_copy(util::sha256_hex(bytes.data(), bytes.size()));
    return true;
}

static bool validate_runtime_schema_stack(const std::string& data_dir, std::string* err) {
    const fs::path schema_dir = fs::path(data_dir) / "schema";
    const fs::path fold_runtime_path = schema_dir / "fold.runtime.json";
    const fs::path runtime_meta_path = schema_dir / "kuhul.fold-runtime.json";
    const fs::path cycle_spec_path = schema_dir / "cycle-hash.spec.json";

    util::JsonPtr fold_runtime;
    util::JsonPtr runtime_meta;
    if (!load_json_object(fold_runtime_path, &fold_runtime, err)) return false;
    if (!load_json_object(runtime_meta_path, &runtime_meta, err)) return false;

    const util::JsonValue* lock = fold_runtime->get("phase_driver_lock");
    if (!lock || !lock->is_object()) {
        if (err) *err = "missing phase_driver_lock object in fold.runtime.json";
        return false;
    }

    const bool lock_enabled = lock->get_bool("enabled").value_or(false);
    const std::string lock_mode = lower_copy(lock->get_string("mode").value_or("audit"));
    if (lock_enabled && lock_mode != "enforce") {
        if (err) *err = "phase_driver_lock.enabled requires mode=enforce";
        return false;
    }

    const util::JsonValue* phases = lock->get("phases");
    if (!phases || !phases->is_array() || phases->as_array().empty()) {
        if (err) *err = "phase_driver_lock.phases must be a non-empty array";
        return false;
    }

    const util::JsonValue* hash_map = runtime_meta->get("phase_driver_sha256");
    if (!hash_map || !hash_map->is_object()) {
        if (err) *err = "missing phase_driver_sha256 object in runtime metadata";
        return false;
    }
    if (lower_copy(hash_map->get_string("algorithm").value_or("")) != "sha256") {
        if (err) *err = "phase_driver_sha256.algorithm must be sha256";
        return false;
    }

    fs::path phase_root = fs::path(lock->get_string("dll_root").value_or("data/drivers/phases"));
    if (!phase_root.is_absolute())
        phase_root = fs::path(data_dir).parent_path() / phase_root;
    phase_root = phase_root.lexically_normal();

    for (const auto& phase_row : phases->as_array()) {
        if (!phase_row || !phase_row->is_object()) continue;
        auto phase_name = phase_row->get_string("phase");
        auto artifact_name = phase_row->get_string("dll");
        if (!phase_name || phase_name->empty() || !artifact_name || artifact_name->empty()) {
            if (err) *err = "phase_driver_lock phase row missing phase/dll";
            return false;
        }
        const fs::path artifact_path = (phase_root / *artifact_name).lexically_normal();
        std::error_code artifact_ec;
        if (!fs::exists(artifact_path, artifact_ec) || artifact_ec) {
            if (err) *err = "phase driver artifact missing: " + artifact_path.string();
            return false;
        }
        const std::string declared_hash = hash_map->get_string(*artifact_name).value_or("");
        if (is_pending_hash(declared_hash)) {
            if (err) *err = "pending or missing phase_driver_sha256 for " + *artifact_name;
            return false;
        }
        std::string actual_hash;
        if (!compute_file_sha256_prefixed(artifact_path, &actual_hash, err)) return false;
        if (strip_sha256_prefix(declared_hash) != strip_sha256_prefix(actual_hash)) {
            if (err) *err = "phase driver hash mismatch for " + *artifact_name;
            return false;
        }
    }

    const util::JsonValue* stack_hashes = runtime_meta->get("sha256");
    if (!stack_hashes || !stack_hashes->is_object()) {
        if (err) *err = "missing sha256 object in runtime metadata";
        return false;
    }
    if (lower_copy(stack_hashes->get_string("algorithm").value_or("")) != "sha256") {
        if (err) *err = "runtime metadata sha256.algorithm must be sha256";
        return false;
    }

    const auto validate_anchor = [&](const std::string& key, const fs::path& target) -> bool {
        const std::string declared = stack_hashes->get_string(key).value_or("");
        if (!declared.starts_with("sha256:") || is_pending_hash(declared)) {
            if (err) *err = "missing hash anchor for " + key;
            return false;
        }
        std::string actual;
        if (!compute_file_sha256_prefixed(target, &actual, err)) return false;
        if (strip_sha256_prefix(declared) != strip_sha256_prefix(actual)) {
            if (err) *err = "hash anchor mismatch for " + key;
            return false;
        }
        return true;
    };

    if (!validate_anchor("fold_runtime", fold_runtime_path)) return false;
    if (!validate_anchor("cycle_hash_spec", cycle_spec_path)) return false;
    return true;
}

static std::optional<std::string> infer_dds_model_identity_from_kson(
    const fs::path& kson_path,
    std::string* err)
{
    std::string text;
    if (!read_text_file(kson_path, &text, err)) return std::nullopt;

    util::JsonPtr root;
    try {
        root = util::parse(text);
    } catch (const std::exception& ex) {
        if (err) *err = "kson parse failed: " + std::string(ex.what());
        return std::nullopt;
    }
    if (!root || !root->is_object()) {
        if (err) *err = "kson root is not an object";
        return std::nullopt;
    }

    if (auto id = root->get_string("source_id"); id && !id->empty())
        return *id;

    const util::JsonValue* arc = root->get("@arc");
    if (arc && arc->is_object()) {
        const util::JsonValue* compiled_weight = arc->get("compiled_weight");
        if (compiled_weight && compiled_weight->is_object()) {
            if (auto src = compiled_weight->get_string("source"); src && !src->empty())
                return *src;
        }
    }

    if (err) *err = "kson missing source identity";
    return std::nullopt;
}

struct KeyStoreCacheFingerprint {
    uint64_t latest_write_s = 0;
    uint64_t total_bytes = 0;
    uint64_t file_count = 0;
};

static uint64_t file_time_to_epoch_seconds(fs::file_time_type ft) {
    using namespace std::chrono;
    const auto raw = duration_cast<seconds>(ft.time_since_epoch()).count();
    return raw > 0 ? static_cast<uint64_t>(raw) : 0ULL;
}

static void fingerprint_file(const fs::path& path, KeyStoreCacheFingerprint* fp) {
    if (!fp) return;
    std::error_code ec;
    if (!fs::exists(path, ec) || ec) return;
    if (!fs::is_regular_file(path, ec) || ec) return;
    const auto size = fs::file_size(path, ec);
    if (ec) return;
    const auto ts = file_time_to_epoch_seconds(fs::last_write_time(path, ec));
    if (ec) return;
    fp->latest_write_s = std::max(fp->latest_write_s, ts);
    fp->total_bytes += static_cast<uint64_t>(size);
    fp->file_count += 1;
}

static void fingerprint_tree_recursive(const fs::path& root, KeyStoreCacheFingerprint* fp) {
    if (!fp) return;
    std::error_code ec;
    if (!fs::exists(root, ec) || ec) return;
    if (!fs::is_directory(root, ec) || ec) return;

    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
    fs::recursive_directory_iterator end;
    while (it != end) {
        if (!ec) {
            const auto& de = *it;
            std::error_code local_ec;
            if (de.is_regular_file(local_ec) && !local_ec) {
                fingerprint_file(de.path(), fp);
            }
        }
        it.increment(ec);
        if (ec) ec.clear();
    }
}

static KeyStoreCacheFingerprint compute_keystore_cache_fingerprint(const std::string& data_dir) {
    KeyStoreCacheFingerprint fp{};
    const fs::path data_path(data_dir);

    // Top-level JSONL datasets consumed by key-store loaders.
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(data_path, ec)) {
        if (ec) break;
        std::error_code local_ec;
        if (!de.is_regular_file(local_ec) || local_ec) continue;
        if (de.path().extension() == ".jsonl")
            fingerprint_file(de.path(), &fp);
    }

    // Dictionary source under data/gram plus docs/schema/model sources.
    fingerprint_file(data_path / "gram" / "english-dictionary.jsonl", &fp);
    fingerprint_tree_recursive(data_path.parent_path() / "docs", &fp);
    fingerprint_tree_recursive(data_path / "schema", &fp);
    fingerprint_tree_recursive(data_path.parent_path() / "models", &fp);

    return fp;
}

static bool key_store_cache_fingerprint_equal(
    const KeyStoreCacheFingerprint& a,
    const KeyStoreCacheFingerprint& b)
{
    return a.latest_write_s == b.latest_write_s &&
           a.total_bytes == b.total_bytes &&
           a.file_count == b.file_count;
}

static bool load_keystore_cache_meta(const std::string& path, KeyStoreCacheFingerprint* out) {
    if (!out || path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::ostringstream ss;
    ss << in.rdbuf();
    auto root = util::parse(ss.str());
    if (!root || !root->is_object()) return false;

    auto latest = root->get_number("latest_write_s");
    auto bytes  = root->get_number("total_bytes");
    auto files  = root->get_number("file_count");
    if (!latest || !bytes || !files) return false;

    out->latest_write_s = (*latest > 0.0) ? static_cast<uint64_t>(*latest) : 0ULL;
    out->total_bytes = (*bytes > 0.0) ? static_cast<uint64_t>(*bytes) : 0ULL;
    out->file_count = (*files > 0.0) ? static_cast<uint64_t>(*files) : 0ULL;
    return true;
}

static bool save_keystore_cache_meta(
    const std::string& path,
    const KeyStoreCacheFingerprint& fp)
{
    if (path.empty()) return false;
    util::JsonObject root;
    root["version"] = util::json_number(1.0);
    root["latest_write_s"] = util::json_number(static_cast<double>(fp.latest_write_s));
    root["total_bytes"] = util::json_number(static_cast<double>(fp.total_bytes));
    root["file_count"] = util::json_number(static_cast<double>(fp.file_count));

    std::ofstream out(path);
    if (!out) return false;
    out << util::stringify(util::json_object(std::move(root)), true);
    return static_cast<bool>(out);
}

struct TopicConfidenceUpdate {
    std::string topic_key;
    float       before = 0.5f;
    float       after = 0.5f;
    float       delta = 0.0f;
    std::string signal = "neutral";
    std::string event = "NONE";
    bool        committed = false;
    std::string commit_phase = "NONE";
    int         relation_updates = 0;
    float       relation_delta = 0.0f;
};

struct MicronautConfidenceUpdate {
    bool        feedback_event = false;
    std::string micronaut_id;
    std::string event = "NONE";
    std::string signal = "neutral";
    float       before = 0.5f;
    float       after = 0.5f;
    float       delta = 0.0f;
    bool        committed = false;
    std::string commit_phase = "NONE";
    bool        target_resolved = false;
    bool        used_last_active = false;
};

struct FactoryMutationOutcome {
    bool attempted = false;
    bool applied = false;
    bool pending_advisor_verify = false;
    std::string micronaut_id;
    std::string result;
    std::vector<std::string> added_capabilities;
    std::vector<std::string> added_governs;
};

static bool contains_phrase_ci(const std::string& text_lower, const std::string& phrase_lower) {
    return text_lower.find(phrase_lower) != std::string::npos;
}

static bool extract_prefixed_payload_ci(
    const std::string& original_text,
    const std::string& lowered_text,
    const std::vector<std::string>& prefixes_lower,
    std::string* payload_out)
{
    if (!payload_out) return false;
    for (const auto& prefix : prefixes_lower) {
        if (prefix.empty()) continue;
        if (lowered_text.rfind(prefix, 0) != 0) continue;
        std::string raw = trim_copy(original_text.substr(prefix.size()));
        while (!raw.empty() &&
               (raw[0] == ':' || raw[0] == '.' || raw[0] == ',' ||
                raw[0] == '-' || raw[0] == '_' || raw[0] == '|')) {
            raw = trim_copy(raw.substr(1));
        }
        *payload_out = raw;
        return true;
    }
    return false;
}

static std::string normalize_web_research_query(const std::string& text) {
    const auto t = trim_copy(text);
    if (t.empty()) return t;
    const auto low = lower_copy(t);

    std::string payload;
    if (extract_prefixed_payload_ci(t, low, {"search_for", "search for", "search_the_web", "search the web",
                                             "@search", "search:", "search", "search web",
                                             "@research", "research:"},
                                    &payload)) {
        return payload.empty() ? t : payload;
    }

    if (extract_prefixed_payload_ci(t, low, {"google_topic", "google topic", "google"}, &payload)) {
        return payload.empty() ? t : payload;
    }

    if (extract_prefixed_payload_ci(t, low, {"recent_news", "recent news", "latest news", "news"}, &payload)) {
        return payload.empty() ? "recent news" : ("recent news " + payload);
    }

    if (extract_prefixed_payload_ci(t, low, {"business_near_me", "business near me"}, &payload)) {
        return payload.empty() ? "business near me" : (payload + " business near me");
    }

    return t;
}

static std::string infer_micronaut_from_text(const std::string& text) {
    const auto low = lower_copy(text);
    if (low.rfind("hello", 0) == 0 ||
        low.rfind("hi ", 0) == 0 ||
        low == "hi" ||
        low.rfind("hey", 0) == 0 ||
        contains_phrase_ci(low, "hows it going") ||
        contains_phrase_ci(low, "how's it going") ||
        contains_phrase_ci(low, "how are you"))
        return "GREETING-µ";
    if (low.rfind("advise", 0) == 0 ||
        low.rfind("advisor ", 0) == 0 ||
        contains_phrase_ci(low, "@advisor") ||
        contains_phrase_ci(low, "advisor-µ") ||
        contains_phrase_ci(low, "ask my advisor"))
        return "ADVISOR-µ";
    if (contains_phrase_ci(low, "chat-u") ||
        contains_phrase_ci(low, "chat-µ") ||
        contains_phrase_ci(low, "help from lfm") ||
        contains_phrase_ci(low, "ask for help from lfm") ||
        contains_phrase_ci(low, "lfm tool help") ||
        contains_phrase_ci(low, " from lfm") ||
        contains_phrase_ci(low, " lfm "))
        return "CHAT-µ";
    if (low.rfind("tools:", 0) == 0 || contains_phrase_ci(low, "@tools") ||
        low.rfind("status:", 0) == 0 || contains_phrase_ci(low, "@status") ||
        low.rfind("graph:", 0) == 0 || contains_phrase_ci(low, "@graph") ||
        low.rfind("semfold:", 0) == 0 || contains_phrase_ci(low, "@semfold") ||
        low.rfind("cube:", 0) == 0 || contains_phrase_ci(low, "@cube") ||
        low.rfind("world:", 0) == 0 || contains_phrase_ci(low, "@world") ||
        low.rfind("field:", 0) == 0 || contains_phrase_ci(low, "@field") ||
        low.rfind("fieldsvg:", 0) == 0 || contains_phrase_ci(low, "@fieldsvg") ||
        low.rfind("relgraph:", 0) == 0 || contains_phrase_ci(low, "@relgraph") ||
        low.rfind("incidence:", 0) == 0 || contains_phrase_ci(low, "@incidence") ||
        low.rfind("fold:", 0) == 0 || contains_phrase_ci(low, "@fold") ||
        low.rfind("keyquery:", 0) == 0 || contains_phrase_ci(low, "@keyquery") ||
        low.rfind("dispatch:", 0) == 0 || contains_phrase_ci(low, "@dispatch") ||
        low.rfind("manifest:", 0) == 0 || contains_phrase_ci(low, "@manifest_query") ||
        low.rfind("confidence:", 0) == 0 || contains_phrase_ci(low, "@confidence") ||
        low.rfind("replay:", 0) == 0 || contains_phrase_ci(low, "@replay") ||
        low.rfind("sandbox:", 0) == 0 || contains_phrase_ci(low, "@sandbox") ||
        low.rfind("infer:", 0) == 0 || contains_phrase_ci(low, "@infer") ||
        low.rfind("listkeys:", 0) == 0 || contains_phrase_ci(low, "@listkeys") ||
        contains_phrase_ci(low, "tool calls"))
        return "TOOL-µ";
    if (low.rfind("code:", 0) == 0 || contains_phrase_ci(low, "@code") ||
        low.rfind("debug:", 0) == 0 || contains_phrase_ci(low, "@debug"))
        return "coder";
    if (low.rfind("math:", 0) == 0 || contains_phrase_ci(low, "@math") ||
        contains_phrase_ci(low, "equation") || contains_phrase_ci(low, "algebra") ||
        contains_phrase_ci(low, "solve ") || contains_phrase_ci(low, "simplify") ||
        contains_phrase_ci(low, "factor ") || contains_phrase_ci(low, "expand ") ||
        contains_phrase_ci(low, "linear system") || contains_phrase_ci(low, "matrix") ||
        contains_phrase_ci(low, "eigenvalue"))
        return "MATH-µ";
    if (low.rfind("summarize:", 0) == 0 || contains_phrase_ci(low, "@summarize") ||
        low.rfind("explain:", 0) == 0 || contains_phrase_ci(low, "@explain") ||
        low.rfind("meeting:", 0) == 0 || contains_phrase_ci(low, "@meeting"))
        return "reasoner";
    if (low.rfind("rewrite:", 0) == 0 || contains_phrase_ci(low, "@rewrite") ||
        low.rfind("translate:", 0) == 0 || contains_phrase_ci(low, "@translate") ||
        low.rfind("brainstorm:", 0) == 0 || contains_phrase_ci(low, "@brainstorm") ||
        low.rfind("email:", 0) == 0 || contains_phrase_ci(low, "@email"))
        return "CREATE-µ";
    if (low.rfind("plan:", 0) == 0 || contains_phrase_ci(low, "@plan"))
        return "PLAN-µ";
    if (contains_phrase_ci(low, "mx2lm") || contains_phrase_ci(low, "@mx2lm"))
        return "Mx2LM";
    if (contains_phrase_ci(low, "hypercube") || contains_phrase_ci(low, "incidence matrix") ||
        contains_phrase_ci(low, "clustering coefficient") || contains_phrase_ci(low, "distributional semantics") ||
        contains_phrase_ci(low, "semantic folding") || contains_phrase_ci(low, "semantic fingerprint") ||
        contains_phrase_ci(low, "semantic graph") || contains_phrase_ci(low, "thinking pattern") ||
        contains_phrase_ci(low, "brain graph"))
        return "REASON-µ";
    if (contains_phrase_ci(low, "app-wizard") || contains_phrase_ci(low, "app wizard") ||
        contains_phrase_ci(low, "@app") || contains_phrase_ci(low, "app:"))
        return "APP-WIZARD-µ";
    if (contains_phrase_ci(low, "@blog") || contains_phrase_ci(low, "blog:") || contains_phrase_ci(low, "blog"))
        return "BLOG-µ";
    if (contains_phrase_ci(low, "search for")     || contains_phrase_ci(low, "search_for") ||
        contains_phrase_ci(low, "search the web") || contains_phrase_ci(low, "search_the_web") ||
        contains_phrase_ci(low, "google topic")   || contains_phrase_ci(low, "google_topic") ||
        contains_phrase_ci(low, "recent news")    || contains_phrase_ci(low, "recent_news") ||
        contains_phrase_ci(low, "latest news")    || contains_phrase_ci(low, "business near me") ||
        contains_phrase_ci(low, "business_near_me") ||
        contains_phrase_ci(low, "web research")   ||
        contains_phrase_ci(low, "@search")         || low.rfind("search:", 0) == 0 ||
        contains_phrase_ci(low, "@research")       || contains_phrase_ci(low, "find information") ||
        contains_phrase_ci(low, "look up ")        || contains_phrase_ci(low, "lookup ") ||
        contains_phrase_ci(low, "citation")        || contains_phrase_ci(low, "no fact found") ||
        contains_phrase_ci(low, "fact miss")       || contains_phrase_ci(low, "research"))
        return "WEB-RESEARCH-µ";
    if (contains_phrase_ci(low, "librarian")) return "Librarian";
    if (contains_phrase_ci(low, "coordinator")) return "Coordinator";
    if (contains_phrase_ci(low, "orch") || contains_phrase_ci(low, "orchestrator")) return "ORCH-µ";
    if (contains_phrase_ci(low, "reasoner")) return "reasoner";
    return "";
}

static std::string resolve_feedback_micronaut_target(
    const std::string& query,
    const std::string& hint_text,
    const std::string& branch_default,
    const std::string& last_active_micronaut,
    bool* used_last_active = nullptr)
{
    if (used_last_active) *used_last_active = false;

    auto direct = infer_micronaut_from_text(query);
    if (!direct.empty()) return direct;

    if (!branch_default.empty()) return branch_default;

    auto hinted = infer_micronaut_from_text(hint_text);
    if (!hinted.empty()) return hinted;

    if (!last_active_micronaut.empty()) {
        if (used_last_active) *used_last_active = true;
        return last_active_micronaut;
    }
    return "";
}

static MicronautConfidenceUpdate apply_micronaut_feedback(
    memory::ConfidenceStore& confidence,
    const TopicConfidenceUpdate& tc,
    const std::string& target_micronaut,
    bool used_last_active)
{
    MicronautConfidenceUpdate out;
    out.event = tc.event;
    out.signal = tc.signal;
    out.delta = tc.delta;
    out.used_last_active = used_last_active;
    out.feedback_event = (tc.event == "BAD_ANSWER" || tc.event == "GOOD_ANSWER");
    if (!out.feedback_event) return out;

    out.micronaut_id = target_micronaut;
    out.target_resolved = !target_micronaut.empty();
    if (!out.target_resolved) {
        out.commit_phase = tc.committed ? "Xul_unresolved" : "Ch'en_pending_unresolved";
        return out;
    }

    out.before = confidence.get(target_micronaut);
    if (tc.committed && tc.delta != 0.0f) {
        confidence.update(target_micronaut, tc.delta);
        out.committed = true;
        out.commit_phase = "Xul";
    } else {
        out.committed = false;
        out.commit_phase = "Ch'en_pending";
    }
    out.after = confidence.get(target_micronaut);
    return out;
}

static void attach_micronaut_confidence(util::JsonPtr obj, const MicronautConfidenceUpdate& mc) {
    if (!obj || !obj->is_object() || !mc.feedback_event) return;
    auto& root = obj->mut_object();
    util::JsonObject mco;
    mco["event"] = util::json_string(mc.event);
    mco["signal"] = util::json_string(mc.signal);
    mco["delta"] = util::json_number((double)mc.delta);
    mco["target_resolved"] = util::json_bool(mc.target_resolved);
    mco["used_last_active"] = util::json_bool(mc.used_last_active);
    mco["committed"] = util::json_bool(mc.committed);
    mco["commit_phase"] = util::json_string(mc.commit_phase);
    if (mc.target_resolved) {
        mco["micronaut"] = util::json_string(mc.micronaut_id);
        mco["before"] = util::json_number((double)mc.before);
        mco["after"] = util::json_number((double)mc.after);
    } else {
        mco["micronaut"] = util::json_null();
        mco["before"] = util::json_null();
        mco["after"] = util::json_null();
    }
    root["micronaut_confidence"] = util::json_object(std::move(mco));
}

struct AppWizardTemplate {
    std::string id;
    std::string app_symbol;
    std::string layout_symbol;
    std::vector<std::string> blocks;
    std::string skin_symbol;
    std::vector<std::string> modules;
    std::string base_track;
    std::vector<std::string> css_macros;
    std::vector<std::string> aliases;
};

static const std::vector<AppWizardTemplate>& app_wizard_catalog() {
    static const std::vector<AppWizardTemplate> kCatalog = {
        {
            "dashboard",
            "Ω_DASHBOARD_APP",
            "Ω_LAYOUT_DASHBOARD",
            {"Ω_UI_HEADER", "Ω_UI_SIDEBAR"},
            "Ω_SKIN_PROFESSIONAL",
            {"Ω_AUTH_SYSTEM"},
            "CREATE-µ",
            {"⟁HTML", "⟁WIDGETGRID", "⟁WIDGET"},
            {"dashboard", "admin", "analytics", "portal", "control panel"}
        },
        {
            "ecommerce",
            "Ω_ECOMMERCE_STORE",
            "Ω_LAYOUT_ECOMMERCE",
            {"Ω_UI_HEADER", "Ω_UI_FOOTER"},
            "Ω_SKIN_MODERN",
            {"Ω_AUTH_SYSTEM", "Ω_PAYMENT_GATEWAY"},
            "CREATE-µ",
            {"⟁HTML", "⟁WIDGETGRID", "⟁WIDGETCARD"},
            {"ecommerce", "store", "shop", "marketplace", "checkout", "cart"}
        },
        {
            "enterprise",
            "Ω_ERP_SYSTEM",
            "Ω_LAYOUT_ADMIN",
            {"Ω_UI_HEADER", "Ω_UI_SIDEBAR"},
            "Ω_SKIN_PROFESSIONAL",
            {"Ω_AUTH_SYSTEM"},
            "CREATE-µ",
            {"⟁HTMLPAGE", "⟁WIDGETDOCK", "⟁WIDGETCARD"},
            {"enterprise", "erp", "internal", "admin", "ops"}
        },
        {
            "social",
            "Ω_SOCIAL_NETWORK",
            "Ω_LAYOUT_SOCIAL",
            {"Ω_UI_HEADER"},
            "Ω_SKIN_MODERN",
            {"Ω_AUTH_SYSTEM"},
            "CREATE-µ",
            {"⟁HTML", "⟁WIDGETGRID", "⟁WIDGETMODAL"},
            {"social", "community", "feed", "network", "timeline"}
        },
        {
            "blog",
            "Ω_BLOG_PLATFORM",
            "Ω_LAYOUT_BLOG",
            {"Ω_UI_HEADER", "Ω_UI_FOOTER", "Ω_POST_EDITOR", "Ω_COMMENT_FEED"},
            "Ω_SKIN_MODERN",
            {"Ω_AUTH_SYSTEM", "Ω_CONTENT_INDEX"},
            "BLOG-µ",
            {"⟁BLOG", "⟁BLOGTOOLBAR", "⟁BLOGMODAL"},
            {"blog", "article", "post", "journal", "newsletter", "publishing"}
        }
    };
    return kCatalog;
}

static const AppWizardTemplate* app_wizard_find_template(const std::string& id) {
    const auto& cat = app_wizard_catalog();
    for (const auto& t : cat) {
        if (t.id == id) return &t;
    }
    return nullptr;
}

static bool has_token(const std::vector<std::string>& toks, const std::string& value) {
    return std::find(toks.begin(), toks.end(), value) != toks.end();
}

static util::JsonArray json_string_array(const std::vector<std::string>& values) {
    util::JsonArray out;
    out.reserve(values.size());
    for (const auto& v : values) out.push_back(util::json_string(v));
    return out;
}

static util::JsonArray app_wizard_options_array() {
    util::JsonArray out;
    for (const auto& t : app_wizard_catalog()) {
        util::JsonObject row;
        row["type"] = util::json_string(t.id);
        row["app_symbol"] = util::json_string(t.app_symbol);
        row["layout"] = util::json_string(t.layout_symbol);
        row["base_track"] = util::json_string(t.base_track);
        row["css_macro"] = util::json_string(t.css_macros.empty() ? "⟁HTML" : t.css_macros.front());
        out.push_back(util::json_object(std::move(row)));
    }
    return out;
}

static std::string to_title_case(std::string text) {
    std::string out;
    out.reserve(text.size());
    bool cap = true;
    for (unsigned char ch : text) {
        if (ch == '_' || ch == '-') {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
            cap = true;
            continue;
        }
        if (std::isspace(ch)) {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
            cap = true;
            continue;
        }
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(cap ? std::toupper(ch) : std::tolower(ch)));
            cap = false;
            continue;
        }
        out.push_back(static_cast<char>(ch));
        cap = true;
    }
    return trim_copy(out);
}

static std::string join_words(const std::vector<std::string>& words, size_t max_words = 0) {
    if (words.empty()) return "";
    const size_t limit = (max_words == 0) ? words.size() : std::min(max_words, words.size());
    std::string out;
    for (size_t i = 0; i < limit; ++i) {
        if (!out.empty()) out.push_back(' ');
        out += words[i];
    }
    return out;
}

static std::string symbol_to_label(const std::string& symbol) {
    std::string raw;
    raw.reserve(symbol.size());
    bool pending_space = false;
    for (unsigned char ch : symbol) {
        if (std::isalnum(ch)) {
            if (pending_space && !raw.empty()) raw.push_back(' ');
            raw.push_back(static_cast<char>(std::tolower(ch)));
            pending_space = false;
            continue;
        }
        if ((ch == '_' || ch == '-' || std::isspace(ch)) && !raw.empty())
            pending_space = true;
    }
    auto cooked = to_title_case(trim_copy(raw));
    return cooked.empty() ? symbol : cooked;
}

static std::vector<std::string> semantic_keywords_from_tokens(const std::vector<std::string>& toks) {
    static const std::unordered_set<std::string> kStop = {
        "app","application","website","platform","template","blueprint","catalog",
        "build","make","create","show","pick","choose","option","options","type","types",
        "the","a","an","to","for","of","in","on","with","and","or","is","are","be",
        "please","want","need","runtime","system","user","from","using","based"
    };

    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (const auto& tok : toks) {
        if (tok.size() < 3) continue;
        if (kStop.count(tok)) continue;
        if (!seen.insert(tok).second) continue;
        out.push_back(tok);
        if (out.size() >= 8) break;
    }
    return out;
}

static bool is_gemma_gguf_path(const std::string& model_path) {
    if (model_path.empty()) return false;
    const auto low = lower_copy(model_path);
    return low.find("gemma") != std::string::npos &&
           low.find(".gguf") != std::string::npos;
}

static std::vector<std::string> build_learning_ngrams(const std::string& text, size_t max_items = 24) {
    std::vector<std::string> out;
    if (text.empty() || max_items == 0) return out;

    const auto toks = micronaut::Mx2LmNGram::tokenize(lower_copy(text));
    if (toks.empty()) return out;

    const auto taxonomy = micronaut::Mx2LmNGram::classify_gram_taxonomy(toks);
    std::unordered_set<std::string> seen;
    auto append_ranked = [&](const micronaut::NGramFreq& freq) {
        for (const auto& [gram, _] : micronaut::Mx2LmNGram::top(freq, max_items)) {
            if (!seen.insert(gram).second) continue;
            out.push_back(gram);
            if (out.size() >= max_items) return;
        }
    };

    append_ranked(taxonomy.bigrams);
    if (out.size() < max_items) append_ranked(taxonomy.trigrams);
    if (out.size() < max_items) append_ranked(taxonomy.unigrams);
    if (out.size() < max_items) append_ranked(taxonomy.quadgrams);
    return out;
}

struct PromptGramChannelCounts {
    size_t unigrams = 0;
    size_t bigrams = 0;
    size_t trigrams = 0;
    size_t quadgrams = 0;
    size_t coarse_grams = 0;
    size_t function_grams = 0;
    size_t tool_grams = 0;
    size_t control_grams = 0;
};

static std::vector<std::pair<std::string, uint32_t>> top_ranked_grams(
    const micronaut::NGramFreq& freq,
    size_t limit)
{
    const auto cap = (limit == 0) ? freq.size() : limit;
    return micronaut::Mx2LmNGram::top(freq, cap);
}

static std::vector<std::string> gram_terms_from_top(
    const micronaut::NGramFreq& freq,
    size_t limit)
{
    std::vector<std::string> out;
    for (const auto& [gram, _] : top_ranked_grams(freq, limit))
        out.push_back(gram);
    return out;
}

static PromptGramChannelCounts record_prompt_gram_channels(
    const std::string& scope_key,
    const micronaut::Mx2LmGramTaxonomy& taxonomy,
    micronaut::Factory* factory,
    kuhul::GramStore* gram_store,
    const std::vector<std::string>& tokens,
    size_t cap_per_class = 24)
{
    PromptGramChannelCounts counts;
    auto record = [&](const std::string& gram_class,
                      const micronaut::NGramFreq& freq,
                      const char* phase,
                      const char* unit,
                      const char* corpus,
                      size_t* count_out) {
        if (count_out) *count_out = freq.size();
        const auto top_rows = top_ranked_grams(freq, cap_per_class);
        for (size_t i = 0; i < top_rows.size(); ++i) {
            const auto& gram = top_rows[i].first;
            const auto count = top_rows[i].second;
            const std::string ordinal = std::to_string(i + 1);

            if (factory) {
                const std::string prefix = "gram:" + gram_class + ":";
                factory->record_key(prefix + gram, scope_key);
                factory->record_key("assoc:" + gram_class + ":" + gram, scope_key);
                if (gram_class == "tool_gram") {
                    factory->record_key("route_hint:tool_gram:" + gram, "TOOL-µ");
                    factory->record_key("route_hint:track:" + gram, "TOOL-µ");
                }
            }

            if (gram_store) {
                gram_store->gramWrite(
                    scope_key + ":" + gram_class + ":" + ordinal,
                    gram_class + "|" + std::to_string(count) + "|" + gram,
                    phase,
                    unit,
                    corpus);
            }
        }
    };

    record("unigram", taxonomy.unigrams,
           "Pop", "semantic_address_layer", "definitions_and_relations", &counts.unigrams);
    record("bigram", taxonomy.bigrams,
           "Wo", "context_association_evidence", "definitions_and_relations", &counts.bigrams);
    record("trigram", taxonomy.trigrams,
           "Wo", "context_association_evidence", "definitions_and_relations", &counts.trigrams);
    record("quadgram", taxonomy.quadgrams,
           "Wo", "context_association_evidence", "definitions_and_relations", &counts.quadgrams);
    record("coarse_gram", taxonomy.coarse_grams,
           "Yax", "coarse_ngram_structure", "definitions_and_relations", &counts.coarse_grams);
    record("function_gram", taxonomy.function_grams,
           "Yax", "function_word_channel", "definitions_and_relations", &counts.function_grams);
    record("tool_gram", taxonomy.tool_grams,
           "Sek", "tool_dispatch_channel", "definitions_and_relations", &counts.tool_grams);

    // Control flow: the key grammar's control words are archived as a
    // phase-bound flow (sugar -> plan -> XCFE lowering), not merely as operator
    // tokens, so a prompt's control structure is addressable and replayable.
    const auto control_plan = micronaut::control::plan_control_flow(tokens);
    counts.control_grams = control_plan.steps.size();
    if (!control_plan.steps.empty()) {
        if (factory) {
            factory->record_key("control:form:" + control_plan.form, scope_key);
            if (!control_plan.condition.empty())
                factory->record_key("control:condition:" + control_plan.condition, scope_key);
            for (const auto& step : control_plan.steps) {
                factory->record_key("control:" + step.word, scope_key);
                factory->record_key("control:" + step.word + ":phase:" + step.phase, scope_key);
                factory->record_key("control:xcfe:" + step.xcfe_op, scope_key);
                if (!step.operand.empty())
                    factory->record_key(
                        "control:operand:" + step.word + ":" + step.operand, scope_key);
            }
        }
        if (gram_store) {
            for (size_t i = 0; i < control_plan.steps.size(); ++i) {
                const auto& step = control_plan.steps[i];
                gram_store->gramWrite(
                    scope_key + ":control_gram:" + std::to_string(i + 1),
                    "control_gram|" + step.xcfe_op + "|" + step.word + "|" + step.operand,
                    step.phase,
                    "control_flow_operator",
                    "key_grammar_control_flow");
            }
        }
    }
    return counts;
}

struct DictionaryLexicon {
    std::unordered_map<std::string, std::string> definition_by_word;
    std::unordered_map<std::string, std::vector<std::string>> similarity_by_word;
};

static bool is_ascii_alpha_word(const std::string& token) {
    if (token.empty()) return false;
    for (unsigned char c : token) {
        if (!std::isalpha(c)) return false;
    }
    return true;
}

static std::string best_dictionary_word_token(const std::string& raw_word) {
    const auto toks = micronaut::Mx2LmNGram::tokenize(lower_copy(raw_word));
    std::string best;
    for (const auto& tok : toks) {
        if (!is_ascii_alpha_word(tok)) continue;
        if (tok.size() > best.size()) best = tok;
    }
    if (best.size() < 2 || best.size() > 32) return {};
    return best;
}

static std::string join_with_delim(
    const std::vector<std::string>& values,
    const std::string& delim,
    size_t max_items = 0)
{
    if (values.empty()) return {};
    const size_t limit = (max_items == 0) ? values.size() : std::min(values.size(), max_items);
    std::string out;
    for (size_t i = 0; i < limit; ++i) {
        if (!out.empty()) out += delim;
        out += values[i];
    }
    return out;
}

static std::vector<std::string> dictionary_similarity_candidates(
    const std::string& head_word,
    const std::string& definition,
    const std::unordered_set<std::string>& known_words,
    size_t max_items = 8)
{
    if (head_word.empty() || definition.empty() || max_items == 0) return {};

    std::string clipped = definition;
    if (clipped.size() > 240) clipped.resize(240);

    static const std::unordered_set<std::string> kStop = {
        "the","a","an","and","or","but","to","of","in","on","at","for","from","with","by","as",
        "is","are","be","being","been","this","that","these","those","it","its","their","there",
        "into","out","over","under","up","down","one","two","three","who","which","what","when",
        "where","whose","whom","his","her","hers","him","them","they","you","your","yours",
        "our","ours","we","i","me","my","mine","he","she","state","act","thing","used","called",
        "person","another","part","kind","sort","manner","quality","power","degree","make","made",
        "give","given","take","taken","put","set","use","used","very","more","most","much","many",
        "same","such","than","then","also","only","not","no"
    };

    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    const auto toks = micronaut::Mx2LmNGram::tokenize(lower_copy(clipped));
    for (const auto& tok : toks) {
        if (tok.size() < 3 || tok.size() > 24) continue;
        if (!is_ascii_alpha_word(tok)) continue;
        if (tok == head_word) continue;
        if (kStop.count(tok)) continue;
        if (known_words.find(tok) == known_words.end()) continue;
        if (!seen.insert(tok).second) continue;
        out.push_back(tok);
        if (out.size() >= max_items) break;
    }
    return out;
}

static DictionaryLexicon load_dictionary_lexicon_from_jsonl(const std::string& dict_path) {
    DictionaryLexicon lex;
    std::ifstream dict_f(dict_path);
    if (!dict_f.is_open()) return lex;

    std::string line;
    while (std::getline(dict_f, line)) {
        if (line.empty()) continue;
        auto obj = util::parse(line);
        if (!obj) continue;

        auto word = obj->get_string("word");
        auto defn = obj->get_string("definition");
        if (!word || !defn || word->empty() || defn->empty()) continue;

        const auto normalized = best_dictionary_word_token(*word);
        if (normalized.empty()) continue;

        auto hit = lex.definition_by_word.find(normalized);
        if (hit == lex.definition_by_word.end() || defn->size() < hit->second.size())
            lex.definition_by_word[normalized] = *defn;
    }

    std::unordered_set<std::string> known_words;
    known_words.reserve(lex.definition_by_word.size());
    for (const auto& [word, _] : lex.definition_by_word) {
        known_words.insert(word);
    }

    for (const auto& [word, defn] : lex.definition_by_word) {
        auto sims = dictionary_similarity_candidates(word, defn, known_words, 8);
        if (!sims.empty())
            lex.similarity_by_word[word] = std::move(sims);
    }

    return lex;
}

static const DictionaryLexicon& dictionary_lexicon_cache(const std::string& data_dir) {
    static std::mutex cache_mu;
    static std::string cache_path;
    static DictionaryLexicon cache;

    const auto dict_path =
        (fs::path(data_dir) / "gram" / "english-dictionary.jsonl").lexically_normal().string();
    std::lock_guard<std::mutex> lk(cache_mu);
    if (cache_path != dict_path) {
        cache = load_dictionary_lexicon_from_jsonl(dict_path);
        cache_path = dict_path;
    }
    return cache;
}

static std::vector<std::string> dictionary_synonyms_for_word(
    const DictionaryLexicon* lexicon,
    const std::string& token,
    size_t max_items = 6)
{
    if (!lexicon || token.empty() || max_items == 0) return {};
    const auto normalized = best_dictionary_word_token(token);
    if (normalized.empty()) return {};
    auto it = lexicon->similarity_by_word.find(normalized);
    if (it == lexicon->similarity_by_word.end() || it->second.empty()) return {};
    std::vector<std::string> out = it->second;
    if (out.size() > max_items) out.resize(max_items);
    return out;
}

static std::string resolve_lexicon_lemma(
    const std::string& token,
    const DictionaryLexicon* lexicon)
{
    auto normalized = best_dictionary_word_token(token);
    if (normalized.empty()) return {};
    if (!lexicon) return normalized;

    auto has_word = [&](const std::string& candidate) {
        return lexicon->definition_by_word.find(candidate) != lexicon->definition_by_word.end();
    };
    if (has_word(normalized)) return normalized;

    if (normalized.size() > 3 &&
        normalized.compare(normalized.size() - 3, 3, "ies") == 0) {
        auto candidate = normalized.substr(0, normalized.size() - 3) + "y";
        if (has_word(candidate)) return candidate;
    }
    if (normalized.size() > 3 &&
        normalized.compare(normalized.size() - 2, 2, "es") == 0) {
        auto candidate = normalized.substr(0, normalized.size() - 2);
        if (has_word(candidate)) return candidate;
    }
    if (normalized.size() > 2 && normalized.back() == 's') {
        auto candidate = normalized.substr(0, normalized.size() - 1);
        if (has_word(candidate)) return candidate;
    }
    return normalized;
}

static std::string lexicon_key_for_word(const std::string& raw_word) {
    const auto lemma = best_dictionary_word_token(raw_word);
    if (lemma.empty()) return {};

    std::string out;
    out.reserve(lemma.size());
    for (unsigned char c : lemma) {
        if (!std::isalnum(c)) continue;
        out.push_back(static_cast<char>(std::toupper(c)));
    }
    return out;
}

static std::string lexicon_class_for_word(
    const std::string& lemma_word,
    const std::string& definition)
{
    if (lemma_word.empty()) return "ENTITY";

    static const std::unordered_set<std::string> kColorValues = {
        "black", "white", "red", "green", "blue", "yellow", "orange", "purple",
        "brown", "gray", "grey", "pink", "teal", "cyan", "magenta", "indigo", "violet"
    };
    static const std::unordered_set<std::string> kProperties = {
        "color", "colour", "size", "shape", "length", "height", "width", "weight",
        "age", "value", "name", "kind", "type", "material", "texture", "temperature"
    };
    static const std::unordered_set<std::string> kAttributeEntities = {
        "fur", "hair", "skin", "coat", "tail", "wing", "paw", "eye", "ear",
        "feather", "horn", "fin", "leaf", "branch", "root"
    };

    if (kColorValues.count(lemma_word)) return "COLOR_VALUE";
    if (kProperties.count(lemma_word)) return "PROPERTY";
    if (kAttributeEntities.count(lemma_word)) return "ATTRIBUTE_ENTITY";

    const auto def_low = lower_copy(definition);
    if (!def_low.empty()) {
        const bool mentions_color =
            contains_phrase_ci(def_low, "color") ||
            contains_phrase_ci(def_low, "colour") ||
            contains_phrase_ci(def_low, "hue");
        if (mentions_color) {
            if (contains_phrase_ci(def_low, "property") ||
                contains_phrase_ci(def_low, "quality")) return "PROPERTY";
            return "COLOR_VALUE";
        }
        if (contains_phrase_ci(def_low, "part of") ||
            contains_phrase_ci(def_low, "covering of")) return "ATTRIBUTE_ENTITY";
    }

    return "ENTITY";
}

static std::string lexicon_book_for_key(const std::string& key) {
    if (key.empty()) return {};
    return "Book(" + key + ")";
}

static util::JsonPtr make_lexicon_entry_payload(
    const std::string& token,
    const DictionaryLexicon* lexicon)
{
    const auto lemma = resolve_lexicon_lemma(token, lexicon);
    if (lemma.empty()) return util::json_null();

    std::string definition;
    if (lexicon) {
        if (auto it = lexicon->definition_by_word.find(lemma);
            it != lexicon->definition_by_word.end()) {
            definition = it->second;
        }
    }
    const auto class_name = lexicon_class_for_word(lemma, definition);
    const auto key = lexicon_key_for_word(lemma);
    const auto book = lexicon_book_for_key(key);
    const auto synonyms = dictionary_synonyms_for_word(lexicon, lemma, 8);

    util::JsonObject obj;
    obj["token"] = util::json_string(best_dictionary_word_token(token));
    obj["lemma"] = util::json_string(lemma);
    obj["@key"] = util::json_string(key);
    obj["@unit"] = util::json_string("unigram");
    obj["@class"] = util::json_string(class_name);
    obj["@book"] = util::json_string(book);
    obj["lexicon_source"] = util::json_string("data/gram/english-dictionary.jsonl");
    if (!definition.empty())
        obj["definition"] = util::json_string(definition);
    if (!synonyms.empty()) {
        util::JsonArray arr;
        for (const auto& s : synonyms) arr.push_back(util::json_string(s));
        obj["synonyms"] = util::json_array(std::move(arr));
    }
    return util::json_object(std::move(obj));
}

static size_t record_lexicon_unigram_keys(
    const std::vector<std::string>& tokens_lower,
    const std::string& scope_key,
    const DictionaryLexicon* lexicon,
    micronaut::Factory* factory,
    kuhul::GramStore* gram_store)
{
    if (!factory) return 0;

    size_t rows = 0;
    size_t words_scanned = 0;
    std::unordered_set<std::string> seen;
    for (const auto& tok : tokens_lower) {
        if (tok.size() < 2) continue;
        const auto lemma = resolve_lexicon_lemma(tok, lexicon);
        if (lemma.empty()) continue;
        if (!seen.insert(lemma).second) continue;
        if (++words_scanned > 64) break;

        auto lex_payload = make_lexicon_entry_payload(tok, lexicon);
        if (!lex_payload || lex_payload->is_null()) continue;

        const auto payload_text = util::stringify(lex_payload);
        const auto key = lex_payload->get_string("@key").value_or(lexicon_key_for_word(lemma));
        const auto book = lex_payload->get_string("@book").value_or(lexicon_book_for_key(key));
        if (key.empty() || book.empty()) continue;

        factory->record_key("lexicon:" + lemma, payload_text);
        factory->record_key("unigram:" + lemma, payload_text);
        factory->record_key("key:" + lemma, key);
        factory->record_key("book:" + lemma, book);
        factory->record_key("book_key:" + key, book);

        if (gram_store) {
            gram_store->gramWrite(
                scope_key + ":lexicon:" + lemma,
                payload_text,
                "Wo",
                "lexicon_unigram",
                "definitions_and_relations");
        }
        ++rows;
    }
    return rows;
}

static size_t record_dictionary_similarity_keys(
    const std::vector<std::string>& tokens_lower,
    const std::string& scope_key,
    const DictionaryLexicon* lexicon,
    micronaut::Factory* factory,
    kuhul::GramStore* gram_store)
{
    if (!lexicon || !factory) return 0;

    size_t rows = 0;
    size_t words_scanned = 0;
    std::unordered_set<std::string> seen;
    for (const auto& tok : tokens_lower) {
        if (tok.size() < 3) continue;
        if (!seen.insert(tok).second) continue;
        if (++words_scanned > 48) break;

        const auto synonyms = dictionary_synonyms_for_word(lexicon, tok, 6);
        if (synonyms.empty()) continue;

        const auto packed = join_with_delim(synonyms, " | ", 6);
        if (packed.empty()) continue;

        factory->record_key("word_similarity:" + tok, packed);
        factory->record_key("synonym:" + tok, packed);
        if (gram_store) {
            gram_store->gramWrite(
                scope_key + ":word_similarity:" + tok,
                packed,
                "Yax",
                "similarity_kernel",
                "definitions_and_relations");
        }
        ++rows;

        const size_t rel_cap = std::min<size_t>(4, synonyms.size());
        for (size_t i = 0; i < rel_cap; ++i) {
            factory->record_key(
                "assoc:semantic:" + tok + "|@related_to|" + synonyms[i],
                scope_key);
            ++rows;
        }
    }
    return rows;
}

struct RuntimeGrammarSpecSummary {
    std::string aiml_ebnf_file;
    std::string scfml_ebnf_file;
    std::string key_grammar_ebnf_file;
};

static bool write_runtime_grammar_specs(
    const fs::path& grammar_dir,
    RuntimeGrammarSpecSummary* out,
    std::string* err)
{
    if (out) *out = {};
    const fs::path aiml_path = grammar_dir / "aiml.runtime-grammar.ebnf";
    const fs::path scfml_path = grammar_dir / "scfml.runtime-grammar.ebnf";
    const fs::path key_json_path = grammar_dir / "key-grammar-json.runtime.ebnf";

    static const char* kAimlEbnf = R"EBNF((* KHANARY AIML Runtime Grammar v4 *)
(* Branch/control markup is delegated to SCFML. See scfml.runtime-grammar.ebnf. *)
aiml_document = "<aiml>", { category }, "</aiml>" ;
category = "<category>", ( aiml_pair | scfml_category ), "</category>" ;
aiml_pair = pattern, [ that ], [ topic ], template ;
scfml_category = scfml_surface_ref ;

pattern = "<pattern>", { pattern_item }, "</pattern>" ;
pattern_item = pattern_text | scfml_surface_ref ;
that = "<that>", pattern_text, "</that>" ;
topic = "<topic>", pattern_text, "</topic>" ;
template = "<template>", { template_item }, "</template>" ;

template_item =
    text | srai | get_var | set_var | star | thatstar | tool_call | graph_call
  | mx2lm_call | scfml_surface_ref ;
srai = "<srai>", pattern_text, "</srai>" ;
get_var = "<get name=\"", ident, "\"/>" ;
set_var = "<set name=\"", ident, "\">", text, "</set>" ;
star = "<star", [ " index=\"", integer, "\"" ], "/>" ;
thatstar = "<thatstar", [ " index=\"", integer, "\"" ], "/>" ;

scfml_surface_ref = ? SCFML semantic control-flow surface; normalized by scfml.runtime-grammar.ebnf before Pattern Learning or XCFE projection ? ;

tool_call = "<tool name=\"", ident, "\">", tool_payload, "</tool>" ;
graph_call = "<edges .../>" | "<getedge .../>" | "<traverse .../>" | "<node .../>" ;
mx2lm_call = "<mx2lm>", mx2lm_payload, "</mx2lm>" ;

mx2lm_payload = { token }, { gram_binding } ;
gram_binding =
    "<gram channel=\"", gram_class, "\" role=\"", gram_role, "\">",
    gram_text,
    "</gram>" ;

gram_class =
    "unigram" | "bigram" | "trigram" | "quadgram" |
    "coarse_gram" | "function_gram" | "tool_gram" ;
gram_role =
    "semantic_address" |
    "context_association" |
    "structural_shape" |
    "function_word_structure" |
    "control_binding" |
    "operator_binding" |
    "tool_dispatch_intent" ;

token = ident ;
ident = ? ASCII letter/digit/underscore/hyphen/apostrophe sequence ? ;
integer = digit, { digit } ;
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
text = { text_char } ;
pattern_text = { text_char } ;
tool_payload = { text_char } ;
gram_text = { text_char } ;
text_char = ? any UTF-8 char except unmatched '<' or '>' ? ;)EBNF";

    static const char* kScfmlEbnf = R"EBNF((* KHANARY SCFML Runtime Grammar v1 *)
(* SCFML = Semantic Control Flow Machine Learning.
   It owns semantic control-flow surfaces and Pattern Learning admission.
   SCFML is above XCFE: it declares branches/pattern candidates, resolves them
   through epistemic state, then projects legal moves to XCFE. *)

scfml_document = { scfml_unit } ;
scfml_unit = scfml_branch_row | scfml_pattern_learning_row | scfml_xml_surface ;

(* ── XML/AIML-compatible surface ─────────────────────────────────────── *)
scfml_xml_surface = branch_element | detached_branch_sequence ;
branch_element = "<branch", [ branch_attrs ], ">", { branch_part }, "</branch>" ;
detached_branch_sequence = branch_part, { branch_part } ;
branch_part = if_block | then_block | else_block | unknown_block ;
if_block = "<if>", { branch_payload }, "</if>" ;
then_block = "<then>", { branch_payload }, "</then>" ;
else_block = "<else>", { branch_payload }, "</else>" ;
unknown_block = "<unknown>", { branch_payload }, "</unknown>" ;
branch_payload = aiml_pattern | aiml_template | scfml_xml_surface | branch_text ;
aiml_pattern = "<pattern>", pattern_text, "</pattern>" ;
aiml_template = "<template>", template_text, "</template>" ;
branch_attrs = ? XML attributes; optional schema/key/track metadata ? ;
branch_text = ? UTF-8 branch text excluding unmatched SCFML branch tags ? ;

(* Surface normalization. Input children may arrive in any order; normalized
   rows use canonical_order = if, then, else, unknown. *)
normalization_rule =
    "if/pattern=>condition"
  | "then/template=>then_action"
  | "else/template=>else_action"
  | "unknown=>unknown_handler"
  | "then/pattern=>then_pattern_candidate"
  | "else/pattern=>else_pattern_candidate"
  | "else/if/then=>nested_else_if"
  | "pattern_with_branch_markup=>structural_after_normalization" ;

(* ── JSON row emitted into key-grammar / gram-store surfaces ─────────── *)
scfml_branch_row =
    "{",
      q_schema, ",", q_key, ",", q_condition, ",", q_then, ",", q_else,
      [ ",", q_track ],
      [ ",", q_epistemic_state ],
      [ ",", q_unknown ],
      [ ",", q_source_surface ],
      [ ",", q_pattern_learning ],
    "}" ;

q_schema = "\"schema\":", ( "\"scfml.branch.v1\"" | "\"key.grammar.semantic_branch.v1\"" ) ;
q_key = "\"key\":", scfml_branch_key ;
q_condition = "\"condition\":", string ;
q_then = "\"then\":", scfml_branch_target ;
q_else = "\"else\":", scfml_branch_target ;
q_track = "\"track\":", string ;
q_epistemic_state = "\"epistemic_state\":", "\"", epistemic_state, "\"" ;
q_unknown = "\"unknown\":", scfml_branch_target ;
q_source_surface = "\"source_surface\":", ( "\"aiml\"" | "\"json\"" | "\"memory_event\"" ) ;
q_pattern_learning = "\"pattern_learning\":", boolean ;
scfml_branch_target = string | "null" ;
scfml_branch_key = "scfml:branch:", epoch_ms, "-", sequence
                 | "scfml:branch:", gram_text
                 | "semantic_branch:", epoch_ms, "-", sequence
                 | "semantic_branch:", gram_text ;

(* ── Pattern Learning rows ───────────────────────────────────────────── *)
scfml_pattern_learning_row =
    "{",
      "\"schema\":", "\"scfml.pattern_learning.v1\"", ",",
      "\"source_event\":", string, ",",
      "\"candidate_kind\":", pattern_candidate_kind, ",",
      "\"requires\":", "\"SUPPORTED_or_user_verified\"",
      [ ",", "\"target\":", string ],
    "}" ;
pattern_candidate_kind =
    "\"aiml_pattern\"" | "\"aiml_template\"" | "\"semantic_branch\""
  | "\"key_grammar\"" | "\"route_pattern\"" ;

(* ── Epistemic branch laws ───────────────────────────────────────────── *)
epistemic_state = "SUPPORTED" | "MISS" | "UNKNOWN" | "CONTRADICTED" ;
scfml_law =
    "SUPPORTED(P)=>then"
  | "CONTRADICTED(P)=>else_as_complement"
  | "MISS(P)!=CONTRADICTED(P)"
  | "UNKNOWN(P)!=else"
  | "else!=not"
  | "contradicting_one_candidate_does_not_prove_another" ;

(* ── Projection boundary ─────────────────────────────────────────────── *)
scfml_pipeline =
    "SCFML_surface", "->", "canonical_branch_row", "->",
    "epistemic_resolution", "->", "XCFE_move_projection" ;
xcfe_projection =
    "guard" | "select_then" | "select_else" | "spawn_unknown" | "preserve"
  | "retrieve" | "defer" | "merge" | "commit" | "rollback" ;

boolean = "true" | "false" ;
string = ? JSON string ? ;
pattern_text = ? AIML pattern text ? ;
template_text = ? AIML template text ? ;
gram_text = ? ordered gram text; token order is semantic ? ;
epoch_ms = ? milliseconds since UNIX epoch ? ;
sequence = ? per-process sequence number ? ;)EBNF";

    static const char* kKeyGrammarJsonEbnf = R"EBNF((* KHANARY key-grammar JSON Runtime EBNF v3 *)
(* Typed gram channels are structured, phase-bound, provenance-carrying
   records — never a bag of words. Every channel declares ordered width,
   first-occurrence position, a semantic role, a K'UHUL phase binding,
   a gram-store unit, a key-address form, and a consumption rule. *)

document = lexicon_row | relation_row | greeting_bank_row | greeting_learning_row
         | control_flow_row | control_plan_row | scfml_branch_row | semantic_branch_row
         | memory_event_row | hyperbolic_event_row ;

(* ── Row forms ───────────────────────────────────────────────────────── *)
lexicon_row =
    "{",
      q_word, ",", q_key, ",", q_unit, ",", q_class, ",", q_book, ",",
      q_definition, ",", q_lexicon_source,
      [ ",", q_synonyms ],
      [ ",", q_definition_ngrams ],
      [ ",", q_definition_keys ],
      [ ",", q_gram_channels ],
      [ ",", q_hyperbolic_event ],
    "}" ;

q_word           = "\"word\":", string ;
q_key            = "\"key\":", string ;
q_unit           = "\"unit\":", "\"unigram\"" ;
q_class          = "\"class\":", string ;
q_book           = "\"book\":", string ;
q_definition     = "\"definition\":", string ;
q_lexicon_source = "\"lexicon_source\":", string ;
q_synonyms            = "\"synonyms\":", string_array ;
q_definition_ngrams   = "\"definition_ngrams\":", string_array ;
q_definition_keys     = "\"definition_keys\":", string_array ;
q_gram_channels       = "\"gram_channels\":", "{", gram_channel, { ",", gram_channel }, "}" ;
q_hyperbolic_event    = "\"hyperbolic_event\":", hyperbolic_event_object ;

(* Hyperbolic key-grammar event projection: the key addresses a future-directed
   candidate continuation from a current origin. Rapidity/direction/interval are
   routing geometry only; semantic_authority must remain false.

   Preferred v1 shape uses nested hyperbolic_metrics + bounds. Legacy flat
   rapidity/direction/interval fields are still accepted for existing rows. *)
hyperbolic_event_row =
    "{",
      "\"$schema\":", "\"urn:kuhul:khanary:schema:key-grammar-event:v1\"", ",",
      "\"generation_id\":", integer, ",",
      "\"topology_target\":", "\"B\"", ",",
      "\"space_signature\":", "\"Minkowski_Minus_Plus_Plus_Plus\"", ",",
      "\"event_record\":", hyperbolic_event_object,
    "}" ;

hyperbolic_event_object =
    "{",
      q_he_type, ",", q_he_origin,
      [ ",", q_he_track ],
      [ ",", q_he_phase ],
      [ ",", q_he_key ],
      [ ",", q_he_gram ],
      [ ",", q_he_event_role ],
      [ ",", q_he_hyperbolic_metrics ],
      [ ",", q_he_bounds ],
      (* Legacy flat event metric fields. *)
      [ ",", q_he_rapidity ],
      [ ",", q_he_direction ],
      [ ",", q_he_minkowski_interval ],
      [ ",", q_he_interval_class ],
      [ ",", q_he_epistemic_state ],
      ",", q_he_semantic_authority,
    "}" ;

q_he_type = "\"@type\":", "\"key_grammar.hyperbolic_event.v1\"" ;
q_he_origin = "\"origin\":", string ;
q_he_track = "\"track\":", string ;
q_he_phase = "\"phase\":", "\"", phase_name, "\"" ;
q_he_key = "\"key\":", string ;
q_he_gram = "\"gram\":", string ;
q_he_event_role = "\"event_role\":", string ;
q_he_rapidity = "\"rapidity\":", number ;
q_he_direction = "\"direction\":", number_array ;
q_he_minkowski_interval = "\"minkowski_interval\":", number ;
q_he_interval_class = "\"interval_class\":", "\"", interval_class, "\"" ;
q_he_epistemic_state = "\"epistemic_state\":", "\"", epistemic_state, "\"" ;
q_he_semantic_authority = "\"semantic_authority\":", "false" ;
q_he_hyperbolic_metrics = "\"hyperbolic_metrics\":", "{",
      "\"rapidity_rho\":", number, ",",
      "\"direction_vector\":", number_array, ",",
      "\"minkowski_interval\":", number,
    "}" ;
q_he_bounds = "\"bounds\":", "{",
      "\"interval_class\":", "\"", interval_class, "\"", ",",
      "\"epistemic_state\":", "\"", epistemic_state, "\"", ",",
      "\"semantic_authority\":", "false",
    "}" ;

interval_class =
    "timelike_candidate" | "spacelike_candidate" | "lightlike_boundary" | "unclassified" ;

gram_channel = "\"", gram_class, "\":", "[", [ typed_gram, { ",", typed_gram } ], "]" ;

(* A typed gram: ordered gram text, frequency, ordered slot width, semantic
   role, first-occurrence position (order evidence), K'UHUL phase binding,
   and provenance of the evidence that produced it. Width counts ordered
   slots: for coarse_gram each structural symbol (c, f:x, ctrl:x, t:x) is
   ONE slot; position is omitted for coarse_gram because structural
   skeletons carry shape, not lexical position. *)
typed_gram =
    "{",
      "\"gram\":", string, ",",
      "\"count\":", number, ",",
      "\"width\":", number, ",",
      "\"role\":", "\"", gram_role, "\"",
      [ ",", "\"position\":", integer ],
      [ ",", "\"phase\":", "\"", phase_name, "\"" ],
      [ ",", "\"provenance\":", string ],
    "}" ;

relation_row =
    "{",
      "\"from\":", string, ",",
      "\"relation\":", string, ",",
      "\"to\":", string, ",",
      "\"source_word\":", string, ",",
      "\"evidence\":", string, ",",
      "\"weight_kind\":", string, ",",
      "\"weight\":", number,
    "}" ;

(* Supplied greeting bank: phase-indexed greeting templates operators can
   extend. The runtime rotates greetings through the C6 phases, mirrors
   greetings the user repeats (reactionary), adapts to learned lingo, and
   synthesizes its own associative words in the same mannerisms. *)
greeting_bank_row =
    "{",
      "\"schema\":", "\"key.grammar.greeting_bank.v1\"", ",",
      [ "\"description\":", string, "," ],
      "\"greetings\":", "[", greeting_phase_entry, { ",", greeting_phase_entry }, "]",
    "}" ;

greeting_phase_entry =
    "{",
      "\"phase\":", "\"", greeting_phase, "\"", ",",
      "\"kind\":", "\"", greeting_kind, "\"", ",",
      "\"templates\":", "[", template_text, { ",", template_text }, "]",
    "}" ;

greeting_phase = "Pop" | "Wo" | "Yax" | "Sek" | "Ch'en" | "Xul"
               | "Pop.reactive_mirror" | "Yax.synthesized_greeting" ;
greeting_kind  = "canonical" | "learned" | "associative" | "reactionary"
               | "synthesized" | "status" | "confirmatory" | "recall" ;
template_text  = ? greeting text with placeholders {cue} {Cue} {greeting}
                  {Greeting} {last} {turns} {lingo_count} {blend} {Blend} ? ;

(* Adaptive greeting learning profile (persisted key + gram record). *)
greeting_learning_row =
    "{",
      "\"language\":", string, ",",
      "\"random_chat_mode\":", boolean, ",",
      "\"salutation_turns\":", number, ",",
      "\"associative_variant_hits\":", number, ",",
      "\"reactive_mirror_hits\":", number, ",",
      "\"synthesized_word_hits\":", number, ",",
      "\"last_user_greeting\":", string, ",",
      "\"learned_lingo\":", "[", lingo_row, { ",", lingo_row }, "]", ",",
      "\"observed_greetings\":", "[", observed_greeting_row, { ",", observed_greeting_row }, "]",
    "}" ;

lingo_row             = "{", "\"token\":", string, ",", "\"hits\":", number, "}" ;
observed_greeting_row = "{", "\"form\":", string, ",", "\"hits\":", number, "}" ;

(* ── Gram channel contract ───────────────────────────────────────────── *)
(* How each channel is USED — order, role, phase, unit, address, consumption.
   Unigrams address meaning; bigrams/trigrams/quadgrams bind ordered context;
   coarse grams match structural shape when exact words miss; function grams
   carry operator/control structure; tool grams carry dispatch intent. *)

gram_class = "unigram" | "bigram" | "trigram" | "quadgram"
           | "coarse_gram" | "function_gram" | "tool_gram" ;
gram_role  = "semantic_address" | "context_association" | "structural_shape"
           | "function_word_structure" | "tool_dispatch_intent" ;
phase_name = "Pop" | "Wo" | "Yax" | "Sek" | "Ch'en" | "Xul" ;

channel_contract =
    unigram_contract | bigram_contract | trigram_contract | quadgram_contract
  | coarse_gram_contract | function_gram_contract | tool_gram_contract ;

unigram_contract =
    "channel=unigram", "width=1", "role=semantic_address", "phase=Pop",
    "unit=semantic_address_layer",
    "address=unigram:<lemma>",
    "consumption=single-token semantic address: lemma resolves to key, word_to_atom:<cue>, atom:<name>, atom_phase_contract:<name>" ;

bigram_contract =
    "channel=bigram", "width=2", "role=context_association", "phase=Wo",
    "unit=context_association_evidence",
    "address=bigram:<left> <right>",
    "consumption=ordered adjacent pair; direction carries meaning — (left,right) is not (right,left)" ;

trigram_contract =
    "channel=trigram", "width=3", "role=context_association", "phase=Wo",
    "unit=context_association_evidence",
    "address=trigram:<t1> <t2> <t3>",
    "consumption=three-token phrase template; wider ordered context than bigram, sparser evidence" ;

quadgram_contract =
    "channel=quadgram", "width=4", "role=context_association", "phase=Wo",
    "unit=context_association_evidence",
    "address=quadgram:<t1> <t2> <t3> <t4>",
    "consumption=four-token phrase template; highest specificity, lowest recall" ;

coarse_gram_contract =
    "channel=coarse_gram", "role=structural_shape", "phase=Yax",
    "unit=coarse_ngram_structure",
    "address=coarse_gram:<c|f:word|ctrl:word> ...",
    "consumption=token-class skeleton (c=content word, f:x=function word, ctrl:x=control word); matches word-order structure when exact grams miss" ;

function_gram_contract =
    "channel=function_gram", "role=function_word_structure", "phase=Yax",
    "unit=function_word_channel",
    "address=function_gram:<glue-token> ...",
    "consumption=grammar-glue and control pivots resolve to XCFE operator authority (CAUSE, VERIFY, ...)" ;

tool_gram_contract =
    "channel=tool_gram", "role=tool_dispatch_intent", "phase=Sek",
    "unit=tool_dispatch_channel",
    "address=tool_gram:<verb> ...",
    "consumption=dispatch intent routes to tool keys k_rpc_<method> / k_mcp_<method> via route_hint:tool_gram:<gram>" ;

(* ── Key-store addressing (runtime memory) ────────────────────────────── *)
key_store_key =
    gram_scope_key | assoc_key | route_hint_key | prompt_key | prompt_ngram_key
  | word_assoc_key | lexicon_key | similarity_key | synonym_key | atom_key
  | greeting_key | lingo_key | style_profile_key | scfml_branch_key | semantic_branch_key | memory_event_key ;

gram_scope_key   = "gram:", gram_class, ":", gram_text ;
assoc_key       = "assoc:", gram_class, ":", gram_text ;
route_hint_key  = "route_hint:tool_gram:", gram_text | "route_hint:track:", gram_text ;
prompt_key      = "prompt:", epoch_ms, "-", sequence ;
prompt_ngram_key = "prompt_ngram:", gram_text ;
word_assoc_key  = "word_assoc:", token, "|", relation, "|", token ;
lexicon_key     = "lexicon:", lemma | "unigram:", lemma | "key:", lemma | "book:", lemma ;
similarity_key  = "word_similarity:", lemma ;
synonym_key     = "synonym:", lemma ;
atom_key        = "word_to_atom:", cue | "atom:", atom | "atom_phase_contract:", atom ;
greeting_key    = "greeting:observed:", greeting_form
                | "greeting:last_user"
                | "greeting:variant:", synthesized_word ;
lingo_key       = "mx2lm:lingo:", lingo_token | "mx2lm:lingo_blend:", synthesized_word ;
style_profile_key = "mx2lm:style_profile:v1" ;
scfml_branch_key = "scfml:branch:", epoch_ms, "-", sequence
                 | "scfml:branch:", gram_text ;
semantic_branch_key = "semantic_branch:", epoch_ms, "-", sequence
                    | "semantic_branch:", gram_text ;
memory_event_key = "memory:event:", memory_event_kind, ":", epoch_ms, "-", sequence ;

(* ── Shared memory-event envelope ────────────────────────────────────── *)
(* Prompts, model responses, and plans are key-grammar events. They are
   event-journal evidence, not automatic fact commits. *)
memory_event_row =
    "{",
      q_me_schema, ",", q_me_key, ",", q_me_event_kind, ",", q_me_source, ",",
      q_me_track, ",", q_me_phase, ",", q_me_memory_scope, ",",
      q_me_fact_authority, ",", q_me_commit_state,
      [ ",", q_me_epistemic_state ],
      [ ",", q_me_observation_authority ],
      [ ",", q_me_epistemic_causes ],
      [ ",", q_me_prompt_key ],
      [ ",", q_me_parent_key ],
      [ ",", q_me_content ],
      [ ",", q_me_hyperbolic_event ],
      [ ",", q_me_payload_json ],
    "}" ;

q_me_schema = "\"schema\":", "\"key.grammar.memory_event.v1\"" ;
q_me_key = "\"key\":", memory_event_key ;
q_me_event_kind = "\"event_kind\":", "\"", memory_event_kind, "\"" ;
q_me_source = "\"source\":", string ;
q_me_track = "\"track\":", string ;
q_me_phase = "\"phase\":", "\"", phase_name, "\"" ;
q_me_memory_scope = "\"memory_scope\":", "\"event_journal\"" ;
q_me_fact_authority = "\"fact_authority\":", boolean ;
q_me_commit_state = "\"commit_state\":", memory_event_commit_state ;
q_me_epistemic_state = "\"epistemic_state\":", "\"", epistemic_state, "\"" ;
q_me_observation_authority = "\"observation_authority\":", boolean ;
q_me_epistemic_causes = "\"epistemic_causes\":", string_array ;
q_me_prompt_key = "\"prompt_key\":", prompt_key ;
q_me_parent_key = "\"parent_key\":", memory_event_key | prompt_key | string ;
q_me_content = "\"content\":", string ;
q_me_hyperbolic_event = "\"hyperbolic_event\":", hyperbolic_event_object ;
q_me_payload_json = "\"payload_json\":", string ;

memory_event_kind = "prompt" | "model_response" | "plan" ;
epistemic_state = "SUPPORTED" | "MISS" | "UNKNOWN" | "CONTRADICTED" ;
memory_event_commit_state =
    "\"journaled_not_fact\"" | "\"candidate\"" | "\"Ch'en_pending\"" | "\"Xul_committed\"" ;

(* ── Prompt archive record (gram-store layout per submitted prompt) ──── *)
(* Every submitted prompt is broken into typed gram channels and archived:
   prompt:<epoch>-<seq> scopes one archive; each channel row keeps class,
   count and gram text with its phase and unit. *)
prompt_archive_record =
    prompt_scope_key, ":", gram_class, ":", ordinal,
    " -> ", gram_class, "|", count, "|", gram_text,
    " @ ", phase_name, " unit=", gram_unit, " corpus=", corpus_name ;

prompt_scope_key = prompt_key ;
gram_unit = "semantic_address_layer" | "context_association_evidence"
          | "coarse_ngram_structure" | "function_word_channel" | "tool_dispatch_channel"
          | "predication_sequence" | "style_adaptation_profile" ;
corpus_name = "definitions_and_relations" | "xcfe_control_program" ;
)EBNF"
// A single string literal is capped by the compiler (MSVC: 16384 bytes), and
// this spec is larger, so it is emitted as adjacent literals that concatenate.
R"EBNF(
(* ── Semantic branch grammar + control flow projection ──────────────── *)
(* @if/@then/@else are semantic branch declarations, not XCFE itself.
   Branching resolves in three layers:
       Semantic Branch Grammar -> Epistemic Resolution -> XCFE Execution.
   @else is a complement branch, not @not(P). MISS is not CONTRADICTED.
   UNKNOWN(P) preserves/spawns/retrieves/defers unless an explicit @unknown
   handler exists; it must not silently select @else. *)

control_flow_row =
    "{",
      q_cf_word, ",", q_cf_key, ",", q_cf_operator, ",", q_cf_book, ",",
      q_cf_role, ",", q_cf_class, ",", q_cf_requires, ",", q_cf_xcfe, ",", q_cf_phase, ",",
      q_cf_authority_phase, ",", q_cf_emit_phase, ",",
      q_cf_block, ",", q_cf_arity, ",", q_cf_sugar,
    "}" ;

q_cf_word       = "\"word\":", control_word ;
q_cf_key        = "\"key\":", string ;
q_cf_operator   = "\"operator\":", string ;
q_cf_book       = "\"book\":", string ;
q_cf_role       = "\"role\":", string ;
q_cf_class      = "\"class\":", xcfe_node_class ;
q_cf_requires   = "\"requires\":", string_array ;
q_cf_xcfe       = "\"xcfe_op\":", control_xcfe_op ;
q_cf_phase      = "\"phase\":", control_phase ;
q_cf_authority_phase = "\"authority_phase\":", control_phase ;
q_cf_emit_phase = "\"emit_phase\":", control_phase ;
q_cf_block      = "\"block_role\":", control_block_role ;
q_cf_arity      = "\"arity\":", control_arity ;
q_cf_sugar      = "\"sugar\":", string ;

control_plan_row =
    "{",
      q_cp_schema, ",", q_cp_form, ",", q_cp_structured, ",", q_cp_condition, ",",
      q_cp_steps,
    "}" ;
q_cp_schema     = "\"schema\":", "\"key.grammar.control_flow_plan.v1\"" ;
q_cp_form       = "\"form\":", control_form ;
q_cp_structured = "\"structured\":", boolean ;
q_cp_condition  = "\"condition\":", string ;
q_cp_steps      = "\"steps\":", "[", [ control_step, { ",", control_step } ], "]" ;

(* AIML/XML surface normalization:
     if/pattern -> condition; then/template -> branch action; then/pattern ->
     pattern candidate; else/if/then -> nested else-if branch. *)
scfml_branch_row = semantic_branch_row ;
semantic_branch_row =
    "{",
      "\"schema\":", ( "\"scfml.branch.v1\"" | "\"key.grammar.semantic_branch.v1\"" ), ",",
      "\"key\":", ( scfml_branch_key | semantic_branch_key ), ",",
      "\"condition\":", string, ",",
      "\"then\":", semantic_branch_target, ",",
      "\"else\":", semantic_branch_target,
      [ ",", "\"track\":", string ],
      [ ",", "\"epistemic_state\":", "\"", epistemic_state, "\"" ],
      [ ",", "\"unknown\":", semantic_branch_target ],
    "}" ;
semantic_branch_target = string | "null" ;

control_step =
    "{",
      q_cs_word, ",", q_cs_operator, ",", q_cs_xcfe, ",", q_cs_role, ",",
      q_cs_class, ",", q_cs_requires, ",", q_cs_phase, ",", q_cs_emit, ",",
      q_cs_authority, ",", q_cs_block, ",", q_cs_position, ",",
      q_cs_arity, ",", q_cs_operand, ",", q_cs_operand_lhs,
    "}" ;
q_cs_word        = "\"word\":", control_word ;
q_cs_operator    = "\"operator\":", string ;
q_cs_xcfe        = "\"xcfe_op\":", control_xcfe_op ;
q_cs_role        = "\"role\":", string ;
q_cs_class       = "\"class\":", xcfe_node_class ;
q_cs_requires    = "\"requires\":", string_array ;
q_cs_phase       = "\"phase\":", control_phase ;
q_cs_emit        = "\"emit_phase\":", control_phase ;
q_cs_authority   = "\"authority_phase\":", control_phase ;
q_cs_block       = "\"block_role\":", control_block_role ;
q_cs_position    = "\"position\":", integer ;
q_cs_arity       = "\"arity\":", control_arity ;
q_cs_operand     = "\"operand\":", string ;
q_cs_operand_lhs = "\"operand_lhs\":", string ;

control_form =
    "if/then" | "if/then/else" | "when/then" | "when/then/else"
  | "compare" | "flat" | "none" ;

(* ── Control-flow sugar: clauses read like English ────────────────────── *)
control_clause =
    conditional_clause | temporal_clause | iteration_clause
  | origin_clause | comparison_clause | query_clause ;

conditional_clause = "if", condition, "then", consequence, [ "else", alternative ] ;
temporal_clause    = "when", condition, "then", consequence, [ "else", alternative ] ;
iteration_clause   = "for", "each", item, "in", collection ;
origin_clause      = "from", source, [ "as", role ] ;
comparison_clause  = expression, comparison_word, expression ;
query_clause       = ( "who" | "why" | "how" ), expression ;

comparison_word = "equals" | "equal" | "is" | "are" ;
condition   = expression ;
consequence = expression ;
alternative = expression ;
expression  = token, { token } ;
item        = token ;
collection  = token, { token } ;
source      = token, { token } ;
role        = token, { token } ;

(* Lowering: a structured control clause becomes an XCFE control node. *)
xcfe_control_node =
    "{", "\"@if\":", string, ",", "\"@then\":", xcfe_branch,
    [ ",", "\"@else\":", xcfe_branch ], "}" ;
xcfe_branch  = string | xcfe_op_node | "null" ;
xcfe_op_node = "{", "\"@fn\":", string, ",", "\"@args\":", string_object, "}" ;
string_object = ? JSON object of string fields ? ;

(* Operator vocabulary + contract bounds. *)
control_word =
    "if" | "else" | "then" | "when" | "how" | "who" | "why" | "for"
  | "from" | "as" | "equals" | "where" | "is" | "are" | "equal"
  | "and" | "or" | "not" | "because" | "requires" | "depends"
  | "evidence" | "shows" | "proves" | "verify" | "check" | "assert" ;
control_phase = "Pop" | "Wo" | "Yax" | "Sek" | "Ch'en" | "Xul" ;
control_xcfe_op =
    "@if" | "@then" | "@else" | "@fn" | "@node" | "@op"
  | "@for" | "@from" | "@as" | "@where" | "@and" | "@or" | "@not"
  | "@direct" | "@cause" ;
control_block_role = "open" | "mid" | "close" | "none" ;
control_arity = "1" | "2" ;

(* Semantic branch epistemic law:
     SUPPORTED(P) may select @then.
     CONTRADICTED(P) may select @else as complement, not as @not(P).
     MISS(P) is not CONTRADICTED(P).
     UNKNOWN(P) must not silently select @else.
     Contradicting one candidate sense only eliminates that sense; it does not
     prove a different candidate sense. *)

(* ── XCFE Node ISA ────────────────────────────────────────────────────── *)
(* Class, operator, flow phase and authority phase are independent
   coordinates. The invariants are:
     class(op) != none                       — every op resolves to a class
     CanonicalPhase(class) != none           — every class has an authority phase
     authority_phase(op) = CanonicalPhase(class(op))   — always
     flow_phase(op) in C6                    — where the construct appears
     flow_phase(op) = authority_phase(op) is LEGAL BUT NEVER REQUIRED
   XcfeNodeClass is deliberately not the Relation ISA family set, and there is
   deliberately no Policy class: operation authority is XCFE's own gate
   (authorize_phase_transition + xcfe_required edges), not a node class.
   The canonical class -> phase mapping is defined once in
   src/kuhul/xcfe_node_class.h; this grammar declares the vocabulary. *)

xcfe_node_class =
    "CONDITION" | "ROUTING" | "CAUSAL" | "STRUCTURE"
  | "SELECTION"
  | "EXECUTION" | "CAPABILITY" | "TRANSFORM"
  | "VERIFICATION" | "CONSEQUENCE"
  | "COMMIT"
  | "MEMORY" | "NEGATION" | "KUHUL_RUNTIME" ;

(* CanonicalPhase(class): the phase where the class's semantics are governed. *)
xcfe_class_authority =
    "CONDITION=>Wo" | "ROUTING=>Wo" | "CAUSAL=>Wo" | "STRUCTURE=>Wo"
  | "SELECTION=>Yax"
  | "EXECUTION=>Sek" | "CAPABILITY=>Sek" | "TRANSFORM=>Sek"
  | "VERIFICATION=>Ch'en" | "CONSEQUENCE=>Ch'en"
  | "COMMIT=>Xul"
  | "MEMORY=>Pop" | "NEGATION=>Pop" | "KUHUL_RUNTIME=>Pop" ;

(* ── Terminals ────────────────────────────────────────────────────────── *)
string_array = "[", [ string, { ",", string } ], "]" ;
number_array = "[", [ number, { ",", number } ], "]" ;
boolean = "true" | "false" ;
string = ? JSON string ? ;
number = ? JSON number ? ;
integer = digit, { digit } ;
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
gram_text = ? ordered gram text; token order is semantic ? ;
lemma = ? lowercase dictionary lemma ? ;
token = ? single lexical token ? ;
cue = ? atom cue token ? ;
atom = ? semantic atom name ? ;
relation = ? relation verb, e.g. @related_to ? ;
epoch_ms = ? milliseconds since UNIX epoch ? ;
sequence = ? per-process prompt sequence number ? ;
ordinal = ? 1-based channel row ordinal ? ;
greeting_form = ? sanitized user greeting form ? ;
lingo_token = ? sanitized learned lingo token (underscores for spaces) ? ;
synthesized_word = ? runtime-blended associative word ? ;
)EBNF";

    {
        std::ofstream aiml_out(aiml_path, std::ios::binary | std::ios::trunc);
        if (!aiml_out) {
            if (err) *err = "unable to write AIML runtime grammar EBNF: " + aiml_path.string();
            return false;
        }
        aiml_out << kAimlEbnf << "\n";
    }
    {
        std::ofstream scfml_out(scfml_path, std::ios::binary | std::ios::trunc);
        if (!scfml_out) {
            if (err) *err = "unable to write SCFML runtime grammar EBNF: " + scfml_path.string();
            return false;
        }
        scfml_out << kScfmlEbnf << "\n";
    }
    {
        std::ofstream key_out(key_json_path, std::ios::binary | std::ios::trunc);
        if (!key_out) {
            if (err) *err = "unable to write key-grammar runtime EBNF: " + key_json_path.string();
            return false;
        }
        key_out << kKeyGrammarJsonEbnf << "\n";
    }
    if (out) {
        out->aiml_ebnf_file = aiml_path.string();
        out->scfml_ebnf_file = scfml_path.string();
        out->key_grammar_ebnf_file = key_json_path.string();
    }
    return true;
}

struct DictionaryKeyGrammarCompileSummary {
    size_t lexicon_rows = 0;
    size_t relation_rows = 0;
    std::string lexicon_file;
    std::string relations_file;
    std::string meta_file;
    std::string aiml_ebnf_file;
    std::string scfml_ebnf_file;
    std::string key_grammar_ebnf_file;
};

static DictionaryKeyGrammarCompileSummary compile_dictionary_key_grammar_files(
    const std::string& data_dir,
    const DictionaryLexicon& lexicon,
    std::string* err)
{
    DictionaryKeyGrammarCompileSummary out;
    fs::path grammar_dir = fs::path(data_dir) / "grammar";
    std::error_code mk_ec;
    fs::create_directories(grammar_dir, mk_ec);
    if (mk_ec) {
        if (err) *err = "unable to create grammar directory: " + grammar_dir.string();
        return out;
    }

    const fs::path lexicon_path = grammar_dir / "english-dictionary.lexicon.key-grammar.jsonl";
    const fs::path relations_path = grammar_dir / "english-dictionary.definition-relations.key-grammar.jsonl";
    const fs::path meta_path = grammar_dir / "english-dictionary.key-grammar.meta.json";

    RuntimeGrammarSpecSummary grammar_specs;
    if (!write_runtime_grammar_specs(grammar_dir, &grammar_specs, err))
        return out;

    std::ofstream lex_out(lexicon_path, std::ios::binary | std::ios::trunc);
    if (!lex_out) {
        if (err) *err = "unable to write lexicon grammar file: " + lexicon_path.string();
        return out;
    }
    std::ofstream rel_out(relations_path, std::ios::binary | std::ios::trunc);
    if (!rel_out) {
        if (err) *err = "unable to write relation grammar file: " + relations_path.string();
        return out;
    }

    // Key-grammar control flow: the operator table as a checked-in artifact, so
    // the EBNF control-flow contract and the runtime operator authority can be
    // validated against each other (mini_transpilers.py control-check).
    {
        const fs::path control_path = grammar_dir / "control-flow.key-grammar.jsonl";
        std::ofstream control_out(control_path, std::ios::binary | std::ios::trunc);
        if (!control_out) {
            if (err) *err = "unable to write control-flow grammar file: " + control_path.string();
            return out;
        }
        for (const auto& row : micronaut::control::operator_rows())
            control_out << util::stringify(row) << "\n";
    }

    std::vector<std::string> words;
    words.reserve(lexicon.definition_by_word.size());
    for (const auto& [word, _] : lexicon.definition_by_word)
        words.push_back(word);
    std::sort(words.begin(), words.end());

    for (const auto& word : words) {
        auto def_it = lexicon.definition_by_word.find(word);
        if (def_it == lexicon.definition_by_word.end()) continue;

        const auto lemma = resolve_lexicon_lemma(word, &lexicon);
        const auto key = lexicon_key_for_word(lemma);
        if (lemma.empty() || key.empty()) continue;

        const auto definition = def_it->second;
        const auto class_name = lexicon_class_for_word(lemma, definition);
        const auto book = lexicon_book_for_key(key);
        const auto synonyms = dictionary_synonyms_for_word(&lexicon, lemma, 8);
        const auto ngrams = build_learning_ngrams(definition, 24);

        std::vector<std::string> related_lemmas;
        std::unordered_set<std::string> related_seen;
        auto def_tokens = micronaut::Mx2LmNGram::tokenize(lower_copy(definition));
        const auto def_taxonomy = micronaut::Mx2LmNGram::classify_gram_taxonomy(def_tokens);
        size_t scanned = 0;
        for (const auto& tok : def_tokens) {
            if (++scanned > 256) break;
            const auto related = resolve_lexicon_lemma(tok, &lexicon);
            if (related.empty() || related == lemma) continue;
            if (lexicon.definition_by_word.find(related) == lexicon.definition_by_word.end()) continue;
            if (!related_seen.insert(related).second) continue;
            related_lemmas.push_back(related);
            if (related_lemmas.size() >= 32) break;
        }

        util::JsonObject row;
        row["word"] = util::json_string(lemma);
        row["key"] = util::json_string(key);
        row["unit"] = util::json_string("unigram");
        row["class"] = util::json_string(class_name);
        row["book"] = util::json_string(book);
        row["definition"] = util::json_string(definition);
        row["lexicon_source"] = util::json_string("data/gram/english-dictionary.jsonl");

        if (!synonyms.empty()) {
            util::JsonArray syn_arr;
            for (const auto& s : synonyms) syn_arr.push_back(util::json_string(s));
            row["synonyms"] = util::json_array(std::move(syn_arr));
        }
        if (!ngrams.empty()) {
            util::JsonArray gram_arr;
            for (const auto& gram : ngrams) gram_arr.push_back(util::json_string(gram));
            row["definition_ngrams"] = util::json_array(std::move(gram_arr));
        }
        if (!related_lemmas.empty()) {
            util::JsonArray related_key_arr;
            for (const auto& rel_lemma : related_lemmas) {
                const auto rel_key = lexicon_key_for_word(rel_lemma);
                if (!rel_key.empty())
                    related_key_arr.push_back(util::json_string(rel_key));
            }
            row["definition_keys"] = util::json_array(std::move(related_key_arr));
        }
        {
            util::JsonObject gram_channels;
            auto push_channel = [&](const std::string& channel,
                                    const micronaut::NGramFreq& freq,
                                    const std::string& role,
                                    const std::string& phase) {
                const auto ranked = top_ranked_grams(freq, 12);
                if (ranked.empty()) return;
                util::JsonArray arr;
                for (const auto& [gram, count] : ranked) {
                    util::JsonObject g;
                    g["gram"] = util::json_string(gram);
                    g["count"] = util::json_number(static_cast<double>(count));
                    // Ordered slot width: coarse symbols (c / f:x / ctrl:x / t:x)
                    // each count as ONE structural slot, matching the engine's
                    // width_from_gram_text rule (split on spaces, not ':').
                    const auto width = static_cast<double>(
                        1 + static_cast<size_t>(std::count(gram.begin(), gram.end(), ' ')));
                    g["width"] = util::json_number(width <= 0.0 ? 1.0 : width);
                    g["role"] = util::json_string(role);
                    // First-occurrence position: order evidence for consumers
                    // that need where (not just whether) the gram appears.
                    // Coarse grams are structural skeletons (c / f:x / ctrl:x
                    // symbols), so lexical position does not apply.
                    const auto gram_toks = micronaut::Mx2LmNGram::tokenize(gram);
                    if (channel != "coarse_gram" && !gram_toks.empty() && !def_tokens.empty()) {
                        size_t first_pos = 0;
                        bool found = false;
                        for (size_t i = 0; i + gram_toks.size() <= def_tokens.size(); ++i) {
                            bool match = true;
                            for (size_t j = 0; j < gram_toks.size(); ++j) {
                                if (def_tokens[i + j] != gram_toks[j]) { match = false; break; }
                            }
                            if (match) { first_pos = i; found = true; break; }
                        }
                        if (found)
                            g["position"] = util::json_number(static_cast<double>(first_pos));
                    }
                    g["phase"] = util::json_string(phase);
                    g["provenance"] = util::json_string("definition_unigram");
                    arr.push_back(util::json_object(std::move(g)));
                }
                gram_channels[channel] = util::json_array(std::move(arr));
            };

            push_channel("unigram", def_taxonomy.unigrams, "semantic_address", "Pop");
            push_channel("bigram", def_taxonomy.bigrams, "context_association", "Wo");
            push_channel("trigram", def_taxonomy.trigrams, "context_association", "Wo");
            push_channel("quadgram", def_taxonomy.quadgrams, "context_association", "Wo");
            push_channel("coarse_gram", def_taxonomy.coarse_grams, "structural_shape", "Yax");
            push_channel("function_gram", def_taxonomy.function_grams, "function_word_structure", "Yax");
            push_channel("tool_gram", def_taxonomy.tool_grams, "tool_dispatch_intent", "Sek");
            if (!gram_channels.empty())
                row["gram_channels"] = util::json_object(std::move(gram_channels));
        }

        lex_out << util::stringify(util::json_object(std::move(row))) << "\n";
        ++out.lexicon_rows;

        for (const auto& rel_lemma : related_lemmas) {
            const auto rel_key = lexicon_key_for_word(rel_lemma);
            if (rel_key.empty()) continue;

            util::JsonObject edge;
            edge["from"] = util::json_string(key);
            edge["relation"] = util::json_string("@related_to");
            edge["to"] = util::json_string(rel_key);
            edge["source_word"] = util::json_string(lemma);
            edge["evidence"] = util::json_string("definition_unigram");
            edge["weight_kind"] = util::json_string("lexical_evidence");
            edge["weight"] = util::json_number(1.0);
            rel_out << util::stringify(util::json_object(std::move(edge))) << "\n";
            ++out.relation_rows;
        }
    }

    util::JsonObject meta;
    meta["schema"] = util::json_string("key.grammar.dictionary.v2");
    meta["dictionary_path"] = util::json_string("data/gram/english-dictionary.jsonl");
    meta["lexicon_file"] = util::json_string(lexicon_path.string());
    meta["relations_file"] = util::json_string(relations_path.string());
    meta["aiml_runtime_ebnf_file"] = util::json_string(grammar_specs.aiml_ebnf_file);
    meta["scfml_runtime_ebnf_file"] = util::json_string(grammar_specs.scfml_ebnf_file);
    meta["key_grammar_runtime_ebnf_file"] = util::json_string(grammar_specs.key_grammar_ebnf_file);
    meta["lexicon_rows"] = util::json_number(static_cast<double>(out.lexicon_rows));
    meta["relation_rows"] = util::json_number(static_cast<double>(out.relation_rows));
    meta["gram_classes"] = util::json_array(json_string_array(std::vector<std::string>{
        "unigram", "bigram", "trigram", "quadgram", "coarse_gram", "function_gram", "tool_gram"
    }));
    meta["gram_roles"] = util::json_array(json_string_array(std::vector<std::string>{
        "semantic_address",
        "context_association",
        "structural_shape",
        "function_word_structure",
        "tool_dispatch_intent"
    }));
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    meta["generated_epoch_ms"] = util::json_string(
        std::to_string(static_cast<long long>(now_ms)));
    {
        std::ofstream meta_out(meta_path, std::ios::binary | std::ios::trunc);
        if (!meta_out) {
            if (err) *err = "unable to write key grammar metadata file: " + meta_path.string();
            return out;
        }
        meta_out << util::stringify(util::json_object(std::move(meta)), true) << "\n";
    }

    out.lexicon_file = lexicon_path.string();
    out.relations_file = relations_path.string();
    out.meta_file = meta_path.string();
    out.aiml_ebnf_file = grammar_specs.aiml_ebnf_file;
    out.scfml_ebnf_file = grammar_specs.scfml_ebnf_file;
    out.key_grammar_ebnf_file = grammar_specs.key_grammar_ebnf_file;
    return out;
}

struct PromptPersistenceSummary {
    std::string prompt_key;
    size_t prompt_ngram_count = 0;
    size_t prompt_unigram_count = 0;
    size_t prompt_bigram_count = 0;
    size_t prompt_trigram_count = 0;
    size_t prompt_quadgram_count = 0;
    size_t prompt_coarse_gram_count = 0;
    size_t prompt_function_gram_count = 0;
    size_t prompt_tool_gram_count = 0;
    size_t prompt_lexicon_count = 0;
    size_t prompt_assoc_count = 0;
    size_t prompt_atom_count = 0;
    size_t prompt_similarity_count = 0;
    size_t file_candidate_count = 0;
    size_t file_ingested_count = 0;
    size_t file_unigram_count = 0;
    size_t file_bigram_count = 0;
    size_t file_trigram_count = 0;
    size_t file_quadgram_count = 0;
    size_t file_coarse_gram_count = 0;
    size_t file_function_gram_count = 0;
    size_t file_tool_gram_count = 0;
    size_t file_lexicon_count = 0;
    size_t file_atom_count = 0;
    size_t file_similarity_count = 0;
    size_t file_pdf_pending_count = 0;
};

static uint64_t& prompt_persistence_sequence() {
    static uint64_t seq = 0;
    return seq;
}

static std::mutex& prompt_persistence_mutex() {
    static std::mutex mu;
    return mu;
}

static std::string next_prompt_archive_key() {
    uint64_t seq = 0;
    {
        std::lock_guard<std::mutex> lk(prompt_persistence_mutex());
        seq = ++prompt_persistence_sequence();
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return "prompt:" + std::to_string(static_cast<long long>(now_ms)) + "-" + std::to_string(seq);
}

static uint64_t& memory_event_sequence() {
    static uint64_t seq = 0;
    return seq;
}

static std::mutex& memory_event_mutex() {
    static std::mutex mu;
    return mu;
}

static std::string normalize_memory_event_kind(const std::string& kind) {
    const std::string k = lower_copy(trim_copy(kind));
    if (k == "prompt") return "prompt";
    if (k == "plan") return "plan";
    return "model_response";
}

static std::string next_memory_event_key(const std::string& kind) {
    uint64_t seq = 0;
    {
        std::lock_guard<std::mutex> lk(memory_event_mutex());
        seq = ++memory_event_sequence();
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return "memory:event:" + normalize_memory_event_kind(kind) + ":" +
           std::to_string(static_cast<long long>(now_ms)) + "-" + std::to_string(seq);
}

static std::string clamp_memory_event_text(std::string text, size_t max_chars = 65536) {
    if (text.size() <= max_chars) return text;
    text.resize(max_chars);
    text += "...[truncated]";
    return text;
}

static bool is_valid_epistemic_state(const std::string& state) {
    return state == "SUPPORTED" || state == "MISS" ||
           state == "UNKNOWN" || state == "CONTRADICTED";
}

static std::string normalize_epistemic_state(std::string state) {
    state = trim_copy(state);
    std::transform(state.begin(), state.end(), state.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return is_valid_epistemic_state(state) ? state : "UNKNOWN";
}

static std::string c6_phase_from_hint(const std::string& hint, const std::string& fallback) {
    const std::string h = lower_copy(trim_copy(hint));
    if (h.rfind("pop", 0) == 0 || h.find("phase:pop") != std::string::npos) return "Pop";
    if (h.rfind("wo", 0) == 0 || h.find("phase:wo") != std::string::npos) return "Wo";
    if (h.rfind("yax", 0) == 0 || h.find("phase:yax") != std::string::npos) return "Yax";
    if (h.rfind("sek", 0) == 0 || h.find("phase:sek") != std::string::npos) return "Sek";
    if (h.rfind("ch'en", 0) == 0 || h.rfind("chen", 0) == 0 ||
        h.find("phase:ch'en") != std::string::npos || h.find("phase:chen") != std::string::npos)
        return "Ch'en";
    if (h.rfind("xul", 0) == 0 || h.find("phase:xul") != std::string::npos) return "Xul";
    return fallback;
}

static std::string default_memory_event_phase(const std::string& kind, const std::string& hint = "") {
    const std::string k = normalize_memory_event_kind(kind);
    if (k == "prompt") return "Pop";
    if (k == "plan") return c6_phase_from_hint(hint, "Yax");
    return c6_phase_from_hint(hint, "Ch'en");
}

static std::string infer_memory_event_kind_from_payload(const util::JsonPtr& payload) {
    if (!payload || !payload->is_object()) return "model_response";
    const std::string status = lower_copy(payload->get_string("status").value_or(""));
    const std::string source = lower_copy(payload->get_string("source").value_or(""));
    if (status.find("plan") != std::string::npos ||
        source.find("plan") != std::string::npos ||
        payload->get("plan_card") ||
        payload->get("research") ||
        payload->get("route_hint"))
        return "plan";
    return "model_response";
}

static std::string infer_memory_event_epistemic_state(
    const util::JsonPtr& payload,
    const std::string& event_kind)
{
    if (!payload || !payload->is_object()) return "UNKNOWN";
    if (auto s = payload->get_string("epistemic_state"); s && !s->empty())
        return normalize_epistemic_state(*s);

    const std::string status = lower_copy(payload->get_string("status").value_or(""));
    const std::string source = lower_copy(payload->get_string("source").value_or(""));
    if (status == "no_fact_found" ||
        status.find("miss") != std::string::npos ||
        status.find("web_research_dispatched") != std::string::npos ||
        source == "miss")
        return "MISS";
    if (source == "key_store")
        return "SUPPORTED";
    (void)event_kind;
    return "UNKNOWN";
}

static std::string memory_event_content_from_payload(const util::JsonPtr& payload) {
    if (!payload || !payload->is_object()) return {};
    static const std::vector<std::string> kPreferred = {
        "response", "text", "status", "source"
    };
    for (const auto& key : kPreferred) {
        if (auto s = payload->get_string(key); s && !s->empty())
            return clamp_memory_event_text(*s, 8192);
    }
    return clamp_memory_event_text(util::stringify(payload), 8192);
}

static std::string record_key_grammar_memory_event(
    const std::string& event_kind_raw,
    const std::string& source,
    const std::string& track,
    const std::string& phase_hint,
    const std::string& prompt_key,
    const std::string& parent_key,
    const std::string& content,
    const std::string& payload_json,
    const std::string& epistemic_state_raw,
    bool observation_authority,
    micronaut::Factory* factory,
    kuhul::GramStore* gram_store,
    io::JROM* jrom)
{
    const std::string event_kind = normalize_memory_event_kind(event_kind_raw);
    const std::string event_key = next_memory_event_key(event_kind);
    const std::string phase = default_memory_event_phase(event_kind, phase_hint);
    const std::string epistemic_state = normalize_epistemic_state(epistemic_state_raw);

    util::JsonObject event;
    event["schema"] = util::json_string("key.grammar.memory_event.v1");
    event["key"] = util::json_string(event_key);
    event["event_kind"] = util::json_string(event_kind);
    event["source"] = util::json_string(source.empty() ? "runtime" : source);
    event["track"] = util::json_string(track.empty() ? "Mx2LM" : track);
    event["phase"] = util::json_string(phase);
    event["memory_scope"] = util::json_string("event_journal");
    event["fact_authority"] = util::json_bool(false);
    event["commit_state"] = util::json_string("journaled_not_fact");
    event["epistemic_state"] = util::json_string(epistemic_state);
    event["observation_authority"] = util::json_bool(observation_authority);
    if (!prompt_key.empty()) event["prompt_key"] = util::json_string(prompt_key);
    if (!parent_key.empty()) event["parent_key"] = util::json_string(parent_key);
    if (!content.empty()) event["content"] = util::json_string(clamp_memory_event_text(content, 8192));
    if (!payload_json.empty()) event["payload_json"] = util::json_string(clamp_memory_event_text(payload_json));

    const std::string event_payload = util::stringify(util::json_object(event));
    if (gram_store) {
        gram_store->gramWrite(
            event_key,
            event_payload,
            phase,
            "memory_event",
            "definitions_and_relations");
    }
    if (factory) {
        factory->record_key(event_key, event_payload);
        factory->record_key("memory:event_kind:" + event_kind, event_key);
        factory->record_key("memory:event:source:" + (source.empty() ? std::string("runtime") : source), event_key);
        factory->record_key("memory:event:track:" + (track.empty() ? std::string("Mx2LM") : track), event_key);
        if (!prompt_key.empty())
            factory->record_key("memory:event:prompt_ref:" + prompt_key, event_key);
        if (!parent_key.empty())
            factory->record_key("memory:event:parent_ref:" + parent_key, event_key);
    }
    if (jrom)
        jrom->log("memory_event", event_payload);
    return event_key;
}

static void attach_memory_event_metadata(
    const util::JsonPtr& payload,
    const std::string& event_key,
    const std::string& event_kind,
    const std::string& prompt_event_key,
    const std::string& prompt_archive_key)
{
    if (!payload || !payload->is_object()) return;
    auto& root = payload->mut_object();
    root["memory_event_key"] = util::json_string(event_key);
    root["memory_event_kind"] = util::json_string(normalize_memory_event_kind(event_kind));
    if (!prompt_event_key.empty())
        root["prompt_event_key"] = util::json_string(prompt_event_key);
    if (!prompt_archive_key.empty())
        root["prompt_archive_key"] = util::json_string(prompt_archive_key);
}

static bool is_prompt_text_extension(const std::string& ext_lower) {
    static const std::unordered_set<std::string> kTextExts = {
        ".md", ".markdown", ".txt", ".text",
        ".html", ".htm",
        ".json", ".jsonl",
        ".grammar", ".ebnf",
        ".xml", ".xsd",
        ".kson", ".khl", ".kuhul",
        ".yaml", ".yml", ".toml", ".ini", ".cfg"
    };
    return kTextExts.count(ext_lower) != 0;
}

static bool is_prompt_pdf_extension(const std::string& ext_lower) {
    return ext_lower == ".pdf";
}

static bool is_prompt_source_extension(const std::string& ext_lower) {
    return is_prompt_text_extension(ext_lower) || is_prompt_pdf_extension(ext_lower);
}

static std::string trim_path_candidate(std::string text) {
    text = trim_copy(text);
    while (!text.empty()) {
        const char c = text.front();
        if (c == '"' || c == '\'' || c == '`' || c == '(' || c == '[' || c == '{' || c == '<') {
            text.erase(text.begin());
            continue;
        }
        break;
    }
    while (!text.empty()) {
        const char c = text.back();
        if (c == '"' || c == '\'' || c == '`' || c == ',' || c == ';' ||
            c == ')' || c == ']' || c == '}' || c == '>') {
            text.pop_back();
            continue;
        }
        break;
    }
    return trim_copy(text);
}

static bool looks_like_local_path_reference(const std::string& candidate) {
    if (candidate.empty()) return false;

    const bool abs_drive_path =
        candidate.size() >= 3 &&
        std::isalpha(static_cast<unsigned char>(candidate[0])) &&
        candidate[1] == ':' &&
        (candidate[2] == '\\' || candidate[2] == '/');
    if (abs_drive_path) return true;

    if (candidate.rfind(".\\", 0) == 0 || candidate.rfind("..\\", 0) == 0 ||
        candidate.rfind("./", 0) == 0 || candidate.rfind("../", 0) == 0) {
        return true;
    }

    const auto dot = candidate.find_last_of('.');
    const bool has_ext = dot != std::string::npos && dot > 0 && dot + 1 < candidate.size();
    if (!has_ext) return false;
    const auto ext = lower_copy(candidate.substr(dot));
    if (!is_prompt_source_extension(ext)) return false;

    const bool has_sep =
        candidate.find('\\') != std::string::npos ||
        candidate.find('/') != std::string::npos;
    return has_sep || candidate.size() <= 260;
}

static std::vector<std::string> extract_path_candidates_from_prompt(const std::string& text) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    auto push = [&](std::string candidate) {
        candidate = trim_path_candidate(std::move(candidate));
        if (candidate.empty()) return;
        if (!looks_like_local_path_reference(candidate)) return;
        auto dedup_key = lower_copy(candidate);
        std::replace(dedup_key.begin(), dedup_key.end(), '/', '\\');
        if (!seen.insert(dedup_key).second) return;
        out.push_back(std::move(candidate));
    };

    for (size_t i = 0; i < text.size(); ++i) {
        const char q = text[i];
        if (q != '"' && q != '\'') continue;
        const auto end = text.find(q, i + 1);
        if (end == std::string::npos) break;
        push(text.substr(i + 1, end - i - 1));
        i = end;
    }

    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) push(tok);
    return out;
}

static std::optional<fs::path> resolve_prompt_path_candidate(
    const std::string& candidate,
    const fs::path& repo_root)
{
    const std::string cleaned = trim_path_candidate(candidate);
    if (cleaned.empty()) return std::nullopt;

    auto try_resolve = [](const fs::path& p) -> std::optional<fs::path> {
        if (p.empty()) return std::nullopt;
        std::error_code ec;
        if (!fs::exists(p, ec) || ec) return std::nullopt;
        if (!fs::is_regular_file(p, ec) || ec) return std::nullopt;
        return p.lexically_normal();
    };

    fs::path raw(cleaned);
    if (raw.is_absolute()) {
        if (auto hit = try_resolve(raw)) return hit;
        return std::nullopt;
    }

    if (auto hit = try_resolve(repo_root / raw)) return hit;
    std::error_code cwd_ec;
    const auto cwd = fs::current_path(cwd_ec);
    if (!cwd_ec) {
        if (auto hit = try_resolve(cwd / raw)) return hit;
    }
    if (auto hit = try_resolve(raw)) return hit;
    return std::nullopt;
}

static std::string strip_html_markup(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool in_tag = false;
    for (char c : text) {
        if (c == '<') {
            in_tag = true;
            out.push_back(' ');
            continue;
        }
        if (c == '>') {
            in_tag = false;
            continue;
        }
        if (!in_tag) out.push_back(c);
    }
    std::string clean;
    clean.reserve(out.size());
    bool ws = false;
    for (char c : out) {
        const bool space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (space) {
            if (!ws) clean.push_back(' ');
            ws = true;
        } else {
            clean.push_back(c);
            ws = false;
        }
    }
    return trim_copy(clean);
}

struct SemanticAtomEvidence {
    std::string atom;
    std::vector<std::string> cues;
};

static std::optional<std::string> semantic_atom_from_cue(const std::string& cue_lower) {
    static const std::unordered_map<std::string, std::string> kCueToAtom = {
        {"if", "CONDITION"},
        {"when", "CONDITION"},
        {"condition", "CONDITION"},
        {"unless", "CONDITION"},
        {"then", "CONSEQUENCE"},
        {"result", "CONSEQUENCE"},
        {"consequence", "CONSEQUENCE"},
        {"therefore", "CONSEQUENCE"},
        {"because", "CAUSAL"},
        {"cause", "CAUSAL"},
        {"reason", "CAUSAL"},
        {"verify", "VERIFICATION"},
        {"verification", "VERIFICATION"},
        {"check", "VERIFICATION"},
        {"confirm", "VERIFICATION"},
        {"validate", "VERIFICATION"},
        {"prove", "VERIFICATION"},
        {"evidence", "VERIFICATION"},
        {"support", "VERIFICATION"},
        {"supported", "VERIFICATION"},
        {"not", "NEGATION"},
        {"no", "NEGATION"},
        {"never", "NEGATION"},
        {"without", "NEGATION"},
        {"absence", "NEGATION"},
        {"prefix:un", "NEGATION"},
        {"prefix:non", "NEGATION"},
        {"prefix:dis", "NEGATION"},
        {"prefix:in", "NEGATION"},
        {"prefix:im", "NEGATION"},
        {"prefix:ir", "NEGATION"},
        {"prefix:il", "NEGATION"},
        {"suffix:able", "CAPABILITY"},
        {"suffix:ible", "CAPABILITY"},
        {"ability", "CAPABILITY"},
        {"capability", "CAPABILITY"},
        {"choose", "SELECTION"},
        {"select", "SELECTION"},
        {"selection", "SELECTION"},
        {"filter", "SELECTION"},
        {"rank", "SELECTION"},
        {"relevance", "SELECTION"},
        {"candidate", "SELECTION"},
        {"execute", "EXECUTION"},
        {"execution", "EXECUTION"},
        {"run", "EXECUTION"},
        {"dispatch", "EXECUTION"},
        {"call", "EXECUTION"},
        {"commit", "COMMIT"},
        {"archive", "COMMIT"},
        {"persist", "COMMIT"},
        {"store", "COMMIT"},
        {"finalize", "COMMIT"},
        {"route", "ROUTING"},
        {"routing", "ROUTING"},
        {"router", "ROUTING"},
        {"delegate", "ROUTING"},
        {"memory", "MEMORY"},
        {"recall", "MEMORY"},
        {"history", "MEMORY"},
        {"kuhul", "KUHUL_RUNTIME"},
        {"k'uhul", "KUHUL_RUNTIME"},
        {"fold", "KUHUL_RUNTIME"},
        {"phase", "KUHUL_RUNTIME"},
        {"runtime", "KUHUL_RUNTIME"},
        {"pop", "KUHUL_RUNTIME"},
        {"wo", "KUHUL_RUNTIME"},
        {"yax", "KUHUL_RUNTIME"},
        {"sek", "KUHUL_RUNTIME"},
        {"ch'en", "KUHUL_RUNTIME"},
        {"xul", "KUHUL_RUNTIME"}
    };
    auto it = kCueToAtom.find(cue_lower);
    if (it == kCueToAtom.end()) return std::nullopt;
    return it->second;
}

static std::vector<std::string> morpheme_cues_for_token(const std::string& token_lower) {
    std::vector<std::string> cues;
    if (token_lower.size() < 4) return cues;

    static const std::vector<std::string> kNegPrefixes = {
        "un", "non", "dis", "in", "im", "ir", "il"
    };
    for (const auto& prefix : kNegPrefixes) {
        if (token_lower.size() <= prefix.size() + 2) continue;
        if (token_lower.rfind(prefix, 0) == 0) {
            cues.push_back("prefix:" + prefix);
            break;
        }
    }

    static const std::vector<std::string> kCapabilitySuffixes = {"able", "ible"};
    for (const auto& suffix : kCapabilitySuffixes) {
        if (token_lower.size() <= suffix.size() + 1) continue;
        if (token_lower.size() >= suffix.size() &&
            token_lower.compare(token_lower.size() - suffix.size(), suffix.size(), suffix) == 0) {
            cues.push_back("suffix:" + suffix);
            break;
        }
    }
    return cues;
}

static std::vector<SemanticAtomEvidence> derive_semantic_atom_grounding(
    const std::vector<std::string>& tokens_lower)
{
    std::unordered_map<std::string, std::unordered_set<std::string>> atom_cues;
    for (const auto& token : tokens_lower) {
        if (token.empty()) continue;
        if (auto atom = semantic_atom_from_cue(token)) {
            atom_cues[*atom].insert(token);
        }
        for (const auto& mcue : morpheme_cues_for_token(token)) {
            if (auto atom = semantic_atom_from_cue(mcue))
                atom_cues[*atom].insert(mcue + "@" + token);
        }
    }

    std::vector<SemanticAtomEvidence> out;
    out.reserve(atom_cues.size());
    for (auto& [atom, cues_set] : atom_cues) {
        SemanticAtomEvidence row;
        row.atom = atom;
        row.cues.assign(cues_set.begin(), cues_set.end());
        std::sort(row.cues.begin(), row.cues.end());
        out.push_back(std::move(row));
    }
    std::sort(out.begin(), out.end(), [](const SemanticAtomEvidence& a, const SemanticAtomEvidence& b) {
        return a.atom < b.atom;
    });
    return out;
}

static std::string canonical_phase_for_semantic_atom(const std::string& atom) {
    return kuhul::canonical_phase_name_for_semantic_atom(atom);
}

static void record_semantic_atom_grounding(
    const std::string& scope_key,
    const std::vector<SemanticAtomEvidence>& atoms,
    micronaut::Factory* factory,
    kuhul::GramStore* gram_store)
{
    if (!factory && !gram_store) return;
    for (size_t i = 0; i < atoms.size(); ++i) {
        const auto& atom = atoms[i];
        if (factory) {
            factory->record_key("atom:" + atom.atom, scope_key);
            const auto phase = canonical_phase_for_semantic_atom(atom.atom);
            if (!phase.empty())
                factory->record_key("atom_phase_contract:" + atom.atom, phase);
            for (const auto& cue : atom.cues)
                factory->record_key("word_to_atom:" + cue, atom.atom);
        }
        if (gram_store) {
            std::string cue_joined;
            for (size_t c = 0; c < atom.cues.size(); ++c) {
                if (!cue_joined.empty()) cue_joined += ",";
                cue_joined += atom.cues[c];
            }
            gram_store->gramWrite(
                scope_key + ":atom:" + std::to_string(i + 1),
                atom.atom + "|" + cue_joined,
                "Wo",
                "taxonomy_geometry",
                "definitions_and_relations");

            const auto phase = canonical_phase_for_semantic_atom(atom.atom);
            if (!phase.empty()) {
                gram_store->gramWrite(
                    scope_key + ":atom_contract:" + std::to_string(i + 1),
                    atom.atom + "->" + phase + " (canonical_only)",
                    "Ch'en",
                    "xcfe_control_program",
                    "xcfe_control_program");
            }
        }
    }
}

static std::string fold_graph_relation_token(const std::string& relation_raw) {
    const auto rel = lower_copy(relation_raw);
    std::string mapped = "@related_to";
    if (rel == "because" || rel == "cause" || rel == "reason")
        mapped = "@causes";
    else if (rel == "verify" || rel == "check" || rel == "confirm" ||
             rel == "validate" || rel == "prove" || rel == "supported" ||
             rel == "support" || rel == "evidence")
        mapped = "@supports";
    else if (rel == "requires" || rel == "depends" || rel == "depends_on")
        mapped = "@requires";
    else if (rel == "not" || rel == "without" || rel == "no")
        mapped = "@contradicts";
    else if (rel == "if" || rel == "when")
        mapped = "@context";
    else if (rel == "then")
        mapped = "@next";
    return micronaut::canonical_relation(mapped);
}

static util::JsonPtr fold_graph_parse_payload(const std::string& text) {
    if (auto delegated = io::fold_graph_parse(text);
        delegated && !delegated->is_null()) {
        return delegated;
    }

    const auto q = trim_copy(text);
    if (q.empty()) return util::json_null();

    auto tokens = micronaut::Mx2LmNGram::tokenize(lower_copy(q));
    if (tokens.size() > 512) tokens.resize(512);

    const auto projection = micronaut::Mx2LmNGram::compose(tokens);
    const auto atoms = derive_semantic_atom_grounding(tokens);
    const auto ngrams = build_learning_ngrams(q, 32);

    std::unordered_map<std::string, std::unordered_set<std::string>> neighborhood;
    for (const auto& rel : projection.relations) {
        if (rel.from.empty() || rel.to.empty()) continue;
        neighborhood[rel.from].insert(rel.to);
        neighborhood[rel.to].insert(rel.from);
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> token_atoms;
    std::unordered_set<std::string> atom_cue_tokens;
    for (const auto& atom : atoms) {
        for (const auto& cue : atom.cues) {
            std::string token = cue;
            const auto at_pos = token.find('@');
            if (at_pos != std::string::npos && at_pos + 1 < token.size())
                token = token.substr(at_pos + 1);
            token = lower_copy(token);
            if (token.empty()) continue;
            token_atoms[token].insert(atom.atom);
            atom_cue_tokens.insert(token);
        }
    }

    util::JsonArray atom_rows;
    for (const auto& atom : atoms) {
        util::JsonObject row;
        row["atom"] = util::json_string(atom.atom);
        row["phase_contract"] = util::json_string(canonical_phase_for_semantic_atom(atom.atom));
        row["execution_authority"] = util::json_string("canonical_only");
        row["cues"] = util::json_array(json_string_array(atom.cues));
        atom_rows.push_back(util::json_object(std::move(row)));
    }

    util::JsonArray relation_rows;
    for (const auto& rel : projection.relations) {
        util::JsonObject row;
        row["from"] = util::json_string(rel.from);
        row["to"] = util::json_string(rel.to);
        row["lexical_relation"] = util::json_string(rel.relation);
        row["typed_relation"] = util::json_string(fold_graph_relation_token(rel.relation));
        row["association_weight"] = util::json_number(static_cast<double>(rel.association_weight));
        relation_rows.push_back(util::json_object(std::move(row)));
    }

    util::JsonArray book_rows;
    {
        std::vector<std::string> tokens_sorted;
        tokens_sorted.reserve(neighborhood.size());
        for (const auto& [tok, _] : neighborhood) tokens_sorted.push_back(tok);
        std::sort(tokens_sorted.begin(), tokens_sorted.end());
        for (const auto& tok : tokens_sorted) {
            auto nh_it = neighborhood.find(tok);
            if (nh_it == neighborhood.end()) continue;
            std::vector<std::string> neighbors(nh_it->second.begin(), nh_it->second.end());
            std::sort(neighbors.begin(), neighbors.end());
            if (neighbors.size() > 8) neighbors.resize(8);

            std::vector<std::string> atom_hits;
            if (auto ait = token_atoms.find(tok); ait != token_atoms.end()) {
                atom_hits.assign(ait->second.begin(), ait->second.end());
                std::sort(atom_hits.begin(), atom_hits.end());
            }

            util::JsonObject row;
            row["token"] = util::json_string(tok);
            row["neighbors"] = util::json_array(json_string_array(neighbors));
            row["atoms"] = util::json_array(json_string_array(atom_hits));
            book_rows.push_back(util::json_object(std::move(row)));
        }
    }

    util::JsonArray candidate_rows;
    for (const auto& atom : atoms) {
        util::JsonObject row;
        row["atom"] = util::json_string(atom.atom);
        row["candidate_phase"] = util::json_string(canonical_phase_for_semantic_atom(atom.atom));
        row["evidence_count"] = util::json_number(static_cast<double>(atom.cues.size()));
        row["selection_policy"] = util::json_string("Yax_WCR_candidate");
        row["phase_binding"] = util::json_string("XCFE_canonical_contract_only");
        candidate_rows.push_back(util::json_object(std::move(row)));
    }

    util::JsonArray unresolved_rows;
    {
        std::unordered_set<std::string> unresolved_set;
        for (const auto& tok : tokens) {
            if (tok.size() < 4) continue;
            if (atom_cue_tokens.count(tok)) continue;
            unresolved_set.insert(tok);
        }
        std::vector<std::string> unresolved(unresolved_set.begin(), unresolved_set.end());
        std::sort(unresolved.begin(), unresolved.end());
        for (const auto& tok : unresolved) {
            util::JsonObject row;
            row["token"] = util::json_string(tok);
            row["miss_kind"] = util::json_string("lexical_root_or_semantic_mapping");
            unresolved_rows.push_back(util::json_object(std::move(row)));
        }
    }

    util::JsonArray provenance_rows;
    {
        util::JsonObject qrow;
        qrow["kind"] = util::json_string("query_text");
        qrow["value"] = util::json_string(q);
        provenance_rows.push_back(util::json_object(std::move(qrow)));
        for (size_t i = 0; i < ngrams.size() && i < 12; ++i) {
            util::JsonObject grow;
            grow["kind"] = util::json_string("ngram");
            grow["value"] = util::json_string(ngrams[i]);
            provenance_rows.push_back(util::json_object(std::move(grow)));
        }
    }

    util::JsonObject parse;
    parse["resolution"] = util::json_string("bounded_recursive_unfold_on_miss");
    parse["execution_authority"] = util::json_string("canonical_kuhul_phase_algebra");
    parse["recognition_only"] = util::json_bool(true);
    parse["atoms"] = util::json_array(std::move(atom_rows));
    parse["relations"] = util::json_array(std::move(relation_rows));
    parse["books"] = util::json_array(std::move(book_rows));
    parse["candidates"] = util::json_array(std::move(candidate_rows));
    parse["unresolved"] = util::json_array(std::move(unresolved_rows));
    parse["provenance"] = util::json_array(std::move(provenance_rows));

    util::JsonObject obj;
    obj["ok"] = util::json_bool(true);
    obj["source"] = util::json_string("fold-graph-parser");
    obj["status"] = util::json_string("fold_graph_parse_ready");
    obj["response"] = util::json_string("Fold graph parse generated.");
    obj["query"] = util::json_string(q);
    obj["fold_graph_parse"] = util::json_object(std::move(parse));
    return util::json_object(std::move(obj));
}

static PromptPersistenceSummary persist_prompt_submission(
    const std::string& prompt_text,
    const fs::path& repo_root,
    const DictionaryLexicon* dictionary_lexicon,
    micronaut::Factory* factory,
    kuhul::GramStore* gram_store)
{
    PromptPersistenceSummary summary;
    if (!gram_store) return summary;
    constexpr size_t kMaxAnalysisChars = 200000;
    const bool factory_available = (factory != nullptr);
    if (factory_available) factory->begin_bulk_update();

    summary.prompt_key = next_prompt_archive_key();

    gram_store->gramWrite(
        summary.prompt_key,
        "stored_in_factory len=" + std::to_string(prompt_text.size()),
        "Pop",
        "predication_sequence",
        "definitions_and_relations");
    if (factory) factory->record_key(summary.prompt_key, prompt_text);

    std::string prompt_analysis_text = prompt_text;
    if (prompt_analysis_text.size() > kMaxAnalysisChars)
        prompt_analysis_text.resize(kMaxAnalysisChars);

    const auto prompt_ngrams = build_learning_ngrams(prompt_analysis_text, 48);
    summary.prompt_ngram_count = prompt_ngrams.size();
    for (size_t i = 0; i < prompt_ngrams.size(); ++i) {
        if (factory) factory->record_key("prompt_ngram:" + prompt_ngrams[i], summary.prompt_key);
    }

    auto prompt_tokens = micronaut::Mx2LmNGram::tokenize(lower_copy(prompt_analysis_text));
    if (prompt_tokens.size() > 512) prompt_tokens.resize(512);
    const auto prompt_taxonomy = micronaut::Mx2LmNGram::classify_gram_taxonomy(prompt_tokens);
    {
        const auto channels = record_prompt_gram_channels(
            summary.prompt_key, prompt_taxonomy, factory, gram_store, prompt_tokens, 32);
        summary.prompt_unigram_count = channels.unigrams;
        summary.prompt_bigram_count = channels.bigrams;
        summary.prompt_trigram_count = channels.trigrams;
        summary.prompt_quadgram_count = channels.quadgrams;
        summary.prompt_coarse_gram_count = channels.coarse_grams;
        summary.prompt_function_gram_count = channels.function_grams;
        summary.prompt_tool_gram_count = channels.tool_grams;
    }
    summary.prompt_lexicon_count = record_lexicon_unigram_keys(
        prompt_tokens,
        summary.prompt_key,
        dictionary_lexicon,
        factory,
        gram_store);
    summary.prompt_similarity_count = record_dictionary_similarity_keys(
        prompt_tokens,
        summary.prompt_key,
        dictionary_lexicon,
        factory,
        gram_store);
    auto prompt_rel = micronaut::Mx2LmNGram::infer_relations(prompt_tokens);
    if (prompt_rel.size() > 96) prompt_rel.resize(96);
    summary.prompt_assoc_count = prompt_rel.size();
    for (size_t i = 0; i < prompt_rel.size(); ++i) {
        const auto& rel = prompt_rel[i];
        const std::string assoc = rel.from + "|" + rel.relation + "|" + rel.to;
        if (factory) factory->record_key("word_assoc:" + assoc, summary.prompt_key);
    }
    const auto prompt_atoms = derive_semantic_atom_grounding(prompt_tokens);
    summary.prompt_atom_count = prompt_atoms.size();
    record_semantic_atom_grounding(summary.prompt_key, prompt_atoms, factory, gram_store);

    const auto path_candidates = extract_path_candidates_from_prompt(prompt_text);
    summary.file_candidate_count = path_candidates.size();
    size_t ingested_index = 0;
    for (const auto& candidate : path_candidates) {
        auto resolved = resolve_prompt_path_candidate(candidate, repo_root);
        if (!resolved) continue;

        std::string path_string = resolved->string();
        std::replace(path_string.begin(), path_string.end(), '/', '\\');

        const auto ext = lower_copy(resolved->extension().string());
        const std::string file_ref_key =
            summary.prompt_key + ":file_ref:" + std::to_string(++ingested_index);
        gram_store->gramWrite(
            file_ref_key,
            path_string,
            "Pop",
            "predication_sequence",
            "definitions_and_relations");

        if (is_prompt_pdf_extension(ext)) {
            summary.file_pdf_pending_count++;
            if (factory) factory->record_key("file_pdf:" + path_string, "pdf_scraper_required");
            gram_store->gramWrite(
                summary.prompt_key + ":file_pdf:" + std::to_string(summary.file_pdf_pending_count),
                "SCRAPER_REQUIRED " + path_string,
                "Wo",
                "predication_sequence",
                "definitions_and_relations");
            continue;
        }
        if (!is_prompt_text_extension(ext)) continue;

        std::string raw_text;
        std::string err;
        if (!read_text_file(*resolved, &raw_text, &err) || raw_text.empty()) continue;

        if (factory) {
            factory->record_key("file:" + path_string, raw_text);
            factory->record_key("file_name:" + resolved->filename().string(), raw_text);
        }

        std::string semantic_text = raw_text;
        if (ext == ".html" || ext == ".htm")
            semantic_text = strip_html_markup(raw_text);
        if (semantic_text.empty()) semantic_text = raw_text;
        if (semantic_text.size() > kMaxAnalysisChars)
            semantic_text.resize(kMaxAnalysisChars);

        auto file_grams = build_learning_ngrams(semantic_text, 64);
        for (const auto& gram : file_grams) {
            if (factory) factory->record_key("file_ngram:" + gram, path_string);
        }

        auto file_tokens = micronaut::Mx2LmNGram::tokenize(lower_copy(semantic_text));
        if (file_tokens.size() > 512) file_tokens.resize(512);
        const std::string file_scope_key =
            summary.prompt_key + ":file:" + std::to_string(summary.file_ingested_count + 1);
        const auto file_taxonomy = micronaut::Mx2LmNGram::classify_gram_taxonomy(file_tokens);
        {
            const auto channels = record_prompt_gram_channels(
                file_scope_key, file_taxonomy, factory, gram_store, file_tokens, 24);
            summary.file_unigram_count += channels.unigrams;
            summary.file_bigram_count += channels.bigrams;
            summary.file_trigram_count += channels.trigrams;
            summary.file_quadgram_count += channels.quadgrams;
            summary.file_coarse_gram_count += channels.coarse_grams;
            summary.file_function_gram_count += channels.function_grams;
            summary.file_tool_gram_count += channels.tool_grams;
        }
        summary.file_lexicon_count += record_lexicon_unigram_keys(
            file_tokens,
            file_scope_key,
            dictionary_lexicon,
            factory,
            gram_store);
        summary.file_similarity_count += record_dictionary_similarity_keys(
            file_tokens,
            file_scope_key,
            dictionary_lexicon,
            factory,
            gram_store);
        auto file_rel = micronaut::Mx2LmNGram::infer_relations(file_tokens);
        if (file_rel.size() > 96) file_rel.resize(96);
        for (const auto& rel : file_rel) {
            const std::string assoc = rel.from + "|" + rel.relation + "|" + rel.to;
            if (factory) factory->record_key("file_assoc:" + assoc, path_string);
        }
        const auto file_atoms = derive_semantic_atom_grounding(file_tokens);
        summary.file_atom_count += file_atoms.size();
        record_semantic_atom_grounding(
            file_scope_key,
            file_atoms,
            factory,
            gram_store);

        summary.file_ingested_count++;
    }

    if (factory_available) factory->end_bulk_update();
    return summary;
}

static bool is_all_upper_ascii(const std::string& token) {
    bool has_alpha = false;
    for (unsigned char c : token) {
        if (!std::isalpha(c)) continue;
        has_alpha = true;
        if (!std::isupper(c)) return false;
    }
    return has_alpha;
}

static bool is_title_ascii(const std::string& token) {
    bool saw_alpha = false;
    bool first_alpha = true;
    for (unsigned char c : token) {
        if (!std::isalpha(c)) continue;
        saw_alpha = true;
        if (first_alpha) {
            if (!std::isupper(c)) return false;
            first_alpha = false;
        } else if (!std::islower(c)) {
            return false;
        }
    }
    return saw_alpha;
}

static std::string apply_token_case_pattern(
    const std::string& source_token,
    std::string replacement_lower)
{
    if (replacement_lower.empty()) return replacement_lower;
    if (is_all_upper_ascii(source_token)) {
        std::transform(replacement_lower.begin(), replacement_lower.end(), replacement_lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return replacement_lower;
    }
    if (is_title_ascii(source_token)) {
        replacement_lower[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(replacement_lower[0])));
        return replacement_lower;
    }
    return replacement_lower;
}

static util::JsonPtr synonym_phase_payload(
    const std::string& text,
    const DictionaryLexicon* dictionary_lexicon)
{
    const auto source = trim_copy(text);
    if (source.empty()) return util::json_null();

    size_t word_count = 0;
    for (size_t i = 0; i < source.size();) {
        if (!std::isalpha(static_cast<unsigned char>(source[i]))) { ++i; continue; }
        ++word_count;
        size_t j = i + 1;
        while (j < source.size()) {
            const unsigned char c = static_cast<unsigned char>(source[j]);
            if (!(std::isalpha(c) || c == '\'' || c == '-')) break;
            ++j;
        }
        i = j;
    }
    const size_t max_replacements = std::min<size_t>(8, std::max<size_t>(1, word_count / 3));

    std::string rewritten;
    rewritten.reserve(source.size() + 32);
    util::JsonArray substitutions;
    size_t replacements = 0;

    for (size_t i = 0; i < source.size();) {
        const unsigned char c = static_cast<unsigned char>(source[i]);
        if (!std::isalpha(c)) {
            rewritten.push_back(static_cast<char>(c));
            ++i;
            continue;
        }

        size_t j = i + 1;
        while (j < source.size()) {
            const unsigned char cc = static_cast<unsigned char>(source[j]);
            if (!(std::isalpha(cc) || cc == '\'' || cc == '-')) break;
            ++j;
        }

        const std::string token = source.substr(i, j - i);
        const auto token_norm = best_dictionary_word_token(token);
        const auto candidates = dictionary_synonyms_for_word(dictionary_lexicon, token_norm, 4);

        bool replaced = false;
        if (!token_norm.empty() && !candidates.empty() && replacements < max_replacements) {
            for (const auto& cand : candidates) {
                if (cand == token_norm) continue;
                const auto replacement = apply_token_case_pattern(token, cand);
                rewritten += replacement;
                ++replacements;
                replaced = true;

                util::JsonObject row;
                row["from"] = util::json_string(token);
                row["to"] = util::json_string(replacement);
                row["candidates"] = util::json_array(json_string_array(candidates));
                row["selection_phase"] = util::json_string("Yax");
                substitutions.push_back(util::json_object(std::move(row)));
                break;
            }
        }

        if (!replaced) rewritten += token;
        i = j;
    }

    const double lexical_preservation =
        (word_count == 0) ? 1.0 :
        std::max(0.0, 1.0 - (static_cast<double>(replacements) / static_cast<double>(word_count)));
    const bool accepted = (replacements == 0) || lexical_preservation >= 0.55;
    const std::string emitted = accepted ? rewritten : source;

    util::JsonObject phase_contract;
    phase_contract["Pop"] = util::json_string("gramRead + lexical_activate_candidates over lexical/morph/syntax units.");
    phase_contract["Wo"] = util::json_string("state_and_constraints + unfold_linguistic_resolution over dictionary equivalence candidates.");
    phase_contract["Yax"] = util::json_string("admit_neighborhood and select contextual substitutions.");
    phase_contract["Sek"] = util::json_string("compose transformed candidate sentence.");
    phase_contract["Ch'en"] = util::json_string("assert_and_verify_candidate with semantic/relation/syntax invariance gates.");
    phase_contract["Xul"] = util::json_string("emit_answer on accepted lexical-equivalence sentence.");

    util::JsonObject obj;
    obj["ok"] = util::json_bool(true);
    obj["source"] = util::json_string("dictionary-lexical-equivalence");
    obj["status"] = util::json_string("synonym_alternate_ready");
    obj["query"] = util::json_string(source);
    obj["response"] = util::json_string(emitted);
    obj["alternate_sentence"] = util::json_string(emitted);
    obj["replacements"] = util::json_number(static_cast<double>(replacements));
    obj["word_count"] = util::json_number(static_cast<double>(word_count));
    obj["lexical_preservation"] = util::json_number(lexical_preservation);
    obj["semantic_invariance_gate"] = util::json_bool(accepted);
    obj["phase_contract"] = util::json_object(std::move(phase_contract));
    obj["substitutions"] = util::json_array(std::move(substitutions));
    return util::json_object(std::move(obj));
}

static util::JsonPtr app_wizard_payload(
    const std::string& text,
    const std::unordered_set<std::string>& known_app_types,
    std::string* selected_type_out = nullptr,
    bool* used_existing_type_out = nullptr)
{
    if (selected_type_out) *selected_type_out = "";
    if (used_existing_type_out) *used_existing_type_out = false;

    const auto t = trim_copy(text);
    if (t.empty()) return util::json_null();

    const auto low = lower_copy(t);
    if (low.rfind("mx2lm:", 0) == 0 || low.rfind("@mx2lm", 0) == 0 || low.find("<mx2lm>") != std::string::npos)
        return util::json_null();

    std::string payload = t;
    std::string payload_low = low;
    bool explicit_marker = false;

    auto pick_after = [&](const std::string& marker, bool colon_after_at = false) {
        auto pos = payload_low.find(marker);
        if (pos == std::string::npos) return false;
        auto start = pos + marker.size();
        auto raw = trim_copy(payload.substr(start));
        if (colon_after_at && !raw.empty() && raw[0] == ':')
            raw = trim_copy(raw.substr(1));
        if (raw.empty()) return false;
        payload = raw;
        payload_low = lower_copy(payload);
        return true;
    };

    auto open = payload_low.find("<app>");
    if (open != std::string::npos) {
        auto start = open + 5;
        auto close = payload_low.find("</app>", start);
        if (close == std::string::npos) close = payload.size();
        payload = trim_copy(payload.substr(start, close - start));
        payload_low = lower_copy(payload);
        explicit_marker = true;
    } else if (payload_low.rfind("app:", 0) == 0) {
        explicit_marker = pick_after("app:");
    } else if (payload_low.rfind("@app", 0) == 0) {
        explicit_marker = pick_after("@app", true);
    } else {
        if (pick_after("app:")) explicit_marker = true;
        else if (pick_after("@app", true)) explicit_marker = true;
    }

    const auto toks = micronaut::Mx2LmNGram::tokenize(payload_low);
    const bool app_intent =
        explicit_marker ||
        has_token(toks, "app") ||
        has_token(toks, "application") ||
        has_token(toks, "website") ||
        has_token(toks, "platform");
    if (!app_intent) return util::json_null();

    const bool wants_options =
        has_token(toks, "option") || has_token(toks, "options") ||
        has_token(toks, "pick") || has_token(toks, "choose") ||
        has_token(toks, "types") || has_token(toks, "catalog");

    const auto& cat = app_wizard_catalog();
    const AppWizardTemplate* selected = nullptr;
    int best_score = 0;

    for (const auto& tdef : cat) {
        int score = 0;
        for (const auto& alias : tdef.aliases) {
            if (contains_phrase_ci(payload_low, lower_copy(alias))) {
                score += (alias.find(' ') == std::string::npos) ? 1 : 2;
            }
        }
        if (score > best_score) {
            best_score = score;
            selected = &tdef;
        }
    }

    const bool dog_domain = has_token(toks, "dog") || has_token(toks, "dogs") ||
                            has_token(toks, "canine") || has_token(toks, "pet");
    const bool walking_domain = has_token(toks, "walk") || has_token(toks, "walking") ||
                                has_token(toks, "walker") || has_token(toks, "leash");
    const bool dog_walking_request = dog_domain && walking_domain;
    const bool has_app_word = has_token(toks, "app") || has_token(toks, "application") ||
                              has_token(toks, "website") || has_token(toks, "platform");
    const bool has_build_signal =
        has_token(toks, "build") || has_token(toks, "create") || has_token(toks, "make") ||
        has_token(toks, "generate") || has_token(toks, "show") || has_token(toks, "pick") ||
        has_token(toks, "choose") || has_token(toks, "produce") || has_token(toks, "compose");
    if (!explicit_marker && !wants_options && best_score == 0 && !dog_walking_request &&
        !(has_app_word && has_build_signal)) {
        return util::json_null();
    }

    std::string selection_rule = "catalog_match";
    std::string selected_type = selected ? selected->id : "blog";
    bool used_existing_type = false;
    bool derived_overlay = false;

    if (!selected) {
        selected = app_wizard_find_template("blog");
        selection_rule = "fallback_blog";
    }

    if (dog_walking_request) {
        selected = app_wizard_find_template("blog");
        selected_type = "dog-walking";
        used_existing_type = (known_app_types.find(selected_type) != known_app_types.end());
        selection_rule = used_existing_type ? "known_domain_template" : "derived_from_blog";
        derived_overlay = !used_existing_type;
    }

    if (!selected) return util::json_null();
    if (selected_type == "blog" && selected->id != "blog")
        selected_type = selected->id;

    auto semantic_keywords = semantic_keywords_from_tokens(toks);
    if (semantic_keywords.empty()) semantic_keywords.push_back(selected_type);

    std::string context_phrase;
    if (dog_walking_request) {
        context_phrase = "dog walking services";
    } else {
        context_phrase = join_words(semantic_keywords, 4);
        if (context_phrase.empty()) context_phrase = selected->id + " application";
    }
    const auto context_title = to_title_case(context_phrase);
    const auto app_type_title = to_title_case(selected_type);
    const auto page_title = context_title + " | " + (app_type_title.empty() ? std::string("App") : app_type_title);
    const auto meta_description =
        "Runtime-composed " + selected_type +
        " application for " + context_phrase +
        " using semantic-slot routing and policy-checked atomic blocks.";

    std::vector<std::string> h2_slots;
    std::vector<std::string> paragraphs;
    if (dog_walking_request) {
        h2_slots = {"Book Walks", "Track Routes", "Manage Pet Profiles"};
        paragraphs.push_back(
            "This app coordinates walkers, pets, and schedules through reusable runtime blocks.");
        paragraphs.push_back(
            "Mx2LM maps intent into deterministic slots and the runtime binds those slots to dog-walking UI macros.");
    } else if (selected->id == "blog") {
        h2_slots = {"Publish Articles", "Organize Topics", "Engage Comments"};
        paragraphs.push_back(
            "This blog blueprint structures writing, moderation, and publishing around reusable block contracts.");
        paragraphs.push_back(
            "Mx2LM routes content semantics into slots while XCFE/K'UHUL governs legal composition and emit phases.");
    } else {
        h2_slots = {"Overview", "Core Features", "Workflow"};
        paragraphs.push_back(
            "This application blueprint is assembled from symbolic recipes and composable interface blocks.");
        paragraphs.push_back(
            "Mx2LM provides semantic routing for content slots, while orchestrator tracks enforce deterministic execution.");
    }

    std::vector<std::string> list_items;
    for (size_t i = 0; i < selected->blocks.size() && i < 3; ++i) {
        list_items.push_back("UI block: " + symbol_to_label(selected->blocks[i]));
    }
    for (size_t i = 0; i < selected->modules.size() && i < 2; ++i) {
        list_items.push_back("Capability: " + symbol_to_label(selected->modules[i]));
    }
    if (dog_walking_request) {
        list_items.push_back("Domain overlay: dog profile cards, route maps, and walk scheduling.");
    }

    util::JsonObject obj;
    obj["ok"] = util::json_bool(true);
    obj["source"] = util::json_string("app-wizard-base");
    obj["catalog_source"] = util::json_string("github:cannaseedus-bot/App-Wizard");
    obj["invoke"] = util::json_string("@app");
    obj["response"] = util::json_string("<app>{intent}</app>");
    obj["mode"] = util::json_string(wants_options ? "guided-selection" : "runtime-build");
    obj["request"] = util::json_string(payload);
    obj["selected_type"] = util::json_string(selected_type);
    obj["selection_rule"] = util::json_string(selection_rule);
    obj["used_existing_type"] = util::json_bool(used_existing_type);
    obj["base_track"] = util::json_string(selected->base_track);

    util::JsonObject recipe;
    recipe["app"] = util::json_string(selected->app_symbol);
    recipe["layout"] = util::json_string(selected->layout_symbol);
    recipe["skin"] = util::json_string(selected->skin_symbol);
    recipe["blocks"] = util::json_array(json_string_array(selected->blocks));
    recipe["code"] = util::json_array(json_string_array(selected->modules));

    util::JsonObject blueprint;
    blueprint["template"] = util::json_string(selected->id);
    blueprint["base_track"] = util::json_string(selected->base_track);
    blueprint["recipe"] = util::json_object(std::move(recipe));
    blueprint["css_macros"] = util::json_array(json_string_array(selected->css_macros));
    blueprint["mx2lm_role"] = util::json_string("semantic_router");
    blueprint["execution_authority"] = util::json_string("XCFE/K'UHUL");

    util::JsonObject semantic_slots;
    semantic_slots["title"] = util::json_string(page_title);
    semantic_slots["meta_description"] = util::json_string(meta_description);
    semantic_slots["h1"] = util::json_string(dog_walking_request ? "Dog Walking Service Hub" : (context_title + " App"));
    semantic_slots["h2"] = util::json_array(json_string_array(h2_slots));
    semantic_slots["paragraphs"] = util::json_array(json_string_array(paragraphs));
    semantic_slots["list_items"] = util::json_array(json_string_array(list_items));
    semantic_slots["keywords"] = util::json_array(json_string_array(semantic_keywords));
    blueprint["semantic_slots"] = util::json_object(std::move(semantic_slots));

    util::JsonObject slot_bindings;
    slot_bindings["title"] = util::json_string("⟁HTMLTITLE");
    slot_bindings["meta_description"] = util::json_string("⟁HTMLMETA");
    slot_bindings["h1"] = util::json_string("⟁H1");
    slot_bindings["h2"] = util::json_string("⟁H2");
    slot_bindings["paragraphs"] = util::json_string("⟁P");
    slot_bindings["list_items"] = util::json_string("⟁LISTITEM");
    slot_bindings["container"] = util::json_string(dog_walking_request ? "⟁DOGWALKAPP" : "⟁BLOG");
    blueprint["atomic_bindings"] = util::json_object(std::move(slot_bindings));

    const auto media_query = dog_walking_request
        ? "dog walking service app"
        : (context_phrase + " " + selected->id + " app");

    util::JsonObject media_research;
    media_research["mode"] = util::json_string("web-research-plan");
    media_research["delegate_track"] = util::json_string("Librarian");
    media_research["policy"] = util::json_string("royalty-free-only");
    media_research["license_gate"] = util::json_string("allow:CC0|CC-BY|Unsplash|Pexels|Pixabay");
    media_research["attribution_tracking"] = util::json_string("required-when-license-demands");

    util::JsonArray providers;
    auto push_provider = [&](const std::string& id,
                             const std::string& name,
                             const std::string& license,
                             const std::string& attribution,
                             const std::string& suffix) {
        util::JsonObject row;
        row["id"] = util::json_string(id);
        row["name"] = util::json_string(name);
        row["license"] = util::json_string(license);
        row["attribution"] = util::json_string(attribution);
        row["query"] = util::json_string(media_query + " " + suffix);
        row["max_results"] = util::json_number(6.0);
        providers.push_back(util::json_object(std::move(row)));
    };

    push_provider("openverse", "Openverse", "CC0/CC-BY", "required-if-cc-by", "hero photo");
    push_provider("wikimedia", "Wikimedia Commons", "varied-open", "follow-source-license", "illustration");
    push_provider("pexels", "Pexels", "Pexels License", "recommended", "app ui background");
    push_provider("pixabay", "Pixabay", "Pixabay License", "not-required-but-recommended", "feature image");
    push_provider("unsplash", "Unsplash", "Unsplash License", "recommended", "cover photo");
    media_research["providers"] = util::json_array(std::move(providers));

    util::JsonArray media_slots;
    auto push_media_slot = [&](const std::string& slot_name,
                               const std::string& atomic_block,
                               const std::string& query,
                               const std::string& aspect,
                               bool require_attribution) {
        util::JsonObject row;
        row["slot"] = util::json_string(slot_name);
        row["atomic_block"] = util::json_string(atomic_block);
        row["query"] = util::json_string(query);
        row["aspect"] = util::json_string(aspect);
        row["require_attribution"] = util::json_bool(require_attribution);
        media_slots.push_back(util::json_object(std::move(row)));
    };

    push_media_slot("hero_image",
                    dog_walking_request ? "⟁DOGWALKROUTE" : "⟁BLOGPOSTCARD",
                    media_query + " hero",
                    "16:9",
                    true);
    push_media_slot("card_image",
                    dog_walking_request ? "⟁DOGWALKCARD" : "⟁WIDGETCARD",
                    media_query + " service card",
                    "4:3",
                    true);
    push_media_slot("list_image",
                    "⟁WIDGET",
                    media_query + " details",
                    "1:1",
                    false);
    media_research["slots"] = util::json_array(std::move(media_slots));
    blueprint["media_research"] = util::json_object(std::move(media_research));

    if (dog_walking_request) {
        std::vector<std::string> dog_blocks = {
            "Ω_DOG_PROFILE_CARD", "Ω_ROUTE_MAP", "Ω_WALK_SCHEDULER", "Ω_BOOKING_PANEL"
        };
        std::vector<std::string> dog_css = {"⟁DOGWALKAPP", "⟁DOGWALKCARD", "⟁DOGWALKROUTE", "⟁DOGWALKSCHED"};
        blueprint["domain_goal"] = util::json_string("dog-walking");
        blueprint["domain_blocks"] = util::json_array(json_string_array(dog_blocks));
        blueprint["domain_css_macros"] = util::json_array(json_string_array(dog_css));
        blueprint["domain_overlay_mode"] = util::json_string(derived_overlay ? "new-overlay" : "reused-known");
    }

    obj["blueprint"] = util::json_object(std::move(blueprint));
    obj["options"] = util::json_array(app_wizard_options_array());

    util::JsonArray watch_stream;
    auto push_step = [&](const std::string& phase, const std::string& op, const std::string& summary) {
        util::JsonObject step;
        step["phase"] = util::json_string(phase);
        step["op"] = util::json_string(op);
        step["summary"] = util::json_string(summary);
        watch_stream.push_back(util::json_object(std::move(step)));
    };
    push_step("Pop",   "capture_request", "Read user app intent and constraints");
    push_step("Wo",    "select_template", "Choose App-Wizard base recipe");
    push_step("Yax",   "map_semantic_slots", "Map title/meta/h1/h2/p/list semantics to atomic slots");
    push_step("Sek",   "compose_blocks",  "Assemble block graph, CSS macro kit, and media slot plan");
    push_step("Ch'en", "verify_media_policy", "Enforce royalty-free license and attribution gates");
    push_step("Xul",   "commit_plan",     "Store selected type and committed runtime block blueprint");
    obj["watch_stream"] = util::json_array(std::move(watch_stream));

    if (selected_type_out) *selected_type_out = selected_type;
    if (used_existing_type_out) *used_existing_type_out = used_existing_type;
    return util::json_object(std::move(obj));
}

struct PopularIntentSpec {
    std::string intent_key;
    std::string invoke_marker;
    std::string keyword;
    std::string delegate_track;
    std::string retrieval_track;
    std::string verifier_track;
    std::string wrapper_tag;
    std::string op;
};

static util::JsonPtr popular_intent_payload(const std::string& text) {
    const auto t = trim_copy(text);
    if (t.empty()) return util::json_null();

    const auto low = lower_copy(t);
    static const std::vector<PopularIntentSpec> kPopular = {
        {"SUMMARIZE:",  "@summarize",  "summarize",  "reasoner",        "",          "INFER-µ", "<summarize>{output}</summarize>", "summarize"},
        {"EXPLAIN:",    "@explain",    "explain",    "reasoner",        "Librarian", "INFER-µ", "<explain>{output}</explain>",     "explain"},
        {"REWRITE:",    "@rewrite",    "rewrite",    "CREATE-µ",        "",          "",        "<rewrite>{output}</rewrite>",     "rewrite"},
        {"TRANSLATE:",  "@translate",  "translate",  "CREATE-µ",        "Librarian", "",        "<translate>{output}</translate>", "translate"},
        {"BRAINSTORM:", "@brainstorm", "brainstorm", "CREATE-µ",        "",          "",        "<brainstorm>{output}</brainstorm>","brainstorm"},
        {"PLAN:",       "@plan",       "plan",       "Coordinator",     "",          "INFER-µ", "<plan>{output}</plan>",           "plan"},
        {"CODE:",       "@code",       "code",       "coder",           "",          "INFER-µ", "<code>{output}</code>",           "implement"},
        {"DEBUG:",      "@debug",      "debug",      "coder",           "",          "INFER-µ", "<debug>{output}</debug>",         "debug"},
        {"MATH:",       "@math",       "math",       "MATH-µ",          "",          "INFER-µ", "<math>{output}</math>",           "math-query"},
        {"EMAIL:",      "@email",      "email",      "CREATE-µ",        "",          "",        "<email>{output}</email>",         "email-draft"},
        {"MEETING:",    "@meeting",    "meeting",    "reasoner",        "",          "Coordinator", "<meeting>{output}</meeting>", "meeting-notes"}
    };

    auto pick_payload = [&](const PopularIntentSpec& spec, std::string* out_payload) {
        const auto k = lower_copy(spec.intent_key);
        const auto m = lower_copy(spec.invoke_marker);
        const auto kw = lower_copy(spec.keyword);

        // key-style prefix: e.g., SUMMARIZE: ...
        if (low.rfind(k, 0) == 0) {
            *out_payload = trim_copy(t.substr(spec.intent_key.size()));
            return true;
        }
        // invoke-style prefix: e.g., @summarize ...
        if (low.rfind(m, 0) == 0) {
            auto raw = trim_copy(t.substr(spec.invoke_marker.size()));
            if (!raw.empty() && raw[0] == ':') raw = trim_copy(raw.substr(1));
            *out_payload = raw;
            return true;
        }
        // natural prefix: e.g., summarize this text
        if (low.rfind(kw + " ", 0) == 0) {
            *out_payload = trim_copy(t.substr(spec.keyword.size()));
            return true;
        }
        return false;
    };

    for (const auto& spec : kPopular) {
        std::string payload;
        if (!pick_payload(spec, &payload)) continue;
        if (payload.empty()) payload = t;

        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["source"] = util::json_string("intent-key");
        obj["status"] = util::json_string("intent_plan_ready");
        obj["intent_key"] = util::json_string(spec.intent_key);
        obj["invoke"] = util::json_string(spec.invoke_marker);
        obj["mode"] = util::json_string("guided-intent");
        obj["op"] = util::json_string(spec.op);
        obj["delegate_track"] = util::json_string(spec.delegate_track);
        if (!spec.retrieval_track.empty())
            obj["retrieval_track"] = util::json_string(spec.retrieval_track);
        if (!spec.verifier_track.empty())
            obj["verifier_track"] = util::json_string(spec.verifier_track);
        obj["query"] = util::json_string(payload);
        obj["response"] = util::json_string(spec.wrapper_tag);

        util::JsonArray watch_stream;
        auto push_step = [&](const std::string& phase,
                             const std::string& op,
                             const std::string& summary) {
            util::JsonObject row;
            row["phase"] = util::json_string(phase);
            row["op"] = util::json_string(op);
            row["summary"] = util::json_string(summary);
            watch_stream.push_back(util::json_object(std::move(row)));
        };
        push_step("Pop", "capture_intent", "Capture normalized intent and payload.");
        push_step("Wo", "route_delegate", "Route to the primary micronaut delegate.");
        push_step("Yax", "compose_plan", "Compose structured output plan.");
        push_step("Sek", "execute_transform", "Run the requested transformation.");
        push_step("Ch'en", "verify_output", "Verify policy, coherence, and constraints.");
        push_step("Xul", "commit_result", "Emit and checkpoint result metadata.");
        obj["watch_stream"] = util::json_array(std::move(watch_stream));
        return util::json_object(std::move(obj));
    }

    return util::json_null();
}

static util::JsonPtr common_tool_call_payload(const std::string& text) {
    const auto t = trim_copy(text);
    if (t.empty()) return util::json_null();

    const auto low = lower_copy(t);
    const auto toks = micronaut::Mx2LmNGram::tokenize(low);
    if (has_token(toks, "timer") || has_token(toks, "countdown")) {
        return util::json_null();
    }
    auto extract_payload = [&](const std::string& key,
                               const std::string& invoke,
                               const std::string& keyword,
                               std::string* out) {
        const auto k = lower_copy(key);
        const auto m = lower_copy(invoke);
        const auto kw = lower_copy(keyword);

        if (low.rfind(k, 0) == 0) {
            *out = trim_copy(t.substr(key.size()));
            return true;
        }
        if (low.rfind(m, 0) == 0) {
            auto raw = trim_copy(t.substr(invoke.size()));
            if (!raw.empty() && raw[0] == ':') raw = trim_copy(raw.substr(1));
            *out = raw;
            return true;
        }
        if (!kw.empty() && low.rfind(kw, 0) == 0) {
            const size_t n = kw.size();
            if (low.size() == n || std::isspace((unsigned char)low[n]) ||
                low[n] == ':' || low[n] == '.' || low[n] == ',' || low[n] == '-') {
                auto raw = trim_copy(t.substr(keyword.size()));
                if (!raw.empty() &&
                    (raw[0] == ':' || raw[0] == '.' || raw[0] == ',' || raw[0] == '-'))
                    raw = trim_copy(raw.substr(1));
                *out = std::move(raw);
                return true;
            }
        }
        return false;
    };

    auto parse_nonneg = [&](const std::string& s, double fallback) {
        if (s.empty()) return fallback;
        try {
            auto v = std::stod(s);
            if (v < 0.0) return fallback;
            return v;
        } catch (...) {
            return fallback;
        }
    };

    auto make_call = [&](const std::string& surface,
                         const std::string& method,
                         util::JsonObject params,
                         const std::string& verb = "") {
        util::JsonObject call;
        call["surface"] = util::json_string(surface);
        call["method"] = util::json_string(method);
        call["params"] = util::json_object(std::move(params));
        if (!verb.empty()) call["verb"] = util::json_string(verb);
        return util::json_object(std::move(call));
    };

    std::string intent_key;
    std::string invoke_marker;
    std::string op;
    std::string payload;
    std::string delegate_track = "TOOL-µ";
    util::JsonArray calls;

    if (extract_payload("TOOLS:", "@tools", "tools", &payload) ||
        extract_payload("STATUS:", "@status", "status", &payload)) {
        intent_key = contains_phrase_ci(low, "@tools") || low.rfind("tools:", 0) == 0 ? "TOOLS:" : "STATUS:";
        invoke_marker = (intent_key == "TOOLS:") ? "@tools" : "@status";
        op = "runtime_status";
        calls.push_back(make_call("mcp", "world_state", {}));
        calls.push_back(make_call("mcp", "sandbox_list", {}));
        calls.push_back(make_call("rpc", "list_keys", {}));
    } else if (extract_payload("CUBE:", "@cube", "cube", &payload) ||
               (low.rfind("field", 0) == 0 && contains_phrase_ci(low, "cube"))) {
        intent_key = "CUBE:";
        invoke_marker = "@cube";
        op = "kcube_inspect";
        if (payload.empty()) payload = "field+cube";
        calls.push_back(make_call("mcp", "world_state", {}, "world"));
        calls.push_back(make_call("mcp", "field_dag", {}));
    } else if (extract_payload("WORLD:", "@world", "world", &payload)) {
        intent_key = "WORLD:";
        invoke_marker = "@world";
        op = "world_state";
        calls.push_back(make_call("mcp", "world_state", {}, "world"));
    } else if (extract_payload("FIELD:", "@field", "field", &payload)) {
        intent_key = "FIELD:";
        invoke_marker = "@field";
        op = "field_dag";
        calls.push_back(make_call("mcp", "field_dag", {}));
    } else if (extract_payload("FIELDSVG:", "@fieldsvg", "", &payload)) {
        intent_key = "FIELDSVG:";
        invoke_marker = "@fieldsvg";
        op = "field_dag_svg";
        calls.push_back(make_call("mcp", "field_dag_svg", {}));
    } else if (extract_payload("RELGRAPH:", "@relgraph", "", &payload)) {
        intent_key = "RELGRAPH:";
        invoke_marker = "@relgraph";
        op = "relational_graph";
        calls.push_back(make_call("mcp", "relational_graph", {}));
    } else if (extract_payload("GRAPH:", "@graph", "graph", &payload)) {
        intent_key = "GRAPH:";
        invoke_marker = "@graph";
        op = "graph_cognition";
        calls.push_back(make_call("mcp", "relational_graph", {}));
        util::JsonObject ip;
        ip["include_phase_folds"] = util::json_bool(true);
        calls.push_back(make_call("mcp", "incidence_matrix", std::move(ip)));
        if (!payload.empty()) {
            util::JsonObject p;
            p["query"] = util::json_string(payload);
            calls.push_back(make_call("rpc", "fold_query", std::move(p), "fold"));
        }
    } else if (extract_payload("SEMFOLD:", "@semfold", "semfold", &payload)) {
        intent_key = "SEMFOLD:";
        invoke_marker = "@semfold";
        op = "semantic_fold";
        util::JsonObject p;
        const auto plow = lower_copy(payload);
        const bool no_folds =
            contains_phrase_ci(plow, "relations-only") ||
            contains_phrase_ci(plow, "relations_only") ||
            contains_phrase_ci(plow, "no-folds") ||
            contains_phrase_ci(plow, "no_folds");
        p["include_phase_folds"] = util::json_bool(!no_folds);

        auto set_pair = [&](const std::string& a, const std::string& b) {
            auto aa = trim_copy(a);
            auto bb = trim_copy(b);
            if (aa.empty() || bb.empty()) return false;
            p["node_a"] = util::json_string(aa);
            p["node_b"] = util::json_string(bb);
            return true;
        };

        bool pair_set = false;
        if (!payload.empty()) {
            auto pos_vs = plow.find(" vs ");
            if (pos_vs != std::string::npos)
                pair_set = set_pair(payload.substr(0, pos_vs), payload.substr(pos_vs + 4));
        }
        if (!pair_set && !payload.empty()) {
            for (char sep : {'|', ','}) {
                auto pos = payload.find(sep);
                if (pos != std::string::npos) {
                    pair_set = set_pair(payload.substr(0, pos), payload.substr(pos + 1));
                    if (pair_set) break;
                }
            }
        }
        if (!pair_set && !payload.empty())
            p["node"] = util::json_string(trim_copy(payload));
        calls.push_back(make_call("mcp", "semantic_fold", std::move(p)));
    } else if (extract_payload("INCIDENCE:", "@incidence", "incidence", &payload)) {
        intent_key = "INCIDENCE:";
        invoke_marker = "@incidence";
        op = "incidence_matrix";
        util::JsonObject p;
        const auto plow = lower_copy(payload);
        const bool no_folds =
            contains_phrase_ci(plow, "relations-only") ||
            contains_phrase_ci(plow, "relations_only") ||
            contains_phrase_ci(plow, "no-folds") ||
            contains_phrase_ci(plow, "no_folds");
        p["include_phase_folds"] = util::json_bool(!no_folds);
        calls.push_back(make_call("mcp", "incidence_matrix", std::move(p)));
    } else if (extract_payload("FOLD:", "@fold", "fold", &payload)) {
        intent_key = "FOLD:";
        invoke_marker = "@fold";
        op = "fold_query";
        if (payload.empty()) payload = "status";
        util::JsonObject p;
        p["query"] = util::json_string(payload);
        calls.push_back(make_call("rpc", "fold_query", std::move(p), "fold"));
    } else if (extract_payload("KEYQUERY:", "@keyquery", "", &payload)) {
        intent_key = "KEYQUERY:";
        invoke_marker = "@keyquery";
        op = "key_query";
        if (payload.empty()) payload = "status";
        util::JsonObject p;
        p["query"] = util::json_string(payload);
        calls.push_back(make_call("rpc", "key_query", std::move(p)));
    } else if (extract_payload("DISPATCH:", "@dispatch", "", &payload)) {
        intent_key = "DISPATCH:";
        invoke_marker = "@dispatch";
        op = "micronaut_dispatch";
        if (payload.empty()) payload = "status";
        util::JsonObject p;
        p["query"] = util::json_string(payload);
        calls.push_back(make_call("rpc", "micronaut_dispatch", std::move(p)));
    } else if (extract_payload("MANIFEST:", "@manifest_query", "", &payload)) {
        intent_key = "MANIFEST:";
        invoke_marker = "@manifest_query";
        op = "resolve_manifest";
        if (payload.empty()) payload = "micronaut";
        util::JsonObject p;
        p["key"] = util::json_string(payload);
        calls.push_back(make_call("rpc", "resolve_manifest", std::move(p), "manifest"));
    } else if (extract_payload("CONFIDENCE:", "@confidence", "confidence", &payload)) {
        intent_key = "CONFIDENCE:";
        invoke_marker = "@confidence";
        op = "confidence_get";
        if (payload.empty()) payload = "global";
        util::JsonObject p;
        p["id"] = util::json_string(payload);
        calls.push_back(make_call("mcp", "confidence_get", std::move(p)));
    } else if (extract_payload("REPLAY:", "@replay", "replay", &payload)) {
        intent_key = "REPLAY:";
        invoke_marker = "@replay";
        op = "jrom_replay";
        util::JsonObject p;
        p["from_seq"] = util::json_number(parse_nonneg(payload, 0.0));
        calls.push_back(make_call("mcp", "jrom_replay", std::move(p)));
    } else if (extract_payload("SANDBOX:", "@sandbox", "sandbox", &payload)) {
        intent_key = "SANDBOX:";
        invoke_marker = "@sandbox";
        op = "sandbox_inspect";
        const auto plow = lower_copy(payload);
        if (payload.empty() || plow == "list") {
            calls.push_back(make_call("mcp", "sandbox_list", {}));
        } else if (plow.rfind("violations", 0) == 0) {
            auto tail = trim_copy(payload.substr(std::string("violations").size()));
            util::JsonObject p;
            p["last_n"] = util::json_number(parse_nonneg(tail, 0.0));
            calls.push_back(make_call("mcp", "sandbox_violations", std::move(p)));
        } else {
            util::JsonObject p;
            p["id"] = util::json_string(payload);
            calls.push_back(make_call("mcp", "sandbox_query", std::move(p)));
        }
    } else if (extract_payload("INFER:", "@infer", "infer", &payload)) {
        intent_key = "INFER:";
        invoke_marker = "@infer";
        op = "infer";
        if (payload.empty()) payload = "status";
        util::JsonObject p;
        p["prompt"] = util::json_string(payload);
        p["max_tokens"] = util::json_number(8.0);
        calls.push_back(make_call("rpc", "infer", std::move(p)));
    } else if (extract_payload("LISTKEYS:", "@listkeys", "", &payload)) {
        intent_key = "LISTKEYS:";
        invoke_marker = "@listkeys";
        op = "list_keys";
        calls.push_back(make_call("rpc", "list_keys", {}));
    } else {
        return util::json_null();
    }

    util::JsonObject tools_payload;
    tools_payload["tool_calls"] = util::json_array(calls);
    const auto tools_json = util::stringify(util::json_object(std::move(tools_payload)));

    util::JsonObject obj;
    obj["ok"] = util::json_bool(true);
    obj["source"] = util::json_string("tool-key");
    obj["status"] = util::json_string("tool_call_plan_ready");
    obj["intent_key"] = util::json_string(intent_key);
    obj["invoke"] = util::json_string(invoke_marker);
    obj["mode"] = util::json_string("command-center-plan");
    obj["op"] = util::json_string(op);
    obj["delegate_track"] = util::json_string(delegate_track);
    if (!payload.empty()) obj["query"] = util::json_string(payload);
    obj["tool_tag"] = util::json_string("tools");
    obj["response"] = util::json_string("<tools>{tool_calls}</tools>");
    obj["tool_calls"] = util::json_array(calls);
    obj["tools_block"] = util::json_string("<tools>" + tools_json + "</tools>");
    obj["call_count"] = util::json_number((double)calls.size());

    util::JsonArray watch_stream;
    auto push_step = [&](const std::string& phase,
                         const std::string& step_op,
                         const std::string& summary) {
        util::JsonObject row;
        row["phase"] = util::json_string(phase);
        row["op"] = util::json_string(step_op);
        row["summary"] = util::json_string(summary);
        watch_stream.push_back(util::json_object(std::move(row)));
    };
    push_step("Pop", "capture_tool_intent", "Capture requested tool operation and payload.");
    push_step("Wo", "normalize_call_plan", "Map key to allowlisted MCP/RPC methods.");
    push_step("Yax", "assemble_tool_calls", "Build deterministic tool_calls JSON payload.");
    push_step("Sek", "execute_via_center", "Dispatch through command center <tools> block.");
    push_step("Ch'en", "verify_tool_results", "Verify tool results and policy constraints.");
    push_step("Xul", "commit_trace", "Persist execution trace for replay/evolve.");
    obj["watch_stream"] = util::json_array(std::move(watch_stream));
    return util::json_object(std::move(obj));
}

struct UtilityTimerEntry {
    std::string id;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point ends_at;
    int64_t duration_seconds = 0;
};

static std::mutex& utility_timer_mutex() {
    static std::mutex mu;
    return mu;
}

static std::vector<UtilityTimerEntry>& utility_timers() {
    static std::vector<UtilityTimerEntry> timers;
    return timers;
}

static uint64_t& utility_timer_sequence() {
    static uint64_t seq = 0;
    return seq;
}

static bool parse_number_token(const std::string& tok, double* out) {
    if (!out) return false;
    try {
        size_t idx = 0;
        const double value = std::stod(tok, &idx);
        if (idx != tok.size()) return false;
        *out = value;
        return true;
    } catch (...) {
        return false;
    }
}

static std::tm local_time_safe(std::time_t ts) {
    std::tm out{};
#if defined(_WIN32)
    localtime_s(&out, &ts);
#else
    localtime_r(&ts, &out);
#endif
    return out;
}

static std::string format_local_time(std::chrono::system_clock::time_point tp, const char* fmt) {
    const auto tt = std::chrono::system_clock::to_time_t(tp);
    const auto tm = local_time_safe(tt);
    char buf[96];
    if (std::strftime(buf, sizeof(buf), fmt, &tm) == 0) return "";
    return std::string(buf);
}

static std::string format_number_compact(double value) {
    const double rounded = std::round(value);
    if (std::fabs(value - rounded) < 1e-9) {
        return std::to_string(static_cast<long long>(rounded));
    }
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(6);
    oss << value;
    auto out = oss.str();
    auto dot = out.find('.');
    if (dot != std::string::npos) {
        while (!out.empty() && out.back() == '0') out.pop_back();
        if (!out.empty() && out.back() == '.') out.pop_back();
    }
    return out.empty() ? "0" : out;
}

static bool parse_math_operator(
    const std::vector<std::string>& toks,
    size_t index,
    std::string* op_out,
    size_t* consumed_out) {
    if (index >= toks.size() || !op_out || !consumed_out) return false;
    const auto& tok = toks[index];
    auto set_op = [&](const std::string& op, size_t consumed) {
        *op_out = op;
        *consumed_out = consumed;
        return true;
    };
    if (tok == "plus" || tok == "add" || tok == "sum") return set_op("add", 1);
    if (tok == "minus" || tok == "subtract" || tok == "less") return set_op("subtract", 1);
    if (tok == "times" || tok == "multiply" || tok == "multiplied" || tok == "x")
        return set_op("multiply", (tok == "multiplied" && index + 1 < toks.size() && toks[index + 1] == "by") ? 2 : 1);
    if (tok == "divide" || tok == "divided" || tok == "over")
        return set_op("divide", (tok == "divided" && index + 1 < toks.size() && toks[index + 1] == "by") ? 2 : 1);
    if (tok == "mod" || tok == "modulo" || tok == "remainder") return set_op("modulo", 1);
    return false;
}

static bool parse_simple_math(
    const std::vector<std::string>& toks,
    double* lhs_out,
    double* rhs_out,
    std::string* op_out) {
    if (!lhs_out || !rhs_out || !op_out) return false;
    for (size_t i = 0; i < toks.size(); ++i) {
        double lhs_value = 0.0;
        if (!parse_number_token(toks[i], &lhs_value)) continue;
        for (size_t op_i = i + 1; op_i < toks.size(); ++op_i) {
            std::string op;
            size_t consumed = 0;
            if (!parse_math_operator(toks, op_i, &op, &consumed)) continue;
            for (size_t rhs_i = op_i + consumed; rhs_i < toks.size(); ++rhs_i) {
                double rhs_value = 0.0;
                if (!parse_number_token(toks[rhs_i], &rhs_value)) continue;
                *lhs_out = lhs_value;
                *rhs_out = rhs_value;
                *op_out = op;
                return true;
            }
        }
    }
    return false;
}

static int64_t parse_timer_duration_seconds(const std::vector<std::string>& toks) {
    constexpr int64_t kDefaultSeconds = 300;
    constexpr int64_t kMaxSeconds = 30LL * 24LL * 60LL * 60LL;
    for (size_t i = 0; i < toks.size(); ++i) {
        double amount = 0.0;
        if (!parse_number_token(toks[i], &amount)) continue;
        if (amount <= 0.0) return 0;
        double multiplier = 60.0;
        if (i + 1 < toks.size()) {
            const auto& unit = toks[i + 1];
            if (unit == "s" || unit == "sec" || unit == "secs" || unit == "second" || unit == "seconds")
                multiplier = 1.0;
            else if (unit == "m" || unit == "min" || unit == "mins" || unit == "minute" || unit == "minutes")
                multiplier = 60.0;
            else if (unit == "h" || unit == "hr" || unit == "hrs" || unit == "hour" || unit == "hours")
                multiplier = 3600.0;
            else if (unit == "d" || unit == "day" || unit == "days")
                multiplier = 86400.0;
        }
        int64_t seconds = static_cast<int64_t>(std::llround(amount * multiplier));
        if (seconds < 1) seconds = 1;
        if (seconds > kMaxSeconds) seconds = kMaxSeconds;
        return seconds;
    }
    return kDefaultSeconds;
}

static util::JsonPtr style_settings_payload(
    const std::string& text,
    micronaut::Factory* style_factory = nullptr,
    kuhul::GramStore* gram_store = nullptr);

static util::JsonPtr utility_runtime_payload(
    const std::string& text,
    micronaut::Factory* style_factory = nullptr,
    kuhul::GramStore* gram_store = nullptr)
{
    const auto q = trim_copy(text);
    if (q.empty()) return util::json_null();

    const auto low = lower_copy(q);
    const auto toks = micronaut::Mx2LmNGram::tokenize(low);
    if (auto style = style_settings_payload(q, style_factory, gram_store); style && !style->is_null())
        return style;
    const auto now_tp = std::chrono::system_clock::now();
    const auto now_tt = std::chrono::system_clock::to_time_t(now_tp);
    const auto now_tm = local_time_safe(now_tt);

    const bool mentions_timer = has_token(toks, "timer") || contains_phrase_ci(low, "countdown");
    if (mentions_timer) {
        const bool has_status_word = has_token(toks, "status") || has_token(toks, "check") ||
                                     has_token(toks, "remaining") || has_token(toks, "left") ||
                                     contains_phrase_ci(low, "time left");
        const bool has_create_word = has_token(toks, "start") || has_token(toks, "set") ||
                                     has_token(toks, "create") || has_token(toks, "make") ||
                                     has_token(toks, "new");
        bool has_number = false;
        for (const auto& tok : toks) {
            double probe = 0.0;
            if (parse_number_token(tok, &probe)) {
                has_number = true;
                break;
            }
        }

        const bool wants_status = has_status_word || (!has_create_word && !has_number);
        if (wants_status) {
            util::JsonObject obj;
            obj["ok"] = util::json_bool(true);
            obj["source"] = util::json_string("utility-runtime");
            obj["status"] = util::json_string("timer_status");
            obj["op"] = util::json_string("timer_status");
            obj["delegate_track"] = util::json_string("TOOL-µ");

            std::lock_guard<std::mutex> lock(utility_timer_mutex());
            auto& timers = utility_timers();
            if (timers.empty()) {
                obj["active_count"] = util::json_number(0.0);
                obj["response"] = util::json_string("No active timers.");
                return util::json_object(std::move(obj));
            }

            util::JsonArray timer_rows;
            int active_count = 0;
            for (const auto& timer : timers) {
                const auto remaining = std::chrono::duration_cast<std::chrono::seconds>(timer.ends_at - now_tp).count();
                const bool active = remaining > 0;
                if (active) active_count++;
                util::JsonObject row;
                row["timer_id"] = util::json_string(timer.id);
                row["duration_seconds"] = util::json_number(static_cast<double>(timer.duration_seconds));
                row["remaining_seconds"] = util::json_number(static_cast<double>(std::max<int64_t>(0, remaining)));
                row["state"] = util::json_string(active ? "running" : "expired");
                row["started_at"] = util::json_string(format_local_time(timer.started_at, "%Y-%m-%dT%H:%M:%S"));
                row["ends_at"] = util::json_string(format_local_time(timer.ends_at, "%Y-%m-%dT%H:%M:%S"));
                timer_rows.push_back(util::json_object(std::move(row)));
            }
            obj["timers"] = util::json_array(std::move(timer_rows));
            obj["active_count"] = util::json_number(static_cast<double>(active_count));
            const auto& latest = timers.back();
            const auto latest_remaining =
                std::chrono::duration_cast<std::chrono::seconds>(latest.ends_at - now_tp).count();
            if (latest_remaining > 0) {
                obj["response"] = util::json_string(
                    latest.id + " has " + std::to_string(static_cast<long long>(latest_remaining)) + " seconds remaining.");
            } else {
                obj["response"] = util::json_string(latest.id + " has expired.");
            }
            return util::json_object(std::move(obj));
        }

        const int64_t duration_seconds = parse_timer_duration_seconds(toks);
        if (duration_seconds <= 0) return util::json_null();

        UtilityTimerEntry timer;
        {
            std::lock_guard<std::mutex> lock(utility_timer_mutex());
            timer.id = "timer-" + std::to_string(++utility_timer_sequence());
            timer.started_at = now_tp;
            timer.ends_at = now_tp + std::chrono::seconds(duration_seconds);
            timer.duration_seconds = duration_seconds;
            utility_timers().push_back(timer);
        }

        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["source"] = util::json_string("utility-runtime");
        obj["status"] = util::json_string("timer_started");
        obj["op"] = util::json_string("timer_create");
        obj["delegate_track"] = util::json_string("TOOL-µ");
        obj["timer_id"] = util::json_string(timer.id);
        obj["duration_seconds"] = util::json_number(static_cast<double>(duration_seconds));
        obj["started_at"] = util::json_string(format_local_time(timer.started_at, "%Y-%m-%dT%H:%M:%S"));
        obj["ends_at"] = util::json_string(format_local_time(timer.ends_at, "%Y-%m-%dT%H:%M:%S"));
        obj["response"] = util::json_string(
            timer.id + " started for " + std::to_string(static_cast<long long>(duration_seconds)) + " seconds.");
        return util::json_object(std::move(obj));
    }

    const bool asks_time = contains_phrase_ci(low, "what time") ||
                           contains_phrase_ci(low, "current time") ||
                           contains_phrase_ci(low, "time is it");
    const bool asks_today =
        contains_phrase_ci(low, "what is today") ||
        contains_phrase_ci(low, "today is what") ||
        (has_token(toks, "today") && (has_token(toks, "what") || has_token(toks, "date") || has_token(toks, "day")));
    const bool asks_first_day_of_week =
        contains_phrase_ci(low, "first day of the week") ||
        contains_phrase_ci(low, "week starts on") ||
        contains_phrase_ci(low, "week start day") ||
        contains_phrase_ci(low, "which day starts the week");
    const bool asks_date = contains_phrase_ci(low, "what date") ||
                           contains_phrase_ci(low, "current date") ||
                           contains_phrase_ci(low, "today date") ||
                           contains_phrase_ci(low, "what is the date");
    const bool asks_day = contains_phrase_ci(low, "what day") ||
                          contains_phrase_ci(low, "day is it") ||
                          has_token(toks, "weekday");
    const bool asks_month = contains_phrase_ci(low, "what month") ||
                            contains_phrase_ci(low, "current month") ||
                            contains_phrase_ci(low, "month is it") ||
                            contains_phrase_ci(low, "what is the month");
    const bool asks_year = contains_phrase_ci(low, "what year") ||
                           contains_phrase_ci(low, "current year") ||
                           contains_phrase_ci(low, "year is it") ||
                           contains_phrase_ci(low, "what is the year");
    if (asks_first_day_of_week) {
        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["source"] = util::json_string("utility-runtime");
        obj["status"] = util::json_string("clock_snapshot");
        obj["op"] = util::json_string("clock_week_start");
        obj["delegate_track"] = util::json_string("TOOL-µ");
        obj["week_standard"] = util::json_string("ISO-8601");
        obj["week_start"] = util::json_string("Monday");
        obj["alternate_common"] = util::json_string("Sunday");
        obj["response"] = util::json_string("ISO-8601 weeks start on Monday. Some calendars use Sunday.");
        return util::json_object(std::move(obj));
    }

    if (asks_time || asks_today || asks_date || asks_day || asks_month || asks_year) {
        const std::string time_local = format_local_time(now_tp, "%H:%M:%S");
        const std::string date_local = format_local_time(now_tp, "%Y-%m-%d");
        const std::string day_name = format_local_time(now_tp, "%A");
        const std::string month_name = format_local_time(now_tp, "%B");
        const int year = now_tm.tm_year + 1900;
        const int month = now_tm.tm_mon + 1;
        const int day = now_tm.tm_mday;

        std::string response = time_local;
        std::string op = "clock_snapshot";
        if (asks_today && !asks_time && !asks_date && !asks_day && !asks_month && !asks_year) {
            response = day_name + ", " + month_name + " " + std::to_string(day) + ", " + std::to_string(year);
            op = "clock_today";
        } else if (asks_day && !asks_time && !asks_date && !asks_month && !asks_year) {
            response = day_name;
            op = "clock_day";
        } else if (asks_month && !asks_time && !asks_date && !asks_day && !asks_year) {
            response = month_name;
            op = "clock_month";
        } else if (asks_year && !asks_time && !asks_date && !asks_day && !asks_month) {
            response = std::to_string(year);
            op = "clock_year";
        } else if (asks_date && !asks_time && !asks_day && !asks_month && !asks_year) {
            response = date_local;
            op = "clock_date";
        } else if (asks_time && !asks_date && !asks_day && !asks_month && !asks_year) {
            response = time_local;
            op = "clock_time";
        } else {
            response = day_name + ", " + month_name + " " + std::to_string(day) + ", " +
                       std::to_string(year) + " " + time_local;
        }

        util::JsonObject now_obj;
        now_obj["iso_local"] = util::json_string(format_local_time(now_tp, "%Y-%m-%dT%H:%M:%S"));
        now_obj["date"] = util::json_string(date_local);
        now_obj["time"] = util::json_string(time_local);
        now_obj["day_name"] = util::json_string(day_name);
        now_obj["month_name"] = util::json_string(month_name);
        now_obj["year"] = util::json_number(static_cast<double>(year));
        now_obj["month"] = util::json_number(static_cast<double>(month));
        now_obj["day"] = util::json_number(static_cast<double>(day));

        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["source"] = util::json_string("utility-runtime");
        obj["status"] = util::json_string("clock_snapshot");
        obj["op"] = util::json_string(op);
        obj["delegate_track"] = util::json_string("TOOL-µ");
        obj["now"] = util::json_object(std::move(now_obj));
        obj["response"] = util::json_string(response);
        return util::json_object(std::move(obj));
    }

    const bool asks_moon_composition =
        contains_phrase_ci(low, "what is the moon made of") ||
        contains_phrase_ci(low, "what's the moon made of") ||
        contains_phrase_ci(low, "what is moon made of") ||
        contains_phrase_ci(low, "whats the moon made of") ||
        contains_phrase_ci(low, "moon made of") ||
        contains_phrase_ci(low, "moon composition") ||
        contains_phrase_ci(low, "composition of the moon");
    const bool asks_sun_composition =
        contains_phrase_ci(low, "what is the sun made of") ||
        contains_phrase_ci(low, "what's the sun made of") ||
        contains_phrase_ci(low, "what is sun made of") ||
        contains_phrase_ci(low, "whats the sun made of") ||
        contains_phrase_ci(low, "sun made of") ||
        contains_phrase_ci(low, "sun composition") ||
        contains_phrase_ci(low, "composition of the sun");
    const bool asks_sun =
        contains_phrase_ci(low, "what is the sun") ||
        contains_phrase_ci(low, "what is a sun") ||
        contains_phrase_ci(low, "tell me about the sun") ||
        contains_phrase_ci(low, "define sun");
    const bool asks_moon =
        contains_phrase_ci(low, "what is the moon") ||
        contains_phrase_ci(low, "what is a moon") ||
        contains_phrase_ci(low, "tell me about the moon") ||
        contains_phrase_ci(low, "define moon");
    const bool mentions_kuhul =
        contains_phrase_ci(low, "k'uhul") ||
        has_token(toks, "kuhul");
    const bool asks_kuhul =
        mentions_kuhul &&
        (contains_phrase_ci(low, "what is") ||
         contains_phrase_ci(low, "define") ||
         contains_phrase_ci(low, "tell me about"));
    if (asks_sun_composition || asks_moon_composition || asks_sun || asks_moon || asks_kuhul) {
        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["source"] = util::json_string("utility-runtime");
        obj["status"] = util::json_string("knowledge_snapshot");
        obj["op"] = util::json_string(asks_kuhul ? "local_file_fact_lookup" : "fact_lookup");
        obj["delegate_track"] = util::json_string(asks_kuhul ? "utility.local_file" : "Librarian");
        if (asks_moon_composition) {
            obj["topic"] = util::json_string("moon_composition");
            obj["response"] = util::json_string(
                "The Moon is mostly silicate rock and minerals: oxygen, silicon, magnesium, iron, calcium, and aluminum. "
                "Its maria are basalt-rich, its highlands are anorthosite-rich, and permanently shadowed polar craters "
                "contain small amounts of water ice.");
        } else if (asks_sun_composition) {
            obj["topic"] = util::json_string("sun_composition");
            obj["response"] = util::json_string(
                "The Sun is mostly hydrogen and helium plasma. By mass, it is about 74% hydrogen and 24% helium, "
                "with roughly 2% heavier elements such as oxygen, carbon, neon, and iron.");
        } else if (asks_sun) {
            obj["topic"] = util::json_string("sun");
            obj["response"] = util::json_string(
                "The Sun is the star at the center of our solar system. It is a hot sphere of plasma powered by nuclear fusion, and it provides most of Earth's light and heat.");
        } else if (asks_moon) {
            obj["topic"] = util::json_string("moon");
            obj["response"] = util::json_string(
                "The Moon is Earth's natural satellite. It orbits Earth about every 27.3 days and influences ocean tides through gravity.");
        } else {
            obj["topic"] = util::json_string("kuhul");
            obj["response"] = util::json_string(
                "K'UHUL is KHANARY's semantic execution layer: phases (Pop, Wo, Yax, Sek, Ch'en, Xul) define runtime operations over folds, while XCFE authorizes control-flow transitions and commits.");
            obj["local_source"] = util::json_string("docs/KHANARY-CPP-SPEC.md, docs/SEMANTIC_FOLDS.md");
        }
        return util::json_object(std::move(obj));
    }

    const bool calc_hint = contains_phrase_ci(low, "calculate") ||
                           contains_phrase_ci(low, "what is") ||
                           contains_phrase_ci(low, "compute") ||
                           contains_phrase_ci(low, "how much is");
    double lhs = 0.0;
    double rhs = 0.0;
    std::string op;
    if (!parse_simple_math(toks, &lhs, &rhs, &op)) return util::json_null();
    const bool has_math_language = calc_hint ||
                                   has_token(toks, "plus") ||
                                   has_token(toks, "minus") ||
                                   has_token(toks, "times") ||
                                   has_token(toks, "multiply") ||
                                   has_token(toks, "multiplied") ||
                                   has_token(toks, "divide") ||
                                   has_token(toks, "divided") ||
                                   has_token(toks, "over") ||
                                   has_token(toks, "mod") ||
                                   has_token(toks, "modulo");
    if (!has_math_language) return util::json_null();

    double result = 0.0;
    if (op == "add")
        result = lhs + rhs;
    else if (op == "subtract")
        result = lhs - rhs;
    else if (op == "multiply")
        result = lhs * rhs;
    else if (op == "divide") {
        if (std::fabs(rhs) < 1e-12) {
            util::JsonObject err;
            err["ok"] = util::json_bool(false);
            err["source"] = util::json_string("utility-runtime");
            err["status"] = util::json_string("calculation_error");
            err["op"] = util::json_string("calculate");
            err["delegate_track"] = util::json_string("MATH-µ");
            err["reason"] = util::json_string("division_by_zero");
            err["response"] = util::json_string("Cannot divide by zero.");
            return util::json_object(std::move(err));
        }
        result = lhs / rhs;
    } else if (op == "modulo") {
        if (std::fabs(rhs) < 1e-12) {
            util::JsonObject err;
            err["ok"] = util::json_bool(false);
            err["source"] = util::json_string("utility-runtime");
            err["status"] = util::json_string("calculation_error");
            err["op"] = util::json_string("calculate");
            err["delegate_track"] = util::json_string("MATH-µ");
            err["reason"] = util::json_string("modulo_by_zero");
            err["response"] = util::json_string("Cannot modulo by zero.");
            return util::json_object(std::move(err));
        }
        result = std::fmod(lhs, rhs);
    } else {
        return util::json_null();
    }

    const std::string lhs_text = format_number_compact(lhs);
    const std::string rhs_text = format_number_compact(rhs);
    const std::string result_text = format_number_compact(result);
    std::string op_symbol = "+";
    if (op == "subtract") op_symbol = "-";
    else if (op == "multiply") op_symbol = "*";
    else if (op == "divide") op_symbol = "/";
    else if (op == "modulo") op_symbol = "%";

    util::JsonObject obj;
    obj["ok"] = util::json_bool(true);
    obj["source"] = util::json_string("utility-runtime");
    obj["status"] = util::json_string("calculation_complete");
    obj["op"] = util::json_string("calculate");
    obj["delegate_track"] = util::json_string("MATH-µ");
    obj["expression"] = util::json_string(lhs_text + " " + op_symbol + " " + rhs_text);
    obj["lhs"] = util::json_number(lhs);
    obj["rhs"] = util::json_number(rhs);
    obj["operator"] = util::json_string(op_symbol);
    obj["result"] = util::json_number(result);
    obj["result_text"] = util::json_string(result_text);
    obj["response"] = util::json_string(result_text);
    return util::json_object(std::move(obj));
}

static std::string topic_anchor_from_text(const std::string& text) {
    static const std::unordered_set<std::string> kStop = {
        "mx2lm","ngram","ngrams",
        "if","else","when","where","how","is","are","equal","for","and","or","not","because","then",
        "the","a","an","to","of","in","on","at","by","with","without","from","into","onto","about",
        "i","you","we","they","he","she","it","this","that","these","those",
        "bad","answer","wrong","right","correct","incorrect","yes","no","can","do","should","would",
        "verify","verified","confirm","confirmed","validate","validated","validation",
        "what","which","who","why","your","my","our","their","was","were","be","been","being"
    };

    auto toks = micronaut::Mx2LmNGram::tokenize(text);
    for (const auto& tok : toks) {
        if (tok.size() < 3) continue;
        if (kStop.count(tok)) continue;
        return tok;
    }
    return "global";
}

static std::string topic_anchor_from_key(const std::string& topic_key) {
    if (topic_key.rfind("topic:", 0) == 0)
        return topic_key.substr(6);
    return topic_key.empty() ? "global" : topic_key;
}

static std::string relation_confidence_key(
    const std::string& topic_anchor,
    const std::string& from,
    const std::string& relation,
    const std::string& to)
{
    return "rel:" + topic_anchor + "|" + from + "|" + relation + "|" + to;
}

static std::vector<std::string> collect_relation_confidence_keys(
    const util::JsonPtr& obj,
    const std::string& topic_anchor)
{
    std::vector<std::string> out;
    if (!obj || !obj->is_object()) return out;
    const auto* rels = obj->get("relations");
    if (!rels || !rels->is_array()) return out;

    std::unordered_set<std::string> dedup;
    for (const auto& rv : rels->as_array()) {
        if (!rv || !rv->is_object()) continue;
        auto from = rv->get_string("from");
        auto rel  = rv->get_string("relation");
        auto to   = rv->get_string("to");
        if (!from || !rel || !to) continue;
        auto key = relation_confidence_key(topic_anchor, *from, *rel, *to);
        if (dedup.insert(key).second) out.push_back(std::move(key));
    }
    return out;
}

static void attach_relation_confidence(
    util::JsonPtr obj,
    memory::ConfidenceStore& confidence,
    const std::string& topic_anchor)
{
    if (!obj || !obj->is_object()) return;
    auto& root = obj->mut_object();
    auto it = root.find("relations");
    if (it == root.end() || !it->second || !it->second->is_array()) return;

    auto& rels = it->second->mut_array();
    for (auto& rv : rels) {
        if (!rv || !rv->is_object()) continue;
        auto from = rv->get_string("from");
        auto rel  = rv->get_string("relation");
        auto to   = rv->get_string("to");
        if (!from || !rel || !to) continue;
        const auto key = relation_confidence_key(topic_anchor, *from, *rel, *to);
        auto& robj = rv->mut_object();
        robj["relation_confidence"] = util::json_number((double)confidence.get(key));
        robj["confidence_key"] = util::json_string(key);
    }
}

static void apply_relation_feedback(
    memory::ConfidenceStore& confidence,
    TopicConfidenceUpdate& tc,
    const std::vector<std::string>& last_relation_keys,
    const std::string& last_topic_anchor)
{
    if (!tc.committed || tc.delta == 0.0f) return;
    if (tc.event != "BAD_ANSWER" && tc.event != "GOOD_ANSWER") return;
    if (last_relation_keys.empty()) return;
    if (topic_anchor_from_key(tc.topic_key) != last_topic_anchor) return;

    // Relation-confidence adapts more conservatively than topic confidence.
    const float rel_delta = tc.delta * 0.5f;
    for (const auto& k : last_relation_keys) {
        confidence.update(k, rel_delta);
        tc.relation_updates++;
    }
    tc.relation_delta = rel_delta;
}

// Validates proposed capabilities against an entry's declared phase, sandbox, and governs scope.
// Returns false and sets reject_reason on any violation.
// (1) Well-formedness: non-empty, no control chars, max 128 chars.
// (2) phase:X token validity: token must be a real C6 phase name. Token-validity only — no
//     comparison to entry.primary_phase; a micronaut may claim needs across phases.
// (3) Sandbox-effect guard: skipped when sandbox.effects is empty (open/undeclared transition case).
// (4) Governs scope: acquire_unfold / miss::* caps require at least one governs token.
static bool validate_proposed_capabilities(
    const micronaut::MicronautEntry& entry,
    const std::vector<std::string>& proposed_caps,
    const std::vector<std::string>& proposed_governs,
    std::string& reject_reason)
{
    static const char* kEffects[] = {
        "gramRead", "gramWrite", "semantic_graph:write",
        "micronaut:create", "micronaut:delete", "route:delegate", "manifest:write", nullptr
    };

    bool needs_governs = false;

    for (const auto& cap : proposed_caps) {
        if (cap.empty()) {
            reject_reason = "proposed capability is empty string";
            return false;
        }
        if (cap.size() > 128) {
            reject_reason = "proposed capability exceeds max length: " + cap.substr(0, 40) + "...";
            return false;
        }
        for (unsigned char c : cap) {
            if (c < 0x20) {
                reject_reason = "proposed capability contains control character";
                return false;
            }
        }

        if (cap.rfind("phase:", 0) == 0) {
            std::string tok = cap.substr(6);
            if (!tok.empty())
                tok[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(tok[0])));
            kuhul::Phase dummy{};
            if (!kuhul::phase_from_name(tok, dummy)) {
                reject_reason = "proposed capability contains unknown phase token: " + cap;
                return false;
            }
        }

        if (!entry.sandbox.effects.empty()) {
            for (int i = 0; kEffects[i]; ++i) {
                const std::string kw = kEffects[i];
                if (cap == kw || cap.rfind(kw + ":", 0) == 0) {
                    if (!entry.sandbox.allows_effect(kw)) {
                        reject_reason = "capability '" + cap + "' requires undeclared sandbox effect '"
                                      + kw + "' for " + entry.id;
                        return false;
                    }
                    break;
                }
            }
        }

        if (cap == "acquire_unfold" || cap.rfind("miss::", 0) == 0)
            needs_governs = true;
    }

    if (needs_governs && entry.governs.empty() && proposed_governs.empty()) {
        reject_reason = "miss-handling capabilities require non-empty governs scope for " + entry.id;
        return false;
    }

    return true;
}

static FactoryMutationOutcome apply_miss_factory_mutation(
    micronaut::Factory& factory,
    micronaut::MicronautRegistry& reg,
    const std::string& micronaut_id,
    const std::string& miss_kind,
    const std::vector<std::string>& required_folds,
    const std::vector<std::string>& govern_seeds,
    double research_pressure)
{
    (void)factory;
    FactoryMutationOutcome out;
    out.attempted = true;
    out.micronaut_id = micronaut_id;

    if (micronaut_id.empty()) {
        out.result = "unresolved_target";
        return out;
    }

    const auto* entry = reg.get(micronaut_id);
    if (!entry) {
        out.result = "target_not_found";
        return out;
    }

    auto has_capability = [&](const std::string& cap) {
        return std::find(entry->capabilities.begin(), entry->capabilities.end(), cap) != entry->capabilities.end();
    };
    auto add_capability = [&](const std::string& cap) {
        if (cap.empty()) return;
        if (has_capability(cap)) return;
        if (std::find(out.added_capabilities.begin(), out.added_capabilities.end(), cap) != out.added_capabilities.end())
            return;
        out.added_capabilities.push_back(cap);
    };

    auto has_govern = [&](const std::string& g) {
        return entry->governs.find(g) != entry->governs.end();
    };
    auto add_govern = [&](const std::string& g) {
        if (g.empty()) return;
        if (has_govern(g)) return;
        if (std::find(out.added_governs.begin(), out.added_governs.end(), g) != out.added_governs.end())
            return;
        out.added_governs.push_back(g);
    };

    add_capability("acquire_unfold");
    if (!miss_kind.empty()) add_capability("miss::" + miss_kind);
    add_capability("research_pressure:" +
        std::string(research_pressure >= 0.66 ? "high" :
                    research_pressure >= 0.33 ? "medium" : "low"));
    for (const auto& rf : required_folds) {
        if (rf.empty()) continue;
        add_capability("needs_fold:" + lower_copy(rf));
    }

    for (const auto& seed : govern_seeds)
        add_govern(lower_copy(seed));

    if (out.added_capabilities.empty() && out.added_governs.empty()) {
        out.result = "no_op_already_known";
        out.applied = true;
        return out;
    }

    // CapabilityValidator: verify seed caps before handing off to advisor.
    // Fail-closed: invalid caps → capability_validation_failed, no pending_advisor_verify.
    std::string cap_reject;
    if (!validate_proposed_capabilities(*entry, out.added_capabilities, out.added_governs, cap_reject)) {
        out.result = "capability_validation_failed: " + cap_reject;
        return out;
    }

    // Seed caps validated. Commit is deferred to advisor verification via factory_advisor_commit RPC.
    // PS will call Invoke-AdvisorEvolveQuery → factory_advisor_commit with semantically verified caps.
    // Fail-closed: advisor offline or empty result → no commit.
    out.pending_advisor_verify = true;
    out.result = "pending_advisor_verify";
    return out;
}

static TopicConfidenceUpdate detect_feedback_signal(const std::string& query) {
    const auto low = lower_copy(query);
    const auto toks = micronaut::Mx2LmNGram::tokenize(low);
    auto has_tok = [&](const std::string& t) {
        return std::find(toks.begin(), toks.end(), t) != toks.end();
    };

    static const std::vector<std::string> kNegative = {
        "bad answer", "wrong answer", "not right", "not correct",
        "bad_answer",
        "incorrect", "that is wrong", "that is incorrect", "that was a bad answer",
        "that answer was wrong", "that answer is wrong", "that answer is not right",
        "that answer is not correct", "that answer is incorrect"
    };
    static const std::vector<std::string> kPositive = {
        "correct", "good answer", "good_answer", "thanks", "thank you", "helpful", "verified", "yes"
    };
    static const std::vector<std::string> kMiss = {
        "fact miss", "missed context", "insufficient context", "not enough context",
        "missing context", "need more context", "need to unfold", "no fact found"
    };

    TopicConfidenceUpdate out;
    for (const auto& p : kNegative) {
        if (contains_phrase_ci(low, p)) {
            out.delta = -0.08f;
            out.signal = "negative_feedback";
            out.event = "BAD_ANSWER";
            return out;
        }
    }
    if (has_tok("wrong") || has_tok("incorrect") || has_tok("bad")) {
        out.delta = -0.08f;
        out.signal = "negative_feedback";
        out.event = "BAD_ANSWER";
        return out;
    }
    for (const auto& p : kPositive) {
        if (contains_phrase_ci(low, p)) {
            out.delta = +0.04f;
            out.signal = "positive_feedback";
            out.event = "GOOD_ANSWER";
            return out;
        }
    }
    if (has_tok("correct") || has_tok("verified") || has_tok("thanks")) {
        out.delta = +0.04f;
        out.signal = "positive_feedback";
        out.event = "GOOD_ANSWER";
        return out;
    }
    for (const auto& p : kMiss) {
        if (contains_phrase_ci(low, p)) {
            out.delta = 0.0f;
            out.signal = "insufficient_context";
            out.event = "MISS";
            return out;
        }
    }
    if (has_tok("miss") && (has_tok("context") || has_tok("missing") || has_tok("unfold"))) {
        out.delta = 0.0f;
        out.signal = "insufficient_context";
        out.event = "MISS";
        return out;
    }
    return out;
}

static std::string resolve_feedback_topic_hint(
    const std::string& query,
    const std::string& topic_hint,
    const std::string& last_topic_anchor)
{
    const auto signal = detect_feedback_signal(query);
    if (signal.delta != 0.0f) {
        const auto q_anchor = topic_anchor_from_text(query);
        if (q_anchor == "global" && !last_topic_anchor.empty() && last_topic_anchor != "global")
            return last_topic_anchor;
    }
    return topic_hint.empty() ? query : topic_hint;
}

static TopicConfidenceUpdate apply_topic_feedback(
    memory::ConfidenceStore& confidence,
    const std::string& query,
    const std::string& topic_hint = "",
    bool verified = false)
{
    auto out = detect_feedback_signal(query);
    const auto anchor = topic_hint.empty() ? topic_anchor_from_text(query)
                                           : topic_anchor_from_text(topic_hint);
    out.topic_key = "topic:" + anchor;
    out.before = confidence.get(out.topic_key);

    if (out.delta != 0.0f) {
        if (verified) {
            confidence.update(out.topic_key, out.delta);
            out.committed = true;
            out.commit_phase = "Xul";
            out.signal += "_committed";
        } else {
            out.committed = false;
            out.commit_phase = "Ch'en_pending";
            out.signal += "_candidate";
        }
    } else if (out.event == "MISS") {
        out.committed = false;
        out.commit_phase = "Ch'en_pending";
    }
    out.after = confidence.get(out.topic_key);
    return out;
}

static std::string chen_classification_from_topic_update(const TopicConfidenceUpdate& tc) {
    if (tc.event == "GOOD_ANSWER") return "VERIFIED";
    if (tc.event == "BAD_ANSWER") return "CONTRADICTED";
    if (tc.event == "MISS") return "MISS";
    if (tc.signal == "insufficient_context") return "INSUFFICIENT_CONTEXT";
    if (tc.signal == "ambiguous_context") return "AMBIGUOUS";
    return "";
}

static std::string chen_transition_from_classification(const std::string& cls) {
    if (cls == "VERIFIED") return "Xul";
    if (cls == "CONTRADICTED") return "Revise";
    if (cls == "MISS" || cls == "INSUFFICIENT_CONTEXT") return "Acquire/Unfold";
    if (cls == "AMBIGUOUS") return "Branch/Disambiguate";
    return "";
}

static void attach_topic_confidence(util::JsonPtr obj, const TopicConfidenceUpdate& tc) {
    if (!obj || !obj->is_object()) return;
    auto& root = obj->mut_object();
    util::JsonObject tco;
    tco["topic"]  = util::json_string(tc.topic_key);
    tco["signal"] = util::json_string(tc.signal);
    tco["event"]  = util::json_string(tc.event);
    tco["before"] = util::json_number((double)tc.before);
    tco["after"]  = util::json_number((double)tc.after);
    tco["delta"]  = util::json_number((double)tc.delta);
    tco["committed"] = util::json_bool(tc.committed);
    tco["commit_phase"] = util::json_string(tc.commit_phase);
    tco["relation_updates"] = util::json_number((double)tc.relation_updates);
    tco["relation_delta"] = util::json_number((double)tc.relation_delta);
    const auto chen_cls = chen_classification_from_topic_update(tc);
    if (!chen_cls.empty()) {
        tco["chen_classification"] = util::json_string(chen_cls);
        tco["chen_transition"] = util::json_string(chen_transition_from_classification(chen_cls));
        if (chen_cls == "MISS")
            tco["miss_is_not_wrong"] = util::json_bool(true);
    }
    root["topic_confidence"] = util::json_object(std::move(tco));
}

static void ensure_governance_metadata(
    util::JsonPtr obj,
    const std::string& default_track,
    bool activation_required,
    const std::string& activation_reason,
    const std::string& default_phase = "")
{
    if (!obj || !obj->is_object()) return;
    auto& root = obj->mut_object();

    std::string track = default_track;
    if (track.empty()) {
        if (auto t = obj->get_string("governing_track"); t && !t->empty()) {
            track = *t;
        } else if (auto d = obj->get_string("delegate_track"); d && !d->empty()) {
            track = *d;
        } else {
            track = "Mx2LM";
        }
    }

    if (root.find("governing_track") == root.end())
        root["governing_track"] = util::json_string(track);

    if (!default_phase.empty() && root.find("response_phase") == root.end())
        root["response_phase"] = util::json_string(default_phase);

    auto ait = root.find("activation");
    if (ait == root.end()) {
        util::JsonObject activation;
        activation["mode"] = util::json_string("on_demand");
        activation["required"] = util::json_bool(activation_required);
        activation["reason"] = util::json_string(activation_reason);
        activation["track"] = util::json_string(track);
        root["activation"] = util::json_object(std::move(activation));
        return;
    }
    if (!ait->second || !ait->second->is_object()) return;

    auto& activation = ait->second->mut_object();
    if (activation.find("mode") == activation.end())
        activation["mode"] = util::json_string("on_demand");
    if (activation.find("required") == activation.end())
        activation["required"] = util::json_bool(activation_required);
    if (activation.find("reason") == activation.end())
        activation["reason"] = util::json_string(activation_reason);
    if (activation.find("track") == activation.end())
        activation["track"] = util::json_string(track);
}

static bool is_feedback_verified(const std::string& query) {
    const auto low = lower_copy(query);
    static const std::vector<std::string> kVerifySignals = {
        "verify", "verified", "confirm", "confirmed", "validated", "validation", "ch'en"
    };
    for (const auto& v : kVerifySignals) {
        if (contains_phrase_ci(low, v)) return true;
    }
    return false;
}

static bool is_question_like_text(const std::string& text) {
    const auto trimmed = trim_copy(text);
    if (trimmed.empty()) return false;
    if (trimmed.find('?') != std::string::npos) return true;

    const auto low = lower_copy(trimmed);
    for (const auto& cue : {
             "what ", "who ", "where ", "when ", "why ", "how ",
             "can ", "could ", "would ", "should ", "do ", "does ", "did ", "is ", "are "
         }) {
        if (low.rfind(cue, 0) == 0) return true;
    }
    return false;
}

static bool contains_token_sequence(
    const std::vector<std::string>& tokens,
    std::initializer_list<const char*> sequence)
{
    if (sequence.size() == 0 || tokens.size() < sequence.size()) return false;
    for (size_t i = 0; i + sequence.size() <= tokens.size(); ++i) {
        bool match = true;
        size_t j = 0;
        for (const char* part : sequence) {
            if (tokens[i + j] != part) {
                match = false;
                break;
            }
            ++j;
        }
        if (match) return true;
    }
    return false;
}

static bool contains_token_sequence(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& sequence)
{
    if (sequence.empty() || tokens.size() < sequence.size()) return false;
    for (size_t i = 0; i + sequence.size() <= tokens.size(); ++i) {
        bool match = true;
        for (size_t j = 0; j < sequence.size(); ++j) {
            if (tokens[i + j] != sequence[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

static bool is_function_word_token(const std::string& tok);

static std::vector<std::string> approximate_syllables_token(const std::string& token);

struct Mx2LmActionGramRule {
    std::vector<std::string> gram_tokens;
    std::string intent;
    std::string op;
    std::string response_clause;
};

struct Mx2LmActionGramHit {
    std::string gram;
    std::string intent;
    std::string op;
    std::string response_clause;
};

struct Mx2LmLearnedFact {
    std::string subject;
    std::string predicate;
};

struct Mx2LmSemanticRelation {
    std::string from;
    std::string relation;
    std::string to;
};

struct Mx2LmSemanticChainFact {
    std::string possessor;
    std::string owner;
    std::string part;
    std::string property;
    std::string value;
};

struct Mx2LmSemanticPropertyQuery {
    std::string possessor;
    std::string owner;
    std::string part;
    std::string property;
};

struct Mx2LmAdaptiveStyleState {
    std::string language = "en";
    bool random_chat_mode = false;
    uint32_t slang_big_dog_hits = 0;
    uint32_t slang_dude_hits = 0;
    uint32_t slang_wave_hits = 0;
    uint32_t salutation_turns = 0;
    uint32_t associative_variant_hits = 0;
    uint32_t reactive_mirror_hits = 0;
    uint32_t synthesized_word_hits = 0;
    std::unordered_map<std::string, uint32_t> learned_lingo_hits;
    std::unordered_map<std::string, uint32_t> observed_greeting_hits;
    std::string last_user_greeting;
    std::vector<std::string> synthesized_words;
};

struct Mx2LmStyleDecision {
    std::string phase;
    std::string clause;
};

static const std::vector<Mx2LmActionGramRule>& mx2lm_action_gram_rules() {
    static const std::vector<Mx2LmActionGramRule> kRules = {
        {{"hello"}, "intent:salutation", "salutation", "Hello."},
        {{"hi"}, "intent:salutation", "salutation", "Hello."},
        {{"hey"}, "intent:salutation", "salutation", "Hello."},
        {{"greetings"}, "intent:salutation", "salutation", "Hello."},
        {{"how", "are", "you"}, "intent:wellbeing_check", "wellbeing_check", "I'm online and ready to help."},
        {{"who", "are", "you"}, "intent:identity_query", "identity_query", "I'm the KHANARY MX2LM runtime shell."},
        {{"what", "are", "you"}, "intent:identity_query", "identity_query", "I'm the KHANARY MX2LM runtime shell."},
        {{"what", "is", "your", "name"}, "intent:identity_query", "identity_query", "I'm the KHANARY MX2LM runtime shell."},
        {{"what", "s", "your", "name"}, "intent:identity_query", "identity_query", "I'm the KHANARY MX2LM runtime shell."},
        {{"what's", "your", "name"}, "intent:identity_query", "identity_query", "I'm the KHANARY MX2LM runtime shell."},
        {{"tell", "me", "your", "name"}, "intent:identity_query", "identity_query", "I'm the KHANARY MX2LM runtime shell."},
        {{"what", "can", "you", "do"}, "intent:capability_query", "capability_query", "I can route semantic key queries, tool plans, and runtime operations."},
        {{"how", "can", "you", "help"}, "intent:capability_query", "capability_query", "I can route semantic key queries, tool plans, and runtime operations."},
        {{"do", "you", "learn", "from", "conversation"}, "intent:learning_query", "learning_query", "I learn local facts from explicit memory entries and validated fact-style statements; unknown facts are routed to WEB-RESEARCH-µ for evidence."},
        {{"do", "you", "learn", "from", "conversations"}, "intent:learning_query", "learning_query", "I learn local facts from explicit memory entries and validated fact-style statements; unknown facts are routed to WEB-RESEARCH-µ for evidence."},
        {{"can", "you", "learn", "from", "conversation"}, "intent:learning_query", "learning_query", "I learn local facts from explicit memory entries and validated fact-style statements; unknown facts are routed to WEB-RESEARCH-µ for evidence."},
        {{"i", "don't", "know", "the", "answer"}, "intent:fact_miss_advisor", "advisor_web_research_trigger", "I don't know the answer to that question. Let me search the web and ask my advisor."},
        {{"i", "don", "t", "know", "the", "answer"}, "intent:fact_miss_advisor", "advisor_web_research_trigger", "I don't know the answer to that question. Let me search the web and ask my advisor."},
        {{"i", "do", "not", "know", "the", "answer"}, "intent:fact_miss_advisor", "advisor_web_research_trigger", "I don't know the answer to that question. Let me search the web and ask my advisor."},
        {{"search", "the", "web", "and", "ask", "my", "advisor"}, "intent:fact_miss_advisor", "advisor_web_research_trigger", "I don't know the answer to that question. Let me search the web and ask my advisor."},
        {{"i", "m", "not", "sure", "what", "tool", "to", "use"}, "intent:tool_uncertain_lfm", "lfm_tool_help_trigger", "I'm not sure what tool to use. Let me ask for help from LFM."},
        {{"i", "am", "not", "sure", "what", "tool", "to", "use"}, "intent:tool_uncertain_lfm", "lfm_tool_help_trigger", "I'm not sure what tool to use. Let me ask for help from LFM."},
        {{"not", "sure", "what", "tool", "to", "use"}, "intent:tool_uncertain_lfm", "lfm_tool_help_trigger", "I'm not sure what tool to use. Let me ask for help from LFM."},
        {{"ask", "for", "help", "from", "lfm"}, "intent:tool_uncertain_lfm", "lfm_tool_help_trigger", "I'm not sure what tool to use. Let me ask for help from LFM."},
        {{"can", "you", "do", "math"}, "intent:math_capability_query", "math_capability_query", "Yes, I can do math. Ask an expression like 'what is 2 plus 2?'."}
    };
    return kRules;
}

static std::mutex& mx2lm_style_mutex() {
    static std::mutex m;
    return m;
}

static Mx2LmAdaptiveStyleState& mx2lm_style_state() {
    static Mx2LmAdaptiveStyleState s;
    return s;
}

static std::once_flag& mx2lm_style_restore_once() {
    static std::once_flag flag;
    return flag;
}

static std::string sanitize_style_lingo_token(std::string token) {
    token = lower_copy(trim_copy(std::move(token)));
    std::string out;
    out.reserve(token.size());
    bool last_sep = false;
    for (unsigned char uc : token) {
        if (std::isalnum(uc)) {
            out.push_back(static_cast<char>(uc));
            last_sep = false;
        } else {
            if (!out.empty() && !last_sep) {
                out.push_back('_');
                last_sep = true;
            }
        }
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

static std::string display_style_lingo_token(const std::string& token) {
    std::string out = token;
    std::replace(out.begin(), out.end(), '_', ' ');
    return out;
}

static util::JsonObject serialize_style_profile(const Mx2LmAdaptiveStyleState& state) {
    util::JsonObject obj;
    obj["language"] = util::json_string(state.language);
    obj["random_chat_mode"] = util::json_bool(state.random_chat_mode);
    obj["slang_big_dog_hits"] = util::json_number(static_cast<double>(state.slang_big_dog_hits));
    obj["slang_dude_hits"] = util::json_number(static_cast<double>(state.slang_dude_hits));
    obj["slang_wave_hits"] = util::json_number(static_cast<double>(state.slang_wave_hits));
    obj["salutation_turns"] = util::json_number(static_cast<double>(state.salutation_turns));
    obj["associative_variant_hits"] = util::json_number(static_cast<double>(state.associative_variant_hits));
    obj["reactive_mirror_hits"] = util::json_number(static_cast<double>(state.reactive_mirror_hits));
    obj["synthesized_word_hits"] = util::json_number(static_cast<double>(state.synthesized_word_hits));
    obj["last_user_greeting"] = util::json_string(state.last_user_greeting);
    util::JsonArray lingo_rows;
    std::vector<std::pair<std::string, uint32_t>> ranked(
        state.learned_lingo_hits.begin(),
        state.learned_lingo_hits.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    for (const auto& [token, hits] : ranked) {
        util::JsonObject row;
        row["token"] = util::json_string(token);
        row["hits"] = util::json_number(static_cast<double>(hits));
        lingo_rows.push_back(util::json_object(std::move(row)));
    }
    obj["learned_lingo"] = util::json_array(std::move(lingo_rows));

    // Observed greeting forms: repeated user greetings become reactionary
    // mirrors — the runtime gives back the type of greeting the user favors.
    util::JsonArray observed_rows;
    {
        std::vector<std::pair<std::string, uint32_t>> observed_ranked(
            state.observed_greeting_hits.begin(),
            state.observed_greeting_hits.end());
        std::sort(observed_ranked.begin(), observed_ranked.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });
        if (observed_ranked.size() > 16) observed_ranked.resize(16);
        for (const auto& [form, hits] : observed_ranked) {
            util::JsonObject row;
            row["form"] = util::json_string(form);
            row["hits"] = util::json_number(static_cast<double>(hits));
            observed_rows.push_back(util::json_object(std::move(row)));
        }
    }
    obj["observed_greetings"] = util::json_array(std::move(observed_rows));

    // Words the runtime synthesized from the user's own lingo.
    util::JsonArray synth_rows;
    {
        std::vector<std::string> synth = state.synthesized_words;
        if (synth.size() > 16) synth.resize(16);
        for (const auto& word : synth)
            synth_rows.push_back(util::json_string(word));
    }
    obj["synthesized_words"] = util::json_array(std::move(synth_rows));
    return obj;
}

static bool deserialize_style_profile(
    const util::JsonPtr& payload,
    Mx2LmAdaptiveStyleState* out)
{
    if (!payload || !payload->is_object() || !out) return false;
    auto state = Mx2LmAdaptiveStyleState{};
    if (auto lang = payload->get_string("language"); lang && !lang->empty())
        state.language = *lang;
    if (auto random_mode = payload->get("random_chat_mode"); random_mode && random_mode->is_bool())
        state.random_chat_mode = random_mode->as_bool();
    if (auto v = payload->get_number("slang_big_dog_hits"); v) state.slang_big_dog_hits = static_cast<uint32_t>(std::max(0.0, *v));
    if (auto v = payload->get_number("slang_dude_hits"); v) state.slang_dude_hits = static_cast<uint32_t>(std::max(0.0, *v));
    if (auto v = payload->get_number("slang_wave_hits"); v) state.slang_wave_hits = static_cast<uint32_t>(std::max(0.0, *v));
    if (auto v = payload->get_number("salutation_turns"); v) state.salutation_turns = static_cast<uint32_t>(std::max(0.0, *v));
    if (auto v = payload->get_number("associative_variant_hits"); v) state.associative_variant_hits = static_cast<uint32_t>(std::max(0.0, *v));
    if (auto v = payload->get_number("reactive_mirror_hits"); v) state.reactive_mirror_hits = static_cast<uint32_t>(std::max(0.0, *v));
    if (auto v = payload->get_number("synthesized_word_hits"); v) state.synthesized_word_hits = static_cast<uint32_t>(std::max(0.0, *v));
    if (auto v = payload->get_string("last_user_greeting"); v) state.last_user_greeting = *v;
    if (const auto* observed = payload->get("observed_greetings"); observed && observed->is_array()) {
        for (const auto& item : observed->as_array()) {
            if (!item || !item->is_object()) continue;
            const auto form_raw = item->get_string("form");
            const auto hits = item->get_number("hits");
            if (!form_raw || !hits) continue;
            const auto form = trim_copy(lower_copy(*form_raw));
            if (form.empty()) continue;
            state.observed_greeting_hits[form] = static_cast<uint32_t>(std::max(0.0, *hits));
        }
    }
    if (const auto* synth = payload->get("synthesized_words"); synth && synth->is_array()) {
        for (const auto& item : synth->as_array()) {
            if (!item || !item->is_string()) continue;
            const auto word = sanitize_style_lingo_token(item->as_string());
            if (word.empty()) continue;
            if (std::find(state.synthesized_words.begin(),
                          state.synthesized_words.end(),
                          word) == state.synthesized_words.end())
                state.synthesized_words.push_back(word);
            if (state.synthesized_words.size() >= 16) break;
        }
    }
    if (const auto* learned = payload->get("learned_lingo"); learned && learned->is_array()) {
        for (const auto& item : learned->as_array()) {
            if (!item || !item->is_object()) continue;
            const auto token_raw = item->get_string("token");
            const auto hits = item->get_number("hits");
            if (!token_raw || !hits) continue;
            const auto token = sanitize_style_lingo_token(*token_raw);
            if (token.empty()) continue;
            state.learned_lingo_hits[token] = static_cast<uint32_t>(std::max(0.0, *hits));
        }
    }
    *out = std::move(state);
    return true;
}

static std::string mx2lm_style_profile_key() {
    return "mx2lm:style_profile:v1";
}

static void restore_style_profile_once(micronaut::Factory* factory, Mx2LmAdaptiveStyleState* state) {
    if (!factory || !state) return;
    std::call_once(mx2lm_style_restore_once(), [&]() {
        const auto persisted = factory->peek_key(mx2lm_style_profile_key());
        if (!persisted || persisted->empty()) return;
        auto parsed = util::parse(*persisted);
        if (!deserialize_style_profile(parsed, state)) return;
    });
}

static void persist_style_profile(
    micronaut::Factory* factory,
    kuhul::GramStore* gram_store,
    const Mx2LmAdaptiveStyleState& state,
    const std::string& phase,
    const std::vector<std::string>& lingo_cues)
{
    auto payload_obj = serialize_style_profile(state);
    const auto payload = util::stringify(util::json_object(payload_obj));
    if (factory) {
        factory->record_key(mx2lm_style_profile_key(), payload);
        for (const auto& cue : lingo_cues) {
            auto it = state.learned_lingo_hits.find(cue);
            if (it == state.learned_lingo_hits.end()) continue;
            factory->record_key("mx2lm:lingo:" + cue, std::to_string(it->second));
        }
        // Greeting learning keys: observed forms, last user greeting, and
        // associative words the runtime synthesized from the user's lingo.
        factory->record_key("greeting:last_user", state.last_user_greeting);
        {
            std::vector<std::pair<std::string, uint32_t>> observed_ranked(
                state.observed_greeting_hits.begin(),
                state.observed_greeting_hits.end());
            std::sort(observed_ranked.begin(), observed_ranked.end(), [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
            });
            if (observed_ranked.size() > 16) observed_ranked.resize(16);
            for (const auto& [form, hits] : observed_ranked)
                factory->record_key("greeting:observed:" + form, std::to_string(hits));
        }
        for (const auto& word : state.synthesized_words) {
            factory->record_key("greeting:variant:" + word, "synthesized_associative_word");
            factory->record_key("mx2lm:lingo_blend:" + word,
                               std::to_string(state.synthesized_word_hits));
        }
    }
    if (gram_store) {
        gram_store->gramWrite(
            "mx2lm_style_profile:" + phase,
            payload,
            "Ch'en",
            "style_adaptation_profile",
            "definitions_and_relations");
    }
}

static bool is_random_chat_signal(const std::string& low) {
    return contains_phrase_ci(low, "random stuff") ||
           contains_phrase_ci(low, "just chat") ||
           contains_phrase_ci(low, "just talk") ||
           contains_phrase_ci(low, "off your chest") ||
           contains_phrase_ci(low, "shoot the breeze") ||
           contains_phrase_ci(low, "talk about anything") ||
           contains_phrase_ci(low, "only here to talk");
}

static bool is_wave_slang_signal(const std::string& low, const std::vector<std::string>& tokens) {
    return contains_phrase_ci(low, "ride some waves") ||
           contains_phrase_ci(low, "ride waves") ||
           contains_phrase_ci(low, "surf") ||
           has_token(tokens, "waves") ||
           has_token(tokens, "wave");
}

static bool is_chinese_language_signal(const std::string& low) {
    return contains_phrase_ci(low, "set language chinese") ||
           contains_phrase_ci(low, "set language mandarin") ||
           contains_phrase_ci(low, "language zh") ||
           contains_phrase_ci(low, "locale zh") ||
           contains_phrase_ci(low, "ni hao") ||
           contains_phrase_ci(low, "nǐ hǎo") ||
           contains_phrase_ci(low, "speak chinese");
}

static bool is_english_language_signal(const std::string& low) {
    return contains_phrase_ci(low, "set language english") ||
           contains_phrase_ci(low, "language en") ||
           contains_phrase_ci(low, "locale en") ||
           contains_phrase_ci(low, "speak english");
}

static std::vector<std::string> extract_salutation_lingo_tokens(const std::vector<std::string>& tokens) {
    static const std::unordered_set<std::string> kIgnore = {
        "hello","hi","hey","greetings","what","s","up","how","are","you","yo","sup",
        "chat","talk","please","thanks","thank"
    };
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;

    auto push = [&](const std::string& raw) {
        const auto token = sanitize_style_lingo_token(raw);
        if (token.size() < 3) return;
        if (kIgnore.count(token)) return;
        if (!seen.insert(token).second) return;
        out.push_back(token);
    };

    for (const auto& tok : tokens) {
        if (tok.empty()) continue;
        if (is_function_word_token(tok)) continue;
        push(tok);
    }

    if (contains_token_sequence(tokens, {"big", "dog"}))
        push("big_dog");
    if (has_token(tokens, "waves") || has_token(tokens, "wave") || has_token(tokens, "surf"))
        push("waves");
    return out;
}

static std::vector<std::string> absorb_adaptive_style_signals(
    const std::string& payload_low,
    const std::vector<std::string>& tokens,
    Mx2LmAdaptiveStyleState* state)
{
    if (!state) return {};

    if (is_chinese_language_signal(payload_low)) state->language = "zh";
    if (is_english_language_signal(payload_low)) state->language = "en";

    if (is_random_chat_signal(payload_low)) state->random_chat_mode = true;
    if (contains_phrase_ci(payload_low, "random chat mode off") ||
        contains_phrase_ci(payload_low, "set chat mode neutral")) {
        state->random_chat_mode = false;
    }

    if (contains_token_sequence(tokens, {"big", "dog"}))
        state->slang_big_dog_hits += 1;
    if (has_token(tokens, "dude"))
        state->slang_dude_hits += 1;
    if (is_wave_slang_signal(payload_low, tokens))
        state->slang_wave_hits += 1;

    auto lingo = extract_salutation_lingo_tokens(tokens);
    for (const auto& cue : lingo)
        state->learned_lingo_hits[cue] += 1;
    return lingo;
}

static std::string build_associative_greeting_variant(
    const Mx2LmAdaptiveStyleState& state,
    const std::vector<std::string>& lingo_cues)
{
    std::string cue;
    if (!lingo_cues.empty()) cue = lingo_cues.front();
    if (cue.empty() && !state.learned_lingo_hits.empty()) {
        std::vector<std::pair<std::string, uint32_t>> ranked(
            state.learned_lingo_hits.begin(),
            state.learned_lingo_hits.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });
        cue = ranked.front().first;
    }
    if (cue.empty()) return {};
    const auto cue_display = display_style_lingo_token(cue);
    const uint32_t seed = state.salutation_turns + state.associative_variant_hits + state.slang_wave_hits;
    switch (seed % 3u) {
        case 0u: return "Hey " + cue_display + ", good to see you again.";
        case 1u: return "Yo " + cue_display + ", ready when you are.";
        default: return "What's good " + cue_display + "? I'm online and locked in.";
    }
}

// ── Phase-supplied greeting bank ───────────────────────────────────────────
// Operators supply different types of greetings per K'UHUL phase in
// data/grammar/greeting-bank.json. The runtime rotates salutations through
// the C6 phases, mirrors greetings the user repeats (reactionary), adapts
// to learned lingo, and synthesizes its own associative words in the same
// mannerisms. Parsing, placeholder resolution, clause selection and the
// salutation policy live in micronaut/greeting_style.{h,cpp} so the greeting
// learning path is unit-testable without the GPU/Factory runtime; the built-in
// fallbacks there keep behavior intact without the bank.

static std::string& mx2lm_greeting_bank_path() {
    static std::string path = "data/grammar/greeting-bank.json";
    return path;
}

static void set_mx2lm_greeting_bank_path(const std::string& path) {
    if (!path.empty()) mx2lm_greeting_bank_path() = path;
}

static const std::vector<greeting::BankEntry>& load_greeting_bank() {
    static const std::vector<greeting::BankEntry> bank = [] {
        std::ifstream f(mx2lm_greeting_bank_path());
        if (!f) return std::vector<greeting::BankEntry>{};
        std::ostringstream ss;
        ss << f.rdbuf();
        return greeting::parse_bank(ss.str());
    }();
    return bank;
}

// Synthesize a new associative word by blending the syllables of the two
// most-observed lingo cues — the runtime creating its own words in the
// user's mannerisms.
static std::string synthesize_associative_word(const Mx2LmAdaptiveStyleState& state) {
    if (state.learned_lingo_hits.size() < 2) return {};
    std::vector<std::pair<std::string, uint32_t>> ranked(
        state.learned_lingo_hits.begin(),
        state.learned_lingo_hits.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    const auto& cue_a = ranked[0].first;
    const auto& cue_b = ranked[1].first;
    if (cue_a == cue_b) return {};

    std::string head;
    std::string tail;
    const auto sylls_a = approximate_syllables_token(cue_a);
    const auto sylls_b = approximate_syllables_token(cue_b);
    if (sylls_a.size() >= 2) {
        const size_t half = (sylls_a.size() + 1) / 2;
        for (size_t i = 0; i < half; ++i) head += sylls_a[i];
    } else if (!cue_a.empty()) {
        head = cue_a.substr(0, (cue_a.size() + 1) / 2);
    }
    if (sylls_b.size() >= 2) {
        const size_t half = sylls_b.size() / 2;
        for (size_t i = half; i < sylls_b.size(); ++i) tail += sylls_b[i];
    } else if (!cue_b.empty()) {
        tail = cue_b.substr(cue_b.size() / 2);
    }

    const auto blend = sanitize_style_lingo_token(head + tail);
    if (blend.size() < 3 || blend.size() > 14) return {};
    if (blend == sanitize_style_lingo_token(cue_a) ||
        blend == sanitize_style_lingo_token(cue_b))
        return {};
    return blend;
}

static void remember_synthesized_word(Mx2LmAdaptiveStyleState* state, const std::string& word) {
    if (!state || word.empty()) return;
    auto& words = state->synthesized_words;
    if (std::find(words.begin(), words.end(), word) != words.end()) return;
    words.push_back(word);
    if (words.size() > 16) words.erase(words.begin());
}

static Mx2LmStyleDecision resolve_adaptive_salutation_phase(
    const std::string& payload_text,
    const std::vector<std::string>& tokens,
    micronaut::Factory* style_factory = nullptr,
    kuhul::GramStore* gram_store = nullptr)
{
    const auto low = lower_copy(payload_text);
    std::lock_guard<std::mutex> lock(mx2lm_style_mutex());
    auto& state = mx2lm_style_state();
    restore_style_profile_once(style_factory, &state);
    const auto lingo = absorb_adaptive_style_signals(low, tokens, &state);
    state.salutation_turns += 1;

    // Archive the observed greeting form. Every submitted salutation prompt
    // already records its typed n-grams (unigram/bigram/trigram/quadgram/
    // coarse/function/tool channels) through prompt persistence; this keeps
    // a greeting-form signature so repeats can be mirrored back.
    const auto greeting_form = greeting::observed_form(tokens);
    uint32_t form_hits = 0;
    if (!greeting_form.empty()) {
        form_hits = ++state.observed_greeting_hits[greeting_form];
        state.last_user_greeting = trim_copy(payload_text).substr(0, 80);
    }

    // Bank cue: the cue the user just used, else the most-observed lingo. The
    // runtime never fakes lingo it has not observed.
    std::string cue;
    if (!lingo.empty()) cue = lingo.front();
    if (cue.empty()) {
        const auto ranked = greeting::rank_lingo_cues(state.learned_lingo_hits);
        if (!ranked.empty()) cue = ranked.front();
    }
    const auto cue_display = display_style_lingo_token(cue);
    const auto user_greeting = trim_copy(payload_text);
    const auto last_greeting = trim_copy(state.last_user_greeting);

    auto bank_clause = [&](const std::string& phase_key,
                           const std::string& synthesized_word) {
        const uint32_t seed = state.salutation_turns + state.associative_variant_hits +
                              state.reactive_mirror_hits;
        const greeting::PlaceholderContext ctx{state.salutation_turns,
                                               state.learned_lingo_hits.size()};
        return greeting::select_clause(load_greeting_bank(), phase_key, cue_display,
                                       user_greeting, last_greeting, synthesized_word,
                                       seed, ctx);
    };
    auto decide = [&](const std::string& phase, const std::string& clause) {
        const Mx2LmStyleDecision d{phase, clause};
        persist_style_profile(style_factory, gram_store, state, d.phase, lingo);
        return d;
    };

    greeting::SalutationInputs inputs;
    inputs.zh = (state.language == "zh");
    inputs.form_hits = form_hits;
    inputs.wave_now = is_wave_slang_signal(low, tokens);
    inputs.slang_wave_hits = state.slang_wave_hits;
    inputs.big_dog_now = contains_token_sequence(tokens, {"big", "dog"});
    inputs.slang_big_dog_hits = state.slang_big_dog_hits;
    inputs.random_chat = state.random_chat_mode || is_random_chat_signal(low);
    inputs.salutation_turns = state.salutation_turns;
    inputs.associative_variant_hits = state.associative_variant_hits;

    // Walk the ordered learning policy and take the first route that can
    // actually answer. The order (mirror before synthesis, synthesis before
    // generic variants, phase cycling last, neutral always) lives in
    // greeting::salutation_candidates so it is unit-testable.
    for (const auto& candidate : greeting::salutation_candidates(inputs)) {
        switch (candidate.route) {
            case greeting::SalutationRoute::LocaleChinese:
                return decide(candidate.phase,
                              "Ni hao. Wo neng wei nin tigong shenme bangzhu?");
            case greeting::SalutationRoute::ReactiveMirror: {
                // The user repeated a greeting form the runtime archived — give
                // the same type of greeting back, in their lingo.
                auto clause = bank_clause(candidate.bank_phase, {});
                if (!clause.empty()) {
                    state.reactive_mirror_hits += 1;
                    return decide(candidate.phase, clause);
                }
                break;
            }
            case greeting::SalutationRoute::LearnedWaveSlang:
                return decide(candidate.phase,
                              "What's up dude, you ready to ride some waves?");
            case greeting::SalutationRoute::LearnedSlang:
                return decide(candidate.phase,
                              "Hey, hey, big dog! What you got for me today?");
            case greeting::SalutationRoute::RandomChat:
                return decide(candidate.phase,
                              "Hi! If you just want to talk, I'm here to chat about "
                              "whatever is on your mind.");
            case greeting::SalutationRoute::SynthesizedGreeting: {
                // Create the runtime's own word by blending the user's top cues.
                const auto blend = synthesize_associative_word(state);
                if (blend.empty()) break;
                auto clause = bank_clause(candidate.bank_phase, blend);
                if (clause.empty()) break;
                state.synthesized_word_hits += 1;
                remember_synthesized_word(&state, blend);
                return decide(candidate.phase, clause);
            }
            case greeting::SalutationRoute::AssociativeGreeting: {
                const auto variant = build_associative_greeting_variant(state, lingo);
                if (variant.empty()) break;
                state.associative_variant_hits += 1;
                return decide(candidate.phase, variant);
            }
            case greeting::SalutationRoute::PhaseCycle: {
                auto clause = bank_clause(candidate.bank_phase, {});
                if (!clause.empty()) return decide(candidate.phase, clause);
                break;
            }
            case greeting::SalutationRoute::Neutral:
                return decide(candidate.phase, "Hello. I'm online and ready to help.");
        }
    }

    return decide("Pop.neutral", "Hello. I'm online and ready to help.");
}

static util::JsonPtr style_settings_payload(
    const std::string& text,
    micronaut::Factory* style_factory,
    kuhul::GramStore* gram_store)
{
    const auto q = trim_copy(text);
    if (q.empty()) return util::json_null();
    const auto low = lower_copy(q);

    const bool requests_reset =
        contains_phrase_ci(low, "reset style profile") ||
        contains_phrase_ci(low, "clear style profile") ||
        contains_phrase_ci(low, "style profile reset");
    const bool requests_language =
        contains_phrase_ci(low, "set language") ||
        contains_phrase_ci(low, "language zh") ||
        contains_phrase_ci(low, "language en") ||
        contains_phrase_ci(low, "locale zh") ||
        contains_phrase_ci(low, "locale en") ||
        contains_phrase_ci(low, "speak chinese") ||
        contains_phrase_ci(low, "speak english");
    const bool requests_chat_mode =
        contains_phrase_ci(low, "set chat mode") ||
        contains_phrase_ci(low, "random chat mode");

    if (!requests_reset && !requests_language && !requests_chat_mode)
        return util::json_null();

    std::string response = "Style profile unchanged.";
    std::string op = "style_profile_view";
    std::string phase = "Sek.profile";

    {
        std::lock_guard<std::mutex> lock(mx2lm_style_mutex());
        auto& state = mx2lm_style_state();
        restore_style_profile_once(style_factory, &state);

        if (requests_reset) {
            state = Mx2LmAdaptiveStyleState{};
            response = "Style profile reset to neutral defaults.";
            op = "style_profile_reset";
            phase = "Xul.reset";
        }

        if (requests_language) {
            if (is_chinese_language_signal(low)) {
                state.language = "zh";
                response = "Language profile set to Chinese greeting pattern.";
                op = "style_profile_language";
                phase = "Sek.locale_language";
            } else if (is_english_language_signal(low)) {
                state.language = "en";
                response = "Language profile set to English greeting pattern.";
                op = "style_profile_language";
                phase = "Sek.locale_language";
            }
        }

        if (requests_chat_mode) {
            if (contains_phrase_ci(low, "off") || contains_phrase_ci(low, "neutral")) {
                state.random_chat_mode = false;
                response = "Chat mode set to neutral.";
            } else {
                state.random_chat_mode = true;
                response = "Chat mode set to random-chat context.";
            }
            op = "style_profile_chat_mode";
            phase = "Wo.random_chat_context";
        }

        persist_style_profile(style_factory, gram_store, state, phase, {});

        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["source"] = util::json_string("utility-runtime");
        obj["status"] = util::json_string("style_profile_updated");
        obj["op"] = util::json_string(op);
        obj["delegate_track"] = util::json_string("Mx2LM");
        obj["phase"] = util::json_string(phase);
        obj["response"] = util::json_string(response);
        obj["style_profile"] = util::json_object(serialize_style_profile(state));
        return util::json_object(std::move(obj));
    }
}

static bool is_function_word_token(const std::string& tok) {
    static const std::unordered_set<std::string> kFunctionWords = {
        "a","an","the","this","that","these","those",
        "i","you","he","she","it","we","they","me","him","her","us","them",
        "my","your","his","its","our","their","mine","yours","ours","theirs",
        "is","am","are","was","were","be","being","been",
        "do","does","did","can","could","will","would","shall","should","may","might","must",
        "have","has","had",
        "to","of","in","on","at","by","for","with","without","from","into","over","under","through","across","about","as",
        "and","or","but","nor","so","yet",
        "if","then","than","because","while","when","where","who","whom","whose","which","what","why","how",
        "not","no","yes"
    };
    return kFunctionWords.find(tok) != kFunctionWords.end();
}

static bool is_basic_vowel_char(char c) {
    const unsigned char uc = static_cast<unsigned char>(c);
    const char low = static_cast<char>(std::tolower(uc));
    return low == 'a' || low == 'e' || low == 'i' || low == 'o' || low == 'u' || low == 'y';
}

static std::vector<std::string> approximate_syllables_token(const std::string& token) {
    std::vector<std::string> out;
    const auto t = trim_copy(lower_copy(token));
    if (t.empty()) return out;
    if (t.size() <= 3) {
        out.push_back(t);
        return out;
    }

    std::string chunk;
    chunk.reserve(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        const char c = t[i];
        if (!std::isalnum(static_cast<unsigned char>(c))) continue;
        chunk.push_back(c);
        if (i + 1 >= t.size()) continue;

        const bool curr_vowel = is_basic_vowel_char(c);
        const bool next_vowel = is_basic_vowel_char(t[i + 1]);
        if (curr_vowel && !next_vowel) {
            bool has_future_vowel = false;
            for (size_t j = i + 1; j < t.size(); ++j) {
                if (is_basic_vowel_char(t[j])) {
                    has_future_vowel = true;
                    break;
                }
            }
            if (has_future_vowel && chunk.size() >= 2) {
                out.push_back(chunk);
                chunk.clear();
            }
        }
    }
    if (!chunk.empty()) out.push_back(chunk);
    if (out.empty()) out.push_back(t);
    return out;
}

static bool is_negation_prefix(const std::string& morph) {
    return morph == "un" || morph == "in" || morph == "im" || morph == "ir" || morph == "il" ||
           morph == "non" || morph == "dis";
}

static std::vector<std::string> segment_morphemes_token(const std::string& token) {
    const auto t = trim_copy(lower_copy(token));
    if (t.size() < 4) return {t};

    static const std::vector<std::string> kPrefixes = {
        "anti","inter","under","over","post","pre","non","dis","un","re","in","im","ir","il"
    };
    static const std::vector<std::string> kSuffixes = {
        "ability","ibility","ation","ition","ment","ness","able","ible","less","ful","ing","edly","edly","ed","ly","er","est","s"
    };

    std::string prefix;
    std::string suffix;
    size_t root_start = 0;
    size_t root_end = t.size();

    for (const auto& p : kPrefixes) {
        if (p.size() + 2 >= t.size()) continue;
        if (t.rfind(p, 0) == 0) {
            prefix = p;
            root_start = p.size();
            break;
        }
    }
    for (const auto& s : kSuffixes) {
        if (s.size() + 2 >= (root_end - root_start)) continue;
        if (t.size() >= s.size() &&
            t.compare(t.size() - s.size(), s.size(), s) == 0) {
            suffix = s;
            root_end = t.size() - s.size();
            break;
        }
    }

    std::vector<std::string> out;
    if (!prefix.empty()) out.push_back(prefix);
    const auto root = t.substr(root_start, root_end - root_start);
    if (!root.empty()) out.push_back(root);
    if (!suffix.empty()) out.push_back(suffix);
    if (out.empty()) out.push_back(t);
    return out;
}

static std::vector<std::string> semantic_atoms_for_morpheme(
    const std::string& morph,
    bool root_slot)
{
    const auto m = trim_copy(lower_copy(morph));
    if (m.empty()) return {};
    if (is_negation_prefix(m)) return {"NEGATION"};
    if (m == "re") return {"REPEAT"};
    if (m == "pre") return {"PRIORITY_TIME"};
    if (m == "post") return {"POST_TIME"};
    if (m == "over") return {"EXCESS"};
    if (m == "under") return {"INSUFFICIENCY"};
    if (m == "able" || m == "ible") return {"CAPABILITY"};
    if (m == "ing") return {"ASPECT_PROGRESSIVE"};
    if (m == "ed") return {"ASPECT_COMPLETED"};
    if (m == "ly") return {"MANNER"};
    if (m == "ness" || m == "ment" || m == "ation" || m == "ition") return {"ABSTRACTION"};
    if (m == "s" || m == "es") return {"NUMBER_PLURAL"};
    if (root_slot) return {"ROOT"};
    return {"AFFIX"};
}

static util::JsonPtr build_language_fold_payload(
    const std::string& payload_text,
    const std::vector<std::string>& tokens,
    const std::string& composed_text,
    const std::string& response_phase,
    const std::string& governing_track,
    size_t relation_count)
{
    util::JsonObject fold;
    fold["contract"] = util::json_string("phase_operates_over_units_not_positions");
    fold["source_text"] = util::json_string(payload_text);
    fold["token_count"] = util::json_number((double)tokens.size());

    util::JsonArray char_fold;
    for (char c : payload_text) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        char_fold.push_back(util::json_string(std::string(1, c)));
    }
    fold["character_fold"] = util::json_array(std::move(char_fold));

    uint32_t function_count = 0;
    uint32_t content_count = 0;
    std::vector<std::string> function_words;
    std::vector<std::string> content_words;
    std::unordered_set<std::string> seen_fn;
    std::unordered_set<std::string> seen_content;

    std::unordered_map<std::string, uint32_t> atom_counts;
    std::vector<std::string> atom_order;
    bool has_multi_morpheme = false;
    auto bump_atom = [&](const std::string& atom) {
        if (atom.empty()) return;
        if (atom_counts.find(atom) == atom_counts.end())
            atom_order.push_back(atom);
        atom_counts[atom] += 1;
    };

    util::JsonArray token_rows;
    util::JsonArray syllable_rows;
    util::JsonArray morpheme_rows;
    for (const auto& tok : tokens) {
        const auto role = is_function_word_token(tok) ? std::string("function_word") : std::string("content_word");
        if (role == "function_word") {
            function_count += 1;
            if (seen_fn.insert(tok).second) function_words.push_back(tok);
        } else {
            content_count += 1;
            if (seen_content.insert(tok).second) content_words.push_back(tok);
        }

        const auto syllables = approximate_syllables_token(tok);
        const auto morphemes = segment_morphemes_token(tok);
        if (morphemes.size() > 1) has_multi_morpheme = true;
        size_t root_idx = 0;
        size_t max_len = 0;
        for (size_t i = 0; i < morphemes.size(); ++i) {
            if (morphemes[i].size() > max_len) {
                max_len = morphemes[i].size();
                root_idx = i;
            }
        }

        std::vector<std::string> semantic_atoms;
        std::unordered_set<std::string> seen_atom_local;
        for (size_t i = 0; i < morphemes.size(); ++i) {
            auto atoms = semantic_atoms_for_morpheme(morphemes[i], i == root_idx);
            for (auto& atom : atoms) {
                if (seen_atom_local.insert(atom).second)
                    semantic_atoms.push_back(atom);
                bump_atom(atom);
            }
        }

        util::JsonObject tok_row;
        tok_row["token"] = util::json_string(tok);
        tok_row["role"] = util::json_string(role);
        tok_row["syllables"] = util::json_array(json_string_array(syllables));
        tok_row["morphemes"] = util::json_array(json_string_array(morphemes));
        tok_row["semantic_atoms"] = util::json_array(json_string_array(semantic_atoms));
        token_rows.push_back(util::json_object(std::move(tok_row)));

        util::JsonObject syl_row;
        syl_row["token"] = util::json_string(tok);
        syl_row["syllables"] = util::json_array(json_string_array(syllables));
        syllable_rows.push_back(util::json_object(std::move(syl_row)));

        util::JsonObject morph_row;
        morph_row["token"] = util::json_string(tok);
        morph_row["morphemes"] = util::json_array(json_string_array(morphemes));
        morpheme_rows.push_back(util::json_object(std::move(morph_row)));
    }

    fold["token_fold"] = util::json_array(std::move(token_rows));
    fold["syllable_fold"] = util::json_array(std::move(syllable_rows));
    fold["morpheme_fold"] = util::json_array(std::move(morpheme_rows));
    fold["function_words"] = util::json_array(json_string_array(function_words));
    fold["content_words"] = util::json_array(json_string_array(content_words));
    fold["function_word_count"] = util::json_number((double)function_count);
    fold["content_word_count"] = util::json_number((double)content_count);

    const double denom = tokens.empty() ? 1.0 : static_cast<double>(tokens.size());
    const double fn_ratio = static_cast<double>(function_count) / denom;
    const double content_ratio = static_cast<double>(content_count) / denom;
    fold["function_word_ratio"] = util::json_number(fn_ratio);
    fold["content_word_ratio"] = util::json_number(content_ratio);

    const auto gram_taxonomy = micronaut::Mx2LmNGram::classify_gram_taxonomy(tokens);
    fold["unigram_grams"] = util::json_array(json_string_array(gram_terms_from_top(gram_taxonomy.unigrams, 12)));
    fold["bigram_grams"] = util::json_array(json_string_array(gram_terms_from_top(gram_taxonomy.bigrams, 12)));
    fold["trigram_grams"] = util::json_array(json_string_array(gram_terms_from_top(gram_taxonomy.trigrams, 12)));
    fold["quadgram_grams"] = util::json_array(json_string_array(gram_terms_from_top(gram_taxonomy.quadgrams, 8)));
    fold["coarse_grams"] = util::json_array(json_string_array(gram_terms_from_top(gram_taxonomy.coarse_grams, 8)));
    fold["function_grams"] = util::json_array(json_string_array(gram_terms_from_top(gram_taxonomy.function_grams, 8)));
    fold["tool_grams"] = util::json_array(json_string_array(gram_terms_from_top(gram_taxonomy.tool_grams, 8)));
    util::JsonObject gram_contract;
    gram_contract["grammar_layer"] = util::json_string("typed_prompt_gram_channels");
    gram_contract["classes"] = util::json_array(json_string_array(std::vector<std::string>{
        "unigram", "bigram", "trigram", "quadgram", "coarse_gram", "function_gram", "tool_gram"
    }));
    gram_contract["control_operator_binding"] = util::json_string("XCFE/K'UHUL deterministic control authority");
    gram_contract["track_alignment"] = util::json_string(governing_track.empty() ? "Mx2LM" : governing_track);
    fold["gram_contract"] = util::json_object(std::move(gram_contract));

    std::vector<std::pair<std::string, uint32_t>> ranked_atoms;
    ranked_atoms.reserve(atom_counts.size());
    for (const auto& [atom, count] : atom_counts) ranked_atoms.push_back({atom, count});
    std::sort(ranked_atoms.begin(), ranked_atoms.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });

    util::JsonArray semantic_fold;
    for (const auto& [atom, count] : ranked_atoms) {
        util::JsonObject row;
        row["atom"] = util::json_string(atom);
        row["count"] = util::json_number((double)count);
        semantic_fold.push_back(util::json_object(std::move(row)));
    }
    fold["semantic_atom_fold"] = util::json_array(std::move(semantic_fold));

    std::vector<std::string> attend_atoms;
    for (const auto& [atom, _] : ranked_atoms) {
        attend_atoms.push_back(atom);
        if (attend_atoms.size() >= 8) break;
    }
    std::vector<std::string> attend_content;
    for (const auto& tok : tokens) {
        if (is_function_word_token(tok)) continue;
        if (std::find(attend_content.begin(), attend_content.end(), tok) != attend_content.end()) continue;
        attend_content.push_back(tok);
        if (attend_content.size() >= 8) break;
    }

    util::JsonObject pop;
    pop["op"] = util::json_string("receive");
    pop["input"] = util::json_string(payload_text);

    util::JsonObject wo;
    wo["op"] = util::json_string("unfold");
    wo["token_count"] = util::json_number((double)tokens.size());
    wo["morpheme_unit_count"] = util::json_number((double)atom_order.size());

    util::JsonObject yax;
    yax["op"] = util::json_string("attend");
    yax["semantic_atoms"] = util::json_array(json_string_array(attend_atoms));
    yax["content_focus"] = util::json_array(json_string_array(attend_content));

    util::JsonObject sek;
    sek["op"] = util::json_string("relate");
    sek["relation_model"] = util::json_string("role_binding_and_cooccurrence");
    sek["content_function_split"] = util::json_string("content_vs_function");

    util::JsonObject chen;
    chen["op"] = util::json_string("verify");
    chen["balanced"] = util::json_bool(fn_ratio >= 0.35 && fn_ratio <= 0.65);
    chen["note"] = util::json_string("function/content ratio is context-sensitive; 0.5 is a prose baseline, not a hard law");

    util::JsonObject xul;
    xul["op"] = util::json_string("collapse");
    xul["response_phase"] = util::json_string(response_phase.empty() ? "Pop.fallback_projection" : response_phase);
    xul["collapsed_text"] = util::json_string(composed_text);

    util::JsonObject phases;
    phases["Pop"] = util::json_object(std::move(pop));
    phases["Wo"] = util::json_object(std::move(wo));
    phases["Yax"] = util::json_object(std::move(yax));
    phases["Sek"] = util::json_object(std::move(sek));
    phases["Ch'en"] = util::json_object(std::move(chen));
    phases["Xul"] = util::json_object(std::move(xul));
    fold["phase_cycle"] = util::json_object(std::move(phases));

    const std::string phi = response_phase.empty() ? "Pop.fallback_projection" : response_phase;
    const std::string mu = governing_track.empty() ? "Mx2LM" : governing_track;
    const std::string active_u = has_multi_morpheme ? "morpheme" : "word";

    util::JsonObject axis_u;
    axis_u["label"] = util::json_string("WHERE");
    axis_u["active"] = util::json_string(active_u);
    axis_u["available"] = util::json_array(json_string_array(
        {"character","syllable","morpheme","word","phrase","sentence"}));

    util::JsonObject axis_g;
    axis_g["label"] = util::json_string("WHAT");
    axis_g["atoms"] = util::json_array(json_string_array(attend_atoms));

    util::JsonObject axis_e;
    axis_e["label"] = util::json_string("HOW");
    axis_e["relation_model"] = util::json_string("typed_relations");
    axis_e["relation_count"] = util::json_number((double)relation_count);

    util::JsonObject axis_phi;
    axis_phi["label"] = util::json_string("DOING");
    axis_phi["phase"] = util::json_string(phi);

    util::JsonObject axis_mu;
    axis_mu["label"] = util::json_string("WHO");
    axis_mu["track"] = util::json_string(mu);

    util::JsonObject axes;
    axes["U"] = util::json_object(std::move(axis_u));
    axes["G"] = util::json_object(std::move(axis_g));
    axes["E"] = util::json_object(std::move(axis_e));
    axes["Phi"] = util::json_object(std::move(axis_phi));
    axes["mu"] = util::json_object(std::move(axis_mu));

    util::JsonObject lang_state;
    lang_state["contract"] = util::json_string("LanguageState=<U,G,E,Phi,mu>");
    lang_state["axes"] = util::json_object(std::move(axes));
    fold["language_state"] = util::json_object(std::move(lang_state));

    return util::json_object(std::move(fold));
}

static bool is_article_token(const std::string& tok) {
    return tok == "a" || tok == "an" || tok == "the";
}

static std::vector<std::string> strip_leading_articles(const std::vector<std::string>& toks) {
    size_t start = 0;
    while (start < toks.size() &&
           (is_article_token(toks[start]) ||
            toks[start] == "even" || toks[start] == "just" ||
            toks[start] == "only" || toks[start] == "also")) {
        ++start;
    }
    return std::vector<std::string>(toks.begin() + static_cast<long long>(start), toks.end());
}

static bool is_subject_pronoun_or_placeholder(const std::string& subject) {
    static const std::unordered_set<std::string> kStopSubjects = {
        "i", "you", "we", "they", "he", "she", "it",
        "this", "that", "these", "those",
        "someone", "somebody", "something",
        "who", "what", "when", "where", "why", "how"
    };
    return kStopSubjects.find(subject) != kStopSubjects.end();
}

static std::string canonical_subject_from_tokens(const std::vector<std::string>& toks) {
    auto cleaned = strip_leading_articles(toks);
    if (cleaned.empty()) return "";
    if (cleaned.size() > 3) return "";
    static const std::unordered_set<std::string> kBlocked = {
        "can", "have", "has", "had", "if", "then", "their", "our", "your", "my",
        "own", "will", "would", "should", "could", "about"
    };
    for (const auto& tok : cleaned) {
        if (kBlocked.find(tok) != kBlocked.end()) return "";
    }
    auto subject = join_words(cleaned, 5);
    return trim_copy(lower_copy(subject));
}

static std::string canonical_predicate_from_tokens(const std::vector<std::string>& toks) {
    if (toks.empty()) return "";
    auto predicate = join_words(toks, 10);
    return trim_copy(lower_copy(predicate));
}

struct Mx2LmOwnerPartTokens {
    std::string possessor;
    std::string owner;
    std::string part;
};

static bool is_possessive_pronoun_token(const std::string& tok) {
    return tok == "my" || tok == "your" || tok == "our" ||
           tok == "their" || tok == "his" || tok == "her";
}

static bool has_possessive_suffix(const std::string& tok) {
    if (tok.size() >= 2 && tok.compare(tok.size() - 2, 2, "'s") == 0) return true;
    return tok.size() >= 2 && tok.back() == '\'' && tok[tok.size() - 2] == 's';
}

static std::string normalize_entity_token(std::string tok) {
    tok = trim_copy(lower_copy(std::move(tok)));
    if (tok.empty()) return {};

    while (!tok.empty() && tok.front() == '\'') tok.erase(tok.begin());
    while (!tok.empty() && tok.back() == '\'') tok.pop_back();

    if (tok.size() >= 2 && tok.compare(tok.size() - 2, 2, "'s") == 0)
        tok.erase(tok.size() - 2);
    else if (tok.size() >= 2 && tok.back() == '\'' && tok[tok.size() - 2] == 's')
        tok.pop_back();

    tok = trim_copy(tok);
    if (tok.empty()) return {};
    if (is_article_token(tok)) return {};
    if (is_subject_pronoun_or_placeholder(tok)) return {};
    return tok;
}

static std::string normalize_property_token(std::string tok) {
    auto property = normalize_entity_token(std::move(tok));
    if (property == "colour") property = "color";
    return property;
}

static std::string append_possessive_suffix(std::string noun) {
    if (noun.empty()) return noun;
    if (!noun.empty() && (noun.back() == 's' || noun.back() == 'S'))
        return noun + "'";
    return noun + "'s";
}

static bool parse_owner_part_tokens(
    const std::vector<std::string>& raw_tokens,
    Mx2LmOwnerPartTokens* out)
{
    if (!out) return false;
    auto toks = strip_leading_articles(raw_tokens);
    if (toks.size() < 2) return false;

    const std::string part = normalize_entity_token(toks.back());
    if (part.empty()) return false;

    const size_t owner_idx = toks.size() - 2;
    const std::string owner_raw = toks[owner_idx];
    if (is_possessive_pronoun_token(owner_raw)) return false;

    std::string possessor;
    if (owner_idx > 0 && is_possessive_pronoun_token(toks[owner_idx - 1]))
        possessor = toks[owner_idx - 1];

    const bool explicit_owner = has_possessive_suffix(owner_raw) || !possessor.empty() || toks.size() == 2;
    if (!explicit_owner) return false;

    const std::string owner = normalize_entity_token(owner_raw);
    if (owner.empty()) return false;

    out->possessor = possessor;
    out->owner = owner;
    out->part = part;
    return true;
}

static std::mutex& semantic_chain_mutex() {
    static std::mutex m;
    return m;
}

static std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>>&
semantic_chain_value_counts() {
    static std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> store;
    return store;
}

static std::string semantic_chain_key(
    const std::string& owner,
    const std::string& part,
    const std::string& property)
{
    const auto o = trim_copy(lower_copy(owner));
    const auto p = trim_copy(lower_copy(part));
    const auto r = trim_copy(lower_copy(property));
    if (o.empty() || p.empty() || r.empty()) return {};
    return o + "\x1f" + p + "\x1f" + r;
}

static bool remember_semantic_chain_fact(const Mx2LmSemanticChainFact& fact) {
    const auto key = semantic_chain_key(fact.owner, fact.part, fact.property);
    const auto val = trim_copy(lower_copy(fact.value));
    if (key.empty() || val.empty()) return false;

    std::lock_guard<std::mutex> lk(semantic_chain_mutex());
    auto& value_map = semantic_chain_value_counts()[key];
    const auto it_before = value_map.find(val);
    value_map[val] += 1;
    return it_before == value_map.end();
}

static std::optional<std::string> lookup_semantic_chain_value(
    const std::string& owner,
    const std::string& part,
    const std::string& property)
{
    const auto key = semantic_chain_key(owner, part, property);
    if (key.empty()) return std::nullopt;

    std::lock_guard<std::mutex> lk(semantic_chain_mutex());
    auto it = semantic_chain_value_counts().find(key);
    if (it == semantic_chain_value_counts().end() || it->second.empty())
        return std::nullopt;

    std::string best_value;
    uint32_t best_count = 0;
    for (const auto& [value, count] : it->second) {
        if (count > best_count || (count == best_count && (best_value.empty() || value < best_value))) {
            best_value = value;
            best_count = count;
        }
    }
    if (best_value.empty()) return std::nullopt;
    return best_value;
}

static std::optional<Mx2LmSemanticChainFact> extract_semantic_chain_fact_from_clause(
    const std::string& clause,
    char terminator)
{
    if (terminator == '?') return std::nullopt;
    const auto trimmed = trim_copy(clause);
    if (trimmed.empty()) return std::nullopt;

    const auto toks = micronaut::Mx2LmNGram::tokenize(trimmed);
    if (toks.size() < 4) return std::nullopt;

    size_t has_idx = std::string::npos;
    for (size_t i = 1; i + 1 < toks.size(); ++i) {
        if (toks[i] == "has" || toks[i] == "have" || toks[i] == "had") {
            has_idx = i;
            break;
        }
    }
    if (has_idx == std::string::npos) return std::nullopt;

    std::vector<std::string> left(toks.begin(), toks.begin() + static_cast<long long>(has_idx));
    std::vector<std::string> right(toks.begin() + static_cast<long long>(has_idx + 1), toks.end());

    Mx2LmOwnerPartTokens owner_part;
    if (!parse_owner_part_tokens(left, &owner_part)) return std::nullopt;

    right = strip_leading_articles(right);
    if (right.size() < 2) return std::nullopt;

    std::string property;
    std::vector<std::string> value_tokens;
    if ((right[0] == "color" || right[0] == "colour") && right.size() >= 2) {
        property = "color";
        value_tokens.assign(right.begin() + 1, right.end());
    } else {
        property = normalize_property_token(right.back());
        value_tokens.assign(right.begin(), right.end() - 1);
    }

    if (property.empty() || value_tokens.empty()) return std::nullopt;
    const std::string value = canonical_predicate_from_tokens(value_tokens);
    if (value.empty()) return std::nullopt;

    Mx2LmSemanticChainFact fact;
    fact.possessor = owner_part.possessor;
    fact.owner = owner_part.owner;
    fact.part = owner_part.part;
    fact.property = property;
    fact.value = value;
    return fact;
}

static void learn_semantic_chain_from_payload(
    const std::string& payload_text,
    std::vector<Mx2LmSemanticChainFact>* learned_semantic_facts_out,
    std::vector<Mx2LmSemanticRelation>* semantic_relations_out)
{
    if (payload_text.empty()) return;
    std::string chunk;
    chunk.reserve(payload_text.size());

    auto emit_relations = [&](const Mx2LmSemanticChainFact& fact) {
        if (!semantic_relations_out) return;
        if (!fact.possessor.empty()) {
            semantic_relations_out->push_back(Mx2LmSemanticRelation{"self", "@possesses", fact.owner});
        }
        semantic_relations_out->push_back(Mx2LmSemanticRelation{fact.owner, "@has_part", fact.part});
        semantic_relations_out->push_back(Mx2LmSemanticRelation{fact.part, "@has_property", fact.property});
        semantic_relations_out->push_back(Mx2LmSemanticRelation{fact.property, "@value", fact.value});
    };

    auto flush_clause = [&](char terminator) {
        const auto parsed = extract_semantic_chain_fact_from_clause(chunk, terminator);
        chunk.clear();
        if (!parsed) return;

        emit_relations(*parsed);
        if (remember_semantic_chain_fact(*parsed) && learned_semantic_facts_out)
            learned_semantic_facts_out->push_back(*parsed);
    };

    for (char c : payload_text) {
        if (c == '.' || c == '!' || c == '?' || c == ';' || c == ',') {
            flush_clause(c);
        } else {
            chunk.push_back(c);
        }
    }
    flush_clause('.');
}

static std::optional<Mx2LmSemanticPropertyQuery> extract_semantic_property_query(
    const std::string& payload_text)
{
    if (payload_text.empty()) return std::nullopt;
    std::string chunk;
    chunk.reserve(payload_text.size());

    auto try_clause = [&](const std::string& clause) -> std::optional<Mx2LmSemanticPropertyQuery> {
        const auto trimmed = trim_copy(clause);
        if (trimmed.empty()) return std::nullopt;

        const auto toks = micronaut::Mx2LmNGram::tokenize(trimmed);
        if (toks.size() < 4 || toks[0] != "what") return std::nullopt;

        size_t copula_idx = std::string::npos;
        for (size_t i = 1; i < toks.size(); ++i) {
            if (toks[i] == "is" || toks[i] == "are") {
                copula_idx = i;
                break;
            }
        }
        if (copula_idx == std::string::npos || copula_idx < 2 || copula_idx + 1 >= toks.size())
            return std::nullopt;

        std::vector<std::string> property_tokens(
            toks.begin() + 1, toks.begin() + static_cast<long long>(copula_idx));
        property_tokens = strip_leading_articles(property_tokens);
        if (property_tokens.empty()) return std::nullopt;

        std::string property = normalize_property_token(property_tokens.back());
        if (property.empty()) return std::nullopt;

        std::vector<std::string> subject_tokens(
            toks.begin() + static_cast<long long>(copula_idx + 1), toks.end());
        Mx2LmOwnerPartTokens owner_part;
        if (!parse_owner_part_tokens(subject_tokens, &owner_part)) return std::nullopt;

        Mx2LmSemanticPropertyQuery out;
        out.possessor = owner_part.possessor;
        out.owner = owner_part.owner;
        out.part = owner_part.part;
        out.property = property;
        return out;
    };

    for (char c : payload_text) {
        if (c == '.' || c == '!' || c == '?' || c == ';') {
            if (auto q = try_clause(chunk); q) return q;
            chunk.clear();
        } else {
            chunk.push_back(c);
        }
    }
    return try_clause(chunk);
}

static std::string compose_semantic_property_answer(
    const Mx2LmSemanticPropertyQuery& query,
    const std::string& value)
{
    const auto value_norm = trim_copy(lower_copy(value));
    const auto owner_norm = trim_copy(lower_copy(query.owner));
    const auto part_norm = trim_copy(lower_copy(query.part));
    const auto property_norm = trim_copy(lower_copy(query.property));
    if (value_norm.empty() || owner_norm.empty() || part_norm.empty() || property_norm.empty())
        return {};

    std::string subject;
    if (query.possessor == "my") {
        subject = "Your " + append_possessive_suffix(owner_norm);
    } else {
        subject = append_possessive_suffix(to_title_case(owner_norm));
    }
    return subject + " " + part_norm + " " + property_norm + " is " + value_norm + ".";
}

static std::mutex& learned_fact_mutex() {
    static std::mutex m;
    return m;
}

static std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>>& learned_fact_counts() {
    static std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> store;
    return store;
}

static bool remember_learned_fact(const std::string& subject, const std::string& predicate) {
    const auto sub = trim_copy(lower_copy(subject));
    const auto pred = trim_copy(lower_copy(predicate));
    if (sub.empty() || pred.empty()) return false;
    if (is_subject_pronoun_or_placeholder(sub)) return false;

    std::lock_guard<std::mutex> lk(learned_fact_mutex());
    auto& subject_map = learned_fact_counts()[sub];
    const auto it_before = subject_map.find(pred);
    subject_map[pred] += 1;
    return it_before == subject_map.end();
}

static std::optional<std::string> lookup_learned_fact(const std::string& subject) {
    const auto sub = trim_copy(lower_copy(subject));
    if (sub.empty()) return std::nullopt;

    std::lock_guard<std::mutex> lk(learned_fact_mutex());
    auto sit = learned_fact_counts().find(sub);
    if (sit == learned_fact_counts().end() || sit->second.empty()) return std::nullopt;

    std::string best_pred;
    uint32_t best_count = 0;
    for (const auto& [pred, count] : sit->second) {
        if (count > best_count || (count == best_count && (best_pred.empty() || pred < best_pred))) {
            best_pred = pred;
            best_count = count;
        }
    }
    if (best_pred.empty()) return std::nullopt;
    return best_pred;
}

static std::optional<std::string> extract_definition_subject_from_payload(const std::string& payload_text) {
    if (payload_text.empty()) return std::nullopt;
    std::string chunk;
    chunk.reserve(payload_text.size());

    auto try_clause = [&](const std::string& clause) -> std::optional<std::string> {
        const auto trimmed = trim_copy(clause);
        if (trimmed.empty()) return std::nullopt;
        const auto toks = micronaut::Mx2LmNGram::tokenize(trimmed);
        if (toks.size() < 3) return std::nullopt;
        if (toks[0] != "what" || (toks[1] != "is" && toks[1] != "are")) return std::nullopt;

        std::vector<std::string> subject_tokens;
        for (size_t i = 2; i < toks.size(); ++i) {
            const auto& tok = toks[i];
            if (tok == "and" || tok == "or" || tok == "can" || tok == "do" ||
                tok == "please" || tok == "tell" || tok == "explain") {
                break;
            }
            subject_tokens.push_back(tok);
        }
        subject_tokens = strip_leading_articles(subject_tokens);
        if (subject_tokens.empty()) return std::nullopt;
        if (subject_tokens.size() > 4) return std::nullopt;

        auto subject = canonical_subject_from_tokens(subject_tokens);
        if (subject.empty() || is_subject_pronoun_or_placeholder(subject)) return std::nullopt;
        return subject;
    };

    for (char c : payload_text) {
        if (c == '.' || c == '!' || c == '?' || c == ';') {
            if (auto s = try_clause(chunk); s) return s;
            chunk.clear();
        } else {
            chunk.push_back(c);
        }
    }
    return try_clause(chunk);
}

static void learn_facts_from_payload(
    const std::string& payload_text,
    std::vector<Mx2LmLearnedFact>* learned_facts_out)
{
    if (payload_text.empty()) return;
    std::string chunk;
    chunk.reserve(payload_text.size());

    auto flush_clause = [&](char terminator) {
        const auto clause = trim_copy(chunk);
        chunk.clear();
        if (clause.empty()) return;
        if (terminator == '?') return;

        const auto toks = micronaut::Mx2LmNGram::tokenize(clause);
        if (toks.size() < 3) return;

        std::string subject;
        std::string predicate;
        bool matched = false;

        for (size_t i = 1; i + 1 < toks.size(); ++i) {
            if (toks[i] != "is" && toks[i] != "are") continue;
            std::vector<std::string> subj_toks(toks.begin(), toks.begin() + static_cast<long long>(i));
            std::vector<std::string> pred_toks(toks.begin() + static_cast<long long>(i + 1), toks.end());
            subject = canonical_subject_from_tokens(subj_toks);
            predicate = canonical_predicate_from_tokens(pred_toks);
            matched = !subject.empty() && !predicate.empty();
            if (matched) break;
        }

        // Zero-copula pattern support (e.g., "the sun a star").
        if (!matched && toks.size() >= 3) {
            if (!is_article_token(toks[0]) && is_article_token(toks[1])) {
                subject = canonical_subject_from_tokens({toks[0]});
                predicate = canonical_predicate_from_tokens(
                    std::vector<std::string>(toks.begin() + 1, toks.end()));
                matched = !subject.empty() && !predicate.empty();
            } else if (toks.size() >= 4 && is_article_token(toks[0]) && is_article_token(toks[2])) {
                subject = canonical_subject_from_tokens({toks[0], toks[1]});
                predicate = canonical_predicate_from_tokens(
                    std::vector<std::string>(toks.begin() + 2, toks.end()));
                matched = !subject.empty() && !predicate.empty();
            }
        }

        if (!matched) return;
        if (is_subject_pronoun_or_placeholder(subject)) return;
        if (subject == predicate) return;

        if (remember_learned_fact(subject, predicate) && learned_facts_out) {
            learned_facts_out->push_back(Mx2LmLearnedFact{subject, predicate});
        }
    };

    for (char c : payload_text) {
        if (c == '.' || c == '!' || c == '?' || c == ';' || c == ',') {
            flush_clause(c);
        } else {
            chunk.push_back(c);
        }
    }
    flush_clause('.');
}

static std::string compose_mx2lm_text_response(
    const std::vector<std::string>& tokens,
    const std::string& payload_text,
    const std::string& fallback_sentence,
    micronaut::Factory* style_factory,
    kuhul::GramStore* style_gram_store,
    std::vector<std::string>* intent_fragments_out,
    std::vector<Mx2LmActionGramHit>* action_gram_hits_out,
    std::vector<Mx2LmLearnedFact>* learned_facts_out,
    std::vector<Mx2LmSemanticChainFact>* learned_semantic_facts_out,
    std::vector<Mx2LmSemanticRelation>* semantic_relations_out,
    std::string* response_phase_out)
{
    auto add_intent = [&](const std::string& intent) {
        if (!intent_fragments_out || intent.empty()) return;
        if (std::find(intent_fragments_out->begin(), intent_fragments_out->end(), intent) == intent_fragments_out->end())
            intent_fragments_out->push_back(intent);
    };

    std::vector<std::string> clauses;
    auto add_clause = [&](const std::string& clause) {
        const auto trimmed = trim_copy(clause);
        if (trimmed.empty()) return;
        if (std::find(clauses.begin(), clauses.end(), trimmed) == clauses.end())
            clauses.push_back(trimmed);
    };

    learn_facts_from_payload(payload_text, learned_facts_out);
    learn_semantic_chain_from_payload(payload_text, learned_semantic_facts_out, semantic_relations_out);

    std::unordered_set<std::string> seen_hits;
    std::unordered_set<std::string> seen_uncertainty_triggers;
    std::vector<size_t> salutation_hit_indices;
    bool has_salutation_or_wellbeing = false;
    bool has_non_salutation_action = false;
    for (const auto& rule : mx2lm_action_gram_rules()) {
        if (!contains_token_sequence(tokens, rule.gram_tokens)) continue;
        const auto gram = join_words(rule.gram_tokens);
        const auto key = gram + "\x1f" + rule.intent + "\x1f" + rule.op;
        if (!seen_hits.insert(key).second) continue;
        if ((rule.op == "advisor_web_research_trigger" || rule.op == "lfm_tool_help_trigger") &&
            !seen_uncertainty_triggers.insert(rule.op).second) {
            continue;
        }

        add_intent(rule.intent);
        const bool adaptive_salutation = (rule.op == "salutation" || rule.op == "wellbeing_check");
        if (adaptive_salutation) has_salutation_or_wellbeing = true;
        else has_non_salutation_action = true;
        if (!adaptive_salutation) add_clause(rule.response_clause);
        if (action_gram_hits_out) {
            action_gram_hits_out->push_back(Mx2LmActionGramHit{
                gram,
                rule.intent,
                rule.op,
                adaptive_salutation ? std::string{} : rule.response_clause
            });
            if (adaptive_salutation)
                salutation_hit_indices.push_back(action_gram_hits_out->size() - 1);
        }
    }

    const bool has_question_boundary =
        payload_text.find('?') != std::string::npos ||
        payload_text.find(',') != std::string::npos ||
        payload_text.find(';') != std::string::npos;
    const bool has_compound_question =
        has_salutation_or_wellbeing &&
        has_non_salutation_action &&
        has_question_boundary;
    if (has_compound_question) {
        add_intent("intent:compound_question");
        if (action_gram_hits_out) {
            action_gram_hits_out->push_back(Mx2LmActionGramHit{
                "token:compound_question_boundary",
                "intent:compound_question",
                "compound_question",
                ""
            });
        }
    }

    if (has_salutation_or_wellbeing) {
        const auto style = resolve_adaptive_salutation_phase(
            payload_text,
            tokens,
            style_factory,
            style_gram_store);
        if (!style.clause.empty()) add_clause(style.clause);
        if (response_phase_out && !style.phase.empty()) *response_phase_out = style.phase;
        if (action_gram_hits_out) {
            for (const auto idx : salutation_hit_indices) {
                if (idx >= action_gram_hits_out->size()) continue;
                (*action_gram_hits_out)[idx].response_clause = style.clause;
            }
        }
    }

    if (auto semantic_query = extract_semantic_property_query(payload_text); semantic_query) {
        add_intent("intent:semantic_property_query");

        std::string semantic_clause;
        if (auto value = lookup_semantic_chain_value(
                semantic_query->owner,
                semantic_query->part,
                semantic_query->property);
            value && !value->empty()) {
            semantic_clause = compose_semantic_property_answer(*semantic_query, *value);
            add_clause(semantic_clause);

            if (semantic_relations_out) {
                if (!semantic_query->possessor.empty()) {
                    semantic_relations_out->push_back(
                        Mx2LmSemanticRelation{"self", "@possesses", semantic_query->owner});
                }
                semantic_relations_out->push_back(
                    Mx2LmSemanticRelation{semantic_query->owner, "@has_part", semantic_query->part});
                semantic_relations_out->push_back(
                    Mx2LmSemanticRelation{semantic_query->part, "@has_property", semantic_query->property});
                semantic_relations_out->push_back(
                    Mx2LmSemanticRelation{semantic_query->property, "@value", *value});
            }
        }

        if (action_gram_hits_out) {
            action_gram_hits_out->push_back(Mx2LmActionGramHit{
                "what " + semantic_query->property + " is " + semantic_query->owner + " " + semantic_query->part,
                "intent:semantic_property_query",
                "semantic_property_query",
                semantic_clause
            });
        }
    }

    if (auto def_subject = extract_definition_subject_from_payload(payload_text); def_subject) {
        add_intent("intent:definition_query");
        std::string definition_clause;
        if (auto learned = lookup_learned_fact(*def_subject); learned && !learned->empty()) {
            definition_clause = to_title_case(*def_subject) + " is " + *learned + ".";
            add_clause(definition_clause);
        }
        if (action_gram_hits_out) {
            action_gram_hits_out->push_back(Mx2LmActionGramHit{
                "what is " + *def_subject,
                "intent:definition_query",
                "definition_query",
                definition_clause
            });
        }
    }

    if (!clauses.empty()) {
        std::string merged;
        for (size_t i = 0; i < clauses.size(); ++i) {
            if (i) merged.push_back(' ');
            merged += clauses[i];
        }
        if (response_phase_out && response_phase_out->empty())
            *response_phase_out = "Pop.default_response";
        return merged;
    }

    if (response_phase_out && response_phase_out->empty())
        *response_phase_out = "Pop.fallback_projection";
    return fallback_sentence;
}

// Parse "key=value key2=value2 ..." from an mx2lm payload string.
// Values may be multi-word; each value extends to the start of the next key= or end of string.
static std::unordered_map<std::string, std::string> parse_mx2lm_params(const std::string& text) {
    std::unordered_map<std::string, std::string> result;
    // Collect all (key_start, eq_pos) pairs — a "key" is a run of non-space chars ending at '='.
    std::vector<std::pair<size_t, size_t>> eq_pos;
    for (size_t i = 0; i < text.size(); ) {
        auto eq = text.find('=', i);
        if (eq == std::string::npos) break;
        size_t ks = eq;
        while (ks > 0 && !std::isspace((unsigned char)text[ks - 1])) --ks;
        if (ks < eq) eq_pos.push_back({ks, eq});
        i = eq + 1;
    }
    for (size_t idx = 0; idx < eq_pos.size(); ++idx) {
        const auto [ks, eq] = eq_pos[idx];
        std::string key = text.substr(ks, eq - ks);
        size_t val_start = eq + 1;
        size_t val_end = (idx + 1 < eq_pos.size()) ? eq_pos[idx + 1].first : text.size();
        while (val_end > val_start && std::isspace((unsigned char)text[val_end - 1])) --val_end;
        std::string value = (val_start <= val_end && val_start < text.size())
            ? text.substr(val_start, val_end - val_start) : "";
        while (!value.empty() && std::isspace((unsigned char)value.front())) value = value.substr(1);
        while (!value.empty() && std::isspace((unsigned char)value.back()))  value.pop_back();
        if (!key.empty()) result[std::move(key)] = std::move(value);
    }
    return result;
}

// Build a typed card object from a memory-u or tool-u payload.
// Returns null if the payload is not a recognised card-emitting track.
static util::JsonPtr build_track_card(const std::string& payload) {
    const auto low = lower_copy(trim_copy(payload));
    const bool is_memory = low.find("memory-u") != std::string::npos;
    const bool is_tool   = !is_memory && low.find("tool-u")   != std::string::npos;
    const bool is_chat   = !is_memory && !is_tool && low.find("chat-u")   != std::string::npos;
    const bool is_email  = !is_memory && !is_tool && !is_chat && low.find("email-u") != std::string::npos;
    const bool is_rick   = !is_memory && !is_tool && !is_chat && !is_email && low.find("rick-u") != std::string::npos;
    if (!is_memory && !is_tool && !is_chat && !is_email && !is_rick) return util::json_null();

    const auto params = parse_mx2lm_params(payload);
    auto get = [&](const char* k) -> std::string {
        auto it = params.find(k);
        return it != params.end() ? it->second : "";
    };

    util::JsonObject card;
    const char* kind = is_memory ? "memory_card.v1"
                     : is_tool   ? "tool_card.v1"
                     : is_chat   ? "chat_card.v1"
                     : is_email  ? "email_card.v1"
                                 : "rick_card.v1";
    card["@kind"] = util::json_string(kind);
    for (const char* k : {"op", "category", "key", "value", "phase", "channel", "tool", "bank",
                           "topic", "reason", "sentiment", "tone", "input",
                           "to", "cc", "from", "subject", "body", "id", "query",
                           "verse", "target", "artist"}) {
        auto v = get(k);
        if (!v.empty()) card[k] = util::json_string(v);
    }
    const auto cv = get("card_view");
    if (!cv.empty()) card["card_view"] = util::json_bool(cv == "true" || cv == "1");
    return util::json_object(std::move(card));
}

static util::JsonPtr mx2lm_ngram_payload(
    const std::string& text,
    const std::string& source,
    const manifest::ManifestRegistry& manifest_reg,
    const DictionaryLexicon* dictionary_lexicon,
    micronaut::Factory* style_factory = nullptr,
    kuhul::GramStore* style_gram_store = nullptr)
{
    const auto t = trim_copy(text);
    if (t.empty()) return util::json_null();

    const auto low = lower_copy(t);
    std::string payload;

    auto pick_after = [&](const std::string& marker, bool colon_after_at = false) {
        auto pos = low.find(marker);
        if (pos == std::string::npos) return;
        auto start = pos + marker.size();
        auto raw = trim_copy(t.substr(start));
        if (colon_after_at && !raw.empty() && raw[0] == ':')
            raw = trim_copy(raw.substr(1));
        payload = std::move(raw);
    };

    // AIML wrapper mode: <mx2lm>...</mx2lm>
    // Also tolerates legacy malformed close marker "<mx2lm>" (no slash).
    auto open = low.find("<mx2lm>");
    if (open != std::string::npos) {
        auto start = open + 7;
        auto close = low.find("</mx2lm>", start);
        if (close == std::string::npos) close = low.find("<mx2lm>", start);
        if (close == std::string::npos) close = t.size();
        payload = trim_copy(t.substr(start, close - start));
    } else if (low.rfind("mx2lm:", 0) == 0) {
        pick_after("mx2lm:");
    } else if (low.rfind("@mx2lm", 0) == 0) {
        pick_after("@mx2lm", true);
    } else {
        // Embedded marker mode so other AIML tracks can forward inline calls.
        pick_after("mx2lm:");
        if (payload.empty()) pick_after("@mx2lm", true);
    }
    if (payload.empty()) return util::json_null();

    auto model = micronaut::Mx2LmNGram::build(payload);
    const auto taxonomy = micronaut::Mx2LmNGram::classify_gram_taxonomy(model.tokens);
    auto top_bi = micronaut::Mx2LmNGram::top(model.bigrams, 6);
    auto top_tri = micronaut::Mx2LmNGram::top(model.trigrams, 6);
    auto top_quad = micronaut::Mx2LmNGram::top(model.quadgrams, 6);
    auto top_coarse = top_ranked_grams(taxonomy.coarse_grams, 6);
    auto top_function = top_ranked_grams(taxonomy.function_grams, 6);
    auto top_tool = top_ranked_grams(taxonomy.tool_grams, 6);
    const auto typed_grams = micronaut::Mx2LmNGram::typed_grams(taxonomy, 6);
    const auto gram_container = micronaut::Mx2LmNGram::compile_gram_container(taxonomy);
    auto projection = micronaut::Mx2LmNGram::compose(model.tokens);
    std::vector<std::string> intent_fragments;
    std::vector<Mx2LmActionGramHit> action_gram_hits;
    std::vector<Mx2LmLearnedFact> learned_facts;
    std::vector<Mx2LmSemanticChainFact> learned_semantic_facts;
    std::vector<Mx2LmSemanticRelation> semantic_relations;
    std::string response_phase;
    const auto inferred_track = infer_micronaut_from_text(payload);
    const std::string governing_track = inferred_track.empty() ? "Mx2LM" : inferred_track;
    const auto pattern_container = micronaut::Mx2LmNGram::compile_pattern_container(
        payload,
        gram_container,
        governing_track,
        governing_track == "GREETING-µ" ? "Pop" : "Wo",
        "response_composed",
        lower_copy(governing_track) == lower_copy("GREETING-µ") ? "greeting" : "");
    const auto vector_projection_container =
        micronaut::Mx2LmNGram::compile_vector_projection_container(
            gram_container,
            pattern_container);
    const auto composed_text = compose_mx2lm_text_response(
        model.tokens,
        payload,
        projection.sentence,
        style_factory,
        style_gram_store,
        &intent_fragments,
        &action_gram_hits,
        &learned_facts,
        &learned_semantic_facts,
        &semantic_relations,
        &response_phase);

    bool has_resolved_action = false;
    bool has_unresolved_definition = false;
    bool has_unresolved_semantic_property_query = false;
    for (const auto& hit : action_gram_hits) {
        const bool resolved_clause = !trim_copy(hit.response_clause).empty();
        if (resolved_clause) has_resolved_action = true;
        if (hit.op == "definition_query" && !resolved_clause)
            has_unresolved_definition = true;
        if (hit.op == "semantic_property_query" && !resolved_clause)
            has_unresolved_semantic_property_query = true;
    }

    const auto payload_low = lower_copy(payload);
    const bool has_question_mark = payload.find('?') != std::string::npos;
    const bool starts_with_question_cue = is_question_like_text(payload_low);

    // MISS = WEB-RESEARCH: unresolved question-like input should fall through to miss handling
    // so key_query can emit web_research dispatch metadata instead of mirror text.
    if (has_unresolved_definition ||
        has_unresolved_semantic_property_query ||
        ((has_question_mark || starts_with_question_cue) && !has_resolved_action)) {
        return util::json_null();
    }

    util::JsonObject obj;
    obj["ok"]        = util::json_bool(true);
    obj["source"]    = util::json_string(source);
    obj["invoke"]    = util::json_string("@mx2lm");
    obj["response"]  = util::json_string("<mx2lm>{ngrams}</mx2lm>");
    obj["text"]      = util::json_string(composed_text);
    if (!response_phase.empty())
        obj["response_phase"] = util::json_string(response_phase);
    obj["governing_track"] = util::json_string(governing_track);
    {
        util::JsonObject activation;
        activation["mode"] = util::json_string("on_demand");
        activation["required"] = util::json_bool(true);
        activation["reason"] = util::json_string("semantic_composition_requested");
        activation["track"] = util::json_string(governing_track);
        obj["activation"] = util::json_object(std::move(activation));
    }
    obj["payload"]   = util::json_string(payload);
    obj["tokens"]    = util::json_number((double)model.tokens.size());
    obj["unigram_role"] = util::json_string("semantic_address_layer");
    obj["ngram_role"] = util::json_string("context_association_evidence_layer");
    obj["full_gram"] = util::json_string(join_words(model.tokens));

    // PLAN-µ is a bounded projection: expose a task-list card alongside the
    // normal ngram evidence. The card carries no execution authority.
    if (payload_low.rfind("plan:", 0) == 0) {
        util::JsonObject card;
        card["id"] = util::json_string("plan");
        card["label"] = util::json_string("Plan");
        card["track"] = util::json_string("PLAN-µ");
        card["goal"] = util::json_string(join_words(model.tokens));
        card["owner"] = util::json_string("PLAN-µ");
        card["phase"] = util::json_string("Wo");
        card["status"] = util::json_string("proposed");
        card["next"] = util::json_string("Yax");
        card["authority"] = util::json_string("none");
        util::JsonArray actions;
        const auto plan_params = parse_mx2lm_params(payload);
        const auto verbs_it = plan_params.find("verbs");
        const std::string declared_verbs = verbs_it == plan_params.end()
            ? "refine,edit,verify,accept,execute,pause" : verbs_it->second;
        size_t verb_start = 0;
        while (verb_start <= declared_verbs.size()) {
            const auto comma = declared_verbs.find(',', verb_start);
            const auto verb = trim_copy(declared_verbs.substr(verb_start,
                comma == std::string::npos ? std::string::npos : comma - verb_start));
            if (!verb.empty()) actions.push_back(util::json_string(verb));
            if (comma == std::string::npos) break;
            verb_start = comma + 1;
        }
        card["actions"] = util::json_array(std::move(actions));
        util::JsonArray tasks;
        util::JsonObject task;
        task["id"] = util::json_string("task-1");
        task["text"] = util::json_string(join_words(model.tokens));
        task["status"] = util::json_string("proposed");
        task["phase"] = util::json_string("Wo");
        tasks.push_back(util::json_object(std::move(task)));
        card["tasks"] = util::json_array(std::move(tasks));
        card["dependencies"] = util::json_array(util::JsonArray{});
        obj["plan_card"] = util::json_object(std::move(card));
    }

    {
        util::JsonArray addresses;
        std::unordered_set<std::string> seen;
        for (const auto& tok : model.tokens) {
            const auto lemma = resolve_lexicon_lemma(tok, dictionary_lexicon);
            if (lemma.empty()) continue;
            if (!seen.insert(lemma).second) continue;

            std::string definition;
            if (dictionary_lexicon) {
                if (auto it = dictionary_lexicon->definition_by_word.find(lemma);
                    it != dictionary_lexicon->definition_by_word.end()) {
                    definition = it->second;
                }
            }
            const auto key = lexicon_key_for_word(lemma);
            const auto class_name = lexicon_class_for_word(lemma, definition);
            const auto synonyms = dictionary_synonyms_for_word(dictionary_lexicon, lemma, 6);

            util::JsonObject row;
            row["token"] = util::json_string(tok);
            row["lemma"] = util::json_string(lemma);
            row["key"] = util::json_string(key);
            row["unit"] = util::json_string("unigram");
            row["class"] = util::json_string(class_name);
            row["book"] = util::json_string(lexicon_book_for_key(key));
            row["role"] = util::json_string("semantic_address");
            if (!definition.empty())
                row["definition"] = util::json_string(definition);
            if (!synonyms.empty()) {
                util::JsonArray syn_arr;
                for (const auto& s : synonyms) syn_arr.push_back(util::json_string(s));
                row["synonyms"] = util::json_array(std::move(syn_arr));
            }
            addresses.push_back(util::json_object(std::move(row)));
        }
        obj["semantic_addresses"] = util::json_array(std::move(addresses));
    }
    if (!intent_fragments.empty()) {
        obj["intent_fragments"] = util::json_array(json_string_array(intent_fragments));
        obj["intent_count"] = util::json_number((double)intent_fragments.size());
    }
    if (!action_gram_hits.empty()) {
        bool has_compound_question_token = false;
        bool has_advisor_web_research_trigger = false;
        bool has_lfm_tool_help_trigger = false;
        util::JsonArray hits_arr;
        for (const auto& hit : action_gram_hits) {
            util::JsonObject row;
            row["gram"] = util::json_string(hit.gram);
            row["intent"] = util::json_string(hit.intent);
            row["op"] = util::json_string(hit.op);
            row["response_clause"] = util::json_string(hit.response_clause);
            hits_arr.push_back(util::json_object(std::move(row)));
            if (hit.op == "compound_question")
                has_compound_question_token = true;
            if (hit.op == "advisor_web_research_trigger")
                has_advisor_web_research_trigger = true;
            if (hit.op == "lfm_tool_help_trigger")
                has_lfm_tool_help_trigger = true;
        }
        obj["action_grams"] = util::json_array(std::move(hits_arr));
        obj["action_gram_count"] = util::json_number((double)action_gram_hits.size());
        if (has_compound_question_token || has_advisor_web_research_trigger || has_lfm_tool_help_trigger) {
            util::JsonArray token_flags;
            if (has_compound_question_token)
                token_flags.push_back(util::json_string("compound_question_boundary"));
            if (has_advisor_web_research_trigger)
                token_flags.push_back(util::json_string("advisor_web_research_trigger"));
            if (has_lfm_tool_help_trigger)
                token_flags.push_back(util::json_string("lfm_tool_help_trigger"));
            obj["token_flags"] = util::json_array(std::move(token_flags));
        }
        if (has_advisor_web_research_trigger || has_lfm_tool_help_trigger) {
            obj["status"] = util::json_string("sandbox_candidate");
            obj["authority"] = util::json_string("none");
            obj["commit_phase"] = util::json_string("Ch'en_pending");
            obj["sandboxed"] = util::json_bool(true);

            util::JsonObject sandbox;
            sandbox["mode"] = util::json_string("response_only");
            sandbox["aiml_mutation"] = util::json_string("candidate_only");
            sandbox["promotion_gate"] = util::json_string("Ch'en_to_Xul_verification_required");
            sandbox["learning_surface"] = util::json_string("aiml_pattern_template_candidate");
            obj["sandbox"] = util::json_object(std::move(sandbox));

            util::JsonArray learned_candidates;
            auto push_candidate = [&](const std::string& event_name,
                                      const std::string& pattern,
                                      const std::string& template_text,
                                      const std::string& track) {
                util::JsonObject row;
                row["event"] = util::json_string(event_name);
                row["pattern"] = util::json_string(pattern);
                row["template"] = util::json_string(template_text);
                row["track"] = util::json_string(track);
                row["authority"] = util::json_string("none");
                row["commit_phase"] = util::json_string("Ch'en_pending");
                row["status"] = util::json_string("candidate");
                learned_candidates.push_back(util::json_object(std::move(row)));
            };

            if (has_advisor_web_research_trigger) {
                push_candidate(
                    "advisor_web_research_trigger",
                    "I DON'T KNOW THE ANSWER TO THAT QUESTION *",
                    "<mx2lm>search: event=advisor_web_research_trigger cue=miss_low_confidence advisor=ADVISOR-µ expert=WEB-RESEARCH-µ phase=Yax next=Ch'en card_view=true authority=none proposal=ready <star/></mx2lm>",
                    "ADVISOR-µ");
            }
            if (has_lfm_tool_help_trigger) {
                push_candidate(
                    "lfm_tool_help_trigger",
                    "I'M NOT SURE WHAT TOOL TO USE *",
                    "<mx2lm>chat-u op=tool_selection_help event=lfm_tool_help_trigger cue=tool_uncertain advisor=ADVISOR-µ expert=CHAT-u phase=Wo next=Sek card_view=true authority=none proposal=ready <star/></mx2lm>",
                    "CHAT-µ");
            }
            obj["learnable_aiml_candidates"] = util::json_array(std::move(learned_candidates));
        }
    }
    if (!learned_facts.empty()) {
        util::JsonArray learned_arr;
        for (const auto& fact : learned_facts) {
            util::JsonObject row;
            row["subject"] = util::json_string(fact.subject);
            row["predicate"] = util::json_string(fact.predicate);
            learned_arr.push_back(util::json_object(std::move(row)));
        }
        obj["learned_facts"] = util::json_array(std::move(learned_arr));
        obj["learned_fact_count"] = util::json_number((double)learned_facts.size());
    }
    if (!learned_semantic_facts.empty()) {
        util::JsonArray learned_arr;
        for (const auto& fact : learned_semantic_facts) {
            util::JsonObject row;
            row["possessor"] = util::json_string(fact.possessor);
            row["owner"] = util::json_string(fact.owner);
            row["part"] = util::json_string(fact.part);
            row["property"] = util::json_string(fact.property);
            row["value"] = util::json_string(fact.value);
            learned_arr.push_back(util::json_object(std::move(row)));
        }
        obj["learned_semantic_facts"] = util::json_array(std::move(learned_arr));
        obj["learned_semantic_fact_count"] = util::json_number((double)learned_semantic_facts.size());
    }

    auto to_ranked_rows = [](const std::vector<std::pair<std::string, uint32_t>>& grams,
                             const std::string& role) {
        util::JsonArray arr;
        for (const auto& [gram, count] : grams) {
            util::JsonObject row;
            row["gram"] = util::json_string(gram);
            row["count"] = util::json_number(static_cast<double>(count));
            row["width"] = util::json_number(static_cast<double>(micronaut::Mx2LmNGram::tokenize(gram).size()));
            row["role"] = util::json_string(role);
            arr.push_back(util::json_object(std::move(row)));
        }
        return arr;
    };
    obj["bigrams"] = util::json_array(to_ranked_rows(top_bi, "context_association"));
    obj["trigrams"] = util::json_array(to_ranked_rows(top_tri, "context_association"));
    obj["quadgrams"] = util::json_array(to_ranked_rows(top_quad, "context_association"));
    obj["coarse_grams"] = util::json_array(to_ranked_rows(top_coarse, "structural_shape"));
    obj["function_grams"] = util::json_array(to_ranked_rows(top_function, "function_word_structure"));
    obj["tool_grams"] = util::json_array(to_ranked_rows(top_tool, "tool_dispatch_intent"));
    obj["gram_channels"] = util::json_array([&]() {
        util::JsonArray arr;
        for (const auto& row : typed_grams) {
            util::JsonObject j;
            j["gram"] = util::json_string(row.gram);
            j["class"] = util::json_string(row.gram_class);
            j["role"] = util::json_string(row.role);
            j["phase"] = util::json_string(row.phase);
            j["count"] = util::json_number(static_cast<double>(row.count));
            j["width"] = util::json_number(static_cast<double>(row.width));
            j["tokens"] = util::json_array(json_string_array(row.tokens));
            arr.push_back(util::json_object(std::move(j)));
        }
        return arr;
    }());
    obj["mx2lm_gram_container"] = util::json_object([&]() {
        util::JsonObject root;
        root["@type"] = util::json_string("mx2lm.gram_container.v1");
        root["key"] = util::json_string(gram_container.key);
        root["namespace"] = util::json_string(gram_container.namespace_id);
        root["authority"] = util::json_string(gram_container.authority);
        root["phase"] = util::json_string(gram_container.phase);
        root["total_grams"] = util::json_number(static_cast<double>(gram_container.grams.size()));
        root["total_observations"] = util::json_number(static_cast<double>(gram_container.total_observations));

        util::JsonObject counts;
        counts["unigram"] = util::json_number(static_cast<double>(gram_container.unigram_count));
        counts["bigram"] = util::json_number(static_cast<double>(gram_container.bigram_count));
        counts["trigram"] = util::json_number(static_cast<double>(gram_container.trigram_count));
        counts["quadgram"] = util::json_number(static_cast<double>(gram_container.quadgram_count));
        counts["coarse_gram"] = util::json_number(static_cast<double>(gram_container.coarse_gram_count));
        counts["function_gram"] = util::json_number(static_cast<double>(gram_container.function_gram_count));
        counts["tool_gram"] = util::json_number(static_cast<double>(gram_container.tool_gram_count));
        root["channel_counts"] = util::json_object(std::move(counts));

        util::JsonObject channels;
        const std::vector<std::string> class_order = {
            "unigram", "bigram", "trigram", "quadgram",
            "coarse_gram", "function_gram", "tool_gram"
        };
        for (const auto& klass : class_order) {
            util::JsonArray rows;
            for (const auto& row : gram_container.grams) {
                if (row.gram_class != klass) continue;
                util::JsonObject j;
                j["gram"] = util::json_string(row.gram);
                j["class"] = util::json_string(row.gram_class);
                j["role"] = util::json_string(row.role);
                j["phase"] = util::json_string(row.phase);
                j["count"] = util::json_number(static_cast<double>(row.count));
                j["width"] = util::json_number(static_cast<double>(row.width));
                j["tokens"] = util::json_array(json_string_array(row.tokens));
                rows.push_back(util::json_object(std::move(j)));
            }
            channels[klass] = util::json_array(std::move(rows));
        }
        root["channels"] = util::json_object(std::move(channels));
        return root;
    }());
    obj["mx2lm_pattern_container"] = util::json_object([&]() {
        util::JsonObject root;
        root["@type"] = util::json_string("mx2lm.pattern_container.v1");
        root["key"] = util::json_string(pattern_container.key);
        root["namespace"] = util::json_string(pattern_container.namespace_id);
        root["authority"] = util::json_string(pattern_container.authority);
        root["phase"] = util::json_string(pattern_container.phase);
        root["event_count"] = util::json_number(static_cast<double>(pattern_container.events.size()));

        util::JsonArray events;
        for (const auto& event : pattern_container.events) {
            util::JsonObject row;
            row["source_track"] = util::json_string(event.source_track);
            row["source_phase"] = util::json_string(event.source_phase);
            row["target_namespace"] = util::json_string(event.target_namespace);
            row["surface"] = util::json_string(event.surface);
            row["roles"] = util::json_array(json_string_array(event.roles));
            row["gram_container"] = util::json_string(event.gram_container_key);
            row["outcome_kind"] = util::json_string(event.outcome_kind);
            row["outcome_status"] = util::json_string(event.outcome_status);
            row["accepted"] = util::json_bool(event.accepted);
            if (!event.card.empty())
                row["card"] = util::json_string(event.card);
            events.push_back(util::json_object(std::move(row)));
        }
        root["events"] = util::json_array(std::move(events));
        return root;
    }());
    obj["vector_projection_container"] = util::json_object([&]() {
        util::JsonObject root;
        root["@type"] = util::json_string("vector.projection_container.v1");
        root["key"] = util::json_string(vector_projection_container.key);
        root["namespace"] = util::json_string(vector_projection_container.namespace_id);
        root["authority"] = util::json_string(vector_projection_container.authority);
        root["phase"] = util::json_string(vector_projection_container.phase);
        root["source_semantic_container"] =
            util::json_string(vector_projection_container.source_semantic_container);
        root["projection_id"] =
            util::json_string(vector_projection_container.projection_id);
        root["generation"] =
            util::json_number(static_cast<double>(vector_projection_container.generation));
        root["learner_version"] =
            util::json_string(vector_projection_container.learner_version);
        root["topology_version"] =
            util::json_string(vector_projection_container.topology_version);
        root["relation_model"] =
            util::json_string(vector_projection_container.relation_model);
        root["alignment"] =
            util::json_string(vector_projection_container.alignment);
        root["similarity_scope"] =
            util::json_string(vector_projection_container.similarity_scope);
        root["dimensions"] =
            util::json_number(static_cast<double>(vector_projection_container.dimensions));
        root["projection_count"] =
            util::json_number(static_cast<double>(vector_projection_container.projections.size()));
        root["semantic_authority"] = util::json_bool(false);
        root["responsibility"] =
            util::json_string("derived_geometry_projection_only");

        util::JsonArray projections;
        for (const auto& p : vector_projection_container.projections) {
            util::JsonObject row;
            row["key"] = util::json_string(p.key);
            row["source_key"] = util::json_string(p.source_key);
            row["source_track"] = util::json_string(p.source_track);
            row["source_phase"] = util::json_string(p.source_phase);
            row["projection_track"] = util::json_string(p.projection_track);
            row["basis"] = util::json_string(p.basis);
            row["x"] = util::json_number(static_cast<double>(p.x));
            row["y"] = util::json_number(static_cast<double>(p.y));
            row["z"] = util::json_number(static_cast<double>(p.z));
            row["norm"] = util::json_number(static_cast<double>(p.norm));
            row["semantic_authority"] = util::json_bool(p.semantic_authority);
            projections.push_back(util::json_object(std::move(row)));
        }
        root["projections"] = util::json_array(std::move(projections));
        return root;
    }());

    util::JsonArray control_arr;
    for (const auto& cb : projection.control_bindings) {
        util::JsonObject row;
        row["token"] = util::json_string(cb.token);
        row["operator"] = util::json_string(cb.operator_name);
        row["book"] = util::json_string(cb.book_key);
        row["detect_phase"] = util::json_string(cb.detect_phase);
        row["emit_phase"] = util::json_string(cb.emit_phase);
        row["semantic_role"] = util::json_string(cb.semantic_role);
        control_arr.push_back(util::json_object(std::move(row)));
    }
    obj["control_bindings"] = util::json_array(std::move(control_arr));

    util::JsonArray node_arr;
    for (const auto& node : projection.nodes) {
        util::JsonObject row;
        row["id"] = util::json_string(node.id);
        row["kind"] = util::json_string(node.kind);
        row["meaning_key"] = util::json_string(node.meaning_key);
        if (!node.operator_name.empty())
            row["operator"] = util::json_string(node.operator_name);
        if (!node.book_key.empty())
            row["book"] = util::json_string(node.book_key);
        node_arr.push_back(util::json_object(std::move(row)));
    }
    obj["nodes"] = util::json_array(std::move(node_arr));

    util::JsonArray rel_arr;
    for (const auto& rel : projection.relations) {
        util::JsonObject row;
        row["from"] = util::json_string(rel.from);
        row["relation"] = util::json_string(rel.relation);
        row["to"] = util::json_string(rel.to);
        row["association_weight"] = util::json_number((double)rel.association_weight);
        row["weight_kind"] = util::json_string(rel.weight_kind);
        // Compatibility key for older clients.
        row["weight"] = util::json_number((double)rel.association_weight);
        rel_arr.push_back(util::json_object(std::move(row)));
    }
    for (const auto& rel : semantic_relations) {
        util::JsonObject row;
        row["from"] = util::json_string(rel.from);
        row["relation"] = util::json_string(rel.relation);
        row["to"] = util::json_string(rel.to);
        row["association_weight"] = util::json_number(1.0);
        row["weight_kind"] = util::json_string("semantic_attention");
        row["weight"] = util::json_number(1.0);
        rel_arr.push_back(util::json_object(std::move(row)));
    }
    obj["relations"] = util::json_array(std::move(rel_arr));

    util::JsonArray key_refs_arr;
    std::unordered_set<std::string> key_ref_dedup;
    auto resolve_manifest_key = [&](const std::string& raw_key) -> std::string {
        const auto k = trim_copy(raw_key);
        if (k.empty()) return "";
        if (manifest_reg.has(k)) return k;

        std::vector<std::string> candidates;
        candidates.push_back(lower_copy(k));
        if (!k.empty() && k[0] != '@') candidates.push_back("@" + k);
        if (!k.empty() && k[0] != '@') candidates.push_back("@" + lower_copy(k));
        if (!k.empty() && k.back() != ':') {
            candidates.push_back(k + ":");
            candidates.push_back(lower_copy(k) + ":");
            std::string up = lower_copy(k);
            std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            candidates.push_back(up + ":");
        }
        for (const auto& c : candidates) {
            if (manifest_reg.has(c)) return c;
        }
        return "";
    };

    auto push_key_reference = [&](const std::string& key,
                                  const std::string& source_kind,
                                  const std::string& source_value) {
        const auto k = trim_copy(key);
        if (k.empty()) return;
        const auto sk = trim_copy(source_kind);
        const auto sv = trim_copy(source_value);
        const auto dedup_key = k + "\x1f" + sk + "\x1f" + sv;
        if (!key_ref_dedup.insert(dedup_key).second) return;

        util::JsonObject row;
        row["key"] = util::json_string(k);
        row["source_kind"] = util::json_string(sk);
        row["source_value"] = util::json_string(sv);

        const auto resolved_key = resolve_manifest_key(k);
        const bool key_exists = !resolved_key.empty();
        row["resolved"] = util::json_bool(key_exists);
        if (key_exists && resolved_key != k)
            row["resolved_key"] = util::json_string(resolved_key);

        util::JsonArray expanded_arr;
        if (key_exists) {
            std::unordered_set<std::string> exp_dedup;
            auto expanded = manifest_reg.resolve(resolved_key);
            for (const auto& v : expanded) {
                if (v.empty()) continue;
                if (!exp_dedup.insert(v).second) continue;
                expanded_arr.push_back(util::json_string(v));
                if (expanded_arr.size() >= 8) break;
            }
        }
        row["expansion"] = util::json_array(std::move(expanded_arr));
        key_refs_arr.push_back(util::json_object(std::move(row)));
    };

    std::unordered_set<std::string> seen_tokens;
    for (const auto& tok : model.tokens) {
        if (!seen_tokens.insert(tok).second) continue;
        push_key_reference(tok, "token", tok);
    }
    for (const auto& [gram, _] : top_bi)
        push_key_reference(gram, "ngram", gram);
    for (const auto& [gram, _] : top_tri)
        push_key_reference(gram, "ngram", gram);
    for (const auto& [gram, _] : top_quad)
        push_key_reference(gram, "ngram", gram);
    for (const auto& [gram, _] : top_coarse)
        push_key_reference(gram, "coarse_gram", gram);
    for (const auto& [gram, _] : top_function)
        push_key_reference(gram, "function_gram", gram);
    for (const auto& [gram, _] : top_tool)
        push_key_reference(gram, "tool_gram", gram);

    for (const auto& cb : projection.control_bindings) {
        push_key_reference(cb.book_key, "control_book", cb.token);
        push_key_reference(cb.semantic_role, "semantic_role", cb.token);
        push_key_reference("@phase:" + cb.detect_phase, "phase_anchor", cb.token);
        push_key_reference("@phase:" + cb.emit_phase, "phase_anchor", cb.token);
    }
    for (const auto& node : projection.nodes)
        push_key_reference(node.meaning_key, "meaning_key", node.id);

    if (!inferred_track.empty()) {
        push_key_reference(inferred_track, "inferred_track", payload);
        push_key_reference("@node:" + inferred_track, "node_anchor", payload);
    }
    obj["key_references"] = util::json_array(std::move(key_refs_arr));

    struct RankedGram {
        std::string gram;
        uint32_t count = 0;
        uint32_t width = 0;
        std::vector<std::string> tokens;
    };

    std::vector<RankedGram> ranked_grams;
    ranked_grams.reserve(top_bi.size() + top_tri.size() + top_quad.size() + top_coarse.size() + top_function.size() + top_tool.size());
    for (const auto& [gram, count] : top_bi) {
        RankedGram rg;
        rg.gram = gram;
        rg.count = count;
        rg.width = 2;
        rg.tokens = micronaut::Mx2LmNGram::tokenize(gram);
        ranked_grams.push_back(std::move(rg));
    }
    for (const auto& [gram, count] : top_tri) {
        RankedGram rg;
        rg.gram = gram;
        rg.count = count;
        rg.width = 3;
        rg.tokens = micronaut::Mx2LmNGram::tokenize(gram);
        ranked_grams.push_back(std::move(rg));
    }
    for (const auto& [gram, count] : top_quad) {
        RankedGram rg;
        rg.gram = gram;
        rg.count = count;
        rg.width = 4;
        rg.tokens = micronaut::Mx2LmNGram::tokenize(gram);
        ranked_grams.push_back(std::move(rg));
    }
    for (const auto& [gram, count] : top_coarse) {
        RankedGram rg;
        rg.gram = gram;
        rg.count = count;
        rg.width = static_cast<uint32_t>(micronaut::Mx2LmNGram::tokenize(gram).size());
        rg.tokens = micronaut::Mx2LmNGram::tokenize(gram);
        ranked_grams.push_back(std::move(rg));
    }
    for (const auto& [gram, count] : top_function) {
        RankedGram rg;
        rg.gram = gram;
        rg.count = count;
        rg.width = static_cast<uint32_t>(micronaut::Mx2LmNGram::tokenize(gram).size());
        rg.tokens = micronaut::Mx2LmNGram::tokenize(gram);
        ranked_grams.push_back(std::move(rg));
    }
    for (const auto& [gram, count] : top_tool) {
        RankedGram rg;
        rg.gram = gram;
        rg.count = count;
        rg.width = static_cast<uint32_t>(micronaut::Mx2LmNGram::tokenize(gram).size());
        rg.tokens = micronaut::Mx2LmNGram::tokenize(gram);
        ranked_grams.push_back(std::move(rg));
    }

    util::JsonArray ngram_ref_arr;
    for (size_t i = 0; i < ranked_grams.size(); ++i) {
        const auto& g = ranked_grams[i];
        util::JsonObject row;
        row["gram"] = util::json_string(g.gram);
        row["count"] = util::json_number((double)g.count);
        row["width"] = util::json_number((double)g.width);
        row["tokens"] = util::json_array(json_string_array(g.tokens));

        util::JsonArray defs;
        const auto resolved_gram = resolve_manifest_key(g.gram);
        if (!resolved_gram.empty()) {
            util::JsonObject ref;
            ref["key"] = util::json_string(resolved_gram);
            if (resolved_gram != g.gram)
                ref["source_value"] = util::json_string(g.gram);
            ref["kind"] = util::json_string("ngram_key");
            auto expanded = manifest_reg.resolve(resolved_gram);
            util::JsonArray expanded_arr;
            std::unordered_set<std::string> exp_dedup;
            for (const auto& v : expanded) {
                if (v.empty()) continue;
                if (!exp_dedup.insert(v).second) continue;
                expanded_arr.push_back(util::json_string(v));
                if (expanded_arr.size() >= 8) break;
            }
            ref["expansion"] = util::json_array(std::move(expanded_arr));
            defs.push_back(util::json_object(std::move(ref)));
        }
        for (const auto& tok : g.tokens) {
            const auto resolved_tok = resolve_manifest_key(tok);
            if (resolved_tok.empty()) continue;
            util::JsonObject ref;
            ref["key"] = util::json_string(resolved_tok);
            if (resolved_tok != tok)
                ref["source_value"] = util::json_string(tok);
            ref["kind"] = util::json_string("token_key");
            auto expanded = manifest_reg.resolve(resolved_tok);
            util::JsonArray expanded_arr;
            std::unordered_set<std::string> exp_dedup;
            for (const auto& v : expanded) {
                if (v.empty()) continue;
                if (!exp_dedup.insert(v).second) continue;
                expanded_arr.push_back(util::json_string(v));
                if (expanded_arr.size() >= 6) break;
            }
            ref["expansion"] = util::json_array(std::move(expanded_arr));
            defs.push_back(util::json_object(std::move(ref)));
        }
        row["definitions"] = util::json_array(std::move(defs));

        util::JsonArray related;
        for (size_t j = 0; j < ranked_grams.size(); ++j) {
            if (i == j) continue;
            bool shares_token = false;
            for (const auto& a : g.tokens) {
                if (std::find(ranked_grams[j].tokens.begin(), ranked_grams[j].tokens.end(), a) != ranked_grams[j].tokens.end()) {
                    shares_token = true;
                    break;
                }
            }
            if (!shares_token) continue;
            util::JsonObject rel;
            rel["gram"] = util::json_string(ranked_grams[j].gram);
            rel["count"] = util::json_number((double)ranked_grams[j].count);
            rel["width"] = util::json_number((double)ranked_grams[j].width);
            related.push_back(util::json_object(std::move(rel)));
            if (related.size() >= 4) break;
        }
        row["related_ngrams"] = util::json_array(std::move(related));
        ngram_ref_arr.push_back(util::json_object(std::move(row)));
    }
    obj["ngram_references"] = util::json_array(std::move(ngram_ref_arr));

    util::JsonObject structured;
    util::JsonArray hierarchy;
    hierarchy.push_back(util::json_string("atom"));
    hierarchy.push_back(util::json_string("gram"));
    hierarchy.push_back(util::json_string("relation"));
    hierarchy.push_back(util::json_string("clause"));
    hierarchy.push_back(util::json_string("sentence"));
    hierarchy.push_back(util::json_string("paragraph"));
    structured["hierarchy"] = util::json_array(std::move(hierarchy));
    structured["control_authority"] = util::json_string("XCFE/K'UHUL");
    structured["node_semantics_authority"] = util::json_string("nodes");
    structured["weight_semantics"] = util::json_string("pattern_strength_only");
    structured["weight_scale"] = util::json_string("[0,1]");
    structured["unigram_layer"] = util::json_string("semantic_addresses");
    structured["ngram_layer"] = util::json_string("context_association_evidence");
    structured["gram_classes"] = util::json_array(json_string_array(std::vector<std::string>{
        "unigram", "bigram", "trigram", "quadgram", "coarse_gram", "function_gram", "tool_gram"
    }));
    structured["coarse_gram_layer"] = util::json_string("structural_shape");
    structured["function_gram_layer"] = util::json_string("function_word_structure");
    structured["tool_gram_layer"] = util::json_string("tool_dispatch_intent");
    structured["tool_gram_count"] = util::json_number(static_cast<double>(taxonomy.tool_grams.size()));
    structured["clause"] = util::json_string(projection.clause);
    structured["sentence"] = util::json_string(projection.sentence);
    structured["paragraph"] = util::json_string(projection.sentence);
    obj["structured_composition"] = util::json_object(std::move(structured));

    if (auto folds = build_language_fold_payload(
            payload,
            model.tokens,
            composed_text,
            response_phase,
            governing_track,
            projection.relations.size());
        folds && !folds->is_null()) {
        obj["language_folds"] = folds;
    }

    if (auto card = build_track_card(payload); card && !card->is_null())
        obj["card"] = card;

    return util::json_object(std::move(obj));
}

struct AppConfig {
    uint16_t    port_mcp  = PORT_MCP;
    uint16_t    port_rpc  = PORT_JSON_RPC;
    uint16_t    port_chat = 17889;  // OpenAI-compat sidecar (tiny-llama); 0 = disabled
    std::string data_dir  = "data";
    std::string weights_dir;
    std::string gguf_model;
    std::string advisor_model;
    std::string dds_kson;
    std::string dds_manifest;
    std::string dds_shards;
    std::string gpu_hint  = "dedicated";
    std::string proj_book;  // resolved to data_dir/projection_abi_v1.json if empty
    std::string gyro_book;  // resolved to data_dir/gyro_abi_v1.json if empty
    bool        build_keycache = false;
    bool        build_lexicon_grammar = false;
    bool        verbose   = false;
};

enum class ArgParseStatus {
    Ok,
    Help,
    Error
};

static void print_help() {
    std::cout << "KHANARY.exe\n"
              << "  --port-mcp <n>        MCP server port (default: 8764)\n"
              << "  --port-rpc <n>        JSON-RPC server port (default: 8787)\n"
              << "  --chat-port <n>       OpenAI-compat sidecar port for /v1/chat/completions (default: 17889, 0=disabled)\n"
              << "  --data-dir <path>     Data directory (default: ./data)\n"
              << "  --weights-dir <path>  SCXQDDS weight shards directory\n"
              << "  --gguf-model <path>   GGUF file preloaded into system RAM\n"
              << "  --dds-kson <path>     KAST/1 .kson driver file\n"
              << "  --dds-manifest <path> DDS shard manifest JSON\n"
              << "  --dds-shards <path>   Directory containing fold_N.dds files\n"
              << "  --gpu <hint>          GPU adapter hint (default: dedicated)\n"
              << "  --proj-book <path>    Projection ABI book JSON (default: data/projection_abi_v1.json)\n"
              << "  --gyro-book <path>    Gyroscope ABI book JSON (default: data/gyro_abi_v1.json)\n"
              << "  --build-keycache      Build key-store binary cache and exit\n"
              << "  --build-lexicon-grammar  Compile dictionary words/definitions into key-grammar files and exit\n"
              << "  --verbose             Verbose logging\n"
              << "  --help                Print help and exit\n";
}

static bool parse_u16_flag_value(
    const std::string& flag,
    const std::string& raw,
    uint16_t* out,
    std::string* err)
{
    if (!out) return false;
    try {
        const unsigned long v = std::stoul(raw);
        if (v == 0 || v > static_cast<unsigned long>(std::numeric_limits<uint16_t>::max())) {
            if (err) *err = flag + " expects a value in [1,65535], got: " + raw;
            return false;
        }
        *out = static_cast<uint16_t>(v);
        return true;
    } catch (const std::exception&) {
        if (err) *err = flag + " expects an integer value, got: " + raw;
        return false;
    }
}

static ArgParseStatus parse_args(
    int argc,
    char* argv[],
    AppConfig& cfg,
    std::string& err)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const std::string& flag_name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                err = "missing value for " + flag_name;
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (arg == "--help") return ArgParseStatus::Help;
        if (arg == "--build-keycache") {
            cfg.build_keycache = true;
            continue;
        }
        if (arg == "--build-lexicon-grammar") {
            cfg.build_lexicon_grammar = true;
            continue;
        }
        if (arg == "--verbose") {
            cfg.verbose = true;
            continue;
        }

        if (arg == "--port-mcp") {
            auto raw = next(arg);
            if (!raw || !parse_u16_flag_value(arg, *raw, &cfg.port_mcp, &err))
                return ArgParseStatus::Error;
            continue;
        }
        if (arg == "--port-rpc") {
            auto raw = next(arg);
            if (!raw || !parse_u16_flag_value(arg, *raw, &cfg.port_rpc, &err))
                return ArgParseStatus::Error;
            continue;
        }
        if (arg == "--chat-port") {
            auto raw = next(arg);
            if (!raw) { err = "--chat-port requires a port number"; return ArgParseStatus::Error; }
            try {
                const unsigned long v = std::stoul(*raw);
                cfg.port_chat = (v > 65535) ? 0 : static_cast<uint16_t>(v);
            } catch (...) { err = "--chat-port expects a number"; return ArgParseStatus::Error; }
            continue;
        }
        if (arg == "--data-dir") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--data-dir expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.data_dir = *raw;
            continue;
        }
        if (arg == "--weights-dir") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--weights-dir expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.weights_dir = *raw;
            continue;
        }
        if (arg == "--gguf-model") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--gguf-model expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.gguf_model = *raw;
            continue;
        }
        if (arg == "--advisor-model") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--advisor-model expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.advisor_model = *raw;
            continue;
        }
        if (arg == "--dds-kson") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--dds-kson expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.dds_kson = *raw;
            continue;
        }
        if (arg == "--dds-manifest") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--dds-manifest expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.dds_manifest = *raw;
            continue;
        }
        if (arg == "--dds-shards") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--dds-shards expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.dds_shards = *raw;
            continue;
        }
        if (arg == "--gpu") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--gpu expects a non-empty adapter hint";
                return ArgParseStatus::Error;
            }
            cfg.gpu_hint = *raw;
            continue;
        }
        if (arg == "--proj-book") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--proj-book expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.proj_book = *raw;
            continue;
        }
        if (arg == "--gyro-book") {
            auto raw = next(arg);
            if (!raw || raw->empty()) {
                if (raw) err = "--gyro-book expects a non-empty path";
                return ArgParseStatus::Error;
            }
            cfg.gyro_book = *raw;
            continue;
        }

        err = "unknown flag: " + arg;
        return ArgParseStatus::Error;
    }
    return ArgParseStatus::Ok;
}

int main(int argc, char* argv[]) {
    // Unbuffer stdout so startup diagnostics are visible even when piped.
    setvbuf(stdout, nullptr, _IONBF, 0);
    AppConfig cfg;
    std::string parse_err;
    const auto parse_status = parse_args(argc, argv, cfg, parse_err);
    if (parse_status == ArgParseStatus::Help) {
        print_help();
        return 0;
    }
    if (parse_status == ArgParseStatus::Error) {
        std::cerr << "ERROR: " << parse_err << "\n\n";
        print_help();
        return 2;
    }

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "KHANARY starting up\n"
              << "  data:    " << cfg.data_dir  << "\n"
              << "  MCP:     " << cfg.port_mcp  << "\n"
              << "  RPC:     " << cfg.port_rpc  << "\n";
    std::vector<uint8_t> gguf_ram_blob;
    if (!cfg.gguf_model.empty()) {
        const bool gguf_exists = fs::exists(cfg.gguf_model);
        std::cout << "  GGUF:    " << cfg.gguf_model
                  << (gguf_exists ? " [present]" : " [missing]") << "\n";
        if (!gguf_exists) {
            std::cerr << "ERROR: --gguf-model path does not exist: " << cfg.gguf_model << "\n";
            return 2;
        }
        if (!cfg.dds_kson.empty()) {
            // DDS identity alignment requires the full model resident in RAM.
            std::string gguf_err;
            if (!preload_file_to_ram(cfg.gguf_model, &gguf_ram_blob, &gguf_err)) {
                std::cerr << "ERROR: GGUF RAM preload failed: " << gguf_err << "\n";
                return 2;
            }
            std::cout << "  GGUF-RAM:" << " "
                      << (gguf_ram_blob.size() / (1024 * 1024)) << " MiB resident\n";
        }
    }

    const std::string gguf_model_tag = normalize_model_tag(cfg.gguf_model);
    std::string dds_identity_raw;
    std::string dds_model_tag;
    bool model_identity_checked = false;
    bool model_identity_aligned = false;
    std::string model_identity_note = "unavailable";
    if (!cfg.dds_kson.empty()) {
        std::string id_err;
        if (auto id = infer_dds_model_identity_from_kson(cfg.dds_kson, &id_err); id && !id->empty()) {
            dds_identity_raw = *id;
            dds_model_tag = normalize_model_tag(*id);
            model_identity_note = "kson_identity";
        } else {
            dds_model_tag = normalize_model_tag(cfg.dds_kson);
            model_identity_note = id_err.empty() ? "kson_fallback" : id_err;
        }
    }
    if (!gguf_model_tag.empty() && !dds_model_tag.empty()) {
        model_identity_checked = true;
        model_identity_aligned =
            gguf_model_tag == dds_model_tag ||
            (gguf_model_tag.size() > 5 && dds_model_tag.find(gguf_model_tag) != std::string::npos) ||
            (dds_model_tag.size() > 5 && gguf_model_tag.find(dds_model_tag) != std::string::npos);
        model_identity_note = model_identity_aligned ? "matched" : "mismatch";
    }

    // ── GPU ───────────────────────────────────────────────────────────────
    gpu::D3D11Device device;
    if (!device.init(cfg.gpu_hint)) {
        std::cerr << "WARN: D3D11 device init failed — GPU features disabled\n";
    } else {
        const auto& ai = device.adapter_info();
        std::cout << "  GPU:     " << ai.description
                  << " (" << (ai.dedicated_video_memory / (1024*1024)) << " MB VRAM)\n";
    }

    gpu::HotSwapManager hot_swap;
    gpu::DMLOps dml_ops(device);
    dml_ops.init();
    gpu::UnifiedTensorRegistry unified_tensors;
    gpu::SvgComputeRuntime svg_compute(device);
    const bool svg_compute_ready =
        svg_compute.init(cfg.data_dir + "/shaders/svg_compute.hlsl");
    if (svg_compute_ready) {
        std::cout << "  SVG-CS:  svg_compute.hlsl runtime projection ready\n";
    } else {
        std::cout << "  SVG-CS:  disabled (" << svg_compute.last_error() << ")\n";
    }

    // ── Manifest ──────────────────────────────────────────────────────────
    manifest::ManifestRegistry manifest_reg;
    manifest::load_all(manifest_reg, cfg.data_dir);
    std::cout << "  Manifest: " << manifest_reg.size() << " keys loaded\n";

    const std::string journal_dir = cfg.data_dir + "/journal";
    fs::create_directories(journal_dir);
    const std::string gram_journal_path = journal_dir + "/gram-events.jsonl";
    const std::string tool_journal_path = journal_dir + "/tool-events.jsonl";

    // ── K'uhul Runtime ────────────────────────────────────────────────────
    kuhul::PhaseEngine   phase_engine;
    kuhul::FoldEngine    fold_engine(phase_engine);
    kuhul::XCFEExecutor  xcfe;
    kuhul::GramStore     gram_store(gram_journal_path);
    kuhul::KuhulVM       vm(phase_engine, fold_engine, xcfe, gram_store);
    kuhul::PhaseDriverLock phase_driver_lock;

    // Pop phase: bulk-load per-expert CODEGEN corpora.
    // Generate K'UHUL+KHL+KLSL+KXML: py -3 tools\scrape.py --domains kuhul --split
    // Generate JSON (project-scoped): py -3 tools\scrape.py --domains json --paths "E:\models\_khanary_inspect\KHANARY.CPP" --split
    // Generate all experts:           py -3 tools\scrape.py --split
    {
        struct ExpertCorpus { const char* file; const char* prefix; };
        static constexpr ExpertCorpus kExperts[] = {
            { "codegen-kuhul.jsonl",   "kuhul"   },  // CODEGEN-KUHUL-µ: K'UHUL phase/fold/XCFE ops
            { "codegen-klsl.jsonl",    "klsl"    },  // CODEGEN-KLSL-µ:  KLSL compiler/shader lang
            { "codegen-khl.jsonl",     "khl"     },  // CODEGEN-KHL-µ:   KHL host-lang constructs
            { "codegen-json.jsonl",    "json"    },  // CODEGEN-JSON-µ:  manifests, contracts, schemas
            { "codegen-cpp.jsonl",     "cpp"     },  // CODEGEN-CPP-µ:   C++ / HLSL GPU code
            { "codegen-h.jsonl",       "cpp"     },  // C++ headers share the cpp expert
            { "codegen-hlsl.jsonl",    "cpp"     },  // HLSL shaders share the cpp expert
            { "codegen-py.jsonl",      "py"      },  // CODEGEN-PY-µ:    Python tools/scripts
            { "codegen-js.jsonl",      "js"      },  // CODEGEN-JS-µ:    JavaScript
            { "codegen-ts.jsonl",      "js"      },  // TypeScript shares the js expert
            { "codegen-cs.jsonl",      "cs"      },  // CODEGEN-CSHARP-µ: C# / .NET
            { "codegen-css.jsonl",     "css"     },  // CODEGEN-CSS-µ:   CSS stylesheets
            { "codegen-html.jsonl",    "html"    },  // CODEGEN-HTML-µ:  HTML markup
            { "codegen-htm.jsonl",     "html"    },  // .htm shares html expert
        };
        size_t total_corpus = 0;
        for (const auto& e : kExperts) {
            const std::string p = cfg.data_dir + "/gram/" + e.file;
            const size_t n = gram_store.load_corpus(p, e.prefix);
            total_corpus += n;
        }
        if (total_corpus > 0)
            std::cout << "  Corpus:  " << total_corpus << " CODEGEN expert bigrams loaded\n";
    }

    {
        std::string runtime_schema_err;
        if (!validate_runtime_schema_stack(cfg.data_dir, &runtime_schema_err)) {
            std::cerr << "ERROR: runtime schema stack gate failed: " << runtime_schema_err << "\n";
            return 1;
        }
        const std::string fold_runtime_path = cfg.data_dir + "/schema/fold.runtime.json";
        const std::string runtime_meta_path = cfg.data_dir + "/schema/kuhul.fold-runtime.json";
        std::string lock_err;
        if (!phase_driver_lock.load(fold_runtime_path, runtime_meta_path, &lock_err)) {
            std::cerr << "ERROR: phase driver lock initialization failed: " << lock_err << "\n";
            return 1;
        }
        vm.set_phase_driver_lock(&phase_driver_lock);
        if (phase_driver_lock.enabled())
            std::cout << "  PhaseDriver: " << phase_driver_lock.summary() << "\n";
    }

    // ── Micronaut Registry ────────────────────────────────────────────────
    micronaut::MicronautRegistry mu_reg;
    micronaut::TrackLoader       track_loader(mu_reg);
    int n_tracks = track_loader.load_dir(cfg.data_dir + "/tracks");
    std::cout << "  Tracks:  " << n_tracks << " semantic tracks loaded\n";

    // The coder micronaut joins the control plane as a µ-call parent
    // (relations-only entry — governs empty, dispatch untouched) and the
    // µ-call graph becomes governed (admission through declared edges).
    micronaut::attach_coder_control_plane(mu_reg);
    std::cout << "  µ-plane: micronaut_coder attached (governed µ-call graph)\n";

    micronaut::AIMLEngine aiml;
    set_mx2lm_greeting_bank_path(cfg.data_dir + "/grammar/greeting-bank.json");
    int n_aiml = aiml.load_dir(cfg.data_dir + "/alice");
    const std::string aiml_state_path = cfg.data_dir + "/aiml_state.json";
    const bool aiml_state_loaded = aiml.load_state(aiml_state_path);
    if (n_aiml > 0)
        std::cout << "  AIML:    " << n_aiml << " files loaded ("
                  << aiml.category_count() << " categories)\n";
    if (aiml_state_loaded)
        std::cout << "  AIML:    state restored from " << aiml_state_path << "\n";

    // Phase-local pattern banks — KuhulVM activates the matching bank on each
    // phase boundary so match() searches only the current phase's categories.
    const std::string aiml_bank_dir = cfg.data_dir + "/alice/word-problem";
    aiml.set_phase_bank_dir(aiml_bank_dir);
    // Augment Yax bank with general question-seed patterns (12 discovery families).
    // add_phase_bank_file accumulates into the existing bank slot — does not replace.
    const std::string qseed_yax = cfg.data_dir + "/alice/question-seeds/yax.aiml";
    if (fs::exists(qseed_yax))
        aiml.add_phase_bank_file("yax", qseed_yax);
    vm.set_aiml_engine(&aiml);
    std::cout << "  AIML:    phase banks wired ("
              << aiml.phase_bank_count("pop")  << " pop / "
              << aiml.phase_bank_count("wo")   << " wo / "
              << aiml.phase_bank_count("yax")  << " yax / "
              << aiml.phase_bank_count("sek")  << " sek / "
              << aiml.phase_bank_count("chen") << " chen / "
              << aiml.phase_bank_count("xul")  << " xul)\n";

    // Build graph-native AIML surface from micronaut registry relations.
    aiml.clear_graph();
    for (const auto& id : mu_reg.ids()) {
        const auto* entry = mu_reg.get(id);
        if (!entry) continue;

        aiml.add_graph_node(entry->id);
        aiml.add_graph_node_attr(entry->id, "phase", kuhul::phase_name(entry->primary_phase));
        aiml.add_graph_node_attr(entry->id, "category", entry->category);
        aiml.add_graph_node_attr(entry->id, "description", entry->description);
        aiml.add_graph_node_attr(entry->id, "static_weight", std::to_string(entry->static_weight));
        aiml.add_graph_node_attr(entry->id, "health", std::to_string(entry->health));
        aiml.add_graph_node_attr(entry->id, "confidence", std::to_string(entry->confidence));

        for (const auto& rel : entry->relations)
            aiml.add_graph_edge(entry->id, rel.target_id, rel.verb, rel.directed);
        for (const auto& legacy : entry->sub_micronauts)
            aiml.add_graph_edge(entry->id, legacy, "@delegates_to", true);
    }
    if (mu_reg.has("Coordinator")) aiml.set_graph_root("Coordinator");

    bool aiml_graphml_loaded = false;
    const std::string aiml_graphml_path = cfg.data_dir + "/alice/aiml.graphml";
    std::error_code graphml_ec;
    if (fs::exists(aiml_graphml_path, graphml_ec) && fs::is_regular_file(aiml_graphml_path, graphml_ec)) {
        aiml_graphml_loaded = aiml.load_graphml(aiml_graphml_path);
    }
    std::cout << "  AIML-G:  " << aiml.graph_node_count() << " nodes, "
              << aiml.graph_edge_count() << " edges"
              << (aiml_graphml_loaded ? " (+GraphML overlay)\n" : "\n");

    // XulGate: all lifecycle commits (keycache build, shutdown persist) are
    // authorized at Xul phase. The decision fields are all true because the
    // process was invoked with explicit intent to build/run — the lifecycle
    // context is the proposal. Runtime topology mutations go through XCFE; this
    // gate governs persistence commits only.
    kuhul::XulGate persist_gate;
    persist_gate.set_authorizer([](kuhul::Phase phase, const std::string&) {
        return phase == kuhul::Phase::Xul;
    });
    auto lifecycle_commit = [&](const std::string& scope,
                                const std::string& proposal) -> std::optional<kuhul::CommitAuthority> {
        kuhul::CommitDecision d;
        d.proposal_verified  = true;
        d.mutation_authorized = true;
        d.scope_permitted    = true;
        d.phase              = kuhul::Phase::Xul;
        return persist_gate.authorize_commit(d, scope, proposal);
    };

    micronaut::Factory factory(mu_reg, CONFIDENCE_DELETE_FLOOR);
    const std::string factory_state_json_path = cfg.data_dir + "/factory_keys.json";
    const std::string factory_state_bin_path = cfg.data_dir + "/factory_keys.bin";
    const std::string factory_cache_meta_path = cfg.data_dir + "/factory_keys.cachemeta.json";

    const auto cache_fingerprint_current = compute_keystore_cache_fingerprint(cfg.data_dir);
    KeyStoreCacheFingerprint cache_fingerprint_saved{};
    const bool cache_meta_loaded = load_keystore_cache_meta(
        factory_cache_meta_path, &cache_fingerprint_saved);
    const bool cache_fingerprint_matches =
        cache_meta_loaded &&
        key_store_cache_fingerprint_equal(cache_fingerprint_current, cache_fingerprint_saved);

    bool factory_state_loaded = false;
    bool factory_state_binary_loaded = false;
    bool factory_state_json_loaded = false;
    if (cache_fingerprint_matches) {
        factory_state_binary_loaded = factory.load_state_binary(factory_state_bin_path);
        factory_state_loaded = factory_state_binary_loaded;
    }

    if (!factory_state_loaded) {
        factory_state_json_loaded = factory.load_state(factory_state_json_path);
        factory_state_loaded = factory_state_json_loaded;
    }

    if (factory_state_binary_loaded) {
        std::cout << "  KeyStore: binary cache restored from " << factory_state_bin_path << "\n";
    } else if (factory_state_loaded) {
        std::cout << "  KeyStore: restored from " << factory_state_json_path << "\n";
    }

    const bool key_store_cache_reuse =
        factory_state_loaded &&
        (cache_fingerprint_matches || factory_state_binary_loaded || factory_state_json_loaded);
    const bool key_store_cache_needs_persist =
        factory_state_loaded &&
        (!factory_state_binary_loaded || !cache_meta_loaded || !cache_fingerprint_matches);

    if (key_store_cache_reuse) {
        std::cout << "  KeyCache: cache accepted; skipping text dataset ingest\n";
    }
    factory.begin_bulk_update();
    aiml.register_tool("manifest_resolve", [&manifest_reg](const std::string& key) -> std::string {
        auto resolved = manifest_reg.resolve(key);
        if (resolved.empty()) return {};
        std::string out;
        for (size_t i = 0; i < resolved.size(); ++i) {
            if (i) out += " | ";
            out += resolved[i];
        }
        return out;
    });
    aiml.register_tool("key_lookup", [&factory](const std::string& query) -> std::string {
        auto hit = factory.lookup_key(query);
        return hit ? *hit : std::string{};
    });
    factory.record_key("Mx2LM:", "<mx2lm>{ngrams}</mx2lm>");
    factory.record_key("@mx2lm", "<mx2lm>{ngrams}</mx2lm>");
    factory.record_key("APP:", "<app>{intent}</app>");
    factory.record_key("@app", "<app>{intent}</app>");
    factory.record_key("memory:", "<memory> @memory");
    factory.record_key("@memory", "<memory> @memory");
    factory.record_key("LEXICON:", "<lexicon>{unigram}</lexicon>");
    factory.record_key("@lexicon", "<lexicon>{unigram}</lexicon>");
    factory.record_key("USERMANIFEST:", "<user_manifest>{scope}</user_manifest>");
    factory.record_key("@user_manifest", "<user_manifest>{scope}</user_manifest>");

    if (!key_store_cache_reuse) {
    // ── Dictionary key-store bulk-load ────────────────────────────────────
    {
        const std::string dict_path = cfg.data_dir + "/gram/english-dictionary.jsonl";
        std::ifstream dict_f(dict_path);
        if (dict_f.is_open()) {
            std::string line;
            size_t loaded = 0;
            while (std::getline(dict_f, line)) {
                if (line.empty()) continue;
                auto obj = util::parse(line);
                if (!obj) continue;
                auto word = obj->get_string("word");
                auto defn = obj->get_string("definition");
                if (word && defn && !word->empty() && !defn->empty()) {
                    std::string hash_hint;
                    if (auto hash = obj->get_string("hash")) hash_hint = *hash;
                    else if (auto hash_hex = obj->get_string("hash_hex")) hash_hint = *hash_hex;
                    else if (auto hash64 = obj->get_string("hash64")) hash_hint = *hash64;
                    factory.record_key(*word, *defn, hash_hint);
                }
                ++loaded;
            }
            std::cout << "  Dict:    " << loaded << " words loaded into key store\n";
        }
    }
    {
        const auto& dict_lexicon = dictionary_lexicon_cache(cfg.data_dir);
        size_t seeded = 0;
        for (const auto& [word, synonyms] : dict_lexicon.similarity_by_word) {
            const auto packed = join_with_delim(synonyms, " | ", 6);
            if (packed.empty()) continue;
            factory.record_key("word_similarity:" + word, packed);
            factory.record_key("synonym:" + word, packed);
            ++seeded;
        }
        if (seeded > 0)
            std::cout << "  DictSim: " << seeded << " similarity keys seeded\n";
    }

    // ── AG News bulk-load (classification) ───────────────────────────────
    {
        static constexpr const char* AG_LABELS[] =
            { "World", "Sports", "Business", "Sci/Tech" };
        const std::string ag_path = cfg.data_dir + "/ag_news_train.jsonl";
        std::ifstream ag_f(ag_path);
        if (ag_f.is_open()) {
            std::string line;
            size_t loaded = 0;
            while (std::getline(ag_f, line)) {
                if (line.empty()) continue;
                util::JsonPtr obj;
                try { obj = util::parse(line); } catch (const std::exception&) { continue; }
                if (!obj) continue;
                auto text = obj->get_string("text");
                auto lnum = obj->get_number("label");
                if (!text || text->empty()) continue;
                int label = lnum ? static_cast<int>(*lnum) : -1;
                const char* cat = (label >= 0 && label < 4) ? AG_LABELS[label] : "News";
                // Key: headline — text up to first \n or 80 chars
                std::string key = *text;
                auto nl = key.find('\n');
                if (nl != std::string::npos) key = key.substr(0, nl);
                if (key.size() > 80) key = key.substr(0, 80);
                // Value: [Category] full text
                std::string val;
                val.reserve(text->size() + 16);
                val += '['; val += cat; val += "] "; val += *text;
                factory.record_key(key, val);
                ++loaded;
            }
            std::cout << "  AGNews:  " << loaded << " articles loaded into key store\n";
        }
    }

    // ── SQuAD bulk-load (QA) ─────────────────────────────────────────────
    {
        const std::string sq_path = cfg.data_dir + "/squad_train.jsonl";
        std::ifstream sq_f(sq_path);
        if (sq_f.is_open()) {
            std::string line;
            size_t loaded = 0;
            while (std::getline(sq_f, line)) {
                if (line.empty()) continue;
                util::JsonPtr obj;
                try { obj = util::parse(line); } catch (const std::exception&) { continue; }
                if (!obj) continue;
                auto question = obj->get_string("question");
                auto context  = obj->get_string("context");
                if (!question || !context || question->size() < 8 || context->empty()) continue;
                factory.record_key(*question, *context);
                ++loaded;
            }
            std::cout << "  SQuAD:   " << loaded << " QA pairs loaded into key store\n";
        }
    }

    // ── WikiText-2 bulk-load (language modeling) ─────────────────────────
    {
        const std::string wt_path = cfg.data_dir + "/wikitext2_train.jsonl";
        std::ifstream wt_f(wt_path);
        if (wt_f.is_open()) {
            std::string line;
            size_t loaded = 0;
            while (std::getline(wt_f, line)) {
                if (line.empty()) continue;
                auto obj = util::parse(line);
                if (!obj) continue;
                auto text = obj->get_string("text");
                if (!text || text->size() < 40) continue;
                // Key: first 60 chars (enough to be unique per passage)
                std::string key = text->substr(0, 60);
                if (key.size() < 8) continue;
                factory.record_key(key, *text);
                ++loaded;
            }
            std::cout << "  Wiki2:   " << loaded << " passages loaded into key store\n";
        }
    }

    // ── Theory JSONL bulk-load (Wikipedia knowledge paragraphs) ─────────
    // Loads any data/*.jsonl with schema {title, title_key, paragraph_id, text}.
    // Key routes: title → intro paragraph, title_key → intro paragraph,
    //             first-60-chars-of-paragraph → full paragraph text.
    {
        std::error_code wiki_ec;
        std::unordered_map<std::string, std::string> title_intro; // title_key → first para text
        size_t wiki_loaded = 0;

        for (const auto& entry : fs::directory_iterator(cfg.data_dir, wiki_ec)) {
            if (entry.path().extension() != ".jsonl") continue;
            const auto stem = entry.path().stem().string();
            // Skip the already-loaded datasets
            if (stem == "ag_news_train" || stem == "squad_train" || stem == "wikitext2_train") continue;
            if (stem == "english-dictionary" || stem == "english-triples") continue;

            std::ifstream jf(entry.path());
            if (!jf.is_open()) continue;

            std::string line;
            while (std::getline(jf, line)) {
                if (line.empty()) continue;
                auto obj = util::parse(line);
                if (!obj) continue;
                auto task  = obj->get_string("task");
                if (!task || *task != "knowledge") continue;

                auto text     = obj->get_string("text");
                auto title    = obj->get_string("title");
                auto titlekey = obj->get_string("title_key");
                auto para_num = obj->get_number("paragraph_id");
                if (!text || text->size() < 20) continue;

                // Per-paragraph key: first 60 chars → full text
                std::string pkey = text->substr(0, 60);
                if (pkey.size() >= 8) {
                    factory.record_key(pkey, *text);
                    ++wiki_loaded;
                }

                // Title-based keys: store once (paragraph_id == 1 = intro)
                int pid = para_num ? static_cast<int>(*para_num) : 0;
                if (pid == 1) {
                    if (title && !title->empty()) {
                        factory.record_key(*title, *text);
                        title_intro[*title] = *text;
                    }
                    if (titlekey && !titlekey->empty()) {
                        // Also store with underscores replaced by spaces for natural lookup
                        std::string tk_spaces = *titlekey;
                        std::replace(tk_spaces.begin(), tk_spaces.end(), '_', ' ');
                        factory.record_key(*titlekey, *text);
                        if (tk_spaces != *titlekey)
                            factory.record_key(tk_spaces, *text);
                    }
                }
            }
        }
        if (wiki_loaded > 0)
            std::cout << "  Theory:  " << wiki_loaded << " knowledge paragraphs loaded into key store\n";
    }

    // ── Docs bulk-load (markdown + HTML sections) ────────────────────────
    {
        const fs::path docs_dir = fs::path(cfg.data_dir).parent_path() / "docs";
        size_t doc_sections = 0;
        std::error_code ec;

        // ── HTML helpers ──────────────────────────────────────────────────
        auto remove_blocks = [](std::string& text, const std::string& tag) {
            const std::string open  = "<" + tag;
            const std::string close = "</" + tag + ">";
            size_t pos = 0;
            while ((pos = text.find(open, pos)) != std::string::npos) {
                size_t end = text.find(close, pos);
                if (end == std::string::npos) break;
                text.erase(pos, end + close.size() - pos);
            }
        };
        auto strip_tags = [](const std::string& s) -> std::string {
            std::string out; out.reserve(s.size());
            bool in_tag = false;
            for (char c : s) {
                if      (c == '<') { in_tag = true; out += ' '; }
                else if (c == '>') { in_tag = false; }
                else if (!in_tag)  { out += c; }
            }
            std::string clean; clean.reserve(out.size());
            bool ws = false;
            for (char c : out) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    if (!ws) clean += ' '; ws = true;
                } else { clean += c; ws = false; }
            }
            return clean;
        };
        auto decode_ents = [](std::string s) -> std::string {
            auto rep = [&](const std::string& from, const std::string& to) {
                size_t p = 0;
                while ((p = s.find(from, p)) != std::string::npos) {
                    s.replace(p, from.size(), to); p += to.size();
                }
            };
            rep("&amp;","&"); rep("&lt;","<"); rep("&gt;",">");
            rep("&quot;","\""); rep("&apos;","'"); rep("&#39;","'");
            rep("&nbsp;"," "); rep("&mdash;","—"); rep("&ndash;","–");
            return s;
        };
        auto trim_str = [](const std::string& s) -> std::string {
            size_t a = s.find_first_not_of(" \t\r\n");
            size_t b = s.find_last_not_of(" \t\r\n");
            return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
        };

        for (const auto& entry : fs::directory_iterator(docs_dir, ec)) {
            const auto ext = entry.path().extension().string();

            // ── Markdown ──────────────────────────────────────────────────
            if (ext == ".md") {
                std::ifstream md_f(entry.path());
                if (!md_f.is_open()) continue;
                std::string current_header, current_body;
                auto trim_hdr = [](std::string h) {
                    while (!h.empty() && (h.back() == ' ' || h.back() == '\r')) h.pop_back();
                    return h;
                };
                auto flush_md = [&]() {
                    if (current_header.size() >= 4 && !current_body.empty()) {
                        size_t s = current_body.find_first_not_of(" \t\r\n");
                        size_t e = current_body.find_last_not_of(" \t\r\n");
                        if (s != std::string::npos)
                            factory.record_key(current_header, current_body.substr(s, e - s + 1));
                        ++doc_sections;
                    }
                };
                std::string line;
                while (std::getline(md_f, line)) {
                    if      (line.rfind("### ", 0) == 0) { flush_md(); current_header = trim_hdr(line.substr(4)); current_body.clear(); }
                    else if (line.rfind("## ",  0) == 0) { flush_md(); current_header = trim_hdr(line.substr(3)); current_body.clear(); }
                    else if (!current_header.empty())     { current_body += line + "\n"; }
                }
                flush_md();
            }

            // ── HTML ──────────────────────────────────────────────────────
            else if (ext == ".html") {
                std::ifstream html_f(entry.path());
                if (!html_f.is_open()) continue;
                std::string html((std::istreambuf_iterator<char>(html_f)),
                                  std::istreambuf_iterator<char>());
                remove_blocks(html, "style");
                remove_blocks(html, "script");
                remove_blocks(html, "nav");

                // Collect all h2/h3 heading positions and their text
                struct HSection { std::string header; size_t body_start = 0; };
                std::vector<HSection> sections;
                size_t pos = 0;
                while (pos < html.size()) {
                    size_t h2 = html.find("<h2", pos);
                    size_t h3 = html.find("<h3", pos);
                    size_t next = std::min(h2, h3);
                    if (next == std::string::npos) break;
                    const std::string ctag = (next == h2) ? "</h2>" : "</h3>";
                    // heading text: between '>' and '</hN>'
                    size_t gt = html.find('>', next);
                    if (gt == std::string::npos) { pos = next + 3; continue; }
                    size_t ce = html.find(ctag, gt);
                    if (ce == std::string::npos) { pos = next + 3; continue; }
                    std::string hdr = trim_str(decode_ents(strip_tags(html.substr(gt + 1, ce - gt - 1))));
                    if (!hdr.empty())
                        sections.push_back({hdr, ce + ctag.size()});
                    pos = next + 3;
                }

                for (size_t i = 0; i < sections.size(); ++i) {
                    size_t body_end = (i + 1 < sections.size())
                        ? sections[i + 1].body_start : html.size();
                    if (body_end <= sections[i].body_start) continue;
                    std::string body = trim_str(decode_ents(strip_tags(
                        html.substr(sections[i].body_start, body_end - sections[i].body_start))));
                    if (sections[i].header.size() >= 4 && body.size() >= 8) {
                        factory.record_key(sections[i].header, body);
                        ++doc_sections;
                    }
                }
            }
        }
        if (doc_sections > 0)
            std::cout << "  Docs:    " << doc_sections << " sections loaded into key store\n";
    }

    // ── Schema graph + model manifest loader ──────────────────────────────
    // Reads internal-schema-graph.json to discover all schema modules, loads
    // each schema's title/description/$defs types into the key store, then
    // walks models/*/MODEL.json to load model capability summaries.
    {
        const fs::path repo_root = fs::path(cfg.data_dir).parent_path();
        size_t schema_loaded = 0;

        // 1. Schema modules via internal-schema-graph.json
        const fs::path sg_path = fs::path(cfg.data_dir) / "schema" / "internal-schema-graph.json";
        {
            std::ifstream sg_f(sg_path);
            if (sg_f.is_open()) {
                std::string sg_text((std::istreambuf_iterator<char>(sg_f)),
                                    std::istreambuf_iterator<char>());
                util::JsonPtr sg;
                try { sg = util::parse(sg_text); } catch (const std::exception&) {}
                const util::JsonValue* mods_v = sg ? sg->get("modules") : nullptr;
                if (mods_v && mods_v->is_array()) {
                    for (const auto& mod_ptr : mods_v->as_array()) {
                        if (!mod_ptr || !mod_ptr->is_object()) continue;
                        auto mname     = mod_ptr->get_string("name");
                        auto schema_id = mod_ptr->get_string("schema_id");
                        auto sf_rel    = mod_ptr->get_string("schema_file");
                        if (!sf_rel) continue;

                        std::ifstream sf(repo_root / *sf_rel);
                        if (!sf.is_open()) continue;
                        std::string sf_text((std::istreambuf_iterator<char>(sf)),
                                            std::istreambuf_iterator<char>());
                        util::JsonPtr schema;
                        try { schema = util::parse(sf_text); } catch (const std::exception&) {}
                        if (!schema) continue;

                        auto title = schema->get_string("title");
                        auto desc  = schema->get_string("description");

                        std::string val;
                        if (title) val = *title;
                        if (desc)  { if (!val.empty()) val += ": "; val += *desc; }
                        if (val.empty() && mname) val = *mname;
                        if (val.empty()) continue;

                        if (mname)     factory.record_key(*mname, val);
                        if (schema_id) factory.record_key(*schema_id, val);
                        if (title && (!mname || *title != *mname)) factory.record_key(*title, val);
                        ++schema_loaded;

                        // Per-type entries from $defs
                        const util::JsonValue* defs_v = schema->get("$defs");
                        if (defs_v && defs_v->is_object()) {
                            for (const auto& [def_name, def_ptr] : defs_v->as_object()) {
                                if (!def_ptr || !def_ptr->is_object()) continue;
                                std::string def_val;
                                auto def_desc = def_ptr->get_string("description");
                                if (def_desc) def_val = *def_desc;

                                const util::JsonValue* enum_v = def_ptr->get("enum");
                                if (enum_v && enum_v->is_array()) {
                                    std::string ev = def_name + " valid values: ";
                                    bool first = true;
                                    for (const auto& ep : enum_v->as_array()) {
                                        if (!first) ev += ", ";
                                        if (ep && ep->is_string()) ev += ep->as_string();
                                        first = false;
                                    }
                                    if (!def_val.empty()) def_val += " — ";
                                    def_val += ev;
                                }

                                const util::JsonValue* props_v = def_ptr->get("properties");
                                if (props_v && props_v->is_object()) {
                                    std::string pv = def_name + " properties: ";
                                    bool first = true;
                                    for (const auto& [pname, pptr] : props_v->as_object()) {
                                        (void)pptr;
                                        if (!first) pv += ", ";
                                        pv += pname;
                                        first = false;
                                    }
                                    if (!def_val.empty()) def_val += " — ";
                                    def_val += pv;
                                }

                                if (!def_val.empty()) {
                                    factory.record_key(def_name, def_val);
                                    ++schema_loaded;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Model manifests: models/*/MODEL.json
        const fs::path models_dir = repo_root / "models";
        std::error_code models_ec;
        for (const auto& entry : fs::directory_iterator(models_dir, models_ec)) {
            if (!entry.is_directory()) continue;
            std::ifstream mf(entry.path() / "MODEL.json");
            if (!mf.is_open()) continue;
            std::string mf_text((std::istreambuf_iterator<char>(mf)),
                                 std::istreambuf_iterator<char>());
            auto mj = util::parse(mf_text);
            if (!mj) continue;

            auto mname   = mj->get_string("name");
            auto version = mj->get_string("version");
            auto profile = mj->get_string("knu_profile");
            if (!mname) continue;

            std::string mval = *mname;
            if (version) { mval += " v"; mval += *version; }
            if (profile) { mval += ": "; mval += *profile; }

            const util::JsonValue* ops_v = mj->get("ops");
            if (ops_v && ops_v->is_object()) {
                mval += " | ops: ";
                bool first = true;
                for (const auto& [op_name, op_ptr] : ops_v->as_object()) {
                    (void)op_ptr;
                    if (!first) mval += ", ";
                    mval += op_name;
                    first = false;
                }
            }
            const util::JsonValue* be_v = mj->get("backends");
            if (be_v && be_v->is_object()) {
                mval += " | backends: ";
                bool first = true;
                for (const auto& [be_name, be_ptr] : be_v->as_object()) {
                    (void)be_ptr;
                    if (!first) mval += ", ";
                    mval += be_name;
                    first = false;
                }
            }

            factory.record_key(*mname, mval);
            factory.record_key(*mname + " model", mval);
            if (version) factory.record_key(*mname + " v" + *version, mval);
            // Space-separated variant so each word hits the word index separately.
            // "khanary geometry" (16 chars) beats "KHANARY Manifest Map" (19 chars)
            // for the word "khanary" since both are phrases and shorter wins.
            {
                std::string mname_sp = *mname;
                std::replace(mname_sp.begin(), mname_sp.end(), '-', ' ');
                if (mname_sp != *mname) {
                    factory.record_key(mname_sp, mval);
                    factory.record_key(mname_sp + " model", mval);
                }
            }
            ++schema_loaded;

            // schema URNs from data.schema_id / data.schema_types_id
            const util::JsonValue* data_v = mj->get("data");
            if (data_v && data_v->is_object()) {
                auto sid  = data_v->get_string("schema_id");
                auto stid = data_v->get_string("schema_types_id");
                if (sid)  factory.record_key(*sid, mval);
                if (stid) factory.record_key(*stid, mval);
            }
        }

        if (schema_loaded > 0)
            std::cout << "  Schemas: " << schema_loaded << " schema entries loaded into key store\n";
    }
    } // !key_store_cache_reuse

    // ── Dataset micronauts ────────────────────────────────────────────────
    // One micronaut per ingested domain, registered on every startup so
    // routing works whether keys came from fresh ingest or binary cache.
    {
        size_t dataset_mu_added = 0;
        auto add_dataset_mu = [&](const char* id, const char* desc,
                                  std::vector<std::string> caps,
                                  std::unordered_set<std::string> govs) {
            if (mu_reg.has(id)) return;
            micronaut::MicronautEntry e;
            e.id            = id;
            e.description   = desc;
            e.primary_phase = kuhul::Phase::Pop;
            e.category      = "storage";
            e.confidence    = 0.8f;
            e.active        = true;
            e.capabilities  = std::move(caps);
            e.governs       = std::move(govs);
            mu_reg.add(std::move(e));
            ++dataset_mu_added;
        };

        add_dataset_mu("DICT-µ",   "dictionary definitions and synonyms",
            {"key_lookup", "word_lookup"},
            {"word", "definition", "lexicon", "synonym", "vocabulary"});
        add_dataset_mu("NEWS-µ",   "AGNews articles by topic and headline",
            {"key_lookup", "news_lookup"},
            {"news", "article", "headline", "topic", "agn"});
        add_dataset_mu("QA-µ",     "SQuAD question-answer pairs",
            {"key_lookup", "qa_lookup"},
            {"question", "answer", "qa", "reading", "comprehension"});
        add_dataset_mu("WIKI-µ",   "Wikitext2 encyclopedic passages",
            {"key_lookup", "passage_lookup"},
            {"passage", "wiki", "encyclopedia", "knowledge"});
        add_dataset_mu("THEORY-µ", "structured knowledge paragraphs",
            {"key_lookup", "knowledge_lookup"},
            {"knowledge", "theory", "title", "paragraph"});
        add_dataset_mu("DOCS-µ",   "documentation sections",
            {"key_lookup", "doc_lookup"},
            {"documentation", "guide", "reference", "docs"});
        add_dataset_mu("SCHEMA-µ", "schema types and model manifests",
            {"key_lookup", "schema_lookup"},
            {"schema", "type", "definition", "model", "manifest"});

        if (dataset_mu_added > 0)
            std::cout << "  DatasetMu: " << dataset_mu_added << " dataset micronauts registered\n";
    }

    factory.end_bulk_update();

    if (!key_store_cache_reuse || key_store_cache_needs_persist) {
        auto ca_cache = lifecycle_commit("keystore", "cache-persist");
        const bool bin_ok = ca_cache &&
            factory.persist_binary(std::move(*ca_cache), factory_state_bin_path);
        const bool meta_ok = save_keystore_cache_meta(factory_cache_meta_path, cache_fingerprint_current);
        if (bin_ok && meta_ok) {
            std::cout << "  KeyCache: compiled binary cache at " << factory_state_bin_path << "\n";
        } else {
            std::cerr << "  WARN: failed to persist binary key cache (" << factory_state_bin_path << ")\n";
        }
    }

    if (cfg.build_lexicon_grammar || cfg.build_keycache) {
        std::string grammar_err;
        auto summary = compile_dictionary_key_grammar_files(
            cfg.data_dir,
            dictionary_lexicon_cache(cfg.data_dir),
            &grammar_err);
        if (!grammar_err.empty()) {
            std::cerr << "  WARN: key-grammar compile failed: " << grammar_err << "\n";
            if (cfg.build_lexicon_grammar) return 1;
        } else {
            std::cout << "  KeyGrammar: " << summary.lexicon_rows << " lexicon rows, "
                      << summary.relation_rows << " definition relations\n";
            std::cout << "  KeyGrammar: " << summary.lexicon_file << "\n";
            std::cout << "  KeyGrammar: " << summary.relations_file << "\n";
            if (!summary.aiml_ebnf_file.empty())
                std::cout << "  KeyGrammar: " << summary.aiml_ebnf_file << "\n";
            if (!summary.scfml_ebnf_file.empty())
                std::cout << "  KeyGrammar: " << summary.scfml_ebnf_file << "\n";
            if (!summary.key_grammar_ebnf_file.empty())
                std::cout << "  KeyGrammar: " << summary.key_grammar_ebnf_file << "\n";
        }
        if (cfg.build_lexicon_grammar && !cfg.build_keycache) {
            std::cout << "  KeyGrammar: build-only mode complete\n";
            return 0;
        }
    }

    if (cfg.build_keycache) {
        auto ca_build = lifecycle_commit("keystore", "keycache-build");
        if (ca_build) factory.persist(std::move(*ca_build), factory_state_json_path, factory_state_bin_path);
        save_keystore_cache_meta(factory_cache_meta_path, compute_keystore_cache_fingerprint(cfg.data_dir));
        std::cout << "  KeyCache: build-only mode complete\n";
        return 0;
    }

    // ── Sandbox Inspector ─────────────────────────────────────────────────
    // Populate from the loaded track entries — each track's declared sandbox
    // is now authoritative for runtime enforcement and JSON inspection.
    micronaut::SandboxInspector sandbox_inspector;
    for (const auto& id : mu_reg.ids()) {
        const auto* e = mu_reg.get(id);
        if (e) sandbox_inspector.register_track(id, e->sandbox);
    }
    std::cout << "  Sandbox: " << mu_reg.ids().size() << " track sandboxes registered\n";

    // Wire sandbox enforcement into the VM — gramRead/gramWrite now gate-checked
    vm.set_sandbox(&sandbox_inspector);

    // ── K-CUBE World State ────────────────────────────────────────────────────
    world::WorldState world_state(mu_reg, sandbox_inspector);
    world_state.rebuild();
    std::cout << "  World:   " << world_state.nodes().size() << " track nodes in K-CUBE\n";

    // ── GML overlay paths (data/*.gml) ───────────────────────────────────────
    std::vector<std::string> gml_paths;
    {
        std::error_code gml_ec;
        for (const auto& de : fs::directory_iterator(cfg.data_dir, gml_ec)) {
            if (de.path().extension() == ".gml")
                gml_paths.push_back(de.path().string());
        }
    }

    // ── Startup projection — write field.xml (K-CUBE semantic field DAG) ────
    {
        world::FieldDag field_dag(world_state, mu_reg);
        field_dag.build();
        for (const auto& gp : gml_paths) field_dag.load_gml(gp);
        auto world_dir = cfg.data_dir + "/world";
        fs::create_directories(world_dir);
        std::ofstream fx(world_dir + "/field.xml");
        if (fx) fx << field_dag.to_xml();

        world::FieldSvgProjection field_projection;
        const bool projected =
            svg_compute_ready &&
            svg_compute.project(field_dag.nodes(), field_dag.edges(), field_projection);

        std::ofstream fs(world_dir + "/field.svg");
        if (fs) {
            if (projected) fs << field_dag.to_svg(&field_projection);
            else           fs << field_dag.to_svg(nullptr);
        }

        std::cout << "  Field:   " << world_dir << "/field.{xml,svg} ("
                  << field_dag.nodes().size() << " nodes, "
                  << field_dag.edges().size() << " edges)"
                  << (projected ? " [svg_compute]" : " [cpu_projection]") << "\n";
        if (!gml_paths.empty())
            std::cout << "  GML:     " << gml_paths.size() << " graph(s) overlaid\n";
    }

    // ── Vector Projection Container — generation 1 (GML topology geometry) ──
    if (!gml_paths.empty()) {
        micronaut::Mx2LmVectorProjectionContainer vpc;
        vpc.generation       = 1;
        vpc.projection_id    = "vector:projection:gml-topology:v1:g1";
        vpc.learner_version  = "static.gml_coordinate_geometry.v1";
        vpc.topology_version = "gml_topology_corpus.v1";

        world::load_gml_topology_vectors(gml_paths, vpc);

        auto vpc_world_dir = cfg.data_dir + "/world";
        std::ofstream vpc_f(vpc_world_dir + "/vector_projection.json");
        if (vpc_f) {
            util::JsonObject root;
            root["@type"]            = util::json_string("vector.projection_container.v1");
            root["key"]              = util::json_string(vpc.key);
            root["generation"]       = util::json_number(static_cast<double>(vpc.generation));
            root["dimensions"]       = util::json_number(static_cast<double>(vpc.dimensions));
            root["projection_id"]    = util::json_string(vpc.projection_id);
            root["learner_version"]  = util::json_string(vpc.learner_version);
            root["topology_version"] = util::json_string(vpc.topology_version);
            root["semantic_authority"] = util::json_bool(false);

            util::JsonArray arr;
            for (const auto& p : vpc.projections) {
                const float fw = p.norm;
                const float rw = (fw > 1e-6f)
                    ? std::sqrt(p.x*p.x + p.y*p.y) / fw : 0.0f;
                const float pw = (fw > 1e-6f)
                    ? 0.5f + 0.5f * (p.z / fw) : 0.5f;
                util::JsonObject row;
                row["key"]              = util::json_string(p.key);
                row["source_track"]     = util::json_string(p.source_track);
                row["basis"]            = util::json_string(p.basis);
                row["x"]               = util::json_number(static_cast<double>(p.x));
                row["y"]               = util::json_number(static_cast<double>(p.y));
                row["z"]               = util::json_number(static_cast<double>(p.z));
                row["fold_weight"]     = util::json_number(static_cast<double>(fw));
                row["relation_weight"] = util::json_number(static_cast<double>(rw));
                row["phase_weight"]    = util::json_number(static_cast<double>(pw));
                arr.push_back(util::json_object(std::move(row)));
            }
            root["projections"] = util::json_array(std::move(arr));

            vpc_f << util::stringify(util::json_object(std::move(root)), true);
        }

        std::cout << "  VPC:     " << vpc.projections.size()
                  << " vectors loaded (gen " << vpc.generation << ") -> "
                  << vpc_world_dir << "/vector_projection.json\n";
    }

    // ── Memory Stores ─────────────────────────────────────────────────────
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence(cfg.data_dir + "/confidence.json");
    memory::SemanticGraph   sem_graph;
    memory::MicronautScorer scorer(perception, confidence, sem_graph);

    // ── Load DDS weight shards ────────────────────────────────────────────
    if (!cfg.weights_dir.empty() && fs::is_directory(cfg.weights_dir)) {
        gpu::DDSLoader dds_loader(device);
        for (const auto& entry : fs::directory_iterator(cfg.weights_dir)) {
            if (entry.path().extension() == ".scxqdds") {
                try {
                    auto shards = dds_loader.load(entry.path().string());
                    for (auto& s : shards) {
                        if (s.crc_ok && s.buffer)
                            hot_swap.make_resident(s.name,
                                std::shared_ptr<gpu::TensorBuffer>(std::move(s.buffer)));
                    }
                    if (cfg.verbose)
                        std::cout << "  Loaded: " << entry.path().filename() << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "  WARN: " << e.what() << "\n";
                }
            }
        }
        std::cout << "  Resident: "
                  << (hot_swap.resident_bytes() / (1024*1024)) << " MiB / "
                  << (hot_swap.ceiling() / (1024*1024)) << " MiB\n";
    }

    // ── DDS Inference Engine ──────────────────────────────────────────────
    std::unique_ptr<inference::InferenceEngine> engine;
    bool dds_ready = false;
    if (!cfg.dds_kson.empty()) {
        engine = std::make_unique<inference::InferenceEngine>(device);
        try {
            engine->load_dds_model(cfg.dds_kson, cfg.dds_manifest, cfg.dds_shards);
            dds_ready = true;
            std::cout << "  DDS:     model loaded — "
                      << cfg.dds_kson.substr(cfg.dds_kson.find_last_of("/\\") + 1) << "\n";
            std::cout << "  DDS-GPU: "
                      << engine->dds_preload_resident_tensors()
                      << "/" << engine->dds_preload_total_tensors()
                      << " tensors resident, "
                      << (engine->dds_resident_bytes() / (1024 * 1024))
                      << " MiB / "
                      << (engine->dds_resident_ceiling_bytes() / (1024 * 1024))
                      << " MiB budget";
            if (engine->dds_preload_failed_tensors() > 0) {
                std::cout << " (" << engine->dds_preload_failed_tensors()
                          << " deferred)";
            }
            std::cout << "\n";
        } catch (const std::exception& ex) {
            std::cerr << "  WARN: DDS model load failed: " << ex.what() << "\n";
        }
    }

    const bool gguf_ram_loaded = !gguf_ram_blob.empty();
    const bool dual_models_present = gguf_ram_loaded && dds_ready;
    std::cout << "  Dual:    gguf_ram=" << (gguf_ram_loaded ? "yes" : "no")
              << ", dds_gpu=" << (dds_ready ? "yes" : "no");
    if (model_identity_checked) {
        std::cout << ", aligned=" << (model_identity_aligned ? "yes" : "no")
                  << " [" << gguf_model_tag << " vs " << dds_model_tag << "]";
    } else if (!model_identity_note.empty()) {
        std::cout << ", identity=" << model_identity_note;
    }
    std::cout << "\n";
    if (model_identity_checked && !model_identity_aligned) {
        std::cerr << "  WARN: GGUF/DDS identity mismatch detected; infer will be blocked until models align\n";
    }

    // ── JROM Journal ─────────────────────────────────────────────────────
    io::JROM jrom(tool_journal_path);

    // ── DirectionStore — rebuilt from JROM; idempotent on each startup ────
    kuhul::DirectionStore direction_store;
    direction_store.rebuild_from_jrom(jrom);

    // ── Projection Engine + Gyroscope ─────────────────────────────────────
    const std::string proj_book_path =
        cfg.proj_book.empty() ? cfg.data_dir + "/projection_abi_v1.json" : cfg.proj_book;
    const std::string gyro_book_path =
        cfg.gyro_book.empty() ? cfg.data_dir + "/gyro_abi_v1.json" : cfg.gyro_book;

    std::unique_ptr<io::ProjectionEngine> proj_engine;
    std::unique_ptr<world::Gyroscope>     gyroscope;
    bool gyro_ready = false;
    try {
        proj_engine = std::make_unique<io::ProjectionEngine>(proj_book_path);
        gyroscope   = std::make_unique<world::Gyroscope>(gyro_book_path);
        gyro_ready  = true;
        std::cout << "  Gyro:    projection+gyroscope ABI books loaded\n";
    } catch (const std::exception& ex) {
        std::cerr << "  WARN: Gyroscope init failed: " << ex.what() << "\n";
    }

    // ── Fold Score Dispatch ──────────────────────────────────────────────
    gpu::FoldScoreDispatch fold_dispatch(device);
    const bool fold_dispatch_ready = device.is_valid() && fold_dispatch.init();
    if (fold_dispatch_ready)
        std::cout << "  Fold:    fold-score dispatch ready\n";

    // ── Persistent MemoryField for fold-score base weights ────────────────
    // Loaded once at startup; grown incrementally per request; saved on each
    // winner update so entry.weight persists across restarts.
    const std::string fold_field_path =
        cfg.data_dir.empty() ? "fold_field.json"
                             : cfg.data_dir + "/fold_field.json";
    micronaut::MemoryField fold_field = micronaut::MemoryField::load(fold_field_path);
    std::unordered_set<std::string> fold_known_names;
    for (const auto& e : fold_field.entries())
        fold_known_names.insert(e.name);
    std::cout << "  FoldField: " << fold_field.size() << " entries loaded\n";

    const std::string fold_mapping_path =
        cfg.data_dir.empty() ? "fold_mapping.json"
                             : cfg.data_dir + "/fold_mapping.json";
    micronaut::AttentionMapping fold_mapping =
        micronaut::AttentionMapping::load(fold_mapping_path);
    std::cout << "  FoldMapping: " << fold_mapping.size() << " rows loaded"
              << " (v" << fold_mapping.candidate_mapping_version << ")\n";

    // ── Servers ───────────────────────────────────────────────────────────
    micronaut::ToolRegistry tool_reg;  // seeded by MCPServer::register_tools()
    io::MCPServer mcp_server(
        cfg.port_mcp, manifest_reg, vm, mu_reg, factory,
        sandbox_inspector, tool_reg, confidence, world_state, jrom,
        svg_compute_ready ? &svg_compute : nullptr);
    for (const auto& gp : gml_paths) mcp_server.add_gml(gp);

    io::JsonRpcServer rpc_server(cfg.port_rpc);

    // GAP-020/021 wiring (T2.1/T2.2): both servers bind loopback by default;
    // KHANARY_RPC_BIND=any opts out; KHANARY_RPC_TOKEN enables server-side
    // bearer auth. A non-loopback bind without a token refuses to start.
    {
        char bind_buf[16]{};
        const DWORD bind_len =
            ::GetEnvironmentVariableA("KHANARY_RPC_BIND", bind_buf, sizeof(bind_buf));
        const bool rpc_loopback =
            !(bind_len > 0 && bind_len < sizeof(bind_buf) &&
              std::strcmp(bind_buf, "any") == 0);
        char tok_buf[256]{};
        const DWORD tok_len =
            ::GetEnvironmentVariableA("KHANARY_RPC_TOKEN", tok_buf, sizeof(tok_buf));
        const bool has_token = tok_len > 0 && tok_len < sizeof(tok_buf);
        rpc_server.set_bind_loopback(rpc_loopback);
        if (has_token) rpc_server.set_auth_token(tok_buf);
        mcp_server.set_bind_loopback(rpc_loopback);
        if (has_token) mcp_server.set_auth_token(tok_buf);
    }

    io::RequestRouter router(mu_reg, scorer, cfg.data_dir);
    // Episode observations (pseudonymous) so H_Ω / A_max are measured, not guessed.
    router.enable_calibration(cfg.data_dir + "/journal/routing-episodes.jsonl");
    router.set_authorizer([&](const io::RouteRequest& req, const std::string& selected) {
        // Policy context is constructed by the host, never copied from RPC JSON.
        const kuhul::XCFEContext context{{"node", selected}};
        return phase_driver_lock.allow_phase(req.phase) &&
               xcfe.authorize_phase_transition(kuhul::Phase::Yax, req.phase, context);
    });

    // DECISION-µ: typed action selection under explicit epistemic uncertainty.
    // The handler may collapse an action choice, but never the underlying UNKNOWN facts.
    router.register_handler("DECISION-µ", [](const std::string&, const io::RouteRequest& req) -> util::JsonPtr {
        return decision::evaluate_request(req.query, req.payload.get());
    });

    // TOOL-µ: Governor advisory → concrete dispatch action
    router.register_handler("TOOL-µ", [&](const std::string&, const io::RouteRequest& req) -> util::JsonPtr {
        const auto& adv = req.query;
        std::string anchor;
        if (req.payload) {
            if (auto a = req.payload->get_string("anchor")) anchor = *a;
        }
        util::JsonObject action;
        if (adv == "anchor") {
            action["action"]   = util::json_string("re_query");
            action["dispatch"] = util::json_string("key_query");
            action["query"]    = util::json_string(anchor);
        } else if (adv == "reduce_omega") {
            action["action"]   = util::json_string("throttle_dds");
            action["dispatch"] = util::json_string("infer");
            action["max_tok"]  = util::json_number(4.0);
        } else if (adv == "reduce_omega_and_anchor") {
            action["action"]   = util::json_string("throttle_and_requery");
            action["dispatch"] = util::json_string("key_query");
            action["query"]    = util::json_string(anchor);
            action["max_tok"]  = util::json_number(4.0);
        } else {
            return util::json_null();
        }
        return util::json_object(std::move(action));
    });

    // ── Grammar micronauts ────────────────────────────────────────────────────
    // Each declaration is parsed from XML, compiled to a MicronautEntry, and
    // installed as an Execute_µ handler. parse() throws on malformed XML so
    // errors are caught and logged rather than crashing startup.
    {
        size_t grammar_mu_ok = 0;
        auto reg_grammar = [&](const char* xml) {
            try {
                router.register_grammar(io::MicronautGrammarCompiler::parse(xml));
                ++grammar_mu_ok;
            } catch (const std::exception& e) {
                std::cerr << "  GrammarMu: parse error: " << e.what() << '\n';
            }
        };

        // REASON-µ: inferential reasoning over a subject (Wo phase, INFER verb)
        reg_grammar(R"(<micronaut id="REASON-µ">
  <pattern>* about *{subject} | * explain *{subject} | why *{subject}</pattern>
  <wildcard name="prefix"/>
  <template>
    <function>INFER</function>
    <input>@subject</input>
  </template>
  <phase>Wo</phase>
  <weights confidence="0.9" relevance="1.0"/>
</micronaut>)");

        // SENSE-µ: lookup / description query (Pop phase, PERCEIVE verb)
        reg_grammar(R"(<micronaut id="SENSE-µ">
  <pattern>what is *{query} | describe *{query} | lookup *{query}</pattern>
  <template>
    <function>PERCEIVE</function>
    <input>@query</input>
  </template>
  <phase>Pop</phase>
  <weights confidence="0.85" relevance="1.0"/>
</micronaut>)");

        // RELATE-µ: graph neighbourhood traversal (Sek phase, TRAVERSE verb)
        reg_grammar(R"(<micronaut id="RELATE-µ">
  <pattern>relate *{node} | neighbors of *{node} | expand *{node}</pattern>
  <template>
    <function>TRAVERSE</function>
    <input>@node</input>
  </template>
  <phase>Sek</phase>
  <weights confidence="0.80" relevance="1.0"/>
</micronaut>)");

        std::cout << "  GrammarMu: " << grammar_mu_ok << " grammar micronauts registered\n";

        // ── Startup Execute_µ smoke-test (direct — bypasses WCR routing) ─────
        // These call MicronautGrammarCompiler::execute() directly so the grammar
        // path is verified regardless of which micronaut wins the routing contest.
        {
            // String domain: "tell me about neural networks"
            //   branch "* about *{subject}" → subject="neural networks"
            auto reason_decl = io::MicronautGrammarCompiler::parse(
                R"(<micronaut id="REASON-µ">
  <pattern>* about *{subject} | * explain *{subject} | why *{subject}</pattern>
  <wildcard name="prefix"/>
  <template>
    <function>INFER</function>
    <input>@subject</input>
  </template>
  <phase>Wo</phase>
  <weights confidence="0.9" relevance="1.0"/>
</micronaut>)");
            io::ExecuteInput ei_str;
            ei_str.text = "tell me about neural networks";
            auto res_str = io::MicronautGrammarCompiler::execute(ei_str, reason_decl);
            std::cout << "  Execute_µ[String]: matched=" << (res_str.matched() ? "yes" : "no")
                      << " domain="
                      << (res_str.domain == io::ExecuteResult::Domain::String ? "string" :
                          res_str.domain == io::ExecuteResult::Domain::None   ? "none" : "other")
                      << " text=\"" << res_str.text << "\"";
            if (res_str.captures) {
                for (const auto& [k, v] : res_str.captures->bindings)
                    std::cout << " [" << k << "=" << v << "]";
            }
            std::cout << '\n';

            // Graph domain: "relate DECISION-µ" → TRAVERSE executor
            auto relate_decl = io::MicronautGrammarCompiler::parse(
                R"(<micronaut id="RELATE-µ">
  <pattern>relate *{node} | neighbors of *{node} | expand *{node}</pattern>
  <template>
    <function>TRAVERSE</function>
    <input>@node</input>
  </template>
  <phase>Sek</phase>
  <weights confidence="0.80" relevance="1.0"/>
</micronaut>)");
            // Build fn_registry with the router's built-in TRAVERSE executor
            // by extracting it from the router via a one-shot registration check.
            // (Simpler: just re-route through router with a forced micronaut.)
            // We register a dummy fn so we can confirm domain::Graph fires:
            io::ExecuteInput ei_gph;
            ei_gph.text = "relate DECISION-µ";
            // Use the full router (which has TRAVERSE in fn_registry_) by
            // routing with a query that exactly matches RELATE-µ's pattern
            io::RouteRequest req_gph;
            req_gph.query  = "relate DECISION-µ";
            req_gph.phase  = kuhul::Phase::Pop;
            req_gph.lane   = "grammar_smoke_test";
            auto rr = router.route(req_gph);
            std::cout << "  RouteTest[Graph]:  selected=" << rr.micronaut_id
                      << " handler=" << rr.handler_id;
            if (!rr.dispatch_allowed)
                std::cout << " blocked=" << rr.dispatch_block_reason;
            else if (rr.response && rr.response->is_object()) {
                if (auto d  = rr.response->get_string("domain"))    std::cout << " domain=" << *d;
                if (auto gr = rr.response->get_string("graph_ref")) std::cout << " ref=" << *gr;
                const auto* ps = rr.response->get("proposed_state");
                if (ps && ps->is_object()) {
                    const auto* nb = ps->get("graph_neighbors");
                    if (nb && nb->is_array())
                        std::cout << " neighbors=" << nb->as_array().size();
                }
            }
            std::cout << '\n';
        }
    }

    // Wire JSON-RPC routes
    rpc_server.register_method("resolve_manifest", [&](const util::JsonPtr& p) {
        auto key = p ? p->get_string("key") : std::nullopt;
        if (!key) return util::json_null();
        auto expanded = manifest_reg.resolve(*key);
        util::JsonArray arr;
        for (const auto& s : expanded) arr.push_back(util::json_string(s));
        return util::json_array(std::move(arr));
    });

    rpc_server.register_method("cycle_identity", [](const util::JsonPtr& p) {
        util::JsonObject result;
        result["@kind"] = util::json_string("khanary.cycle.identity.v1");
        try {
            if (!p) throw std::runtime_error("cycle identity: missing input");
            result["hash"] = util::json_string(util::cycle_identity_hash(*p));
            result["ok"] = util::json_bool(true);
        } catch (const std::exception& e) {
            result["ok"] = util::json_bool(false);
            result["error"] = util::json_string(e.what());
        }
        return util::json_object(std::move(result));
    });

    rpc_server.register_method("fold_query", [&](const util::JsonPtr& p) {
        auto query = p ? p->get_string("query") : std::nullopt;
        if (!query) return util::json_null();
        io::RouteRequest req;
        req.query = *query;
        req.phase = kuhul::Phase::Pop;
        req.payload = p;
        req.lane = "key_query_pop";
        auto result = router.route(req);
        util::JsonObject obj;
        obj["micronaut"]     = util::json_string(result.micronaut_id);
        obj["handler"]       = util::json_string(result.handler_id);
        obj["selected_task_op"] = util::json_string(result.selected_task_op);
        obj["resolver_policy"] = util::json_string(result.resolver_policy);
        obj["required_primitive"] = util::json_string(result.required_primitive);
        obj["policy_filter_applied"] = util::json_bool(result.policy_filter_applied);
        obj["capability_miss"] = util::json_bool(result.capability_miss);
        obj["selected_capable"] = util::json_bool(result.selected_capable);
        obj["policy_satisfied"] = util::json_bool(result.policy_satisfied);
        obj["xcfe_authorized"] = util::json_bool(result.xcfe_authorized);
        obj["dispatch_allowed"] = util::json_bool(result.dispatch_allowed);
        obj["dispatch_block_reason"] = util::json_string(result.dispatch_block_reason);
        obj["score"]         = util::json_number(result.score);
        obj["fold_weight"]   = util::json_number(result.fold_weight);
        obj["phase_driver_weight"] = util::json_number(result.phase_driver_weight);
        obj["pattern_strength"] = util::json_number(result.pattern_strength);
        obj["static_weight"] = util::json_number(result.static_weight);
        obj["confidence_score"] = util::json_number(result.confidence_score);
        obj["health_score"] = util::json_number(result.health_score);
        obj["relevance_score"] = util::json_number(result.relevance_score);
        obj["bow_score"] = util::json_number(result.bow_score);
        obj["distributional_score"] = util::json_number(result.distributional_score);
        obj["lexical_semantic_score"] = util::json_number(result.lexical_semantic_score);
        obj["nsm_prime_score"] = util::json_number(result.nsm_prime_score);
        obj["function_word_score"] = util::json_number(result.function_word_score);
        obj["content_word_ratio"] = util::json_number(result.content_word_ratio);
        obj["function_word_ratio"] = util::json_number(result.function_word_ratio);
        obj["content_function_binding_ratio"] = util::json_number(result.content_function_binding_ratio);
        obj["word_sense_score"] = util::json_number(result.word_sense_score);
        obj["compositional_phrase_score"] = util::json_number(result.compositional_phrase_score);
        obj["coarse_ngram_score"] = util::json_number(result.coarse_ngram_score);
        obj["tool_gram_score"] = util::json_number(result.tool_gram_score);
        obj["miss_dispatch_score"] = util::json_number(result.miss_dispatch_score);
        obj["semantic_score"]= util::json_number(result.semantic_score);
        obj["score_policy"]  = util::json_string(io::ROUTE_SCORE_POLICY);
        obj["required_capabilities"] = util::json_array(json_string_array(result.required_capabilities));
        obj["eligible_resolvers"] = util::json_array(json_string_array(result.eligible_resolvers));
        obj["ineligible_resolvers"] = util::json_array(json_string_array(result.ineligible_resolvers));
        // LBS blend: [{track, weight}, ...] normalized bone weights
        util::JsonArray blend_arr;
        for (const auto& [tid, w] : result.blend) {
            util::JsonObject bone;
            bone["track"]  = util::json_string(tid);
            bone["weight"] = util::json_number(static_cast<double>(w));
            blend_arr.push_back(util::json_object(std::move(bone)));
        }
        obj["blend"] = util::json_array(std::move(blend_arr));
        return util::json_object(std::move(obj));
    });

    // Agent key store: record model-generated key→response pairs
    rpc_server.register_method("record_key", [&](const util::JsonPtr& p) {
        auto key  = p ? p->get_string("key")      : std::nullopt;
        auto resp = p ? p->get_string("response")  : std::nullopt;
        if (!key || !resp) {
            util::JsonObject err;
            err["ok"] = util::json_bool(false);
            err["error"] = util::json_string("requires 'key' and 'response' fields");
            return util::json_object(std::move(err));
        }
        auto hash = p ? p->get_string("hash") : std::nullopt;
        factory.record_key(*key, *resp, hash.value_or(""));
        util::JsonObject obj;
        obj["ok"]  = util::json_bool(true);
        obj["key"] = util::json_string(*key);
        if (hash) obj["hash"] = util::json_string(*hash);
        return util::json_object(std::move(obj));
    });

    // Advisor-verified factory evolve. PS calls this after Invoke-AdvisorEvolveQuery rewrites seed caps.
    // Fail-closed: empty accepted_capabilities + accepted_governs → rejected, no commit.
    rpc_server.register_method("factory_advisor_commit", [&](const util::JsonPtr& p) {
        auto mid = p ? p->get_string("micronaut_id") : std::nullopt;
        auto make_err = [](const std::string& msg) {
            util::JsonObject e;
            e["ok"]    = util::json_bool(false);
            e["error"] = util::json_string(msg);
            return util::json_object(std::move(e));
        };
        if (!mid || mid->empty())
            return make_err("requires non-empty 'micronaut_id'");

        std::vector<std::string> accepted_caps;
        std::vector<std::string> accepted_governs;
        if (p) {
            const auto* caps_node = p->get("accepted_capabilities");
            if (caps_node && caps_node->is_array()) {
                for (const auto& item : caps_node->as_array())
                    if (item && item->is_string() && !item->as_string().empty())
                        accepted_caps.push_back(item->as_string());
            }
            const auto* govs_node = p->get("accepted_governs");
            if (govs_node && govs_node->is_array()) {
                for (const auto& item : govs_node->as_array())
                    if (item && item->is_string() && !item->as_string().empty())
                        accepted_governs.push_back(item->as_string());
            }
        }
        if (accepted_caps.empty() && accepted_governs.empty())
            return make_err("advisor returned no accepted capabilities or governs");

        auto candidate = factory.propose_evolve(*mid, accepted_caps, accepted_governs);
        if (!candidate)
            return make_err("propose_evolve rejected: micronaut not found or id empty");

        // CapabilityValidator: gate proposal_verified on cap/sandbox/governs consistency.
        // Fail-closed: invalid advisor-returned caps → proposal_verified=false → XulGate rejects.
        std::string cap_reject;
        const bool caps_valid = validate_proposed_capabilities(
            candidate->entry, accepted_caps, accepted_governs, cap_reject);

        kuhul::CommitDecision d;
        d.proposal_verified   = caps_valid;
        d.mutation_authorized = true;
        d.scope_permitted     = true;
        d.phase               = kuhul::Phase::Xul;
        auto ca = persist_gate.authorize_commit(d, candidate->scope, candidate->proposal_identity);
        if (!ca)
            return make_err(caps_valid ? "commit not authorized by XulGate"
                                       : "capability validation failed: " + cap_reject);

        auto r = factory.commit(std::move(*candidate), std::move(*ca));
        util::JsonObject obj;
        obj["ok"]           = util::json_bool(r.ok);
        obj["micronaut_id"] = util::json_string(*mid);
        obj["result"]       = util::json_string(r.message);
        obj["applied"]      = util::json_bool(r.ok);

        // FoldDelta extraction. Precondition: r.ok == true (GOOD verdict only).
        // Measure the verified transition — do not decide what ought to have been learned.
        // accepted_caps / accepted_governs are still valid (candidate was moved, not these).
        if (r.ok) {
            kuhul::FoldDelta delta;
            delta.micronaut_id   = *mid;
            delta.verdict        = "GOOD";
            delta.extracted_at   = format_local_time(
                std::chrono::system_clock::now(), "%Y-%m-%dT%H:%M:%S");
            delta.weights.micronaut_id    = *mid;  // delta_W = 0.0f: propose_evolve unchanged
            delta.confidence.micronaut_id = *mid;  // delta_C = 0.0f: propose_evolve unchanged

            for (const auto& cap : accepted_caps)
                delta.relations.added_capabilities.push_back({*mid, cap});
            for (const auto& gov : accepted_governs)
                delta.relations.added_governs.push_back({*mid, gov});

            // Serialize and log to JROM. Returned seq is the provenance handle.
            util::JsonObject dlog;
            dlog["micronaut_id"]  = util::json_string(delta.micronaut_id);
            dlog["verdict"]       = util::json_string(delta.verdict);
            dlog["extracted_at"]  = util::json_string(delta.extracted_at);
            dlog["delta_W"]       = util::json_number(0.0);
            dlog["delta_C"]       = util::json_number(0.0);
            {
                util::JsonArray arr;
                for (const auto& e : delta.relations.added_capabilities)
                    arr.push_back(util::json_string(e.cap_token));
                dlog["added_capabilities"] = util::json_array(std::move(arr));
            }
            {
                util::JsonArray arr;
                for (const auto& e : delta.relations.added_governs)
                    arr.push_back(util::json_string(e.scope_token));
                dlog["added_governs"] = util::json_array(std::move(arr));
            }
            delta.jrom_seq = jrom.log("fold_delta",
                util::stringify(util::json_object(std::move(dlog))));

            direction_store.ingest(delta);

            // Include provenance in the RPC response.
            util::JsonObject fdout;
            fdout["jrom_seq"]     = util::json_number(static_cast<double>(delta.jrom_seq));
            fdout["micronaut_id"] = util::json_string(delta.micronaut_id);
            fdout["extracted_at"] = util::json_string(delta.extracted_at);
            fdout["delta_W"]      = util::json_number(0.0);
            fdout["delta_C"]      = util::json_number(0.0);
            {
                util::JsonArray arr;
                for (const auto& e : delta.relations.added_capabilities)
                    arr.push_back(util::json_string(e.cap_token));
                fdout["added_capabilities"] = util::json_array(std::move(arr));
            }
            {
                util::JsonArray arr;
                for (const auto& e : delta.relations.added_governs)
                    arr.push_back(util::json_string(e.scope_token));
                fdout["added_governs"] = util::json_array(std::move(arr));
            }
            obj["fold_delta"] = util::json_object(std::move(fdout));
        }

        return util::json_object(std::move(obj));
    });

    rpc_server.register_method("fold_graph_parse", [&](const util::JsonPtr& p) {
        auto query = p ? p->get_string("query") : std::nullopt;
        if (!query || trim_copy(*query).empty()) {
            util::JsonObject err;
            err["ok"] = util::json_bool(false);
            err["error"] = util::json_string("requires non-empty 'query' field");
            return util::json_object(std::move(err));
        }
        auto payload = fold_graph_parse_payload(*query);
        ensure_governance_metadata(
            payload,
            "FGP-µ",
            false,
            "fold_graph_parse_requested",
            "Wo.fold_graph_parse");
        return payload;
    });

    const fs::path prompt_ingest_root = fs::path(cfg.data_dir).parent_path();
    const auto& dictionary_lexicon = dictionary_lexicon_cache(cfg.data_dir);

    // key_query: respond from stored key without calling model
    std::string last_topic_anchor = "global";
    std::vector<std::string> last_relation_confidence_keys;
    std::string last_active_micronaut;
    std::unordered_set<std::string> known_app_types = {
        "dashboard", "ecommerce", "enterprise", "social", "blog"
    };
    rpc_server.register_method("key_query", [&](const util::JsonPtr& p) {
        auto query = p ? p->get_string("query") : std::nullopt;
        if (!query) return util::json_null();
        const bool verified = is_feedback_verified(*query);
        const auto qtrim_cmd = trim_copy(*query);
        const auto qlow_cmd = lower_copy(qtrim_cmd);
        const auto prompt_archive = persist_prompt_submission(
            *query, prompt_ingest_root, &dictionary_lexicon, &factory, &vm.gram_store());
        {
            util::JsonObject plog;
            plog["source"] = util::json_string("key_query");
            plog["prompt_key"] = util::json_string(prompt_archive.prompt_key);
            plog["prompt_len"] = util::json_number(static_cast<double>(query->size()));
            plog["prompt_ngrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_ngram_count));
            plog["prompt_unigrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_unigram_count));
            plog["prompt_bigrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_bigram_count));
            plog["prompt_trigrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_trigram_count));
            plog["prompt_quadgrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_quadgram_count));
            plog["prompt_coarse_grams"] = util::json_number(static_cast<double>(prompt_archive.prompt_coarse_gram_count));
            plog["prompt_function_grams"] = util::json_number(static_cast<double>(prompt_archive.prompt_function_gram_count));
            plog["prompt_tool_grams"] = util::json_number(static_cast<double>(prompt_archive.prompt_tool_gram_count));
            plog["prompt_lexicon"] = util::json_number(static_cast<double>(prompt_archive.prompt_lexicon_count));
            plog["prompt_assoc"] = util::json_number(static_cast<double>(prompt_archive.prompt_assoc_count));
            plog["prompt_atoms"] = util::json_number(static_cast<double>(prompt_archive.prompt_atom_count));
            plog["prompt_similarity"] = util::json_number(static_cast<double>(prompt_archive.prompt_similarity_count));
            plog["file_candidates"] = util::json_number(static_cast<double>(prompt_archive.file_candidate_count));
            plog["file_ingested"] = util::json_number(static_cast<double>(prompt_archive.file_ingested_count));
            plog["file_unigrams"] = util::json_number(static_cast<double>(prompt_archive.file_unigram_count));
            plog["file_bigrams"] = util::json_number(static_cast<double>(prompt_archive.file_bigram_count));
            plog["file_trigrams"] = util::json_number(static_cast<double>(prompt_archive.file_trigram_count));
            plog["file_quadgrams"] = util::json_number(static_cast<double>(prompt_archive.file_quadgram_count));
            plog["file_coarse_grams"] = util::json_number(static_cast<double>(prompt_archive.file_coarse_gram_count));
            plog["file_function_grams"] = util::json_number(static_cast<double>(prompt_archive.file_function_gram_count));
            plog["file_tool_grams"] = util::json_number(static_cast<double>(prompt_archive.file_tool_gram_count));
            plog["file_lexicon"] = util::json_number(static_cast<double>(prompt_archive.file_lexicon_count));
            plog["file_atoms"] = util::json_number(static_cast<double>(prompt_archive.file_atom_count));
            plog["file_similarity"] = util::json_number(static_cast<double>(prompt_archive.file_similarity_count));
            plog["pdf_pending"] = util::json_number(static_cast<double>(prompt_archive.file_pdf_pending_count));
            jrom.log("prompt_archive", util::stringify(util::json_object(std::move(plog))));
        }
        const std::string prompt_event_key = record_key_grammar_memory_event(
            "prompt",
            "key_query",
            "USER-µ",
            "Pop",
            prompt_archive.prompt_key,
            "",
            *query,
            "",
            "UNKNOWN",
            false,
            &factory,
            &vm.gram_store(),
            &jrom);

        auto finalize_key_query_event = [&](util::JsonPtr out,
                                            const std::string& kind_hint,
                                            const std::string& track_hint,
                                            const std::string& phase_hint) -> util::JsonPtr {
            if (!out || out->is_null() || !out->is_object()) return out;
            const std::string event_kind =
                kind_hint.empty() ? infer_memory_event_kind_from_payload(out) : normalize_memory_event_kind(kind_hint);

            std::string track = track_hint;
            if (track.empty()) {
                track = out->get_string("governing_track").value_or("");
                if (track.empty()) track = out->get_string("delegate_track").value_or("");
            }
            if (track.empty())
                track = event_kind == "plan" ? "PLAN-µ" : "Mx2LM";

            std::string phase = phase_hint;
            if (phase.empty())
                phase = out->get_string("response_phase").value_or("");

            std::string source = out->get_string("source").value_or("key_query");
            if (source.empty()) source = "key_query";

            const std::string epistemic_state = infer_memory_event_epistemic_state(out, event_kind);
            bool observation_authority = out->get_bool("observation_authority").value_or(false);
            if (!observation_authority &&
                epistemic_state == "SUPPORTED" &&
                out->get_string("source").value_or("") == "key_store")
                observation_authority = true;

            const std::string event_key = record_key_grammar_memory_event(
                event_kind,
                source,
                track,
                phase,
                prompt_archive.prompt_key,
                prompt_event_key,
                memory_event_content_from_payload(out),
                util::stringify(out),
                epistemic_state,
                observation_authority,
                &factory,
                &vm.gram_store(),
                &jrom);
            attach_memory_event_metadata(
                out,
                event_key,
                event_kind,
                prompt_event_key,
                prompt_archive.prompt_key);
            return out;
        };

        std::string fold_parse_payload;
        const bool fold_parse_prefixed =
            extract_prefixed_payload_ci(
                qtrim_cmd,
                qlow_cmd,
                {"fgp:", "@fgp", "fold_graph_parse:", "fold:"},
                &fold_parse_payload);
        const bool fold_parse_phrase =
            contains_phrase_ci(qlow_cmd, "fold graph parse") ||
            contains_phrase_ci(qlow_cmd, "parse fold graph");
        if (fold_parse_prefixed || fold_parse_phrase) {
            const std::string parse_query =
                (!fold_parse_payload.empty() ? fold_parse_payload : qtrim_cmd);
            auto payload = fold_graph_parse_payload(parse_query);
            ensure_governance_metadata(
                payload,
                "FGP-µ",
                false,
                "fold_graph_parse_requested",
                "Wo.fold_graph_parse");
            return finalize_key_query_event(payload, "", "FGP-µ", "Wo");
        }

        std::string synonymize_payload;
        const bool synonymize_prefixed =
            extract_prefixed_payload_ci(
                qtrim_cmd,
                qlow_cmd,
                {"synonymize:", "synonym:", "syn:", "@syn", "rewrite_words:", "rewrite-words:"},
                &synonymize_payload);
        const bool synonymize_phrase =
            contains_phrase_ci(qlow_cmd, "rewrite with synonyms") ||
            contains_phrase_ci(qlow_cmd, "change words keep meaning") ||
            contains_phrase_ci(qlow_cmd, "replace words keep meaning");
        if (synonymize_prefixed || synonymize_phrase) {
            const std::string source_text =
                (!synonymize_payload.empty() ? synonymize_payload : qtrim_cmd);
            auto payload = synonym_phase_payload(source_text, &dictionary_lexicon);
            ensure_governance_metadata(
                payload,
                "Mx2LM",
                false,
                "lexical_equivalence_requested",
                "Wo.synonymize");
            return finalize_key_query_event(payload, "", "Mx2LM", "Wo");
        }

        std::string lexicon_payload;
        const bool lexicon_prefixed =
            extract_prefixed_payload_ci(
                qtrim_cmd,
                qlow_cmd,
                {"lexicon:", "lex:", "unigram:", "lemma:"},
                &lexicon_payload);
        if (lexicon_prefixed) {
            const auto payload_low = lower_copy(lexicon_payload);
            const bool list_mode =
                lexicon_payload.empty() || payload_low == "list" ||
                payload_low == "status" || payload_low == "show" ||
                payload_low == "?" || payload_low == "*";

            util::JsonObject obj;
            obj["ok"] = util::json_bool(true);
            obj["source"] = util::json_string("lexicon");
            obj["lexicon_path"] = util::json_string("data/gram/english-dictionary.jsonl");

            if (list_mode) {
                obj["status"] = util::json_string("lexicon_status");
                obj["entry_count"] = util::json_number(
                    static_cast<double>(dictionary_lexicon.definition_by_word.size()));
                obj["similarity_entry_count"] = util::json_number(
                    static_cast<double>(dictionary_lexicon.similarity_by_word.size()));

                std::vector<std::string> words;
                words.reserve(dictionary_lexicon.definition_by_word.size());
                for (const auto& [word, _] : dictionary_lexicon.definition_by_word)
                    words.push_back(word);
                std::sort(words.begin(), words.end());

                util::JsonArray sample;
                const size_t cap = std::min<size_t>(16, words.size());
                for (size_t i = 0; i < cap; ++i)
                    sample.push_back(util::json_string(words[i]));
                obj["sample_words"] = util::json_array(std::move(sample));
            } else {
                auto entry = make_lexicon_entry_payload(lexicon_payload, &dictionary_lexicon);
                if (!entry || entry->is_null()) {
                    obj["status"] = util::json_string("lexicon_miss");
                    obj["query"] = util::json_string(lexicon_payload);
                } else {
                    obj["status"] = util::json_string("lexicon_entry");
                    obj["entry"] = entry;

                    const auto lemma = entry->get_string("lemma").value_or("");
                    const auto key = entry->get_string("@key").value_or("");
                    const auto book = entry->get_string("@book").value_or("");

                    const auto packed = util::stringify(entry);
                    if (!lemma.empty()) {
                        factory.record_key("lexicon:" + lemma, packed);
                        factory.record_key("unigram:" + lemma, packed);
                        if (!key.empty()) factory.record_key("key:" + lemma, key);
                        if (!book.empty()) factory.record_key("book:" + lemma, book);
                        obj["key_id"] = util::json_string(factory.key_id_for(lemma));
                    }
                }
            }
            return finalize_key_query_event(
                util::json_object(std::move(obj)), "", "Mx2LM", "Pop");
        }

        std::string user_manifest_payload;
        const bool user_manifest_prefixed =
            extract_prefixed_payload_ci(
                qtrim_cmd,
                qlow_cmd,
                {"usermanifest:", "user_manifest:", "user.manifest:"},
                &user_manifest_payload);
        if (user_manifest_prefixed) {
            const auto filter = sanitize_user_manifest_segment(user_manifest_payload);
            const bool list_mode =
                user_manifest_payload.empty() ||
                filter == "list" || filter == "show" ||
                filter == "status" || filter == "global" ||
                filter == "?";

            util::JsonObject obj;
            obj["ok"] = util::json_bool(true);
            obj["source"] = util::json_string("user.manifest");
            obj["status"] = util::json_string("user_manifest_snapshot");
            obj["namespace"] = util::json_string("user.manifest:rlhf:*");
            obj["sandboxed"] = util::json_bool(true);

            util::JsonArray rows;
            for (const auto& k : vm.gram_store().keys()) {
                const auto kl = lower_copy(k);
                if (kl.rfind("user.manifest:", 0) != 0) continue;
                if (!list_mode && filter != "global" && kl.find(filter) == std::string::npos) continue;
                auto g = vm.gram_store().gramRead(k);
                if (!g) continue;
                util::JsonObject row;
                row["key"] = util::json_string(g->key);
                row["value"] = util::json_string(g->value);
                row["phase"] = util::json_string(g->phase);
                rows.push_back(util::json_object(std::move(row)));
                if (rows.size() >= 32) break;
            }

            obj["count"] = util::json_number(static_cast<double>(rows.size()));
            obj["rows"] = util::json_array(std::move(rows));
            return finalize_key_query_event(
                util::json_object(std::move(obj)), "", "USER-µ", "Pop");
        }

        if (qlow_cmd.rfind("memory:", 0) == 0 || qlow_cmd.rfind("@memory", 0) == 0) {
            std::string payload;
            if (qlow_cmd.rfind("memory:", 0) == 0) payload = trim_copy(qtrim_cmd.substr(7));
            else {
                payload = trim_copy(qtrim_cmd.substr(7)); // "@memory"
                if (!payload.empty() && payload.front() == ':')
                    payload = trim_copy(payload.substr(1));
            }
            const auto payload_low = lower_copy(payload);
            const bool recall_mode =
                payload.empty() || payload_low == "list" || payload_low == "recall" ||
                payload_low == "show" || payload_low == "?";

            util::JsonObject obj;
            obj["ok"] = util::json_bool(true);
            obj["source"] = util::json_string("memory_store");
            obj["memory_tag"] = util::json_string("<memory> @memory");

            if (recall_mode) {
                util::JsonArray memories;
                for (const auto& k : vm.gram_store().keys()) {
                    if (lower_copy(k).rfind("memory:", 0) != 0) continue;
                    auto g = vm.gram_store().gramRead(k);
                    if (!g) continue;
                    util::JsonObject row;
                    row["key"] = util::json_string(g->key);
                    row["value"] = util::json_string(g->value);
                    row["phase"] = util::json_string(g->phase);
                    memories.push_back(util::json_object(std::move(row)));
                    if (memories.size() >= 24) break;
                }
                obj["status"] = util::json_string("memory_recall");
                obj["count"] = util::json_number((double)memories.size());
                obj["memories"] = util::json_array(std::move(memories));
                {
                    util::JsonObject recall_log;
                    recall_log["mode"]  = util::json_string("explicit");
                    recall_log["query"] = util::json_string(qtrim_cmd);
                    jrom.log("memory_recall",
                             util::stringify(util::json_object(std::move(recall_log))));
                }
            } else {
                const std::string memory_key = "memory: " + payload;
                vm.gram_store().gramWrite(memory_key, payload, "Ch'en");
                factory.record_key(memory_key, payload);
                factory.record_key("@memory " + payload, payload);

                obj["status"] = util::json_string("memory_recorded");
                obj["memory_key"] = util::json_string(memory_key);
                obj["value"] = util::json_string(payload);

                util::JsonObject mlog;
                mlog["key"] = util::json_string(memory_key);
                mlog["value"] = util::json_string(payload);
                mlog["mode"] = util::json_string("append");
                jrom.log("memory_append", util::stringify(util::json_object(std::move(mlog))));
            }

            return finalize_key_query_event(
                util::json_object(std::move(obj)), "", "MEMORY-µ", "Ch'en");
        }

        auto update_last_topic = [&](const TopicConfidenceUpdate& tc) {
            if (tc.topic_key.rfind("topic:", 0) != 0) return;
            auto anchor = tc.topic_key.substr(6);
            if (!anchor.empty()) last_topic_anchor = anchor;
        };

        auto resolved_hint = [&](const std::string& hint) {
            return resolve_feedback_topic_hint(*query, hint, last_topic_anchor);
        };

        auto update_last_micronaut = [&](const std::string& id) {
            if (!id.empty()) last_active_micronaut = id;
        };

        auto log_micronaut_feedback = [&](const MicronautConfidenceUpdate& mc) {
            if (mc.feedback_event) {
                const auto event_key = sanitize_user_manifest_segment(mc.event);
                const auto topic_key = sanitize_user_manifest_segment(last_topic_anchor);
                const auto user_key = "user.manifest:rlhf:" + event_key + ":" + topic_key;

                util::JsonObject rlhf;
                rlhf["event"] = util::json_string(mc.event);
                rlhf["micronaut"] = util::json_string(mc.micronaut_id);
                rlhf["delta"] = util::json_number(mc.delta);
                rlhf["before"] = util::json_number(mc.before);
                rlhf["after"] = util::json_number(mc.after);
                rlhf["topic_anchor"] = util::json_string(last_topic_anchor);
                rlhf["commit_phase"] = util::json_string("Ch'en_pending");
                rlhf["sandbox"] = util::json_string("user-manifest");
                const auto payload = util::stringify(util::json_object(std::move(rlhf)));

                vm.gram_store().gramWrite(
                    user_key,
                    payload,
                    "Ch'en",
                    "user_manifest_rlhf",
                    "definitions_and_relations");
                factory.record_key(user_key, payload);
                factory.record_key("user:rlhf:" + event_key + ":" + topic_key, payload);
            }

            if (!mc.committed || !mc.target_resolved) return;
            jrom.log("micronaut_confidence_update",
                     "{\"micronaut\":\"" + mc.micronaut_id +
                     "\",\"delta\":" + std::to_string(mc.delta) +
                     ",\"event\":\"" + mc.event + "\"}");
        };

        // ── Explicit web-research intent ──────────────────────────────────
        if (infer_micronaut_from_text(*query) == "WEB-RESEARCH-µ" && !verified) {
            const auto qtrim = trim_copy(*query);
            const auto research_query = normalize_web_research_query(qtrim);
            auto tc = apply_topic_feedback(confidence, *query, resolved_hint(research_query), verified);
            bool used_last = false;
            auto target_mu = resolve_feedback_micronaut_target(
                *query, research_query, "WEB-RESEARCH-µ", last_active_micronaut, &used_last);
            auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
            apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);

            jrom.log("web_research",
                     "{\"query\":\"" + research_query +
                     "\",\"delegate\":\"WEB-RESEARCH-µ\",\"retrieval\":\"Librarian\",\"source\":\"user_request\"}");

            util::JsonObject obj;
            obj["ok"]     = util::json_bool(true);
            obj["source"] = util::json_string("web-research");
            obj["status"] = util::json_string("research_plan_ready");
            obj["text"]   = util::json_string("Research plan queued for: " + research_query);
            obj["response"] = util::json_string("Research plan queued for: " + research_query);

            util::JsonObject rp;
            rp["mode"]             = util::json_string("user-requested");
            rp["delegate_track"]   = util::json_string("WEB-RESEARCH-µ");
            rp["retrieval_track"]  = util::json_string("Librarian");
            rp["policy"]           = util::json_string("royalty-free-only");
            rp["license_gate"]     = util::json_string("allow:CC0|CC-BY|Unsplash|Pexels|Pixabay");
            rp["citation_required"]= util::json_bool(true);
            rp["query"]            = util::json_string(research_query);
            obj["research"] = util::json_object(std::move(rp));

            last_relation_confidence_keys.clear();
            update_last_topic(tc);
            update_last_micronaut("WEB-RESEARCH-µ");
            log_micronaut_feedback(mc);
            auto out = util::json_object(std::move(obj));
            ensure_governance_metadata(
                out,
                "WEB-RESEARCH-µ",
                true,
                "user_requested_research",
                "Pop.research_dispatch");
            attach_topic_confidence(out, tc);
            attach_micronaut_confidence(out, mc);
            return finalize_key_query_event(out, "plan", "WEB-RESEARCH-µ", "Pop");
        }

        // ── Explicit popular AI intent keys ───────────────────────────────
        if (auto intent = popular_intent_payload(*query); intent && !intent->is_null()) {
            auto delegate_track = intent->get_string("delegate_track").value_or("");
            auto topic_hint = intent->get_string("query").value_or(*query);
            if (delegate_track.empty()) delegate_track = infer_micronaut_from_text(*query);

            auto tc = apply_topic_feedback(confidence, *query, resolved_hint(topic_hint), verified);
            bool used_last = false;
            auto target_mu = resolve_feedback_micronaut_target(
                *query, topic_hint, delegate_track, last_active_micronaut, &used_last);
            auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
            apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
            ensure_governance_metadata(
                intent,
                delegate_track,
                true,
                "intent_plan_requested",
                "Wo.intent_plan");
            attach_topic_confidence(intent, tc);
            attach_micronaut_confidence(intent, mc);
            last_relation_confidence_keys.clear();
            update_last_topic(tc);
            update_last_micronaut(delegate_track);
            log_micronaut_feedback(mc);

            util::JsonObject ilog;
            ilog["query"] = util::json_string(*query);
            ilog["delegate_track"] = util::json_string(delegate_track);
            ilog["intent_key"] = util::json_string(intent->get_string("intent_key").value_or(""));
            ilog["op"] = util::json_string(intent->get_string("op").value_or(""));
            jrom.log("popular_intent_plan", util::stringify(util::json_object(std::move(ilog))));
            return finalize_key_query_event(intent, "plan", delegate_track, "Wo");
        }

        // ── Explicit common tool-call keys ────────────────────────────────
        if (auto tool_plan = common_tool_call_payload(*query); tool_plan && !tool_plan->is_null()) {
            auto delegate_track = tool_plan->get_string("delegate_track").value_or("TOOL-µ");
            auto topic_hint = tool_plan->get_string("query").value_or(*query);
            auto tc = apply_topic_feedback(confidence, *query, resolved_hint(topic_hint), verified);
            bool used_last = false;
            auto target_mu = resolve_feedback_micronaut_target(
                *query, topic_hint, delegate_track, last_active_micronaut, &used_last);
            auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
            apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
            ensure_governance_metadata(
                tool_plan,
                delegate_track,
                true,
                "tool_plan_requested",
                "Wo.tool_plan");
            attach_topic_confidence(tool_plan, tc);
            attach_micronaut_confidence(tool_plan, mc);
            last_relation_confidence_keys.clear();
            update_last_topic(tc);
            update_last_micronaut(delegate_track);
            log_micronaut_feedback(mc);

            util::JsonObject tlog;
            tlog["query"] = util::json_string(*query);
            tlog["delegate_track"] = util::json_string(delegate_track);
            tlog["intent_key"] = util::json_string(tool_plan->get_string("intent_key").value_or(""));
            tlog["op"] = util::json_string(tool_plan->get_string("op").value_or(""));
            tlog["call_count"] = util::json_number(
                tool_plan->get_number("call_count").value_or(0.0));
            jrom.log("common_tool_call_plan", util::stringify(util::json_object(std::move(tlog))));
            return finalize_key_query_event(tool_plan, "plan", delegate_track, "Wo");
        }

        if (auto utility = utility_runtime_payload(*query, &factory, &vm.gram_store()); utility && !utility->is_null()) {
            auto delegate_track = utility->get_string("delegate_track").value_or("TOOL-µ");
            ensure_governance_metadata(
                utility,
                delegate_track,
                false,
                "deterministic_utility_path",
                "Pop.utility_resolution");
            auto topic_hint = utility->get_string("response").value_or(*query);
            auto tc = apply_topic_feedback(confidence, *query, resolved_hint(topic_hint), verified);
            bool used_last = false;
            auto target_mu = resolve_feedback_micronaut_target(
                *query, topic_hint, delegate_track, last_active_micronaut, &used_last);
            auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
            apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
            attach_topic_confidence(utility, tc);
            attach_micronaut_confidence(utility, mc);
            last_relation_confidence_keys.clear();
            update_last_topic(tc);
            update_last_micronaut(delegate_track);
            log_micronaut_feedback(mc);

            util::JsonObject ulog;
            ulog["query"] = util::json_string(*query);
            ulog["delegate_track"] = util::json_string(delegate_track);
            ulog["status"] = util::json_string(utility->get_string("status").value_or(""));
            ulog["op"] = util::json_string(utility->get_string("op").value_or(""));
            jrom.log("utility_intent", util::stringify(util::json_object(std::move(ulog))));
            return finalize_key_query_event(utility, "", delegate_track, "Pop");
        }

        std::string app_type;
        bool used_existing_type = false;
        if (auto app = app_wizard_payload(*query, known_app_types, &app_type, &used_existing_type);
            app && !app->is_null()) {
            auto topic_hint = app_type.empty() ? *query : app_type;
            auto tc = apply_topic_feedback(confidence, *query, resolved_hint(topic_hint), verified);
            bool used_last = false;
            auto target_mu = resolve_feedback_micronaut_target(
                *query, topic_hint, "APP-WIZARD-µ", last_active_micronaut, &used_last);
            auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
            apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
            ensure_governance_metadata(
                app,
                "APP-WIZARD-µ",
                true,
                "app_wizard_plan_requested",
                "Wo.app_wizard_plan");
            attach_topic_confidence(app, tc);
            attach_micronaut_confidence(app, mc);
            last_relation_confidence_keys.clear();
            update_last_topic(tc);
            update_last_micronaut("APP-WIZARD-µ");
            log_micronaut_feedback(mc);
            if (!app_type.empty()) known_app_types.insert(app_type);
            jrom.log("compose_plan",
                     "{\"source\":\"app-wizard\",\"app_type\":\"" + (app_type.empty() ? std::string("unknown") : app_type) +
                     "\",\"ops\":[\"compose\",\"scaffold\",\"emit\",\"publish\",\"blueprint\",\"template\",\"policy\"]" +
                     ",\"known\":" + std::string(used_existing_type ? "true" : "false") + "}");
            return finalize_key_query_event(app, "plan", "APP-WIZARD-µ", "Wo");
        }

        if (auto mx = mx2lm_ngram_payload(
                *query,
                "mx2lm-direct",
                manifest_reg,
                &dictionary_lexicon,
                &factory,
                &vm.gram_store());
            mx && !mx->is_null()) {
            auto tc = apply_topic_feedback(confidence, *query, resolved_hint(*query), verified);
            bool used_last = false;
            auto target_mu = resolve_feedback_micronaut_target(
                *query, *query, "Mx2LM", last_active_micronaut, &used_last);
            auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
            apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
            const auto anchor = topic_anchor_from_key(tc.topic_key);
            attach_relation_confidence(mx, confidence, anchor);
            attach_topic_confidence(mx, tc);
            attach_micronaut_confidence(mx, mc);
            last_relation_confidence_keys = collect_relation_confidence_keys(mx, anchor);
            update_last_topic(tc);
            update_last_micronaut("Mx2LM");
            log_micronaut_feedback(mc);
            {
                // nodes[] → ⟁EVIDENCE: event name gives "semantic"+"graph"+"query" scoring
                util::JsonObject sgq;
                sgq["source"] = util::json_string("mx2lm");
                sgq["anchor"] = util::json_string(anchor);
                jrom.log("semantic_graph_query",
                         util::stringify(util::json_object(std::move(sgq))));
            }
            // C_ij commit → ⟁VERIFY: log only when relation confidence was actually updated
            if (tc.relation_updates > 0) {
                util::JsonObject prov_log;
                prov_log["confidence_keys"] = util::json_number((double)tc.relation_updates);
                prov_log["rel_delta"]       = util::json_number(tc.relation_delta);
                jrom.log("check_provenance_weights",
                         util::stringify(util::json_object(std::move(prov_log))));
            }
            return finalize_key_query_event(mx, "", "Mx2LM", "Ch'en");
        }

        const bool question_like_after_mx2lm = is_question_like_text(*query);

        if (!question_like_after_mx2lm) {
            auto hit = factory.lookup_key(*query);
            if (hit) {
                auto tc = apply_topic_feedback(confidence, *query, resolved_hint(*hit), verified);
                auto branch_default = infer_micronaut_from_text(*hit);
                if (branch_default.empty()) branch_default = infer_micronaut_from_text(*query);
                bool used_last = false;
                auto target_mu = resolve_feedback_micronaut_target(
                    *query, *hit, branch_default, last_active_micronaut, &used_last);
                auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
                apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
                util::JsonObject obj;
                obj["ok"]       = util::json_bool(true);
                obj["response"] = util::json_string(*hit);
                obj["source"]   = util::json_string("key_store");
                auto out = util::json_object(std::move(obj));
                attach_topic_confidence(out, tc);
                attach_micronaut_confidence(out, mc);
                update_last_topic(tc);
                update_last_micronaut(branch_default);
                log_micronaut_feedback(mc);
                return finalize_key_query_event(out, "model_response", branch_default, "Ch'en");
            }

            auto aiml_hit = aiml.match(*query);
            if (!aiml_hit.empty()) {
                if (auto mx = mx2lm_ngram_payload(
                        aiml_hit,
                        "aiml-mx2lm",
                        manifest_reg,
                        &dictionary_lexicon,
                        &factory,
                        &vm.gram_store());
                    mx && !mx->is_null()) {
                    auto tc = apply_topic_feedback(confidence, *query, resolved_hint(aiml_hit), verified);
                    bool used_last = false;
                    auto target_mu = resolve_feedback_micronaut_target(
                        *query, aiml_hit, "Mx2LM", last_active_micronaut, &used_last);
                    auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
                    apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
                    const auto anchor = topic_anchor_from_key(tc.topic_key);
                    attach_relation_confidence(mx, confidence, anchor);
                    attach_topic_confidence(mx, tc);
                    attach_micronaut_confidence(mx, mc);
                    last_relation_confidence_keys = collect_relation_confidence_keys(mx, anchor);
                    update_last_topic(tc);
                    update_last_micronaut("Mx2LM");
                    log_micronaut_feedback(mc);
                    {
                        util::JsonObject sgq;
                        sgq["source"] = util::json_string("mx2lm");
                        sgq["anchor"] = util::json_string(anchor);
                        jrom.log("semantic_graph_query",
                                 util::stringify(util::json_object(std::move(sgq))));
                    }
                    if (tc.relation_updates > 0) {
                        util::JsonObject prov_log;
                        prov_log["confidence_keys"] = util::json_number((double)tc.relation_updates);
                        prov_log["rel_delta"]       = util::json_number(tc.relation_delta);
                        jrom.log("check_provenance_weights",
                                 util::stringify(util::json_object(std::move(prov_log))));
                    }
                    return finalize_key_query_event(mx, "", "Mx2LM", "Ch'en");
                }

                auto tc = apply_topic_feedback(confidence, *query, resolved_hint(aiml_hit), verified);
                auto branch_default = infer_micronaut_from_text(aiml_hit);
                if (branch_default.empty()) branch_default = infer_micronaut_from_text(*query);
                bool used_last = false;
                auto target_mu = resolve_feedback_micronaut_target(
                    *query, aiml_hit, branch_default, last_active_micronaut, &used_last);
                auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
                apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);
                util::JsonObject obj;
                obj["ok"]       = util::json_bool(true);
                obj["response"] = util::json_string(aiml_hit);
                obj["source"]   = util::json_string("aiml");
                auto out = util::json_object(std::move(obj));
                ensure_governance_metadata(
                    out,
                    branch_default,
                    true,
                    "aiml_response_projection",
                    "Wo.aiml_projection");
                attach_topic_confidence(out, tc);
                attach_micronaut_confidence(out, mc);
                update_last_topic(tc);
                update_last_micronaut(branch_default);
                log_micronaut_feedback(mc);
                return finalize_key_query_event(out, "model_response", branch_default, "Wo");
            }
        }

        auto tc = apply_topic_feedback(confidence, *query, resolved_hint(*query), verified);
        auto branch_default = infer_micronaut_from_text(*query);
        bool used_last = false;
        auto target_mu = resolve_feedback_micronaut_target(
            *query, *query, branch_default, last_active_micronaut, &used_last);
        auto mc = apply_micronaut_feedback(confidence, tc, target_mu, used_last);
        apply_relation_feedback(confidence, tc, last_relation_confidence_keys, last_topic_anchor);

        util::JsonObject obj;
        obj["ok"]     = util::json_bool(false);
        obj["source"] = util::json_string("miss");
        obj["status"] = util::json_string("no_fact_found");
        obj["response"] = util::json_string("= no fact found");

        std::string active_micronaut = branch_default;
        const bool feedback_event = (tc.event == "BAD_ANSWER" || tc.event == "GOOD_ANSWER");
        if (!feedback_event) {
            tc.event = "MISS";
            tc.signal = "insufficient_context";
            if (tc.commit_phase == "NONE")
                tc.commit_phase = "Ch'en_pending";

            io::RouteRequest miss_req;
            miss_req.query = *query;
            miss_req.phase = kuhul::Phase::Pop;
            miss_req.lane = "web_research_miss";
            auto miss_route = router.route(miss_req);
            const std::string delegate_track = "WEB-RESEARCH-µ";

            util::JsonObject route_hint;
            route_hint["micronaut"] = util::json_string(
                miss_route.micronaut_id.empty() ? "Librarian" : miss_route.micronaut_id);
            route_hint["handler"] = util::json_string(miss_route.handler_id);
            route_hint["selected_task_op"] = util::json_string(miss_route.selected_task_op);
            route_hint["resolver_policy"] = util::json_string(miss_route.resolver_policy);
            route_hint["required_primitive"] = util::json_string(miss_route.required_primitive);
            route_hint["policy_filter_applied"] = util::json_bool(miss_route.policy_filter_applied);
            route_hint["capability_miss"] = util::json_bool(miss_route.capability_miss);
            route_hint["selected_capable"] = util::json_bool(miss_route.selected_capable);
            route_hint["policy_satisfied"] = util::json_bool(miss_route.policy_satisfied);
            route_hint["xcfe_authorized"] = util::json_bool(miss_route.xcfe_authorized);
            route_hint["dispatch_allowed"] = util::json_bool(miss_route.dispatch_allowed);
            route_hint["dispatch_block_reason"] = util::json_string(miss_route.dispatch_block_reason);
            route_hint["score"] = util::json_number(miss_route.score);
            route_hint["fold_weight"] = util::json_number(miss_route.fold_weight);
            route_hint["phase_driver_weight"] = util::json_number(miss_route.phase_driver_weight);
            route_hint["pattern_strength"] = util::json_number(miss_route.pattern_strength);
            route_hint["static_weight"] = util::json_number(miss_route.static_weight);
            route_hint["confidence_score"] = util::json_number(miss_route.confidence_score);
            route_hint["health_score"] = util::json_number(miss_route.health_score);
            route_hint["relevance_score"] = util::json_number(miss_route.relevance_score);
            route_hint["bow_score"] = util::json_number(miss_route.bow_score);
            route_hint["distributional_score"] = util::json_number(miss_route.distributional_score);
            route_hint["lexical_semantic_score"] = util::json_number(miss_route.lexical_semantic_score);
            route_hint["nsm_prime_score"] = util::json_number(miss_route.nsm_prime_score);
            route_hint["function_word_score"] = util::json_number(miss_route.function_word_score);
            route_hint["content_word_ratio"] = util::json_number(miss_route.content_word_ratio);
            route_hint["function_word_ratio"] = util::json_number(miss_route.function_word_ratio);
            route_hint["content_function_binding_ratio"] = util::json_number(miss_route.content_function_binding_ratio);
            route_hint["word_sense_score"] = util::json_number(miss_route.word_sense_score);
            route_hint["compositional_phrase_score"] = util::json_number(miss_route.compositional_phrase_score);
            route_hint["coarse_ngram_score"] = util::json_number(miss_route.coarse_ngram_score);
            route_hint["tool_gram_score"] = util::json_number(miss_route.tool_gram_score);
            route_hint["miss_dispatch_score"] = util::json_number(miss_route.miss_dispatch_score);
            route_hint["semantic_score"] = util::json_number(miss_route.semantic_score);
            route_hint["score_policy"] = util::json_string(io::ROUTE_SCORE_POLICY);
            route_hint["required_capabilities"] = util::json_array(
                json_string_array(miss_route.required_capabilities));
            route_hint["eligible_resolvers"] = util::json_array(
                json_string_array(miss_route.eligible_resolvers));
            route_hint["ineligible_resolvers"] = util::json_array(
                json_string_array(miss_route.ineligible_resolvers));

            util::JsonArray blend_arr;
            for (const auto& [tid, w] : miss_route.blend) {
                util::JsonObject row;
                row["track"] = util::json_string(tid);
                row["weight"] = util::json_number((double)w);
                blend_arr.push_back(util::json_object(std::move(row)));
            }
            route_hint["blend"] = util::json_array(std::move(blend_arr));
            obj["route_hint"] = util::json_object(std::move(route_hint));

            const auto qtrim = trim_copy(*query);
            const auto qlow = lower_copy(qtrim);
            auto key_terms = semantic_keywords_from_tokens(micronaut::Mx2LmNGram::tokenize(qlow));
            const auto topic_anchor = topic_anchor_from_key(tc.topic_key);
            if (key_terms.empty() && topic_anchor != "global")
                key_terms.push_back(topic_anchor);
            const auto phrase = join_words(key_terms, 4);

            const auto q_tokens = micronaut::Mx2LmNGram::tokenize(qlow);
            auto has_tok = [&](const std::string& t) {
                return std::find(q_tokens.begin(), q_tokens.end(), t) != q_tokens.end();
            };

            std::vector<std::string> resolved_context;
            if (topic_anchor != "global") resolved_context.push_back(topic_anchor);
            if (!delegate_track.empty()) resolved_context.push_back(delegate_track);
            if (!miss_route.micronaut_id.empty() && miss_route.micronaut_id != delegate_track)
                resolved_context.push_back(miss_route.micronaut_id);
            if (resolved_context.empty()) resolved_context.push_back("route_hint_only");

            std::vector<std::string> missing_context;
            auto push_missing = [&](const std::string& s) {
                if (std::find(missing_context.begin(), missing_context.end(), s) == missing_context.end())
                    missing_context.push_back(s);
            };

            if (has_tok("imply") || has_tok("implies") || has_tok("therefore") || has_tok("entails"))
                push_missing("implication chain evidence");
            if (has_tok("evidence") || has_tok("source") || has_tok("provenance") || has_tok("citation") || has_tok("cite"))
                push_missing("source provenance");
            if (has_tok("graph") || has_tok("incidence") || has_tok("topology") || has_tok("relation"))
                push_missing("graph/incidence relation path");
            if (has_tok("recall") || has_tok("memory") || has_tok("history"))
                push_missing("memory recall trace");
            if (has_tok("math") || has_tok("equation") || has_tok("proof") || has_tok("deterministic"))
                push_missing("deterministic math execution trace");
            if (has_tok("delegate") || has_tok("capability") || has_tok("micronaut"))
                push_missing("capability delegation path");
            if (miss_route.capability_miss) {
                push_missing("resolver capability gap");
                if (!miss_route.required_primitive.empty())
                    push_missing("required primitive " + lower_copy(miss_route.required_primitive));
                for (const auto& cap : miss_route.required_capabilities)
                    push_missing("required capability " + lower_copy(cap));
            }
            if (missing_context.empty())
                push_missing("domain evidence");

            std::vector<std::string> required_folds;
            auto push_fold = [&](const std::string& f) {
                if (std::find(required_folds.begin(), required_folds.end(), f) == required_folds.end())
                    required_folds.push_back(f);
            };
            for (const auto& m : missing_context) {
                if (m.find("implication") != std::string::npos) push_fold("IMPLICATION");
                if (m.find("evidence") != std::string::npos || m.find("provenance") != std::string::npos) push_fold("EVIDENCE");
                if (m.find("graph") != std::string::npos || m.find("incidence") != std::string::npos) push_fold("TOPOLOGY");
                if (m.find("memory") != std::string::npos) push_fold("MEMORY");
                if (m.find("math") != std::string::npos || m.find("deterministic") != std::string::npos) push_fold("MATH");
                if (m.find("capability") != std::string::npos) push_fold("CAPABILITY");
            }
            if (required_folds.empty()) push_fold("EVIDENCE");

            std::string miss_kind = "semantic";
            if (std::find(required_folds.begin(), required_folds.end(), "TOPOLOGY") != required_folds.end()) miss_kind = "topology";
            else if (std::find(required_folds.begin(), required_folds.end(), "MEMORY") != required_folds.end()) miss_kind = "memory";
            else if (std::find(required_folds.begin(), required_folds.end(), "MATH") != required_folds.end()) miss_kind = "mathematical";
            else if (std::find(required_folds.begin(), required_folds.end(), "CAPABILITY") != required_folds.end()) miss_kind = "capability";
            else if (std::find(required_folds.begin(), required_folds.end(), "EVIDENCE") != required_folds.end()) miss_kind = "evidence";
            if (miss_route.capability_miss) miss_kind = "capability";

            io::RuntimeAttentionInputs attn_in;
            attn_in.miss_kind = miss_kind;
            attn_in.required_folds = required_folds;
            attn_in.query_lower = qlow;
            attn_in.relevance_cold_start = key_terms.empty() && topic_anchor == "global";
            attn_in.missing_context_count = missing_context.size();
            attn_in.relevance_score = miss_route.relevance_score;
            attn_in.bow_score = miss_route.bow_score;
            attn_in.distributional_score = miss_route.distributional_score;
            attn_in.confidence_score = miss_route.confidence_score;
            attn_in.static_weight = miss_route.static_weight;
            attn_in.health_score = miss_route.health_score;

            const auto runtime_attention = io::compute_runtime_attention(attn_in);
            const double D = runtime_attention.D;
            const double R = runtime_attention.R;
            const double U = runtime_attention.U;
            const double P = runtime_attention.P;
            const double lexical_prior = runtime_attention.lexical_prior;
            const bool relevance_cold_start = runtime_attention.relevance_cold_start;
            const double P_route = runtime_attention.P_route;
            const double P_query = runtime_attention.P_query;
            const double research_pressure = runtime_attention.research_pressure;
            const auto& focus_pick = runtime_attention.focus;
            const auto& focus_candidates = runtime_attention.candidates;
            const double kRuntimeAttentionAdmission = runtime_attention.admission_threshold;
            const double runtime_attention_score = runtime_attention.score;
            const bool runtime_attention_admitted = runtime_attention.admitted;

            // Legacy aliases retained for downstream payload shape compatibility.
            const double M = D;
            const double I = R;

            const bool code_like =
                contains_phrase_ci(qlow, "code") || contains_phrase_ci(qlow, "coding") ||
                contains_phrase_ci(qlow, "function") || contains_phrase_ci(qlow, "compile") ||
                contains_phrase_ci(qlow, "syntax") || contains_phrase_ci(qlow, "api") ||
                contains_phrase_ci(qlow, "c++") || contains_phrase_ci(qlow, "cpp") ||
                contains_phrase_ci(qlow, "javascript") || contains_phrase_ci(qlow, "typescript") ||
                contains_phrase_ci(qlow, "html") || contains_phrase_ci(qlow, "css") ||
                contains_phrase_ci(qlow, "python");

            std::vector<std::string> govern_seeds = key_terms;
            if (topic_anchor != "global")
                govern_seeds.push_back(topic_anchor);

            const auto factory_mutation = apply_miss_factory_mutation(
                factory, mu_reg, delegate_track, miss_kind, required_folds, govern_seeds, research_pressure);

            util::JsonArray query_variants;
            query_variants.push_back(util::json_string(qtrim));
            if (!phrase.empty()) {
                query_variants.push_back(util::json_string(phrase + " overview"));
                query_variants.push_back(util::json_string(phrase + " evidence"));
                query_variants.push_back(util::json_string(phrase + " mechanism"));
            }

            util::JsonArray manifest_expansions;
            std::unordered_set<std::string> seen_terms;
            for (const auto& term : key_terms) {
                if (!seen_terms.insert(term).second) continue;
                auto expanded = manifest_reg.resolve(term);
                util::JsonArray expanded_arr;
                std::unordered_set<std::string> dedup;
                for (const auto& v : expanded) {
                    if (v.empty()) continue;
                    if (!dedup.insert(v).second) continue;
                    expanded_arr.push_back(util::json_string(v));
                    if (expanded_arr.size() >= 8) break;
                }
                if (expanded_arr.empty()) continue;
                util::JsonObject row;
                row["term"] = util::json_string(term);
                row["expanded"] = util::json_array(std::move(expanded_arr));
                manifest_expansions.push_back(util::json_object(std::move(row)));
                if (manifest_expansions.size() >= 4) break;
            }


            util::JsonObject fallback;
            fallback["mode"] = util::json_string("web-research-fallback");
            fallback["delegate_track"] = util::json_string(delegate_track);
            fallback["retrieval_track"] = util::json_string("Librarian");
            fallback["confidence_state"] = util::json_string("provisional");
            fallback["citation_required"] = util::json_bool(true);
            fallback["query"] = util::json_string(qtrim);
            fallback["topic_anchor"] = util::json_string(topic_anchor);
            fallback["research_pressure"] = util::json_number(research_pressure);
            fallback["runtime_attention_score"] = util::json_number(runtime_attention_score);
            fallback["runtime_attention_admitted"] = util::json_bool(runtime_attention_admitted);
            fallback["query_variants"] = util::json_array(std::move(query_variants));
            fallback["manifest_expansions"] = util::json_array(std::move(manifest_expansions));

            util::JsonArray providers;
            auto push_provider = [&](const std::string& id,
                                     const std::string& name,
                                     const std::string& query_suffix) {
                util::JsonObject p;
                p["id"] = util::json_string(id);
                p["name"] = util::json_string(name);
                p["query"] = util::json_string(
                    phrase.empty() ? (qtrim + " " + query_suffix) : (phrase + " " + query_suffix));
                providers.push_back(util::json_object(std::move(p)));
            };
            if (code_like) {
                push_provider("w3schools", "W3Schools", "site:w3schools.com");
                push_provider("cppreference", "cppreference", "site:en.cppreference.com");
                push_provider("github_docs", "GitHub Docs", "site:docs.github.com");
            }
            push_provider("sciencedirect", "ScienceDirect", "site:sciencedirect.com");
            push_provider("arxiv", "arXiv", "site:arxiv.org");
            push_provider("nasa", "NASA", "site:nasa.gov");
            push_provider("esa", "ESA", "site:esa.int");
            push_provider("pubchem", "PubChem", "site:pubchem.ncbi.nlm.nih.gov");
            push_provider("openalex", "OpenAlex", "site:openalex.org");
            fallback["providers"] = util::json_array(std::move(providers));

            util::JsonArray watch_stream;
            auto push_step = [&](const std::string& phase,
                                 const std::string& op,
                                 const std::string& summary) {
                util::JsonObject step;
                step["phase"] = util::json_string(phase);
                step["op"] = util::json_string(op);
                step["summary"] = util::json_string(summary);
                watch_stream.push_back(util::json_object(std::move(step)));
            };
            push_step("Pop", "mark_fact_miss", "Record miss and infer nearest route candidate.");
            push_step("Wo", "expand_manifest_aliases", "Expand key terms through manifest aliases.");
            push_step("Yax", "draft_research_queries", "Generate query variants for cited research.");
            push_step("Sek", "delegate_librarian", "Delegate retrieval plan to Librarian track.");
            push_step("Ch'en", "verify_sources", "Require source quality and citation provenance checks.");
            push_step("Xul", "record_replay_candidate", "Store query as replay candidate for when fact is committed.");
            push_step("Xul", "evolve_track", "Evolve delegate track capability from confirmed fact gap.");
            fallback["watch_stream"] = util::json_array(std::move(watch_stream));

            // Replay descriptor — re-fires key_query once a result is committed
            util::JsonObject replay;
            replay["method"]     = util::json_string("key_query");
            replay["query"]      = util::json_string(qtrim);
            replay["trigger_on"] = util::json_string("web_research_commit");
            replay["seq"]        = util::json_number((double)jrom.last_seq());
            fallback["replay"]   = util::json_object(std::move(replay));

            // Evolve descriptor — expands delegate track on fact gap confirmation
            util::JsonObject evolve;
            evolve["track"]      = util::json_string(delegate_track);
            evolve["capability"] = util::json_string("fact_retrieval::" + (phrase.empty() ? qtrim.substr(0, 32) : phrase));
            evolve["trigger_on"] = util::json_string("web_research_commit");
            fallback["evolve"]   = util::json_object(std::move(evolve));

            obj["research_fallback"] = util::json_object(std::move(fallback));

            util::JsonObject miss_deficit;
            miss_deficit["@status"] = util::json_string("MISS");
            miss_deficit["@research_needed"] = util::json_bool(true);
            miss_deficit["@query"] = util::json_string(qtrim);
            {
                util::JsonArray arr;
                for (const auto& s : resolved_context) arr.push_back(util::json_string(s));
                miss_deficit["resolved"] = util::json_array(std::move(arr));
            }
            {
                util::JsonArray arr;
                for (const auto& s : missing_context) arr.push_back(util::json_string(s));
                miss_deficit["missing"] = util::json_array(std::move(arr));
            }
            {
                util::JsonArray arr;
                for (const auto& s : required_folds) arr.push_back(util::json_string(s));
                miss_deficit["required_folds"] = util::json_array(std::move(arr));
            }
            miss_deficit["confidence"] = util::json_number((double)miss_route.confidence_score);
            miss_deficit["miss_kind"] = util::json_string(miss_kind);
            miss_deficit["research_pressure"] = util::json_number(research_pressure);
            {
                util::JsonObject rho;
                rho["R_relevance"] = util::json_number(R);
                rho["U_unresolvedness"] = util::json_number(U);
                rho["D_dependency_importance"] = util::json_number(D);
                rho["P_priority"] = util::json_number(P);
                rho["relevance_cold_start"] = util::json_bool(relevance_cold_start);
                rho["relevance_lexical_prior"] = util::json_number(lexical_prior);
                rho["priority_route"] = util::json_number(P_route);
                rho["priority_query"] = util::json_number(P_query);
                // Legacy aliases retained for consumers not yet migrated.
                rho["M_missing_magnitude"] = util::json_number(M);
                rho["I_importance"] = util::json_number(I);
                rho["U_uncertainty"] = util::json_number(U);
                miss_deficit["research_pressure_components"] = util::json_object(std::move(rho));
            }
            obj["miss_deficit"] = util::json_object(std::move(miss_deficit));

            {
                util::JsonObject ra;
                ra["model"] = util::json_string("SGSFG_RSA");
                ra["focus_state"] = util::json_string("MISS");
                ra["focus_node"] = util::json_string(focus_pick.id);
                ra["focus_kind"] = util::json_string(focus_pick.kind);
                ra["focus_slot"] = util::json_string(focus_pick.slot);
                ra["resolver"] = util::json_string(delegate_track);
                ra["return_to"] = util::json_string(
                    branch_default.empty() ? std::string("key_query") : branch_default);
                ra["score"] = util::json_number(runtime_attention_score);
                ra["formula"] = util::json_string("A(v)=R*U*D*P");
                ra["combine_mode"] = util::json_string("product");
                ra["admission_threshold"] = util::json_number(kRuntimeAttentionAdmission);
                ra["admitted"] = util::json_bool(runtime_attention_admitted);
                ra["tie_break"] = util::json_string("score_desc,D_desc,R_desc,node_id_asc");
                ra["unresolvedness_model"] = util::json_string("continuous:clamp(max(0.15,1-confidence))");
                ra["relevance_cold_start"] = util::json_bool(relevance_cold_start);

                util::JsonObject factors;
                factors["R_relevance"] = util::json_number(focus_pick.R);
                factors["U_unresolvedness"] = util::json_number(focus_pick.U);
                factors["D_dependency_pressure"] = util::json_number(focus_pick.D);
                factors["P_priority"] = util::json_number(focus_pick.P);
                factors["priority_route"] = util::json_number(P_route);
                factors["priority_query"] = util::json_number(P_query);
                ra["factors"] = util::json_object(std::move(factors));

                util::JsonObject resolver;
                resolver["formula"] = util::json_string(
                    "S_mu(v)=static_weight*pattern_strength*fold_weight*phase_driver_weight*confidence_score*health_score*relevance_score");
                resolver["candidate"] = util::json_string(
                    miss_route.micronaut_id.empty() ? std::string("Librarian") : miss_route.micronaut_id);
                resolver["delegate_policy"] = util::json_string(delegate_track);
                resolver["score_policy"] = util::json_string(io::ROUTE_SCORE_POLICY);
                resolver["route_score"] = util::json_number((double)miss_route.score);
                ra["resolver_selection"] = util::json_object(std::move(resolver));

                util::JsonArray candidates;
                for (const auto& c : focus_candidates) {
                    util::JsonObject row;
                    row["node"] = util::json_string(c.id);
                    row["kind"] = util::json_string(c.kind);
                    row["slot"] = util::json_string(c.slot);
                    row["R"] = util::json_number(c.R);
                    row["U"] = util::json_number(c.U);
                    row["D"] = util::json_number(c.D);
                    row["P"] = util::json_number(c.P);
                    row["score"] = util::json_number(c.score);
                    candidates.push_back(util::json_object(std::move(row)));
                }
                ra["candidates"] = util::json_array(std::move(candidates));
                obj["runtime_attention"] = util::json_object(std::move(ra));
            }

            {
                util::JsonObject fm;
                fm["attempted"] = util::json_bool(factory_mutation.attempted);
                fm["applied"] = util::json_bool(factory_mutation.applied);
                fm["pending_advisor_verify"] = util::json_bool(factory_mutation.pending_advisor_verify);
                fm["micronaut"] = util::json_string(factory_mutation.micronaut_id);
                fm["result"] = util::json_string(factory_mutation.result);
                util::JsonArray caps;
                for (const auto& c : factory_mutation.added_capabilities)
                    caps.push_back(util::json_string(c));
                fm["added_capabilities"] = util::json_array(std::move(caps));
                util::JsonArray govs;
                for (const auto& g : factory_mutation.added_governs)
                    govs.push_back(util::json_string(g));
                fm["added_governs"] = util::json_array(std::move(govs));
                obj["factory_mutation"] = util::json_object(std::move(fm));
            }

            util::JsonObject chen_obj;
            chen_obj["classification"] = util::json_string("MISS");
            chen_obj["subclassification"] = util::json_string("INSUFFICIENT_CONTEXT");
            chen_obj["transition"] = util::json_string("Acquire/Unfold");
            chen_obj["topology_mutation"] = util::json_bool(false);
            chen_obj["confidence_penalty_applied"] = util::json_bool(false);
            chen_obj["note"] = util::json_string("MISS != WRONG");
            obj["chen"] = util::json_object(std::move(chen_obj));

            // Promote miss → active web_research dispatch
            obj["ok"]       = util::json_bool(true);
            obj["source"]   = util::json_string("web_research");
            obj["status"]   = util::json_string("web_research_dispatched");
            const std::string dispatch_text =
                "I don't have a confirmed local fact for \"" + qtrim +
                "\" yet. Routed to WEB-RESEARCH-µ for evidence retrieval.";
            obj["text"] = util::json_string(dispatch_text);
            obj["response"] = util::json_string(dispatch_text);

            util::JsonObject miss_log;
            miss_log["query"] = util::json_string(qtrim);
            miss_log["topic_anchor"] = util::json_string(topic_anchor);
            miss_log["route_candidate"] = util::json_string(
                miss_route.micronaut_id.empty() ? delegate_track : miss_route.micronaut_id);
            miss_log["delegate_track"] = util::json_string(delegate_track);
            miss_log["selected_task_op"] = util::json_string(miss_route.selected_task_op);
            miss_log["resolver_policy"] = util::json_string(miss_route.resolver_policy);
            miss_log["required_primitive"] = util::json_string(miss_route.required_primitive);
            miss_log["policy_filter_applied"] = util::json_bool(miss_route.policy_filter_applied);
            miss_log["capability_miss"] = util::json_bool(miss_route.capability_miss);
            miss_log["dispatch_block_reason"] = util::json_string(miss_route.dispatch_block_reason);
            miss_log["selected_capable"] = util::json_bool(miss_route.selected_capable);
            miss_log["policy_satisfied"] = util::json_bool(miss_route.policy_satisfied);
            miss_log["xcfe_authorized"] = util::json_bool(miss_route.xcfe_authorized);
            miss_log["dispatch_allowed"] = util::json_bool(miss_route.dispatch_allowed);
            miss_log["score_policy"] = util::json_string(io::ROUTE_SCORE_POLICY);
            miss_log["fold_weight"] = util::json_number((double)miss_route.fold_weight);
            miss_log["phase_driver_weight"] = util::json_number((double)miss_route.phase_driver_weight);
            miss_log["pattern_strength"] = util::json_number((double)miss_route.pattern_strength);
            miss_log["static_weight"] = util::json_number((double)miss_route.static_weight);
            miss_log["confidence_score"] = util::json_number((double)miss_route.confidence_score);
            miss_log["health_score"] = util::json_number((double)miss_route.health_score);
            miss_log["relevance_score"] = util::json_number((double)miss_route.relevance_score);
            miss_log["bow_score"] = util::json_number((double)miss_route.bow_score);
            miss_log["distributional_score"] = util::json_number((double)miss_route.distributional_score);
            miss_log["lexical_semantic_score"] = util::json_number((double)miss_route.lexical_semantic_score);
            miss_log["nsm_prime_score"] = util::json_number((double)miss_route.nsm_prime_score);
            miss_log["function_word_score"] = util::json_number((double)miss_route.function_word_score);
            miss_log["content_word_ratio"] = util::json_number((double)miss_route.content_word_ratio);
            miss_log["function_word_ratio"] = util::json_number((double)miss_route.function_word_ratio);
            miss_log["content_function_binding_ratio"] = util::json_number((double)miss_route.content_function_binding_ratio);
            miss_log["word_sense_score"] = util::json_number((double)miss_route.word_sense_score);
            miss_log["compositional_phrase_score"] = util::json_number((double)miss_route.compositional_phrase_score);
            miss_log["coarse_ngram_score"] = util::json_number((double)miss_route.coarse_ngram_score);
            miss_log["tool_gram_score"] = util::json_number((double)miss_route.tool_gram_score);
            miss_log["miss_dispatch_score"] = util::json_number((double)miss_route.miss_dispatch_score);
            miss_log["semantic_score"] = util::json_number((double)miss_route.semantic_score);
            miss_log["runtime_attention_formula"] = util::json_string("A(v)=R*U*D*P");
            miss_log["runtime_attention_score"] = util::json_number(runtime_attention_score);
            miss_log["runtime_attention_admitted"] = util::json_bool(runtime_attention_admitted);
            miss_log["runtime_attention_threshold"] = util::json_number(kRuntimeAttentionAdmission);
            miss_log["relevance_cold_start"] = util::json_bool(relevance_cold_start);
            miss_log["focus_node"] = util::json_string(focus_pick.id);
            miss_log["focus_kind"] = util::json_string(focus_pick.kind);
            miss_log["focus_slot"] = util::json_string(focus_pick.slot);
            miss_log["R_relevance"] = util::json_number(focus_pick.R);
            miss_log["U_unresolvedness"] = util::json_number(focus_pick.U);
            miss_log["D_dependency_pressure"] = util::json_number(focus_pick.D);
            miss_log["P_priority"] = util::json_number(focus_pick.P);
            {
                util::JsonArray req_caps;
                for (const auto& cap : miss_route.required_capabilities)
                    req_caps.push_back(util::json_string(cap));
                miss_log["required_capabilities"] = util::json_array(std::move(req_caps));
            }
            jrom.log("fact_miss", util::stringify(util::json_object(std::move(miss_log))));

            util::JsonObject fallback_log;
            fallback_log["delegate_track"] = util::json_string(delegate_track);
            fallback_log["retrieval_track"] = util::json_string("Librarian");
            fallback_log["query"] = util::json_string(qtrim);
            fallback_log["topic_anchor"] = util::json_string(topic_anchor);
            fallback_log["variant_count"] = util::json_number(phrase.empty() ? 1.0 : 4.0);
            jrom.log("research_fallback_plan", util::stringify(util::json_object(std::move(fallback_log))));

            active_micronaut = delegate_track;

            {
                util::JsonObject gap_log;
                gap_log["track"]  = util::json_string(active_micronaut);
                gap_log["reason"] = util::json_string("fact_miss");
                jrom.log("knowledge_gap",
                         util::stringify(util::json_object(std::move(gap_log))));
            }
            {
                util::JsonObject row;
                row["classification"] = util::json_string("MISS");
                row["subclassification"] = util::json_string("INSUFFICIENT_CONTEXT");
                row["transition"] = util::json_string("Acquire/Unfold");
                row["research_pressure"] = util::json_number(research_pressure);
                row["runtime_attention_score"] = util::json_number(runtime_attention_score);
                row["runtime_attention_admitted"] = util::json_bool(runtime_attention_admitted);
                row["miss_kind"] = util::json_string(miss_kind);
                jrom.log("chen_resolution", util::stringify(util::json_object(std::move(row))));
            }
            {
                util::JsonObject row;
                row["query"] = util::json_string(qtrim);
                row["topic_anchor"] = util::json_string(topic_anchor);
                row["track"] = util::json_string(active_micronaut);
                row["miss_kind"] = util::json_string(miss_kind);
                row["research_pressure"] = util::json_number(research_pressure);
                row["missing_count"] = util::json_number((double)missing_context.size());
                row["required_fold_count"] = util::json_number((double)required_folds.size());
                jrom.log("miss_deficit_vector", util::stringify(util::json_object(std::move(row))));
            }
            {
                util::JsonObject row;
                row["track"] = util::json_string(delegate_track);
                row["attempted"] = util::json_bool(factory_mutation.attempted);
                row["applied"] = util::json_bool(factory_mutation.applied);
                row["result"] = util::json_string(factory_mutation.result);
                row["added_capability_count"] = util::json_number(
                    (double)factory_mutation.added_capabilities.size());
                row["added_govern_count"] = util::json_number(
                    (double)factory_mutation.added_governs.size());
                jrom.log("factory_mutation", util::stringify(util::json_object(std::move(row))));
            }
            {
                util::JsonObject wr_log, rc_log, ev_log;
                wr_log["query"]     = util::json_string(qtrim);
                wr_log["delegate"]  = util::json_string(delegate_track);
                wr_log["retrieval"] = util::json_string("Librarian");
                jrom.log("web_research",
                         util::stringify(util::json_object(std::move(wr_log))));
                rc_log["query"]     = util::json_string(qtrim);
                rc_log["method"]    = util::json_string("key_query");
                rc_log["trigger_on"] = util::json_string("web_research_commit");
                jrom.log("replay_candidate",
                         util::stringify(util::json_object(std::move(rc_log))));
                ev_log["track"]      = util::json_string(delegate_track);
                ev_log["capability"] = util::json_string(
                    "fact_retrieval::" + (phrase.empty() ? qtrim.substr(0, 32) : phrase));
                ev_log["trigger_on"] = util::json_string("web_research_commit");
                jrom.log("evolve_candidate",
                         util::stringify(util::json_object(std::move(ev_log))));
            }
            {
                util::JsonObject msl_log;
                msl_log["track"]            = util::json_string(active_micronaut);
                msl_log["rule"]             = util::json_string("MISS_not_WRONG");
                msl_log["confidence_delta"] = util::json_number(0.0);
                msl_log["action"]           = util::json_string("expand_unfold");
                jrom.log("miss_resolution_learning",
                         util::stringify(util::json_object(std::move(msl_log))));
            }
        }

        auto out = util::json_object(std::move(obj));
        const auto out_status = out->get_string("status").value_or("");
        const bool research_dispatch = out_status == "web_research_dispatched";
        ensure_governance_metadata(
            out,
            active_micronaut,
            research_dispatch,
            research_dispatch ? "knowledge_gap_research_dispatch" : "local_resolution_pending",
            research_dispatch ? "Ch'en.miss_dispatch" : "Pop.fallback_projection");
        attach_topic_confidence(out, tc);
        attach_micronaut_confidence(out, mc);
        update_last_topic(tc);
        update_last_micronaut(active_micronaut);
        log_micronaut_feedback(mc);
        return finalize_key_query_event(
            out,
            research_dispatch ? "plan" : "model_response",
            active_micronaut,
            research_dispatch ? "Ch'en" : "Pop");
    });

    // list_keys: enumerate registered agent keys
    rpc_server.register_method("list_keys", [&](const util::JsonPtr&) {
        auto keys = factory.list_keys();
        util::JsonArray arr;
        for (const auto& k : keys) arr.push_back(util::json_string(k));
        return util::json_array(std::move(arr));
    });

    // key_encode: deterministic key -> binary keyspace identifier
    rpc_server.register_method("key_encode", [&](const util::JsonPtr& p) {
        auto key = p ? p->get_string("key") : std::nullopt;
        if (!key || trim_copy(*key).empty()) {
            util::JsonObject err;
            err["ok"] = util::json_bool(false);
            err["error"] = util::json_string("requires non-empty 'key' field");
            return util::json_object(std::move(err));
        }

        const std::string key_trimmed = trim_copy(*key);
        const std::string key_norm = lower_copy(key_trimmed);
        const std::string key_id = factory.key_id_for(key_trimmed);
        const auto candidates = factory.keys_for_id(key_id);

        bool exact_exists = false;
        for (const auto& candidate : candidates) {
            if (lower_copy(candidate) == key_norm) {
                exact_exists = true;
                break;
            }
        }

        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["key"] = util::json_string(key_trimmed);
        obj["normalized_key"] = util::json_string(key_norm);
        obj["key_id"] = util::json_string(key_id);
        obj["key_id_hex"] = util::json_string("0x" + key_id);
        obj["lookup_hash_query"] = util::json_string("hash:" + key_id);
        obj["exists"] = util::json_bool(exact_exists);
        obj["candidate_count"] = util::json_number(static_cast<double>(candidates.size()));
        if (exact_exists) {
            obj["hits"] = util::json_number(static_cast<double>(factory.key_hits(key_trimmed)));
            obj["confidence"] = util::json_number(static_cast<double>(factory.key_confidence(key_trimmed)));
        }
        obj["binary_contract"] = util::json_string("keyed_then_serialized");
        return util::json_object(std::move(obj));
    });

    // key_decode: keyspace identifier -> known keys
    rpc_server.register_method("key_decode", [&](const util::JsonPtr& p) {
        auto key_id = p ? p->get_string("key_id") : std::nullopt;
        if (!key_id || trim_copy(*key_id).empty()) {
            util::JsonObject err;
            err["ok"] = util::json_bool(false);
            err["error"] = util::json_string("requires non-empty 'key_id' field");
            return util::json_object(std::move(err));
        }

        const bool include_response = p ? p->get_bool("include_response").value_or(false) : false;
        size_t limit = 12;
        if (auto n = p ? p->get_number("limit") : std::nullopt; n.has_value()) {
            if (*n > 0.0) limit = static_cast<size_t>(std::min<double>(*n, 256.0));
        }

        auto keys = factory.keys_for_id(trim_copy(*key_id));
        util::JsonArray resolved;
        size_t emitted = 0;
        for (const auto& k : keys) {
            if (emitted >= limit) break;
            util::JsonObject row;
            row["key"] = util::json_string(k);
            row["key_id"] = util::json_string(factory.key_id_for(k));
            row["hits"] = util::json_number(static_cast<double>(factory.key_hits(k)));
            row["confidence"] = util::json_number(static_cast<double>(factory.key_confidence(k)));
            if (include_response) {
                if (auto response = factory.peek_key(k)) {
                    row["response"] = util::json_string(*response);
                }
            }
            resolved.push_back(util::json_object(std::move(row)));
            ++emitted;
        }

        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["input_key_id"] = util::json_string(trim_copy(*key_id));
        obj["status"] = util::json_string(resolved.empty() ? "key_id_miss" : "key_id_resolved");
        obj["resolved_count"] = util::json_number(static_cast<double>(resolved.size()));
        obj["resolved_keys"] = util::json_array(std::move(resolved));
        return util::json_object(std::move(obj));
    });

    // keyspace_status: key store persistence contract and artifact presence
    rpc_server.register_method("keyspace_status", [&](const util::JsonPtr&) {
        util::JsonObject obj;
        obj["ok"] = util::json_bool(true);
        obj["key_count"] = util::json_number(static_cast<double>(factory.list_keys().size()));
        obj["factory_keys_json_path"] = util::json_string(factory_state_json_path);
        obj["factory_keys_bin_path"] = util::json_string(factory_state_bin_path);

        std::error_code ec_json, ec_bin;
        const bool json_exists = fs::exists(factory_state_json_path, ec_json);
        const bool bin_exists = fs::exists(factory_state_bin_path, ec_bin);
        obj["factory_keys_json_exists"] = util::json_bool(json_exists);
        obj["factory_keys_bin_exists"] = util::json_bool(bin_exists);
        obj["factory_keys_json_access_error"] = util::json_string(ec_json ? ec_json.message() : std::string());
        obj["factory_keys_bin_access_error"] = util::json_string(ec_bin ? ec_bin.message() : std::string());
        obj["binary_encoding_note"] = util::json_string("factory_keys.bin stores keyed records serialized to bytes; tokenization/keying occurs before binary serialization.");
        return util::json_object(std::move(obj));
    });

    rpc_server.register_method("micronaut_dispatch", [&](const util::JsonPtr& p) {
        auto query = p ? p->get_string("query") : std::nullopt;
        if (!query) return util::json_null();
        io::RouteRequest req;
        req.query = *query;
        req.phase = kuhul::Phase::Wo;
        req.payload = p;
        req.lane = "key_query_wo";
        auto result = router.route(req);
        return result.response ? result.response : util::json_null();
    });

    // Runtime projection of micronaut route contracts into key-grammar rows.
    // This mirrors tools/grammar/mini_transpilers.py micronaut-transpile without
    // writing artifacts or mutating registry state.
    rpc_server.register_method("micronaut_transpile", [&](const util::JsonPtr& p) {
        util::JsonObject out;
        out["@kind"] = util::json_string("khanary.micronaut_transpile.v1");
        out["authority_boundary"] = util::json_string(
            "route_contract_declares_callable_surface_not_fact");
        if (p && p->is_object()) {
            if (auto id = p->get_string("id")) {
                const auto* entry = mu_reg.get(*id);
                if (!entry) return util::json_null();
                out["rows"] = util::json_array({micronaut::MicronautTranspiler::compile_entry(*entry)});
                out["count"] = util::json_number(1.0);
                return util::json_object(std::move(out));
            }
        }
        const bool include_inactive = p && p->get_bool("include_inactive").value_or(false);
        auto rows = micronaut::MicronautTranspiler::compile_registry(
            mu_reg, micronaut::MicronautTranspileOptions{include_inactive});
        out["count"] = util::json_number(static_cast<double>(rows.size()));
        out["rows"] = util::json_array(std::move(rows));
        return util::json_object(std::move(out));
    });

    // JROM semantic-envelope audit lane. JROM remains the tape/provenance
    // journal: it records routed observations and proposals; it does not grant
    // semantic authority, execute SQL, or commit state.
    rpc_server.register_method("journal_semantic_envelope", [&](const util::JsonPtr& p) {
        if (!p || !p->is_object()) return util::json_null();
        static const std::unordered_set<std::string> kEnvelopeTypes = {
            "think", "reason", "plan", "task", "commands", "sql",
            "grammar", "route", "decision", "memory", "verify", "commit"
        };
        static const std::unordered_set<std::string> kEpistemicTypes = {
            "observation", "preference", "plan", "generated_output",
            "verified_fact", "failed_retrieval", "capability_metric",
            "execution_trace"
        };
        const std::string type = p->get_string("type").value_or("unknown");
        const std::string safe_type = kEnvelopeTypes.count(type) ? type : "unknown";
        const std::string epistemic_type = p->get_string("epistemic_type").value_or("generated_output");
        const std::string safe_epistemic_type =
            kEpistemicTypes.count(epistemic_type) ? epistemic_type : "generated_output";
        const std::string disposition = p->get_string("disposition").value_or("proposal_only");
        const std::string status = p->get_string("status").value_or("observed");
        const std::string content_json = p->get("content")
            ? util::stringify(*p->get("content")) : "null";
        const std::string result_json = p->get("result")
            ? util::stringify(*p->get("result")) : "null";

        util::JsonObject event;
        event["@kind"] = util::json_string("khanary.semantic_envelope.v1");
        event["envelope_type"] = util::json_string(safe_type);
        event["epistemic_type"] = util::json_string(safe_epistemic_type);
        event["disposition"] = util::json_string(disposition);
        event["status"] = util::json_string(status);
        event["content_json"] = util::json_string(content_json);
        event["result_json"] = util::json_string(result_json);
        event["fact_authority"] = util::json_bool(false);
        event["observation_authority"] = util::json_bool(false);
        event["authority_boundary"] = util::json_string(
            "jrom_records_routed_observation_not_semantic_authority");
        const auto seq = jrom.log("semantic_envelope",
                                  util::stringify(util::json_object(std::move(event))),
                                  safe_epistemic_type);

        util::JsonObject out;
        out["ok"] = util::json_bool(true);
        out["jrom_seq"] = util::json_number(static_cast<double>(seq));
        out["event_type"] = util::json_string("semantic_envelope");
        return util::json_object(std::move(out));
    });

    // Hyperbolic key-grammar event journal. This records a future-directed
    // candidate continuation address (origin/direction/rapidity/interval) as
    // replay evidence. It is never a verified fact or topology commit.
    rpc_server.register_method("journal_hyperbolic_event", [&](const util::JsonPtr& p) {
        util::JsonObject out;
        out["@kind"] = util::json_string("khanary.hyperbolic_event.response.v1");
        if (!p || !p->is_object()) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("missing hyperbolic event spec");
            return util::json_object(std::move(out));
        }

        const util::JsonValue* spec = p.get();
        if (const auto* nested = p->get("hyperbolic_event"))
            if (nested->is_object()) spec = nested;
        if (const auto* record = p->get("event_record"))
            if (record->is_object()) spec = record;
        const util::JsonValue* metrics = spec->get("hyperbolic_metrics");
        const util::JsonValue* bounds = spec->get("bounds");

        const auto get_string = [&](const char* key) -> std::optional<std::string> {
            return spec->get_string(key);
        };
        const auto get_number = [&](const char* key) -> std::optional<double> {
            return spec->get_number(key);
        };

        const auto key = get_string("key").value_or("");
        const auto origin = get_string("origin").value_or("");
        if (key.empty() || origin.empty()) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("key and origin are required");
            return util::json_object(std::move(out));
        }
        const auto semantic_authority = spec->get_bool("semantic_authority");
        const auto bounds_authority = bounds ? bounds->get_bool("semantic_authority") : std::optional<bool>{};
        if (!semantic_authority || *semantic_authority ||
            spec->get_bool("fact_authority").value_or(false) ||
            (bounds && (!bounds_authority || *bounds_authority))) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("hyperbolic events cannot claim semantic or fact authority");
            return util::json_object(std::move(out));
        }

        double rapidity = get_number("rapidity").value_or(0.0);
        if (metrics) {
            if (auto nested_rapidity = metrics->get_number("rapidity_rho"))
                rapidity = *nested_rapidity;
        }
        if (!std::isfinite(rapidity) || rapidity < 0.0) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("rapidity must be a nonnegative finite number");
            return util::json_object(std::move(out));
        }

        util::JsonArray direction_json;
        bool has_direction = false;
        const util::JsonValue* direction = spec->get("direction");
        if (!direction && metrics) direction = metrics->get("direction_vector");
        if (direction) {
            if (!direction->is_array() || direction->as_array().size() != 3) {
                out["ok"] = util::json_bool(false);
                out["error"] = util::json_string("direction must be [x,y,z]");
                return util::json_object(std::move(out));
            }
            double norm_sq = 0.0;
            for (const auto& component : direction->as_array()) {
                if (!component || !component->is_number() || !std::isfinite(component->as_number())) {
                    out["ok"] = util::json_bool(false);
                    out["error"] = util::json_string("direction components must be finite numbers");
                    return util::json_object(std::move(out));
                }
                const double v = component->as_number();
                norm_sq += v * v;
                direction_json.push_back(util::json_number(v));
            }
            if (norm_sq <= 1e-12) {
                out["ok"] = util::json_bool(false);
                out["error"] = util::json_string("direction must be nonzero");
                return util::json_object(std::move(out));
            }
            has_direction = true;
        }

        static const std::unordered_set<std::string> kIntervalClasses = {
            "timelike_candidate", "spacelike_candidate", "lightlike_boundary", "unclassified"
        };
        std::string interval_class = get_string("interval_class").value_or("unclassified");
        if (bounds) {
            if (auto nested_interval = bounds->get_string("interval_class"))
                interval_class = *nested_interval;
        }
        if (!kIntervalClasses.count(interval_class)) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("invalid interval_class");
            return util::json_object(std::move(out));
        }
        const auto expected_interval_class = [](double interval) -> std::string {
            if (interval < -1e-9) return "timelike_candidate";
            if (interval > 1e-9) return "spacelike_candidate";
            return "lightlike_boundary";
        };

        std::string epistemic_state_raw = get_string("epistemic_state").value_or("UNKNOWN");
        if (bounds) {
            if (auto nested_state = bounds->get_string("epistemic_state"))
                epistemic_state_raw = *nested_state;
        }
        const std::string epistemic_state = normalize_epistemic_state(epistemic_state_raw);
        std::optional<double> minkowski_interval = get_number("minkowski_interval");
        if (!minkowski_interval && metrics)
            minkowski_interval = metrics->get_number("minkowski_interval");
        if (minkowski_interval && !std::isfinite(*minkowski_interval)) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("minkowski_interval must be finite");
            return util::json_object(std::move(out));
        }
        if (minkowski_interval && interval_class != "unclassified" &&
            expected_interval_class(*minkowski_interval) != interval_class) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("interval_class does not match minkowski_interval sign");
            return util::json_object(std::move(out));
        }

        util::JsonObject event;
        event["@kind"] = util::json_string("key_grammar.hyperbolic_event.v1");
        event["@type"] = util::json_string("key_grammar.hyperbolic_event.v1");
        event["key"] = util::json_string(key);
        event["origin"] = util::json_string(origin);
        event["track"] = util::json_string(get_string("track").value_or(""));
        event["phase"] = util::json_string(get_string("phase").value_or(""));
        event["event_role"] = util::json_string(get_string("event_role").value_or("candidate_continuation"));
        event["rapidity"] = util::json_number(rapidity);
        if (has_direction)
            event["direction"] = util::json_array(std::move(direction_json));
        if (minkowski_interval)
            event["minkowski_interval"] = util::json_number(*minkowski_interval);
        event["interval_class"] = util::json_string(interval_class);
        event["epistemic_state"] = util::json_string(epistemic_state);
        event["semantic_authority"] = util::json_bool(false);
        event["fact_authority"] = util::json_bool(false);
        event["authority_boundary"] = util::json_string(
            "hyperbolic_event_is_candidate_not_fact");

        const auto seq = jrom.log("hyperbolic_event",
                                  util::stringify(util::json_object(std::move(event))),
                                  "observation");
        out["ok"] = util::json_bool(true);
        out["jrom_seq"] = util::json_number(static_cast<double>(seq));
        out["event_type"] = util::json_string("hyperbolic_event");
        out["epistemic_type"] = util::json_string("observation");
        out["epistemic_state"] = util::json_string(epistemic_state);
        return util::json_object(std::move(out));
    });

    // Unified tensor registry: logical tensor metadata above TensorBuffer/DMLOps.
    // This is a KXC/executor surface; lifecycle labels are execution states only.
    auto phase_to_tensor_lifecycle = [](const std::string& phase) -> std::string {
        if (phase == "Pop" || phase == "POP") return "POP";
        if (phase == "Ch'en" || phase == "CHEN") return "CHEN";
        if (phase == "Xul" || phase == "XUL") return "XUL";
        if (phase == "KNU") return "KNU";
        // Wo/Yax/Sek are all executable materialization states here. This is
        // not a semantic phase collapse; it is the UnifiedTensor lifecycle label.
        return "SEK";
    };

    auto parse_tensor_shape_request = [](const util::JsonValue& value,
                                         std::vector<uint32_t>* out) -> bool {
        const auto* shape_node = value.get("shape");
        if (!shape_node || !shape_node->is_array() || shape_node->as_array().empty())
            return false;
        std::vector<uint32_t> parsed;
        parsed.reserve(shape_node->as_array().size());
        for (const auto& dim : shape_node->as_array()) {
            if (!dim || !dim->is_number()) return false;
            const double raw = dim->as_number();
            if (!std::isfinite(raw) || raw <= 0.0 || raw > 4294967295.0) return false;
            parsed.push_back(static_cast<uint32_t>(raw));
        }
        *out = std::move(parsed);
        return true;
    };

    rpc_server.register_method("tensor_register", [&](const util::JsonPtr& p) {
        util::JsonObject out;
        out["@kind"] = util::json_string("khanary.unified_tensor.register.v1");
        if (!p || !p->is_object()) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("missing tensor spec");
            return util::json_object(std::move(out));
        }
        auto tensor = gpu::UnifiedTensor::from_json(*p);
        if (!tensor) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("invalid tensor spec");
            return util::json_object(std::move(out));
        }
        const bool overwrite = p->get_bool("overwrite").value_or(false);
        const std::string id = tensor->id();
        if (!unified_tensors.register_tensor(std::move(*tensor), overwrite)) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("tensor exists or invalid");
            out["id"] = util::json_string(id);
            return util::json_object(std::move(out));
        }
        auto stored = unified_tensors.get(id);
        stored->set_cso_id(stored->compute_cso_id());
        out["ok"] = util::json_bool(true);
        out["tensor"] = stored->to_json();
        jrom.log("unified_tensor_register", util::stringify(*stored->to_json()));
        return util::json_object(std::move(out));
    });

    rpc_server.register_method("tensor_slot_materialize", [&](const util::JsonPtr& p) {
        util::JsonObject out;
        out["@kind"] = util::json_string("khanary.unified_tensor.slot_materialize.v1");
        if (!p || !p->is_object()) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("missing tensor slot materialization spec");
            return util::json_object(std::move(out));
        }

        const auto requested_slot_id = p->get_string("slot_id");
        const auto requested_micronaut_id = p->get_string("micronaut_id");
        if (!requested_slot_id && !requested_micronaut_id) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("slot_id or micronaut_id is required");
            return util::json_object(std::move(out));
        }

        std::vector<uint32_t> shape;
        if (!parse_tensor_shape_request(*p, &shape)) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("shape is required for slot materialization");
            return util::json_object(std::move(out));
        }

        const fs::path slot_registry_path =
            fs::path(cfg.data_dir) / "world" / "micronaut_tensor_slots.json";
        std::string registry_text;
        std::string registry_err;
        if (!read_text_file(slot_registry_path, &registry_text, &registry_err)) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("tensor slot registry unavailable: " + registry_err);
            return util::json_object(std::move(out));
        }

        util::JsonPtr slot_registry;
        try {
            slot_registry = util::parse(registry_text);
        } catch (const std::exception& exc) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string(std::string("tensor slot registry parse failed: ") + exc.what());
            return util::json_object(std::move(out));
        }

        const util::JsonValue* matched_slot = nullptr;
        if (slot_registry && slot_registry->is_object()) {
            if (const auto* slots = slot_registry->get("slots");
                slots && slots->is_array()) {
                for (const auto& slot_ptr : slots->as_array()) {
                    if (!slot_ptr || !slot_ptr->is_object()) continue;
                    const auto sid = slot_ptr->get_string("slot_id");
                    const auto mid = slot_ptr->get_string("micronaut_id");
                    const bool slot_match =
                        requested_slot_id && sid && *requested_slot_id == *sid;
                    const bool micronaut_match =
                        requested_micronaut_id && mid && *requested_micronaut_id == *mid;
                    if (slot_match || micronaut_match) {
                        matched_slot = slot_ptr.get();
                        break;
                    }
                }
            }
        }

        if (!matched_slot) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("tensor slot not found");
            if (requested_slot_id) out["slot_id"] = util::json_string(*requested_slot_id);
            if (requested_micronaut_id) out["micronaut_id"] = util::json_string(*requested_micronaut_id);
            return util::json_object(std::move(out));
        }

        const std::string slot_id = matched_slot->get_string("slot_id").value_or("");
        const std::string micronaut_id = matched_slot->get_string("micronaut_id").value_or("");
        const std::string phase = matched_slot->get_string("phase").value_or("Sek");
        const std::string materialization = matched_slot->get_string("materialization").value_or("unknown");
        if (slot_id.empty() || micronaut_id.empty()) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("matched tensor slot is malformed");
            return util::json_object(std::move(out));
        }

        util::JsonArray shape_json;
        for (uint32_t dim : shape)
            shape_json.push_back(util::json_number(static_cast<double>(dim)));

        const std::string tensor_id =
            p->get_string("id").value_or("tensor:" + slot_id);
        util::JsonObject spec;
        spec["id"] = util::json_string(tensor_id);
        spec["slot_id"] = util::json_string(slot_id);
        spec["micronaut_id"] = util::json_string(micronaut_id);
        spec["shape"] = util::json_array(std::move(shape_json));
        spec["dtype"] = util::json_string(p->get_string("dtype").value_or("f32"));
        spec["layout"] = util::json_string(p->get_string("layout").value_or("row_major"));
        spec["device"] = util::json_string(p->get_string("device").value_or("cpu"));
        spec["backend"] = util::json_string(p->get_string("backend").value_or("host"));
        spec["lifecycle"] = util::json_string(
            p->get_string("lifecycle").value_or(phase_to_tensor_lifecycle(phase)));
        if (const auto* data = p->get("data")) {
            if (!data->is_array()) {
                out["ok"] = util::json_bool(false);
                out["error"] = util::json_string("data must be an array when provided");
                return util::json_object(std::move(out));
            }
            util::JsonArray copied_data;
            copied_data.reserve(data->as_array().size());
            for (const auto& item : data->as_array()) {
                if (!item || !item->is_number() || !std::isfinite(item->as_number())) {
                    out["ok"] = util::json_bool(false);
                    out["error"] = util::json_string("data contains non-finite or non-number entry");
                    return util::json_object(std::move(out));
                }
                copied_data.push_back(util::json_number(item->as_number()));
            }
            spec["data"] = util::json_array(std::move(copied_data));
        }

        auto tensor = gpu::UnifiedTensor::from_json(*util::json_object(std::move(spec)));
        if (!tensor) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("slot-backed tensor spec is invalid");
            return util::json_object(std::move(out));
        }
        const bool overwrite = p->get_bool("overwrite").value_or(false);
        if (!unified_tensors.register_tensor(std::move(*tensor), overwrite)) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("slot-backed tensor exists or is invalid");
            out["id"] = util::json_string(tensor_id);
            return util::json_object(std::move(out));
        }

        auto stored = unified_tensors.get(tensor_id);
        stored->set_cso_id(stored->compute_cso_id());

        util::JsonObject slot_json;
        slot_json["slot_id"] = util::json_string(slot_id);
        slot_json["micronaut_id"] = util::json_string(micronaut_id);
        slot_json["phase"] = util::json_string(phase);
        slot_json["materialization"] = util::json_string(materialization);
        slot_json["authority_boundary"] = util::json_string(
            "slot_materialization_binds_tensor_runtime_state_not_semantic_truth");

        out["ok"] = util::json_bool(true);
        out["slot"] = util::json_object(slot_json);
        out["tensor"] = stored->to_json();

        util::JsonObject event;
        event["slot_id"] = util::json_string(slot_id);
        event["micronaut_id"] = util::json_string(micronaut_id);
        event["tensor_id"] = util::json_string(tensor_id);
        event["cso_id"] = util::json_string(stored->cso_id());
        event["semantic_authority"] = util::json_bool(false);
        event["authority_boundary"] = util::json_string(
            "unified_tensor_slot_materialize_records_runtime_projection_only");
        jrom.log("unified_tensor_slot_materialize",
                 util::stringify(util::json_object(std::move(event))));
        return util::json_object(std::move(out));
    });

    rpc_server.register_method("tensor_info", [&](const util::JsonPtr& p) {
        const auto id = p ? p->get_string("id") : std::nullopt;
        return id ? unified_tensors.info_json(*id) : util::json_null();
    });

    rpc_server.register_method("tensor_list", [&](const util::JsonPtr&) {
        return unified_tensors.list_json();
    });

    rpc_server.register_method("tensor_stats", [&](const util::JsonPtr&) {
        return unified_tensors.stats_json();
    });

    rpc_server.register_method("tensor_gemm", [&](const util::JsonPtr& p) {
        util::JsonObject out;
        out["@kind"] = util::json_string("khanary.unified_tensor.gemm.v1");
        if (!p || !p->is_object()) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("missing gemm params");
            return util::json_object(std::move(out));
        }
        const auto a_id = p->get_string("A");
        const auto b_id = p->get_string("B");
        const auto c_id = p->get_string("C");
        if (!a_id || !b_id || !c_id) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("A, B, and C ids are required");
            return util::json_object(std::move(out));
        }
        auto A = unified_tensors.get(*a_id);
        auto B = unified_tensors.get(*b_id);
        auto C = unified_tensors.get(*c_id);
        if (!A || !B) {
            out["ok"] = util::json_bool(false);
            out["error"] = util::json_string("A or B tensor not registered");
            return util::json_object(std::move(out));
        }
        if (!C) {
            if (A->shape().size() != 2 || B->shape().size() != 2 ||
                A->shape()[1] != B->shape()[0]) {
                out["ok"] = util::json_bool(false);
                out["error"] = util::json_string("A/B shapes are not GEMM-compatible");
                return util::json_object(std::move(out));
            }
            gpu::UnifiedTensorDesc desc;
            desc.id = *c_id;
            desc.shape = {A->shape()[0], B->shape()[1]};
            desc.dtype = gpu::TensorDType::F32;
            desc.layout = gpu::TensorLayout::RowMajor;
            gpu::UnifiedTensor tensor(std::move(desc));
            if (!unified_tensors.register_tensor(std::move(tensor), true)) {
                out["ok"] = util::json_bool(false);
                out["error"] = util::json_string("failed to register C tensor");
                return util::json_object(std::move(out));
            }
            C = unified_tensors.get(*c_id);
        }
        const float alpha = static_cast<float>(p->get_number("alpha").value_or(1.0));
        const float beta = static_cast<float>(p->get_number("beta").value_or(0.0));
        // Phase 1 keeps RPC GEMM host-deterministic by default. GPU residency is
        // opt-in until the unified registry ↔ DMLOps bridge has a dedicated
        // conformance lane for the target adapter.
        const bool prefer_gpu = p->get_bool("prefer_gpu").value_or(false);
        std::string backend;
        const bool ok = gpu::gemm_f32_unified(
            *A, *B, *C, alpha, beta,
            (prefer_gpu && device.is_valid()) ? &device : nullptr,
            (prefer_gpu && device.is_valid()) ? &dml_ops : nullptr,
            &backend);
        out["ok"] = util::json_bool(ok);
        out["backend"] = util::json_string(backend);
        out["A"] = util::json_string(*a_id);
        out["B"] = util::json_string(*b_id);
        out["C"] = util::json_string(*c_id);
        if (ok) {
            out["tensor"] = C->to_json();
            util::JsonObject event;
            event["A"] = util::json_string(*a_id);
            event["B"] = util::json_string(*b_id);
            event["C"] = util::json_string(*c_id);
            event["backend"] = util::json_string(backend);
            event["cso_id"] = util::json_string(C->cso_id());
            jrom.log("unified_tensor_gemm", util::stringify(util::json_object(std::move(event))));
        } else {
            out["error"] = util::json_string("gemm execution failed");
        }
        return util::json_object(std::move(out));
    });

    const bool gemma_learning_enabled = is_gemma_gguf_path(cfg.gguf_model);
    const std::string gemma_model_name =
        cfg.gguf_model.empty() ? std::string("gemma.gguf") : fs::path(cfg.gguf_model).filename().string();
    const std::string dds_identity_report =
        dds_identity_raw.empty() ? dds_model_tag : dds_identity_raw;
    auto make_model_presence = [&]() {
        util::JsonObject presence;
        const bool identity_ok = !model_identity_checked || model_identity_aligned;
        const bool dual_ready = gguf_ram_loaded && dds_ready && identity_ok;
        presence["gguf_ram_loaded"] = util::json_bool(gguf_ram_loaded);
        presence["dds_gpu_loaded"] = util::json_bool(dds_ready);
        presence["dual_models_present"] = util::json_bool(dual_models_present);
        presence["identity_checked"] = util::json_bool(model_identity_checked);
        presence["identity_aligned"] = util::json_bool(identity_ok);
        presence["dual_ready"] = util::json_bool(dual_ready);
        presence["gguf_model"] = util::json_string(cfg.gguf_model);
        presence["gguf_model_tag"] = util::json_string(gguf_model_tag);
        presence["dds_kson"] = util::json_string(cfg.dds_kson);
        presence["dds_identity"] = util::json_string(dds_identity_report);
        presence["dds_model_tag"] = util::json_string(dds_model_tag);
        presence["identity_note"] = util::json_string(model_identity_note);
        return util::json_object(std::move(presence));
    };

    uint64_t last_infer_end_seq = 0;  // exclusive lower bound for next call's trace window
    rpc_server.register_method("infer", [&](const util::JsonPtr& p) {
        if (!dds_ready || !engine) {
            util::JsonObject err;
            err["ok"]    = util::json_bool(false);
            err["error"] = util::json_string("DDS model not loaded — start with --dds-kson/--dds-manifest/--dds-shards");
            err["model_presence"] = make_model_presence();
            err["dual_inference_required"] = util::json_bool(true);
            return util::json_object(std::move(err));
        }
        if (!gguf_ram_loaded) {
            util::JsonObject err;
            err["ok"]    = util::json_bool(false);
            err["error"] = util::json_string("GGUF base model not loaded in RAM — start with --gguf-model <path>");
            err["model_presence"] = make_model_presence();
            err["dual_inference_required"] = util::json_bool(true);
            return util::json_object(std::move(err));
        }
        if (model_identity_checked && !model_identity_aligned) {
            util::JsonObject err;
            err["ok"]    = util::json_bool(false);
            err["error"] = util::json_string(
                "GGUF/DDS model identity mismatch — load matching conversion artifacts before infer");
            err["model_presence"] = make_model_presence();
            err["dual_inference_required"] = util::json_bool(true);
            return util::json_object(std::move(err));
        }
        auto prompt = p ? p->get_string("prompt") : std::nullopt;
        if (!prompt) {
            util::JsonObject err;
            err["ok"]    = util::json_bool(false);
            err["error"] = util::json_string("missing 'prompt' field");
            err["model_presence"] = make_model_presence();
            return util::json_object(std::move(err));
        }
        const auto prompt_archive = persist_prompt_submission(
            *prompt, prompt_ingest_root, &dictionary_lexicon, &factory, &vm.gram_store());
        {
            util::JsonObject plog;
            plog["source"] = util::json_string("infer");
            plog["prompt_key"] = util::json_string(prompt_archive.prompt_key);
            plog["prompt_len"] = util::json_number(static_cast<double>(prompt->size()));
            plog["prompt_ngrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_ngram_count));
            plog["prompt_unigrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_unigram_count));
            plog["prompt_bigrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_bigram_count));
            plog["prompt_trigrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_trigram_count));
            plog["prompt_quadgrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_quadgram_count));
            plog["prompt_coarse_grams"] = util::json_number(static_cast<double>(prompt_archive.prompt_coarse_gram_count));
            plog["prompt_function_grams"] = util::json_number(static_cast<double>(prompt_archive.prompt_function_gram_count));
            plog["prompt_tool_grams"] = util::json_number(static_cast<double>(prompt_archive.prompt_tool_gram_count));
            plog["prompt_lexicon"] = util::json_number(static_cast<double>(prompt_archive.prompt_lexicon_count));
            plog["prompt_assoc"] = util::json_number(static_cast<double>(prompt_archive.prompt_assoc_count));
            plog["prompt_atoms"] = util::json_number(static_cast<double>(prompt_archive.prompt_atom_count));
            plog["prompt_similarity"] = util::json_number(static_cast<double>(prompt_archive.prompt_similarity_count));
            plog["file_candidates"] = util::json_number(static_cast<double>(prompt_archive.file_candidate_count));
            plog["file_ingested"] = util::json_number(static_cast<double>(prompt_archive.file_ingested_count));
            plog["file_unigrams"] = util::json_number(static_cast<double>(prompt_archive.file_unigram_count));
            plog["file_bigrams"] = util::json_number(static_cast<double>(prompt_archive.file_bigram_count));
            plog["file_trigrams"] = util::json_number(static_cast<double>(prompt_archive.file_trigram_count));
            plog["file_quadgrams"] = util::json_number(static_cast<double>(prompt_archive.file_quadgram_count));
            plog["file_coarse_grams"] = util::json_number(static_cast<double>(prompt_archive.file_coarse_gram_count));
            plog["file_function_grams"] = util::json_number(static_cast<double>(prompt_archive.file_function_gram_count));
            plog["file_tool_grams"] = util::json_number(static_cast<double>(prompt_archive.file_tool_gram_count));
            plog["file_lexicon"] = util::json_number(static_cast<double>(prompt_archive.file_lexicon_count));
            plog["file_atoms"] = util::json_number(static_cast<double>(prompt_archive.file_atom_count));
            plog["file_similarity"] = util::json_number(static_cast<double>(prompt_archive.file_similarity_count));
            plog["pdf_pending"] = util::json_number(static_cast<double>(prompt_archive.file_pdf_pending_count));
            jrom.log("prompt_archive", util::stringify(util::json_object(std::move(plog))));
        }
        const std::string infer_prompt_event_key = record_key_grammar_memory_event(
            "prompt",
            "infer",
            "USER-µ",
            "Pop",
            prompt_archive.prompt_key,
            "",
            *prompt,
            "",
            "UNKNOWN",
            false,
            &factory,
            &vm.gram_store(),
            &jrom);

        auto finalize_infer_event = [&](util::JsonObject obj,
                                        const std::string& response_text,
                                        bool response_ok,
                                        const std::string& phase_hint) -> util::JsonPtr {
            auto out = util::json_object(std::move(obj));
            const std::string epistemic_state =
                (response_ok && !response_text.empty()) ? "UNKNOWN" : "MISS";
            const std::string event_key = record_key_grammar_memory_event(
                "model_response",
                "infer",
                "Mx2LM",
                phase_hint,
                prompt_archive.prompt_key,
                infer_prompt_event_key,
                response_text,
                util::stringify(out),
                epistemic_state,
                false,
                &factory,
                &vm.gram_store(),
                &jrom);
            attach_memory_event_metadata(
                out,
                event_key,
                "model_response",
                infer_prompt_event_key,
                prompt_archive.prompt_key);
            return out;
        };
        // Optional max_tokens override; defaults to 8 for DDS path (full 256 is very slow)
        auto mt_opt = p ? p->get_number("max_tokens") : std::nullopt;
        uint32_t max_tok = mt_opt ? static_cast<uint32_t>(*mt_opt) : 8u;

        auto attach_gemma_learning = [&](util::JsonObject& obj, const std::string& prompt_text, const std::string& response_text) {
            if (!gemma_learning_enabled) return;
            if (response_text.empty()) return;

            auto grams = build_learning_ngrams(response_text, 24);
            if (grams.empty()) return;

            auto make_gram_array = [&](size_t limit) {
                util::JsonArray arr;
                const size_t capped = std::min(limit, grams.size());
                for (size_t i = 0; i < capped; ++i)
                    arr.push_back(util::json_string(grams[i]));
                return arr;
            };

            obj["ngrams"] = util::json_array(make_gram_array(grams.size()));

            util::JsonObject replay;
            replay["method"] = util::json_string("infer");
            replay["query"] = util::json_string(prompt_text);
            replay["track"] = util::json_string("GEMMA-TOOLCALL-µ");
            replay["trigger_on"] = util::json_string("infer_commit");
            replay["ngrams"] = util::json_array(make_gram_array(12));
            obj["replay_candidate"] = util::json_object(std::move(replay));

            util::JsonObject evolve;
            evolve["track"] = util::json_string("GEMMA-TOOLCALL-µ");
            evolve["model"] = util::json_string(gemma_model_name);
            evolve["capability"] = util::json_string("toolcall_response_ngrams");
            evolve["trigger_on"] = util::json_string("infer_commit");
            evolve["ngrams"] = util::json_array(make_gram_array(12));
            obj["evolve_candidate"] = util::json_object(std::move(evolve));

            util::JsonObject replay_log;
            replay_log["query"] = util::json_string(prompt_text);
            replay_log["method"] = util::json_string("infer");
            replay_log["track"] = util::json_string("GEMMA-TOOLCALL-µ");
            replay_log["model"] = util::json_string(gemma_model_name);
            replay_log["ngram_count"] = util::json_number((double)grams.size());
            jrom.log("replay_candidate", util::stringify(util::json_object(std::move(replay_log))));

            util::JsonObject evolve_log;
            evolve_log["track"] = util::json_string("GEMMA-TOOLCALL-µ");
            evolve_log["model"] = util::json_string(gemma_model_name);
            evolve_log["capability"] = util::json_string("toolcall_response_ngrams");
            evolve_log["trigger_on"] = util::json_string("infer_commit");
            evolve_log["ngram_count"] = util::json_number((double)grams.size());
            jrom.log("evolve_candidate", util::stringify(util::json_object(std::move(evolve_log))));
        };

        util::JsonObject attention_obj;
        bool attention_attempted = false;
        auto run_semantic_attention = [&]() {
            if (!engine) return;
            auto candidates = inference::collect_prompt_attention_candidates(
                vm.gram_store(), prompt_archive.prompt_key, 64);
            if (candidates.empty()) return;

            // GAP-040: validate prompt envelope and candidate eligibility before dispatch.
            if (prompt_archive.prompt_key.empty() ||
                prompt_archive.prompt_key.rfind("prompt:", 0) != 0) {
                attention_obj["dispatch_rejected"] = util::json_bool(true);
                attention_obj["reject_reason"] = util::json_string("invalid_prompt_key");
                attention_attempted = true;
                return;
            }
            const bool any_eligible = std::any_of(candidates.begin(), candidates.end(),
                [](const auto& c) { return c.policy_ok && c.capable && c.xcfe_ok; });
            if (!any_eligible) {
                attention_obj["dispatch_rejected"] = util::json_bool(true);
                attention_obj["reject_reason"] = util::json_string("no_eligible_candidates");
                attention_attempted = true;
                return;
            }

            // GAP-042: GPU failure carries a typed status; it is never
            // indistinguishable from a legitimately empty candidate set.
            auto attention = engine->score_prompt_attention(
                vm.gram_store(), candidates, 0.3f, 0.35f);
            attention_attempted = true;
            const auto& scores = attention.scores;

            size_t admitted_count = 0;
            util::JsonArray top;
            const size_t top_count = std::min<size_t>(8, scores.size());
            for (size_t i = 0; i < scores.size(); ++i)
                if (scores[i].admitted) ++admitted_count;
            for (size_t i = 0; i < top_count; ++i) {
                util::JsonObject row;
                row["key"] = util::json_string(scores[i].key);
                row["semantic_score"] = util::json_number(scores[i].semantic_score);
                row["base_score"] = util::json_number(scores[i].base_score);
                row["admitted"] = util::json_bool(scores[i].admitted);
                top.push_back(util::json_object(std::move(row)));
            }

            const auto* boost = engine->lorentz_transition();
            attention_obj["candidate_count"] = util::json_number(static_cast<double>(candidates.size()));
            attention_obj["gpu_ready"] = util::json_bool(engine->semantic_attention_ready());
            attention_obj["gpu_status"] = util::json_string(
                gpu::gram_gpu_status_name(attention.status));
            attention_obj["dispatched"] = util::json_bool(
                attention.ok() && !scores.empty());
            if (!attention.ok())
                attention_obj["gpu_error"] = util::json_string(attention.error);
            attention_obj["admitted_count"] = util::json_number(static_cast<double>(admitted_count));
            attention_obj["lorentz_enabled"] = util::json_bool(boost && boost->enabled);
            attention_obj["top"] = util::json_array(std::move(top));
            attention_obj["semantic_authority"] = util::json_bool(false);
            attention_obj["responsibility"] = util::json_string("projection_only_attention_bias");
            jrom.log("semantic_attention",
                     util::stringify(util::json_object(attention_obj)),
                     "observation");
        };

        util::JsonObject fold_obj;
        bool  fold_attempted  = false;
        float fold_confidence = 1.0f;
        auto run_fold_selection = [&]() {
            if (!fold_dispatch_ready) return;
            if (prompt_archive.prompt_key.empty() ||
                    prompt_archive.prompt_key.rfind("prompt:", 0) != 0) return;

            // Incrementally populate the persistent field with new GramStore entries.
            // Entries already in fold_known_names are skipped to avoid duplicates.
            bool field_grew = false;
            const std::string fld_prefix = prompt_archive.prompt_key + ":";
            for (const auto& key : vm.gram_store().keys()) {
                if (key.rfind(fld_prefix, 0) != 0) continue;
                if (key.find(":vector:") != std::string::npos) continue;
                if (fold_known_names.count(key)) continue;
                const auto gram = vm.gram_store().gramRead(key);
                if (!gram) continue;
                fold_field.add(key, gram->value, "", gram->phase, gram->channel);
                fold_known_names.insert(key);
                field_grew = true;
            }
            if (fold_field.size() == 0) return;

            static micronaut::ElizaOperatorBank s_fold_eliza;
            static micronaut::CompareGrammar s_fold_compare =
                micronaut::CompareGrammar::standard();
            micronaut::NeighborhoodSelector sel(
                fold_field, vm.gram_store(), s_fold_eliza, s_fold_compare);
            auto nhood = sel.select(*prompt, prompt_archive.prompt_key, "", 16);
            if (nhood.empty()) {
                if (field_grew) fold_field.save(fold_field_path);
                return;
            }
            fold_mapping.reset_batch();
            const uint64_t pre_map_version = fold_mapping.candidate_mapping_version;
            auto cands = micronaut::promote_candidates(
                nhood, prompt_archive.prompt_key, fold_mapping, 0);
            std::vector<float> wcr_buf(cands.size());
            for (size_t i = 0; i < cands.size(); ++i)
                wcr_buf[i] = cands[i].wcr_score;
            micronaut::segmented_softmax(wcr_buf.data(), wcr_buf.size());
            micronaut::apply_attention_scores(cands, wcr_buf.data(), wcr_buf.size());
            const uint32_t total = static_cast<uint32_t>(cands.size());
            std::vector<float> retention_buf(total, 1.0f);
            for (const auto& c : cands) {
                const auto* entry = fold_field.get(c.candidate_id);
                if (entry && c.attention_offset < total)
                    retention_buf[c.attention_offset] = entry->weight;
            }
            const auto result = fold_dispatch.dispatch_score_only(cands, total, fold_confidence, retention_buf);
            if (result.ok()) {
                for (auto& c : cands) {
                    if (c.attention_offset < result.scores.size())
                        c.combined = result.scores[c.attention_offset];
                }
            }
            const auto* winner = micronaut::select_best(cands);
            micronaut::feedback_selection(winner, sel);
            bool any_loser_learned = false;
            for (const auto& c : cands) {
                if (!c.legal) continue;
                if (winner && c.candidate_id == winner->candidate_id) continue;
                sel.confirm(c.neighborhood_id, c.intent, c.candidate_id, 0.0f);
                if (fold_field.learn(c.candidate_id, -0.3f))
                    any_loser_learned = true;
            }

            // Package selection into a SemanticAccumulator for the trainer's
            // provenance chain.  Runs at Phase::Chen (inference-commit); closed
            // at Phase::Xul to mark the fold done before logging.
            {
                kuhul::SemanticAccumulator fold_acc = kuhul::make_accumulator(
                    prompt_archive.prompt_key,
                    "",
                    "fold_score_dispatch",
                    kuhul::Phase::Chen);
                fold_acc.semantic.event_key = prompt_archive.prompt_key;
                fold_acc.semantic.intent    = winner ? winner->intent : "";
                kuhul::set_field(fold_acc, "prompt_key", prompt_archive.prompt_key);
                kuhul::set_field(fold_acc, "gpu_status",
                    gpu::gram_gpu_status_name(result.status));

                kuhul::append_event(fold_acc, kuhul::Phase::Chen, "promote",
                    "n=" + std::to_string(cands.size()));
                kuhul::append_event(fold_acc, kuhul::Phase::Chen, "gpu_dispatch",
                    gpu::gram_gpu_status_name(result.status));

                std::vector<kuhul::CandidateRecord> fold_records;
                fold_records.reserve(cands.size());
                for (const auto& c : cands) {
                    kuhul::CandidateRecord cr;
                    cr.candidate_id = c.candidate_id;
                    cr.gram_edge_id = c.gram_edge_id;
                    cr.combined     = c.combined;
                    cr.wcr_score    = c.wcr_score;
                    cr.legal        = c.legal;
                    fold_records.push_back(std::move(cr));
                }
                kuhul::record_candidates(fold_acc, fold_records,
                    winner ? winner->candidate_id : "");
                kuhul::append_event(fold_acc, kuhul::Phase::Xul, "select",
                    winner ? winner->candidate_id : "none");
                kuhul::close_fold(fold_acc);

                jrom.log("fold_accumulator",
                    kuhul::serialize_accumulator(fold_acc),
                    "observation");
            }

            // Update base weight in the persistent field and flush to disk.
            bool weight_updated = false;
            if (winner)
                weight_updated = fold_field.learn(winner->candidate_id, 1.0f);
            if (weight_updated || any_loser_learned || field_grew)
                fold_field.save(fold_field_path);

            // Persist mapping when new candidate IDs were registered this batch.
            const bool mapping_grew =
                fold_mapping.candidate_mapping_version != pre_map_version;
            if (mapping_grew)
                fold_mapping.save(fold_mapping_path);

            fold_attempted = true;
            fold_obj["dispatched"]      = util::json_bool(result.ok());
            fold_obj["candidate_count"] = util::json_number(
                static_cast<double>(cands.size()));
            fold_obj["gpu_status"]      = util::json_string(
                gpu::gram_gpu_status_name(result.status));
            fold_obj["confidence"]      = util::json_number(
                static_cast<double>(fold_confidence));
            if (!result.ok())
                fold_obj["gpu_error"] = util::json_string(result.error);
            fold_obj["mapping_rows"] = util::json_number(
                static_cast<double>(fold_mapping.size()));
            fold_obj["mapping_version"] = util::json_number(
                static_cast<double>(fold_mapping.candidate_mapping_version));
            if (winner) {
                fold_obj["winner"]         = util::json_string(winner->candidate_id);
                fold_obj["winner_row"]     = util::json_number(
                    static_cast<double>(winner->embedding_row));
                fold_obj["combined"]       = util::json_number(
                    static_cast<double>(winner->combined));
                fold_obj["weight_updated"] = util::json_bool(weight_updated);
            }
            {
                std::vector<size_t> idx(cands.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                    return cands[a].combined > cands[b].combined;
                });
                const size_t top_n = std::min<size_t>(8, cands.size());
                util::JsonArray top;
                top.reserve(top_n);
                for (size_t k = 0; k < top_n; ++k) {
                    const auto& c = cands[idx[k]];
                    util::JsonObject row;
                    row["candidate_id"]    = util::json_string(c.candidate_id);
                    row["combined"]        = util::json_number(static_cast<double>(c.combined));
                    row["wcr_score"]       = util::json_number(static_cast<double>(c.wcr_score));
                    row["edge_weight"]     = util::json_number(static_cast<double>(c.edge_weight));
                    row["attention_score"] = util::json_number(static_cast<double>(c.attention_score));
                    row["retention"]       = util::json_number(static_cast<double>(
                        c.attention_offset < retention_buf.size()
                            ? retention_buf[c.attention_offset] : 1.0f));
                    row["selected"]        = util::json_bool(
                        winner && c.candidate_id == winner->candidate_id);
                    top.push_back(util::json_object(std::move(row)));
                }
                fold_obj["top"] = util::json_array(std::move(top));
            }
            jrom.log("fold_selection",
                     util::stringify(util::json_object(fold_obj)),
                     "observation");
        };

        auto apply_lorentz_transition = [&](const world::GyroResult& gr,
                                            const std::string& source) {
            if (!engine) return;
            // Fold the 6D ABI residual into a deterministic 3D route normal:
            // field/counterexample → x, evidence/verify → y, infer/commit → z.
            std::array<float, 3> direction{
                gr.abi_residual[static_cast<size_t>(world::AbiDim::Field)] +
                    gr.abi_residual[static_cast<size_t>(world::AbiDim::Counterexample)],
                gr.abi_residual[static_cast<size_t>(world::AbiDim::Evidence)] +
                    gr.abi_residual[static_cast<size_t>(world::AbiDim::Verify)],
                gr.abi_residual[static_cast<size_t>(world::AbiDim::Infer)] +
                    gr.abi_residual[static_cast<size_t>(world::AbiDim::Xul)]
            };
            const float direction_norm2 =
                direction[0] * direction[0] +
                direction[1] * direction[1] +
                direction[2] * direction[2];
            const float rapidity = std::clamp(gr.r, 0.0f, 2.5f);
            const float alignment_mix = gr.dynamically_stable ? 1.0f : 0.5f;
            const float mass_scale = 1.0f;

            if (rapidity <= 1e-6f || direction_norm2 <= 1e-12f) {
                engine->clear_lorentz_transition();
            } else {
                engine->set_lorentz_transition(
                    rapidity, direction, alignment_mix, mass_scale);
            }

            util::JsonObject log;
            log["source"] = util::json_string(source);
            log["rapidity"] = util::json_number(rapidity);
            util::JsonArray dir;
            for (float v : direction) dir.push_back(util::json_number(v));
            log["direction"] = util::json_array(std::move(dir));
            log["alignment_mix"] = util::json_number(alignment_mix);
            log["mass_scale"] = util::json_number(mass_scale);
            log["r"] = util::json_number(gr.r);
            log["gamma"] = util::json_number(gr.gamma);
            log["enabled"] = util::json_bool(
                engine->lorentz_transition() && engine->lorentz_transition()->enabled);
            log["semantic_authority"] = util::json_bool(false);
            jrom.log("lorentz_transition",
                     util::stringify(util::json_object(std::move(log))),
                     "observation");
        };

        // ── Projection: build JROM_projection before inference ────────────
        util::JsonObject gyro_obj;
        if (gyro_ready && proj_engine && gyroscope) {
            // Capture window start before logging new events for this call
            uint64_t window_start = last_infer_end_seq;

            // decompose_task carries real prompt terms so s_proj varies per request
            auto prompt_toks = world::Gyroscope::tokenize(*prompt);
            std::string task_query;
            for (size_t i = 0; i < std::min<size_t>(6, prompt_toks.size()); ++i) {
                if (!task_query.empty()) task_query += ' ';
                task_query += prompt_toks[i];
            }
            {
                util::JsonObject dt_log;
                dt_log["query"] = util::json_string(task_query);
                jrom.log("decompose_task",
                         util::stringify(util::json_object(std::move(dt_log))));
            }

            // Sliding window: only events since end of previous infer call (O(window) not O(n²))
            std::vector<io::JROMEvent> trace;
            jrom.replay(window_start, [&](const io::JROMEvent& e){ trace.push_back(e); });
            io::ProjectionRequest proj_req;
            auto proj = proj_engine->project_v(trace, proj_req);

            // ── Build context-conditioned prompt ──────────────────────────
            // Strip ⟁DIM lines (E2 9F 81 prefix) — scoring metadata, not model input
            std::string ctx_prompt = *prompt;
            if (!proj.text.empty()) {
                std::string proj_ctx;
                for (size_t i = 0; i < proj.text.size(); ) {
                    size_t end = proj.text.find('\n', i);
                    if (end == std::string::npos) end = proj.text.size();
                    bool is_dim = (end - i >= 3 &&
                                   (unsigned char)proj.text[i]   == 0xE2 &&
                                   (unsigned char)proj.text[i+1] == 0x9F &&
                                   (unsigned char)proj.text[i+2] == 0x81);
                    if (!is_dim && end > i) { proj_ctx.append(proj.text, i, end - i); proj_ctx += '\n'; }
                    i = (end == proj.text.size()) ? end : end + 1;
                }
                if (!proj_ctx.empty())
                    ctx_prompt = "[Context]\n" + proj_ctx + "---\n" + *prompt;
            }

            // ── Pre-dispatch frame estimate: projection vs current prompt ────
            // This uses the live projection context to set the Lorentz frame
            // before semantic-attention scoring.  It is projection geometry,
            // not semantic authority.
            const auto& gov = gyroscope->make_governs_sets();
            auto proj_tokens = world::Gyroscope::tokenize(proj.text);
            auto s_proj = world::Gyroscope::score_tokens(
                proj_tokens,
                gov,
                &dictionary_lexicon.similarity_by_word,
                0.65f);
            auto s_prompt = world::Gyroscope::score_tokens(
                prompt_toks,
                gov,
                &dictionary_lexicon.similarity_by_word,
                0.65f);
            const float pre_l_bar = engine->mean_token_latency_ms(32) / 1000.0f;
            const float pre_sigma = engine->router().routing_variance(16);
            world::GyroInputs pre_inputs{ .omega=1.0f,
                                          .m_dds = pre_l_bar * (1.0f + pre_sigma),
                                          .k     = 2.0f };
            const auto pre_gr = gyroscope->observe(s_proj, s_prompt, pre_inputs);
            apply_lorentz_transition(pre_gr, "projection_prompt_residual");
            run_semantic_attention();
            fold_confidence = std::clamp(pre_gr.r / 2.5f, 0.1f, 1.0f)
                              * (pre_gr.dynamically_stable ? 1.0f : 0.5f);
            run_fold_selection();

            // ── DDS inference conditioned on projection context ────────────
            auto result = engine->run(ctx_prompt, max_tok);
            if (!result.ok || result.text.empty()) {
                {
                    util::JsonObject kg_log;
                    kg_log["reason"] = util::json_string("dds_empty");
                    kg_log["query"]  = util::json_string(task_query);
                    jrom.log("knowledge_gap",
                             util::stringify(util::json_object(std::move(kg_log))));
                }
                {
                    util::JsonObject wr_log;
                    wr_log["query"]    = util::json_string(task_query);
                    wr_log["delegate"] = util::json_string("Librarian");
                    jrom.log("web_research",
                             util::stringify(util::json_object(std::move(wr_log))));
                }
            } else {
                util::JsonObject vr_log;
                vr_log["ok"] = util::json_bool(true);
                jrom.log("verify_result",
                         util::stringify(util::json_object(std::move(vr_log))));
            }

            // ── Score completion with natural-language governs vocabulary ──
            auto dds_tokens  = world::Gyroscope::tokenize(result.text);
            auto s_dds = world::Gyroscope::score_tokens(
                dds_tokens,
                gov,
                &dictionary_lexicon.similarity_by_word,
                0.65f);

            // ── GyroInputs: L̄_token (s) × (1 + σ_route); ω=1 placeholder ─
            float l_bar = engine->mean_token_latency_ms(32) / 1000.0f;
            float sigma = engine->router().routing_variance(16);
            world::GyroInputs inputs{ .omega=1.0f,
                                      .m_dds = l_bar * (1.0f + sigma),
                                      .k     = 2.0f };
            auto gr = gyroscope->observe(s_proj, s_dds, inputs);
            apply_lorentz_transition(gr, "projection_completion_residual");

            // ── Log gyro_observe to JROM ──────────────────────────────────
            std::string gyro_json =
                "{\"r\":" + std::to_string(gr.r) +
                ",\"gamma\":" + std::to_string(gr.gamma) +
                ",\"stable\":" + (gr.dynamically_stable ? "true" : "false") +
                ",\"aligned\":" + (gr.aligned ? "true" : "false") +
                ",\"anchor\":\"" + gr.dominant_anchor + "\""
                ",\"proj_hash\":\"" + proj.projection_hash + "\""
                "}";
            jrom.log("gyro_observe", gyro_json);

            // ── Governor advisory ─────────────────────────────────────────
            std::string gov_rec;
            if (!gr.dynamically_stable && !gr.aligned)
                gov_rec = "reduce_omega_and_anchor";
            else if (!gr.dynamically_stable)
                gov_rec = "reduce_omega";
            else if (!gr.aligned && !gr.dominant_anchor.empty())
                gov_rec = "anchor";
            else
                gov_rec = "nominal";
            std::string gov_jlog =
                "{\"stable\":"  + std::string(gr.dynamically_stable ? "true" : "false") +
                ",\"aligned\":" + std::string(gr.aligned ? "true" : "false") +
                ",\"gamma\":"   + std::to_string(gr.gamma) +
                ",\"r\":"       + std::to_string(gr.r) +
                ",\"dominant_anchor\":\"" + gr.dominant_anchor + "\"" +
                ",\"recommendation\":\"" + gov_rec + "\"}";
            jrom.log("governor_advisory", gov_jlog);

            // ── TOOL-µ: execute advisory if non-nominal ───────────────────
            util::JsonPtr tool_result;
            if (gov_rec != "nominal") {
                io::RouteRequest tool_req;
                tool_req.query = gov_rec;
                tool_req.phase = kuhul::Phase::Sek;
                tool_req.lane = "tool_governor";
                util::JsonObject tool_payload;
                tool_payload["anchor"] = util::json_string(gr.dominant_anchor);
                tool_req.payload = util::json_object(std::move(tool_payload));
                auto tool_route = router.route(tool_req);
                tool_result = tool_route.response;
                jrom.log("dispatch_call",
                         "{\"tool\":\"TOOL-\xC2\xB5\",\"advisory\":\"" + gov_rec +
                         "\",\"anchor\":\"" + gr.dominant_anchor +
                         "\",\"handler\":\"" + tool_route.handler_id + "\"}");
            }

            // ── Build RPC response with gyro + governor summary ───────────
            util::JsonObject obj;
            obj["ok"]   = util::json_bool(result.ok);
            obj["text"] = util::json_string(result.text);
            if (!result.error.empty())
                obj["error"] = util::json_string(result.error);
            util::JsonArray toks;
            for (auto id : result.token_ids)
                toks.push_back(util::json_number(static_cast<double>(id)));
            obj["token_ids"] = util::json_array(std::move(toks));
            if (result.text.empty() || !result.ok) {
                util::JsonObject rp;
                rp["needed"]   = util::json_bool(true);
                rp["delegate"] = util::json_string("Librarian");
                rp["query"]    = util::json_string(task_query);
                rp["reason"]   = util::json_string("dds_empty");
                obj["research"] = util::json_object(std::move(rp));
            }

            gyro_obj["r"]               = util::json_number(static_cast<double>(gr.r));
            gyro_obj["gamma"]           = util::json_number(static_cast<double>(gr.gamma));
            gyro_obj["stable"]          = util::json_bool(gr.dynamically_stable);
            gyro_obj["aligned"]         = util::json_bool(gr.aligned);
            gyro_obj["dominant_anchor"] = util::json_string(gr.dominant_anchor);
            gyro_obj["proj_hash"]       = util::json_string(proj.projection_hash);
            gyro_obj["anchor_satisfied"]= util::json_bool(proj.anchor_satisfied);
            obj["gyro"] = util::json_object(std::move(gyro_obj));

            util::JsonObject gov_resp;
            gov_resp["stable"]          = util::json_bool(gr.dynamically_stable);
            gov_resp["aligned"]         = util::json_bool(gr.aligned);
            gov_resp["gamma"]           = util::json_number(static_cast<double>(gr.gamma));
            gov_resp["r"]               = util::json_number(static_cast<double>(gr.r));
            gov_resp["dominant_anchor"] = util::json_string(gr.dominant_anchor);
            gov_resp["recommendation"]  = util::json_string(gov_rec);
            obj["governor"] = util::json_object(std::move(gov_resp));
            if (tool_result && !tool_result->is_null())
                obj["tool"] = std::move(tool_result);
            attach_gemma_learning(obj, *prompt, result.text);
            obj["model_presence"] = make_model_presence();
            if (attention_attempted)
                obj["semantic_attention"] = util::json_object(std::move(attention_obj));
            if (fold_attempted)
                obj["fold_selection"] = util::json_object(std::move(fold_obj));

            last_infer_end_seq = jrom.last_seq() + 1;
            return finalize_infer_event(std::move(obj), result.text, result.ok, "Ch'en");
        }

        // Gyroscope not available — plain inference path.  If a previous gyro
        // pass installed a frame, semantic attention still dispatches through it;
        // otherwise the scorer uses the disabled identity transition.
        run_semantic_attention();
        run_fold_selection();
        auto result = engine->run(*prompt, max_tok);
        util::JsonObject obj;
        obj["ok"]   = util::json_bool(result.ok);
        obj["text"] = util::json_string(result.text);
        if (!result.error.empty())
            obj["error"] = util::json_string(result.error);
        util::JsonArray toks;
        for (auto id : result.token_ids)
            toks.push_back(util::json_number(static_cast<double>(id)));
        obj["token_ids"] = util::json_array(std::move(toks));
        attach_gemma_learning(obj, *prompt, result.text);
        obj["model_presence"] = make_model_presence();
        if (attention_attempted)
            obj["semantic_attention"] = util::json_object(std::move(attention_obj));
        if (fold_attempted)
            obj["fold_selection"] = util::json_object(std::move(fold_obj));
        return finalize_infer_event(std::move(obj), result.text, result.ok, "Ch'en");
    });

    // llm_infer — auto-swapping in-process llama.cpp inference.
    // Selects model by: (1) micronaut_id param → preferred_model_gguf from track file,
    // (2) advisor_like content routing → cfg.advisor_model, (3) default → cfg.gguf_model.
    // Dismounts the previous model when a different path is requested.
    // Request: {"prompt": "...", "sys": "...", "micronaut_id": "CHAT-µ"} (sys, micronaut_id optional)
    khanary::llm::ModelSwapper llm_swapper;
    rpc_server.register_method("llm_infer", [&](const util::JsonPtr& p) {
        auto prompt_opt = p ? p->get_string("prompt") : std::nullopt;
        util::JsonObject res;
        if (!prompt_opt) {
            res["ok"]    = util::json_bool(false);
            res["error"] = util::json_string("missing 'prompt' field");
            return util::json_object(std::move(res));
        }
        std::string sys = (p ? p->get_string("sys") : std::nullopt).value_or("");
        // Auto-inject semantic driver meta when no sys prompt is provided.
        // Factory-created drivers in drivers/micronauts/semantic/ feed the meta pipeline.
        if (sys.empty()) {
            if (auto match = io::find_best_semantic_meta(*prompt_opt, cfg.data_dir)) {
                sys = match->response;
            }
        }
        const std::string qlow = lower_copy(*prompt_opt);

        auto word_ci = [](const std::string& s, const std::string& w) -> bool {
            size_t p2 = 0;
            while ((p2 = s.find(w, p2)) != std::string::npos) {
                const bool pre  = (p2 == 0) || !std::isalnum((unsigned char)s[p2 - 1]);
                const bool post = (p2 + w.size() >= s.size()) || !std::isalnum((unsigned char)s[p2 + w.size()]);
                if (pre && post) return true;
                p2 += w.size();
            }
            return false;
        };

        // Select model path and template.
        std::string model_path;
        std::string model_tmpl;  // "chatml" or "gemma"
        std::string active_track = "CHAT-µ";

        // Priority 1: micronaut_id param → track's preferred GGUF.
        const auto mu_id = p ? p->get_string("micronaut_id") : std::nullopt;
        if (mu_id) {
            const auto* mu = mu_reg.get(*mu_id);
            if (mu && !mu->preferred_model_gguf.empty()) {
                model_path   = mu->preferred_model_gguf;
                model_tmpl   = mu->preferred_model_template;
                active_track = mu->id;
            }
        }

        // Priority 2: advisor_like content routing → cfg.advisor_model.
        if (model_path.empty()) {
            const bool advisor_like =
                word_ci(qlow, "advise")  || word_ci(qlow, "advisor")  ||
                word_ci(qlow, "analyze") || word_ci(qlow, "analyse")  ||
                word_ci(qlow, "reason")  || word_ci(qlow, "reasoning")||
                word_ci(qlow, "propose") || word_ci(qlow, "proposal") ||
                word_ci(qlow, "gap")     || word_ci(qlow, "evaluate") ||
                word_ci(qlow, "compare") || word_ci(qlow, "tradeoff") ||
                word_ci(qlow, "pros and cons");
            if (advisor_like && !cfg.advisor_model.empty() && fs::exists(cfg.advisor_model)) {
                model_path   = cfg.advisor_model;
                model_tmpl   = "chatml"; // Qwen uses ChatML
                active_track = "ADVISOR-µ";
            }
        }

        // Priority 3: server default --gguf-model.
        if (model_path.empty() && !cfg.gguf_model.empty() && fs::exists(cfg.gguf_model)) {
            model_path = cfg.gguf_model;
            // LFM2.5 (or any ChatML model) declared in the server config gets chatml.
            // Simple heuristic: if path contains "LFM" or "Qwen" → chatml, else gemma.
            const auto& mp = model_path;
            if (mp.find("LFM") != std::string::npos || mp.find("Qwen") != std::string::npos ||
                mp.find("lfm") != std::string::npos || mp.find("qwen") != std::string::npos)
                model_tmpl = "chatml";
        }

        if (model_path.empty()) {
            res["ok"]    = util::json_bool(false);
            res["error"] = util::json_string(
                "no GGUF model configured — start with --gguf-model or set model.gguf in track file");
            return util::json_object(std::move(res));
        }

        // Augment sys with AIML/ELIZA pattern context when the active track declares
        // pattern_source: "aiml" / "eliza" / "aiml+eliza".
        // Pattern meta is prepended so LFM sees the domain grammar before any
        // semantic-driver context injected above.
        {
            std::string pat_src;
            if (const auto* mu = mu_reg.get(active_track))
                pat_src = mu->pattern_source;
            if (!pat_src.empty()) {
                const std::string pat = io::load_pattern_meta_for_track(
                    pat_src, *prompt_opt, cfg.data_dir);
                if (!pat.empty()) {
                    sys = sys.empty() ? pat : (pat + "\n---\n" + sys);
                }
            }
        }

        khanary::llm::LlamaConfig llama_cfg;
        llama_cfg.ctx_size     = 2048;
        llama_cfg.max_tokens   = 512;
        llama_cfg.temperature  = (active_track == "ADVISOR-µ") ? 0.3f : 0.7f;
        llama_cfg.chat_template = (model_tmpl == "chatml")
            ? khanary::llm::ChatTemplate::ChatML
            : khanary::llm::ChatTemplate::Gemma;

        if (!llm_swapper.swap_to(model_path, llama_cfg)) {
            res["ok"]    = util::json_bool(false);
            res["error"] = util::json_string("model failed to load: " + model_path);
            return util::json_object(std::move(res));
        }

        if (active_track == "ADVISOR-µ") {
                const auto prompt_archive = persist_prompt_submission(
                    *prompt_opt, prompt_ingest_root, &dictionary_lexicon, &factory, &vm.gram_store());
                const std::string pek = record_key_grammar_memory_event(
                    "prompt", "llm_infer", "USER-µ", "Pop",
                    prompt_archive.prompt_key, "", *prompt_opt, "",
                    "UNKNOWN", false, &factory, &vm.gram_store(), &jrom);
                const std::string text = llm_swapper.call_fn()(sys, *prompt_opt);
                record_key_grammar_memory_event(
                    "model_response", "llm_infer", "ADVISOR-µ", "Yax",
                    prompt_archive.prompt_key, pek, text, "",
                    "UNKNOWN", false, &factory, &vm.gram_store(), &jrom);
                const auto grams = build_learning_ngrams(text, 24);
                for (const auto& g : grams) {
                    vm.gram_store().gramWrite(g,
                        text.size() > 120 ? text.substr(0, 120) : text,
                        "Yax", "ADVISOR-µ", "definitions_and_relations");
                    factory.record_key("response_ngram:" + g, prompt_archive.prompt_key);
                }
                {
                    util::JsonObject rl;
                    rl["query"]       = util::json_string(*prompt_opt);
                    rl["method"]      = util::json_string("llm_infer");
                    rl["track"]       = util::json_string("ADVISOR-µ");
                    rl["model"]       = util::json_string(gemma_model_name);
                    rl["ngram_count"] = util::json_number(static_cast<double>(grams.size()));
                    jrom.log("replay_candidate", util::stringify(util::json_object(std::move(rl))));
                    util::JsonObject el;
                    el["track"]       = util::json_string("ADVISOR-µ");
                    el["model"]       = util::json_string(gemma_model_name);
                    el["capability"]  = util::json_string("advisory_response_ngrams");
                    el["trigger_on"]  = util::json_string("infer_commit");
                    el["ngram_count"] = util::json_number(static_cast<double>(grams.size()));
                    jrom.log("evolve_candidate", util::stringify(util::json_object(std::move(el))));
                }
                util::JsonObject res2;
                res2["ok"]           = util::json_bool(!text.empty());
                res2["text"]         = util::json_string(text);
                res2["track"]        = util::json_string("ADVISOR-µ");
                res2["prompt_key"]   = util::json_string(prompt_archive.prompt_key);
                res2["prompt_ngrams"]= util::json_number(static_cast<double>(prompt_archive.prompt_ngram_count));
                if (text.empty())
                    res2["error"] = util::json_string("advisor model generated empty response");
                if (!grams.empty()) {
                    util::JsonArray arr;
                    for (const auto& g : grams) arr.push_back(util::json_string(g));
                    res2["ngrams"] = util::json_array(std::move(arr));
                }
                return util::json_object(std::move(res2));
        }

        const auto prompt_archive = persist_prompt_submission(
            *prompt_opt, prompt_ingest_root, &dictionary_lexicon, &factory, &vm.gram_store());
        const std::string prompt_event_key = record_key_grammar_memory_event(
            "prompt", "llm_infer", "USER-µ", "Pop",
            prompt_archive.prompt_key, "", *prompt_opt, "",
            "UNKNOWN", false, &factory, &vm.gram_store(), &jrom);

        const std::string text = llm_swapper.call_fn()(sys, *prompt_opt);
        res["ok"]            = util::json_bool(!text.empty());
        res["text"]          = util::json_string(text);
        res["prompt_key"]    = util::json_string(prompt_archive.prompt_key);
        res["prompt_ngrams"] = util::json_number(static_cast<double>(prompt_archive.prompt_ngram_count));
        if (text.empty()) {
            res["error"] = util::json_string("llama generated empty response");
            return util::json_object(std::move(res));
        }

        // After any MEMORY-µ dispatch, flush live gram state to the semantic driver
        // so the next call picks up fresh memory context.
        if (active_track == "MEMORY-\xc2\xb5" || active_track == "MEMORY-u")
            vm.gram_store().flush_memory_driver(cfg.data_dir);

        // Detect code vs chat to route ngrams to the right backend slot.
        const bool code_like =
            text.find("```") != std::string::npos ||
            word_ci(qlow, "code")       || word_ci(qlow, "coding")     ||
            word_ci(qlow, "function")   || word_ci(qlow, "compile")    ||
            word_ci(qlow, "syntax")     || word_ci(qlow, "api")        ||
            word_ci(qlow, "c++")        || word_ci(qlow, "cpp")        ||
            word_ci(qlow, "javascript") || word_ci(qlow, "typescript") ||
            word_ci(qlow, "html")       || word_ci(qlow, "css")        ||
            word_ci(qlow, "python")     || word_ci(qlow, "debug");
        const std::string response_track = code_like ? "CODEGEN-µ" : "CHAT-µ";
        const std::string response_phase = code_like ? "Wo"        : "Pop";

        record_key_grammar_memory_event(
            "model_response", "llm_infer", response_track, "Ch'en",
            prompt_archive.prompt_key, prompt_event_key, text, "",
            "UNKNOWN", false, &factory, &vm.gram_store(), &jrom);

        // Write response ngrams directly into gram_store and key store.
        const auto grams = build_learning_ngrams(text, 24);
        for (const auto& g : grams) {
            vm.gram_store().gramWrite(
                g,
                text.size() > 120 ? text.substr(0, 120) : text,
                response_phase, response_track, "definitions_and_relations");
            factory.record_key("response_ngram:" + g, prompt_archive.prompt_key);
        }

        // jrom: replay + evolve candidates for track evolution pipeline.
        {
            util::JsonObject rl;
            rl["query"]       = util::json_string(*prompt_opt);
            rl["method"]      = util::json_string("llm_infer");
            rl["track"]       = util::json_string(response_track);
            rl["model"]       = util::json_string(gemma_model_name);
            rl["ngram_count"] = util::json_number(static_cast<double>(grams.size()));
            jrom.log("replay_candidate", util::stringify(util::json_object(std::move(rl))));

            util::JsonObject el;
            el["track"]       = util::json_string(response_track);
            el["model"]       = util::json_string(gemma_model_name);
            el["capability"]  = util::json_string(code_like ? "codegen_response_ngrams" : "chat_response_ngrams");
            el["trigger_on"]  = util::json_string("infer_commit");
            el["ngram_count"] = util::json_number(static_cast<double>(grams.size()));
            jrom.log("evolve_candidate", util::stringify(util::json_object(std::move(el))));
        }

        // Response shape — mirrors infer handler's attach_gemma_learning output.
        res["track"] = util::json_string(response_track);
        if (!grams.empty()) {
            util::JsonArray arr;
            for (const auto& g : grams) arr.push_back(util::json_string(g));
            res["ngrams"] = util::json_array(std::move(arr));

            auto make_ngram_slice = [&](size_t n) {
                util::JsonArray a;
                for (size_t i = 0; i < std::min(n, grams.size()); ++i)
                    a.push_back(util::json_string(grams[i]));
                return a;
            };
            util::JsonObject replay;
            replay["method"]     = util::json_string("llm_infer");
            replay["query"]      = util::json_string(*prompt_opt);
            replay["track"]      = util::json_string(response_track);
            replay["trigger_on"] = util::json_string("infer_commit");
            replay["ngrams"]     = util::json_array(make_ngram_slice(12));
            res["replay_candidate"] = util::json_object(std::move(replay));

            util::JsonObject evolve;
            evolve["track"]      = util::json_string(response_track);
            evolve["model"]      = util::json_string(gemma_model_name);
            evolve["capability"] = util::json_string(code_like ? "codegen_response_ngrams" : "chat_response_ngrams");
            evolve["trigger_on"] = util::json_string("infer_commit");
            evolve["ngrams"]     = util::json_array(make_ngram_slice(12));
            res["evolve_candidate"] = util::json_object(std::move(evolve));
        }
        return util::json_object(std::move(res));
    });

    // dds_probe — minimal forward pass test that only requires dds_ready (no GGUF).
    // Usage: {"prompt": "Hello", "max_tokens": 1}
    rpc_server.register_method("dds_probe", [&](const util::JsonPtr& p) {
        util::JsonObject res;
        if (!dds_ready || !engine) {
            res["ok"]    = util::json_bool(false);
            res["error"] = util::json_string("DDS model not loaded — start with --dds-kson/--dds-manifest/--dds-shards");
            return util::json_object(std::move(res));
        }
        auto prompt_str = p ? p->get_string("prompt") : std::nullopt;
        if (!prompt_str) {
            res["ok"]    = util::json_bool(false);
            res["error"] = util::json_string("missing 'prompt' field");
            return util::json_object(std::move(res));
        }
        uint32_t max_tok = 1;
        if (p) { if (auto v = p->get_number("max_tokens")) max_tok = static_cast<uint32_t>(*v); }
        auto ir = engine->run(*prompt_str, max_tok);
        res["ok"]               = util::json_bool(ir.ok);
        res["text"]             = util::json_string(ir.text);
        if (!ir.error.empty())
            res["error"]        = util::json_string(ir.error);
        res["preload_resident"] = util::json_number(static_cast<double>(engine->dds_preload_resident_tensors()));
        res["preload_total"]    = util::json_number(static_cast<double>(engine->dds_preload_total_tensors()));
        res["resident_mib"]     = util::json_number(static_cast<double>(engine->dds_resident_bytes() / (1024ULL * 1024ULL)));
        return util::json_object(std::move(res));
    });

    // Register /v1/chat/completions on the RPC server port.
    // key_query is called directly (no network hop); the sidecar (tiny-llama or
    // compatible) on port_chat handles prompts the semantic engine can't satisfy.
    io::ChatCompletionHandler chat_handler(
        [&](const std::string& method, const util::JsonPtr& params) {
            return rpc_server.call_method(method, params);
        },
        cfg.port_chat);
    rpc_server.register_path("/v1/chat/completions",
        [&](const std::string& body) { return chat_handler.handle(body); });

    // Start servers
    if (!mcp_server.start())
        std::cerr << "WARN: MCP server failed to start on port " << cfg.port_mcp << "\n";
    else
        std::cout << "  MCP:     listening on :" << cfg.port_mcp << "\n";

    if (!rpc_server.start())
        std::cerr << "WARN: JSON-RPC server failed to start on port " << cfg.port_rpc << "\n";
    else
        std::cout << "  RPC:     listening on :" << cfg.port_rpc << "\n";
    if (cfg.port_chat > 0)
        std::cout << "  CHAT:    sidecar on :" << cfg.port_chat << "\n";

    {
        util::JsonObject ports_row;
        auto ports_obj = util::json_object();
        ports_obj->mut_object()["mcp"] = util::json_number((double)cfg.port_mcp);
        ports_obj->mut_object()["rpc"] = util::json_number((double)cfg.port_rpc);
        ports_row["ports"] = ports_obj;
        jrom.log("startup",
                 util::stringify(util::json_object(std::move(ports_row))));
    }

    std::cout << "\nKHANARY ready. Press Ctrl+C to stop.\n";

    // Main loop
    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nShutting down...\n";
    mcp_server.stop();
    rpc_server.stop();
    {
        auto ca_aiml = lifecycle_commit("aiml-state", "shutdown-persist");
        if (ca_aiml) aiml.persist(std::move(*ca_aiml), aiml_state_path);
    }
    {
        auto ca_factory = lifecycle_commit("keystore", "shutdown-persist");
        if (ca_factory) factory.persist(std::move(*ca_factory), factory_state_json_path, factory_state_bin_path);
    }
    save_keystore_cache_meta(factory_cache_meta_path, compute_keystore_cache_fingerprint(cfg.data_dir));
    confidence.save(cfg.data_dir + "/confidence.json");
    gram_store.gramCheckpoint();
    jrom.log("shutdown", "{}");

    return 0;
}
