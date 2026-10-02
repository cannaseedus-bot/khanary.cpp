#include "phase_driver_lock.h"
#include "../util/json.h"
#include "../util/sha256.h"
#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

namespace khanary::kuhul {
namespace {
namespace fs = std::filesystem;

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool read_text_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

std::string strip_sha256_prefix(std::string s) {
    constexpr const char* kPrefix = "sha256:";
    if (s.rfind(kPrefix, 0) == 0) s.erase(0, 7);
    return lower_copy(s);
}

bool is_pending_hash(const std::string& s) {
    const auto low = lower_copy(s);
    return low.empty() || low.find("__pending__") != std::string::npos;
}

std::string file_sha256_hex(const fs::path& path, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (err) *err = "cannot open file for hashing: " + path.string();
        return {};
    }
    std::string bytes((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    return util::sha256_hex(bytes.data(), bytes.size());
}

const util::JsonValue* read_hash_map(const util::JsonPtr& runtime_meta_json) {
    if (!runtime_meta_json || !runtime_meta_json->is_object()) return nullptr;
    return runtime_meta_json->get("phase_driver_sha256");
}

std::string lookup_declared_hash(const util::JsonValue* hash_map, const std::string& dll_name) {
    if (!hash_map || !hash_map->is_object()) return {};
    const auto& obj = hash_map->as_object();
    auto it = obj.find(dll_name);
    if (it != obj.end() && it->second && it->second->is_string())
        return it->second->as_string();
    return {};
}

} // namespace

bool PhaseDriverLock::load(const std::string& fold_runtime_path,
                           const std::string& runtime_metadata_path,
                           std::string* err)
{
    enabled_ = false;
    strict_ = false;
    ready_ = true;
    entries_.clear();
    phase_index_.clear();

    std::string runtime_text;
    if (!read_text_file(fold_runtime_path, runtime_text)) {
        return true; // no lock contract present, keep disabled
    }

    util::JsonPtr runtime_json;
    try {
        runtime_json = util::parse(runtime_text);
    } catch (const std::exception&) {
        runtime_json.reset();
    }
    if (!runtime_json || !runtime_json->is_object()) {
        if (err) *err = "invalid JSON in " + fold_runtime_path;
        return false;
    }

    const util::JsonValue* lock_cfg = runtime_json->get("phase_driver_lock");
    if (!lock_cfg || !lock_cfg->is_object()) {
        return true; // lock section not present, keep disabled
    }

    enabled_ = lock_cfg->get_bool("enabled").value_or(false);
    const std::string mode = lower_copy(lock_cfg->get_string("mode").value_or("audit"));
    strict_ = (mode == "enforce") || lock_cfg->get_bool("strict").value_or(false);
    if (!enabled_) return true;

    const std::string dll_root = lock_cfg->get_string("dll_root").value_or("data/drivers/phases");
    const util::JsonValue* phases_val = lock_cfg->get("phases");
    if (!phases_val || !phases_val->is_array()) {
        ready_ = false;
        if (strict_) {
            if (err) *err = "phase_driver_lock.enabled=true but phases array is missing";
            return false;
        }
        return true;
    }

    util::JsonPtr runtime_meta_json;
    std::string runtime_meta_text;
    if (read_text_file(runtime_metadata_path, runtime_meta_text)) {
        try {
            runtime_meta_json = util::parse(runtime_meta_text);
        } catch (const std::exception&) {
            runtime_meta_json.reset();
        }
    }
    const util::JsonValue* hash_map = read_hash_map(runtime_meta_json);

    const fs::path dll_root_path = fs::path(dll_root);
    for (const auto& phase_ptr : phases_val->as_array()) {
        if (!phase_ptr || !phase_ptr->is_object()) continue;

        auto phase_name_opt = phase_ptr->get_string("phase");
        auto dll_name_opt   = phase_ptr->get_string("dll");
        if (!phase_name_opt || !dll_name_opt || phase_name_opt->empty() || dll_name_opt->empty())
            continue;

        PhaseDriverEntryStatus row;
        row.phase_name = *phase_name_opt;
        row.dll_name = *dll_name_opt;
        row.required = phase_ptr->get_bool("required_in_enforce_mode").value_or(true);

        Phase phase{};
        if (!phase_from_name(row.phase_name, phase)) {
            row.error = "unknown phase in lock config: " + row.phase_name;
            row.verified = false;
            entries_.push_back(row);
            continue;
        }
        row.phase = phase;

        fs::path dll_path(row.dll_name);
        if (!dll_path.is_absolute())
            dll_path = dll_root_path / dll_path;
        row.dll_path = dll_path.lexically_normal().string();

        std::error_code ec;
        row.exists = fs::exists(dll_path, ec) && !ec;

        row.expected_sha256 = lookup_declared_hash(hash_map, row.dll_name);
        const std::string expected_norm = strip_sha256_prefix(row.expected_sha256);
        row.hash_declared = !is_pending_hash(row.expected_sha256);

        if (row.exists) {
            std::string hash_err;
            row.actual_sha256 = lower_copy(file_sha256_hex(dll_path, &hash_err));
            if (!hash_err.empty()) row.error = hash_err;
        }

        if (row.hash_declared && !row.actual_sha256.empty())
            row.hash_match = (expected_norm == row.actual_sha256);
        row.verified = row.exists && row.hash_declared && row.hash_match;

        if (row.error.empty()) {
            if (!row.exists) row.error = "missing phase driver artifact";
            else if (!row.hash_declared) row.error = "hash missing or pending in runtime metadata";
            else if (!row.hash_match) row.error = "sha256 mismatch";
        }

        phase_index_[static_cast<int>(phase)] = entries_.size();
        entries_.push_back(std::move(row));
    }

    bool required_ok = true;
    for (const auto& row : entries_) {
        if (row.required && !row.verified) {
            required_ok = false;
            break;
        }
    }
    ready_ = required_ok;

    if (strict_ && !ready_) {
        if (err) *err = "phase_driver_lock enforce mode failed: " + summary();
        return false;
    }
    return true;
}

bool PhaseDriverLock::allow_phase(Phase p, std::string* err) const {
    if (!enabled_) return true;

    auto it = phase_index_.find(static_cast<int>(p));
    if (it == phase_index_.end()) {
        if (strict_) {
            if (err) *err = "missing phase binding for " + std::string(phase_name(p));
            return false;
        }
        return true;
    }

    const auto& row = entries_[it->second];
    if (row.verified) return true;

    if (strict_ && row.required) {
        if (err) {
            *err = "phase " + row.phase_name + " driver rejected: " + row.error +
                   " (" + row.dll_path + ")";
        }
        return false;
    }
    return true;
}

std::string PhaseDriverLock::summary() const {
    if (!enabled_) return "disabled";

    size_t verified = 0;
    size_t required = 0;
    for (const auto& row : entries_) {
        if (row.verified) ++verified;
        if (row.required) ++required;
    }

    std::ostringstream ss;
    ss << "mode=" << (strict_ ? "enforce" : "audit")
       << ", verified=" << verified << "/" << entries_.size()
       << ", required=" << required
       << ", ready=" << (ready_ ? "true" : "false");
    return ss.str();
}

} // namespace khanary::kuhul
