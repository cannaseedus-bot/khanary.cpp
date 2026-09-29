#include "manifest_loader.h"
#include <filesystem>
#include <stdexcept>
#include <array>

namespace khanary::manifest {

namespace fs = std::filesystem;

namespace {

static bool has_manifest_suffix(const fs::path& p) {
    const std::string name = p.filename().string();
    constexpr const char* kSuffix = ".manifest.json";
    const size_t n = name.size();
    const size_t m = std::char_traits<char>::length(kSuffix);
    return n >= m && name.compare(n - m, m, kSuffix) == 0;
}

static fs::path normalize_manifest_path(const std::string& path) {
    fs::path p(path);
    if (p.empty()) throw std::runtime_error("manifest_loader: empty manifest path");
    p = p.lexically_normal();
    if (!has_manifest_suffix(p)) {
        throw std::runtime_error(
            "manifest_loader: expected *.manifest.json file, got: " + p.string());
    }
    return p;
}

} // namespace

void load_file(ManifestRegistry& reg, const std::string& path) {
    const fs::path p = normalize_manifest_path(path);
    if (!fs::exists(p)) {
        throw std::runtime_error("manifest_loader: manifest file not found: " + p.string());
    }
    if (!fs::is_regular_file(p)) {
        throw std::runtime_error("manifest_loader: manifest path is not a regular file: " + p.string());
    }
    reg.load(p.string());
}

void load_all(ManifestRegistry& reg, const std::string& data_dir) {
    static constexpr std::array<const char*, 11> manifests = {
        "css.manifest.json",
        "kuhul.manifest.json",
        "words.manifest.json",
        "micronaut.manifest.json",
        "user.manifest.json",
        "aiml.manifest.json",
        "mx2lm.manifest.json",
        "eliza.manifest.json",
        "tools.manifest.json",
        "nodes.manifest.json",
        "folds.manifest.json",
    };
    fs::path base = fs::path(data_dir) / "manifest";
    if (!fs::exists(base) || !fs::is_directory(base)) {
        return; // optional manifest directory
    }
    for (const char* name : manifests) {
        fs::path p = base / name;
        if (fs::exists(p)) {
            if (!fs::is_regular_file(p)) {
                throw std::runtime_error(
                    "manifest_loader: expected regular manifest file, got: " + p.string());
            }
            load_file(reg, p.string());
        }
        // Missing standard manifest files are intentionally skipped — not all are required.
    }
}

} // namespace khanary::manifest
