#include "manifest_loader.h"
#include <filesystem>
#include <stdexcept>

namespace khanary::manifest {

namespace fs = std::filesystem;

void load_file(ManifestRegistry& reg, const std::string& path) {
    reg.load(path);
}

void load_all(ManifestRegistry& reg, const std::string& data_dir) {
    static const char* manifests[] = {
        "css.manifest.json",
        "kuhul.manifest.json",
        "words.manifest.json",
        "micronaut.manifest.json",
    };
    fs::path base = fs::path(data_dir) / "manifest";
    for (auto* name : manifests) {
        fs::path p = base / name;
        if (fs::exists(p)) {
            reg.load(p.string());
        }
        // Missing manifest files are silently skipped — not all are required
    }
}

} // namespace khanary::manifest
