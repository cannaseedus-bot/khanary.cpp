#pragma once
#include "manifest_registry.h"
#include <string>

namespace khanary::manifest {

// Loads all 4 standard manifests from the data/manifest/ directory
// into the given registry.
// Manifests are treated as declarative data only; execution authority is outside this loader.
void load_all(ManifestRegistry& reg, const std::string& data_dir);

// Load a single manifest file
// Accepted files must be regular files with ".manifest.json" suffix.
void load_file(ManifestRegistry& reg, const std::string& path);

} // namespace khanary::manifest
