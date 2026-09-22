#pragma once
#include "manifest_registry.h"
#include <string>

namespace khanary::manifest {

// Loads all 4 standard manifests from the data/manifest/ directory
// into the given registry.
void load_all(ManifestRegistry& reg, const std::string& data_dir);

// Load a single manifest file
void load_file(ManifestRegistry& reg, const std::string& path);

} // namespace khanary::manifest
