#pragma once
#include "micronaut_registry.h"
#include <string>
#include <filesystem>

namespace khanary::micronaut {

// Load micronaut.semantic-tracks.v1.json files
// Each file defines a micronaut: id, phase, governs, folds, memory patterns,
// and optional semantic Key vector metadata used by W·C·R routing.
class TrackLoader {
public:
    explicit TrackLoader(MicronautRegistry& reg);

    // Load a single track file
    bool load_file(const std::string& path);

    // Load all .semantic-tracks.v1.json files in a directory
    int load_dir(const std::string& dir);

private:
    MicronautRegistry& reg_;
};

} // namespace khanary::micronaut
