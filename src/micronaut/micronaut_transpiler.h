#pragma once

#include "micronaut_registry.h"
#include "../util/json.h"

namespace khanary::micronaut {

struct MicronautTranspileOptions {
    bool include_inactive = false;
};

// MicronautTranspiler projects MicronautEntry declarations into the canonical
// route-contract key-grammar row shape used by the Python artifact compiler.
// It is a normalization/projection surface only: it does not mutate the
// registry, execute a micronaut, or grant semantic/fact authority.
class MicronautTranspiler {
public:
    static util::JsonPtr compile_entry(const MicronautEntry& entry);
    static util::JsonArray compile_registry(
        const MicronautRegistry& registry,
        MicronautTranspileOptions options = {});
};

} // namespace khanary::micronaut
