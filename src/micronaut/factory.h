#pragma once
#include "micronaut_registry.h"
#include "../kuhul/phase_engine.h"
#include <string>
#include <vector>

namespace khanary::micronaut {

enum class FactoryOp {
    Create,   // Wo phase
    Evolve,   // Yax phase
    List,     // Pop phase
    Read,     // Pop phase
    Edit,     // Sek phase
    Patch,    // Sek phase
    Status,   // Ch'en phase
    Delete,   // Xul phase (requires confidence >= floor)
};

struct FactoryResult {
    bool        ok;
    std::string id;
    std::string message;
};

class Factory {
public:
    Factory(MicronautRegistry& reg, float confidence_floor = 0.7f);

    // Create a new micronaut
    FactoryResult create(const std::string& id,
                         const std::string& description,
                         kuhul::Phase primary_phase,
                         const std::vector<std::string>& capabilities,
                         const std::vector<std::string>& governs);

    // Evolve: add capabilities or governs tokens to existing micronaut
    FactoryResult evolve(const std::string& id,
                         const std::vector<std::string>& new_capabilities,
                         const std::vector<std::string>& new_governs);

    // List all micronaut ids
    std::vector<std::string> list() const;

    // Read a micronaut definition
    const MicronautEntry* read(const std::string& id) const;

    // Edit: replace capabilities
    FactoryResult edit(const std::string& id,
                       const std::vector<std::string>& capabilities);

    // Patch: update confidence
    FactoryResult patch(const std::string& id, float confidence);

    // Status: returns active flag
    FactoryResult status(const std::string& id) const;

    // Delete: only if confidence >= floor
    FactoryResult remove(const std::string& id);

    kuhul::Phase op_phase(FactoryOp op) const;

private:
    MicronautRegistry& reg_;
    float confidence_floor_;
};

} // namespace khanary::micronaut
