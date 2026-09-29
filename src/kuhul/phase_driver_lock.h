#pragma once
#include "phase_engine.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace khanary::kuhul {

struct PhaseDriverEntryStatus {
    Phase       phase = Phase::Pop;
    std::string phase_name;
    std::string dll_name;
    std::string dll_path;
    std::string expected_sha256;
    std::string actual_sha256;
    bool        exists = false;
    bool        hash_declared = false;
    bool        hash_match = false;
    bool        verified = false;
    bool        required = true;
    std::string error;
};

class PhaseDriverLock {
public:
    // Loads lock policy from fold runtime config and hash anchors from runtime metadata.
    // Returns false only for malformed lock config or strict enforce-mode failures.
    bool load(const std::string& fold_runtime_path,
              const std::string& runtime_metadata_path,
              std::string* err = nullptr);

    bool enabled() const { return enabled_; }
    bool strict()  const { return strict_;  }
    bool ready()   const { return ready_;   }

    // True if the phase is permitted to execute under current lock policy.
    bool allow_phase(Phase p, std::string* err = nullptr) const;

    const std::vector<PhaseDriverEntryStatus>& entries() const { return entries_; }
    std::string summary() const;

private:
    bool enabled_ = false;
    bool strict_ = false;
    bool ready_ = true;
    std::vector<PhaseDriverEntryStatus> entries_;
    std::unordered_map<int, size_t> phase_index_;
};

} // namespace khanary::kuhul
