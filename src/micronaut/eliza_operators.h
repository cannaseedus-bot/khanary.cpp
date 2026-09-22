#pragma once
#include "../kuhul/phase_engine.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace khanary::micronaut {

// ELIZA-1: 16 reasoning operator pairs
// Each has Pop-phase detection cues and Sek-phase emission directive
struct ElizaOperator {
    std::string              name;
    std::vector<std::string> pop_governs;    // detection cue tokens (Pop phase)
    std::string              sek_directive;  // emission template (Sek phase)
    kuhul::Phase             detect_phase = kuhul::Phase::Pop;
    kuhul::Phase             emit_phase   = kuhul::Phase::Sek;
};

struct OperatorMatch {
    const ElizaOperator* op = nullptr;
    float                score = 0.0f;
    std::vector<std::string> matched_cues;
};

class ElizaOperatorBank {
public:
    ElizaOperatorBank();

    // Detect which operator matches a token sequence (Pop phase)
    std::optional<OperatorMatch> detect(const std::vector<std::string>& tokens) const;

    // Emit a directive from a matched operator (Sek phase)
    std::string emit(const ElizaOperator& op, const std::string& context = "") const;

    // All 16 operators
    const std::vector<ElizaOperator>& operators() const { return ops_; }

    const ElizaOperator* find(const std::string& name) const;

private:
    std::vector<ElizaOperator> ops_;
    std::unordered_map<std::string, size_t> name_index_;

    void init();
};

} // namespace khanary::micronaut
