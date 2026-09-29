#pragma once
#include "../config.h"
#include <string>
#include <cmath>

namespace khanary::kuhul {

enum class Phase : int {
    Pop  = 0,  // 0
    Wo   = 1,  // π/3
    Yax  = 2,  // 2π/3
    Sek  = 3,  // π
    Chen = 4,  // 4π/3  (Ch'en)
    Xul  = 5,  // 5π/3
};

constexpr int PHASE_COUNT = 6;

// C6 cycle angle (radians) for a Phase
constexpr double phase_angle(Phase p) {
    switch (p) {
        case Phase::Pop:  return PHASE_POP;
        case Phase::Wo:   return PHASE_WO;
        case Phase::Yax:  return PHASE_YAX;
        case Phase::Sek:  return PHASE_SEK;
        case Phase::Chen: return PHASE_CHEN;
        case Phase::Xul:  return PHASE_XUL;
    }
    return 0.0;
}

// arc_weight(from, to) = cos(angle(to) - angle(from))
inline double arc_weight(Phase from, Phase to) {
    return std::cos(phase_angle(to) - phase_angle(from));
}

// Advance one step in G_phase = C6
constexpr Phase next_phase(Phase p) {
    return Phase(((int)p + 1) % PHASE_COUNT);
}

constexpr Phase prev_phase(Phase p) {
    return Phase(((int)p + PHASE_COUNT - 1) % PHASE_COUNT);
}

const char* phase_name(Phase p);
Phase phase_from_name(const std::string& name);
bool  phase_from_name(const std::string& name, Phase& out);

// Canonical semantic-atom -> phase binding (execution authority).
// Returns false for unknown atoms.
bool canonical_phase_for_semantic_atom(const std::string& atom, Phase& out);
std::string canonical_phase_name_for_semantic_atom(const std::string& atom);

// Phase difference in units of π/3 steps (mod 6, unsigned)
constexpr int phase_distance(Phase from, Phase to) {
    return ((int)to - (int)from + PHASE_COUNT) % PHASE_COUNT;
}

class PhaseEngine {
public:
    PhaseEngine();

    Phase current() const { return current_; }
    void  set(Phase p)    { current_ = p; }

    // Step to next phase in C6 — returns new phase
    Phase advance();

    // Compute effective arc weight for a fold from → to
    double effective_weight(Phase from, Phase to) const;

    // True if from→to is a vertical (Δθ = π/3) step
    static bool is_vertical(Phase from, Phase to);

    // True if from==to (horizontal fold, Δθ=0)
    static bool is_horizontal(Phase from, Phase to);

private:
    Phase current_ = Phase::Pop;
};

} // namespace khanary::kuhul
