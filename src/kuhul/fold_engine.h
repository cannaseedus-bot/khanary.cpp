#pragma once
#include "phase_engine.h"
#include <string>
#include <functional>

namespace khanary::kuhul {

// FoldResult captures what a fold produces
struct FoldResult {
    std::string node_a;
    std::string node_b;
    Phase       phase;
    double      weight;      // arc weight applied
    bool        authorized;  // XCFE authorization (vertical only)
    std::string error;
};

using XCFEAuthFn = std::function<bool(Phase from, Phase to, const std::string& node)>;

class FoldEngine {
public:
    explicit FoldEngine(PhaseEngine& pe);

    // Install an XCFE authorization callback
    // Vertical folds require authorization; horizontal folds do not
    void set_xcfe_auth(XCFEAuthFn fn);

    // Horizontal fold: same-phase traversal, Δθ=0, weight=+1.0
    // Connects node_a ↔ node_b at the given phase
    FoldResult horizontal_fold(
        const std::string& node_a,
        const std::string& node_b,
        Phase phase);

    // Vertical fold: phase transition, Δθ=π/3
    // Requires XCFE authorization to cross the phase boundary
    FoldResult vertical_fold(
        const std::string& node,
        Phase from_phase,
        Phase to_phase);

    // Effective weight for any fold pair
    double effective_weight(Phase from, Phase to) const;

    // Semantic attention gate aligned with runtime W·C·R scoring.
    double wcr_weighted_score(double static_weight,
                              double confidence,
                              double relevance) const;

    PhaseEngine& phase_engine() { return pe_; }

private:
    PhaseEngine& pe_;
    XCFEAuthFn   auth_;
};

} // namespace khanary::kuhul
