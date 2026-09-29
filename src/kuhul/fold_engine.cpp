#include "fold_engine.h"
#include <algorithm>

namespace khanary::kuhul {

FoldEngine::FoldEngine(PhaseEngine& pe) : pe_(pe) {}

void FoldEngine::set_xcfe_auth(XCFEAuthFn fn) {
    auth_ = std::move(fn);
}

double FoldEngine::effective_weight(Phase from, Phase to) const {
    return arc_weight(from, to);
}

double FoldEngine::wcr_weighted_score(double static_weight,
                                      double confidence,
                                      double relevance) const {
    const double W = std::clamp(static_weight, 0.0, 1.0);
    const double C = std::clamp(confidence, 0.0, 1.0);
    const double R = std::clamp(relevance, 0.0, 1.0);
    return std::clamp(W * C * R, 0.0, 1.0);
}

FoldResult FoldEngine::horizontal_fold(
    const std::string& node_a,
    const std::string& node_b,
    Phase phase)
{
    // Horizontal: Δθ = 0, weight = cos(0) = +1.0
    // No XCFE authorization required — same-phase traversal is free
    return FoldResult{
        .node_a     = node_a,
        .node_b     = node_b,
        .phase      = phase,
        .weight     = 1.0,
        .authorized = true,
        .error      = {},
    };
}

FoldResult FoldEngine::vertical_fold(
    const std::string& node,
    Phase from_phase,
    Phase to_phase)
{
    // Vertical fold requires exactly one phase step (primitive edge)
    int dist = phase_distance(from_phase, to_phase);
    if (dist != 1) {
        return FoldResult{
            .node_a     = node,
            .node_b     = node,
            .phase      = from_phase,
            .weight     = 0.0,
            .authorized = false,
            .error      = "vertical_fold: non-primitive step (dist=" + std::to_string(dist) + ")",
        };
    }

    // XCFE authorization gate
    bool ok = true;
    if (auth_) ok = auth_(from_phase, to_phase, node);

    double w = arc_weight(from_phase, to_phase);
    return FoldResult{
        .node_a     = node,
        .node_b     = node,
        .phase      = to_phase,
        .weight     = w,
        .authorized = ok,
        .error      = ok ? std::string{} : "XCFE authorization denied for phase transition " +
                          std::string(phase_name(from_phase)) + " → " + phase_name(to_phase),
    };
}

} // namespace khanary::kuhul
