#include "conformance.h"
#include "../io/runtime_attention.h"

#include <cmath>
#include <iostream>

using namespace khanary::io;

namespace {

bool expect_true(const std::string& label, bool value) {
    if (value) return true;
    std::cerr << "[FAIL] " << label << ": expected true\n";
    return false;
}

bool expect_near(const std::string& label, double got, double want, double eps = 1e-9) {
    if (std::fabs(got - want) <= eps) return true;
    std::cerr << "[FAIL] " << label << ": expected " << want << " got " << got << "\n";
    return false;
}

} // namespace

static int acceptance_main() {
    {
        RuntimeAttentionInputs in;
        in.miss_kind = "semantic";
        in.required_folds = {"EVIDENCE"};
        in.relevance_cold_start = true;
        in.missing_context_count = 1;
        in.relevance_score = 0.0f;
        in.bow_score = 0.0f;
        in.distributional_score = 0.0f;
        in.confidence_score = 1.0f;
        in.static_weight = 1.0f;
        in.health_score = 1.0f;

        const auto out = compute_runtime_attention(in);
        if (!expect_near("cold-start R floor", out.R, 0.30)) return 1;
        if (!expect_near("unresolvedness floor", out.U, 0.15)) return 1;
        if (!expect_near("query priority default", out.P_query, 0.50)) return 1;
        if (!expect_true("cold-start marked", out.relevance_cold_start)) return 1;
        if (!expect_true("below admission threshold", !out.admitted)) return 1;
    }

    {
        RuntimeAttentionInputs in;
        in.miss_kind = "semantic";
        in.required_folds = {"EVIDENCE", "CAPABILITY"};
        in.query_lower = "this is urgent and a blocker right now";
        in.relevance_cold_start = false;
        in.missing_context_count = 1;
        in.relevance_score = 0.8f;
        in.bow_score = 0.7f;
        in.distributional_score = 0.8f;
        in.confidence_score = 0.2f;
        in.static_weight = 1.0f;
        in.health_score = 1.0f;

        const auto out = compute_runtime_attention(in);
        if (!expect_near("urgent query priority", out.P_query, 0.90)) return 1;
        // CAPABILITY and EVIDENCE are non-primary when miss_kind=semantic; tie falls back to node id asc.
        if (!expect_true("lexicographic tie-break",
                         out.focus.id == "fold:CAPABILITY")) return 1;
    }

    {
        RuntimeAttentionInputs in;
        in.miss_kind = "evidence";
        in.required_folds = {"EVIDENCE"};
        in.query_lower = "urgent";
        in.relevance_cold_start = false;
        in.missing_context_count = 1;
        in.relevance_score = 1.0f;
        in.bow_score = 1.0f;
        in.distributional_score = 1.0f;
        in.confidence_score = 0.0f;
        in.static_weight = 1.0f;
        in.health_score = 1.0f;

        const auto out = compute_runtime_attention(in);
        if (!expect_true("admission threshold reached", out.admitted)) return 1;
        if (!expect_true("attention score above threshold", out.score >= out.admission_threshold)) return 1;
    }

    std::cout << "Runtime attention tests passed.\n";
    return 0;
}

int main() { return khanary::test::legacy("attention", [] { return acceptance_main(); }); }
