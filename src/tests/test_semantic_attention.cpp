#include "conformance.h"
#include "inference/semantic_attention.h"
#include "kuhul/gram_store.h"
#include "gpu/d3d11_device.h"
#include "gpu/gram_gpu_scorer.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

using namespace khanary;

namespace {

bool finite_nonnegative(float v) {
    return std::isfinite(v) && v >= 0.0f;
}

} // namespace

int main() {
    return test::run("semantic-attention", [](test::Suite& suite) {
        kuhul::GramStore store;
        store.gramWrite("prompt:test:trigram:1",
                        "trigram|2|alpha beta gamma",
                        "Wo", "context_association_evidence", "definitions_and_relations");
        store.gramWrite("prompt:test:bigram:2",
                        "bigram|1|semantic attention",
                        "Wo", "context_association_evidence", "definitions_and_relations");
        store.gramWrite("prompt:test:unigram:1",
                        "unigram|4|alpha",
                        "Pop", "semantic_address_layer", "definitions_and_relations");
        store.gramWrite("prompt:test:file_ref:1",
                        "C:\\tmp\\not-a-semantic-candidate.txt",
                        "Pop", "predication_sequence", "definitions_and_relations");

        const auto candidates = inference::collect_prompt_attention_candidates(
            store, "prompt:test", 8);
        suite.check("candidate_count_excludes_file_rows", candidates.size() == 3);
        suite.check("trigram_ranked_first",
                    !candidates.empty() && candidates[0].key == "prompt:test:trigram:1");
        suite.check("candidate_relevance_bounded",
                    std::all_of(candidates.begin(), candidates.end(), [](const auto& c) {
                        return c.relevance >= 0.0f && c.relevance <= 1.0f;
                    }));

        const auto v1 = inference::prompt_attention_vector_for("bigram", "semantic attention");
        const auto v2 = inference::prompt_attention_vector_for("bigram", "semantic attention");
        const float n1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
        suite.check("prompt_vector_deterministic", v1 == v2);
        suite.check("prompt_vector_unit_norm", std::fabs(n1 - 1.0f) <= 1e-5f);

        const auto overrides = inference::prompt_attention_vector_overrides(candidates);
        suite.check("override_map_matches_candidates", overrides.size() == candidates.size());

        gpu::D3D11Device device;
        if (!device.init("auto")) {
            suite.check("gpu_device_unavailable_skip", true);
            return;
        }

        gpu::GramGpuScorer scorer(device);
        const bool ready = scorer.init();
        suite.check("gpu_scorer_init_when_device_available", ready);
        if (!ready) return;

        std::vector<std::string> keys;
        std::vector<float> relevance;
        for (const auto& candidate : candidates) {
            keys.push_back(candidate.key);
            relevance.push_back(candidate.relevance);
        }
        const auto boost = gpu::LorentzBoostRows::make_pure(
            0.25f, {1.0f, 0.0f, 0.0f});
        const auto scores = scorer.score_batch(
            store, keys, relevance,
            0.0f,
            0.5f,
            0.35f,
            nullptr,
            0.0f,
            0.0f,
            &boost,
            &overrides);
        suite.check("gpu_score_batch_dispatched", scores.size() == candidates.size());
        suite.check("gpu_scores_finite_nonnegative",
                    std::all_of(scores.begin(), scores.end(), [](const auto& score) {
                        return finite_nonnegative(score.semantic_score) &&
                               finite_nonnegative(score.base_score);
                    }));
        suite.check("gpu_zero_threshold_admits_legal_rows",
                    std::all_of(scores.begin(), scores.end(), [](const auto& score) {
                        return score.admitted;
                    }));
    });
}
