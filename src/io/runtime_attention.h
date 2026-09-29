#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace khanary::io {

struct SemanticVector3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool valid = false;
};

struct RuntimeAttentionInputs {
    std::string miss_kind;
    std::vector<std::string> required_folds;
    std::string query_lower;
    bool relevance_cold_start = false;
    std::size_t missing_context_count = 0;

    float relevance_score = 0.0f;
    float bow_score = 0.0f;
    float distributional_score = 0.0f;
    float confidence_score = 0.0f;
    float static_weight = 1.0f;
    float health_score = 1.0f;
};

struct RuntimeAttentionCandidate {
    std::string id;
    std::string kind;
    std::string slot;
    double R = 0.0;
    double U = 0.0;
    double D = 0.0;
    double P = 0.0;
    double score = 0.0;
};

struct RuntimeAttentionResult {
    double R = 0.0;
    double U = 0.0;
    double D = 0.0;
    double P = 0.0;
    double lexical_prior = 0.0;
    double P_route = 0.0;
    double P_query = 0.0;
    double research_pressure = 0.0;
    double score = 0.0;
    double admission_threshold = 0.35;
    bool admitted = false;
    bool relevance_cold_start = false;

    RuntimeAttentionCandidate focus;
    std::vector<RuntimeAttentionCandidate> candidates;
};

struct WcrSelectionInputs {
    double static_weight = 1.0; // W
    double confidence = 0.5;    // C
    double relevance = 0.5;     // R
};

std::string fold_to_attention_kind(const std::string& fold);
std::string kind_to_attention_slot(const std::string& kind);
RuntimeAttentionResult compute_runtime_attention(const RuntimeAttentionInputs& in);
double compute_wcr_score(const WcrSelectionInputs& in);
double compute_vector_relevance(const SemanticVector3& query_vector, const SemanticVector3& key_vector);
double compute_wcr_qkv_score(const SemanticVector3& query_vector,
                             const SemanticVector3& key_vector,
                             double static_weight,
                             double confidence);
// Deterministic cold-start vector fallback used only when semantic key vectors
// are unavailable from declared/runtime metadata.
SemanticVector3 derive_vector_from_tokens(const std::vector<std::string>& tokens_lower);
// Convenience wrapper around derive_vector_from_tokens for raw text fallback.
SemanticVector3 derive_vector_from_text(const std::string& text);

} // namespace khanary::io
