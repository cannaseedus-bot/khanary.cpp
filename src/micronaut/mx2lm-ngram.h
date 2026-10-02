#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace khanary::micronaut {

using NGramFreq = std::unordered_map<std::string, uint32_t>;

struct Mx2LmNGramModel {
    std::vector<std::string> tokens;
    NGramFreq                bigrams;
    NGramFreq                trigrams;
    NGramFreq                quadgrams;
};

struct Mx2LmGramTaxonomy {
    NGramFreq unigrams;
    NGramFreq bigrams;
    NGramFreq trigrams;
    NGramFreq quadgrams;
    NGramFreq coarse_grams;
    NGramFreq function_grams;
    NGramFreq tool_grams;
};

struct Mx2LmTypedGram {
    std::string gram;
    std::string gram_class;  // unigram|bigram|trigram|quadgram|coarse_gram|function_gram|tool_gram
    std::string role;        // semantic/control/operator/tool binding role
    std::string phase;       // Pop|Wo|Yax|Sek phase interpretation for this gram class
    uint32_t    count = 0;
    uint32_t    width = 0;
    std::vector<std::string> tokens;
};

struct Mx2LmGramContainer {
    std::string namespace_id = "Mx2LM";
    std::string key = "mx2lm:gram_container:v1";
    std::string authority = "mx2lm:ngram-build";
    std::string phase = "Wo";
    std::vector<Mx2LmTypedGram> grams;

    uint32_t total_observations = 0;
    uint32_t unigram_count = 0;
    uint32_t bigram_count = 0;
    uint32_t trigram_count = 0;
    uint32_t quadgram_count = 0;
    uint32_t coarse_gram_count = 0;
    uint32_t function_gram_count = 0;
    uint32_t tool_gram_count = 0;
};

struct Mx2LmPatternEvent {
    std::string source_track = "Mx2LM";
    std::string source_phase = "Wo";
    std::string target_namespace = "Mx2LM";
    std::string surface;
    std::vector<std::string> roles;
    std::string gram_container_key = "mx2lm:gram_container:v1";
    std::string outcome_kind = "pattern_observed";
    std::string outcome_status = "observed";
    std::string card;
    bool accepted = true;
};

struct Mx2LmPatternContainer {
    std::string namespace_id = "Mx2LM";
    std::string key = "mx2lm:pattern_container:v1";
    std::string authority = "mx2lm:pattern-learn";
    std::string phase = "Xul";
    std::vector<Mx2LmPatternEvent> events;
    uint32_t event_count = 0;
};

struct Mx2LmVectorProjection {
    std::string key;
    std::string source_key;
    std::string source_track;
    std::string source_phase;
    std::string projection_track = "VECTOR-µ";
    std::string basis = "deterministic_semantic_hash_projection";
    float       x = 0.0f;
    float       y = 0.0f;
    float       z = 0.0f;
    float       norm = 0.0f;
    bool        semantic_authority = false;
};

struct Mx2LmVectorProjectionContainer {
    std::string namespace_id = "VECTOR-µ";
    std::string key = "vector:projection_container:v1";
    std::string authority = "vector:projection-only";
    std::string phase = "Wo";
    std::string source_semantic_container = "mx2lm:pattern_container:v1";
    std::string projection_id = "vector:projection:bootstrap:v1:g0";
    std::string learner_version = "bootstrap.deterministic_hash.v1";
    std::string topology_version = "runtime.semantic_containers";
    std::string relation_model = "relationisa_conditioned_geometry";
    std::string alignment = "bootstrap_no_cross_generation_alignment";
    std::string similarity_scope = "semantic_similarity_subset_of_relatedness";
    uint32_t    generation = 0;
    uint32_t    dimensions = 3;
    std::vector<Mx2LmVectorProjection> projections;
};

struct Mx2LmControlBinding {
    std::string token;          // observed lexical token ("if", "because", ...)
    std::string operator_name;  // deterministic operator authority (e.g., "CAUSE")
    std::string book_key;       // semantic/operator book anchor (e.g., "ELIZA-1:CAUSE")
    std::string detect_phase;   // Pop-phase cue binding
    std::string emit_phase;     // Sek/Wo/... execution binding
    std::string semantic_role;  // human-readable structural role
    std::string xcfe_op;        // XCFE lowering target ("@if", "@then", ...)
    uint32_t    arity = 1;      // operands the control word consumes
};

struct Mx2LmSemanticNode {
    std::string id;             // token identity
    std::string kind;           // "semantic_node" | "control_node"
    std::string meaning_key;    // semantic meaning anchor for this node
    std::string operator_name;  // control operator when kind=control_node
    std::string book_key;       // operator Book binding when kind=control_node
};

struct Mx2LmRelation {
    std::string from;
    std::string relation;
    std::string to;
    float       association_weight = 1.0f;            // [0,1] pattern strength
    std::string weight_kind = "pattern_strength";     // never semantic meaning
};

struct Mx2LmStructuredProjection {
    std::vector<Mx2LmSemanticNode>   nodes;
    std::vector<Mx2LmControlBinding> control_bindings;
    std::vector<Mx2LmRelation>       relations;
    std::string                      clause;
    std::string                      sentence;
};

class Mx2LmNGram {
public:
    static std::vector<std::string> tokenize(std::string_view text);
    static Mx2LmNGramModel build(std::string_view text);
    static std::vector<std::pair<std::string, uint32_t>> top(
        const NGramFreq& freq, size_t top_k);
    static Mx2LmGramTaxonomy classify_gram_taxonomy(
        const std::vector<std::string>& tokens);
    static Mx2LmGramTaxonomy classify_gram_taxonomy(std::string_view text);
    static std::vector<Mx2LmTypedGram> typed_grams(
        const Mx2LmGramTaxonomy& taxonomy,
        size_t top_k_per_class = 8);
    static Mx2LmGramContainer compile_gram_container(
        const Mx2LmGramTaxonomy& taxonomy,
        size_t top_k_per_class = 0);
    static Mx2LmGramContainer compile_gram_container(
        std::string_view text,
        size_t top_k_per_class = 0);
    static Mx2LmPatternContainer compile_pattern_container(
        std::string_view surface,
        const Mx2LmGramContainer& gram_container,
        const std::string& source_track = "Mx2LM",
        const std::string& source_phase = "Wo",
        const std::string& outcome_kind = "pattern_observed",
        const std::string& card = "",
        bool accepted = true);
    static Mx2LmVectorProjectionContainer compile_vector_projection_container(
        const Mx2LmGramContainer& gram_container,
        const Mx2LmPatternContainer& pattern_container);

    // Resolve control words to deterministic operator authority.
    // This keeps control semantics anchored in XCFE/K'UHUL operators,
    // rather than purely statistical n-gram co-occurrence.
    static std::vector<Mx2LmControlBinding> resolve_control_bindings(
        const std::vector<std::string>& tokens);

    // Infer relational edges from control pivots and associative adjacency.
    static std::vector<Mx2LmRelation> infer_relations(
        const std::vector<std::string>& tokens);

    // Build a structured projection hierarchy (atoms/grams/relations + control).
    static Mx2LmStructuredProjection compose(
        const std::vector<std::string>& tokens);
};

} // namespace khanary::micronaut
