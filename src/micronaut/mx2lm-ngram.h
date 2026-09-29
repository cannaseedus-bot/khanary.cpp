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
    uint32_t    count = 0;
    uint32_t    width = 0;
    std::vector<std::string> tokens;
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
