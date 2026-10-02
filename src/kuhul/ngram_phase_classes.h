#pragma once
#include "phase_engine.h"
#include "gram_store.h"
#include "../micronaut/mx2lm-ngram.h"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace khanary::kuhul {

// ─── Phase result types ────────────────────────────────────────────────────

struct PopNGramResult {
    std::string                 surface;
    std::vector<std::string>    tokens;
    micronaut::Mx2LmNGramModel  model;
};

struct WoNGramResult {
    PopNGramResult                         pop;
    micronaut::Mx2LmGramTaxonomy           taxonomy;
    std::vector<micronaut::Mx2LmTypedGram> typed;
    micronaut::Mx2LmGramContainer          gram_container;
    micronaut::Mx2LmPatternContainer       pattern_container;
    micronaut::Mx2LmVectorProjectionContainer vector_projection_container;
};

struct YaxNGramResult {
    WoNGramResult                               wo;
    std::vector<micronaut::Mx2LmTypedGram>      admitted;       // activated subset = resolved sense
    std::vector<micronaut::Mx2LmControlBinding> control_bindings;
    std::string                                 resolved_sense; // dominant gram_class after gating
};

struct SekNGramResult {
    YaxNGramResult                               yax;
    std::vector<micronaut::Mx2LmRelation>        relations;
    std::vector<std::pair<std::string, float>>   ranked; // gram text → similarity score
};

enum class ChenNGramStatus { PASS, FAIL, UNKNOWN };

// Miss kinds matching fold.runtime.json#miss_deficit_contract.miss_kind_enum.
// MISS != WRONG — it is an epistemic deficit: RequiredContext(Q) - ResolvedContext(F_Q).
enum class ChenMissKind {
    evidence,     // τ too narrow — no grams passed the Yax gate
    semantic,     // grams admitted but no relational/control structure
    topology,     // relation endpoints disconnect from admitted gram tokens
    factual,      // control binding present but book_key unanchored
    mathematical, // control op is a math/KLSL operator
    capability,   // control op is a non-standard capability expression
    relation,     // inferred relation has empty or unrecognized verb
    memory,       // grams admitted but all scores below noise floor (history not replayed)
    none          // no miss — status is PASS or FAIL
};

const char* chen_miss_kind_name(ChenMissKind k);
ChenMissKind chen_miss_kind_from_name(const std::string& name);

// Advisor delegation: fired when Ch'en detects a MISS.
// The advisor (e.g. web-research completion callback) returns acquired evidence.
struct ChenMissContext {
    ChenMissKind             miss_kind     = ChenMissKind::none;
    std::string              miss_kind_str;
    float                    research_pressure = 0.0f;  // ρ = D * R * U ∈ [0,1]
    std::vector<std::string> missing_keys;              // absent context terms
    std::string              topology_hash;
    std::string              resolved_sense;
};

struct ChenAdvisorResponse {
    bool                     evidence_obtained = false;
    std::vector<std::string> acquired_keys;   // new evidence terms from research
    std::string              source;          // "web_research" | "gram_store" | "book_fold" | ...
    float                    confidence = 0.0f; // [0,1] quality of acquired evidence
};

using ChenAdvisorFn = std::function<ChenAdvisorResponse(const ChenMissContext&)>;

struct ChenNGramCertificate {
    ChenNGramStatus      status          = ChenNGramStatus::UNKNOWN;
    std::string          assertion_class;   // STRUCTURAL | SEMANTIC | STATE | AUTHORITY
    std::vector<std::string> evidence_keys;
    std::string          topology_hash;
    // Miss fields — populated when status == UNKNOWN
    ChenMissKind         miss_kind_enum  = ChenMissKind::none;
    std::string          miss_kind;         // string form of miss_kind_enum
    float                research_pressure = 0.0f; // ρ_research = D * R * U
    // Advisor fields — populated when an advisor was consulted
    bool                 advisor_consulted = false;
    std::string          advisor_source;
    std::vector<std::string> advisor_keys;
    float                advisor_confidence = 0.0f;
};

struct ChenNGramResult {
    SekNGramResult       sek;
    ChenNGramCertificate certificate;
};

struct XulNGramResult {
    ChenNGramResult      chen;
    bool                 committed = false;
    std::vector<std::string> written_keys;
    std::string          rejection_reason;
};

// ─── Per-phase classes ────────────────────────────────────────────────────

// Pop: ingest raw text → tokens + ngram model
class PopNGram {
public:
    static constexpr Phase kPhase = Phase::Pop;
    PopNGramResult process(const std::string& text) const;
};

// Wo: declare semantic space → gram taxonomy + typed grams (top_k per class)
class WoNGram {
public:
    static constexpr Phase kPhase = Phase::Wo;
    WoNGramResult process(const PopNGramResult& pop, size_t top_k = 8) const;
};

// Yax: admission gate — neighborhood ∩ typed grams = admitted (WSD)
//   neighborhood: context strings from B_τ[h,*] fold projection
class YaxNGram {
public:
    static constexpr Phase kPhase = Phase::Yax;
    YaxNGramResult process(
        const WoNGramResult& wo,
        const std::vector<std::string>& neighborhood) const;
};

// Sek: similarity scoring + relational edge inference over admitted grams
class SekNGram {
public:
    static constexpr Phase kPhase = Phase::Sek;
    SekNGramResult process(const YaxNGramResult& yax) const;
};

// Ch'en: verify gram evidence → assertion certificate (PASS/FAIL/UNKNOWN).
//
// When status == UNKNOWN, a ChenMissContext is built and the miss kind is
// classified from the evidence topology (8 kinds, matching fold.runtime.json).
// Research pressure ρ = D * R * U is computed.
//
// If an advisor function is registered (set_advisor), it is called on any MISS.
// The advisor may return acquired evidence (e.g. after web research completes).
// If advisor.confidence >= 0.5 and evidence_obtained, status upgrades to PASS
// under assertion_class AUTHORITY.
class ChenNGram {
public:
    static constexpr Phase kPhase = Phase::Chen;

    ChenNGramResult process(
        const SekNGramResult& sek,
        const std::string& topology_hash = "") const;

    void set_advisor(ChenAdvisorFn fn) { advisor_ = std::move(fn); }
    bool has_advisor() const           { return advisor_.has_value(); }

private:
    std::optional<ChenAdvisorFn> advisor_;

    // Classify the miss kind from what evidence is present/absent.
    ChenMissKind classify_miss(const SekNGramResult& sek) const;

    // ρ_research = D * R * U  (each component in [0,1])
    float compute_research_pressure(const SekNGramResult& sek, ChenMissKind k) const;

    // Build the ChenMissContext for the advisor call.
    ChenMissContext build_miss_context(
        const SekNGramResult& sek,
        ChenMissKind k,
        const std::string& topology_hash) const;
};

// Xul: compare-and-commit — writes verified grams to GramStore
//   T_Xul commits only when Verified(V) ∧ Authorized(V) ∧ Fresh(V)
class XulNGram {
public:
    static constexpr Phase kPhase = Phase::Xul;
    explicit XulNGram(GramStore& store) : store_(store) {}
    XulNGramResult process(const ChenNGramResult& chen) const;
private:
    GramStore& store_;
};

// ─── Pipeline ─────────────────────────────────────────────────────────────

struct NGramPipelineResult {
    PopNGramResult  pop;
    WoNGramResult   wo;
    YaxNGramResult  yax;
    SekNGramResult  sek;
    ChenNGramResult chen;
    XulNGramResult  xul;
};

class NGramPipeline {
public:
    explicit NGramPipeline(GramStore& store) : xul_(store) {}

    // Run all six phases in order.
    // fold_neighborhood: B_τ[h,*] context strings for Yax admission gate.
    NGramPipelineResult run(
        const std::string& text,
        const std::vector<std::string>& fold_neighborhood = {},
        const std::string& source_track = "Mx2LM",
        const std::string& source_phase = "Wo",
        const std::string& outcome_kind = "pattern_observed",
        const std::string& card = "",
        bool accepted = true) const;

    // Inject an advisor into Ch'en for miss-delegation.
    void set_chen_advisor(ChenAdvisorFn fn) { chen_.set_advisor(std::move(fn)); }

private:
    PopNGram  pop_;
    WoNGram   wo_;
    YaxNGram  yax_;
    SekNGram  sek_;
    ChenNGram chen_;
    XulNGram  xul_;
};

} // namespace khanary::kuhul
