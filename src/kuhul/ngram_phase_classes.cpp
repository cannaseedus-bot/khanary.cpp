#include "ngram_phase_classes.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace khanary::kuhul {

// ─── Miss kind helpers ────────────────────────────────────────────────────

const char* chen_miss_kind_name(ChenMissKind k) {
    switch (k) {
        case ChenMissKind::evidence:     return "evidence";
        case ChenMissKind::semantic:     return "semantic";
        case ChenMissKind::topology:     return "topology";
        case ChenMissKind::factual:      return "factual";
        case ChenMissKind::mathematical: return "mathematical";
        case ChenMissKind::capability:   return "capability";
        case ChenMissKind::relation:     return "relation";
        case ChenMissKind::memory:       return "memory";
        case ChenMissKind::none:         return "none";
    }
    return "none";
}

ChenMissKind chen_miss_kind_from_name(const std::string& name) {
    if (name == "evidence")     return ChenMissKind::evidence;
    if (name == "semantic")     return ChenMissKind::semantic;
    if (name == "topology")     return ChenMissKind::topology;
    if (name == "factual")      return ChenMissKind::factual;
    if (name == "mathematical") return ChenMissKind::mathematical;
    if (name == "capability")   return ChenMissKind::capability;
    if (name == "relation")     return ChenMissKind::relation;
    if (name == "memory")       return ChenMissKind::memory;
    return ChenMissKind::none;
}

// ─── Pop ──────────────────────────────────────────────────────────────────

PopNGramResult PopNGram::process(const std::string& text) const {
    PopNGramResult r;
    r.tokens = micronaut::Mx2LmNGram::tokenize(text);
    r.model  = micronaut::Mx2LmNGram::build(text);
    return r;
}

// ─── Wo ───────────────────────────────────────────────────────────────────

WoNGramResult WoNGram::process(const PopNGramResult& pop, size_t top_k) const {
    WoNGramResult r;
    r.pop      = pop;
    r.taxonomy = micronaut::Mx2LmNGram::classify_gram_taxonomy(pop.tokens);
    r.typed    = micronaut::Mx2LmNGram::typed_grams(r.taxonomy, top_k);
    return r;
}

// ─── Yax ──────────────────────────────────────────────────────────────────
//
// Admission: a typed gram is admitted when at least one of its tokens
// overlaps with the fold neighborhood (B_τ[h,*]).  The intersection is the
// activated subset — that IS the resolved word sense.

YaxNGramResult YaxNGram::process(
    const WoNGramResult& wo,
    const std::vector<std::string>& neighborhood) const
{
    YaxNGramResult r;
    r.wo = wo;

    const std::unordered_set<std::string> nb(neighborhood.begin(), neighborhood.end());

    for (const auto& tg : wo.typed) {
        bool admitted = false;
        if (nb.empty()) {
            admitted = true; // open-world fallback
        } else {
            for (const auto& tok : tg.tokens) {
                if (nb.count(tok)) { admitted = true; break; }
            }
        }
        if (admitted) r.admitted.push_back(tg);
    }

    r.control_bindings = micronaut::Mx2LmNGram::resolve_control_bindings(wo.pop.tokens);

    std::unordered_map<std::string, uint32_t> class_counts;
    for (const auto& g : r.admitted)
        class_counts[g.gram_class] += g.count;

    if (!class_counts.empty()) {
        r.resolved_sense = std::max_element(
            class_counts.begin(), class_counts.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; }
        )->first;
    }

    return r;
}

// ─── Sek ──────────────────────────────────────────────────────────────────

SekNGramResult SekNGram::process(const YaxNGramResult& yax) const {
    SekNGramResult r;
    r.yax = yax;
    r.relations = micronaut::Mx2LmNGram::infer_relations(yax.wo.pop.tokens);

    std::unordered_set<std::string> relation_tokens;
    for (const auto& rel : r.relations) {
        relation_tokens.insert(rel.from);
        relation_tokens.insert(rel.to);
    }

    for (const auto& g : yax.admitted) {
        float score = 0.0f;
        for (const auto& tok : g.tokens)
            if (relation_tokens.count(tok)) score += 1.0f;
        score = (g.tokens.empty() ? 0.0f : score / float(g.tokens.size()));
        score = 0.6f * score + 0.4f * std::min(1.0f, float(g.count) / 20.0f);
        r.ranked.push_back({g.gram, score});
    }

    std::sort(r.ranked.begin(), r.ranked.end(),
        [](const auto& a, const auto& b){ return a.second > b.second; });

    return r;
}

// ─── Ch'en ────────────────────────────────────────────────────────────────

namespace {

// Known math xcfe_op prefixes
bool is_math_op(const std::string& xcfe_op) {
    static const std::unordered_set<std::string> math_ops = {
        "@mul","@div","@add","@sub","@pow","@sqrt","@mod","@abs",
        "@sum","@mean","@max","@min","@dot","@norm"
    };
    return math_ops.count(xcfe_op) > 0;
}

// Standard XCFE ops (non-math, non-empty = capability)
bool is_standard_xcfe_op(const std::string& xcfe_op) {
    static const std::unordered_set<std::string> standard_ops = {
        "@if","@then","@else","@because","@when","@so","@thus",
        "@but","@and","@or","@not","@while","@until","@for",
        "@delegate","@route","@emit","@check","@assert","@commit"
    };
    return standard_ops.count(xcfe_op) > 0;
}

float ranked_entropy_norm(const std::vector<std::pair<std::string,float>>& ranked) {
    if (ranked.size() <= 1) return 1.0f;
    float total = 0.0f;
    for (const auto& [g, s] : ranked) total += s;
    if (total < 1e-6f) return 1.0f;
    float H = 0.0f;
    for (const auto& [g, s] : ranked) {
        float p = s / total;
        if (p > 1e-9f) H -= p * std::log2f(p);
    }
    float H_max = std::log2f(float(ranked.size()));
    return H_max < 1e-6f ? 0.0f : H / H_max;
}

} // namespace

// Priority order: mathematical > capability > factual > topology > relation
//                > semantic > memory > evidence
ChenMissKind ChenNGram::classify_miss(const SekNGramResult& sek) const {
    const bool has_admitted  = !sek.yax.admitted.empty();
    const bool has_relations = !sek.relations.empty();
    const bool has_control   = !sek.yax.control_bindings.empty();

    // Check control binding quality first (highest specificity)
    for (const auto& cb : sek.yax.control_bindings) {
        if (!cb.xcfe_op.empty() && is_math_op(cb.xcfe_op))
            return ChenMissKind::mathematical;
    }
    for (const auto& cb : sek.yax.control_bindings) {
        if (!cb.xcfe_op.empty() && !is_standard_xcfe_op(cb.xcfe_op) && !is_math_op(cb.xcfe_op))
            return ChenMissKind::capability;
    }
    for (const auto& cb : sek.yax.control_bindings) {
        if (cb.book_key.empty())
            return ChenMissKind::factual;
    }

    // Topology: relations exist but their endpoints don't appear in any admitted gram
    if (has_relations && has_admitted) {
        std::unordered_set<std::string> admitted_tokens;
        for (const auto& g : sek.yax.admitted)
            for (const auto& t : g.tokens) admitted_tokens.insert(t);
        bool any_connected = false;
        for (const auto& rel : sek.relations) {
            if (admitted_tokens.count(rel.from) || admitted_tokens.count(rel.to)) {
                any_connected = true; break;
            }
        }
        if (!any_connected) return ChenMissKind::topology;
    }

    // Relation: inferred relations have empty/blank verb
    for (const auto& rel : sek.relations) {
        if (rel.relation.empty() || rel.relation == "unknown")
            return ChenMissKind::relation;
    }

    // Semantic: admitted grams exist but no structural edges at all
    if (has_admitted && !has_relations && !has_control)
        return ChenMissKind::semantic;

    // Memory: admitted grams present but all scores below noise floor
    if (has_admitted && !sek.ranked.empty()) {
        bool all_below = true;
        for (const auto& [g, s] : sek.ranked)
            if (s >= 0.05f) { all_below = false; break; }
        if (all_below) return ChenMissKind::memory;
    }

    // Evidence: nothing at all
    return ChenMissKind::evidence;
}

float ChenNGram::compute_research_pressure(
    const SekNGramResult& sek, ChenMissKind) const
{
    const size_t typed_count    = sek.yax.wo.typed.size();
    const size_t admitted_count = sek.yax.admitted.size();

    // D = deficit fraction — how much of the typed space was NOT admitted
    float D = (typed_count == 0) ? 1.0f
            : 1.0f - std::min(1.0f, float(admitted_count) / float(typed_count));

    // R = relevance — top ranked score (0 if nothing ranked)
    float R = sek.ranked.empty() ? 0.0f : sek.ranked.front().second;

    // U = uncertainty — normalised entropy of the ranked distribution
    float U = ranked_entropy_norm(sek.ranked);

    return std::min(1.0f, D * R * U);
}

ChenMissContext ChenNGram::build_miss_context(
    const SekNGramResult& sek,
    ChenMissKind k,
    const std::string& topology_hash) const
{
    ChenMissContext ctx;
    ctx.miss_kind       = k;
    ctx.miss_kind_str   = chen_miss_kind_name(k);
    ctx.research_pressure = compute_research_pressure(sek, k);
    ctx.topology_hash   = topology_hash;
    ctx.resolved_sense  = sek.yax.resolved_sense;

    // Populate missing_keys: tokens from admitted grams that have no relation
    std::unordered_set<std::string> relation_tokens;
    for (const auto& rel : sek.relations) {
        relation_tokens.insert(rel.from);
        relation_tokens.insert(rel.to);
    }
    for (const auto& g : sek.yax.admitted) {
        for (const auto& t : g.tokens) {
            if (!relation_tokens.count(t))
                ctx.missing_keys.push_back(t);
        }
    }
    // Deduplicate
    std::sort(ctx.missing_keys.begin(), ctx.missing_keys.end());
    ctx.missing_keys.erase(
        std::unique(ctx.missing_keys.begin(), ctx.missing_keys.end()),
        ctx.missing_keys.end());

    return ctx;
}

ChenNGramResult ChenNGram::process(
    const SekNGramResult& sek,
    const std::string& topology_hash) const
{
    ChenNGramResult r;
    r.sek = sek;

    auto& cert = r.certificate;
    cert.topology_hash = topology_hash;

    const bool has_admitted  = !sek.yax.admitted.empty();
    const bool has_relations = !sek.relations.empty();
    const bool has_control   = !sek.yax.control_bindings.empty();

    // ── Contradiction check ───────────────────────────────────────────────
    bool contradicted = false;
    std::unordered_map<std::string, std::string> token_relation;
    for (const auto& rel : sek.relations) {
        auto it = token_relation.find(rel.from);
        if (it != token_relation.end() && it->second != rel.relation) {
            contradicted = true; break;
        }
        token_relation[rel.from] = rel.relation;
    }
    if (contradicted) {
        cert.status          = ChenNGramStatus::FAIL;
        cert.assertion_class = "SEMANTIC";
        cert.miss_kind_enum  = ChenMissKind::none;
        cert.miss_kind       = "none";
        return r;
    }

    // ── PASS path ─────────────────────────────────────────────────────────
    if (has_admitted) {
        for (size_t i = 0; i < sek.ranked.size() && i < 6; ++i)
            cert.evidence_keys.push_back(sek.ranked[i].first);

        cert.status = ChenNGramStatus::PASS;
        if (has_relations && has_control)
            cert.assertion_class = "SEMANTIC";
        else if (has_relations)
            cert.assertion_class = "STRUCTURAL";
        else if (has_control)
            cert.assertion_class = "AUTHORITY";
        else
            cert.assertion_class = "STRUCTURAL";

        // Still check for memory miss even on a nominal PASS path
        if (!sek.ranked.empty()) {
            bool all_below = true;
            for (const auto& [g, s] : sek.ranked)
                if (s >= 0.05f) { all_below = false; break; }
            if (all_below) {
                // Scores too thin to trust — downgrade to UNKNOWN/memory
                cert.status         = ChenNGramStatus::UNKNOWN;
                cert.assertion_class = "STATE";
                cert.miss_kind_enum = ChenMissKind::memory;
                cert.miss_kind      = "memory";
                cert.research_pressure = compute_research_pressure(sek, ChenMissKind::memory);
            }
        }

        if (cert.status == ChenNGramStatus::PASS)
            return r;
    }

    // ── UNKNOWN / MISS path ────────────────────────────────────────────────
    if (cert.status != ChenNGramStatus::UNKNOWN) {
        cert.status = ChenNGramStatus::UNKNOWN;
        cert.assertion_class = "STATE";
    }

    const ChenMissKind mk = (cert.miss_kind_enum == ChenMissKind::none ||
                              cert.miss_kind_enum == ChenMissKind::memory)
                              ? (has_admitted ? cert.miss_kind_enum : classify_miss(sek))
                              : cert.miss_kind_enum;

    if (cert.miss_kind_enum == ChenMissKind::none)
        cert.miss_kind_enum = classify_miss(sek);
    cert.miss_kind      = chen_miss_kind_name(cert.miss_kind_enum);
    cert.research_pressure = compute_research_pressure(sek, cert.miss_kind_enum);
    (void)mk;

    // ── Advisor delegation ────────────────────────────────────────────────
    if (advisor_.has_value()) {
        const auto ctx = build_miss_context(sek, cert.miss_kind_enum, topology_hash);
        const auto resp = (*advisor_)(ctx);

        cert.advisor_consulted  = true;
        cert.advisor_source     = resp.source;
        cert.advisor_keys       = resp.acquired_keys;
        cert.advisor_confidence = resp.confidence;

        // Upgrade to PASS under AUTHORITY if advisor returned confident evidence
        if (resp.evidence_obtained && resp.confidence >= 0.5f) {
            cert.status          = ChenNGramStatus::PASS;
            cert.assertion_class = "AUTHORITY";
            for (const auto& k : resp.acquired_keys)
                cert.evidence_keys.push_back(k);
        }
    }

    return r;
}

// ─── Xul ──────────────────────────────────────────────────────────────────

XulNGramResult XulNGram::process(const ChenNGramResult& chen) const {
    XulNGramResult r;
    r.chen = chen;

    const auto& cert = chen.certificate;

    if (cert.status != ChenNGramStatus::PASS) {
        r.committed = false;
        r.rejection_reason = (cert.status == ChenNGramStatus::FAIL)
            ? "ChenNGram FAIL"
            : "ChenNGram UNKNOWN: " + cert.miss_kind +
              (cert.advisor_consulted ? " (advisor consulted, insufficient evidence)" : "");
        return r;
    }

    const auto& sek = chen.sek;
    for (const auto& [gram_text, score] : sek.ranked) {
        if (score < 0.05f) continue;
        const std::string key = "ngram:" + gram_text;
        store_.gramWrite(key, gram_text, "Xul",
            cert.assertion_class == "AUTHORITY" ? "control" : "similarity",
            "ngram_phase_pipeline");
        r.written_keys.push_back(key);
    }

    // Also commit advisor-acquired keys (source is the external authority)
    for (const auto& ak : cert.advisor_keys) {
        store_.gramWrite(ak, ak, "Xul", "authority", cert.advisor_source);
        r.written_keys.push_back(ak);
    }

    for (const auto& cb : sek.yax.control_bindings) {
        const std::string key = "ctrl:" + cb.token;
        store_.gramWrite(key, cb.operator_name, "Xul", "control", cb.book_key);
        r.written_keys.push_back(key);
    }

    r.committed = true;
    return r;
}

// ─── NGramPipeline ────────────────────────────────────────────────────────

NGramPipelineResult NGramPipeline::run(
    const std::string& text,
    const std::vector<std::string>& fold_neighborhood) const
{
    NGramPipelineResult res;
    res.pop  = pop_.process(text);
    res.wo   = wo_.process(res.pop);
    res.yax  = yax_.process(res.wo, fold_neighborhood);
    res.sek  = sek_.process(res.yax);
    res.chen = chen_.process(res.sek);
    res.xul  = xul_.process(res.chen);
    return res;
}

} // namespace khanary::kuhul
