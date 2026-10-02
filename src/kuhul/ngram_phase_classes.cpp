#include "ngram_phase_classes.h"
#include "../util/json.h"
#include <algorithm>
#include <cctype>
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

namespace {

util::JsonArray string_array_json(const std::vector<std::string>& values) {
    util::JsonArray arr;
    arr.reserve(values.size());
    for (const auto& value : values)
        arr.push_back(util::json_string(value));
    return arr;
}

util::JsonObject gram_channel_count_json(const micronaut::Mx2LmGramContainer& c) {
    util::JsonObject counts;
    counts["unigram"] = util::json_number(static_cast<double>(c.unigram_count));
    counts["bigram"] = util::json_number(static_cast<double>(c.bigram_count));
    counts["trigram"] = util::json_number(static_cast<double>(c.trigram_count));
    counts["quadgram"] = util::json_number(static_cast<double>(c.quadgram_count));
    counts["coarse_gram"] = util::json_number(static_cast<double>(c.coarse_gram_count));
    counts["function_gram"] = util::json_number(static_cast<double>(c.function_gram_count));
    counts["tool_gram"] = util::json_number(static_cast<double>(c.tool_gram_count));
    return counts;
}

std::string serialize_mx2lm_gram_container(const micronaut::Mx2LmGramContainer& c) {
    util::JsonObject root;
    root["@type"] = util::json_string("mx2lm.gram_container.v1");
    root["key"] = util::json_string(c.key);
    root["namespace"] = util::json_string(c.namespace_id);
    root["authority"] = util::json_string(c.authority);
    root["phase"] = util::json_string(c.phase);
    root["total_grams"] = util::json_number(static_cast<double>(c.grams.size()));
    root["total_observations"] = util::json_number(static_cast<double>(c.total_observations));
    root["channel_counts"] = util::json_object(gram_channel_count_json(c));

    util::JsonObject channels;
    const std::vector<std::string> class_order = {
        "unigram", "bigram", "trigram", "quadgram",
        "coarse_gram", "function_gram", "tool_gram"
    };

    for (const auto& klass : class_order) {
        util::JsonArray rows;
        for (const auto& g : c.grams) {
            if (g.gram_class != klass) continue;
            util::JsonObject row;
            row["gram"] = util::json_string(g.gram);
            row["class"] = util::json_string(g.gram_class);
            row["role"] = util::json_string(g.role);
            row["phase"] = util::json_string(g.phase);
            row["count"] = util::json_number(static_cast<double>(g.count));
            row["width"] = util::json_number(static_cast<double>(g.width));
            row["tokens"] = util::json_array(string_array_json(g.tokens));
            rows.push_back(util::json_object(std::move(row)));
        }
        channels[klass] = util::json_array(std::move(rows));
    }

    root["channels"] = util::json_object(std::move(channels));
    return util::stringify(util::json_object(std::move(root)));
}

std::string mx2lm_gram_key(const micronaut::Mx2LmTypedGram& g) {
    return "mx2lm:gram:" + g.gram_class + ":" + g.gram;
}

util::JsonObject pattern_event_json(const micronaut::Mx2LmPatternEvent& event) {
    util::JsonObject row;
    row["source_track"] = util::json_string(event.source_track);
    row["source_phase"] = util::json_string(event.source_phase);
    row["target_namespace"] = util::json_string(event.target_namespace);
    row["surface"] = util::json_string(event.surface);
    row["roles"] = util::json_array(string_array_json(event.roles));
    row["gram_container"] = util::json_string(event.gram_container_key);
    row["outcome_kind"] = util::json_string(event.outcome_kind);
    row["outcome_status"] = util::json_string(event.outcome_status);
    row["accepted"] = util::json_bool(event.accepted);
    if (!event.card.empty())
        row["card"] = util::json_string(event.card);
    return row;
}

std::string serialize_mx2lm_pattern_container(const micronaut::Mx2LmPatternContainer& c) {
    util::JsonObject root;
    root["@type"] = util::json_string("mx2lm.pattern_container.v1");
    root["key"] = util::json_string(c.key);
    root["namespace"] = util::json_string(c.namespace_id);
    root["authority"] = util::json_string(c.authority);
    root["phase"] = util::json_string(c.phase);
    root["event_count"] = util::json_number(static_cast<double>(c.events.size()));

    util::JsonArray events;
    for (const auto& event : c.events)
        events.push_back(util::json_object(pattern_event_json(event)));
    root["events"] = util::json_array(std::move(events));
    return util::stringify(util::json_object(std::move(root)));
}

util::JsonObject vector_projection_json(const micronaut::Mx2LmVectorProjection& p) {
    util::JsonObject row;
    row["key"] = util::json_string(p.key);
    row["source_key"] = util::json_string(p.source_key);
    row["source_track"] = util::json_string(p.source_track);
    row["source_phase"] = util::json_string(p.source_phase);
    row["projection_track"] = util::json_string(p.projection_track);
    row["basis"] = util::json_string(p.basis);
    row["x"] = util::json_number(static_cast<double>(p.x));
    row["y"] = util::json_number(static_cast<double>(p.y));
    row["z"] = util::json_number(static_cast<double>(p.z));
    row["norm"] = util::json_number(static_cast<double>(p.norm));
    row["semantic_authority"] = util::json_bool(p.semantic_authority);
    return row;
}

std::string serialize_vector_projection_container(
    const micronaut::Mx2LmVectorProjectionContainer& c)
{
    util::JsonObject root;
    root["@type"] = util::json_string("vector.projection_container.v1");
    root["key"] = util::json_string(c.key);
    root["namespace"] = util::json_string(c.namespace_id);
    root["authority"] = util::json_string(c.authority);
    root["phase"] = util::json_string(c.phase);
    root["source_semantic_container"] = util::json_string(c.source_semantic_container);
    root["projection_id"] = util::json_string(c.projection_id);
    root["generation"] = util::json_number(static_cast<double>(c.generation));
    root["learner_version"] = util::json_string(c.learner_version);
    root["topology_version"] = util::json_string(c.topology_version);
    root["relation_model"] = util::json_string(c.relation_model);
    root["alignment"] = util::json_string(c.alignment);
    root["similarity_scope"] = util::json_string(c.similarity_scope);
    root["dimensions"] = util::json_number(static_cast<double>(c.dimensions));
    root["projection_count"] = util::json_number(static_cast<double>(c.projections.size()));
    root["semantic_authority"] = util::json_bool(false);
    root["responsibility"] = util::json_string("derived_geometry_projection_only");

    util::JsonArray projections;
    for (const auto& p : c.projections)
        projections.push_back(util::json_object(vector_projection_json(p)));
    root["projections"] = util::json_array(std::move(projections));
    return util::stringify(util::json_object(std::move(root)));
}

std::string key_safe(std::string value) {
    std::string out;
    out.reserve(value.size());
    bool last_sep = false;
    for (unsigned char ch : value) {
        if (std::isalnum(ch) != 0) {
            out.push_back(static_cast<char>(std::tolower(ch)));
            last_sep = false;
        } else if (!last_sep) {
            out.push_back('_');
            last_sep = true;
        }
        if (out.size() >= 64) break;
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? "pattern" : out;
}

std::string mx2lm_pattern_key(const micronaut::Mx2LmPatternEvent& event) {
    return "mx2lm:pattern:" + key_safe(event.source_track) + ":" +
           key_safe(event.source_phase) + ":" + key_safe(event.surface);
}

} // namespace

// ─── Pop ──────────────────────────────────────────────────────────────────

PopNGramResult PopNGram::process(const std::string& text) const {
    PopNGramResult r;
    r.surface = text;
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
    r.gram_container = micronaut::Mx2LmNGram::compile_gram_container(r.taxonomy);
    r.pattern_container = micronaut::Mx2LmNGram::compile_pattern_container(
        pop.surface,
        r.gram_container);
    r.vector_projection_container =
        micronaut::Mx2LmNGram::compile_vector_projection_container(
            r.gram_container,
            r.pattern_container);
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
    const auto& container = sek.yax.wo.gram_container;
    if (!container.grams.empty()) {
        store_.gramWrite(
            container.key,
            serialize_mx2lm_gram_container(container),
            "Xul",
            "mx2lm_gram_container",
            container.authority);
        r.written_keys.push_back(container.key);

        for (const auto& gram : container.grams) {
            const std::string key = mx2lm_gram_key(gram);
            store_.gramWrite(
                key,
                gram.gram,
                "Xul",
                "mx2lm_gram_container",
                container.authority);
            r.written_keys.push_back(key);
        }
    }

    const auto& pattern_container = sek.yax.wo.pattern_container;
    if (!pattern_container.events.empty()) {
        store_.gramWrite(
            pattern_container.key,
            serialize_mx2lm_pattern_container(pattern_container),
            "Xul",
            "mx2lm_pattern_container",
            pattern_container.authority);
        r.written_keys.push_back(pattern_container.key);

        for (const auto& event : pattern_container.events) {
            const std::string key = mx2lm_pattern_key(event);
            store_.gramWrite(
                key,
                event.surface,
                "Xul",
                "mx2lm_pattern_container",
                pattern_container.authority);
            r.written_keys.push_back(key);
        }
    }

    const auto& vector_container = sek.yax.wo.vector_projection_container;
    if (!vector_container.projections.empty()) {
        store_.gramWrite(
            vector_container.key,
            serialize_vector_projection_container(vector_container),
            "Xul",
            "vector_projection_container",
            vector_container.authority);
        r.written_keys.push_back(vector_container.key);

        for (const auto& projection : vector_container.projections) {
            store_.gramWriteVector(
                projection.key,
                projection.x,
                projection.y,
                projection.z,
                "Xul",
                "vector_projection",
                vector_container.authority);
            r.written_keys.push_back(projection.key);
        }
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
    const std::vector<std::string>& fold_neighborhood,
    const std::string& source_track,
    const std::string& source_phase,
    const std::string& outcome_kind,
    const std::string& card,
    bool accepted) const
{
    NGramPipelineResult res;
    res.pop  = pop_.process(text);
    res.wo   = wo_.process(res.pop);
    res.wo.pattern_container = micronaut::Mx2LmNGram::compile_pattern_container(
        res.pop.surface,
        res.wo.gram_container,
        source_track,
        source_phase,
        outcome_kind,
        card,
        accepted);
    res.wo.vector_projection_container =
        micronaut::Mx2LmNGram::compile_vector_projection_container(
            res.wo.gram_container,
            res.wo.pattern_container);
    res.yax  = yax_.process(res.wo, fold_neighborhood);
    res.sek  = sek_.process(res.yax);
    res.chen = chen_.process(res.sek);
    res.xul  = xul_.process(res.chen);
    return res;
}

} // namespace khanary::kuhul
