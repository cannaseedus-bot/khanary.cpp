#include "conformance.h"
#include "../micronaut/mx2lm-ngram.h"
#include "../kuhul/ngram_phase_classes.h"
#include "../kuhul/gram_store.h"
#include "../kuhul/fold_delta.h"

#include <cmath>
#include <iostream>

using namespace khanary::micronaut;
using namespace khanary::kuhul;

namespace {

// Shannon entropy over ranked gram scores; returns H in bits.
float gram_entropy(const std::vector<std::pair<std::string, float>>& ranked) {
    float total = 0.0f;
    for (const auto& [g, s] : ranked) total += s;
    if (total < 1e-6f) return 0.0f;
    float H = 0.0f;
    for (const auto& [g, s] : ranked) {
        float p = s / total;
        if (p > 1e-9f) H -= p * std::log2f(p);
    }
    return H;
}

// H_norm = H / H_max; 1.0 = perfectly uniform (AIML needed), 0.0 = single winner.
float gram_entropy_norm(const std::vector<std::pair<std::string, float>>& ranked) {
    if (ranked.size() <= 1) return 1.0f;
    float H_max = std::log2f(float(ranked.size()));
    return H_max < 1e-6f ? 0.0f : gram_entropy(ranked) / H_max;
}

// True when the gram distribution is too flat for the ngram pipeline to decide
// alone — entropy-delegates to AIML/Eliza for pattern resolution.
bool needs_aiml(const SekNGramResult& sek) {
    return gram_entropy_norm(sek.ranked) >= 0.5f;
}

} // namespace

int main() {
    return khanary::test::run("mx2lm-gram-channels", [](khanary::test::Suite& suite) {

        // ── Existing Mx2LmNGram channel tests ────────────────────────────

        auto model = Mx2LmNGram::build("alpha beta gamma delta");
        suite.check("quadgram_built", model.quadgrams.count("alpha beta gamma delta") == 1);

        auto tokens = Mx2LmNGram::tokenize("hey can you run tool call web research now");
        auto taxonomy = Mx2LmNGram::classify_gram_taxonomy(tokens);
        suite.check("unigram_present", taxonomy.unigrams.count("tool") == 1);
        suite.check("bigram_present", taxonomy.bigrams.count("tool call") == 1);
        suite.check("trigram_present", taxonomy.trigrams.count("tool call web") == 1);
        suite.check("quadgram_present", taxonomy.quadgrams.count("run tool call web") == 1);
        suite.check("coarse_grams_present", !taxonomy.coarse_grams.empty());
        suite.check("function_grams_present", !taxonomy.function_grams.empty());
        suite.check("tool_grams_present", !taxonomy.tool_grams.empty());

        const auto typed = Mx2LmNGram::typed_grams(taxonomy, 8);
        bool has_tool_channel = false;
        bool has_function_channel = false;
        bool has_coarse_channel = false;
        for (const auto& row : typed) {
            if (row.gram_class == "tool_gram"     && row.role == "tool_dispatch_intent")
                has_tool_channel = true;
            if (row.gram_class == "function_gram" && row.role == "function_word_structure")
                has_function_channel = true;
            if (row.gram_class == "coarse_gram"   && row.role == "structural_shape")
                has_coarse_channel = true;
        }
        suite.check("typed_tool_channel",     has_tool_channel);
        suite.check("typed_function_channel", has_function_channel);
        suite.check("typed_coarse_channel",   has_coarse_channel);

        // ── FoldDelta → VECTOR-µ boundary contract ───────────────────────
        const auto causes_geo =
            RelationConditionedGeometrySpec::from_relation("@causes");
        suite.check("fold_delta_relation_causes_translational",
            causes_geo.kind == RelationGeometryKind::Translational &&
            causes_geo.asymmetric &&
            causes_geo.valid_projection_boundary());

        const auto contradicts_geo =
            RelationConditionedGeometrySpec::from_relation("@contradicts");
        suite.check("fold_delta_relation_contradicts_rotational",
            contradicts_geo.kind == RelationGeometryKind::Rotational &&
            !contradicts_geo.asymmetric &&
            contradicts_geo.valid_projection_boundary());

        const auto part_geo =
            RelationConditionedGeometrySpec::from_relation("@part_of");
        suite.check("fold_delta_relation_part_of_bilinear",
            part_geo.kind == RelationGeometryKind::Bilinear &&
            part_geo.asymmetric);

        VectorMultiSignalObjective objective;
        suite.check("vector_multisignal_objective_valid", objective.valid());

        SemanticManifoldProjectionContract manifold;
        suite.check("semantic_manifold_projection_boundary_safe",
            manifold.safe() &&
            manifold.topology_symbol == "B" &&
            manifold.learned_geometry_symbol == "M_mu" &&
            !manifold.delta_manifold_is_delta_topology);

        LocalSemanticMetricSpec local_metric;
        local_metric.phase = "Yax";
        local_metric.relation = "@causes";
        suite.check("semantic_local_metric_phase_relation_conditioned",
            local_metric.valid_projection_boundary() &&
            relation_geometry_kind_for(local_metric.relation) ==
                RelationGeometryKind::Translational);

        SemanticGeodesicProjectionSpec geodesic;
        suite.check("semantic_geodesic_projection_not_authority",
            geodesic.valid_projection_boundary() &&
            !geodesic.semantic_authority &&
            !geodesic.can_verify_relation);

        VectorDecisionRiskSpec risk_ok;
        risk_ok.prior_mass = 0.65f;
        risk_ok.source_risk_bound = 1.0f;
        risk_ok.target_risk_bound = 1.0f;
        risk_ok.loss_cost = 0.95f;
        suite.check("vector_decision_risk_admissible",
            risk_ok.valid() && !risk_ok.inadmissible_minimax_rule());

        VectorDecisionRiskSpec risk_bad;
        risk_bad.prior_mass = 0.5f;
        risk_bad.source_risk_bound = 0.10f;
        risk_bad.target_risk_bound = 0.20f;
        risk_bad.loss_cost = 0.90f;
        suite.check("vector_decision_risk_minimax_rejects",
            risk_bad.valid() && risk_bad.inadmissible_minimax_rule());

        DifferentialComparisonProjectionSpec diff;
        diff.mutation = differential_mutation_op_for("modify");
        diff.base_version = 1;
        suite.check("differential_comparison_projection_boundary",
            diff.valid_projection_boundary() &&
            !diff.can_commit_topology);

        DifferentialComparisonProjectionSpec diff_invalid;
        diff_invalid.mutation = differential_mutation_op_for("warp");
        suite.check("differential_comparison_rejects_invalid_mutation",
            !diff_invalid.valid_projection_boundary());

        VectorProjectionGeneration gen0;
        VectorProjectionGeneration gen0_same;
        VectorProjectionGeneration gen1 = gen0;
        gen1.generation = 1;
        gen1.projection_id = "vector:projection:bootstrap:v1:g1";
        VectorProjectionGeneration gen_topology_changed = gen1;
        gen_topology_changed.topology_version = "runtime.semantic_containers.v2";

        suite.check("vector_generation_same_comparable",
            gen0.comparable_without_alignment(gen0_same) &&
            gen0.alignment_policy_for(gen0_same) ==
                VectorProjectionAlignmentPolicy::SameGenerationOnly);
        suite.check("vector_generation_same_topology_aligns",
            gen0.alignment_policy_for(gen1) ==
                VectorProjectionAlignmentPolicy::OrthogonalProcrustes);
        suite.check("vector_generation_topology_regenerates",
            gen0.alignment_policy_for(gen_topology_changed) ==
                VectorProjectionAlignmentPolicy::AuthoritativeRegeneration);

        FoldDelta good_delta;
        good_delta.verdict = "GOOD";
        suite.check("fold_delta_vector_boundary_safe",
            good_delta.vector_boundary.safe() &&
            good_delta.vector_projection_allowed());
        FoldDelta unknown_delta;
        unknown_delta.verdict = "UNKNOWN";
        suite.check("fold_delta_vector_boundary_rejects_unknown",
            !unknown_delta.vector_projection_allowed());

        const auto container = Mx2LmNGram::compile_gram_container(taxonomy);
        const size_t expected_container_size =
            taxonomy.unigrams.size() +
            taxonomy.bigrams.size() +
            taxonomy.trigrams.size() +
            taxonomy.quadgrams.size() +
            taxonomy.coarse_grams.size() +
            taxonomy.function_grams.size() +
            taxonomy.tool_grams.size();
        suite.check("mx2lm_container_namespace",
            container.namespace_id == "Mx2LM" &&
            container.key == "mx2lm:gram_container:v1");
        suite.check("mx2lm_container_all_channels",
            container.grams.size() == expected_container_size);
        bool container_has_bigram = false;
        bool container_phases_set = true;
        for (const auto& row : container.grams) {
            if (row.gram == "tool call" &&
                row.gram_class == "bigram" &&
                row.phase == "Wo")
                container_has_bigram = true;
            if (row.phase.empty()) container_phases_set = false;
        }
        suite.check("mx2lm_container_has_bigram_phase", container_has_bigram);
        suite.check("mx2lm_container_phases_set", container_phases_set);

        const auto pattern_container = Mx2LmNGram::compile_pattern_container(
            "Hello, hows it going?",
            container,
            "GREETING-µ",
            "Pop",
            "card_rendered",
            "greeting",
            true);
        suite.check("mx2lm_pattern_container_key",
            pattern_container.key == "mx2lm:pattern_container:v1" &&
            pattern_container.namespace_id == "Mx2LM");
        suite.check("mx2lm_pattern_event_track",
            pattern_container.events.size() == 1 &&
            pattern_container.events.front().source_track == "GREETING-µ" &&
            pattern_container.events.front().source_phase == "Pop");
        bool pattern_has_track_role = false;
        if (!pattern_container.events.empty()) {
            for (const auto& role : pattern_container.events.front().roles) {
                if (role == "track:GREETING-µ") {
                    pattern_has_track_role = true;
                    break;
                }
            }
        }
        suite.check("mx2lm_pattern_event_roles", pattern_has_track_role);

        const auto vector_container = Mx2LmNGram::compile_vector_projection_container(
            container,
            pattern_container);
        suite.check("vector_projection_container_key",
            vector_container.key == "vector:projection_container:v1" &&
            vector_container.namespace_id == "VECTOR-µ" &&
            vector_container.dimensions == 3);
        suite.check("vector_projection_generation_contract",
            vector_container.projection_id == "vector:projection:bootstrap:v1:g0" &&
            vector_container.generation == 0 &&
            vector_container.learner_version == "bootstrap.deterministic_hash.v1" &&
            vector_container.similarity_scope ==
                "semantic_similarity_subset_of_relatedness");
        suite.check("vector_projection_is_projection_only",
            vector_container.authority == "vector:projection-only" &&
            !vector_container.projections.empty() &&
            !vector_container.projections.front().semantic_authority);
        suite.check("vector_projection_is_finite_unit_vector",
            !vector_container.projections.empty() &&
            std::isfinite(vector_container.projections.front().x) &&
            std::isfinite(vector_container.projections.front().y) &&
            std::isfinite(vector_container.projections.front().z) &&
            std::fabs(vector_container.projections.front().norm - 1.0f) < 1e-4f);

        // ── Per-phase ngram classes ───────────────────────────────────────

        const std::string sentence = "if you understand then execute and verify the output";
        const std::vector<std::string> nb = {"execute", "verify", "output", "understand"};

        // Pop — tokenise + ngram model
        PopNGram pop_cls;
        auto pop_r = pop_cls.process(sentence);
        suite.check("pop_tokens_nonempty", !pop_r.tokens.empty());
        suite.check("pop_bigrams_built",   !pop_r.model.bigrams.empty());

        // Wo — gram taxonomy declared
        WoNGram wo_cls;
        auto wo_r = wo_cls.process(pop_r, 8);
        suite.check("wo_typed_nonempty", !wo_r.typed.empty());
        suite.check("wo_taxonomy_unigrams", !wo_r.taxonomy.unigrams.empty());
        suite.check("wo_mx2lm_container_nonempty", !wo_r.gram_container.grams.empty());
        suite.check("wo_mx2lm_container_is_all_grams",
            wo_r.gram_container.grams.size() >= wo_r.typed.size());
        suite.check("wo_mx2lm_pattern_container_nonempty",
            !wo_r.pattern_container.events.empty());
        suite.check("wo_vector_projection_container_nonempty",
            !wo_r.vector_projection_container.projections.empty() &&
            wo_r.vector_projection_container.source_semantic_container ==
                "mx2lm:pattern_container:v1");

        // Yax — admission gate; neighbourhood filters to resolved sense
        YaxNGram yax_cls;
        auto yax_r = yax_cls.process(wo_r, nb);
        suite.check("yax_admitted_nonempty", !yax_r.admitted.empty());
        suite.check("yax_resolved_sense_set", !yax_r.resolved_sense.empty());

        // Yax open-world: empty neighbourhood admits everything
        auto yax_open = yax_cls.process(wo_r, {});
        suite.check("yax_open_world_admits_all",
            yax_open.admitted.size() == wo_r.typed.size());

        // Sek — scoring + relational edges
        SekNGram sek_cls;
        auto sek_r = sek_cls.process(yax_r);
        suite.check("sek_ranked_nonempty", !sek_r.ranked.empty());
        // Scores must be in [0,1]
        bool scores_bounded = true;
        for (const auto& [g, s] : sek_r.ranked)
            if (s < 0.0f || s > 1.0f) { scores_bounded = false; break; }
        suite.check("sek_scores_bounded", scores_bounded);

        // Ch'en — certificate from admitted evidence
        ChenNGram chen_cls;
        auto chen_r = chen_cls.process(sek_r, "topology_hash_test");
        suite.check("chen_status_known",
            chen_r.certificate.status != ChenNGramStatus::UNKNOWN ||
            sek_r.yax.admitted.empty());

        // Ch'en on empty admitted → UNKNOWN
        SekNGram sek_cls2;
        YaxNGram yax_empty;
        auto yax_no_nb = yax_empty.process(wo_r, {"zzz_nothing_matches"});
        auto sek_empty = sek_cls2.process(yax_no_nb);
        auto chen_empty = chen_cls.process(sek_empty, "");
        suite.check("chen_empty_is_unknown",
            chen_empty.certificate.status == ChenNGramStatus::UNKNOWN);
        suite.check("chen_empty_miss_kind_set",
            !chen_empty.certificate.miss_kind.empty());

        // Xul — commits on PASS, rejects otherwise
        GramStore gs;
        XulNGram xul_cls(gs);
        auto xul_r = xul_cls.process(chen_r);
        // Committed iff certificate was PASS
        suite.check("xul_commit_matches_chen",
            xul_r.committed == (chen_r.certificate.status == ChenNGramStatus::PASS));
        if (xul_r.committed)
            suite.check("xul_keys_written", !xul_r.written_keys.empty());
        suite.check("xul_writes_mx2lm_container",
            !xul_r.committed || gs.has("mx2lm:gram_container:v1"));
        bool wrote_namespaced_gram = false;
        for (const auto& key : xul_r.written_keys) {
            if (key.rfind("mx2lm:gram:", 0) == 0) {
                wrote_namespaced_gram = true;
                break;
            }
        }
        suite.check("xul_writes_namespaced_mx2lm_grams",
            !xul_r.committed || wrote_namespaced_gram);
        suite.check("xul_writes_mx2lm_pattern_container",
            !xul_r.committed || gs.has("mx2lm:pattern_container:v1"));
        suite.check("xul_writes_vector_projection_container",
            !xul_r.committed || gs.has("vector:projection_container:v1"));
        bool wrote_pattern_key = false;
        bool wrote_vector_key = false;
        for (const auto& key : xul_r.written_keys) {
            if (key.rfind("mx2lm:pattern:", 0) == 0) {
                wrote_pattern_key = true;
            }
            if (key.rfind("vector:projection:", 0) == 0) {
                wrote_vector_key = true;
            }
        }
        suite.check("xul_writes_namespaced_mx2lm_patterns",
            !xul_r.committed || wrote_pattern_key);
        suite.check("xul_writes_namespaced_vector_projection",
            !xul_r.committed || wrote_vector_key);
        if (xul_r.committed && wrote_vector_key) {
            for (const auto& key : xul_r.written_keys) {
                if (key.rfind("vector:projection:", 0) != 0) continue;
                const auto vec = gs.gramVector(key);
                suite.check("xul_vector_projection_readback",
                    vec.has_value() &&
                    std::isfinite((*vec)[0]) &&
                    std::isfinite((*vec)[1]) &&
                    std::isfinite((*vec)[2]));
                break;
            }
        }

        // Xul on UNKNOWN → no commit
        auto xul_reject = xul_cls.process(chen_empty);
        suite.check("xul_rejects_unknown", !xul_reject.committed);
        suite.check("xul_rejection_reason_set", !xul_reject.rejection_reason.empty());

        // ── NGramPipeline ──────────────────────────────────────────────────

        GramStore gs2;
        NGramPipeline pipeline(gs2);
        auto pipe_r = pipeline.run(sentence, nb, "GREETING-µ", "Pop", "card_rendered", "greeting", true);
        suite.check("pipeline_pop_nonempty",  !pipe_r.pop.tokens.empty());
        suite.check("pipeline_wo_nonempty",   !pipe_r.wo.typed.empty());
        suite.check("pipeline_pattern_track",
            !pipe_r.wo.pattern_container.events.empty() &&
            pipe_r.wo.pattern_container.events.front().source_track == "GREETING-µ");
        suite.check("pipeline_vector_track",
            !pipe_r.wo.vector_projection_container.projections.empty() &&
            pipe_r.wo.vector_projection_container.projections.front().projection_track ==
                "VECTOR-µ");
        suite.check("pipeline_yax_admitted",  !pipe_r.yax.admitted.empty());
        suite.check("pipeline_sek_ranked",    !pipe_r.sek.ranked.empty());
        suite.check("pipeline_chen_complete",
            pipe_r.chen.certificate.status != ChenNGramStatus::UNKNOWN ||
            pipe_r.yax.admitted.empty());
        suite.check("pipeline_xul_consistent",
            pipe_r.xul.committed ==
            (pipe_r.chen.certificate.status == ChenNGramStatus::PASS));

        // ── Entropy ↔ AIML coupling ────────────────────────────────────────
        //
        // The ngram pipeline produces a ranked score distribution.
        // High H_norm (≥0.5) = distribution is too flat to route alone
        //   → entropy_delegates_to ELIZA for pattern resolution.
        // Low H_norm (<0.5)  = distribution is peaked = ngram result is final.
        //
        // Scoring blends relation-endpoint overlap (0.6) + frequency (0.4).
        // Control-pivot sentences create relation endpoints → some grams score
        // much higher than average → lower entropy.  Generic function-word
        // sentences produce no relation pivots → uniform scores → high entropy.

        // Confident case: sentence with an explicit causal pivot ("because")
        // so infer_relations() creates non-empty relation endpoints, causing
        // at least one gram to score significantly higher than the rest.
        GramStore gs3;
        NGramPipeline pipe_conf(gs3);
        auto conf_r = pipe_conf.run(
            "because you ran the tool call it verified the web research output",
            {"tool", "call", "verified", "output", "ran"});
        float conf_H = gram_entropy_norm(conf_r.sek.ranked);

        // Ambiguous case: generic function-word sentence, no control pivots.
        GramStore gs4;
        NGramPipeline pipe_amb(gs4);
        auto amb_r = pipe_amb.run("what do you think about it", {});
        float amb_H = gram_entropy_norm(amb_r.sek.ranked);
        bool aiml_needed = needs_aiml(amb_r.sek);

        // Key invariant: the structured sentence has ≤ entropy than the
        // function-word sentence (relative ordering, not absolute threshold).
        suite.check("entropy_structured_le_ambiguous",
            conf_r.sek.ranked.empty() || amb_r.sek.ranked.empty() ||
            conf_H <= amb_H + 0.05f);

        // Ambiguous case: either flat distribution or no mass → AIML handles it
        suite.check("entropy_ambiguous_delegates",
            aiml_needed || amb_r.sek.ranked.empty());

        // Entropy determinism: running the same text twice gives the same H_norm
        GramStore gs5;
        NGramPipeline pipe_det(gs5);
        auto det_r = pipe_det.run(
            "because you ran the tool call it verified the web research output",
            {"tool", "call", "verified", "output", "ran"});
        float det_H = gram_entropy_norm(det_r.sek.ranked);
        suite.check("entropy_deterministic", conf_H == det_H);

        // Entropy is always in [0, 1] when normalised
        suite.check("entropy_norm_bounded_conf", conf_H >= 0.0f && conf_H <= 1.0f);
        suite.check("entropy_norm_bounded_amb",  amb_H  >= 0.0f && amb_H  <= 1.0f);

        // ── Miss kind classification ───────────────────────────────────────
        //
        // evidence:  no grams passed the Yax gate (τ too narrow)
        // semantic:  grams admitted but no relations and no control bindings
        // memory:    grams admitted, scores all below noise floor

        // evidence miss: hard neighbourhood mismatch → empty admitted
        {
            PopNGram p; WoNGram w; YaxNGram y; SekNGram s;
            ChenNGram c;
            auto pr = p.process("run the tool call");
            auto wr = w.process(pr);
            auto yr = y.process(wr, {"zzz_nothing_matches"});
            auto sr = s.process(yr);
            auto cr = c.process(sr, "");
            suite.check("miss_kind_evidence",
                cr.certificate.miss_kind_enum == ChenMissKind::evidence);
            suite.check("miss_kind_str_evidence",
                cr.certificate.miss_kind == "evidence");
            suite.check("miss_rho_bounded",
                cr.certificate.research_pressure >= 0.0f &&
                cr.certificate.research_pressure <= 1.0f);
        }

        // semantic miss: admitted grams (open world) but no relations / control
        {
            PopNGram p; WoNGram w; YaxNGram y; SekNGram s;
            ChenNGram c;
            // "alpha beta gamma delta" — no control words → no relations, no control bindings
            auto pr = p.process("alpha beta gamma delta");
            auto wr = w.process(pr);
            auto yr = y.process(wr, {});           // open world → admit all
            auto sr = s.process(yr);
            auto cr = c.process(sr, "");
            // Semantic miss when admitted but no structure; may fall through to
            // memory if all scores are below noise.  Either is acceptable.
            suite.check("miss_kind_semantic_or_memory",
                cr.certificate.miss_kind_enum == ChenMissKind::semantic ||
                cr.certificate.miss_kind_enum == ChenMissKind::memory   ||
                cr.certificate.status == ChenNGramStatus::PASS);
            suite.check("miss_kind_semantic_rho_bounded",
                cr.certificate.research_pressure >= 0.0f &&
                cr.certificate.research_pressure <= 1.0f);
        }

        // miss kind round-trip: name ↔ enum
        suite.check("miss_name_evidence",
            std::string(chen_miss_kind_name(ChenMissKind::evidence)) == "evidence");
        suite.check("miss_name_topology",
            std::string(chen_miss_kind_name(ChenMissKind::topology)) == "topology");
        suite.check("miss_from_name_factual",
            chen_miss_kind_from_name("factual") == ChenMissKind::factual);
        suite.check("miss_from_name_mathematical",
            chen_miss_kind_from_name("mathematical") == ChenMissKind::mathematical);

        // ── Advisor delegation ────────────────────────────────────────────
        //
        // Set an advisor on ChenNGram. When a MISS fires the advisor is called
        // with a ChenMissContext. If the advisor returns evidence_obtained=true
        // with confidence >= 0.5, Ch'en upgrades to PASS/AUTHORITY and Xul
        // commits the advisor-acquired keys.

        {
            PopNGram p; WoNGram w; YaxNGram y; SekNGram s;
            ChenNGram c;

            bool advisor_called = false;
            std::string advisor_saw_kind;
            float       advisor_saw_pressure = -1.0f;

            c.set_advisor([&](const ChenMissContext& ctx) -> ChenAdvisorResponse {
                advisor_called       = true;
                advisor_saw_kind     = ctx.miss_kind_str;
                advisor_saw_pressure = ctx.research_pressure;
                ChenAdvisorResponse resp;
                resp.evidence_obtained = true;
                resp.acquired_keys     = {"web:result_alpha", "web:result_beta"};
                resp.source            = "web_research";
                resp.confidence        = 0.85f;
                return resp;
            });

            // Trigger a MISS: strict neighbourhood that blocks all grams
            auto pr = p.process("delta epsilon zeta");
            auto wr = w.process(pr);
            auto yr = y.process(wr, {"zzz_nothing"});
            auto sr = s.process(yr);
            auto cr = c.process(sr, "topo_hash_advisor_test");

            suite.check("advisor_was_called",      advisor_called);
            suite.check("advisor_saw_miss_kind",   !advisor_saw_kind.empty());
            suite.check("advisor_pressure_valid",
                advisor_saw_pressure >= 0.0f && advisor_saw_pressure <= 1.0f);
            suite.check("advisor_upgrades_to_pass",
                cr.certificate.status == ChenNGramStatus::PASS);
            suite.check("advisor_assertion_authority",
                cr.certificate.assertion_class == "AUTHORITY");
            suite.check("advisor_keys_in_cert",
                !cr.certificate.advisor_keys.empty());
            suite.check("advisor_source_set",
                cr.certificate.advisor_source == "web_research");

            // Xul should now commit including the advisor-acquired keys
            GramStore gs_adv;
            XulNGram xul_adv(gs_adv);
            auto xul_adv_r = xul_adv.process(cr);
            suite.check("advisor_xul_commits", xul_adv_r.committed);
            bool has_web_key = false;
            for (const auto& k : xul_adv_r.written_keys)
                if (k == "web:result_alpha" || k == "web:result_beta")
                    { has_web_key = true; break; }
            suite.check("advisor_keys_committed", has_web_key);
        }

        // Advisor with low confidence → UNKNOWN stays UNKNOWN
        {
            PopNGram p; WoNGram w; YaxNGram y; SekNGram s;
            ChenNGram c;

            c.set_advisor([](const ChenMissContext&) -> ChenAdvisorResponse {
                ChenAdvisorResponse resp;
                resp.evidence_obtained = true;
                resp.acquired_keys     = {"weak_result"};
                resp.source            = "web_research";
                resp.confidence        = 0.3f; // below 0.5 threshold
                return resp;
            });

            auto pr = p.process("unclear ambiguous phrase");
            auto wr = w.process(pr);
            auto yr = y.process(wr, {"zzz_nothing"});
            auto sr = s.process(yr);
            auto cr = c.process(sr, "");

            suite.check("advisor_low_confidence_stays_unknown",
                cr.certificate.status == ChenNGramStatus::UNKNOWN);
            suite.check("advisor_low_confidence_consulted",
                cr.certificate.advisor_consulted);
        }
    });
}
