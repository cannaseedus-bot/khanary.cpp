#include "conformance.h"
#include "../graphscript/attention_algebra.h"
#include "../graphscript/graph_algebra.h"
#include "../inference/mu_attention.h"
#include "../inference/semantic_attention.h"
#include "../kuhul/gram_store.h"
#include "../micronaut/internal_micronauts.h"
#include "../micronaut/micronaut_call_graph.h"
#include <cmath>
#include <string>
#include <vector>

using namespace khanary;

namespace {

graphscript::RowMatrix Q = {{1.0, 0.0}, {0.0, 1.0}};
graphscript::RowMatrix K = {{1.0, 0.0}, {0.0, 1.0}};
graphscript::RowMatrix V = {{2.0, 20.0}, {4.0, 40.0}};

} // namespace

int main() {
    return test::run("attention_algebra", [&](test::Suite& suite) {
        using graphscript::attention;
        using graphscript::scaled_dot_scores;
        using graphscript::softmax_row;
        using graphscript::weighted_fold;

        // ── MAP: scaled dot-product (the box's QKᵀ/√d) ───────────────────
        const auto scores = scaled_dot_scores(Q, K);
        suite.check("map_scaled_scores",
            scores.size() == 2 && std::abs(scores[0][0] - 0.70710678118654752) < 1e-12 &&
            std::abs(scores[1][1] - 0.70710678118654752) < 1e-12);

        // ── FILTER: stable softmax (soft selection) ──────────────────────
        const auto soft = softmax_row({1000.0, -1000.0});
        suite.check("filter_softmax_stability",
            abs(soft[0] - 1.0) < 1e-12 && soft[1] == 0.0);

        // ── FOLD: the weighted aggregation Σ w·V ─────────────────────────
        const auto folded = weighted_fold({0.25, 0.75}, V);
        suite.check("fold_weighted_aggregation",
            abs(folded[0] - 3.5) < 1e-12 && abs(folded[1] - 35.0) < 1e-12);

        // ── The composed box: attention == map∘filter∘fold; and the mix
        // arithmetic: with d=2 the scaled dot is 0.7071, so the softmax
        // spreads 67/33 — the output is a MIX, inside the values' hull.
        const auto whole = attention(Q, K, V);
        const auto w0 = softmax_row(scores[0]);
        suite.check("softmax_normalizes_and_favors_aligned",
            abs(w0[0] + w0[1] - 1.0) < 1e-12 && w0[0] > 0.5 && w0[0] < 1.0);
        suite.check("box_formula_exact",
            whole.size() == 2 &&
            abs(whole[0][0] - (w0[0] * 2.0 + w0[1] * 4.0)) < 1e-12 &&
            abs(whole[0][1] - (w0[0] * 20.0 + w0[1] * 40.0)) < 1e-12 &&
            whole[0][0] > 2.0 && whole[0][0] < 4.0);   // inside the hull
        // identity Q=K → the softmax concentrates on the diagonal → V rows out.
        const auto manual = [&] {
            std::vector<std::vector<double>> out;
            for (size_t i = 0; i < 2; ++i) {
                const auto w = softmax_row(scores[i]);
                out.push_back(weighted_fold(w, V));
            }
            return out;
        }();
        suite.check("decomposition_equals_whole", manual == whole);

        // ── The mask = the crisp filter (declared edges gate attention) ──
        const auto masked = attention(Q, K, V, {{true, false}, {false, true}});
        suite.check("mask_is_crisp_filter",
            abs(masked[0][0] - 2.0) < 1e-12 && abs(masked[0][1] - 20.0) < 1e-12 &&
            abs(masked[1][0] - 4.0) < 1e-12 && abs(masked[1][1] - 40.0) < 1e-12 &&
            abs(masked[0][1] - 20.0) < 1e-12);   // mass from the surviving key only

        // mask + DIFFERENT value rows: the renormalization over survivors.
        const auto renorm = attention(Q, K, {{1.0}, {3.0}}, {{true, true}, {true, true}});
        suite.check("mask_full_keeps_both",
            abs(renorm[0][0] - (softmax_row(scores[0])[0] * 1.0 +
                                softmax_row(scores[0])[1] * 3.0)) < 1e-12);

        const auto half_masked = attention(Q, K, {{1.0}, {7.0}}, {{true, false}, {true, false}});
        suite.check("mask_weights_renormalize",
            abs(half_masked[0][0] - 1.0) < 1e-12 && abs(half_masked[1][0] - 1.0) < 1e-12);

        // All-masked row → the named violation (no fabricated mass).
        suite.rejects("all_masked_row_named_violation", [&] {
            (void)attention(Q, K, V, {{false, false}, {true, true}});
        });

        // ── Causal: query i sees keys j ≤ i ─────────────────────────────
        const auto causal = attention(Q, K, V, {}, true);
        suite.check("causal_policy",
            abs(causal[0][0] - 2.0) < 1e-9 && abs(causal[0][1] - 20.0) < 1e-9 &&          // row 0: one key → weight 1
            abs(causal[1][0] - (softmax_row(scores[1])[0] * 2.0 +
                                softmax_row(scores[1])[1] * 4.0)) < 1e-12);               // row 1: both keys

        // ── The WCR bridge: W·C·cos(Q,K) and the cos variant ────────────
        using graphscript::wcr_scores;
        const auto wcr = wcr_scores(Q, K, {0.5, 0.5}, {1.0, 1.0});
        suite.check("wcr_score_kernel",
            abs(wcr[0][0] - 0.5) < 1e-12 && abs(wcr[1][1] - 0.5) < 1e-12);
        // cosine scores on already-normalized rows == the dot form (no scaling):
        graphscript::RowMatrix qn = Q, kn = K;
        const auto local_dot = [](const std::vector<double>& a, const std::vector<double>& b) {
            double sum = 0.0;
            for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) sum += a[i] * b[i];
            return sum;
        };
        const auto cosish = wcr_scores(qn, kn, {1.0, 1.0}, {1.0, 1.0});
        suite.check("wcr_cos_matches_dot_on_unit_rows",
            abs(cosish[0][0] - local_dot(qn[0], kn[0])) < 1e-12);

        // ── The field tie: self-attention over the declared graph ───────
        // The algebra fixture: 3 vertices, edges 0-1 (0.5), 1-2 (0.8).
        const std::vector<std::vector<float>> payloads =
            {{-2.0f, 1.0f}, {4.0f, 3.0f}, {-0.5f, -1.0f}};
        const std::vector<std::vector<float>> adjacency =
            {{1.0f, 0.5f, 0.0f}, {0.5f, 1.0f, 0.8f}, {0.0f, 0.8f, 1.0f}};
        const auto field_out =
            graphscript::self_attention_over_field(payloads, adjacency);

        // Vertex 0's mask = {T, T, F}: softmax over [3.5355, -3.5355] (the
        // scaled dots) → w ≈ {0.9772, 0.0228} → Σ w·v:
        const double w00 = [](double a, double b) {
            const double ea = std::exp(a - a), eb = std::exp(b - a);
            return ea / (ea + eb);
        }(3.5355339059327378, -3.5355339059327378);
        suite.check("field_masked_self_attention",
            field_out.size() == 3 &&
            std::abs(field_out[0][0] - (w00 * -2.0 + (1.0 - w00) * 4.0)) < 1e-6 &&
            std::abs(field_out[0][1] - (w00 * 1.0 + (1.0 - w00) * 3.0)) < 1e-6);
        // The unmasked form differs (vertex 0 would also see key 2's mass) —
        // the declared-edges mask is doing real work.
        const auto unmasked_out = attention(
            {std::vector<double>(payloads[0].begin(), payloads[0].end())},
            {std::vector<double>(payloads[0].begin(), payloads[0].end()),
             std::vector<double>(payloads[1].begin(), payloads[1].end()),
             std::vector<double>(payloads[2].begin(), payloads[2].end())},
            {std::vector<double>(payloads[0].begin(), payloads[0].end()),
             std::vector<double>(payloads[1].begin(), payloads[1].end()),
             std::vector<double>(payloads[2].begin(), payloads[2].end())});
        suite.check("mask_changes_the_mechanics",
            std::abs(unmasked_out[0][0] - field_out[0][0]) > 1e-6);

        // Map/filter/fold shape invariants under attention: |out| = |Q| rows,
        // d_v columns — the map/fold law holds for the mechanism.
        suite.check("attention_shape_invariants",
            attention(Q, K, V).size() == Q.size() &&
            attention(Q, K, V)[0].size() == V[0].size());

        // ══ The CONCRETE Q/K/V: the micronaut lane's own attention ════════
        {
            micronaut::register_internal_micronauts();
            using inference::PromptAttentionCandidate;
            using inference::micronaut_attention;

            // Map(encode, input): the µ's 3-D encode — deterministic.
            const auto encoded = inference::prompt_attention_vector_for(
                "similarity", "fold reachability");
            suite.check("mu_encode_deterministic",
                encoded == inference::prompt_attention_vector_for(
                    "similarity", "fold reachability"));

            std::vector<PromptAttentionCandidate> candidates = {
                {"gram::fold",    0.9f,  {2.0f, 0.0f, 0.0f},    true, true, true},
                {"gram::phase",   0.8f,  {0.0f, 0.0f, 2.0f},    true, true, true},
                {"gram::blocked", 0.95f, {100.0f, 0.0f, 0.0f},  false, true, true},  // policy-dropped
                {"gram::ghost",   0.0f,  {1.0f, 1.0f, 1.0f},    true, true, true},   // relevance 0
            };
            const auto outcome = micronaut_attention(encoded, candidates);

            // The FILTER: capability flags kill candidates regardless of
            // their magnetic match; relevance 0 filters the ghost.
            suite.check("mu_capability_mask",
                outcome.ok &&
                outcome.admitted_keys ==
                    std::vector<std::string>({"gram::fold", "gram::phase"}) &&
                outcome.winner_key != "gram::blocked" &&
                outcome.winner_key != "gram::ghost");

            // The soft filter + the fold: the output lies in the admitted
            // rows' convex hull: |out| ≤ max |v_j| (Σ w = 1, w ≥ 0).
            const double max_norm = 2.0;
            const double out_norm = std::sqrt(
                static_cast<double>(outcome.output[0]) * outcome.output[0] +
                static_cast<double>(outcome.output[1]) * outcome.output[1] +
                static_cast<double>(outcome.output[2]) * outcome.output[2]);
            suite.check("mu_fold_convex_hull", out_norm <= max_norm + 1e-9);

            // The route decision left µ-graph evidence (REASON-µ @decide).
            const auto graph = micronaut::MicronautCallGraph::instance().graph_json();
            suite.check("mu_route_decided_on_graph",
                util::stringify(graph).find("\"@decide\"") != std::string::npos);

            // The empty-eligible refusal is named.
            std::vector<PromptAttentionCandidate> locked = {
                {"gram::sealed", 0.9f, {1.0f, 0.0f, 0.0f}, false, true, true},
            };
            const auto refused = micronaut_attention(encoded, locked);
            suite.check("mu_refusal_named",
                !refused.ok && refused.error.find("no eligible candidates") != std::string::npos);

            // ══ The GramStore chain: store → collect → @attend (ATTN-µ) ═══
            kuhul::GramStore store;   // in-memory (no journal pollution)
            // The collector's row contract: key = <prompt>:<class>[:ordinal];
            // value = <class>|<rank>|<text>; authority ≠ projection-only.
            store.gramWrite("arch:trigram:1",  "trigram|1|fold reachability", "Xul");
            store.gramWrite("arch:atom:1",     "atom|1|reach the fold",       "Xul");
            store.gramWrite("arch:pattern:1",  "pattern|1|WHO * YOU ARE",     "Xul");   // the 2D parent (wildcard key)
            store.gramWrite("arch:template:1", "template|1|I AM THE ANSWER",  "Xul");   // the 2D child
            store.gramWrite("arch:unigram:1",  "unigram|1|why",               "Xul",
                             "general", "vector:projection-only");   // → xcfe_ok = false
            store.gramWrite("arch:file:notes", "file|1|noise", "Xul");     // → skipped
            store.gramWrite("arch:vector:0", "x|1|y", "Xul");               // → skipped

            const auto collected = inference::collect_prompt_attention_candidates(store, "arch");
            suite.check("store_collect_rows",
                collected.size() == 5 &&                 // trigram, atom, pattern, template, unigram-flagged
                collected[0].key == "arch:trigram:1" &&  // relevance: .95 > .90 > .86 > .80 > .72
                collected[0].relevance > 0.9f &&
                collected[1].key == "arch:atom:1" &&
                collected[2].key == "arch:pattern:1" &&  // the PARENT outranks its CHILD (0.86 > 0.80)
                collected[3].key == "arch:template:1" &&
                !collected[4].xcfe_ok);                  // the projection-only gram carries its filter flag into attention

            // The replaced rows: quadgram/coarse_gram grams are no longer
            // attention candidates (class_weight 0 = the collector's skip);
            // the AIML pattern/template designations stay collectible.
            store.gramWrite("arch:quadgram:1", "quadgram|1|alpha beta gamma delta", "Xul");
            const auto replaced = inference::collect_prompt_attention_candidates(store, "arch");
            suite.check("old_gram_classes_no_longer_collected", replaced.size() == 5);

            // The µ's @encode supplies Q; the @attend supplies the pipeline.
            const auto encode_result = micronaut::mu_call("router", "ATTN-µ", "@encode",
                util::json_object({
                    {"semantic_class", util::json_string("similarity")},
                    {"gram_text",      util::json_string("fold reachability")},
                }));
            suite.check("mu_encode_verb",
                encode_result.success &&
                encode_result.result->get("vector")->as_array().size() == 3);

            // Assemble the candidate JSON rows for the µ-call.
            util::JsonArray candidate_rows;
            for (const auto& cand : collected) {
                candidate_rows.push_back(util::json_object({
                    {"key",       util::json_string(cand.key)},
                    {"relevance", util::json_number(static_cast<double>(cand.relevance))},
                    {"vector",    util::json_array({
                        util::json_number(static_cast<double>(cand.vector[0])),
                        util::json_number(static_cast<double>(cand.vector[1])),
                        util::json_number(static_cast<double>(cand.vector[2])),
                    })},
                    {"capable",   util::json_bool(cand.capable)},
                    {"policy_ok", util::json_bool(cand.policy_ok)},
                    {"xcfe_ok",   util::json_bool(cand.xcfe_ok)},
                }));
            }
            const auto attends = micronaut::mu_call("micronaut_coder", "ATTN-µ", "@attend",
                util::json_object({
                    {"query", util::json_array({
                        util::json_number(static_cast<double>(encoded[0])),
                        util::json_number(static_cast<double>(encoded[1])),
                        util::json_number(static_cast<double>(encoded[2])),
                    })},
                    {"candidates", util::json_array(std::move(candidate_rows))},
                }));
            suite.check("store_chain_via_mutron",
                attends.success &&
                attends.result->get("admitted")->as_array().size() == 4 &&  // the flagged unigram dropped by the store's authority
                attends.result->get_string("winner").value_or("").rfind("arch:", 0) == 0 &&
                attends.result->get_string("winner").value_or("") != "arch:unigram:1" &&
                attends.result->get("output")->as_array().size() == 3);

            // The µ-graph now shows the ATTN-µ node (the lane's own µ).
            const auto attn_graph = micronaut::MicronautCallGraph::instance().graph_json();
            suite.check("mutron_graph_has_attn_node",
                util::stringify(attn_graph).find("ATTN-µ") != std::string::npos);
        }

        // ══ INCIDENCE_PRODUCT: A = B · W_E · Bᵀ (fold-theoretic attention) ══
        {
            using graphscript::incidence_product;
            using graphscript::diagonal_weights;

            // 2 vertices, 3 contexts; W = diag(1, 2, 3)
            // B = [[1,0,0],[0,1,1]]
            const graphscript::RowMatrix B2 = {{1.0, 0.0, 0.0}, {0.0, 1.0, 1.0}};
            const auto W3 = diagonal_weights({1.0, 2.0, 3.0});
            const auto A = incidence_product(B2, W3);
            // BW = [[1,0,0],[0,2,3]]; A[0][0]=1, A[0][1]=0, A[1][1]=2+3=5
            suite.check("incidence_product_diagonal",
                A.size() == 2 && A[0].size() == 2 &&
                std::abs(A[0][0] - 1.0) < 1e-12 && std::abs(A[0][1]) < 1e-12 &&
                std::abs(A[1][0]) < 1e-12 && std::abs(A[1][1] - 5.0) < 1e-12);

            // Identity weight: A = B Bᵀ; A[0][0]=1, A[1][1]=2, A[0][1]=0
            const auto W_id = diagonal_weights({1.0, 1.0, 1.0});
            const auto A_id = incidence_product(B2, W_id);
            suite.check("incidence_product_identity_weight",
                std::abs(A_id[0][0] - 1.0) < 1e-12 &&
                std::abs(A_id[1][1] - 2.0) < 1e-12 &&
                std::abs(A_id[0][1]) < 1e-12);

            // Non-diagonal symmetric W_E: exercises the full C×C code path and
            // verifies A = Aᵀ with a nonzero off-diagonal entry (not trivially 0).
            // B = [[1,2],[3,1]], W_sym = [[1,0.5],[0.5,2]]
            // BW[0]=[1+1, 0.5+4]=[2.0,4.5]; BW[1]=[3+0.5, 1.5+2]=[3.5,3.5]
            // A[0][0]=2*1+4.5*2=11; A[0][1]=2*3+4.5*1=10.5; A[1][1]=3.5*3+3.5*1=14
            const graphscript::RowMatrix B_nd  = {{1.0, 2.0}, {3.0, 1.0}};
            const graphscript::RowMatrix W_sym = {{1.0, 0.5}, {0.5, 2.0}};
            const auto A_sym = incidence_product(B_nd, W_sym);
            suite.check("incidence_product_symmetric_nontrivial",
                std::abs(A_sym[0][0] - 11.0) < 1e-12 &&
                std::abs(A_sym[1][1] - 14.0) < 1e-12 &&
                std::abs(A_sym[0][1] - 10.5) < 1e-12 &&
                std::abs(A_sym[0][1] - A_sym[1][0]) < 1e-12);  // A = Aᵀ, nonzero

            // Triple-matmul parity: incidence_product agrees with naive B*W*Bᵀ to 1e-12.
            {
                const size_t nv2 = B_nd.size(), nc2 = B_nd[0].size();
                graphscript::RowMatrix BW2(nv2, std::vector<double>(nc2, 0.0));
                for (size_t i = 0; i < nv2; ++i)
                    for (size_t c = 0; c < nc2; ++c)
                        for (size_t d = 0; d < nc2; ++d)
                            BW2[i][d] += B_nd[i][c] * W_sym[c][d];
                graphscript::RowMatrix A_triple(nv2, std::vector<double>(nv2, 0.0));
                for (size_t i = 0; i < nv2; ++i)
                    for (size_t j = 0; j < nv2; ++j)
                        for (size_t c = 0; c < nc2; ++c)
                            A_triple[i][j] += BW2[i][c] * B_nd[j][c];
                suite.check("incidence_product_triple_parity",
                    std::abs(A_triple[0][0] - A_sym[0][0]) < 1e-12 &&
                    std::abs(A_triple[0][1] - A_sym[0][1]) < 1e-12 &&
                    std::abs(A_triple[1][0] - A_sym[1][0]) < 1e-12 &&
                    std::abs(A_triple[1][1] - A_sym[1][1]) < 1e-12);
            }

            // Asymmetric W_E: A ≠ Aᵀ; pins W_E[c][d] index order.
            // B = I_2, W_asym = [[1,2],[3,4]]: A = B * W_asym * I = W_asym
            // → A[0][1]=2, A[1][0]=3 (different)
            const graphscript::RowMatrix B_eye  = {{1.0, 0.0}, {0.0, 1.0}};
            const graphscript::RowMatrix W_asym = {{1.0, 2.0}, {3.0, 4.0}};
            const auto A_asym = incidence_product(B_eye, W_asym);
            suite.check("incidence_product_asymmetric",
                std::abs(A_asym[0][1] - 2.0) < 1e-12 &&
                std::abs(A_asym[1][0] - 3.0) < 1e-12 &&
                std::abs(A_asym[0][1] - A_asym[1][0]) > 0.5);  // NOT symmetric

            // Entry-wise bilinear form: A[i][j] = B_i · W · B_jᵀ is bilinear
            // in (B_i, B_j).  Test: replace row 0 with α*b0 + β*c0 and verify
            // A[0][1] scales correctly.
            // b0=[1,0,0], c0=[0,1,0], combo=2*b0+3*c0=[2,3,0]
            const graphscript::RowMatrix B_b0   = {{1.0, 0.0, 0.0}, {0.0, 1.0, 1.0}};
            const graphscript::RowMatrix B_c0   = {{0.0, 1.0, 0.0}, {0.0, 1.0, 1.0}};
            const graphscript::RowMatrix B_combo= {{2.0, 3.0, 0.0}, {0.0, 1.0, 1.0}};
            const auto A_b0    = incidence_product(B_b0,    W3);
            const auto A_c0    = incidence_product(B_c0,    W3);
            const auto A_combi = incidence_product(B_combo, W3);
            // A[0][1] for b0   : BW[0]=[1,0,0] · B[1]=[0,1,1] = 0
            // A[0][1] for c0   : BW[0]=[0,2,0] · B[1]=[0,1,1] = 2
            // A[0][1] for combo: BW[0]=[2,6,0] · B[1]=[0,1,1] = 6
            // 2*0 + 3*2 = 6  ✓
            suite.check("incidence_product_entry_bilinear",
                std::abs(A_combi[0][1] - (2.0 * A_b0[0][1] + 3.0 * A_c0[0][1])) < 1e-12);

            // Violation: W_E dimension != B columns
            suite.rejects("incidence_product_dimension_violation", [&] {
                (void)incidence_product(B2, graphscript::RowMatrix({{1.0, 2.0}, {3.0, 4.0}}));
            });
        }
    });
}
