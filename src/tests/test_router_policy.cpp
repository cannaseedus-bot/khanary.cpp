#include "../io/request_router.h"
#include "../kuhul/xul_gate.h"
#include "../memory/perception.h"
#include "../memory/confidence.h"
#include "../memory/semantic_graph.h"
#include "../memory/scoring.h"
#include "../micronaut/micronaut_registry.h"
#include "../micronaut/factory.h"
#include "../util/json.h"
#include "conformance.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using namespace khanary;

namespace {

bool expect_true(const std::string& label, bool value) {
    if (value) return true;
    std::cerr << "[FAIL] " << label << ": expected true\n";
    return false;
}

bool expect_eq(const std::string& label, const std::string& got, const std::string& want) {
    if (got == want) return true;
    std::cerr << "[FAIL] " << label << ": expected '" << want << "' got '" << got << "'\n";
    return false;
}

bool contains_value(const std::vector<std::string>& values, const std::string& target) {
    return std::find(values.begin(), values.end(), target) != values.end();
}

util::JsonPtr matrix_required_payload() {
    util::JsonObject p;
    p["op"] = util::json_string("SOLVE_LINEAR_SYSTEM");
    p["resolver_policy"] = util::json_string("matrix_buffer.required");
    p["required_primitive"] = util::json_string("matrix_buffer");
    util::JsonArray caps;
    caps.push_back(util::json_string("MATRIX_BUFFER_SOLVE"));
    p["required_capabilities"] = util::json_array(std::move(caps));
    return util::json_object(std::move(p));
}

micronaut::MicronautEntry make_coder_entry() {
    micronaut::MicronautEntry e;
    e.id = "coder";
    e.description = "general code track";
    e.primary_phase = kuhul::Phase::Sek;
    e.category = "cognitive";
    e.capabilities = {"code-generation", "debug", "ast-parse"};
    e.governs = {"solve", "linear", "system", "matrix"};
    e.static_weight = 1.0f;
    e.health = 1.0f;
    e.confidence = 1.0f;
    e.no_eval = true;
    e.active = true;
    return e;
}

micronaut::MicronautEntry make_math_entry() {
    micronaut::MicronautEntry e;
    e.id = "MATH-u";
    e.description = "math resolver";
    e.primary_phase = kuhul::Phase::Sek;
    e.category = "cognitive";
    e.capabilities = {
        "MATH_SYMBOLIC_SOLVE",
        "MATRIX_BUFFER_SOLVE",
        "MATRIX_BUFFER_MATMUL"
    };
    e.governs = {"solve", "linear", "system", "matrix", "equation"};
    e.static_weight = 1.0f;
    e.health = 1.0f;
    e.confidence = 0.5f;
    e.no_eval = true;
    e.active = true;
    return e;
}

micronaut::MicronautEntry make_tool_entry() {
    micronaut::MicronautEntry e;
    e.id = "TOOL-u";
    e.description = "tool routing and command execution";
    e.primary_phase = kuhul::Phase::Sek;
    e.category = "utility";
    e.capabilities = {"tool-execution", "web-research", "command-runner"};
    e.governs = {"tool", "call", "run", "command", "web", "research", "execute"};
    e.static_weight = 1.0f;
    e.health = 1.0f;
    e.confidence = 0.7f;
    e.no_eval = true;
    e.active = true;
    return e;
}

micronaut::MicronautEntry make_chat_entry() {
    micronaut::MicronautEntry e;
    e.id = "CHAT-u";
    e.description = "general conversational fallback";
    e.primary_phase = kuhul::Phase::Wo;
    e.category = "cognitive";
    e.capabilities = {"smalltalk", "chat"};
    e.governs = {"hello", "chat", "talk", "conversation"};
    e.static_weight = 1.0f;
    e.health = 1.0f;
    e.confidence = 0.9f;
    e.no_eval = true;
    e.active = true;
    return e;
}

bool test_policy_filters_then_ranks() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    reg.add(make_coder_entry());
    reg.add(make_math_entry());
    confidence.set("coder", 1.0f);   // intentionally higher, should still be filtered out
    confidence.set("MATH-u", 0.2f);

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("coder", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("coder");
    });
    router.register_handler("MATH-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("math");
    });

    io::RouteRequest req;
    req.query = "SOLVE LINEAR SYSTEM A x = b";
    req.phase = kuhul::Phase::Sek;
    req.payload = matrix_required_payload();

    const auto result = router.route(req);
    if (!expect_true("policy filter applied", result.policy_filter_applied)) return false;
    if (!expect_true("no capability miss", !result.capability_miss)) return false;
    if (!expect_eq("selected resolver", result.micronaut_id, "MATH-u")) return false;
    if (!expect_true("dispatch allowed", result.dispatch_allowed)) return false;
    if (!expect_true("math eligible", contains_value(result.eligible_resolvers, "MATH-u"))) return false;
    if (!expect_true("coder filtered out", contains_value(result.ineligible_resolvers, "coder"))) return false;
    return true;
}

bool test_empty_eligibility_is_capability_miss() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    reg.add(make_coder_entry());
    confidence.set("coder", 1.0f);

    io::RequestRouter router(reg, scorer, "data");
    router.register_handler("coder", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("coder");
    });

    io::RouteRequest req;
    req.query = "SOLVE LINEAR SYSTEM A x = b";
    req.phase = kuhul::Phase::Sek;
    req.payload = matrix_required_payload();

    const auto result = router.route(req);
    if (!expect_true("capability miss raised", result.capability_miss)) return false;
    if (!expect_true("dispatch denied", !result.dispatch_allowed)) return false;
    if (!expect_eq("block reason", result.dispatch_block_reason, "MISS_capability")) return false;
    if (!result.response || !result.response->is_object()) {
        std::cerr << "[FAIL] blocked response shape\n";
        return false;
    }
    const auto status = result.response->get_string("status").value_or("");
    if (!expect_eq("blocked status", status, "miss_capability")) return false;
    return true;
}

bool test_tool_gram_channel_biases_tool_track() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    reg.add(make_tool_entry());
    reg.add(make_chat_entry());

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("TOOL-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("tool");
    });
    router.register_handler("CHAT-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("chat");
    });

    io::RouteRequest req;
    req.query = "please run tool call web research and execute command";
    req.phase = kuhul::Phase::Sek;
    auto result = router.route(req);
    if (!expect_eq("tool track selected", result.micronaut_id, "TOOL-u")) return false;
    if (!expect_true("tool gram score raised", result.tool_gram_score > 0.05f)) return false;
    if (!expect_true("dispatch allowed", result.dispatch_allowed)) return false;
    return true;
}

} // namespace

int main() {
    return test::run("router", [](test::Suite& suite) {
        suite.check("policy_filters_then_ranks", test_policy_filters_then_ranks());
        suite.check("empty_eligibility_is_capability_miss", test_empty_eligibility_is_capability_miss());
        suite.check("tool_gram_channel_biases_tool_track", test_tool_gram_channel_biases_tool_track());
        memory::PerceptionStore perception;
        memory::ConfidenceStore confidence;
        memory::SemanticGraph graph;
        memory::MicronautScorer scorer(perception, confidence, graph);
        micronaut::MicronautRegistry registry;
        registry.add(make_math_entry());
        io::RequestRouter router(registry, scorer, "data");
        int calls = 0;
        router.register_handler("MATH-u", [&](const std::string&, const io::RouteRequest&) {
            ++calls; return util::json_string("executed");
        });
        io::RouteRequest req;
        req.query = "SOLVE LINEAR SYSTEM A x = b";
        req.phase = kuhul::Phase::Sek;
        req.payload = matrix_required_payload();
        suite.check("unbound_denied", !router.route(req).dispatch_allowed && calls == 0);
        req.payload->mut_object()["xcfe_authorized"] = util::json_bool(true);
        req.payload->mut_object()["xcfe_authorize"] = util::json_bool(true);
        suite.check("forged_true_denied", !router.route(req).dispatch_allowed && calls == 0);
        router.set_authorizer([](const io::RouteRequest&, const std::string&) { return false; });
        suite.check("host_denial_not_overridden", !router.route(req).dispatch_allowed && calls == 0);
        router.set_authorizer([](const io::RouteRequest&, const std::string&) -> bool { throw std::runtime_error("policy unavailable"); });
        suite.check("policy_error_denied", !router.route(req).dispatch_allowed && calls == 0);
        router.set_authorizer([](const io::RouteRequest&, const std::string& id) { return id == "MATH-u"; });
        req.payload->mut_object()["xcfe_authorize"] = util::json_bool(false);
        suite.check("either_denial_wins", !router.route(req).dispatch_allowed && calls == 0);
        req.payload->mut_object().erase("xcfe_authorize");
        req.payload->mut_object()["xcfe_authorized"] = util::json_string("true");
        suite.check("malformed_flag_denied", !router.route(req).dispatch_allowed && calls == 0);
        req.payload->mut_object().erase("xcfe_authorized");
        suite.check("host_grant_without_payload_authority", router.route(req).dispatch_allowed && calls == 1);

        // ── Entropy measurement layer (diagnostic; never authorizes) ────────
        // Concentrated distribution: H = 0, normalized 0, no throttle.
        const auto concentrated = io::routing_entropy({4.0, 0.0001});
        suite.check("entropy_concentrated_is_zero", concentrated.bits < 0.01 &&
                    concentrated.normalized < 0.01 && !concentrated.omega_throttled);
        // Uniform over 2 candidates: H = 1 bit, normalized 1.0 -> throttled.
        const auto uniform2 = io::routing_entropy({1.0, 1.0});
        suite.check("entropy_uniform_two_is_one_bit",
                    std::fabs(uniform2.bits - 1.0) < 1e-9 &&
                    std::fabs(uniform2.normalized - 1.0) < 1e-9);
        suite.check("entropy_uniform_is_throttled", uniform2.omega_throttled);
        suite.check("entropy_counts_eligible", uniform2.eligible_count == 2);
        // Uniform over 4: H = 2 bits but normalized still 1.0 — the normalized
        // form is what keeps candidate-count differences readable.
        const auto uniform4 = io::routing_entropy({1.0, 1.0, 1.0, 1.0});
        suite.check("entropy_uniform_four_is_two_bits",
                    std::fabs(uniform4.bits - 2.0) < 1e-9 &&
                    std::fabs(uniform4.normalized - 1.0) < 1e-9);
        // |Eligible| <= 1 and empty sets are not ambiguous.
        const auto single = io::routing_entropy({0.9});
        suite.check("entropy_single_candidate_no_throttle",
                    single.eligible_count == 1 && single.bits == 0.0 && !single.omega_throttled);
        const auto empty = io::routing_entropy({});
        suite.check("entropy_empty_set_no_throttle",
                    empty.eligible_count == 0 && !empty.omega_throttled);
        suite.check("entropy_zero_scores_no_entropy", io::routing_entropy({0.0, 0.0}).bits == 0.0);
        // The threshold is advisory and configurable.
        suite.check("entropy_threshold_is_configurable",
                    !io::routing_entropy({1.0, 1.0}, 1.01).omega_throttled &&
                    io::routing_entropy({1.0, 1.0}, 0.5).omega_throttled);

        // ── Ω gate: blocking, bounded, and never a bypass ───────────────────
        const io::RoutingEntropy flat = io::routing_entropy({1.0, 1.0, 1.0});
        suite.check("gate_flat_field_is_blocked",
                    io::ambiguity_gate(flat, 0) == io::AmbiguityGate::BlockAmbiguity);
        const io::RoutingEntropy collapsed = io::routing_entropy({4.0, 0.0001, 0.0001});
        suite.check("gate_collapsed_field_allows",
                    io::ambiguity_gate(collapsed, 0) == io::AmbiguityGate::Allow);
        // A single eligible candidate is not ambiguous, even if the threshold is met.
        io::RoutingEntropy one;
        one.eligible_count = 1;
        one.omega_throttled = true;
        suite.check("gate_single_candidate_allows",
                    io::ambiguity_gate(one, 99) == io::AmbiguityGate::Allow);
        // Exhaustion is explicit and still blocks — it never falls through to Allow.
        suite.check("gate_exhausts_after_budget",
                    io::ambiguity_gate(flat, io::kAmbiguityMaxAttempts) ==
                        io::AmbiguityGate::ExhaustedAmbiguity);
        suite.check("gate_exhaustion_never_allows",
                    io::ambiguity_gate(flat, 1000) != io::AmbiguityGate::Allow &&
                    io::ambiguity_gate(flat, 1000) == io::AmbiguityGate::ExhaustedAmbiguity);
        suite.check("gate_blocks_until_budget",
                    io::ambiguity_gate(flat, io::kAmbiguityMaxAttempts - 1) ==
                        io::AmbiguityGate::BlockAmbiguity);

        // ── Calibration tuple: whole-field entropy vs local top-two margin ──
        const auto wide = io::wcr_field_stats({0.26, 0.25, 0.25, 0.24});
        suite.check("margin_wide_field_is_thin",
                    wide.top1 == 0.26 && wide.top2 == 0.25 &&
                    std::fabs(wide.margin - 0.01) < 1e-12);
        const auto peaky = io::wcr_field_stats({0.82, 0.07, 0.06, 0.05});
        suite.check("margin_peaky_field_is_wide",
                    std::fabs(peaky.top1 - 0.82) < 1e-12 &&
                    std::fabs(peaky.margin - 0.75) < 1e-12);
        // Similar entropy, different local geometry: the two signals differ.
        const double h_wide = io::routing_entropy({0.26, 0.25, 0.25, 0.24}).normalized;
        const double h_peaky = io::routing_entropy({0.82, 0.07, 0.06, 0.05}).normalized;
        suite.check("entropy_and_margin_are_distinct_signals",
                    h_wide > h_peaky && wide.margin < peaky.margin);
        suite.check("margin_single_candidate_is_its_score",
                    io::wcr_field_stats({0.5}).margin == 0.5 &&
                    io::wcr_field_stats({}).margin == 0.0);
        // Calibration buckets, including the allowed ones.
        suite.check("outcome_bucket_allowed",
                    std::string(io::ambiguity_outcome_label("", 4)) == "allowed");
        suite.check("outcome_bucket_single_eligible",
                    std::string(io::ambiguity_outcome_label("", 1)) == "single_eligible");
        suite.check("outcome_bucket_omega_blocked",
                    std::string(io::ambiguity_outcome_label("MISS_ambiguity", 4)) ==
                        "omega_blocked");
        suite.check("outcome_bucket_omega_exhausted",
                    std::string(io::ambiguity_outcome_label("MISS_ambiguity_exhausted", 4)) ==
                        "omega_exhausted");
        suite.check("outcome_bucket_miss_capability",
                    std::string(io::ambiguity_outcome_label("MISS_capability", 0)) ==
                        "miss_capability");

        // ── Episode identity: pseudonymous, joinable per episode, not per run ─
        const std::string canonical = io::canonical_routing_query("  If   Mode EQUALS Debug  ");
        suite.check("episode_canonicalizes_whitespace_and_case",
                    canonical == "if mode equals debug");
        const std::string id_a = io::episode_id_for(canonical, "nonceA");
        const std::string id_b = io::episode_id_for(canonical, "nonceA");
        const std::string id_c = io::episode_id_for(canonical, "nonceB");
        const std::string id_other = io::episode_id_for("why did it fail", "nonceA");
        suite.check("episode_id_is_deterministic", id_a == id_b);
        suite.check("episode_id_is_run_scoped", id_a != id_c);
        suite.check("episode_id_separates_queries", id_a != id_other);
        suite.check("episode_id_is_truncated_hex",
                    id_a.size() == io::kEpisodeIdChars &&
                    id_a.find_first_not_of("0123456789abcdef") == std::string::npos);
        // Different spellings of the same question share one episode.
        suite.check("episode_joins_attempts",
                    io::episode_id_for(io::canonical_routing_query("IF MODE equals debug"),
                                       "nonceA") == id_a);

        // ── Xul gate: B_{t+1} != B_t => Xul AND Authorized(dB) ───────────────
        // Non-manufacturability and non-reusability are compile-time properties.
        static_assert(!std::is_default_constructible_v<kuhul::CommitAuthority>,
                      "a writer must not be able to manufacture commit authority");
        static_assert(!std::is_constructible_v<kuhul::CommitAuthority, std::string, std::string>,
                      "only the Xul gate may mint commit authority");
        static_assert(!std::is_copy_constructible_v<kuhul::CommitAuthority>,
                      "commit authority is a one-shot certificate, not a reusable permission");

        std::size_t writes = 0;
        const auto mutate = [&writes] { ++writes; };
        const std::string scope = "tracks";
        const std::string proposal = "proposal-A";
        kuhul::CommitDecision ok;
        ok.proposal_verified = true;
        ok.mutation_authorized = true;
        ok.scope_permitted = true;
        ok.phase = kuhul::Phase::Xul;

        // An unbound gate denies; it never grants by default.
        kuhul::XulGate unbound;
        suite.check("xul_unbound_gate_denies",
                    !unbound.authorize_commit(ok, scope, proposal).has_value());

        kuhul::XulGate gate;
        gate.set_authorizer([](kuhul::Phase phase, const std::string& s) {
            return phase == kuhul::Phase::Xul && !s.empty();
        });

        // Every non-Xul phase is denied even with a permissive authorizer.
        for (auto phase : {kuhul::Phase::Pop, kuhul::Phase::Wo, kuhul::Phase::Yax,
                           kuhul::Phase::Sek, kuhul::Phase::Chen}) {
            auto decision = ok;
            decision.phase = phase;
            suite.check(std::string("xul_denies_phase_") + kuhul::phase_name(phase),
                        !gate.authorize_commit(decision, scope, proposal).has_value());
        }
        // Xul alone is not enough: each precondition is required, not suggested.
        for (std::size_t i = 0; i < 3; ++i) {
            auto decision = ok;
            if (i == 0) decision.proposal_verified = false;
            if (i == 1) decision.mutation_authorized = false;
            if (i == 2) decision.scope_permitted = false;
            suite.check(std::string("xul_requires_precondition_") + std::to_string(i),
                        !gate.authorize_commit(decision, scope, proposal).has_value());
        }
        suite.check("xul_denies_empty_scope",
                    !gate.authorize_commit(ok, "", proposal).has_value());
        suite.check("xul_denies_empty_proposal_identity",
                    !gate.authorize_commit(ok, scope, "").has_value());

        // A denying authorizer at Xul still denies.
        kuhul::XulGate denying;
        denying.set_authorizer([](kuhul::Phase, const std::string&) { return false; });
        suite.check("xul_denying_authorizer_denies",
                    !denying.authorize_commit(ok, scope, proposal).has_value());

        // The authorized path: mint, then write, and the mutation happens once.
        auto authority = gate.authorize_commit(ok, scope, proposal);
        suite.check("xul_authorized_commit_mints", authority.has_value());
        suite.check("xul_write_without_authority_denied",
                    !gate.write(std::nullopt, scope, proposal, mutate) && writes == 0);

        // Transaction binding: a token minted for proposal A cannot commit B.
        auto for_a = gate.authorize_commit(ok, scope, proposal);
        suite.check("xul_identity_cannot_swap",
                    !gate.write(std::move(for_a), scope, "proposal-B", mutate) && writes == 0);

        suite.check("xul_write_with_authority_allowed",
                    gate.write(std::move(authority), scope, proposal, mutate) && writes == 1);
        // Consumed by the commit: the moved-from token carries no usable
        // scope/identity, so it cannot authorize an unrelated future commit.
        suite.check("xul_authority_is_single_use",
                    !gate.write(std::move(authority), scope, proposal, mutate) && writes == 1);

        // A fresh token cannot be replayed against a wider scope.
        auto for_scope = gate.authorize_commit(ok, scope, proposal);
        suite.check("xul_scope_cannot_widen",
                    !gate.write(std::move(for_scope), "keys", proposal, mutate) && writes == 1);

        // ── Key-store invariant: usage is not evidence ──────────────────────
        // Load(Store(C,U)) = (C,U), never (max(C, f(U)), U). Accumulated
        // retrievals must not resurrect a poorly-supported key, and a key that is
        // highly familiar but contradicted must remain representable.
        {
            micronaut::MicronautRegistry probe_reg;
            const auto dir = std::filesystem::temp_directory_path();
            const auto in_path = (dir / "khanary_key_probe.json").string();
            const auto out_path = (dir / "khanary_key_probe_out.json").string();
            util::JsonObject entry;
            entry["key"] = util::json_string("teenager paradox probe");
            entry["response"] = util::json_string("a familiar answer");
            entry["hash"] = util::json_string("deadbeef");
            entry["hits"] = util::json_number(100000.0);   // highly familiar
            entry["confidence"] = util::json_number(0.05); // poorly supported
            util::JsonObject root;
            root["entries"] = util::json_array({util::json_object(std::move(entry))});
            {
                std::ofstream sink(in_path);
                sink << util::stringify(util::json_object(std::move(root)));
            }

            micronaut::Factory probe(probe_reg, 0.7f);
            const std::string k = "teenager paradox probe";
            suite.check("keystore_load_accepts_entries", probe.load_state(in_path));
            // Evidence confidence below 0.5 is representable: clamp01's second
            // argument is a fallback for non-finite input, not a lower bound.
            suite.check("keystore_evidence_confidence_is_representable",
                        std::fabs(probe.key_confidence(k) - 0.05f) < 1e-6f);

            const uint64_t hits_before = probe.key_hits(k);
            const float conf_before = probe.key_confidence(k);
            const auto recalled = probe.lookup_key(k);
            suite.check("keystore_retrieval_succeeds", recalled.has_value());
            suite.check("keystore_retrieval_increases_usage", probe.key_hits(k) > hits_before);

            // Usage and evidence are independent axes: retrieval increments U,
            // but does not promote C. Confidence changes belong to feedback.
            suite.check("keystore_retrieval_preserves_evidence_confidence",
                        std::fabs(probe.key_confidence(k) - conf_before) < 1e-6f);
            std::error_code ignored;
            std::filesystem::remove(in_path, ignored);
            std::filesystem::remove(out_path, ignored);
        }
    }, 63);
}
