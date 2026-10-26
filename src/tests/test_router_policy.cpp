#include "../io/request_router.h"
#include "../kuhul/xul_gate.h"
#include "../memory/perception.h"
#include "../memory/confidence.h"
#include "../memory/semantic_graph.h"
#include "../memory/scoring.h"
#include "../micronaut/micronaut_registry.h"
#include "../micronaut/micronaut_transpiler.h"
#include "../micronaut/mou_router.h"
#include "../micronaut/factory.h"
#include "../decision/decision_mu.h"
#include "../util/json.h"
#include "conformance.h"

#include "../kuhul/state_algebra.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numbers>
#include <optional>
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

micronaut::MicronautEntry make_mou_entry(
    const std::string& id,
    kuhul::Phase phase,
    float x,
    float y,
    float z,
    bool active = true)
{
    micronaut::MicronautEntry e;
    e.id = id;
    e.description = "mou test entry";
    e.primary_phase = phase;
    e.category = "cognitive";
    e.capabilities = {"route"};
    e.governs = {"route", "test"};
    e.vector_meta.x = x;
    e.vector_meta.y = y;
    e.vector_meta.z = z;
    e.vector_meta.declared = true;
    e.vector_meta.source = "test";
    e.static_weight = 1.0f;
    e.health = 1.0f;
    e.confidence = 0.5f;
    e.no_eval = true;
    e.active = active;
    return e;
}

micronaut::MicronautEntry make_scheme_entry(
    const std::string& id,
    kuhul::Phase phase,
    const std::vector<std::string>& schemes,
    const std::vector<std::string>& verbs,
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& governs,
    const std::vector<std::string>& capabilities)
{
    micronaut::MicronautEntry e;
    e.id = id;
    e.description = "scheme test entry";
    e.primary_phase = phase;
    e.category = "utility";
    e.route_contract.schemes = schemes;
    e.route_contract.verbs = verbs;
    e.route_contract.tokens = tokens;
    e.governs.insert(governs.begin(), governs.end());
    e.capabilities = capabilities;
    e.static_weight = 1.0f;
    e.health = 1.0f;
    e.confidence = 0.8f;
    e.no_eval = true;
    e.active = true;
    return e;
}

micronaut::MicronautEntry make_decision_entry() {
    return make_scheme_entry("DECISION-µ", kuhul::Phase::Sek,
                             {"decision"}, {"choose", "select", "evaluate"},
                             {"decision", "choose", "utility", "loss", "branch"},
                             {"decision", "choose", "action", "utility", "loss"},
                             {"DECISION_EXPECTED_UTILITY", "DECISION_MAXIMIN",
                              "DECISION_MINIMAX", "DECISION_UNDER_UNCERTAINTY"});
}

float vec_norm_for_test(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

bool json_array_contains_string(const util::JsonPtr& root,
                                const std::string& key,
                                const std::string& value) {
    if (!root || !root->is_object()) return false;
    const auto* arr = root->get(key);
    if (!arr || !arr->is_array()) return false;
    for (const auto& item : arr->as_array()) {
        if (item && item->is_string() && item->as_string() == value) return true;
    }
    return false;
}

util::JsonPtr make_decision_payload(bool include_probabilities,
                                    bool explicit_expected_utility,
                                    bool unknown_rain_branch) {
    util::JsonObject rain;
    rain["id"] = util::json_string("rain");
    rain["epistemic_state"] = util::json_string(unknown_rain_branch ? "UNKNOWN" : "SUPPORTED");
    if (include_probabilities) rain["probability"] = util::json_number(0.7);

    util::JsonObject clear;
    clear["id"] = util::json_string("clear");
    clear["epistemic_state"] = util::json_string("SUPPORTED");
    if (include_probabilities) clear["probability"] = util::json_number(0.3);

    util::JsonObject carry_outcomes;
    carry_outcomes["rain"] = util::json_object({{"utility", util::json_number(10.0)}});
    carry_outcomes["clear"] = util::json_object({{"utility", util::json_number(-1.0)}});
    util::JsonObject carry;
    carry["id"] = util::json_string("carry_umbrella");
    carry["outcomes"] = util::json_object(std::move(carry_outcomes));

    util::JsonObject leave_outcomes;
    leave_outcomes["rain"] = util::json_object({{"utility", util::json_number(-8.0)}});
    leave_outcomes["clear"] = util::json_object({{"utility", util::json_number(2.0)}});
    util::JsonObject leave;
    leave["id"] = util::json_string("leave_umbrella");
    leave["outcomes"] = util::json_object(std::move(leave_outcomes));

    util::JsonObject decision;
    decision["objective"] = util::json_string("maximize_utility");
    if (explicit_expected_utility)
        decision["decision_model"] = util::json_string("expected_utility");
    decision["branches"] = util::json_array({
        util::json_object(std::move(rain)),
        util::json_object(std::move(clear))
    });
    decision["actions"] = util::json_array({
        util::json_object(std::move(carry)),
        util::json_object(std::move(leave))
    });

    util::JsonObject root;
    root["decision"] = util::json_object(std::move(decision));
    return util::json_object(std::move(root));
}

std::optional<kuhul::CommitAuthority> make_test_commit_authority(
    const std::string& scope,
    const std::string& identity) {
    kuhul::XulGate gate;
    gate.set_authorizer([](kuhul::Phase phase, const std::string& value) {
        return phase == kuhul::Phase::Xul && !value.empty();
    });
    kuhul::CommitDecision decision;
    decision.proposal_verified = true;
    decision.mutation_authorized = true;
    decision.scope_permitted = true;
    decision.phase = kuhul::Phase::Xul;
    return gate.authorize_commit(decision, scope, identity);
}

bool commit_factory_candidate(micronaut::Factory& factory,
                              micronaut::FactoryCandidate candidate) {
    auto authority = make_test_commit_authority(candidate.scope, candidate.proposal_identity);
    if (!authority) return false;
    return factory.commit(std::move(candidate), std::move(*authority)).ok;
}

io::RouteResult route_decision_payload(const util::JsonPtr& payload,
                                       const std::string& query) {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;
    reg.add(make_decision_entry());
    confidence.set("DECISION-µ", 0.9f);

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("DECISION-µ", [](const std::string&, const io::RouteRequest& req) {
        return decision::evaluate_request(req.query, req.payload.get());
    });

    io::RouteRequest req;
    req.query = query;
    req.phase = kuhul::Phase::Sek;
    req.payload = payload;
    return router.route(req);
}

bool test_factory_vector_track_commits_monotonic_frame() {
    micronaut::MicronautRegistry reg;
    reg.add(make_mou_entry("VECTOR-TEST-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
    micronaut::Factory factory(reg);

    micronaut::VectorTrackObservation observation;
    observation.x = 0.0f;
    observation.y = 1.0f;
    observation.z = 0.0f;
    observation.evidence_weight = 0.5f;
    observation.run_id = "run-1";
    observation.source = "unit";
    observation.frame_id = 1;

    auto candidate = factory.propose_vector_track("VECTOR-TEST-u", observation);
    if (!expect_true("vector track proposal exists", candidate.has_value())) return false;
    if (!expect_eq("vector track scope", candidate->scope, "micronaut-vector-tracking")) return false;
    if (!expect_eq("vector track identity", candidate->proposal_identity,
                   "vector_track:VECTOR-TEST-u:1")) return false;
    if (!expect_true("vector track commit", commit_factory_candidate(factory, std::move(*candidate)))) return false;

    const auto* entry = reg.get("VECTOR-TEST-u");
    if (!expect_true("vector track entry exists", entry != nullptr)) return false;
    if (!expect_true("vector track declares vector", entry->vector_meta.declared)) return false;
    if (!expect_eq("vector track source", entry->vector_meta.source, "unit:run-1:f1")) return false;
    const auto frame = factory.vector_tracking_frame("VECTOR-TEST-u");
    if (!expect_true("vector tracking frame stored", frame.has_value())) return false;
    if (!expect_true("vector tracking frame id", frame->frame_id == 1)) return false;
    if (!expect_true("vector tracking base frame", frame->base_frame == 0)) return false;

    auto duplicate = factory.propose_vector_track("VECTOR-TEST-u", observation);
    if (!expect_true("vector track rejects non-monotonic frame", !duplicate.has_value())) return false;
    return true;
}

bool test_factory_vector_track_bounds_drift_and_normalizes() {
    micronaut::MicronautRegistry reg;
    reg.add(make_mou_entry("VECTOR-DRIFT-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
    micronaut::Factory factory(reg);

    micronaut::VectorTrackObservation observation;
    observation.x = -100.0f;
    observation.y = 0.0f;
    observation.z = 0.0f;
    observation.evidence_weight = 1.0f;
    observation.run_id = "run-drift";
    observation.source = "unit";
    observation.frame_id = 1;

    auto candidate = factory.propose_vector_track("VECTOR-DRIFT-u", observation);
    if (!expect_true("drift proposal exists", candidate.has_value())) return false;
    if (!expect_true("drift is bounded", candidate->vector_frame.drift <= 0.2501f)) return false;
    if (!expect_true("drift proposal normalized",
                     candidate->vector_frame.x > 0.9f &&
                     candidate->vector_frame.y > 0.2f &&
                     std::fabs(candidate->vector_frame.z) < 1e-3f &&
                     std::fabs(vec_norm_for_test(candidate->vector_frame.x,
                                                 candidate->vector_frame.y,
                                                 candidate->vector_frame.z) - 1.0f) < 1e-3f)) return false;
    if (!expect_true("drift proposal commits", commit_factory_candidate(factory, std::move(*candidate)))) return false;
    const auto* entry = reg.get("VECTOR-DRIFT-u");
    if (!expect_true("drift entry normalized",
                     entry && entry->vector_meta.x > 0.9f && entry->vector_meta.y > 0.2f)) return false;
    return true;
}

bool test_factory_vector_tracking_state_roundtrip() {
    const auto base = std::filesystem::current_path();
    const auto json_path = base / "factory-vector-state-test.json";
    const auto bin_path = base / "factory-vector-state-test.bin";

    {
        micronaut::MicronautRegistry reg;
        reg.add(make_mou_entry("VECTOR-PERSIST-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
        micronaut::Factory factory(reg);
        micronaut::VectorTrackObservation observation;
        observation.x = 0.0f;
        observation.y = 1.0f;
        observation.z = 0.0f;
        observation.evidence_weight = 0.25f;
        observation.run_id = "run-persist";
        observation.source = "unit";
        observation.frame_id = 1;
        auto candidate = factory.propose_vector_track("VECTOR-PERSIST-u", observation);
        if (!expect_true("persist proposal exists", candidate.has_value())) return false;
        if (!expect_true("persist proposal commits", commit_factory_candidate(factory, std::move(*candidate)))) return false;
        auto authority = make_test_commit_authority("keystore", "factory-vector-persist-test");
        if (!expect_true("persist authority minted", authority.has_value())) return false;
        if (!expect_true("factory state persisted",
                         factory.persist(std::move(*authority), json_path.string(), bin_path.string()))) return false;
    }

    {
        micronaut::MicronautRegistry reg;
        reg.add(make_mou_entry("VECTOR-PERSIST-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
        micronaut::Factory restored(reg);
        if (!expect_true("json state restored", restored.load_state(json_path.string()))) return false;
        const auto frame = restored.vector_tracking_frame("VECTOR-PERSIST-u");
        const auto* entry = reg.get("VECTOR-PERSIST-u");
        if (!expect_true("json frame restored", frame && frame->frame_id == 1)) return false;
        if (!expect_true("json vector applied", entry && entry->vector_meta.y > 0.05f)) return false;
    }

    {
        micronaut::MicronautRegistry reg;
        reg.add(make_mou_entry("VECTOR-PERSIST-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
        micronaut::Factory restored(reg);
        if (!expect_true("binary state restored", restored.load_state_binary(bin_path.string()))) return false;
        const auto frame = restored.vector_tracking_frame("VECTOR-PERSIST-u");
        const auto* entry = reg.get("VECTOR-PERSIST-u");
        if (!expect_true("binary frame restored", frame && frame->frame_id == 1)) return false;
        if (!expect_true("binary vector applied", entry && entry->vector_meta.y > 0.05f)) return false;
    }

    std::error_code ec;
    std::filesystem::remove(json_path, ec);
    std::filesystem::remove(bin_path, ec);
    return true;
}

bool test_micronaut_attraction_capture_formula() {
    micronaut::GravityField field;
    field.register_nodes("HIGH-u", {"solve", "matrix"}, kuhul::Phase::Sek,
                         1.0f, 0.0f, 0.0f, true,
                         1.0f, 0.9f, 1.0f, 0.0f,
                         true, true, true);
    field.register_nodes("LOW-u", {"solve", "matrix"}, kuhul::Phase::Sek,
                         1.0f, 0.0f, 0.0f, true,
                         1.0f, 0.2f, 1.0f, 0.0f,
                         true, true, true);
    field.register_nodes("ENTROPY-u", {"solve", "matrix"}, kuhul::Phase::Sek,
                         1.0f, 0.0f, 0.0f, true,
                         1.0f, 1.0f, 1.0f, 999.0f,
                         true, true, true);
    field.register_nodes("DENIED-u", {"solve", "matrix"}, kuhul::Phase::Sek,
                         1.0f, 0.0f, 0.0f, true,
                         1.0f, 1.0f, 1.0f, 0.0f,
                         false, true, true);

    const auto rows = field.capture({"solve", "matrix"}, kuhul::Phase::Sek, 0.1f);
    auto find = [&](const std::string& id) -> const micronaut::MicronautAttraction* {
        for (const auto& row : rows)
            if (row.micronaut_id == id) return &row;
        return nullptr;
    };

    const auto* high = find("HIGH-u");
    const auto* low = find("LOW-u");
    const auto* entropy = find("ENTROPY-u");
    const auto* denied = find("DENIED-u");
    if (!expect_true("attraction high exists", high != nullptr)) return false;
    if (!expect_true("attraction low exists", low != nullptr)) return false;
    if (!expect_true("attraction entropy exists", entropy != nullptr)) return false;
    if (!expect_true("attraction denied exists", denied != nullptr)) return false;

    const float expected_phi =
        high->semantic_mass * high->geometric_alignment * high->wcr_prior;
    if (!expect_true("attraction formula multiplies mass alignment wcr",
                     std::fabs(high->phi - expected_phi) < 1e-6f)) return false;
    if (!expect_true("high confidence increases attraction",
                     high->capture_score > low->capture_score)) return false;
    if (!expect_true("entropy overbalances capture",
                     !entropy->captured && entropy->capture_score == 0.0f)) return false;
    if (!expect_true("policy denial blocks capture",
                     !denied->legal && !denied->captured && denied->capture_score == 0.0f)) return false;
    if (!expect_eq("best match uses attraction", field.best_match({"solve", "matrix"},
                                                                  kuhul::Phase::Sek, 0.1f),
                   "HIGH-u")) return false;
    return true;
}

bool test_micronaut_transpiler_compiles_key_contract() {
    auto entry = make_decision_entry();
    entry.route_contract.key_patterns = {"decision:choose:*", "decision:choose:*"};
    entry.route_contract.key_templates = {"decision:result:selected"};
    entry.route_contract.key_grammar = {"schema:scfml.pattern_learning.v1"};
    util::JsonObject profile;
    profile["projection_only"] = util::json_bool(true);
    profile["authority_boundary"] = util::json_string("test_profile");
    entry.profile_projection = util::json_object(std::move(profile));

    const auto row = micronaut::MicronautTranspiler::compile_entry(entry);
    if (!expect_true("micronaut transpiler row is object", row && row->is_object())) return false;
    if (!expect_eq("micronaut transpiler schema", row->get_string("schema").value_or(""),
                   "key.grammar.micronaut_route.v1")) return false;
    if (!expect_eq("micronaut transpiler track", row->get_string("track").value_or(""),
                   "DECISION-µ")) return false;
    if (!expect_true("micronaut transpiler fact boundary",
                     !row->get_bool("fact_authority").value_or(true))) return false;
    const auto* contract = row->get("route_contract");
    if (!expect_true("micronaut transpiler contract object", contract && contract->is_object())) return false;
    const auto* patterns = contract->get("key_patterns");
    if (!expect_true("micronaut transpiler key patterns", patterns && patterns->is_array() &&
                     patterns->as_array().size() == 1)) return false;
    const auto* grammar = contract->get("key_grammar");
    if (!expect_true("micronaut transpiler key grammar", grammar && grammar->is_array() &&
                     grammar->as_array().size() == 1)) return false;
    const auto* klass = row->get("micronaut_class");
    if (!expect_true("micronaut class projection object", klass && klass->is_object())) return false;
    if (!expect_eq("micronaut class schema", klass->get_string("schema").value_or(""),
                   "micronaut.class_projection.example.v1")) return false;
    if (!expect_true("micronaut class is example-only",
                     klass->get_bool("example_only").value_or(false))) return false;
    const auto* identity = klass->get("identity");
    if (!expect_true("micronaut class identity", identity && identity->is_object() &&
                     identity->get_string("name").value_or("") == "DECISION-µ")) return false;
    const auto* profile_row = row->get("micronaut_profile");
    if (!expect_true("micronaut profile projection surfaced",
                     profile_row && profile_row->is_object() &&
                     profile_row->get_bool("projection_only").value_or(false))) return false;
    return true;
}

bool test_mou_router_phase_filter_and_weight_norm() {
    micronaut::MicronautRegistry reg;
    memory::ConfidenceStore confidence;

    reg.add(make_mou_entry("CHAT-u", kuhul::Phase::Wo, 1.0f, 0.0f, 0.0f));
    reg.add(make_mou_entry("TOOL-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
    reg.add(make_mou_entry("MATH-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));

    confidence.set("CHAT-u", 1.0f);
    confidence.set("TOOL-u", 0.8f);
    confidence.set("MATH-u", 0.2f);

    micronaut::MouRouter router(reg, confidence);
    router.sync_index();
    router.sync_bias();

    const auto allocs = router.route(kuhul::Phase::Sek, {"tool", "call", "web"}, 2);
    if (!expect_true("mou phase route nonempty", !allocs.empty())) return false;
    if (!expect_true("mou phase route bounded", allocs.size() <= 2)) return false;

    bool all_at_phase = true;
    float sum = 0.0f;
    for (const auto& alloc : allocs) {
        sum += alloc.weight;
        const auto* entry = reg.get(alloc.id);
        if (!entry || entry->primary_phase != kuhul::Phase::Sek) {
            all_at_phase = false;
            break;
        }
    }

    if (!expect_true("mou phase filter keeps requested phase", all_at_phase)) return false;
    if (!expect_true("mou weights normalized", std::fabs(sum - 1.0f) < 1e-4f)) return false;
    if (!expect_eq("mou phase route prefers higher confidence", allocs.front().id, "TOOL-u")) return false;
    return true;
}

bool test_mou_router_phase_fallback_unfiltered() {
    micronaut::MicronautRegistry reg;
    memory::ConfidenceStore confidence;

    reg.add(make_mou_entry("CHAT-u", kuhul::Phase::Wo, 1.0f, 0.0f, 0.0f));
    reg.add(make_mou_entry("TOOL-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));

    confidence.set("CHAT-u", 1.0f);
    confidence.set("TOOL-u", 0.5f);

    micronaut::MouRouter router(reg, confidence);
    router.sync_index();
    router.sync_bias();

    const auto allocs = router.route(kuhul::Phase::Chen, {"status"}, 2);
    if (!expect_true("mou phase fallback nonempty", !allocs.empty())) return false;

    bool contains_non_phase = false;
    float sum = 0.0f;
    for (const auto& alloc : allocs) {
        sum += alloc.weight;
        const auto* entry = reg.get(alloc.id);
        if (entry && entry->primary_phase != kuhul::Phase::Chen)
            contains_non_phase = true;
    }

    if (!expect_true("mou fallback returns unfiltered entries", contains_non_phase)) return false;
    if (!expect_true("mou fallback weights normalized", std::fabs(sum - 1.0f) < 1e-4f)) return false;
    return true;
}

bool test_mou_router_sync_bias_refresh() {
    micronaut::MicronautRegistry reg;
    memory::ConfidenceStore confidence;

    reg.add(make_mou_entry("A-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
    reg.add(make_mou_entry("B-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));

    confidence.set("A-u", 0.1f);
    confidence.set("B-u", 0.9f);

    micronaut::MouRouter router(reg, confidence);
    router.sync_index();
    router.sync_bias();

    auto first = router.route(kuhul::Phase::Sek, {"bias", "probe"}, 1);
    if (!expect_true("mou bias initial route exists", !first.empty())) return false;
    if (!expect_eq("mou bias initial winner", first.front().id, "B-u")) return false;

    confidence.set("A-u", 1.0f);
    confidence.set("B-u", 0.0f);

    auto stale = router.route(kuhul::Phase::Sek, {"bias", "probe"}, 1);
    if (!expect_true("mou stale route exists", !stale.empty())) return false;
    if (!expect_eq("mou bias remains stale before sync", stale.front().id, "B-u")) return false;

    router.sync_bias();
    auto refreshed = router.route(kuhul::Phase::Sek, {"bias", "probe"}, 1);
    if (!expect_true("mou refreshed route exists", !refreshed.empty())) return false;
    if (!expect_eq("mou bias winner flips after sync", refreshed.front().id, "A-u")) return false;
    return true;
}

bool test_mou_router_dispatch_variance_contract() {
    micronaut::MicronautRegistry reg;
    memory::ConfidenceStore confidence;

    reg.add(make_mou_entry("ALPHA-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
    reg.add(make_mou_entry("BETA-u", kuhul::Phase::Sek, 0.0f, 1.0f, 0.0f));
    confidence.set("ALPHA-u", 0.7f);
    confidence.set("BETA-u", 0.6f);

    micronaut::MouRouter router(reg, confidence);
    router.sync_index();
    router.sync_bias();

    for (int i = 0; i < 12; ++i) {
        const auto allocs = router.route(kuhul::Phase::Sek, {"variance", "probe", std::to_string(i % 2)}, 2);
        if (!allocs.empty())
            router.record_dispatch(allocs.front().id);
    }

    // Unknown IDs are ignored (no throw, no mutation requirement).
    router.record_dispatch("UNKNOWN-u");

    const float sigma2 = router.routing_variance(8);
    if (!expect_true("mou variance is finite and non-negative",
                     std::isfinite(sigma2) && sigma2 >= 0.0f)) return false;
    if (!expect_true("mou index count tracks registry", router.n_micronauts() == 2u)) return false;
    return true;
}

bool test_scheme_uri_routes_to_declared_micronaut() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    reg.add(make_scheme_entry("MATH-u", kuhul::Phase::Sek,
                              {"math"}, {"solve"}, {"algebra"},
                              {"algebra", "equation"}, {"solve"}));
    reg.add(make_scheme_entry("CHAT-u", kuhul::Phase::Sek,
                              {"chat"}, {"talk"}, {"chat"},
                              {"chat", "talk"}, {"smalltalk"}));
    confidence.set("MATH-u", 0.9f);
    confidence.set("CHAT-u", 0.3f);

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("MATH-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("math");
    });
    router.register_handler("CHAT-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("chat");
    });

    io::RouteRequest req;
    req.query = "math://solve?expr=2%2B2 algebra";
    req.phase = kuhul::Phase::Sek;
    const auto result = router.route(req);
    if (!expect_eq("scheme uri selected resolver", result.micronaut_id, "MATH-u")) return false;
    if (!expect_eq("scheme uri selected op", result.selected_task_op, "solve")) return false;
    if (!expect_true("scheme uri dispatch allowed", result.dispatch_allowed)) return false;
    return true;
}

bool test_scheme_uri_verb_filtering() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    reg.add(make_scheme_entry("MATH-u", kuhul::Phase::Sek,
                              {"math"}, {"solve"}, {"algebra"},
                              {"algebra"}, {"solve"}));
    reg.add(make_scheme_entry("REASON-u", kuhul::Phase::Sek,
                              {"math"}, {"prove"}, {"theorem"},
                              {"theorem", "proof"}, {"prove"}));
    confidence.set("MATH-u", 0.9f);
    confidence.set("REASON-u", 0.8f);

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("MATH-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("math");
    });
    router.register_handler("REASON-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("reason");
    });

    io::RouteRequest req;
    req.query = "math://prove?topic=theorem";
    req.phase = kuhul::Phase::Sek;
    const auto result = router.route(req);
    if (!expect_eq("scheme uri verb-filter resolver", result.micronaut_id, "REASON-u")) return false;
    if (!expect_eq("scheme uri verb-filter op", result.selected_task_op, "prove")) return false;
    return true;
}

bool test_scheme_uri_scopes_explicit_decision_field() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    reg.add(make_decision_entry());
    reg.add(make_scheme_entry("MATH-u", kuhul::Phase::Sek,
                              {"math"}, {"solve"}, {"algebra"},
                              {"algebra", "equation"}, {"solve"}));
    reg.add(make_scheme_entry("PLAN-u", kuhul::Phase::Sek,
                              {"plan"}, {"outline"}, {"plan"},
                              {"plan", "outline"}, {"outline"}));
    reg.add(make_scheme_entry("CHAT-u", kuhul::Phase::Sek,
                              {"chat"}, {"talk"}, {"chat"},
                              {"chat", "talk"}, {"smalltalk"}));
    confidence.set("DECISION-µ", 0.9f);
    confidence.set("MATH-u", 0.9f);
    confidence.set("PLAN-u", 0.9f);
    confidence.set("CHAT-u", 0.9f);

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("DECISION-µ", [](const std::string&, const io::RouteRequest& req) {
        return decision::evaluate_request(req.query, req.payload.get());
    });
    router.register_handler("MATH-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("math");
    });
    router.register_handler("PLAN-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("plan");
    });
    router.register_handler("CHAT-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("chat");
    });

    io::RouteRequest req;
    req.query = "decision://choose";
    req.phase = kuhul::Phase::Sek;
    req.payload = make_decision_payload(true, true, false);
    const auto result = router.route(req);
    if (!expect_eq("explicit decision uri resolver", result.micronaut_id, "DECISION-µ")) return false;
    if (!expect_true("explicit decision uri dispatch allowed", result.dispatch_allowed)) return false;
    if (!expect_true("explicit decision uri scopes eligible field", result.eligible_count == 1)) return false;
    if (!expect_eq("explicit decision uri handler", result.handler_id, "DECISION-µ")) return false;
    if (!expect_eq("explicit decision uri action",
                   result.response->get_string("action").value_or(""), "carry_umbrella")) return false;
    return true;
}

bool test_scheme_uri_id_alias_default() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    auto plan = make_scheme_entry("PLAN-u", kuhul::Phase::Sek,
                                  {}, {}, {"outline"},
                                  {"plan", "outline"}, {"outline"});
    reg.add(plan);
    reg.add(make_scheme_entry("CHAT-u", kuhul::Phase::Sek,
                              {"chat"}, {"talk"}, {"chat"},
                              {"chat"}, {"smalltalk"}));
    confidence.set("PLAN-u", 0.6f);
    confidence.set("CHAT-u", 0.6f);

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("PLAN-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("plan");
    });
    router.register_handler("CHAT-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("chat");
    });

    io::RouteRequest req;
    req.query = "plan://outline";
    req.phase = kuhul::Phase::Sek;
    const auto result = router.route(req);
    if (!expect_eq("scheme uri id-alias resolver", result.micronaut_id, "PLAN-u")) return false;
    return true;
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

bool test_epistemic_contract_for_miss_and_unknown() {
    io::RouteResult miss;
    miss.dispatch_block_reason = "MISS_capability";
    miss.epistemic_state = "MISS";
    miss.observation_authority = false;
    miss.epistemic_causes = {"local_miss", "insufficient_observability"};

    util::JsonObject miss_response;
    miss_response["status"] = util::json_string("miss_capability");
    miss_response["epistemic_state"] = util::json_string(miss.epistemic_state);
    miss_response["observation_authority"] = util::json_bool(miss.observation_authority);
    util::JsonArray miss_causes;
    for (const auto& cause : miss.epistemic_causes)
        miss_causes.push_back(util::json_string(cause));
    miss_response["epistemic_causes"] = util::json_array(std::move(miss_causes));
    miss.response = util::json_object(std::move(miss_response));

    if (!expect_eq("capability miss epistemic state", miss.epistemic_state, "MISS")) return false;
    if (!expect_true("capability miss denies observation authority", !miss.observation_authority)) return false;
    if (!expect_eq("capability miss response state", miss.response->get_string("epistemic_state").value_or(""), "MISS")) return false;

    io::RouteResult unknown;
    unknown.dispatch_block_reason = "MISS_ambiguity";
    unknown.epistemic_state = "UNKNOWN";
    unknown.observation_authority = false;
    unknown.epistemic_causes = {"local_miss", "insufficient_observability", "cross_track_evidence_not_checked"};
    util::JsonObject unknown_response;
    unknown_response["status"] = util::json_string("miss_ambiguity");
    unknown_response["epistemic_state"] = util::json_string(unknown.epistemic_state);
    unknown_response["observation_authority"] = util::json_bool(unknown.observation_authority);
    util::JsonArray unknown_causes;
    for (const auto& cause : unknown.epistemic_causes)
        unknown_causes.push_back(util::json_string(cause));
    unknown_response["epistemic_causes"] = util::json_array(std::move(unknown_causes));
    unknown.response = util::json_object(std::move(unknown_response));

    if (!expect_eq("ambiguity epistemic state", unknown.epistemic_state, "UNKNOWN")) return false;
    if (!expect_true("ambiguity gate blocks flat field",
                     io::ambiguity_gate(io::routing_entropy({1.0, 1.0}), 0) == io::AmbiguityGate::BlockAmbiguity)) return false;
    if (!expect_eq("ambiguity response state", unknown.response->get_string("epistemic_state").value_or(""), "UNKNOWN")) return false;
    return true;
}

bool test_decision_expected_utility_selects_action_without_fact_collapse() {
    const auto result = route_decision_payload(
        make_decision_payload(true, true, false), "decision://choose");
    if (!expect_eq("decision scheme resolver", result.micronaut_id, "DECISION-µ")) return false;
    if (!expect_eq("decision scheme op", result.selected_task_op, "choose")) return false;
    if (!expect_true("decision dispatch allowed", result.dispatch_allowed)) return false;
    if (!result.response || !result.response->is_object()) {
        std::cerr << "[FAIL] decision response shape\n";
        return false;
    }
    if (!expect_eq("decision status", result.response->get_string("status").value_or(""),
                   "decision_selected")) return false;
    if (!expect_eq("decision action", result.response->get_string("action").value_or(""),
                   "carry_umbrella")) return false;
    if (!expect_eq("decision model", result.response->get_string("decision_model").value_or(""),
                   "expected_utility")) return false;
    if (!expect_eq("decision epistemic state",
                   result.response->get_string("epistemic_state").value_or(""),
                   "SUPPORTED")) return false;
    if (!expect_eq("decision probability state",
                   result.response->get_string("probability_state").value_or(""),
                   "SPECIFIED")) return false;
    if (!expect_true("decision does not collapse facts",
                     !result.response->get_bool("fact_collapsed").value_or(true))) return false;
    return true;
}

bool test_decision_unknown_probability_uses_robust_action_only() {
    const auto result = route_decision_payload(
        make_decision_payload(false, false, true), "decision://select");
    if (!expect_eq("decision robust resolver", result.micronaut_id, "DECISION-µ")) return false;
    if (!expect_eq("decision robust op", result.selected_task_op, "select")) return false;
    if (!expect_true("decision robust dispatch allowed", result.dispatch_allowed)) return false;
    if (!result.response || !result.response->is_object()) {
        std::cerr << "[FAIL] robust decision response shape\n";
        return false;
    }
    if (!expect_eq("robust decision status", result.response->get_string("status").value_or(""),
                   "decision_selected")) return false;
    if (!expect_eq("robust decision action", result.response->get_string("action").value_or(""),
                   "carry_umbrella")) return false;
    if (!expect_eq("robust decision model", result.response->get_string("decision_model").value_or(""),
                   "maximin")) return false;
    if (!expect_eq("robust epistemic state",
                   result.response->get_string("epistemic_state").value_or(""),
                   "UNKNOWN")) return false;
    if (!expect_eq("robust probability state",
                   result.response->get_string("probability_state").value_or(""),
                   "UNKNOWN")) return false;
    if (!expect_true("robust decision preserves unknown branch",
                     json_array_contains_string(result.response, "unresolved_branches", "rain"))) return false;
    if (!expect_true("robust decision does not collapse facts",
                     !result.response->get_bool("fact_collapsed").value_or(true))) return false;
    return true;
}

bool test_decision_expected_utility_rejects_missing_probability() {
    const auto result = route_decision_payload(
        make_decision_payload(false, true, false), "decision://evaluate");
    if (!expect_eq("decision missing-probability resolver", result.micronaut_id, "DECISION-µ")) return false;
    if (!expect_true("decision missing-probability dispatch allowed", result.dispatch_allowed)) return false;
    if (!result.response || !result.response->is_object()) {
        std::cerr << "[FAIL] missing-probability decision response shape\n";
        return false;
    }
    if (!expect_eq("missing-probability status", result.response->get_string("status").value_or(""),
                   "decision_unknown")) return false;
    if (!expect_true("missing-probability has no action",
                     !result.response->get_bool("action_selected").value_or(true))) return false;
    if (!expect_true("missing-probability has no decision collapse",
                     !result.response->get_bool("decision_collapse").value_or(true))) return false;
    if (!expect_eq("missing-probability probability state",
                   result.response->get_string("probability_state").value_or(""),
                   "UNKNOWN")) return false;
    if (!expect_true("missing-probability requires probability evidence",
                     json_array_contains_string(result.response, "required_evidence",
                                                "branch_probability:rain"))) return false;
    if (!expect_true("missing-probability does not collapse facts",
                     !result.response->get_bool("fact_collapsed").value_or(true))) return false;
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

// GAP-096: a registry mutation without an explicit sync_index() must not route
// from a stale expert index — route() auto-resyncs; explicit sync stays valid.
bool test_mou_router_stale_index_guard() {
    micronaut::MicronautRegistry reg;
    memory::ConfidenceStore confidence;

    reg.add(make_mou_entry("A-u", kuhul::Phase::Sek, 1.0f, 0.0f, 0.0f));
    reg.add(make_mou_entry("B-u", kuhul::Phase::Sek, 0.0f, 1.0f, 0.0f));
    confidence.set("A-u", 0.8f);
    confidence.set("B-u", 0.7f);

    micronaut::MouRouter router(reg, confidence);
    router.sync_index();
    router.sync_bias();

    auto contains = [](const std::vector<micronaut::MouAlloc>& allocs,
                       const std::string& id) {
        for (const auto& a : allocs)
            if (a.id == id) return true;
        return false;
    };

    auto baseline = router.route(kuhul::Phase::Sek, {"route", "probe"}, 4);
    if (!expect_true("stale guard baseline nonempty", !baseline.empty())) return false;
    if (!expect_true("stale guard baseline has no C", !contains(baseline, "C-u"))) return false;

    // Add without explicit sync: route() must auto-resync before dispatch.
    const uint64_t v_before = reg.mutation_version();
    reg.add(make_mou_entry("C-u", kuhul::Phase::Sek, 0.0f, 0.0f, 1.0f));
    if (!expect_true("registry version advances on add",
                     reg.mutation_version() > v_before)) return false;
    confidence.set("C-u", 0.9f);
    auto after_add = router.route(kuhul::Phase::Sek, {"route", "probe"}, 4);
    if (!expect_true("route auto-syncs new expert", contains(after_add, "C-u"))) return false;

    // Remove without explicit sync: likewise never routes the removed expert.
    if (!expect_true("remove succeeds", reg.remove("C-u", 0.5f))) return false;
    auto after_remove = router.route(kuhul::Phase::Sek, {"route", "probe"}, 4);
    if (!expect_true("route drops removed expert", !contains(after_remove, "C-u"))) return false;

    // Explicit sync still restores a consistent index.
    router.sync_index();
    auto explicit_sync = router.route(kuhul::Phase::Sek, {"route", "probe"}, 4);
    if (!expect_true("explicit sync route nonempty", !explicit_sync.empty())) return false;
    return true;
}

// Parity test: ga_phase_arc_scalar must equal cos(Δθ) for the winning track's
// primary_phase relative to the request phase. This is the C6→Cl(2,0) cross-check
// declared in the algebra-isa contracts (GA_PRODUCT, phase_policy:"none").
bool test_ga_telemetry_arc_weight_parity() {
    memory::PerceptionStore perception;
    memory::ConfidenceStore confidence;
    memory::SemanticGraph graph;
    memory::MicronautScorer scorer(perception, confidence, graph);
    micronaut::MicronautRegistry reg;

    // Register a single track at Phase::Yax (index 2).
    micronaut::MicronautEntry yax_entry;
    yax_entry.id = "YAX-u";
    yax_entry.description = "yax-phase test track";
    yax_entry.primary_phase = kuhul::Phase::Yax;
    yax_entry.category = "cognitive";
    yax_entry.capabilities = {"expand"};
    yax_entry.governs = {"expand", "grow", "yax"};
    yax_entry.static_weight = 1.0f;
    yax_entry.health = 1.0f;
    yax_entry.confidence = 0.9f;
    yax_entry.no_eval = true;
    yax_entry.active = true;
    reg.add(yax_entry);

    io::RequestRouter router(reg, scorer, "data");
    router.set_authorizer([](const io::RouteRequest&, const std::string&) { return true; });
    router.register_handler("YAX-u", [](const std::string&, const io::RouteRequest&) {
        return util::json_string("yax");
    });

    // Route from Pop (phase 0) → track at Yax (phase 2): Δθ = 2 * π/3
    io::RouteRequest req;
    req.query = "expand grow yax";
    req.phase = kuhul::Phase::Pop;
    req.lane = "ga_parity_test";

    const auto result = router.route(req);

    // GA must not affect routing.
    if (!expect_true("ga_parity: dispatch_allowed stays correct", result.dispatch_allowed)) return false;
    if (!expect_eq("ga_parity: winning track is yax", result.micronaut_id, "YAX-u")) return false;
    if (!expect_true("ga_parity: dispatched_phase is Yax",
                     result.dispatched_phase == kuhul::Phase::Yax)) return false;

    // Core parity: ga_phase_arc_scalar == cos(Δθ) == cos(2π/3) == -0.5
    const double delta = (2 - 0) * (std::numbers::pi / 3.0);
    const float expected_scalar   = static_cast<float>(std::cos(delta));   // -0.5
    const float expected_bivector = static_cast<float>(-std::sin(delta));  // -√3/2 ≈ -0.866
    constexpr float kTol = 1e-5f;
    if (!expect_true("ga_parity: arc_scalar == cos(2π/3)",
                     std::fabs(result.ga_phase_arc_scalar - expected_scalar) < kTol)) return false;
    if (!expect_true("ga_parity: arc_bivector == -sin(2π/3)",
                     std::fabs(result.ga_phase_arc_bivector - expected_bivector) < kTol)) return false;

    // orient is zero when both Q and K xy-components are zero (default zero vectors).
    if (!expect_true("ga_parity: orient_xy is finite",
                     std::isfinite(result.ga_orient_xy_query_key))) return false;
    return true;
}

} // namespace

int main() {
    return test::run("router", [](test::Suite& suite) {
        suite.check("policy_filters_then_ranks", test_policy_filters_then_ranks());
        suite.check("epistemic_contract_for_miss_and_unknown", test_epistemic_contract_for_miss_and_unknown());
        suite.check("decision_expected_utility_selects_action_without_fact_collapse", test_decision_expected_utility_selects_action_without_fact_collapse());
        suite.check("decision_unknown_probability_uses_robust_action_only", test_decision_unknown_probability_uses_robust_action_only());
        suite.check("decision_expected_utility_rejects_missing_probability", test_decision_expected_utility_rejects_missing_probability());
        suite.check("empty_eligibility_is_capability_miss", test_empty_eligibility_is_capability_miss());
        suite.check("tool_gram_channel_biases_tool_track", test_tool_gram_channel_biases_tool_track());
        suite.check("factory_vector_track_commits_monotonic_frame", test_factory_vector_track_commits_monotonic_frame());
        suite.check("factory_vector_track_bounds_drift_and_normalizes", test_factory_vector_track_bounds_drift_and_normalizes());
        suite.check("factory_vector_tracking_state_roundtrip", test_factory_vector_tracking_state_roundtrip());
        suite.check("micronaut_attraction_capture_formula", test_micronaut_attraction_capture_formula());
        suite.check("micronaut_transpiler_compiles_key_contract", test_micronaut_transpiler_compiles_key_contract());
        suite.check("mou_router_phase_filter_and_weight_norm", test_mou_router_phase_filter_and_weight_norm());
        suite.check("mou_router_phase_fallback_unfiltered", test_mou_router_phase_fallback_unfiltered());
        suite.check("mou_router_sync_bias_refresh", test_mou_router_sync_bias_refresh());
        suite.check("mou_router_dispatch_variance_contract", test_mou_router_dispatch_variance_contract());
        suite.check("mou_router_stale_index_guard", test_mou_router_stale_index_guard());
        suite.check("ga_telemetry_arc_weight_parity", test_ga_telemetry_arc_weight_parity());
        suite.check("scheme_uri_routes_to_declared_micronaut", test_scheme_uri_routes_to_declared_micronaut());
        suite.check("scheme_uri_verb_filtering", test_scheme_uri_verb_filtering());
        suite.check("scheme_uri_scopes_explicit_decision_field", test_scheme_uri_scopes_explicit_decision_field());
        suite.check("scheme_uri_id_alias_default", test_scheme_uri_id_alias_default());
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
    }, 82);
}
