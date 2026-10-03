#include "conformance.h"
#include "io/kql.h"

#include <string>
#include <unordered_map>
#include <vector>

using khanary::io::KQLOptions;
using khanary::io::KQLRow;
using khanary::io::evaluate_select;
using khanary::io::is_kql_authority_table;
using khanary::io::make_user_mesh_id;
using khanary::io::parse_select;

namespace {

bool has_value(
    const std::unordered_map<std::string, std::string>& row,
    const std::string& key,
    const std::string& value)
{
    const auto it = row.find(key);
    return it != row.end() && it->second == value;
}

KQLRow memory_event(
    const std::string& user_mesh_id,
    const std::string& key,
    const std::string& vector_key,
    const std::string& event_kind,
    const std::string& phase)
{
    KQLRow row;
    row.table = "kql_memory_events";
    row.user_mesh_id = user_mesh_id;
    row.key = key;
    row.vector_key = vector_key;
    row.commit_state = "journaled_not_fact";
    row.fields = {
        {"event_kind", event_kind},
        {"phase", phase},
        {"track", "USER-mu"}
    };
    return row;
}

} // namespace

int main() {
    return khanary::test::run("kql", [](khanary::test::Suite& suite) {
        const auto user_a = make_user_mesh_id("google_apps_script", "gas-user-a", "local-salt");
        const auto user_a_again = make_user_mesh_id("google_apps_script", "gas-user-a", "local-salt");
        const auto user_b = make_user_mesh_id("google_apps_script", "gas-user-b", "local-salt");

        suite.check("user_mesh_id_stable", user_a == user_a_again &&
            user_a.rfind("user_mesh:sha256:", 0) == 0);
        suite.check("user_mesh_id_separates_users", user_a != user_b);
        suite.rejects("user_mesh_id_requires_components", [] {
            (void)make_user_mesh_id("", "gas-user-a", "local-salt");
        });

        std::vector<KQLRow> rows;
        rows.push_back(memory_event(
            user_a,
            "memory:event:prompt:1-1",
            "vec:prompt:1",
            "prompt",
            "Pop"));
        rows.push_back(memory_event(
            user_a,
            "memory:event:model_response:1-2",
            "vec:model_response:1",
            "model_response",
            "Yax"));
        rows.push_back(memory_event(
            user_b,
            "memory:event:prompt:2-1",
            "vec:prompt:2",
            "prompt",
            "Pop"));

        KQLRow hyper;
        hyper.table = "kql_hyperbolic_events";
        hyper.user_mesh_id = user_a;
        hyper.key = "kg:advisor:yax:why_should_i";
        hyper.vector_key = "vec:hyper:why_should_i";
        hyper.commit_state = "candidate";
        hyper.fields = {
            {"interval_class", "timelike_candidate"},
            {"rapidity_rho", "0.73"},
            {"phase", "Yax"}
        };
        rows.push_back(hyper);

        KQLRow static_model;
        static_model.table = "kql_static_api_models";
        static_model.user_mesh_id = user_a;
        static_model.key = "static-api:example:echo-v1";
        static_model.vector_key = "vec:static-api:echo-v1";
        static_model.commit_state = "candidate";
        static_model.fields = {
            {"model_id", "static-api:example:echo-v1"},
            {"provider", "example"},
            {"capability", "text.generate"},
            {"endpoint_host", "api.example.invalid"}
        };
        rows.push_back(static_model);

        KQLRow fact;
        fact.table = "kql_fact_commits";
        fact.user_mesh_id = user_a;
        fact.key = "fact:chen:xul:1";
        fact.commit_state = "Xul_committed";
        fact.fact_authority = true;
        fact.semantic_authority = true;
        fact.fields = {{"claim", "verified-test-fact"}};
        rows.push_back(fact);

        const auto prompt_query = parse_select(
            "SELECT key, event_kind, phase "
            "FROM kql_memory_events "
            "WHERE user_mesh_id = :user AND event_kind = 'prompt' "
            "LIMIT 50");
        suite.check("select_parses_fields_table_where_limit",
            prompt_query.fields.size() == 3 &&
            prompt_query.fields[0] == "key" &&
            prompt_query.table == "kql_memory_events" &&
            prompt_query.where.at("event_kind") == "prompt" &&
            prompt_query.limit == 50);

        const auto case_query = parse_select(
            "select KEY from KQL_MEMORY_EVENTS where EVENT_KIND = 'prompt' limit 1");
        suite.check("select_identifiers_are_normalized",
            case_query.fields.size() == 1 &&
            case_query.fields[0] == "key" &&
            case_query.table == "kql_memory_events" &&
            case_query.where.at("event_kind") == "prompt");

        suite.rejects("select_rejects_limit_before_where", [] {
            (void)parse_select(
                "SELECT key FROM kql_memory_events LIMIT 1 WHERE event_kind = 'prompt'");
        });

        KQLOptions scoped;
        scoped.user_mesh_id = user_a;

        const auto prompt_result = evaluate_select(prompt_query, rows, scoped);
        suite.check("select_scopes_to_bound_user",
            prompt_result.rows.size() == 1 &&
            has_value(prompt_result.rows[0], "key", "memory:event:prompt:1-1"));
        suite.check("select_projects_event_fields",
            prompt_result.rows.size() == 1 &&
            has_value(prompt_result.rows[0], "event_kind", "prompt") &&
            has_value(prompt_result.rows[0], "phase", "Pop"));

        suite.rejects("select_requires_user_scope", [&] {
            KQLOptions missing_scope;
            (void)evaluate_select(prompt_query, rows, missing_scope);
        });

        const auto cross_user = parse_select(
            "SELECT key FROM kql_memory_events WHERE user_mesh_id = '" + user_b + "'");
        suite.rejects("select_rejects_cross_user_predicate", [&] {
            (void)evaluate_select(cross_user, rows, scoped);
        });

        const auto all_memory = parse_select(
            "SELECT key FROM kql_memory_events WHERE user_mesh_id = :user");

        KQLOptions by_event_key = scoped;
        by_event_key.vector_candidate_keys.insert("memory:event:model_response:1-2");
        const auto event_key_result = evaluate_select(all_memory, rows, by_event_key);
        suite.check("vector_filter_accepts_event_key",
            event_key_result.vector_filtered &&
            event_key_result.rows.size() == 1 &&
            has_value(event_key_result.rows[0], "key", "memory:event:model_response:1-2"));

        KQLOptions by_vector_key = scoped;
        by_vector_key.vector_candidate_keys.insert("vec:prompt:1");
        const auto vector_key_result = evaluate_select(all_memory, rows, by_vector_key);
        suite.check("vector_filter_accepts_vector_key",
            vector_key_result.vector_filtered &&
            vector_key_result.rows.size() == 1 &&
            has_value(vector_key_result.rows[0], "key", "memory:event:prompt:1-1"));

        KQLOptions miss_vector = scoped;
        miss_vector.vector_candidate_keys.insert("vec:not-present");
        const auto miss_result = evaluate_select(all_memory, rows, miss_vector);
        suite.check("vector_filter_excludes_non_candidates",
            miss_result.vector_filtered && miss_result.rows.empty());

        const auto hyper_query = parse_select(
            "SELECT event_key, interval_class, rapidity_rho "
            "FROM kql_hyperbolic_events "
            "WHERE user_mesh_id = :user AND interval_class = 'timelike_candidate' "
            "LIMIT 32");
        const auto hyper_result = evaluate_select(hyper_query, rows, scoped);
        suite.check("hyperbolic_event_rows_are_queryable",
            hyper_result.rows.size() == 1 &&
            has_value(hyper_result.rows[0], "event_key", "kg:advisor:yax:why_should_i") &&
            has_value(hyper_result.rows[0], "rapidity_rho", "0.73"));

        const auto static_model_query = parse_select(
            "SELECT model_id, provider, capability, endpoint_host "
            "FROM kql_static_api_models "
            "WHERE user_mesh_id = :user AND capability = 'text.generate' "
            "LIMIT 25");
        const auto static_model_result = evaluate_select(static_model_query, rows, scoped);
        suite.check("static_api_model_contributions_are_queryable_candidates",
            static_model_result.rows.size() == 1 &&
            has_value(static_model_result.rows[0], "model_id", "static-api:example:echo-v1") &&
            has_value(static_model_result.rows[0], "provider", "example") &&
            has_value(static_model_result.rows[0], "endpoint_host", "api.example.invalid"));

        const auto limited = parse_select(
            "SELECT key FROM kql_memory_events WHERE user_mesh_id = :user LIMIT 1");
        suite.check("limit_applies_after_scope_filter",
            evaluate_select(limited, rows, scoped).rows.size() == 1);

        suite.check("authority_tables_are_marked",
            is_kql_authority_table("kql_fact_commits") &&
            is_kql_authority_table("kql_topology_commits"));
        suite.check("memory_tables_are_not_authority",
            !is_kql_authority_table("kql_memory_events"));

        const auto star = parse_select(
            "SELECT * FROM kql_memory_events WHERE user_mesh_id = :user AND event_kind = 'prompt'");
        const auto star_result = evaluate_select(star, rows, scoped);
        suite.check("star_projection_preserves_non_authority_boundary",
            star_result.rows.size() == 1 &&
            has_value(star_result.rows[0], "semantic_authority", "false") &&
            has_value(star_result.rows[0], "fact_authority", "false") &&
            has_value(star_result.rows[0], "vector_key", "vec:prompt:1"));

        const auto fact_query = parse_select(
            "SELECT fact_key, fact_authority, semantic_authority "
            "FROM kql_fact_commits WHERE user_mesh_id = :user");
        const auto fact_result = evaluate_select(fact_query, rows, scoped);
        suite.check("xul_committed_fact_rows_project_authority",
            fact_result.rows.size() == 1 &&
            has_value(fact_result.rows[0], "fact_key", "fact:chen:xul:1") &&
            has_value(fact_result.rows[0], "fact_authority", "true") &&
            has_value(fact_result.rows[0], "semantic_authority", "true"));
    }, 20);
}
