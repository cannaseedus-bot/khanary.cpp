#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace khanary::io {

// KQL is a user-scoped table/query layer over KHANARY's existing vector engine.
// It does not own semantic authority; it filters event/vector evidence before
// Ch'en/Xul verification can promote rows into fact/topology authority tables.
struct KQLRow {
    std::string table;
    std::string user_mesh_id;
    std::string key;
    std::string vector_key;
    std::string commit_state;
    bool fact_authority = false;
    bool semantic_authority = false;
    std::unordered_map<std::string, std::string> fields;
};

struct KQLQuery {
    std::vector<std::string> fields;
    std::string table;
    std::unordered_map<std::string, std::string> where;
    std::size_t limit = 0; // 0 means unbounded after user/table/vector filters.
};

struct KQLOptions {
    std::string user_mesh_id;
    std::unordered_set<std::string> vector_candidate_keys;
    bool require_user_scope = true;
};

struct KQLResult {
    std::vector<std::unordered_map<std::string, std::string>> rows;
    bool vector_filtered = false;
};

std::string make_user_mesh_id(
    const std::string& provider,
    const std::string& provider_user_key,
    const std::string& local_salt);

KQLQuery parse_select(const std::string& text);
KQLResult evaluate_select(
    const KQLQuery& query,
    const std::vector<KQLRow>& rows,
    const KQLOptions& options);

bool is_kql_authority_table(const std::string& table);

} // namespace khanary::io
