#include "kql.h"
#include "../util/sha256.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace khanary::io {
namespace {

std::string trim_copy(const std::string& s) {
    std::size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    std::size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string lower_copy(std::string s) {
    for (char& ch : s)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

std::string squeeze_ws(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_ws = false;
    for (unsigned char ch : s) {
        if (std::isspace(ch)) {
            if (!in_ws) out.push_back(' ');
            in_ws = true;
        } else {
            out.push_back(static_cast<char>(ch));
            in_ws = false;
        }
    }
    return trim_copy(out);
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t pos = s.find(',', start);
        const std::string part = trim_copy(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (!part.empty()) out.push_back(part);
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return out;
}

std::string strip_quotes(std::string s) {
    s = trim_copy(s);
    if (s.size() >= 2) {
        const char a = s.front();
        const char b = s.back();
        if ((a == '\'' && b == '\'') || (a == '"' && b == '"'))
            return s.substr(1, s.size() - 2);
    }
    return s;
}

std::string normalize_identifier(std::string s, const char* label) {
    s = trim_copy(s);
    if (s.empty())
        throw std::invalid_argument(std::string(label) + " cannot be empty");
    return lower_copy(s);
}

std::vector<std::string> split_and_predicates(const std::string& s) {
    std::vector<std::string> out;
    const std::string low = lower_copy(s);
    std::size_t start = 0;
    while (start < s.size()) {
        std::size_t pos = low.find(" and ", start);
        std::string part = trim_copy(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (!part.empty()) out.push_back(part);
        if (pos == std::string::npos) break;
        start = pos + 5;
    }
    return out;
}

std::string builtin_field(const KQLRow& row, const std::string& field) {
    if (field == "table") return row.table;
    if (field == "user_mesh_id") return row.user_mesh_id;
    if (field == "key" || field == "event_key" || field == "fact_key" || field == "topology_key") return row.key;
    if (field == "vector_key") return row.vector_key;
    if (field == "commit_state") return row.commit_state;
    if (field == "fact_authority") return row.fact_authority ? "true" : "false";
    if (field == "semantic_authority") return row.semantic_authority ? "true" : "false";
    auto it = row.fields.find(field);
    return it == row.fields.end() ? std::string() : it->second;
}

bool row_matches_predicates(
    const KQLRow& row,
    const std::unordered_map<std::string, std::string>& where,
    const KQLOptions& options)
{
    for (const auto& [field, raw_expected] : where) {
        std::string expected = raw_expected;
        if (expected == ":user") expected = options.user_mesh_id;
        if (builtin_field(row, field) != expected) return false;
    }
    return true;
}

std::unordered_map<std::string, std::string> project_row(
    const KQLRow& row,
    const std::vector<std::string>& fields)
{
    std::unordered_map<std::string, std::string> out;
    if (fields.size() == 1 && fields[0] == "*") {
        out = row.fields;
        out["table"] = row.table;
        out["user_mesh_id"] = row.user_mesh_id;
        out["key"] = row.key;
        if (!row.vector_key.empty()) out["vector_key"] = row.vector_key;
        if (!row.commit_state.empty()) out["commit_state"] = row.commit_state;
        out["fact_authority"] = row.fact_authority ? "true" : "false";
        out["semantic_authority"] = row.semantic_authority ? "true" : "false";
        return out;
    }

    for (const auto& field : fields)
        out[field] = builtin_field(row, field);
    return out;
}

} // namespace

std::string make_user_mesh_id(
    const std::string& provider,
    const std::string& provider_user_key,
    const std::string& local_salt)
{
    if (provider.empty() || provider_user_key.empty() || local_salt.empty())
        throw std::invalid_argument("provider, provider_user_key, and local_salt are required");
    return "user_mesh:sha256:" + util::sha256_hex(
        "khanary:user:" + provider + ":" + provider_user_key + ":" + local_salt);
}

KQLQuery parse_select(const std::string& text) {
    std::string q = squeeze_ws(text);
    if (!q.empty() && q.back() == ';') q.pop_back();
    q = trim_copy(q);

    const std::string low = lower_copy(q);
    if (low.rfind("select ", 0) != 0)
        throw std::invalid_argument("KQL currently supports SELECT only");

    const std::size_t from_pos = low.find(" from ");
    if (from_pos == std::string::npos)
        throw std::invalid_argument("SELECT requires FROM");

    KQLQuery out;
    out.fields = split_commas(q.substr(7, from_pos - 7));
    if (out.fields.empty()) throw std::invalid_argument("SELECT requires at least one field");
    for (auto& field : out.fields) {
        if (field != "*") field = normalize_identifier(field, "SELECT field");
    }

    const std::size_t where_pos = low.find(" where ", from_pos + 6);
    const std::size_t limit_pos = low.find(" limit ", from_pos + 6);
    if (where_pos != std::string::npos &&
        limit_pos != std::string::npos &&
        limit_pos < where_pos)
        throw std::invalid_argument("LIMIT must follow WHERE");

    const std::size_t table_end = std::min(
        where_pos == std::string::npos ? q.size() : where_pos,
        limit_pos == std::string::npos ? q.size() : limit_pos);
    out.table = normalize_identifier(q.substr(from_pos + 6, table_end - (from_pos + 6)), "FROM table");
    if (out.table.empty()) throw std::invalid_argument("FROM requires a table");

    if (where_pos != std::string::npos) {
        const std::size_t where_start = where_pos + 7;
        const std::size_t where_end = limit_pos == std::string::npos ? q.size() : limit_pos;
        const std::string where_text = trim_copy(q.substr(where_start, where_end - where_start));
        for (const auto& pred : split_and_predicates(where_text)) {
            const std::size_t eq = pred.find('=');
            if (eq == std::string::npos)
                throw std::invalid_argument("WHERE predicates must use equality");
            const std::string field = normalize_identifier(pred.substr(0, eq), "WHERE field");
            const std::string value = strip_quotes(pred.substr(eq + 1));
            if (field.empty() || value.empty())
                throw std::invalid_argument("WHERE predicate has empty field or value");
            out.where[field] = value;
        }
    }

    if (limit_pos != std::string::npos) {
        const std::string value = trim_copy(q.substr(limit_pos + 7));
        if (value.empty()) throw std::invalid_argument("LIMIT requires a number");
        std::size_t consumed = 0;
        const unsigned long long n = std::stoull(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("LIMIT must be an integer");
        out.limit = static_cast<std::size_t>(n);
    }

    return out;
}

KQLResult evaluate_select(
    const KQLQuery& query,
    const std::vector<KQLRow>& rows,
    const KQLOptions& options)
{
    if (options.require_user_scope && options.user_mesh_id.empty())
        throw std::invalid_argument("KQL reads require user_mesh_id scope");

    if (auto it = query.where.find("user_mesh_id");
        it != query.where.end() && it->second != ":user" && it->second != options.user_mesh_id)
        throw std::invalid_argument("KQL user_mesh_id predicate does not match bound user");

    KQLResult result;
    result.vector_filtered = !options.vector_candidate_keys.empty();

    for (const auto& row : rows) {
        if (row.table != query.table) continue;
        if (options.require_user_scope && row.user_mesh_id != options.user_mesh_id) continue;
        if (result.vector_filtered &&
            options.vector_candidate_keys.count(row.key) == 0 &&
            options.vector_candidate_keys.count(row.vector_key) == 0)
            continue;
        if (!row_matches_predicates(row, query.where, options)) continue;

        result.rows.push_back(project_row(row, query.fields));
        if (query.limit != 0 && result.rows.size() >= query.limit) break;
    }

    return result;
}

bool is_kql_authority_table(const std::string& table) {
    return table == "kql_fact_commits" || table == "kql_topology_commits";
}

} // namespace khanary::io
