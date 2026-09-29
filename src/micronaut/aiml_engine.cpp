#include "aiml_engine.h"
#include "eliza_operators.h"
#include "relation_isa.h"
#include "../util/json.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <functional>
#include <regex>
#include <filesystem>
#include <vector>
#include <unordered_set>
#include <queue>

namespace khanary::micronaut {

namespace fs = std::filesystem;
using namespace khanary::util;

AIMLEngine::AIMLEngine() {
    vars_["topic"] = "*";
}

static std::string xml_text_between(const std::string& s, const std::string& tag) {
    std::string open  = "<" + tag + ">";
    std::string close = "</" + tag + ">";
    size_t a = s.find(open);
    if (a == std::string::npos) return "";
    a += open.size();
    size_t b = s.find(close, a);
    if (b == std::string::npos) return s.substr(a);
    return s.substr(a, b - a);
}

static std::string lower_ascii(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::unordered_map<std::string, std::string> parse_xml_attrs(const std::string& raw) {
    std::unordered_map<std::string, std::string> out;
    static const std::regex attr_re(
        "([A-Za-z_:\\-][A-Za-z0-9_:\\-]*)\\s*=\\s*\"([^\"]*)\"",
        std::regex::icase);
    for (auto it = std::sregex_iterator(raw.begin(), raw.end(), attr_re);
         it != std::sregex_iterator{}; ++it) {
        const auto key = lower_ascii((*it)[1].str());
        out[key] = (*it)[2].str();
    }
    return out;
}

static std::string attr_value(
    const std::unordered_map<std::string, std::string>& attrs,
    const std::string& key,
    const std::string& fallback = {})
{
    auto it = attrs.find(lower_ascii(key));
    return it == attrs.end() ? fallback : it->second;
}

static std::string join_tokens(
    const std::vector<std::string>& toks,
    size_t start,
    size_t end_exclusive)
{
    if (start >= end_exclusive || end_exclusive > toks.size()) return {};
    std::string out;
    for (size_t i = start; i < end_exclusive; ++i) {
        if (!out.empty()) out.push_back(' ');
        out += toks[i];
    }
    return out;
}

static std::string capture_at(const std::vector<std::string>& caps, size_t one_based_index) {
    if (one_based_index == 0) return {};
    const size_t idx = one_based_index - 1;
    return (idx < caps.size()) ? caps[idx] : std::string{};
}

static void replace_capture_tags(std::string& tpl, const char* tag_name, const std::vector<std::string>& caps) {
    if (tpl.empty()) return;

    const std::string tag(tag_name);
    const std::regex indexed_re(
        "<" + tag + "\\s+index\\s*=\\s*[\"']([0-9]+)[\"']\\s*/>",
        std::regex::icase);
    std::string out;
    auto begin = std::sregex_iterator(tpl.begin(), tpl.end(), indexed_re);
    auto end = std::sregex_iterator{};
    size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        const auto& m = *it;
        out += tpl.substr(last, m.position() - last);
        const size_t idx = static_cast<size_t>(std::strtoul(m[1].str().c_str(), nullptr, 10));
        out += capture_at(caps, idx);
        last = m.position() + m.length();
    }
    out += tpl.substr(last);
    tpl = std::move(out);

    const std::string plain = "<" + tag + "/>";
    size_t pos = 0;
    while ((pos = tpl.find(plain, pos)) != std::string::npos) {
        tpl.replace(pos, plain.size(), capture_at(caps, 1));
        pos += capture_at(caps, 1).size();
    }
}

std::string AIMLEngine::normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_ws = false;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '*' || c == '_') {
            out += static_cast<char>(std::toupper(c));
            in_ws = false;
            continue;
        }
        if (std::isspace(c) || std::ispunct(c)) {
            if (!in_ws) { out += ' '; in_ws = true; }
        }
    }
    size_t start = 0;
    while (start < out.size() && out[start] == ' ') ++start;
    size_t end = out.size();
    while (end > start && out[end - 1] == ' ') --end;
    return out.substr(start, end - start);
}

bool AIMLEngine::pattern_match(
    const std::string& pattern,
    const std::string& input,
    std::vector<std::string>& captures)
{
    std::vector<std::string> ptoks, itoks;
    {
        std::istringstream ps(pattern), is(input);
        std::string t;
        while (ps >> t) ptoks.push_back(t);
        while (is >> t) itoks.push_back(t);
    }

    captures.clear();

    if (ptoks.size() == 1 && ptoks[0] == "*") { captures.push_back(input); return true; }
    if (ptoks.size() == 1 && ptoks[0] == "_") {
        if (itoks.size() != 1) return false;
        captures.push_back(itoks[0]);
        return true;
    }

    // Recursive backtracking match; captures are collected in left-to-right wildcard order.
    std::function<bool(size_t, size_t, std::vector<std::string>&)> rec =
        [&](size_t pi, size_t ii, std::vector<std::string>& caps) -> bool {
        if (pi == ptoks.size() && ii == itoks.size()) return true;
        if (pi == ptoks.size()) return false;
        if (ptoks[pi] == "*") {
            // Greedy wildcard: consume 0..N tokens.
            for (size_t j = itoks.size(); ; --j) {
                if (j < ii) break;
                std::vector<std::string> next_caps = caps;
                next_caps.push_back(join_tokens(itoks, ii, j));
                if (rec(pi + 1, j, next_caps)) {
                    caps = std::move(next_caps);
                    return true;
                }
                if (j == 0) break;
            }
            return false;
        }
        if (ptoks[pi] == "_") {
            if (ii >= itoks.size()) return false;
            std::vector<std::string> next_caps = caps;
            next_caps.push_back(itoks[ii]);
            if (rec(pi + 1, ii + 1, next_caps)) {
                caps = std::move(next_caps);
                return true;
            }
            return false;
        }
        if (ii >= itoks.size()) return false;
        if (ptoks[pi] != itoks[ii]) return false;
        return rec(pi + 1, ii + 1, caps);
    };
    std::vector<std::string> cap_accum;
    if (!rec(0, 0, cap_accum)) return false;
    captures = std::move(cap_accum);
    return true;
}

void AIMLEngine::parse_aiml(const std::string& content) {
    // Very light XML parser: find <category>...</category> blocks
    size_t pos = 0;
    while (true) {
        size_t cat_start = content.find("<category>", pos);
        if (cat_start == std::string::npos) break;
        size_t cat_end = content.find("</category>", cat_start);
        if (cat_end == std::string::npos) break;
        std::string cat_body = content.substr(cat_start + 10, cat_end - cat_start - 10);
        pos = cat_end + 11;

        AIMLCategory cat;
        cat.pattern = normalize(xml_text_between(cat_body, "pattern"));
        cat.that    = normalize(xml_text_between(cat_body, "that"));
        if (cat.that.empty()) cat.that = "*";
        cat.topic   = normalize(xml_text_between(cat_body, "topic"));
        cat.tmpl    = xml_text_between(cat_body, "template");
        if (!cat.pattern.empty() && !cat.tmpl.empty())
            categories_.push_back(std::move(cat));
    }
}

bool AIMLEngine::load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    parse_aiml(ss.str());
    return true;
}

int AIMLEngine::load_dir(const std::string& dir) {
    int count = 0;
    if (!fs::is_directory(dir)) return 0;
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".aiml")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& file : files)
        if (load_file(file.string())) ++count;
    return count;
}

std::string AIMLEngine::match(const std::string& input, const std::string& that) const {
    std::string norm_in   = normalize(input);
    std::string norm_that = that.empty() ? "*" : normalize(that);
    const std::string active_topic = get("topic");
    auto attempt_match = [&](const std::string& normalized_candidate) -> std::string {
        if (normalized_candidate.empty()) return {};
        for (const auto& cat : categories_) {
            if (!cat.topic.empty() && cat.topic != "*" && cat.topic != active_topic) continue;

            std::vector<std::string> caps;
            if (!pattern_match(cat.pattern, normalized_candidate, caps)) continue;

            std::vector<std::string> that_caps;
            if (cat.that != "*") {
                if (!pattern_match(cat.that, norm_that, that_caps)) continue;
            }

            std::string tpl = cat.tmpl;
            replace_capture_tags(tpl, "star", caps);
            replace_capture_tags(tpl, "thatstar", that_caps);
            return expand(tpl, input);
        }
        return {};
    };

    if (auto direct = attempt_match(norm_in); !direct.empty())
        return direct;

    std::vector<std::string> tokens;
    {
        std::istringstream iss(norm_in);
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);
    }
    if (!tokens.empty()) {
        static const ElizaOperatorBank kOpBank;
        const auto op_matches = kOpBank.detect_all(tokens, 0.25f, 3);
        for (const auto& opm : op_matches) {
            if (!opm.op) continue;

            const std::string op_name = normalize(opm.op->name);
            if (auto op_hit = attempt_match(op_name); !op_hit.empty())
                return op_hit;

            if (!opm.canonical_relation.empty()) {
                const std::string rel_name = normalize(opm.canonical_relation);
                if (auto rel_hit = attempt_match(rel_name); !rel_hit.empty())
                    return rel_hit;
            }

            if (!opm.semantic_atom.empty()) {
                const std::string atom_name = normalize(opm.semantic_atom);
                if (auto atom_hit = attempt_match(atom_name); !atom_hit.empty())
                    return atom_hit;
            }

            const std::string directive = normalize(kOpBank.emit(*opm.op, opm.context));
            if (auto directive_hit = attempt_match(directive); !directive_hit.empty())
                return directive_hit;
        }
    }

    SemanticFallbackFn fallback;
    {
        std::lock_guard<std::mutex> lk(fallback_mtx_);
        fallback = semantic_fallback_;
    }
    if (fallback) return fallback(norm_in);
    return "";
}

std::string AIMLEngine::expand(const std::string& tmpl, const std::string& input) const {
    static thread_local int s_depth = 0;
    if (s_depth >= 8) return tmpl;
    ++s_depth;
    struct DepthGuard {
        int& depth;
        ~DepthGuard() { --depth; }
    } guard{s_depth};
    return expand_impl(tmpl, input);
}

std::string AIMLEngine::expand_impl(const std::string& tmpl, const std::string& /*input*/) const {
    std::string result = tmpl;
    // <srai>...</srai> → recursive match
    size_t pos = 0;
    while (true) {
        size_t s = result.find("<srai>", pos);
        if (s == std::string::npos) break;
        size_t e = result.find("</srai>", s);
        if (e == std::string::npos) break;
        std::string inner = result.substr(s + 6, e - s - 6);
        std::string sub = match(inner);
        const std::string resolved = sub.empty() ? inner : sub;
        result.replace(s, e - s + 7, resolved);
        pos = s + resolved.size();
    }

    auto parse_size_t_attr = [&](const std::string& text, size_t fallback) {
        if (text.empty()) return fallback;
        try {
            return static_cast<size_t>(std::stoull(text));
        } catch (...) {
            return fallback;
        }
    };

    auto graph_neighbors = [&](const std::string& source,
                               const std::string& relation_filter,
                               const std::string& direction_filter,
                               const std::string& family_filter) {
        std::vector<std::string> out;
        if (source.empty()) return out;

        const std::string rel = relation_filter.empty()
            ? std::string{}
            : infer_relation_from_lexical(relation_filter);
        const std::string dir = lower_ascii(direction_filter.empty() ? "out" : direction_filter);
        const auto fam = relation_family_from_name(family_filter);
        const bool fam_filter_active = !family_filter.empty() && fam != RelationFamily::Unknown;

        std::lock_guard<std::mutex> lk(graph_mtx_);
        if (graph_nodes_.find(source) == graph_nodes_.end()) return out;

        auto accept_rel = [&](const AIMLGraphEdge& e) {
            if (!rel.empty() && !relation_matches(e.verb, rel)) return false;
            if (fam_filter_active && relation_family_of(e.verb) != fam) return false;
            return true;
        };

        for (const auto& e : graph_edges_) {
            if (!accept_rel(e)) continue;

            const bool dir_out = (dir == "out" || dir == "forward" || dir == "both");
            const bool dir_in  = (dir == "in" || dir == "reverse" || dir == "both");
            if (!dir_out && !dir_in) continue;

            if (dir_out && e.source == source) out.push_back(e.target);
            if (dir_in && e.target == source) out.push_back(e.source);

            if (!e.directed) {
                if (dir_out && e.target == source) out.push_back(e.source);
                if (dir_in && e.source == source) out.push_back(e.target);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };

    auto graph_traverse = [&](const std::string& start,
                              const std::string& relation_filter,
                              const std::string& direction_filter,
                              const std::string& family_filter,
                              size_t depth,
                              size_t limit) {
        std::vector<std::string> out;
        if (start.empty() || depth == 0 || limit == 0) return out;

        std::queue<std::pair<std::string, size_t>> q;
        std::unordered_set<std::string> visited;
        q.push({start, 0});
        visited.insert(start);

        while (!q.empty()) {
            auto [cur, level] = q.front();
            q.pop();
            if (level >= depth) continue;

            auto next = graph_neighbors(cur, relation_filter, direction_filter, family_filter);
            for (const auto& node_id : next) {
                if (!visited.insert(node_id).second) continue;
                out.push_back(node_id);
                if (out.size() >= limit) return out;
                q.push({node_id, level + 1});
            }
        }
        return out;
    };

    auto graph_attr_read = [&](const std::string& node_id, const std::string& field) -> std::string {
        if (node_id.empty() || field.empty()) return {};
        std::lock_guard<std::mutex> lk(graph_mtx_);
        auto it = graph_nodes_.find(node_id);
        if (it == graph_nodes_.end()) return {};
        auto fit = it->second.find(field);
        return fit == it->second.end() ? std::string{} : fit->second;
    };

    auto graph_attr_write = [&](const std::string& node_id, const std::string& field, const std::string& value) {
        if (node_id.empty() || field.empty()) return;
        const_cast<AIMLEngine*>(this)->add_graph_node_attr(node_id, field, value);
    };

    auto graph_root = [&]() -> std::string {
        std::lock_guard<std::mutex> lk(graph_mtx_);
        return graph_root_;
    };

    auto graph_level = [&](const std::string& node_id) -> int {
        if (node_id.empty()) return -1;
        std::string root;
        std::vector<AIMLGraphEdge> edges;
        {
            std::lock_guard<std::mutex> lk(graph_mtx_);
            if (graph_nodes_.find(node_id) == graph_nodes_.end()) return -1;
            root = graph_root_;
            edges = graph_edges_;
        }
        if (root.empty()) return -1;
        if (root == node_id) return 0;

        std::queue<std::pair<std::string, int>> q;
        std::unordered_set<std::string> visited;
        q.push({root, 0});
        visited.insert(root);

        while (!q.empty()) {
            auto [cur, level] = q.front();
            q.pop();
            for (const auto& e : edges) {
                if (e.source == cur) {
                    if (e.target == node_id) return level + 1;
                    if (visited.insert(e.target).second) q.push({e.target, level + 1});
                }
                if (!e.directed && e.target == cur) {
                    if (e.source == node_id) return level + 1;
                    if (visited.insert(e.source).second) q.push({e.source, level + 1});
                }
            }
        }
        return -1;
    };

    // <node id="..."/> — bind/set current graph node and echo node id.
    {
        std::regex node_re("<node\\s+([^>/]*)/>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), node_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            const auto attrs = parse_xml_attrs(m[1].str());
            const std::string node_id = attr_value(attrs, "id");
            bool exists = false;
            if (!node_id.empty()) {
                std::lock_guard<std::mutex> lk(graph_mtx_);
                exists = graph_nodes_.find(node_id) != graph_nodes_.end();
            }
            if (exists) {
                const_cast<AIMLEngine*>(this)->set("graph.current", node_id);
                out += node_id;
            }
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <getroot/>
    {
        const std::string token = "<getroot/>";
        size_t p = 0;
        while ((p = result.find(token, p)) != std::string::npos) {
            const std::string root = graph_root();
            result.replace(p, token.size(), root);
            p += root.size();
        }
    }

    // <edges .../> and <getchildren .../>
    {
        std::regex edges_re("<(edges|getchildren)\\s+([^>/]*)/>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), edges_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            auto attrs = parse_xml_attrs(m[2].str());
            std::string source = attr_value(attrs, "source");
            if (source.empty()) source = attr_value(attrs, "id");
            if (source.empty()) source = get("graph.current");
            std::string relation = attr_value(attrs, "relation");
            if (relation.empty()) relation = attr_value(attrs, "relation_type");
            if (relation.empty()) relation = attr_value(attrs, "verb");
            const std::string direction = attr_value(attrs, "direction", "out");
            const std::string relation_family = attr_value(attrs, "relation_family");
            const std::string join = attr_value(attrs, "join", " ");
            size_t limit = 0;
            const auto limit_s = attr_value(attrs, "limit");
            if (!limit_s.empty()) {
                try { limit = static_cast<size_t>(std::stoul(limit_s)); } catch (...) { limit = 0; }
            }

            auto children = graph_neighbors(source, relation, direction, relation_family);
            if (limit > 0 && children.size() > limit) children.resize(limit);

            std::string joined;
            for (size_t i = 0; i < children.size(); ++i) {
                if (i) joined += join;
                joined += children[i];
            }
            out += joined;
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <getedge .../>
    {
        std::regex edge_re("<getedge\\s+([^>/]*)/>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), edge_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            auto attrs = parse_xml_attrs(m[1].str());
            std::string source = attr_value(attrs, "source");
            if (source.empty()) source = get("graph.current");
            std::string relation = attr_value(attrs, "relation");
            if (relation.empty()) relation = attr_value(attrs, "relation_type");
            if (relation.empty()) relation = attr_value(attrs, "verb");
            const std::string direction = attr_value(attrs, "direction", "out");
            const std::string relation_family = attr_value(attrs, "relation_family");
            size_t index = 1;
            const auto idx_s = attr_value(attrs, "index");
            if (!idx_s.empty()) {
                try { index = static_cast<size_t>(std::stoul(idx_s)); } catch (...) { index = 1; }
            }
            std::string picked;
            auto children = graph_neighbors(source, relation, direction, relation_family);
            if (index > 0 && index <= children.size()) picked = children[index - 1];
            out += picked;
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <traverse .../>
    {
        std::regex traverse_re("<traverse\\s+([^>/]*)/>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), traverse_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);

            auto attrs = parse_xml_attrs(m[1].str());
            std::string start = attr_value(attrs, "start");
            if (start.empty()) start = attr_value(attrs, "source");
            if (start.empty()) start = attr_value(attrs, "id");
            if (start.empty()) start = get("graph.current");

            std::string relation = attr_value(attrs, "relation");
            if (relation.empty()) relation = attr_value(attrs, "relation_type");
            if (relation.empty()) relation = attr_value(attrs, "verb");

            const std::string direction = attr_value(attrs, "direction", "out");
            const std::string relation_family = attr_value(attrs, "relation_family");
            const std::string join = attr_value(attrs, "join", " ");
            const std::string target = attr_value(attrs, "target");

            size_t depth = parse_size_t_attr(attr_value(attrs, "depth"), 1);
            if (depth == 0) depth = 1;
            size_t limit = parse_size_t_attr(attr_value(attrs, "limit"), 16);
            if (limit == 0) limit = 16;

            std::string emitted;
            auto nodes = graph_traverse(start, relation, direction, relation_family, depth, limit);
            if (!target.empty()) {
                if (std::find(nodes.begin(), nodes.end(), target) != nodes.end()) {
                    emitted = target;
                } else if (!relation.empty() && !relation_matches(relation, "@related_to")) {
                    auto weak_nodes = graph_traverse(start, "@related_to", direction, relation_family, depth, limit);
                    if (std::find(weak_nodes.begin(), weak_nodes.end(), target) != weak_nodes.end()) {
                        emitted = "MISS_relation(requested=" + infer_relation_from_lexical(relation) +
                                  ",available=@related_to,start=" + start +
                                  ",target=" + target + ")";
                    }
                }
            } else {
                for (size_t i = 0; i < nodes.size(); ++i) {
                    if (i) emitted += join;
                    emitted += nodes[i];
                }
                if (emitted.empty() && !relation.empty() && !relation_matches(relation, "@related_to")) {
                    auto weak_nodes = graph_traverse(start, "@related_to", direction, relation_family, depth, limit);
                    if (!weak_nodes.empty()) {
                        emitted = "MISS_relation(requested=" + infer_relation_from_lexical(relation) +
                                  ",available=@related_to,start=" + start + ")";
                    }
                }
            }

            out += emitted;
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <getlevel .../>
    {
        std::regex level_re("<getlevel\\s*([^>/]*)/>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), level_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            auto attrs = parse_xml_attrs(m[1].str());
            std::string node_id = attr_value(attrs, "id");
            if (node_id.empty()) node_id = get("graph.current");
            const int level = graph_level(node_id);
            out += (level >= 0) ? std::to_string(level) : std::string{};
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <getindex .../>
    {
        std::regex index_re("<getindex\\s+([^>/]*)/>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), index_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            auto attrs = parse_xml_attrs(m[1].str());
            std::string source = attr_value(attrs, "within");
            if (source.empty()) source = get("graph.current");
            std::string target = attr_value(attrs, "id");
            if (target.empty()) target = attr_value(attrs, "target");
            std::string relation = attr_value(attrs, "relation");
            if (relation.empty()) relation = attr_value(attrs, "relation_type");
            if (relation.empty()) relation = attr_value(attrs, "verb");
            const std::string direction = attr_value(attrs, "direction", "out");
            const std::string relation_family = attr_value(attrs, "relation_family");
            size_t idx = 0;
            auto children = graph_neighbors(source, relation, direction, relation_family);
            for (size_t i = 0; i < children.size(); ++i) {
                if (children[i] == target) { idx = i + 1; break; }
            }
            out += std::to_string(idx);
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <node-attr .../>
    {
        std::regex attr_re("<node-attr\\s+([^>/]*)/>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), attr_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            auto attrs = parse_xml_attrs(m[1].str());
            std::string node_id = attr_value(attrs, "id");
            if (node_id.empty()) node_id = get("graph.current");
            const std::string field = attr_value(attrs, "field");
            out += graph_attr_read(node_id, field);
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <node-write ...>...</node-write>
    {
        std::regex write_re("<node-write\\s+([^>]*)>([\\s\\S]*?)</node-write>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), write_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            auto attrs = parse_xml_attrs(m[1].str());
            std::string node_id = attr_value(attrs, "id");
            if (node_id.empty()) node_id = get("graph.current");
            const std::string field = attr_value(attrs, "field");
            const std::string raw_value = m[2].str();
            const std::string value = expand(raw_value);
            graph_attr_write(node_id, field, value);
            out += value;
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }

    // <tool name="...">...</tool>
    {
        std::regex tool_re("<tool\\s+name\\s*=\\s*\"([^\"]+)\"\\s*>([\\s\\S]*?)</tool>", std::regex::icase);
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), tool_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);

            const std::string tool_name = lower_ascii(m[1].str());
            const std::string raw_arg = m[2].str();
            const std::string arg = expand(raw_arg);
            ToolFn tool;
            {
                std::lock_guard<std::mutex> lk(tools_mtx_);
                auto tf = tools_.find(tool_name);
                if (tf != tools_.end()) tool = tf->second;
            }
            if (!tool) {
                out += "[tool_error:" + tool_name + "]";
            } else {
                out += tool(arg);
            }
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }
    // <get name="..."/>
    {
        std::regex get_re("<get name=\"([^\"]+)\"\\s*/>");
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), get_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            out += get(m[1].str());
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }
    // <set name="...">...</set>
    {
        std::regex set_re("<set name=\"([^\"]+)\">([^<]*)</set>");
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), set_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            const_cast<AIMLEngine*>(this)->set(m[1].str(), m[2].str());
            out += m[2].str();
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }
    return result;
}

void AIMLEngine::set(const std::string& name, const std::string& value) {
    if (name.empty()) return;
    std::lock_guard<std::mutex> lk(vars_mtx_);
    vars_[name] = value;
}

std::string AIMLEngine::get(const std::string& name) const {
    if (name.empty()) return "";
    std::lock_guard<std::mutex> lk(vars_mtx_);
    auto it = vars_.find(name);
    return it != vars_.end() ? it->second : "";
}

void AIMLEngine::register_tool(const std::string& name, ToolFn fn) {
    if (name.empty() || !fn) return;
    std::lock_guard<std::mutex> lk(tools_mtx_);
    tools_[lower_ascii(name)] = std::move(fn);
}

bool AIMLEngine::has_tool(const std::string& name) const {
    if (name.empty()) return false;
    std::lock_guard<std::mutex> lk(tools_mtx_);
    return tools_.find(lower_ascii(name)) != tools_.end();
}

void AIMLEngine::set_semantic_fallback(SemanticFallbackFn fn) {
    std::lock_guard<std::mutex> lk(fallback_mtx_);
    semantic_fallback_ = std::move(fn);
}

bool AIMLEngine::persist(kuhul::CommitAuthority authority, const std::string& path) const {
    (void)std::move(authority);
    return save_state(path);
}

bool AIMLEngine::save_state(const std::string& path) const {
    if (path.empty()) return false;

    std::unordered_map<std::string, std::string> snapshot;
    {
        std::lock_guard<std::mutex> lk(vars_mtx_);
        snapshot = vars_;
    }

    JsonObject root;
    for (const auto& [k, v] : snapshot) {
        if (k.empty()) continue;
        root[k] = json_string(v);
    }
    std::ofstream f(path);
    if (!f) return false;
    f << stringify(json_object(std::move(root)), true);
    return true;
}

bool AIMLEngine::load_state(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    auto root = parse(ss.str());
    if (!root || !root->is_object()) return false;

    std::unordered_map<std::string, std::string> loaded;
    for (const auto& [k, v] : root->as_object()) {
        if (k.empty() || !v || !v->is_string()) continue;
        loaded[k] = v->as_string();
    }
    if (!loaded.count("topic")) loaded["topic"] = "*";

    std::lock_guard<std::mutex> lk(vars_mtx_);
    vars_.swap(loaded);
    return true;
}

void AIMLEngine::clear_graph() {
    std::lock_guard<std::mutex> lk(graph_mtx_);
    graph_nodes_.clear();
    graph_edges_.clear();
    graph_root_.clear();
}

void AIMLEngine::set_graph_root(const std::string& node_id) {
    if (node_id.empty()) return;
    std::lock_guard<std::mutex> lk(graph_mtx_);
    if (graph_nodes_.find(node_id) == graph_nodes_.end()) return;
    graph_root_ = node_id;
}

void AIMLEngine::add_graph_node(const std::string& node_id) {
    if (node_id.empty()) return;
    std::lock_guard<std::mutex> lk(graph_mtx_);
    graph_nodes_.try_emplace(node_id);
    if (graph_root_.empty()) graph_root_ = node_id;
}

void AIMLEngine::add_graph_node_attr(
    const std::string& node_id,
    const std::string& field,
    const std::string& value)
{
    if (node_id.empty() || field.empty()) return;
    std::lock_guard<std::mutex> lk(graph_mtx_);
    graph_nodes_[node_id][field] = value;
    if (graph_root_.empty()) graph_root_ = node_id;
}

void AIMLEngine::add_graph_edge(
    const std::string& source,
    const std::string& target,
    const std::string& verb,
    bool directed)
{
    if (source.empty() || target.empty()) return;
    std::lock_guard<std::mutex> lk(graph_mtx_);
    graph_nodes_.try_emplace(source);
    graph_nodes_.try_emplace(target);
    if (graph_root_.empty()) graph_root_ = source;

    AIMLGraphEdge e;
    e.source = source;
    e.target = target;
    const auto canon = canonical_relation(verb);
    if (is_supported_relation(canon)) e.verb = canon;
    else {
        const auto inferred = infer_relation_from_lexical(verb);
        if (is_supported_relation(inferred))
            e.verb = inferred;
        else {
            const auto normalized = normalize_relation_token(verb);
            e.verb = normalized.empty() ? "@related_to" : normalized;
        }
    }
    e.directed = directed;
    graph_edges_.push_back(std::move(e));
}

bool AIMLEngine::load_graphml(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream f(path);
    if (!f) return false;
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (text.empty()) return false;

    bool default_directed = true;
    {
        std::regex graph_re("<graph\\s+([^>]*)>", std::regex::icase);
        std::smatch m;
        if (std::regex_search(text, m, graph_re)) {
            const auto attrs = parse_xml_attrs(m[1].str());
            const auto edgedefault = lower_ascii(attr_value(attrs, "edgedefault"));
            if (edgedefault == "undirected") default_directed = false;
        }
    }

    size_t loaded_nodes = 0;
    size_t loaded_edges = 0;
    {
        std::regex node_re("<node\\s+([^>]*)>", std::regex::icase);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), node_re);
             it != std::sregex_iterator{}; ++it) {
            const auto attrs = parse_xml_attrs((*it)[1].str());
            const auto id = attr_value(attrs, "id");
            if (id.empty()) continue;
            add_graph_node(id);
            ++loaded_nodes;
        }
    }
    {
        std::regex edge_re("<edge\\s+([^>]*)>", std::regex::icase);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), edge_re);
             it != std::sregex_iterator{}; ++it) {
            const auto attrs = parse_xml_attrs((*it)[1].str());
            const auto source = attr_value(attrs, "source");
            const auto target = attr_value(attrs, "target");
            if (source.empty() || target.empty()) continue;

            const auto directed_s = lower_ascii(attr_value(attrs, "directed"));
            bool directed = default_directed;
            if (!directed_s.empty()) {
                directed = !(directed_s == "false" || directed_s == "0" || directed_s == "no");
            }

            std::string verb = attr_value(attrs, "verb");
            if (verb.empty()) verb = attr_value(attrs, "relation");
            if (verb.empty()) verb = attr_value(attrs, "relation_type");
            if (verb.empty()) verb = attr_value(attrs, "label");
            if (verb.empty()) verb = attr_value(attrs, "type");
            if (verb.empty()) verb = directed ? "@delegates_to" : "@related_to";
            add_graph_edge(source, target, verb, directed);
            ++loaded_edges;
        }
    }

    return (loaded_nodes > 0) || (loaded_edges > 0);
}

size_t AIMLEngine::graph_node_count() const {
    std::lock_guard<std::mutex> lk(graph_mtx_);
    return graph_nodes_.size();
}

size_t AIMLEngine::graph_edge_count() const {
    std::lock_guard<std::mutex> lk(graph_mtx_);
    return graph_edges_.size();
}

} // namespace khanary::micronaut
