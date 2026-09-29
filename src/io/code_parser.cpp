#include "code_parser.h"
#include "runtime_attention.h"
#include "../kuhul/phase_engine.h"
#include "../micronaut/mx2lm-ngram.h"
#include "../micronaut/relation_isa.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <optional>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace khanary::io {

namespace {

std::string trim_copy(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::vector<std::string> tokenize_distributional_text(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(24);
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '_') {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (!cur.empty()) {
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

size_t line_of(const std::string& src, size_t pos) {
    size_t line = 1;
    for (size_t i = 0; i < pos && i < src.size(); ++i)
        if (src[i] == '\n') ++line;
    return line;
}

size_t col_of(const std::string& src, size_t pos) {
    size_t col = 0;
    for (size_t i = pos; i > 0; --i) {
        if (src[i - 1] == '\n') break;
        ++col;
    }
    return col;
}

util::JsonArray json_string_array(const std::vector<std::string>& values) {
    util::JsonArray out;
    out.reserve(values.size());
    for (const auto& v : values) out.push_back(util::json_string(v));
    return out;
}

util::JsonPtr json_vector(const SemanticVector3& v) {
    util::JsonObject obj;
    obj["x"] = util::json_number(v.x);
    obj["y"] = util::json_number(v.y);
    obj["z"] = util::json_number(v.z);
    obj["valid"] = util::json_bool(v.valid);
    return util::json_object(std::move(obj));
}

std::vector<std::string> build_learning_ngrams(const std::string& text, size_t max_items = 24) {
    std::vector<std::string> out;
    if (text.empty() || max_items == 0) return out;

    const auto toks = micronaut::Mx2LmNGram::tokenize(lower_ascii(text));
    if (toks.empty()) return out;

    std::unordered_set<std::string> seen;
    auto append_width = [&](size_t width) {
        if (width == 0 || toks.size() < width) return;
        for (size_t i = 0; i + width <= toks.size(); ++i) {
            std::string gram;
            for (size_t j = 0; j < width; ++j) {
                if (!gram.empty()) gram.push_back(' ');
                gram += toks[i + j];
            }
            if (!seen.insert(gram).second) continue;
            out.push_back(std::move(gram));
            if (out.size() >= max_items) return;
        }
    };

    append_width(2);
    if (out.size() < max_items) append_width(3);
    if (out.size() < max_items) append_width(1);
    if (out.size() < max_items) append_width(4);
    return out;
}

std::optional<std::string> semantic_atom_from_cue(const std::string& cue_lower) {
    static const std::unordered_map<std::string, std::string> kCueToAtom = {
        {"if", "CONDITION"},
        {"when", "CONDITION"},
        {"condition", "CONDITION"},
        {"unless", "CONDITION"},
        {"then", "CONSEQUENCE"},
        {"result", "CONSEQUENCE"},
        {"consequence", "CONSEQUENCE"},
        {"therefore", "CONSEQUENCE"},
        {"because", "CAUSAL"},
        {"cause", "CAUSAL"},
        {"reason", "CAUSAL"},
        {"verify", "VERIFICATION"},
        {"verification", "VERIFICATION"},
        {"check", "VERIFICATION"},
        {"confirm", "VERIFICATION"},
        {"validate", "VERIFICATION"},
        {"prove", "VERIFICATION"},
        {"evidence", "VERIFICATION"},
        {"support", "VERIFICATION"},
        {"supported", "VERIFICATION"},
        {"not", "NEGATION"},
        {"no", "NEGATION"},
        {"never", "NEGATION"},
        {"without", "NEGATION"},
        {"absence", "NEGATION"},
        {"prefix:un", "NEGATION"},
        {"prefix:non", "NEGATION"},
        {"prefix:dis", "NEGATION"},
        {"prefix:in", "NEGATION"},
        {"prefix:im", "NEGATION"},
        {"prefix:ir", "NEGATION"},
        {"prefix:il", "NEGATION"},
        {"suffix:able", "CAPABILITY"},
        {"suffix:ible", "CAPABILITY"},
        {"ability", "CAPABILITY"},
        {"capability", "CAPABILITY"},
        {"choose", "SELECTION"},
        {"select", "SELECTION"},
        {"selection", "SELECTION"},
        {"filter", "SELECTION"},
        {"rank", "SELECTION"},
        {"relevance", "SELECTION"},
        {"candidate", "SELECTION"},
        {"execute", "EXECUTION"},
        {"execution", "EXECUTION"},
        {"run", "EXECUTION"},
        {"dispatch", "EXECUTION"},
        {"call", "EXECUTION"},
        {"commit", "COMMIT"},
        {"archive", "COMMIT"},
        {"persist", "COMMIT"},
        {"store", "COMMIT"},
        {"finalize", "COMMIT"},
        {"route", "ROUTING"},
        {"routing", "ROUTING"},
        {"router", "ROUTING"},
        {"delegate", "ROUTING"},
        {"memory", "MEMORY"},
        {"recall", "MEMORY"},
        {"history", "MEMORY"},
        {"kuhul", "KUHUL_RUNTIME"},
        {"k'uhul", "KUHUL_RUNTIME"},
        {"fold", "KUHUL_RUNTIME"},
        {"phase", "KUHUL_RUNTIME"},
        {"runtime", "KUHUL_RUNTIME"},
        {"pop", "KUHUL_RUNTIME"},
        {"wo", "KUHUL_RUNTIME"},
        {"yax", "KUHUL_RUNTIME"},
        {"sek", "KUHUL_RUNTIME"},
        {"ch'en", "KUHUL_RUNTIME"},
        {"xul", "KUHUL_RUNTIME"}
    };
    auto it = kCueToAtom.find(cue_lower);
    if (it == kCueToAtom.end()) return std::nullopt;
    return it->second;
}

std::vector<std::string> morpheme_cues_for_token(const std::string& token_lower) {
    std::vector<std::string> cues;
    if (token_lower.size() < 4) return cues;

    static const std::vector<std::string> kNegPrefixes = {
        "un", "non", "dis", "in", "im", "ir", "il"
    };
    for (const auto& prefix : kNegPrefixes) {
        if (token_lower.size() <= prefix.size() + 2) continue;
        if (token_lower.rfind(prefix, 0) == 0) {
            cues.push_back("prefix:" + prefix);
            break;
        }
    }

    static const std::vector<std::string> kCapabilitySuffixes = {"able", "ible"};
    for (const auto& suffix : kCapabilitySuffixes) {
        if (token_lower.size() <= suffix.size() + 1) continue;
        if (token_lower.size() >= suffix.size() &&
            token_lower.compare(token_lower.size() - suffix.size(), suffix.size(), suffix) == 0) {
            cues.push_back("suffix:" + suffix);
            break;
        }
    }
    return cues;
}

std::string fold_graph_relation_token(const std::string& relation_raw) {
    return micronaut::infer_relation_from_lexical(relation_raw);
}

} // namespace

CodeGraph parse_with_grammar(const util::JsonPtr& grammar_json,
                              const std::string& source)
{
    CodeGraph graph;
    if (!grammar_json || !grammar_json->is_object()) return graph;

    if (auto s = grammar_json->get_string("language"))
        graph.language = *s;

    const util::JsonValue* tokens_val = grammar_json->get("tokens");
    const util::JsonValue* edges_val  = grammar_json->get("edges");

    if (!tokens_val || !tokens_val->is_array()) return graph;

    size_t node_counter = 0;

    // Match all token patterns against source
    for (const auto& tok_ptr : tokens_val->as_array()) {
        if (!tok_ptr || !tok_ptr->is_object()) continue;

        auto type_opt    = tok_ptr->get_string("type");
        auto pattern_opt = tok_ptr->get_string("pattern");
        if (!type_opt || !pattern_opt) continue;

        const std::string& tok_type    = *type_opt;
        const std::string& raw_pattern = *pattern_opt;

        std::vector<std::string> cap_names;
        const util::JsonValue* caps_val = tok_ptr->get("captures");
        if (caps_val && caps_val->is_array()) {
            for (const auto& cv : caps_val->as_array()) {
                if (cv && cv->is_string())
                    cap_names.push_back(cv->as_string());
            }
        }

        try {
            std::regex re(raw_pattern, std::regex_constants::ECMAScript);
            auto beg = std::sregex_iterator(source.begin(), source.end(), re);
            const std::sregex_iterator end{};

            for (auto it = beg; it != end; ++it) {
                const auto& m = *it;
                CodeNode node;
                node.id   = "n" + std::to_string(node_counter++);
                node.type = tok_type;
                node.text = m[0].str();
                node.line = line_of(source, static_cast<size_t>(m.position(0)));
                node.col  = col_of(source,  static_cast<size_t>(m.position(0)));

                for (size_t gi = 0; gi < cap_names.size() && gi + 1 < m.size(); ++gi) {
                    if (m[gi + 1].matched)
                        node.captures[cap_names[gi]] = m[gi + 1].str();
                }

                graph.nodes.push_back(std::move(node));
            }
        } catch (const std::regex_error&) {
            // skip malformed pattern
        }
    }

    // Sort by source position
    std::sort(graph.nodes.begin(), graph.nodes.end(),
        [](const CodeNode& a, const CodeNode& b) {
            return a.line != b.line ? a.line < b.line : a.col < b.col;
        });

    if (!edges_val || !edges_val->is_array()) return graph;

    // Index nodes by type
    std::unordered_map<std::string, std::vector<size_t>> by_type;
    for (size_t i = 0; i < graph.nodes.size(); ++i)
        by_type[graph.nodes[i].type].push_back(i);

    // Apply edge rules
    for (const auto& edge_ptr : edges_val->as_array()) {
        if (!edge_ptr || !edge_ptr->is_object()) continue;

        auto from_type_opt = edge_ptr->get_string("from_type");
        auto to_type_opt   = edge_ptr->get_string("to_type");
        auto kind_opt      = edge_ptr->get_string("kind");
        auto cond_opt      = edge_ptr->get_string("condition");

        if (!from_type_opt || !to_type_opt || !kind_opt) continue;

        const std::string& from_type = *from_type_opt;
        const std::string& to_type   = *to_type_opt;
        const std::string& kind      = *kind_opt;
        const std::string  condition = cond_opt ? *cond_opt : "sequence";

        const auto& from_nodes = by_type[from_type];
        const auto& to_nodes   = by_type[to_type];

        if (from_type == to_type) {
            // Sequential edges within same type
            if (condition == "sequence") {
                for (size_t i = 0; i + 1 < from_nodes.size(); ++i) {
                    graph.edges.push_back({
                        graph.nodes[from_nodes[i]].id,
                        graph.nodes[from_nodes[i + 1]].id,
                        kind, 1.0f
                    });
                }
            }
            continue;
        }

        if (condition == "sequence") {
            // Each from_node → first to_node appearing after it in source
            for (size_t fi : from_nodes) {
                const auto& fn = graph.nodes[fi];
                for (size_t ti : to_nodes) {
                    const auto& tn = graph.nodes[ti];
                    if (tn.line > fn.line || (tn.line == fn.line && tn.col > fn.col)) {
                        graph.edges.push_back({ fn.id, tn.id, kind, 1.0f });
                        break;
                    }
                }
            }
        } else if (condition == "precedes") {
            // from_node on the line immediately before to_node
            for (size_t fi : from_nodes) {
                const auto& fn = graph.nodes[fi];
                for (size_t ti : to_nodes) {
                    const auto& tn = graph.nodes[ti];
                    if (tn.line == fn.line + 1 ||
                        (tn.line == fn.line && tn.col > fn.col)) {
                        graph.edges.push_back({ fn.id, tn.id, kind, 1.0f });
                    }
                }
            }
        } else {
            // "file" or unknown: all from × all to
            for (size_t fi : from_nodes) {
                for (size_t ti : to_nodes) {
                    if (graph.nodes[fi].id != graph.nodes[ti].id) {
                        graph.edges.push_back({
                            graph.nodes[fi].id, graph.nodes[ti].id,
                            kind, 1.0f
                        });
                    }
                }
            }
        }
    }

    return graph;
}

std::vector<SemanticAtomEvidence> derive_semantic_atom_grounding(
    const std::vector<std::string>& tokens_lower)
{
    std::unordered_map<std::string, std::unordered_set<std::string>> atom_cues;
    for (const auto& raw_token : tokens_lower) {
        const auto token = lower_ascii(raw_token);
        if (token.empty()) continue;
        if (auto atom = semantic_atom_from_cue(token)) {
            atom_cues[*atom].insert(token);
        }
        for (const auto& mcue : morpheme_cues_for_token(token)) {
            if (auto atom = semantic_atom_from_cue(mcue))
                atom_cues[*atom].insert(mcue + "@" + token);
        }
    }

    std::vector<SemanticAtomEvidence> out;
    out.reserve(atom_cues.size());
    for (auto& [atom, cues_set] : atom_cues) {
        SemanticAtomEvidence row;
        row.atom = atom;
        row.cues.assign(cues_set.begin(), cues_set.end());
        std::sort(row.cues.begin(), row.cues.end());
        out.push_back(std::move(row));
    }
    std::sort(out.begin(), out.end(), [](const SemanticAtomEvidence& a, const SemanticAtomEvidence& b) {
        return a.atom < b.atom;
    });
    return out;
}

util::JsonPtr fold_graph_parse(const std::string& text) {
    const auto q = trim_copy(text);
    if (q.empty()) return util::json_null();

    auto tokens = micronaut::Mx2LmNGram::tokenize(lower_ascii(q));
    if (tokens.size() > 512) tokens.resize(512);

    const auto projection = micronaut::Mx2LmNGram::compose(tokens);
    const auto atoms = derive_semantic_atom_grounding(tokens);
    const auto ngrams = build_learning_ngrams(q, 32);
    const auto query_vector = derive_vector_from_tokens(tokens);

    std::unordered_map<std::string, std::unordered_set<std::string>> neighborhood;
    for (const auto& rel : projection.relations) {
        if (rel.from.empty() || rel.to.empty()) continue;
        neighborhood[rel.from].insert(rel.to);
        neighborhood[rel.to].insert(rel.from);
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> token_atoms;
    std::unordered_set<std::string> atom_cue_tokens;
    for (const auto& atom : atoms) {
        for (const auto& cue : atom.cues) {
            std::string token = cue;
            const auto at_pos = token.find('@');
            if (at_pos != std::string::npos && at_pos + 1 < token.size())
                token = token.substr(at_pos + 1);
            token = lower_ascii(token);
            if (token.empty()) continue;
            token_atoms[token].insert(atom.atom);
            atom_cue_tokens.insert(token);
        }
    }

    util::JsonArray atom_rows;
    util::JsonArray candidate_rows;
    for (const auto& atom : atoms) {
        const std::string phase_contract = kuhul::canonical_phase_name_for_semantic_atom(atom.atom);

        util::JsonObject row;
        row["atom"] = util::json_string(atom.atom);
        row["phase_contract"] = util::json_string(phase_contract);
        row["execution_authority"] = util::json_string("canonical_only");
        row["cues"] = util::json_array(json_string_array(atom.cues));
        atom_rows.push_back(util::json_object(std::move(row)));

        const double cue_strength = std::clamp(0.20 + 0.20 * static_cast<double>(atom.cues.size()), 0.0, 1.0);
        const double relevance = std::clamp(
            static_cast<double>(atom.cues.size()) / static_cast<double>(std::max<std::size_t>(1, tokens.size())) * 3.0,
            0.05, 1.0);
        std::vector<std::string> atom_tokens = tokenize_distributional_text(atom.atom);
        for (const auto& cue : atom.cues) {
            std::string cue_text = cue;
            const auto at_pos = cue_text.find('@');
            if (at_pos != std::string::npos && at_pos + 1 < cue_text.size())
                cue_text = cue_text.substr(at_pos + 1);
            auto cue_tokens = tokenize_distributional_text(cue_text);
            atom_tokens.insert(atom_tokens.end(), cue_tokens.begin(), cue_tokens.end());
        }
        const auto key_vector = derive_vector_from_tokens(atom_tokens);
        const double vector_relevance = compute_vector_relevance(query_vector, key_vector);
        const double wcr_score = compute_wcr_qkv_score(query_vector, key_vector, 1.0, cue_strength);

        util::JsonObject candidate;
        candidate["atom"] = util::json_string(atom.atom);
        candidate["candidate_phase"] = util::json_string(phase_contract);
        candidate["evidence_count"] = util::json_number(static_cast<double>(atom.cues.size()));
        candidate["selection_policy"] = util::json_string("Yax_WCR_candidate");
        candidate["phase_binding"] = util::json_string("XCFE_canonical_contract_only");
        candidate["query_vector"] = json_vector(query_vector);
        candidate["key_vector"] = json_vector(key_vector);
        candidate["vector_relevance"] = util::json_number(vector_relevance);
        candidate["wcr_formula"] = util::json_string("S=W*C*cos(Q,K)");
        candidate["wcr_score"] = util::json_number(wcr_score);
        candidate["legacy_relevance"] = util::json_number(relevance);
        candidate_rows.push_back(util::json_object(std::move(candidate)));
    }

    util::JsonArray relation_rows;
    for (const auto& rel : projection.relations) {
        const auto typed = fold_graph_relation_token(rel.relation);
        util::JsonObject row;
        row["from"] = util::json_string(rel.from);
        row["to"] = util::json_string(rel.to);
        row["lexical_relation"] = util::json_string(rel.relation);
        row["typed_relation"] = util::json_string(typed);
        row["relation_phase_hint"] = util::json_string(micronaut::relation_phase_hint(typed));
        row["association_weight"] = util::json_number(static_cast<double>(rel.association_weight));
        relation_rows.push_back(util::json_object(std::move(row)));
    }

    util::JsonArray book_rows;
    std::vector<std::string> tokens_sorted;
    tokens_sorted.reserve(neighborhood.size());
    for (const auto& [tok, _] : neighborhood) tokens_sorted.push_back(tok);
    std::sort(tokens_sorted.begin(), tokens_sorted.end());
    for (const auto& tok : tokens_sorted) {
        auto nh_it = neighborhood.find(tok);
        if (nh_it == neighborhood.end()) continue;
        std::vector<std::string> neighbors(nh_it->second.begin(), nh_it->second.end());
        std::sort(neighbors.begin(), neighbors.end());
        if (neighbors.size() > 8) neighbors.resize(8);

        std::vector<std::string> atom_hits;
        if (auto ait = token_atoms.find(tok); ait != token_atoms.end()) {
            atom_hits.assign(ait->second.begin(), ait->second.end());
            std::sort(atom_hits.begin(), atom_hits.end());
        }

        util::JsonObject row;
        row["token"] = util::json_string(tok);
        row["neighbors"] = util::json_array(json_string_array(neighbors));
        row["atoms"] = util::json_array(json_string_array(atom_hits));
        book_rows.push_back(util::json_object(std::move(row)));
    }

    util::JsonArray unresolved_rows;
    std::unordered_set<std::string> unresolved_set;
    for (const auto& tok : tokens) {
        if (tok.size() < 4) continue;
        if (atom_cue_tokens.count(tok)) continue;
        unresolved_set.insert(tok);
    }
    std::vector<std::string> unresolved(unresolved_set.begin(), unresolved_set.end());
    std::sort(unresolved.begin(), unresolved.end());
    for (const auto& tok : unresolved) {
        util::JsonObject row;
        row["token"] = util::json_string(tok);
        row["miss_kind"] = util::json_string("lexical_root_or_semantic_mapping");
        unresolved_rows.push_back(util::json_object(std::move(row)));
    }

    util::JsonArray provenance_rows;
    {
        util::JsonObject qrow;
        qrow["kind"] = util::json_string("query_text");
        qrow["value"] = util::json_string(q);
        provenance_rows.push_back(util::json_object(std::move(qrow)));
    }
    for (size_t i = 0; i < ngrams.size() && i < 12; ++i) {
        util::JsonObject grow;
        grow["kind"] = util::json_string("ngram");
        grow["value"] = util::json_string(ngrams[i]);
        provenance_rows.push_back(util::json_object(std::move(grow)));
    }

    util::JsonObject parse;
    parse["resolution"] = util::json_string("bounded_recursive_unfold_on_miss");
    parse["execution_authority"] = util::json_string("canonical_kuhul_phase_algebra");
    parse["recognition_only"] = util::json_bool(true);
    parse["atoms"] = util::json_array(std::move(atom_rows));
    parse["relations"] = util::json_array(std::move(relation_rows));
    parse["books"] = util::json_array(std::move(book_rows));
    parse["candidates"] = util::json_array(std::move(candidate_rows));
    parse["unresolved"] = util::json_array(std::move(unresolved_rows));
    parse["provenance"] = util::json_array(std::move(provenance_rows));

    util::JsonObject obj;
    obj["ok"] = util::json_bool(true);
    obj["source"] = util::json_string("fold-graph-parser");
    obj["status"] = util::json_string("fold_graph_parse_ready");
    obj["response"] = util::json_string("Fold graph parse generated.");
    obj["query"] = util::json_string(q);
    obj["semantic_vector_contract"] = semantic_vector_contract();
    obj["fold_graph_parse"] = util::json_object(std::move(parse));
    return util::json_object(std::move(obj));
}

util::JsonPtr semantic_vector_contract() {
    util::JsonObject contract;
    contract["formula"] = util::json_string("S=W*C*cos(Q,K)");
    contract["mapping"] = util::json_string("Q=query_task_vector;K=semantic_key_vector_meta;V=book_payload");
    contract["selection_phase"] = util::json_string("Yax");
    contract["w_role"] = util::json_string("static_weight");
    contract["c_role"] = util::json_string("confidence");
    contract["r_role"] = util::json_string("cosine_relevance");
    return util::json_object(std::move(contract));
}

} // namespace khanary::io
