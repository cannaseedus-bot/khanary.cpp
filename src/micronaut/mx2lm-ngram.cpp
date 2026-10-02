#include "mx2lm-ngram.h"
#include "control_flow.h"
#include "eliza_operators.h"
#include "../kuhul/phase_engine.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace khanary::micronaut {

namespace {

bool is_token_char(unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '\'' || ch == '_';
}

std::string lower_ascii(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

float clamp01(float value) {
    if (!std::isfinite(value)) return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

void count_ngrams(const std::vector<std::string>& tokens, size_t width, NGramFreq& out) {
    if (width == 0 || tokens.size() < width) return;
    for (size_t i = 0; i + width <= tokens.size(); ++i) {
        std::string key = tokens[i];
        for (size_t j = 1; j < width; ++j) {
            key.push_back(' ');
            key += tokens[i + j];
        }
        ++out[key];
    }
}

struct ControlRule {
    std::string op_name;
    std::string semantic_role;
    std::string xcfe_op;
    std::string phase;
    uint32_t    arity = 1;
};

// The control-flow operator table in micronaut/control_flow.{h,cpp} is the one
// authority for control semantics (role, book, phase binding, XCFE lowering).
const std::unordered_map<std::string, ControlRule>& control_rules() {
    static const std::unordered_map<std::string, ControlRule> kRules = [] {
        std::unordered_map<std::string, ControlRule> rules;
        for (const auto& op : control::control_operators())
            rules.emplace(op.word, ControlRule{op.op_name, op.semantic_role,
                                               op.xcfe_op, op.phase, op.arity});
        return rules;
    }();
    return kRules;
}

bool is_control_token(const std::string& tok) {
    return control_rules().find(lower_ascii(tok)) != control_rules().end();
}

const std::unordered_set<std::string>& function_words() {
    static const std::unordered_set<std::string> kFunctionWords = {
        "a","an","the","this","that","these","those",
        "i","you","he","she","it","we","they","me","him","her","us","them",
        "my","your","his","its","our","their","mine","yours","ours","theirs",
        "is","am","are","was","were","be","being","been",
        "do","does","did","can","could","will","would","shall","should","may","might","must",
        "have","has","had",
        "to","of","in","on","at","by","for","with","without","from","into","over","under","through","across","about","as",
        "and","or","but","nor","so","yet",
        "if","then","than","because","while","when","where","who","whom","whose","which","what","why","how",
        "not","no","yes"
    };
    return kFunctionWords;
}

bool is_function_word_token(const std::string& tok) {
    return function_words().count(lower_ascii(tok)) != 0;
}

const std::unordered_set<std::string>& tool_tokens() {
    static const std::unordered_set<std::string> kToolTokens = {
        "tool","tools","function","functions","call","invoke","run","execute","command","commands",
        "mcp","api","rpc","plugin","route","router","delegate","dispatch","plan",
        "search","research","lookup","query","fetch","scrape","read","write",
        "build","compile","test","lint","deploy","convert","quantize",
        "web","file","shell","terminal","runtime"
    };
    return kToolTokens;
}

const std::unordered_set<std::string>& tool_action_tokens() {
    static const std::unordered_set<std::string> kToolActionTokens = {
        "call","invoke","run","execute","dispatch","delegate","search","research","lookup","query","fetch",
        "build","compile","test","lint","deploy","convert","quantize","read","write","plan"
    };
    return kToolActionTokens;
}

const std::unordered_set<std::string>& tool_target_tokens() {
    static const std::unordered_set<std::string> kToolTargetTokens = {
        "tool","tools","function","functions","command","commands","api","mcp","plugin","route","router",
        "web","file","shell","terminal","runtime","model","grammar","dataset"
    };
    return kToolTargetTokens;
}

bool is_tool_token(const std::string& tok) {
    return tool_tokens().count(lower_ascii(tok)) != 0;
}

std::string coarse_symbol_for_token(const std::string& tok) {
    const auto t = lower_ascii(tok);
    if (t.empty()) return "c";
    if (is_control_token(t)) return "ctrl:" + t;
    if (is_function_word_token(t)) return "f:" + t;
    if (is_tool_token(t)) return "t:" + t;
    return "c";
}

std::string join_tokens(const std::vector<std::string>& tokens) {
    std::string out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) out.push_back(' ');
        out += tokens[i];
    }
    return out;
}

std::vector<Mx2LmRelation> collapse_relations(const std::vector<Mx2LmRelation>& input) {
    std::vector<Mx2LmRelation> out;
    std::unordered_map<std::string, size_t> index;
    for (const auto& rel : input) {
        std::string key = rel.from + "\x1f" + rel.relation + "\x1f" + rel.to;
        auto it = index.find(key);
        if (it == index.end()) {
            index[key] = out.size();
            out.push_back(rel);
        } else {
            out[it->second].association_weight += rel.association_weight;
        }
    }
    for (auto& rel : out) {
        if (rel.association_weight < 0.0f) rel.association_weight = 0.0f;
        if (rel.association_weight > 1.0f) rel.association_weight = 1.0f;
    }
    return out;
}

std::string join_window(const std::vector<std::string>& tokens, size_t start, size_t width) {
    std::string gram;
    for (size_t i = 0; i < width; ++i) {
        if (!gram.empty()) gram.push_back(' ');
        gram += tokens[start + i];
    }
    return gram;
}

uint32_t width_from_gram_text(const std::string& gram) {
    if (gram.empty()) return 0;
    uint32_t width = 1;
    for (char c : gram) {
        if (c == ' ' || c == '>') width += 1;
    }
    return width;
}

std::string phase_for_gram_class(const std::string& gram_class) {
    if (gram_class == "unigram") return "Pop";
    if (gram_class == "coarse_gram" || gram_class == "function_gram") return "Yax";
    if (gram_class == "tool_gram") return "Sek";
    return "Wo";
}

std::vector<std::string> pattern_roles_from_container(
    const Mx2LmGramContainer& gram_container,
    const std::string& source_track)
{
    std::vector<std::string> roles;
    std::unordered_set<std::string> seen;
    auto push = [&](const std::string& role) {
        if (role.empty()) return;
        if (!seen.insert(role).second) return;
        roles.push_back(role);
    };

    push("pattern_observation");
    push("track:" + (source_track.empty() ? std::string("Mx2LM") : source_track));
    for (const auto& g : gram_container.grams) {
        push(g.role);
        if (g.gram_class == "tool_gram") push("tool_dispatch_pattern");
        if (g.gram_class == "function_gram") push("function_word_pattern");
        if (g.gram_class == "coarse_gram") push("structural_shape_pattern");
        if (roles.size() >= 12) break;
    }
    return roles;
}

std::uint64_t fnv1a64(const std::string& value) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : value) {
        h ^= static_cast<std::uint64_t>(ch);
        h *= 1099511628211ull;
    }
    return h;
}

float signed_component(std::uint64_t h, unsigned shift) {
    const auto part = static_cast<std::uint32_t>((h >> shift) & 0xffffu);
    return (static_cast<float>(part) / 32767.5f) - 1.0f;
}

void mix_projection_seed(const std::string& seed,
                         float weight,
                         float& x,
                         float& y,
                         float& z)
{
    if (seed.empty() || weight <= 0.0f) return;
    const auto h0 = fnv1a64(seed);
    const auto h1 = fnv1a64("y:" + seed);
    const auto h2 = fnv1a64("z:" + seed);
    x += signed_component(h0, 0) * weight;
    y += signed_component(h1, 8) * weight;
    z += signed_component(h2, 16) * weight;
}

std::string vector_key_safe(std::string value) {
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
    return out.empty() ? "projection" : out;
}

Mx2LmVectorProjection project_event_geometry(
    const Mx2LmPatternEvent& event,
    const Mx2LmGramContainer& gram_container)
{
    Mx2LmVectorProjection row;
    row.source_key = event.gram_container_key.empty()
        ? gram_container.key
        : event.gram_container_key;
    row.source_track = event.source_track.empty() ? "Mx2LM" : event.source_track;
    row.source_phase = event.source_phase.empty() ? "Wo" : event.source_phase;
    row.key = "vector:projection:" +
              vector_key_safe(row.source_track) + ":" +
              vector_key_safe(row.source_phase) + ":" +
              vector_key_safe(event.surface);

    mix_projection_seed("surface:" + event.surface, 1.0f, row.x, row.y, row.z);
    mix_projection_seed("track:" + row.source_track, 0.75f, row.x, row.y, row.z);
    mix_projection_seed("phase:" + row.source_phase, 0.50f, row.x, row.y, row.z);
    for (const auto& role : event.roles)
        mix_projection_seed("role:" + role, 0.40f, row.x, row.y, row.z);

    size_t grams_seen = 0;
    for (const auto& gram : gram_container.grams) {
        const float width = std::max(1.0f, static_cast<float>(gram.width));
        const float weight = std::min(1.0f, static_cast<float>(gram.count) / width);
        mix_projection_seed("gram:" + gram.gram_class + ":" + gram.gram,
                            0.25f * weight,
                            row.x,
                            row.y,
                            row.z);
        if (++grams_seen >= 32) break;
    }

    const float n = std::sqrt(row.x * row.x + row.y * row.y + row.z * row.z);
    if (std::isfinite(n) && n > 1e-6f) {
        row.x /= n;
        row.y /= n;
        row.z /= n;
        row.norm = 1.0f;
    } else {
        row.x = 1.0f;
        row.y = 0.0f;
        row.z = 0.0f;
        row.norm = 1.0f;
    }
    row.semantic_authority = false;
    return row;
}

} // namespace

std::vector<std::string> Mx2LmNGram::tokenize(std::string_view text) {
    std::vector<std::string> tokens;
    std::string current;
    current.reserve(32);

    for (char raw : text) {
        const auto ch = static_cast<unsigned char>(raw);
        if (is_token_char(ch)) {
            current.push_back(static_cast<char>(std::tolower(ch)));
        } else if (!current.empty()) {
            tokens.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

Mx2LmNGramModel Mx2LmNGram::build(std::string_view text) {
    Mx2LmNGramModel model;
    model.tokens = tokenize(text);
    count_ngrams(model.tokens, 2, model.bigrams);
    count_ngrams(model.tokens, 3, model.trigrams);
    count_ngrams(model.tokens, 4, model.quadgrams);
    return model;
}

std::vector<std::pair<std::string, uint32_t>> Mx2LmNGram::top(
    const NGramFreq& freq, size_t top_k)
{
    std::vector<std::pair<std::string, uint32_t>> ranked(freq.begin(), freq.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    if (ranked.size() > top_k) ranked.resize(top_k);
    return ranked;
}

Mx2LmGramTaxonomy Mx2LmNGram::classify_gram_taxonomy(const std::vector<std::string>& tokens) {
    Mx2LmGramTaxonomy out;
    if (tokens.empty()) return out;

    std::vector<std::string> norm;
    norm.reserve(tokens.size());
    for (const auto& raw : tokens) {
        const auto tok = lower_ascii(raw);
        if (!tok.empty()) norm.push_back(tok);
    }
    if (norm.empty()) return out;

    for (const auto& tok : norm)
        ++out.unigrams[tok];

    count_ngrams(norm, 2, out.bigrams);
    count_ngrams(norm, 3, out.trigrams);
    count_ngrams(norm, 4, out.quadgrams);

    std::vector<std::string> coarse_seq;
    coarse_seq.reserve(norm.size());
    for (const auto& tok : norm)
        coarse_seq.push_back(coarse_symbol_for_token(tok));
    count_ngrams(coarse_seq, 2, out.coarse_grams);
    count_ngrams(coarse_seq, 3, out.coarse_grams);
    count_ngrams(coarse_seq, 4, out.coarse_grams);

    for (size_t n = 1; n <= 3; ++n) {
        if (norm.size() < n) break;
        for (size_t i = 0; i + n <= norm.size(); ++i) {
            bool all_function = true;
            for (size_t k = 0; k < n; ++k) {
                if (!is_function_word_token(norm[i + k])) {
                    all_function = false;
                    break;
                }
            }
            if (!all_function) continue;
            ++out.function_grams[join_window(norm, i, n)];
        }
    }

    for (size_t n = 1; n <= 4; ++n) {
        if (norm.size() < n) break;
        for (size_t i = 0; i + n <= norm.size(); ++i) {
            bool has_tool = false;
            bool has_action = false;
            bool has_target = false;
            for (size_t k = 0; k < n; ++k) {
                const auto& tok = norm[i + k];
                if (is_tool_token(tok)) has_tool = true;
                if (tool_action_tokens().count(tok)) has_action = true;
                if (tool_target_tokens().count(tok)) has_target = true;
            }
            if (!has_tool) continue;
            if (n > 1 && !(has_action && has_target)) continue;
            ++out.tool_grams[join_window(norm, i, n)];
        }
    }

    return out;
}

Mx2LmGramTaxonomy Mx2LmNGram::classify_gram_taxonomy(std::string_view text) {
    return classify_gram_taxonomy(tokenize(text));
}

std::vector<Mx2LmTypedGram> Mx2LmNGram::typed_grams(
    const Mx2LmGramTaxonomy& taxonomy,
    size_t top_k_per_class)
{
    std::vector<Mx2LmTypedGram> out;
    auto append_class = [&](const NGramFreq& freq,
                            const std::string& gram_class,
                            const std::string& role,
                            uint32_t default_width) {
        std::vector<std::pair<std::string, uint32_t>> ranked(freq.begin(), freq.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });
        if (top_k_per_class > 0 && ranked.size() > top_k_per_class)
            ranked.resize(top_k_per_class);
        for (const auto& [gram, count] : ranked) {
            Mx2LmTypedGram row;
            row.gram = gram;
            row.gram_class = gram_class;
            row.role = role;
            row.phase = phase_for_gram_class(gram_class);
            row.count = count;
            row.width = (default_width > 0) ? default_width : width_from_gram_text(gram);
            row.tokens = tokenize(gram);
            out.push_back(std::move(row));
        }
    };

    append_class(taxonomy.unigrams, "unigram", "semantic_address", 1);
    append_class(taxonomy.bigrams, "bigram", "context_association", 2);
    append_class(taxonomy.trigrams, "trigram", "context_association", 3);
    append_class(taxonomy.quadgrams, "quadgram", "context_association", 4);
    append_class(taxonomy.coarse_grams, "coarse_gram", "structural_shape", 0);
    append_class(taxonomy.function_grams, "function_gram", "function_word_structure", 0);
    append_class(taxonomy.tool_grams, "tool_gram", "tool_dispatch_intent", 0);
    return out;
}

Mx2LmGramContainer Mx2LmNGram::compile_gram_container(
    const Mx2LmGramTaxonomy& taxonomy,
    size_t top_k_per_class)
{
    Mx2LmGramContainer c;
    c.grams = typed_grams(taxonomy, top_k_per_class);
    c.unigram_count = static_cast<uint32_t>(taxonomy.unigrams.size());
    c.bigram_count = static_cast<uint32_t>(taxonomy.bigrams.size());
    c.trigram_count = static_cast<uint32_t>(taxonomy.trigrams.size());
    c.quadgram_count = static_cast<uint32_t>(taxonomy.quadgrams.size());
    c.coarse_gram_count = static_cast<uint32_t>(taxonomy.coarse_grams.size());
    c.function_gram_count = static_cast<uint32_t>(taxonomy.function_grams.size());
    c.tool_gram_count = static_cast<uint32_t>(taxonomy.tool_grams.size());

    uint32_t observations = 0;
    for (const auto& g : c.grams) observations += g.count;
    c.total_observations = observations;
    return c;
}

Mx2LmGramContainer Mx2LmNGram::compile_gram_container(
    std::string_view text,
    size_t top_k_per_class)
{
    return compile_gram_container(classify_gram_taxonomy(text), top_k_per_class);
}

Mx2LmPatternContainer Mx2LmNGram::compile_pattern_container(
    std::string_view surface,
    const Mx2LmGramContainer& gram_container,
    const std::string& source_track,
    const std::string& source_phase,
    const std::string& outcome_kind,
    const std::string& card,
    bool accepted)
{
    Mx2LmPatternContainer c;
    Mx2LmPatternEvent event;
    event.source_track = source_track.empty() ? "Mx2LM" : source_track;
    event.source_phase = source_phase.empty() ? "Wo" : source_phase;
    event.target_namespace = c.namespace_id;
    event.surface = std::string(surface);
    event.roles = pattern_roles_from_container(gram_container, event.source_track);
    event.gram_container_key = gram_container.key;
    event.outcome_kind = outcome_kind.empty() ? "pattern_observed" : outcome_kind;
    event.outcome_status = accepted ? "accepted" : "rejected";
    event.card = card;
    event.accepted = accepted;
    c.events.push_back(std::move(event));
    c.event_count = static_cast<uint32_t>(c.events.size());
    return c;
}

Mx2LmVectorProjectionContainer Mx2LmNGram::compile_vector_projection_container(
    const Mx2LmGramContainer& gram_container,
    const Mx2LmPatternContainer& pattern_container)
{
    Mx2LmVectorProjectionContainer c;
    c.source_semantic_container = pattern_container.key.empty()
        ? std::string("mx2lm:pattern_container:v1")
        : pattern_container.key;

    for (const auto& event : pattern_container.events)
        c.projections.push_back(project_event_geometry(event, gram_container));

    if (c.projections.empty() && !gram_container.grams.empty()) {
        Mx2LmPatternEvent event;
        event.source_track = "Mx2LM";
        event.source_phase = gram_container.phase.empty() ? "Wo" : gram_container.phase;
        event.surface = "mx2lm:gram_container:v1";
        event.roles = pattern_roles_from_container(gram_container, event.source_track);
        event.gram_container_key = gram_container.key;
        c.projections.push_back(project_event_geometry(event, gram_container));
    }

    return c;
}

std::vector<Mx2LmControlBinding> Mx2LmNGram::resolve_control_bindings(
    const std::vector<std::string>& tokens)
{
    std::vector<Mx2LmControlBinding> out;
    const auto& rules = control_rules();

    for (const auto& raw_tok : tokens) {
        const std::string tok = lower_ascii(raw_tok);
        if (tok.empty()) continue;

        auto it = rules.find(tok);
        if (it == rules.end()) continue;

        Mx2LmControlBinding b;
        b.token = tok;
        b.operator_name = it->second.op_name;
        b.book_key = "ELIZA-1:" + b.operator_name;
        // Phase-bound control flow: the operator declares the phase its
        // construct runs in, and its effect lands in the C6 successor.
        b.detect_phase = it->second.phase;
        b.emit_phase   = control::phase_after(it->second.phase);
        b.semantic_role = it->second.semantic_role;
        b.xcfe_op = it->second.xcfe_op;
        b.arity = it->second.arity;
        out.push_back(std::move(b));
    }
    return out;
}

std::vector<Mx2LmRelation> Mx2LmNGram::infer_relations(
    const std::vector<std::string>& tokens)
{
    std::vector<Mx2LmRelation> edges;
    if (tokens.empty()) return edges;

    std::vector<std::string> norm;
    norm.reserve(tokens.size());
    for (const auto& t : tokens) {
        const std::string lowered = lower_ascii(t);
        if (!lowered.empty()) norm.push_back(lowered);
    }
    if (norm.empty()) return edges;

    // 1) Control-centered relation edges:
    //    noun/control/noun -> from -(control)-> to
    for (size_t i = 1; i + 1 < norm.size(); ++i) {
        const auto& a = norm[i - 1];
        const auto& c = norm[i];
        const auto& b = norm[i + 1];
        if (!is_control_token(c)) continue;
        if (is_control_token(a) || is_control_token(b)) continue;
        Mx2LmRelation rel;
        rel.from = a;
        rel.relation = c;
        rel.to = b;
        rel.association_weight = clamp01(1.0f);
        edges.push_back(std::move(rel));
    }

    // 2) Associative adjacency edges for non-control lexical flow.
    for (size_t i = 0; i + 1 < norm.size(); ++i) {
        const auto& a = norm[i];
        const auto& b = norm[i + 1];
        if (is_control_token(a) || is_control_token(b)) continue;
        Mx2LmRelation rel;
        rel.from = a;
        rel.relation = "assoc";
        rel.to = b;
        rel.association_weight = clamp01(0.5f);
        edges.push_back(std::move(rel));
    }

    return collapse_relations(edges);
}

Mx2LmStructuredProjection Mx2LmNGram::compose(const std::vector<std::string>& tokens) {
    Mx2LmStructuredProjection out;
    out.control_bindings = resolve_control_bindings(tokens);
    out.relations = infer_relations(tokens);

    std::unordered_map<std::string, Mx2LmControlBinding> control_by_token;
    for (const auto& cb : out.control_bindings) {
        if (!control_by_token.count(cb.token))
            control_by_token[cb.token] = cb;
    }

    std::unordered_set<std::string> seen;
    for (const auto& tok : tokens) {
        if (seen.count(tok)) continue;
        seen.insert(tok);
        Mx2LmSemanticNode node;
        node.id = tok;
        auto it = control_by_token.find(tok);
        if (it != control_by_token.end()) {
            node.kind = "control_node";
            node.meaning_key = it->second.semantic_role;
            node.operator_name = it->second.operator_name;
            node.book_key = it->second.book_key;
        } else {
            node.kind = "semantic_node";
            node.meaning_key = tok;
        }
        out.nodes.push_back(std::move(node));
    }

    out.clause = join_tokens(tokens);
    out.sentence = out.clause;
    if (!out.sentence.empty()) {
        out.sentence[0] = (char)std::toupper((unsigned char)out.sentence[0]);
        const char last = out.sentence.back();
        if (last != '.' && last != '!' && last != '?')
            out.sentence.push_back('.');
    }
    return out;
}

} // namespace khanary::micronaut
