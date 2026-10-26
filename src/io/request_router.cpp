#include "request_router.h"
#include "../kuhul/state_algebra.h"
#include <filesystem>
#include <fstream>
#include "../util/json.h"
#include "../util/lexicon.h"
#include <sstream>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iterator>
#include <filesystem>

namespace khanary::io {

namespace {

namespace fs = std::filesystem;

static bool ends_with(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

static const std::unordered_map<std::string, std::string>& lexical_aliases() {
    static const std::unordered_map<std::string, std::string> kAliases = {
        {"apps", "app"}, {"application", "app"}, {"applications", "app"}, {"platform", "app"},
        {"website", "app"}, {"websites", "app"}, {"site", "app"}, {"sites", "app"},
        {"program", "app"}, {"programs", "app"}, {"software", "app"}, {"tool", "app"}, {"tools", "app"},
        {"query", "search"}, {"lookup", "search"}, {"find", "search"}, {"retrieval", "search"},
        {"retrieved", "search"}, {"searching", "search"},
        {"build", "create"}, {"builder", "create"}, {"generate", "create"}, {"make", "create"},
        {"debug", "fix"}, {"bugfix", "fix"}, {"repair", "fix"}, {"troubleshoot", "fix"},
        {"memory", "recall"}, {"remember", "recall"}, {"recalled", "recall"},
        {"research", "explore"}, {"investigate", "explore"}, {"analysis", "explore"},
    };
    return kAliases;
}

static bool is_function_word(const std::string& token) {
    static const std::unordered_set<std::string> kFunctionWords = {
        "a","an","the","this","that","these","those",
        "i","me","you","he","him","she","her","it","we","us","they","them",
        "my","your","his","its","our","their","mine","yours","hers","ours","theirs",
        "myself","yourself","himself","herself","itself","ourselves","themselves",
        "who","whom","whose","which","what","when","where","why","how",
        "in","on","at","by","for","from","to","of","with","without","under","over",
        "above","below","before","after","between","among","into","onto","through",
        "across","around","about","towards","toward","as","if","than","then",
        "and","or","but","nor","so","yet","because","while","although","though",
        "unless","until","since","however","thus",
        "is","am","are","was","were","be","been","being",
        "do","does","did","doing","have","has","had","having",
        "can","could","may","might","must","shall","should","will","would","ought",
        "not","no","yes","ok","okay",
        "up","down","out","off","on",
        "oh","ah","eh","there","here"
    };
    return kFunctionWords.count(token) > 0;
}

struct ContentFunctionProfile {
    float content_ratio = 0.0f;
    float function_ratio = 0.0f;
    float function_binding_ratio = 0.0f;
};

static ContentFunctionProfile profile_content_function_tokens(const std::vector<std::string>& raw_tokens) {
    ContentFunctionProfile out;
    if (raw_tokens.empty()) return out;

    std::vector<bool> is_fn;
    is_fn.reserve(raw_tokens.size());

    size_t content_count = 0;
    size_t function_count = 0;
    for (const auto& tok : raw_tokens) {
        if (tok.empty()) continue;
        const bool fn = is_function_word(tok);
        is_fn.push_back(fn);
        if (fn) ++function_count;
        else ++content_count;
    }

    const size_t total = content_count + function_count;
    if (total == 0) return out;

    out.content_ratio = static_cast<float>(content_count) / static_cast<float>(total);
    out.function_ratio = static_cast<float>(function_count) / static_cast<float>(total);

    if (function_count == 0 || total < 2) return out;

    size_t function_with_content_neighbor = 0;
    for (size_t i = 0; i < is_fn.size(); ++i) {
        if (!is_fn[i]) continue;
        const bool left_content = (i > 0) && !is_fn[i - 1];
        const bool right_content = (i + 1 < is_fn.size()) && !is_fn[i + 1];
        if (left_content || right_content) ++function_with_content_neighbor;
    }
    out.function_binding_ratio =
        static_cast<float>(function_with_content_neighbor) / static_cast<float>(function_count);
    return out;
}

struct RouterVector3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool valid = false;
};

static RouterVector3 phase_unit_vector(kuhul::Phase phase) {
    constexpr double kPi = 3.14159265358979323846;
    const double angle = (2.0 * kPi / 6.0) * static_cast<int>(phase);
    RouterVector3 v;
    v.x = std::cos(angle);
    v.y = std::sin(angle);
    v.z = 0.0;
    v.valid = true;
    return v;
}

static RouterVector3 key_vector_for_entry(const micronaut::MicronautEntry& entry) {
    RouterVector3 v;
    if (entry.vector_meta.declared) {
        v.x = entry.vector_meta.x;
        v.y = entry.vector_meta.y;
        v.z = entry.vector_meta.z;
        v.valid = true;
        return v;
    }
    return phase_unit_vector(entry.primary_phase);
}

static RouterVector3 derive_router_fallback_vector(const std::vector<std::string>& tokens_lower) {
    if (tokens_lower.empty()) return {};
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    std::size_t kept = 0;
    for (std::size_t i = 0; i < tokens_lower.size(); ++i) {
        const auto& token = tokens_lower[i];
        if (token.empty()) continue;
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : token) {
            h ^= static_cast<uint64_t>(c);
            h *= 1099511628211ull;
        }
        sx += std::sin(static_cast<double>(h & 0xffffull) * 0.0009 + static_cast<double>(i) * 0.17);
        sy += std::cos(static_cast<double>((h >> 16) & 0xffffull) * 0.0011 + static_cast<double>(i) * 0.13);
        sz += std::sin(static_cast<double>((h >> 32) & 0xffffull) * 0.0013 + static_cast<double>(i) * 0.11);
        ++kept;
    }
    if (kept == 0) return {};
    const double nx = sx / static_cast<double>(kept);
    const double ny = sy / static_cast<double>(kept);
    const double nz = sz / static_cast<double>(kept);
    const double n = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (!(n > 1e-9)) return {};
    return {nx / n, ny / n, nz / n, true};
}

static double router_cosine_relevance(const RouterVector3& query_vector, const RouterVector3& key_vector) {
    if (!query_vector.valid || !key_vector.valid) return 0.5;
    const double dot = query_vector.x * key_vector.x +
                       query_vector.y * key_vector.y +
                       query_vector.z * key_vector.z;
    const double cosine = std::clamp(dot, -1.0, 1.0);
    return std::clamp(0.5 * (cosine + 1.0), 0.0, 1.0);
}

static double router_wcr_score(double static_weight, double confidence, double relevance) {
    return std::clamp(
        std::clamp(static_weight, 0.0, 1.0) *
        std::clamp(confidence, 0.0, 1.0) *
        std::clamp(relevance, 0.0, 1.0),
        0.0, 1.0);
}

struct SenseOption {
    std::string sense_id;
    std::vector<std::string> cues;
};

struct SenseDecision {
    std::string term;
    std::string sense_id;
    std::unordered_set<std::string> cues;
    float confidence = 0.0f;
};

static const std::unordered_map<std::string, std::vector<SenseOption>>& wsd_lexeme_specs() {
    static const std::unordered_map<std::string, std::vector<SenseOption>> kSpecs = {
        {"simulation", {
            {"modeling_runtime", {"model", "modeling", "runtime", "engine", "compute", "training", "inference", "physics", "digital", "synthetic", "data"}},
            {"pretend_imitation", {"pretend", "imitation", "fake", "roleplay", "acting", "theater", "game", "fiction"}},
            {"medical_training", {"medical", "clinical", "patient", "surgery", "hospital", "anatomy", "procedure"}}
        }},
        {"play", {
            {"theater_drama", {"theater", "stage", "actor", "drama", "script", "romeo", "juliet", "scene"}},
            {"sports_tactic", {"coach", "team", "offense", "defense", "defensive", "quarterback", "strategy", "formation"}},
            {"recreation_activity", {"children", "park", "fun", "toy", "recess", "outside", "recreation"}}
        }},
        {"model", {
            {"ai_architecture", {"ai", "neural", "weights", "inference", "training", "runtime", "token"}},
            {"representation_schema", {"schema", "representation", "semantic", "graph", "shape", "abstraction"}},
            {"person_object", {"person", "fashion", "pose", "photo", "camera"}}
        }},
        {"field", {
            {"domain_area", {"domain", "discipline", "subject", "context", "topic"}},
            {"spatial_region", {"space", "region", "area", "map", "world", "inside"}},
            {"physical_vector_field", {"vector", "physics", "force", "gradient", "flow"}}
        }}
    };
    return kSpecs;
}

static std::string canonical_lexeme(const std::string& token) {
    if (token.empty()) return token;
    auto lemma = token;

    const auto& aliases = lexical_aliases();
    if (auto it = aliases.find(lemma); it != aliases.end()) return it->second;

    if (lemma.size() > 5 && ends_with(lemma, "ing")) {
        lemma.erase(lemma.size() - 3);
    } else if (lemma.size() > 4 && ends_with(lemma, "ied")) {
        lemma.replace(lemma.size() - 3, 3, "y");
    } else if (lemma.size() > 4 && ends_with(lemma, "ed")) {
        lemma.erase(lemma.size() - 2);
    } else if (lemma.size() > 4 && ends_with(lemma, "es")) {
        lemma.erase(lemma.size() - 2);
    } else if (lemma.size() > 3 && ends_with(lemma, "s")) {
        lemma.erase(lemma.size() - 1);
    }

    if (auto it = aliases.find(lemma); it != aliases.end()) return it->second;
    return lemma;
}

std::vector<std::string> tokenize_distributional_text(const std::string& text) {
    // Lexical authority moved into the grammar: the "alnum_plain" @profile
    // (word_plain class) declared in data/grammar/cpp.code-grammar.ebnf,
    // mirrored in the compiled-in core table (src/util/lexicon.cpp). The two
    // same-named copies in request_router.cpp / code_parser.cpp had already
    // drifted (underscore handling); the shared profile table makes that
    // choice visible and named instead of hidden.
    return lexicon::core_tokenize(text, "alnum_plain");
}

struct MicronautUriCall {
    bool valid = false;
    std::string scheme;
    std::string kind; // action | verb | pattern | template | grammar
    std::string op;
    std::vector<std::string> segments;
};

static std::string uri_lower_ascii_copy(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::string trim_ascii_copy(const std::string& input) {
    size_t b = 0;
    while (b < input.size() &&
           std::isspace(static_cast<unsigned char>(input[b]))) ++b;
    size_t e = input.size();
    while (e > b &&
           std::isspace(static_cast<unsigned char>(input[e - 1]))) --e;
    return input.substr(b, e - b);
}

static std::vector<std::string> split_non_empty_segments(const std::string& path) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : path) {
        if (c == '/') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static MicronautUriCall parse_micronaut_uri_call(const std::string& raw_query) {
    MicronautUriCall out;
    const std::string trimmed = trim_ascii_copy(raw_query);
    if (trimmed.empty()) return out;

    const size_t sep_global = trimmed.find("://");
    if (sep_global == std::string::npos || sep_global == 0) return out;

    size_t begin = sep_global;
    while (begin > 0 &&
           !std::isspace(static_cast<unsigned char>(trimmed[begin - 1]))) --begin;
    size_t end = sep_global + 3;
    while (end < trimmed.size() &&
           !std::isspace(static_cast<unsigned char>(trimmed[end]))) ++end;

    const std::string uri = trimmed.substr(begin, end - begin);
    const size_t sep = uri.find("://");
    if (sep == std::string::npos || sep == 0) return out;

    out.scheme = uri_lower_ascii_copy(uri.substr(0, sep));
    std::string rest = uri.substr(sep + 3);
    if (!rest.empty() && rest.front() == '/') rest.erase(rest.begin());
    const size_t qmark = rest.find('?');
    if (qmark != std::string::npos) rest = rest.substr(0, qmark);
    out.segments = split_non_empty_segments(rest);
    for (auto& seg : out.segments) seg = uri_lower_ascii_copy(seg);

    out.kind = "action";
    if (!out.segments.empty()) {
        if (out.segments[0] == "verb" ||
            out.segments[0] == "pattern" ||
            out.segments[0] == "template" ||
            out.segments[0] == "grammar") {
            out.kind = out.segments[0];
            if (out.segments.size() > 1) out.op = out.segments[1];
        } else {
            out.op = out.segments[0];
        }
    }

    out.valid = !out.scheme.empty();
    return out;
}

static std::string normalize_scheme_alias(std::string value) {
    value = uri_lower_ascii_copy(value);
    std::string out;
    out.reserve(value.size());
    bool last_dash = false;
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(c));
            last_dash = false;
        } else if (c == '-' || c == '_') {
            if (!last_dash && !out.empty()) {
                out.push_back('-');
                last_dash = true;
            }
        }
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out;
}

static void push_unique_ci(std::vector<std::string>& out, const std::string& value) {
    if (value.empty()) return;
    const std::string canon = uri_lower_ascii_copy(value);
    for (const auto& existing : out) {
        if (uri_lower_ascii_copy(existing) == canon) return;
    }
    out.push_back(value);
}

static std::vector<std::string> scheme_aliases_for_entry(const micronaut::MicronautEntry& entry) {
    std::vector<std::string> aliases;
    for (const auto& s : entry.route_contract.schemes)
        push_unique_ci(aliases, normalize_scheme_alias(s));

    const std::string normalized_id = normalize_scheme_alias(entry.id);
    if (!normalized_id.empty()) push_unique_ci(aliases, normalized_id);
    if (auto dash = normalized_id.find('-'); dash != std::string::npos && dash > 0) {
        push_unique_ci(aliases, normalized_id.substr(0, dash));
    }

    std::string trimmed = normalized_id;
    if (trimmed.size() > 2 && trimmed.ends_with("-u")) {
        trimmed = trimmed.substr(0, trimmed.size() - 2);
        if (!trimmed.empty()) push_unique_ci(aliases, trimmed);
    }
    if (trimmed.size() > 6 && trimmed.ends_with("-agent")) {
        trimmed = trimmed.substr(0, trimmed.size() - 6);
        if (!trimmed.empty()) push_unique_ci(aliases, trimmed);
    }
    return aliases;
}

static bool route_contract_contains_ci(const std::vector<std::string>& rows, const std::string& value) {
    if (value.empty()) return false;
    const std::string canon = uri_lower_ascii_copy(value);
    for (const auto& row : rows) {
        if (uri_lower_ascii_copy(row) == canon) return true;
    }
    return false;
}

static bool entry_supports_uri_call(const micronaut::MicronautEntry& entry,
                                    const MicronautUriCall& call) {
    if (!call.valid) return false;
    bool scheme_match = false;
    for (const auto& alias : scheme_aliases_for_entry(entry)) {
        if (alias == call.scheme) {
            scheme_match = true;
            break;
        }
    }
    if (!scheme_match) return false;
    if (call.op.empty()) return true;

    if (call.kind == "pattern")
        return !entry.route_contract.aiml_patterns.empty();
    if (call.kind == "template")
        return !entry.route_contract.aiml_templates.empty();
    if (!entry.route_contract.verbs.empty())
        return route_contract_contains_ci(entry.route_contract.verbs, call.op);

    for (const auto& cap : entry.capabilities) {
        const auto lcap = uri_lower_ascii_copy(cap);
        if (lcap == call.op) return true;
        if (lcap.rfind(call.op + "_", 0) == 0) return true;
        if (lcap.rfind(call.op + ":", 0) == 0) return true;
    }
    for (const auto& g : entry.governs)
        if (uri_lower_ascii_copy(g) == call.op) return true;
    return false;
}

static bool is_composition_bridge(const std::string& token) {
    static const std::unordered_set<std::string> kBridge = {
        "in","of","for","with","to","from","on","into","within","about","between","through","across"
    };
    return kBridge.count(token) > 0;
}

static const std::unordered_map<std::string, std::vector<std::string>>& compositional_aliases() {
    static const std::unordered_map<std::string, std::vector<std::string>> kAliases = {
        {"organic_life_in_space", {"astrobiology", "exobiology", "biosignature", "habitability"}},
        {"life_in_space", {"astrobiology", "exobiology", "biosignature"}},
        {"search_for_life", {"biosignature", "habitability", "astrobiology"}},
        {"signs_of_life", {"biosignature", "bio_marker", "habitability"}}
    };
    return kAliases;
}

static std::unordered_set<std::string> extract_compositional_signatures(
    const std::vector<std::string>& raw_tokens)
{
    std::unordered_set<std::string> out;
    if (raw_tokens.empty()) return out;

    for (size_t i = 0; i < raw_tokens.size(); ++i) {
        const auto& tok_i = raw_tokens[i];
        if (tok_i.size() < 2) continue;
        if (is_function_word(tok_i)) continue;

        std::string sig = canonical_lexeme(tok_i);
        size_t content_count = 1;
        for (size_t j = i + 1; j < raw_tokens.size() && j < i + 5; ++j) {
            const auto& tj = raw_tokens[j];
            if (tj.size() < 2) break;
            if (is_function_word(tj)) {
                if (!is_composition_bridge(tj)) break;
                sig += "_" + tj;
                continue;
            }
            sig += "_" + canonical_lexeme(tj);
            ++content_count;
            if (content_count >= 2) out.insert(sig);
        }
    }

    const auto& aliases = compositional_aliases();
    std::vector<std::string> seeds(out.begin(), out.end());
    for (const auto& sig : seeds) {
        auto it = aliases.find(sig);
        if (it == aliases.end()) continue;
        for (const auto& alias : it->second) {
            if (alias.empty()) continue;
            out.insert(alias);
            out.insert(canonical_lexeme(alias));
        }
    }
    return out;
}

static std::unordered_set<std::string> extract_coarse_ngrams(
    const std::vector<std::string>& raw_tokens,
    size_t min_n = 2,
    size_t max_n = 3)
{
    std::unordered_set<std::string> out;
    if (raw_tokens.empty() || min_n == 0 || max_n < min_n) return out;

    std::vector<std::string> coarse_seq;
    coarse_seq.reserve(raw_tokens.size());
    for (const auto& tok : raw_tokens) {
        if (tok.size() < 2) continue;
        if (is_function_word(tok)) coarse_seq.push_back("f:" + tok);
        else coarse_seq.push_back("c");
    }
    if (coarse_seq.size() < min_n) return out;

    for (size_t n = min_n; n <= max_n; ++n) {
        if (coarse_seq.size() < n) break;
        for (size_t i = 0; i + n <= coarse_seq.size(); ++i) {
            std::string gram = coarse_seq[i];
            for (size_t k = 1; k < n; ++k)
                gram += ">" + coarse_seq[i + k];
            out.insert(std::move(gram));
        }
    }
    return out;
}

static std::unordered_set<std::string> extract_tool_grams(
    const std::vector<std::string>& raw_tokens,
    size_t min_n = 1,
    size_t max_n = 4)
{
    static const std::unordered_set<std::string> kToolTokens = {
        "tool","tools","function","functions","call","invoke","run","execute","command","commands",
        "mcp","api","rpc","plugin","route","router","delegate","dispatch","plan",
        "search","research","lookup","query","fetch","scrape","read","write",
        "build","compile","test","lint","deploy","convert","quantize",
        "web","file","shell","terminal","runtime","model","grammar"
    };
    static const std::unordered_set<std::string> kToolActionTokens = {
        "call","invoke","run","execute","dispatch","delegate","search","research","lookup",
        "query","fetch","build","compile","test","lint","deploy","convert","quantize","read","write","plan"
    };
    static const std::unordered_set<std::string> kToolTargetTokens = {
        "tool","tools","function","functions","command","commands","api","mcp","plugin",
        "web","file","shell","terminal","runtime","model","grammar"
    };

    std::unordered_set<std::string> out;
    if (raw_tokens.empty() || min_n == 0 || max_n < min_n) return out;

    for (size_t n = min_n; n <= max_n; ++n) {
        if (raw_tokens.size() < n) break;
        for (size_t i = 0; i + n <= raw_tokens.size(); ++i) {
            bool has_tool = false;
            bool has_action = false;
            bool has_target = false;
            std::string gram;
            for (size_t k = 0; k < n; ++k) {
                const auto tok = canonical_lexeme(raw_tokens[i + k]);
                if (tok.empty()) continue;
                if (!gram.empty()) gram.push_back(' ');
                gram += tok;
                if (kToolTokens.count(tok)) has_tool = true;
                if (kToolActionTokens.count(tok)) has_action = true;
                if (kToolTargetTokens.count(tok)) has_target = true;
            }
            if (gram.empty()) continue;
            if (!has_tool) continue;
            if (n > 1 && !(has_action && has_target)) continue;
            out.insert(std::move(gram));
        }
    }
    return out;
}

static std::vector<SenseDecision> resolve_query_senses(
    const std::unordered_set<std::string>& query_lexemes,
    const std::vector<std::string>& query_raw_tokens)
{
    std::vector<SenseDecision> out;
    std::unordered_set<std::string> qset = query_lexemes;
    for (const auto& tok : query_raw_tokens) {
        if (tok.size() < 2) continue;
        qset.insert(tok);
        qset.insert(canonical_lexeme(tok));
    }

    const auto& specs = wsd_lexeme_specs();
    for (const auto& [term, options] : specs) {
        if (options.empty()) continue;
        if (!qset.count(term)) continue;

        const SenseOption* picked = &options.front();
        size_t picked_hits = 0;
        for (const auto& opt : options) {
            size_t hits = 0;
            for (const auto& cue : opt.cues) {
                const auto cc = canonical_lexeme(cue);
                if (qset.count(cue) || qset.count(cc)) ++hits;
            }
            if (hits > picked_hits) {
                picked = &opt;
                picked_hits = hits;
            }
        }

        SenseDecision d;
        d.term = term;
        d.sense_id = picked->sense_id;
        for (const auto& cue : picked->cues) {
            d.cues.insert(cue);
            d.cues.insert(canonical_lexeme(cue));
        }

        const float cue_norm = picked->cues.empty()
            ? 0.0f
            : static_cast<float>(picked_hits) / static_cast<float>(picked->cues.size());
        d.confidence = (picked_hits > 0)
            ? std::clamp(0.45f + 0.55f * cue_norm, 0.45f, 1.0f)
            : 0.35f;
        out.push_back(std::move(d));
    }
    return out;
}

static const std::unordered_map<std::string, std::string>& nsm_prime_aliases() {
    static const std::unordered_map<std::string, std::string> kPrimeAliases = {
        {"i","i"}, {"you","you"}, {"someone","someone"}, {"somebody","someone"}, {"people","people"},
        {"something","something"}, {"thing","something"}, {"body","body"}, {"kind","kind"}, {"part","part"},
        {"this","this"}, {"same","the_same"}, {"other","other"}, {"else","other"}, {"another","other"},
        {"one","one"}, {"two","two"}, {"some","some"}, {"all","all"}, {"much","much_many"},
        {"many","much_many"}, {"little","little_few"}, {"few","little_few"},
        {"good","good"}, {"bad","bad"}, {"big","big"}, {"small","small"},
        {"think","think"}, {"know","know"}, {"want","want"}, {"feel","feel"}, {"see","see"}, {"hear","hear"},
        {"say","say"}, {"word","words"}, {"words","words"}, {"true","true"},
        {"do","do"}, {"happen","happen"}, {"move","move"}, {"live","live"}, {"die","die"},
        {"when","when_time"}, {"time","when_time"}, {"now","now"}, {"before","before"}, {"after","after"},
        {"moment","moment"}, {"where","where_place"}, {"place","where_place"}, {"here","here"},
        {"above","above"}, {"below","below"}, {"far","far"}, {"near","near"}, {"side","side"},
        {"inside","inside"}, {"touch","touch"}, {"contact","touch"},
        {"not","not"}, {"maybe","maybe"}, {"can","can"}, {"because","because"}, {"if","if"},
        {"very","very"}, {"more","more"}, {"like","like_as_way"}, {"as","like_as_way"}, {"way","like_as_way"},
        {"there","there_is"}, {"is","be"}, {"be","be"}, {"mine","mine"}
    };
    return kPrimeAliases;
}

static std::unordered_set<std::string> extract_nsm_primes(const std::string& text) {
    std::unordered_set<std::string> out;
    const auto toks = tokenize_distributional_text(text);
    if (toks.empty()) return out;

    const auto& aliases = nsm_prime_aliases();
    for (const auto& tok : toks) {
        const auto canon = canonical_lexeme(tok);
        if (auto it = aliases.find(canon); it != aliases.end()) out.insert(it->second);
        else if (auto jt = aliases.find(tok); jt != aliases.end()) out.insert(jt->second);
    }

    for (size_t i = 0; i < toks.size(); ++i) {
        if (i + 1 < toks.size() && toks[i] == "the" && toks[i + 1] == "same")
            out.insert("the_same");
        if (i + 1 < toks.size() && toks[i] == "there" && toks[i + 1] == "is")
            out.insert("there_is");
        if (i + 1 < toks.size() && toks[i] == "not" && toks[i + 1] == "want")
            out.insert("dont_want");
        if (i + 2 < toks.size() && toks[i] == "don" && toks[i + 1] == "t" && toks[i + 2] == "want")
            out.insert("dont_want");
        if (i + 2 < toks.size() && toks[i] == "a" && toks[i + 1] == "long" && toks[i + 2] == "time")
            out.insert("a_long_time");
        if (i + 2 < toks.size() && toks[i] == "a" && toks[i + 1] == "short" && toks[i + 2] == "time")
            out.insert("a_short_time");
        if (i + 2 < toks.size() && toks[i] == "for" && toks[i + 1] == "some" && toks[i + 2] == "time")
            out.insert("for_some_time");
    }
    return out;
}

struct ResolverPolicyContext {
    std::string op;
    std::string resolver_policy;
    std::string required_primitive;
    std::vector<std::string> required_capabilities;

    bool constrained() const {
        return !resolver_policy.empty() || !required_primitive.empty() || !required_capabilities.empty();
    }
};

struct EligibilityCheck {
    bool capable = true;
    bool primitive_ok = true;
    std::vector<std::string> missing;
};

static std::string lower_ascii_copy(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static bool contains_phrase_lc(const std::string& text_lower, const std::string& phrase_lower) {
    return !phrase_lower.empty() && text_lower.find(phrase_lower) != std::string::npos;
}

static void push_unique_token(std::vector<std::string>& out, const std::string& value) {
    if (value.empty()) return;
    const auto canon = lower_ascii_copy(value);
    for (const auto& v : out) {
        if (lower_ascii_copy(v) == canon) return;
    }
    out.push_back(value);
}

static void add_required_capability(ResolverPolicyContext& ctx, const std::string& cap) {
    if (cap.empty()) return;
    push_unique_token(ctx.required_capabilities, cap);
}

static void parse_capability_list_text(ResolverPolicyContext& ctx, const std::string& text) {
    std::string token;
    auto flush = [&]() {
        if (token.empty()) return;
        add_required_capability(ctx, token);
        token.clear();
    };
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '_' || c == ':' || c == '-') {
            token.push_back(static_cast<char>(c));
        } else {
            flush();
        }
    }
    flush();
}

static ResolverPolicyContext derive_resolver_policy_context(const RouteRequest& req) {
    ResolverPolicyContext ctx;

    if (req.payload && req.payload->is_object()) {
        if (auto op = req.payload->get_string("op")) ctx.op = *op;
        if (auto policy = req.payload->get_string("resolver_policy")) ctx.resolver_policy = *policy;
        if (auto prim = req.payload->get_string("required_primitive")) ctx.required_primitive = *prim;
        if (auto single = req.payload->get_string("requires")) parse_capability_list_text(ctx, *single);

        if (const auto* req_caps = req.payload->get("required_capabilities");
            req_caps && req_caps->is_array()) {
            for (const auto& rv : req_caps->as_array()) {
                if (rv && rv->is_string()) add_required_capability(ctx, rv->as_string());
            }
        }
        if (const auto* req_caps = req.payload->get("requirements");
            req_caps && req_caps->is_array()) {
            for (const auto& rv : req_caps->as_array()) {
                if (rv && rv->is_string()) add_required_capability(ctx, rv->as_string());
            }
        }
    }

    const auto qlow = lower_ascii_copy(req.query);
    if ((contains_phrase_lc(qlow, "solve linear system") ||
         contains_phrase_lc(qlow, "linear system")) &&
        !contains_phrase_lc(qlow, "symbolic only")) {
        if (ctx.op.empty()) ctx.op = "SOLVE_LINEAR_SYSTEM";
        if (ctx.required_primitive.empty()) ctx.required_primitive = "matrix_buffer";
        if (ctx.resolver_policy.empty()) ctx.resolver_policy = "matrix_buffer.required";
        add_required_capability(ctx, "MATRIX_BUFFER_SOLVE");
    }
    if (contains_phrase_lc(qlow, "matrix multiply") || contains_phrase_lc(qlow, "matmul")) {
        if (ctx.op.empty()) ctx.op = "MATRIX_MULTIPLY";
        if (ctx.required_primitive.empty()) ctx.required_primitive = "matrix_buffer";
        if (ctx.resolver_policy.empty()) ctx.resolver_policy = "matrix_buffer.required";
        add_required_capability(ctx, "MATRIX_BUFFER_MATMUL");
    }
    if (contains_phrase_lc(qlow, "eigenvalue") || contains_phrase_lc(qlow, "eigenvalues")) {
        if (ctx.op.empty()) ctx.op = "EIGENVALUES";
        if (ctx.required_primitive.empty()) ctx.required_primitive = "matrix_buffer";
        if (ctx.resolver_policy.empty()) ctx.resolver_policy = "matrix_buffer.required";
        add_required_capability(ctx, "MATRIX_BUFFER_APPLY_OP");
    }

    const auto policy_l = lower_ascii_copy(ctx.resolver_policy);
    if (ctx.required_primitive.empty()) {
        if (contains_phrase_lc(policy_l, "matrix_buffer.required")) ctx.required_primitive = "matrix_buffer";
        else if (contains_phrase_lc(policy_l, "symbolic_math.required")) ctx.required_primitive = "symbolic_math";
    }

    return ctx;
}

static bool has_capability_ci(const micronaut::MicronautEntry& entry, const std::string& required) {
    const auto need = lower_ascii_copy(required);
    for (const auto& cap : entry.capabilities) {
        if (lower_ascii_copy(cap) == need) return true;
    }
    return false;
}

static bool has_required_primitive(
    const micronaut::MicronautEntry& entry,
    const std::string& required_primitive)
{
    if (required_primitive.empty()) return true;
    const auto prim = lower_ascii_copy(required_primitive);

    for (const auto& cap_raw : entry.capabilities) {
        const auto cap = lower_ascii_copy(cap_raw);
        if (cap == prim) return true;
        if (cap.rfind(prim + "_", 0) == 0) return true;
        if (cap.rfind(prim + ":", 0) == 0) return true;
    }

    if (prim == "matrix_buffer") {
        for (const auto& cap_raw : entry.capabilities) {
            const auto cap = lower_ascii_copy(cap_raw);
            if (cap.rfind("matrix_buffer_", 0) == 0) return true;
        }
    }
    if (prim == "symbolic_math") {
        for (const auto& cap_raw : entry.capabilities) {
            const auto cap = lower_ascii_copy(cap_raw);
            if (cap.rfind("math_symbolic_", 0) == 0) return true;
        }
    }
    return false;
}

static EligibilityCheck evaluate_eligibility(
    const micronaut::MicronautEntry& entry,
    const ResolverPolicyContext& policy)
{
    EligibilityCheck out;
    for (const auto& req : policy.required_capabilities) {
        if (!has_capability_ci(entry, req)) {
            out.capable = false;
            push_unique_token(out.missing, req);
        }
    }

    out.primitive_ok = has_required_primitive(entry, policy.required_primitive);
    if (!out.primitive_ok) {
        out.capable = false;
        if (!policy.required_primitive.empty())
            push_unique_token(out.missing, "primitive:" + policy.required_primitive);
    }
    return out;
}

// Convert Execute_µ result to JsonPtr response. Returns nullptr on Domain::None
// (no pattern match) so the handler chain falls through to the next handler.
static util::JsonPtr grammar_result_to_json(const ExecuteResult& r, const std::string& id) {
    if (r.domain == ExecuteResult::Domain::None)
        return nullptr;

    util::JsonObject obj;
    obj["@kind"]        = util::json_string("khanary.grammar.execute.v1");
    obj["micronaut_id"] = util::json_string(id);
    obj["matched"]      = util::json_bool(r.matched());

    switch (r.domain) {
    case ExecuteResult::Domain::String:
        obj["domain"] = util::json_string("string");
        obj["text"]   = util::json_string(r.text);
        break;
    case ExecuteResult::Domain::Graph:
        obj["domain"]    = util::json_string("graph");
        obj["graph_ref"] = util::json_string(r.text);
        break;
    case ExecuteResult::Domain::Vector:
        obj["domain"] = util::json_string("vector");
        obj["text"]   = util::json_string(r.text);
        break;
    case ExecuteResult::Domain::Error:
        obj["domain"] = util::json_string("error");
        obj["error"]  = util::json_string(r.error);
        break;
    default: break;
    }

    if (r.captures) {
        util::JsonObject caps;
        for (const auto& [k, v] : r.captures->bindings)
            caps[k] = util::json_string(v);
        obj["captures"] = util::json_object(std::move(caps));
    }

    if (!r.next_address.micronaut_id.empty())
        obj["next_address"] = util::json_string(r.next_address.micronaut_id);

    if (r.proposed_state)
        obj["proposed_state"] = r.proposed_state;

    return util::json_object(std::move(obj));
}

} // namespace

RequestRouter::RequestRouter(
    micronaut::MicronautRegistry& mu_reg,
    memory::MicronautScorer& scorer,
    const std::string& data_dir)
    : mu_reg_(mu_reg), scorer_(scorer)
{
    phase_driver_weights_.fill(1.0f);
    phase_wsd_weights_.fill(1.0f);
    phase_function_channel_weights_.fill(1.0f);
    phase_tool_channel_weights_.fill(1.0f);
    phase_miss_dispatch_weights_.fill(1.0f);
    load_phase_driver_weights(data_dir);
    load_track_gram_pools(data_dir);

    // ── Built-in Graph domain executor ─────────────────────────────────────────
    // Registered for TRAVERSE / RELATE / EXPAND / NEIGHBOR verbs.
    // Resolves the input_capture binding as a node id and walks the relational
    // graph via mu_reg_.relational_dispatch().
    GrammarFn graph_exec = [this](const ExecuteInput& input,
                                   const MatchCaptures& caps,
                                   const MicronautGrammarDecl& decl) -> ExecuteResult {
        ExecuteResult r;
        r.captures = caps;

        std::string node_id;
        if (!decl.input_capture.empty()) {
            auto it = caps.bindings.find(decl.input_capture);
            node_id = (it != caps.bindings.end()) ? it->second : input.text;
        } else {
            node_id = input.text;
        }

        util::JsonArray neighbors_arr;
        for (const char* verb : {"@delegates_to", "@is_a", "@supports", "@relates_to"}) {
            for (const auto& n : mu_reg_.relational_dispatch(node_id, verb)) {
                util::JsonObject nb;
                nb["id"]   = util::json_string(n);
                nb["verb"] = util::json_string(verb);
                neighbors_arr.push_back(util::json_object(std::move(nb)));
            }
        }

        r.domain = ExecuteResult::Domain::Graph;
        r.text   = node_id;

        util::JsonObject state;
        state["source_node"]     = util::json_string(node_id);
        state["graph_neighbors"] = util::json_array(std::move(neighbors_arr));
        r.proposed_state = util::json_object(std::move(state));

        return r;
    };
    for (const char* verb : {"TRAVERSE", "RELATE", "EXPAND", "NEIGHBOR"})
        fn_registry_[verb] = graph_exec;

    // ── Built-in Vector domain executor ────────────────────────────────────────
    // Registered for RANK / SCORE / WCR / WEIGHT verbs.
    // Tokenizes the input_capture binding and runs dispatch_blend() for WCR ranking.
    GrammarFn vector_exec = [this](const ExecuteInput& input,
                                    const MatchCaptures& caps,
                                    const MicronautGrammarDecl& decl) -> ExecuteResult {
        ExecuteResult r;
        r.captures = caps;

        std::string query;
        if (!decl.input_capture.empty()) {
            auto it = caps.bindings.find(decl.input_capture);
            query = (it != caps.bindings.end()) ? it->second : input.text;
        } else {
            query = input.text;
        }

        const auto tokens = tokenize_query(query);
        const auto blend  = mu_reg_.dispatch_blend(kuhul::Phase::Pop, tokens, 0.05f, 4);

        util::JsonArray ranked;
        for (const auto& [id, wt] : blend) {
            util::JsonObject row;
            row["id"]     = util::json_string(id);
            row["weight"] = util::json_number(static_cast<double>(wt));
            ranked.push_back(util::json_object(std::move(row)));
        }

        r.domain = ExecuteResult::Domain::Vector;
        r.text   = blend.empty() ? "" : blend[0].first;

        if (!blend.empty())
            r.next_address.micronaut_id = blend[0].first;

        util::JsonObject state;
        state["query"]  = util::json_string(query);
        state["ranked"] = util::json_array(std::move(ranked));
        r.proposed_state = util::json_object(std::move(state));

        return r;
    };
    for (const char* verb : {"RANK", "SCORE", "WCR", "WEIGHT"})
        fn_registry_[verb] = vector_exec;
}

void RequestRouter::register_grammar_fn(const std::string& verb, GrammarFn fn) {
    fn_registry_[verb] = std::move(fn);
}

void RequestRouter::register_grammar(
    const MicronautGrammarDecl& decl,
    const std::unordered_map<std::string, GrammarFn>& fn_overrides)
{
    // Compile → MicronautEntry; idempotent (guard with has)
    auto entry = MicronautGrammarCompiler::compile(decl);
    if (!mu_reg_.has(decl.id))
        mu_reg_.add(std::move(entry));

    grammar_registry_[decl.id]      = decl;
    grammar_fn_overrides_[decl.id]  = fn_overrides;

    // Install a ResponseFn that dispatches through Execute_µ.
    // fn_registry_ and grammar_fn_overrides_ are merged at call time so that
    // register_grammar_fn() calls made after registration take effect.
    handlers_[decl.id] = [this, id = decl.id](const std::string&,
                                               const RouteRequest& req) -> util::JsonPtr {
        auto git = grammar_registry_.find(id);
        if (git == grammar_registry_.end()) return nullptr;
        const auto& d = git->second;

        // Merge: global fn_registry_ as base, per-grammar overrides on top
        auto merged = fn_registry_;
        if (auto oit = grammar_fn_overrides_.find(id); oit != grammar_fn_overrides_.end())
            for (const auto& [verb, fn] : oit->second)
                merged[verb] = fn;

        ExecuteInput ei;
        ei.text  = req.query;
        ei.state = req.payload;

        auto res = MicronautGrammarCompiler::execute(ei, d, merged);
        return grammar_result_to_json(res, id);
    };
}

void RequestRouter::register_handler(const std::string& id, ResponseFn fn) {
    handlers_[id] = std::move(fn);
}

void RequestRouter::set_default_handler(ResponseFn fn) {
    default_handler_ = std::move(fn);
}

std::vector<std::string> RequestRouter::tokenize_query(const std::string& query) const {
    std::vector<std::string> tokens;
    std::istringstream ss(query);
    std::string tok;
    while (ss >> tok) {
        std::string lc;
        for (char c : tok) lc += (char)tolower((unsigned char)c);
        tokens.push_back(lc);
    }
    return tokens;
}

float RequestRouter::phase_driver_weight(kuhul::Phase phase) const {
    const size_t idx = static_cast<size_t>(phase);
    if (idx >= phase_driver_weights_.size()) return 1.0f;
    return std::clamp(phase_driver_weights_[idx], 0.0f, 1.0f);
}

float RequestRouter::phase_wsd_weight(kuhul::Phase phase) const {
    const size_t idx = static_cast<size_t>(phase);
    if (idx >= phase_wsd_weights_.size()) return 1.0f;
    return std::clamp(phase_wsd_weights_[idx], 0.0f, 2.0f);
}

float RequestRouter::phase_function_channel_weight(kuhul::Phase phase) const {
    const size_t idx = static_cast<size_t>(phase);
    if (idx >= phase_function_channel_weights_.size()) return 1.0f;
    return std::clamp(phase_function_channel_weights_[idx], 0.0f, 2.0f);
}

float RequestRouter::phase_tool_channel_weight(kuhul::Phase phase) const {
    const size_t idx = static_cast<size_t>(phase);
    if (idx >= phase_tool_channel_weights_.size()) return 1.0f;
    return std::clamp(phase_tool_channel_weights_[idx], 0.0f, 2.0f);
}

float RequestRouter::phase_miss_dispatch_weight(kuhul::Phase phase) const {
    const size_t idx = static_cast<size_t>(phase);
    if (idx >= phase_miss_dispatch_weights_.size()) return 1.0f;
    return std::clamp(phase_miss_dispatch_weights_[idx], 0.0f, 2.0f);
}

void RequestRouter::load_phase_driver_weights(const std::string& data_dir) {
    namespace fs = std::filesystem;
    const std::string path = data_dir + "/schema/fold.runtime.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) return;

    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    util::JsonPtr root;
    try {
        root = util::parse(text);
    } catch (...) {
        root.reset();
    }
    if (!root || !root->is_object()) return;

    const util::JsonValue* weights = root->get("phase_driver_weights");
    if (weights && weights->is_object()) {
        for (const auto& [k, v] : weights->as_object()) {
            if (!v || !v->is_number()) continue;
            kuhul::Phase p{};
            if (!kuhul::phase_from_name(k, p)) continue;
            const size_t idx = static_cast<size_t>(p);
            if (idx >= phase_driver_weights_.size()) continue;
            phase_driver_weights_[idx] = std::clamp(static_cast<float>(v->as_number()), 0.0f, 1.0f);
        }
    }

    const auto load_contract_file = [&](kuhul::Phase expected_phase, const std::string& raw_path) {
        const size_t idx = static_cast<size_t>(expected_phase);
        if (idx >= phase_driver_weights_.size()) return;

        fs::path contract_path(raw_path);
        if (!contract_path.is_absolute())
            contract_path = fs::path(data_dir).parent_path() / contract_path;
        contract_path = contract_path.lexically_normal();

        std::ifstream pf(contract_path, std::ios::binary);
        if (!pf) return;
        std::string ptext((std::istreambuf_iterator<char>(pf)),
                          std::istreambuf_iterator<char>());
        util::JsonPtr pj;
        try {
            pj = util::parse(ptext);
        } catch (...) {
            pj.reset();
        }
        if (!pj || !pj->is_object()) return;

        if (auto phase_name = pj->get_string("phase")) {
            kuhul::Phase declared_phase{};
            if (!kuhul::phase_from_name(*phase_name, declared_phase)) return;
            if (declared_phase != expected_phase) return;
        }

        if (auto w = pj->get_number("driver_weight"))
            phase_driver_weights_[idx] = std::clamp(static_cast<float>(*w), 0.0f, 1.0f);

        const util::JsonValue* channels = pj->get("semantic_channels");
        if (!channels || !channels->is_object()) return;

        const auto apply_channel_weight = [&](const char* channel_name, std::array<float, 6>& target) {
            const util::JsonValue* channel = channels->get(channel_name);
            if (!channel || !channel->is_object()) return;
            auto w = channel->get_number("weight");
            if (!w) return;
            target[idx] = std::clamp(static_cast<float>(*w), 0.0f, 2.0f);
        };

        apply_channel_weight("word_sense_disambiguation", phase_wsd_weights_);
        apply_channel_weight("function_gram", phase_function_channel_weights_);
        apply_channel_weight("tool_gram", phase_tool_channel_weights_);
        apply_channel_weight("high_confidence_miss", phase_miss_dispatch_weights_);
    };

    const auto load_contract_entry = [&](kuhul::Phase expected_phase, const util::JsonValue* entry) {
        if (!entry) return;
        if (entry->is_string()) {
            load_contract_file(expected_phase, entry->as_string());
            return;
        }
        if (entry->is_array()) {
            for (const auto& item : entry->as_array()) {
                if (!item || !item->is_string()) continue;
                load_contract_file(expected_phase, item->as_string());
            }
            return;
        }
        if (entry->is_object()) {
            if (auto path_s = entry->get_string("path"))
                load_contract_file(expected_phase, *path_s);
            const util::JsonValue* paths = entry->get("paths");
            if (paths && paths->is_array()) {
                for (const auto& p : paths->as_array()) {
                    if (!p || !p->is_string()) continue;
                    load_contract_file(expected_phase, p->as_string());
                }
            }
        }
    };

    const auto load_contract_map = [&](const util::JsonValue* map_root) {
        if (!map_root || !map_root->is_object()) return;
        for (const auto& [phase_name, phase_entry] : map_root->as_object()) {
            kuhul::Phase phase{};
            if (!kuhul::phase_from_name(phase_name, phase)) continue;
            load_contract_entry(phase, phase_entry.get());
        }
    };

    const util::JsonValue* base_contracts = root->get("phase_driver_contracts");
    const util::JsonValue* selected_profile_contracts = nullptr;
    std::string selected_profile = "default";
    if (auto profile = root->get_string("phase_driver_profile"))
        selected_profile = *profile;

    const util::JsonValue* profiles = root->get("phase_driver_contract_profiles");
    if (profiles && profiles->is_object()) {
        if (const util::JsonValue* p = profiles->get(selected_profile); p && p->is_object()) {
            selected_profile_contracts = p;
        } else if (const util::JsonValue* p_default = profiles->get("default");
                   p_default && p_default->is_object()) {
            selected_profile_contracts = p_default;
        }
    }

    load_contract_map(base_contracts);
    if (selected_profile_contracts && selected_profile_contracts != base_contracts)
        load_contract_map(selected_profile_contracts);
}

void RequestRouter::load_track_gram_pools(const std::string& data_dir) {
    namespace fs = std::filesystem;
    const fs::path registry_path =
        (fs::path(data_dir) / "mx2lm" / "gram_pool_registry.json").lexically_normal();
    std::ifstream rf(registry_path, std::ios::binary);
    if (!rf) return;
    const std::string rtext((std::istreambuf_iterator<char>(rf)),
                            std::istreambuf_iterator<char>());
    util::JsonPtr reg;
    try { reg = util::parse(rtext); } catch (...) { return; }
    if (!reg || !reg->is_object()) return;

    const auto* tracks = reg->get("tracks");
    if (!tracks || !tracks->is_object()) return;

    for (const auto& [track_name, track_val] : tracks->as_object()) {
        if (!track_val || !track_val->is_object()) continue;
        const auto* artifacts = track_val->get("artifacts");
        if (!artifacts || !artifacts->is_object()) continue;
        const auto gc_str = artifacts->get_string("gram_container");
        if (!gc_str || gc_str->empty()) continue;

        fs::path gc_path(*gc_str);
        if (!gc_path.is_absolute())
            gc_path = (fs::path(data_dir).parent_path() / gc_path).lexically_normal();

        std::ifstream gf(gc_path, std::ios::binary);
        if (!gf) continue;
        const std::string gtext((std::istreambuf_iterator<char>(gf)),
                                std::istreambuf_iterator<char>());
        util::JsonPtr gc;
        try { gc = util::parse(gtext); } catch (...) { continue; }
        if (!gc || !gc->is_object()) continue;

        const auto* channels = gc->get("channels");
        if (!channels || !channels->is_object()) continue;

        TrackGramPool pool;
        const auto extract_channel = [&](const char* name,
                                         std::unordered_set<std::string>& dest) {
            const auto* ch = channels->get(name);
            if (!ch || !ch->is_array()) return;
            for (const auto& item : ch->as_array()) {
                if (!item || !item->is_object()) continue;
                if (const auto g = item->get_string("gram"); g && !g->empty())
                    dest.insert(*g);
            }
        };
        extract_channel("unigram",      pool.unigrams);
        extract_channel("coarse_gram",  pool.coarse_grams);
        extract_channel("function_gram", pool.function_grams);
        extract_channel("tool_gram",    pool.tool_grams);

        if (!pool.unigrams.empty() || !pool.tool_grams.empty())
            track_gram_pools_[track_name] = std::move(pool);
    }
}

RouteResult RequestRouter::route(const RouteRequest& req) {
    RouteResult result;
    result.dispatched_phase = req.phase;
    result.selected_task_op.clear();
    result.resolver_policy.clear();
    result.required_primitive.clear();
    result.required_capabilities.clear();
    result.eligible_resolvers.clear();
    result.ineligible_resolvers.clear();
    result.policy_filter_applied = false;
    result.capability_miss = false;
    result.selected_capable = false;
    result.policy_satisfied = true;
    result.xcfe_authorized = false;
    result.dispatch_allowed = false;
    result.dispatch_block_reason.clear();
    result.score = 0.0f;
    result.fold_weight = 0.0f;
    result.phase_driver_weight = 0.0f;
    result.semantic_score = 0.0f;
    result.static_weight = 0.0f;
    result.pattern_strength = 0.0f;
    result.confidence_score = 0.0f;
    result.health_score = 0.0f;
    result.relevance_score = 0.0f;
    result.bow_score = 0.0f;
    result.distributional_score = 0.0f;
    result.lexical_semantic_score = 0.0f;
    result.nsm_prime_score = 0.0f;
    result.function_word_score = 0.0f;
    result.content_word_ratio = 0.0f;
    result.function_word_ratio = 0.0f;
    result.content_function_binding_ratio = 0.0f;
    result.word_sense_score = 0.0f;
    result.compositional_phrase_score = 0.0f;
    result.coarse_ngram_score = 0.0f;
    result.tool_gram_score = 0.0f;
    result.miss_dispatch_score = 0.0f;
    result.vector_relevance_score = 0.0f;
    result.wcr_attention_score = 0.0f;
    result.query_vector_x = 0.0f;
    result.query_vector_y = 0.0f;
    result.query_vector_z = 0.0f;
    result.key_vector_x = 0.0f;
    result.key_vector_y = 0.0f;
    result.key_vector_z = 0.0f;

    const auto policy_ctx = derive_resolver_policy_context(req);
    const bool policy_constrained = policy_ctx.constrained();
    result.selected_task_op = policy_ctx.op;
    result.resolver_policy = policy_ctx.resolver_policy;
    result.required_primitive = policy_ctx.required_primitive;
    result.required_capabilities = policy_ctx.required_capabilities;
    result.policy_filter_applied = policy_constrained;
    result.policy_satisfied = !policy_constrained;

    auto tokens = req.tokens.empty() ? tokenize_query(req.query) : req.tokens;
    const auto uri_call = parse_micronaut_uri_call(req.query);
    if (uri_call.valid) {
        tokens.push_back(uri_call.scheme);
        if (!uri_call.op.empty()) tokens.push_back(uri_call.op);
        for (const auto& seg : uri_call.segments)
            if (!seg.empty()) tokens.push_back(seg);
        if (result.selected_task_op.empty() && !uri_call.op.empty())
            result.selected_task_op = uri_call.op;
    }
    std::vector<std::string> query_raw_tokens = tokenize_distributional_text(req.query);
    if (query_raw_tokens.empty()) {
        for (const auto& qtok : tokens) {
            auto parts = tokenize_distributional_text(qtok);
            query_raw_tokens.insert(query_raw_tokens.end(), parts.begin(), parts.end());
        }
    }
    const std::vector<std::string> query_vector_fallback_tokens =
        query_raw_tokens.empty() ? tokens : query_raw_tokens;
    RouterVector3 query_vector{};

    const auto query_cf_profile = profile_content_function_tokens(query_raw_tokens);
    result.content_word_ratio = query_cf_profile.content_ratio;
    result.function_word_ratio = query_cf_profile.function_ratio;
    result.content_function_binding_ratio = query_cf_profile.function_binding_ratio;

    std::unordered_set<std::string> query_lexemes;
    std::unordered_set<std::string> query_function_words;
    for (const auto& qtok : query_raw_tokens) {
        if (qtok.size() < 2) continue;
        if (is_function_word(qtok)) {
            query_function_words.insert(qtok);
            continue;
        }
        query_lexemes.insert(canonical_lexeme(qtok));
    }
    if (query_lexemes.empty()) {
        for (const auto& qtok : query_raw_tokens) {
            if (qtok.size() < 2) continue;
            query_lexemes.insert(canonical_lexeme(qtok));
        }
    }
    const auto query_senses = resolve_query_senses(query_lexemes, query_raw_tokens);
    const auto query_compositional = extract_compositional_signatures(query_raw_tokens);
    const auto query_coarse_ngrams = extract_coarse_ngrams(query_raw_tokens);
    const auto query_tool_grams = extract_tool_grams(query_raw_tokens);

    auto query_nsm_primes = extract_nsm_primes(req.query);
    if (query_nsm_primes.empty()) {
        for (const auto& qtok : tokens) {
            auto part = extract_nsm_primes(qtok);
            query_nsm_primes.insert(part.begin(), part.end());
        }
    }

    // Distributional channel over track-text corpora:
    // similarity(query_context_distribution, track_context_distribution).
    struct DistDoc {
        std::unordered_map<std::string, float> tf;
        std::unordered_set<std::string> lexemes;
        std::unordered_set<std::string> nsm_primes;
        std::unordered_set<std::string> function_words;
        std::unordered_set<std::string> compositional_signatures;
        std::unordered_set<std::string> coarse_ngrams;
        std::unordered_set<std::string> tool_grams;
    };
    std::unordered_map<std::string, DistDoc> dist_docs;
    std::unordered_map<std::string, size_t> doc_freq;
    size_t active_doc_count = 0;

    auto add_text_tokens = [](std::unordered_map<std::string, float>& tf,
                              std::unordered_set<std::string>& lexemes,
                              std::unordered_set<std::string>& nsm_primes,
                              std::unordered_set<std::string>& function_words,
                              std::unordered_set<std::string>& compositional_signatures,
                              std::unordered_set<std::string>& coarse_ngrams,
                              std::unordered_set<std::string>& tool_grams,
                              const std::string& text) {
        const auto raw = tokenize_distributional_text(text);
        for (const auto& tok : raw) {
            if (tok.size() < 2) continue;
            if (is_function_word(tok)) {
                function_words.insert(tok);
                continue;
            }
            tf[tok] += 1.0f;
            lexemes.insert(canonical_lexeme(tok));
        }
        auto primes = extract_nsm_primes(text);
        nsm_primes.insert(primes.begin(), primes.end());
        auto comps = extract_compositional_signatures(raw);
        compositional_signatures.insert(comps.begin(), comps.end());
        auto coarse = extract_coarse_ngrams(raw);
        coarse_ngrams.insert(coarse.begin(), coarse.end());
        auto tool = extract_tool_grams(raw);
        tool_grams.insert(tool.begin(), tool.end());
    };

    for (const auto& id : mu_reg_.ids()) {
        const auto* e = mu_reg_.get(id);
        if (!e || !e->active) continue;
        ++active_doc_count;
        auto& tf = dist_docs[id].tf;
        auto& lex = dist_docs[id].lexemes;
        auto& nsm = dist_docs[id].nsm_primes;
        auto& fnw = dist_docs[id].function_words;
        auto& comp = dist_docs[id].compositional_signatures;
        auto& coarse = dist_docs[id].coarse_ngrams;
        auto& tool = dist_docs[id].tool_grams;
        add_text_tokens(tf, lex, nsm, fnw, comp, coarse, tool, e->id);
        add_text_tokens(tf, lex, nsm, fnw, comp, coarse, tool, e->description);
        add_text_tokens(tf, lex, nsm, fnw, comp, coarse, tool, e->category);
        for (const auto& g : e->governs) add_text_tokens(tf, lex, nsm, fnw, comp, coarse, tool, g);
        for (const auto& c : e->capabilities) add_text_tokens(tf, lex, nsm, fnw, comp, coarse, tool, c);

        std::unordered_set<std::string> uniq;
        for (const auto& [tok, count] : tf)
            if (count > 0.0f) uniq.insert(tok);
        for (const auto& tok : uniq) ++doc_freq[tok];
    }

    // METRIC-µ local pool enrichment: supplement structural gram channels from
    // the per-track pool before any mesh bridge scoring.
    // doc_freq is already computed; only non-TF channels are enriched here so
    // IDF weights are unaffected.
    for (auto& [id, doc] : dist_docs) {
        const auto* e = mu_reg_.get(id);
        if (!e) continue;
        const auto pit = track_gram_pools_.find(e->id);
        if (pit == track_gram_pools_.end()) continue;
        const auto& pool = pit->second;
        doc.coarse_ngrams.insert(pool.coarse_grams.begin(), pool.coarse_grams.end());
        doc.tool_grams.insert(pool.tool_grams.begin(), pool.tool_grams.end());
        doc.function_words.insert(pool.function_grams.begin(), pool.function_grams.end());
    }

    std::unordered_map<std::string, float> query_tf;
    for (const auto& qtok : query_raw_tokens) {
        if (qtok.size() < 2) continue;
        if (is_function_word(qtok)) continue;
        query_tf[qtok] += 1.0f;
    }
    if (query_tf.empty()) {
        for (const auto& qtok : query_raw_tokens) {
            if (qtok.size() < 2) continue;
            query_tf[qtok] += 1.0f;
        }
    }

    auto idf = [&](const std::string& tok) -> float {
        const auto it = doc_freq.find(tok);
        const double df = (it == doc_freq.end()) ? 0.0 : static_cast<double>(it->second);
        const double n = static_cast<double>(std::max<size_t>(active_doc_count, 1));
        return static_cast<float>(std::log((1.0 + n) / (1.0 + df)) + 1.0);
    };

    std::unordered_map<std::string, float> query_tfidf;
    float query_norm_sq = 0.0f;
    for (const auto& [tok, tf] : query_tf) {
        const float w = tf * idf(tok);
        query_tfidf[tok] = w;
        query_norm_sq += w * w;
    }
    const float query_norm = std::sqrt(std::max(0.0f, query_norm_sq));

    // LBS soft dispatch as primary pattern-strength candidates.
    auto gravity_blend = mu_reg_.dispatch_blend(req.phase, tokens);

    struct Candidate {
        float pattern = 0.0f;
    };
    std::unordered_map<std::string, Candidate> candidates;
    for (const auto& [track_id, w] : gravity_blend)
        candidates[track_id].pattern = std::max(candidates[track_id].pattern, std::clamp(w, 0.0f, 1.0f));

    // Sparse adaptive mesh expansion (1-hop neighbors), bounded fan-out.
    constexpr size_t kMaxMeshNeighbors = 8;
    size_t mesh_added = 0;
    for (const auto& [track_id, w] : gravity_blend) {
        if (mesh_added >= kMaxMeshNeighbors) break;
        const auto* src = mu_reg_.get(track_id);
        if (!src) continue;
        for (const auto& rel : src->relations) {
            if (mesh_added >= kMaxMeshNeighbors) break;
            const auto* dst = mu_reg_.get(rel.target_id);
            if (!dst || !dst->active) continue;
            const float relay_gain = rel.directed ? 0.35f : 0.45f;
            const float rel_pattern = std::clamp(w * relay_gain, 0.02f, 1.0f);
            auto it = candidates.find(rel.target_id);
            if (it == candidates.end()) {
                candidates[rel.target_id] = Candidate{rel_pattern};
                ++mesh_added;
            } else {
                it->second.pattern = std::max(it->second.pattern, rel_pattern);
            }
        }
    }

    // Scheme URI fast-path (math://..., plan://..., tools://..., decision://..., etc):
    // explicit scheme-targeted calls become first-class candidate seeds, then scope
    // the field to callable endpoints so an explicit address does not decay into Ω
    // ambiguity among unrelated tracks.
    std::unordered_set<std::string> uri_matched_ids;
    if (uri_call.valid) {
        for (const auto& id : mu_reg_.ids()) {
            const auto* entry = mu_reg_.get(id);
            if (!entry || !entry->active) continue;
            if (!entry_supports_uri_call(*entry, uri_call)) continue;

            float uri_pattern = 0.95f;
            if (!entry->route_contract.tokens.empty()) {
                size_t token_hits = 0;
                for (const auto& t : entry->route_contract.tokens) {
                    for (const auto& q : query_raw_tokens) {
                        if (lower_ascii_copy(q) == lower_ascii_copy(t)) {
                            ++token_hits;
                            break;
                        }
                    }
                }
                if (token_hits > 0) uri_pattern = 1.0f;
            }
            candidates[id].pattern = std::max(candidates[id].pattern, uri_pattern);
            uri_matched_ids.insert(id);
        }

        if (!uri_matched_ids.empty()) {
            for (auto it = candidates.begin(); it != candidates.end();) {
                if (!uri_matched_ids.count(it->first)) it = candidates.erase(it);
                else ++it;
            }
        }
    }

    // Fallback candidates when no gravity/mesh hit.
    if (candidates.empty()) {
        for (const auto& id : mu_reg_.ids()) {
            const auto* e = mu_reg_.get(id);
            if (!e || !e->active) continue;
            const float arc = static_cast<float>(
                std::max(0.0, kuhul::arc_weight(e->primary_phase, req.phase)));
            if (arc <= 0.0f) continue;
            candidates[id].pattern = std::max(candidates[id].pattern, arc);
        }
    }

    // Yax policy-aware eligibility gate:
    // filter impossible resolvers before W-C-R ranking.
    if (policy_constrained) {
        std::unordered_map<std::string, Candidate> eligible;
        std::unordered_set<std::string> scope_ids;
        scope_ids.reserve(candidates.size() + 8);
        for (const auto& [track_id, _] : candidates)
            scope_ids.insert(track_id);
        for (const auto& id : mu_reg_.ids())
            scope_ids.insert(id);

        eligible.reserve(scope_ids.size());
        for (const auto& track_id : scope_ids) {
            const auto* entry = mu_reg_.get(track_id);
            if (!entry || !entry->active) continue;

            float pattern = 0.0f;
            if (auto it = candidates.find(track_id); it != candidates.end()) {
                pattern = std::clamp(it->second.pattern, 0.0f, 1.0f);
            } else {
                const float arc = static_cast<float>(
                    std::max(0.0, kuhul::arc_weight(entry->primary_phase, req.phase)));
                pattern = std::clamp(arc, 0.0f, 1.0f);
            }
            if (pattern <= 0.0f) continue;

            const auto check = evaluate_eligibility(*entry, policy_ctx);
            if (check.capable) {
                eligible[track_id] = Candidate{pattern};
                result.eligible_resolvers.push_back(track_id);
            } else {
                result.ineligible_resolvers.push_back(track_id);
            }
        }
        std::sort(result.eligible_resolvers.begin(), result.eligible_resolvers.end());
        std::sort(result.ineligible_resolvers.begin(), result.ineligible_resolvers.end());
        candidates.swap(eligible);

        if (candidates.empty()) {
            result.capability_miss = true;
            result.policy_satisfied = false;
            result.dispatch_block_reason = "MISS_capability";
        }
    }

    // Canonical query vector = semantic centroid over policy-admitted candidate key vectors.
    // Hash-based token vector remains a cold-start fallback when no semantic vectors resolved.
    {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        double sw = 0.0;
        for (const auto& [track_id, cand] : candidates) {
            const auto* entry = mu_reg_.get(track_id);
            if (!entry || !entry->active) continue;
            const auto kv = key_vector_for_entry(*entry);
            if (!kv.valid) continue;
            const double w = std::clamp(static_cast<double>(cand.pattern), 0.0, 1.0);
            if (!(w > 1e-9)) continue;
            sx += kv.x * w;
            sy += kv.y * w;
            sz += kv.z * w;
            sw += w;
        }

        if (sw > 1e-9) {
            const double nx = sx / sw;
            const double ny = sy / sw;
            const double nz = sz / sw;
            const double n = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (n > 1e-9) {
                query_vector = {nx / n, ny / n, nz / n, true};
            }
        }
    }
    if (!query_vector.valid) {
        query_vector = derive_router_fallback_vector(query_vector_fallback_tokens);
    }
    if (!query_vector.valid) {
        query_vector = phase_unit_vector(req.phase);
    }
    result.query_vector_x = static_cast<float>(query_vector.x);
    result.query_vector_y = static_cast<float>(query_vector.y);
    result.query_vector_z = static_cast<float>(query_vector.z);

    // Build visible blend list from candidate pattern channel (normalized).
    {
        std::vector<std::pair<std::string, float>> ranked;
        ranked.reserve(candidates.size());
        for (const auto& [id, c] : candidates)
            ranked.push_back({id, c.pattern});
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) {
                      if (std::fabs(a.second - b.second) > 1e-6f)
                          return a.second > b.second;
                      return a.first < b.first;
                  });
        if (ranked.size() > 8) ranked.resize(8);
        float sum = 0.0f;
        for (const auto& kv : ranked) sum += kv.second;
        const float inv = (sum > 1e-6f) ? (1.0f / sum) : 0.0f;
        result.blend.reserve(ranked.size());
        for (const auto& [id, w] : ranked)
            result.blend.push_back({id, w * inv});
    }

    auto bow_relevance = [&](const micronaut::MicronautEntry& entry) -> float {
        if (query_lexemes.empty()) return 0.5f;
        if (entry.governs.empty()) return 0.5f;
        std::unordered_set<std::string> govern_lexemes;
        for (const auto& g : entry.governs) {
            for (const auto& tok : tokenize_distributional_text(g)) {
                if (tok.size() < 2) continue;
                if (is_function_word(tok)) continue;
                govern_lexemes.insert(canonical_lexeme(tok));
            }
        }
        if (govern_lexemes.empty()) return 0.05f;
        size_t hits = 0;
        for (const auto& qlex : query_lexemes)
            if (govern_lexemes.count(qlex)) ++hits;
        if (hits == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(hits) / static_cast<float>(query_lexemes.size()),
            0.05f, 1.0f);
    };

    auto distributional_similarity = [&](const std::string& track_id) -> float {
        if (query_norm <= 1e-6f) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.tf.empty()) return 0.05f;
        const auto& tf = it_doc->second.tf;
        float dot = 0.0f;
        float doc_norm_sq = 0.0f;
        for (const auto& [tok, tfv] : tf) {
            const float w = tfv * idf(tok);
            doc_norm_sq += w * w;
            auto it_q = query_tfidf.find(tok);
            if (it_q != query_tfidf.end()) dot += it_q->second * w;
        }
        const float doc_norm = std::sqrt(std::max(0.0f, doc_norm_sq));
        if (doc_norm <= 1e-6f) return 0.05f;
        return std::clamp(dot / (query_norm * doc_norm), 0.0f, 1.0f);
    };

    auto lexical_similarity = [&](const std::string& track_id) -> float {
        if (query_lexemes.empty()) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.lexemes.empty()) return 0.05f;
        const auto& dlex = it_doc->second.lexemes;
        size_t intersection = 0;
        for (const auto& qlex : query_lexemes)
            if (dlex.count(qlex)) ++intersection;
        if (intersection == 0) return 0.05f;
        const size_t uni = query_lexemes.size() + dlex.size() - intersection;
        if (uni == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(intersection) / static_cast<float>(uni),
            0.05f, 1.0f);
    };

    auto nsm_similarity = [&](const std::string& track_id) -> float {
        if (query_nsm_primes.empty()) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.nsm_primes.empty()) return 0.05f;
        const auto& dpr = it_doc->second.nsm_primes;
        size_t intersection = 0;
        for (const auto& qpr : query_nsm_primes)
            if (dpr.count(qpr)) ++intersection;
        if (intersection == 0) return 0.05f;
        const size_t uni = query_nsm_primes.size() + dpr.size() - intersection;
        if (uni == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(intersection) / static_cast<float>(uni),
            0.05f, 1.0f);
    };

    auto function_similarity = [&](const std::string& track_id) -> float {
        if (query_function_words.empty()) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.function_words.empty()) return 0.05f;
        const auto& dfn = it_doc->second.function_words;
        size_t intersection = 0;
        for (const auto& qfn : query_function_words)
            if (dfn.count(qfn)) ++intersection;
        if (intersection == 0) return 0.05f;
        const size_t uni = query_function_words.size() + dfn.size() - intersection;
        if (uni == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(intersection) / static_cast<float>(uni),
            0.05f, 1.0f);
    };

    auto word_sense_similarity = [&](const std::string& track_id) -> float {
        if (query_senses.empty()) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.lexemes.empty()) return 0.05f;
        const auto& dlex = it_doc->second.lexemes;

        float accum = 0.0f;
        size_t used = 0;
        for (const auto& sd : query_senses) {
            if (sd.cues.empty()) continue;
            size_t cue_hits = 0;
            for (const auto& cue : sd.cues)
                if (dlex.count(cue)) ++cue_hits;
            const float cue_cov = static_cast<float>(cue_hits) / static_cast<float>(sd.cues.size());
            const float term_hit = dlex.count(sd.term) ? 0.20f : 0.0f;
            const float local = std::clamp(
                0.60f * cue_cov + term_hit + 0.20f * sd.confidence,
                0.05f, 1.0f);
            accum += local;
            ++used;
        }
        if (used == 0) return 0.05f;
        return std::clamp(accum / static_cast<float>(used), 0.05f, 1.0f);
    };

    auto compositional_similarity = [&](const std::string& track_id) -> float {
        if (query_compositional.empty()) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.compositional_signatures.empty()) return 0.05f;
        const auto& ds = it_doc->second.compositional_signatures;
        size_t intersection = 0;
        for (const auto& qsig : query_compositional)
            if (ds.count(qsig)) ++intersection;
        if (intersection == 0) return 0.05f;
        const size_t uni = query_compositional.size() + ds.size() - intersection;
        if (uni == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(intersection) / static_cast<float>(uni),
            0.05f, 1.0f);
    };

    auto coarse_ngram_similarity = [&](const std::string& track_id) -> float {
        if (query_coarse_ngrams.empty()) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.coarse_ngrams.empty()) return 0.05f;
        const auto& dg = it_doc->second.coarse_ngrams;
        size_t intersection = 0;
        for (const auto& qg : query_coarse_ngrams)
            if (dg.count(qg)) ++intersection;
        if (intersection == 0) return 0.05f;
        const size_t uni = query_coarse_ngrams.size() + dg.size() - intersection;
        if (uni == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(intersection) / static_cast<float>(uni),
            0.05f, 1.0f);
    };

    auto tool_gram_similarity = [&](const std::string& track_id) -> float {
        if (query_tool_grams.empty()) return 0.5f;
        auto it_doc = dist_docs.find(track_id);
        if (it_doc == dist_docs.end() || it_doc->second.tool_grams.empty()) return 0.05f;
        const auto& tg = it_doc->second.tool_grams;
        size_t intersection = 0;
        for (const auto& qg : query_tool_grams)
            if (tg.count(qg)) ++intersection;
        if (intersection == 0) return 0.05f;
        const size_t uni = query_tool_grams.size() + tg.size() - intersection;
        if (uni == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(intersection) / static_cast<float>(uni),
            0.05f, 1.0f);
    };

    // Sek high_confidence_miss channel: capability-based affinity for MISS-routing dispatch.
    // Checks track capabilities for dispatch/research/acquisition keywords.
    // Phase-neutral score; phase_miss_dispatch_weight(Sek)=1.15 provides the phase lift.
    auto miss_dispatch_affinity = [&](const std::string& track_id) -> float {
        const auto* entry = mu_reg_.get(track_id);
        if (!entry || entry->capabilities.empty()) return 0.05f;
        static const std::unordered_set<std::string> kMissKws = {
            "dispatch", "search", "research", "fetch", "acquire",
            "web", "rag", "external", "miss", "fact", "retrieve", "recall"
        };
        size_t hits = 0;
        for (const auto& cap : entry->capabilities) {
            std::string lc = cap;
            std::transform(lc.begin(), lc.end(), lc.begin(),
                           [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            for (const auto& kw : kMissKws) {
                if (lc.find(kw) != std::string::npos) { ++hits; break; }
            }
        }
        if (hits == 0) return 0.05f;
        return std::clamp(
            static_cast<float>(hits) /
            static_cast<float>(entry->capabilities.size()),
            0.05f, 1.0f);
    };

    auto track_phase_fold_weight = [&](const micronaut::MicronautEntry& entry) -> float {
        // Fold declarations are capability manifests. Route selection can start from any phase;
        // estimate compatibility by projecting declared fold phases onto the requested phase.
        if (entry.folds.empty()) {
            const float fallback = static_cast<float>(
                std::max(0.0, kuhul::arc_weight(entry.primary_phase, req.phase)));
            return std::clamp(fallback, 0.0f, 1.0f);
        }

        float best = 0.0f;
        for (const auto& fold : entry.folds) {
            const float w = std::clamp(fold.weight, 0.0f, 1.0f);
            const float phase_arc = static_cast<float>(
                std::max(0.0, kuhul::arc_weight(fold.phase, req.phase)));
            best = std::max(best, std::clamp(w * phase_arc, 0.0f, 1.0f));
        }
        return std::clamp(best, 0.0f, 1.0f);
    };

    constexpr float kEps = 1e-6f;
    bool picked = false;
    std::vector<double> eligible_wcr;  // W·C·R per eligible candidate (entropy input)
    for (const auto& [track_id, cand] : candidates) {
        const auto* entry = mu_reg_.get(track_id);
        if (!entry || !entry->active) continue;
        auto sc = scorer_.score(track_id);
        const float static_w = std::clamp(entry->static_weight, 0.0f, 1.0f);
        const float pattern = std::clamp(cand.pattern, 0.0f, 1.0f);
        const float fold_w = track_phase_fold_weight(*entry);
        if (fold_w <= 0.0f) continue;
        const float driver_w = phase_driver_weight(req.phase);
        const float conf    = std::clamp(sc.c_mu, 0.0f, 1.0f);
        const float health  = std::clamp(entry->health, 0.0f, 1.0f);
        const auto key_vector = key_vector_for_entry(*entry);
        const float vector_rel = static_cast<float>(router_cosine_relevance(query_vector, key_vector));
        const float vector_gate = 0.75f + 0.25f * vector_rel;
        const float wcr_gate = static_cast<float>(router_wcr_score(static_w, conf, vector_rel));
        eligible_wcr.push_back(static_cast<double>(wcr_gate));
        const float bow_rel = bow_relevance(*entry);
        const float dist_rel = distributional_similarity(track_id);
        const float lex_rel = lexical_similarity(track_id);
        const float nsm_rel = nsm_similarity(track_id);
        const float fn_rel = std::clamp(
            function_similarity(track_id) * phase_function_channel_weight(req.phase),
            0.0f, 1.0f);
        const float wsd_rel = std::clamp(
            word_sense_similarity(track_id) * phase_wsd_weight(req.phase),
            0.0f, 1.0f);
        const float comp_rel = compositional_similarity(track_id);
        const float coarse_rel = coarse_ngram_similarity(track_id);
        const float tool_rel = std::clamp(
            tool_gram_similarity(track_id) * phase_tool_channel_weight(req.phase),
            0.0f, 1.0f);
        const float miss_rel = std::clamp(
            miss_dispatch_affinity(track_id) * phase_miss_dispatch_weight(req.phase),
            0.0f, 1.0f);
        const float rel     = std::clamp(
            0.22f * std::clamp(sc.r_mu, 0.0f, 1.0f) +
            0.11f * bow_rel +
            0.14f * dist_rel +
            0.11f * lex_rel +
            0.09f * nsm_rel +
            0.07f * fn_rel +
            0.08f * wsd_rel +
            0.08f * comp_rel +
            0.04f * coarse_rel +
            0.06f * tool_rel +
            0.05f * miss_rel,
            0.0f, 1.0f);
        const float total   = static_w * pattern * fold_w * driver_w * conf * health * rel * vector_gate;

        if (!picked ||
            (total > result.score + kEps) ||
            (std::fabs(total - result.score) <= kEps && pattern > result.pattern_strength + kEps) ||
            (std::fabs(total - result.score) <= kEps &&
             std::fabs(pattern - result.pattern_strength) <= kEps &&
             sc.total > result.semantic_score + kEps)) {
            picked = true;
            result.micronaut_id = track_id;
            result.dispatched_phase = entry->primary_phase;
            result.score = total;
            result.fold_weight = fold_w;
            result.phase_driver_weight = driver_w;
            result.pattern_strength = pattern;
            result.static_weight = static_w;
            result.confidence_score = conf;
            result.health_score = health;
            result.relevance_score = rel;
            result.bow_score = bow_rel;
            result.distributional_score = dist_rel;
            result.lexical_semantic_score = lex_rel;
            result.nsm_prime_score = nsm_rel;
            result.function_word_score = fn_rel;
            result.word_sense_score = wsd_rel;
            result.compositional_phrase_score = comp_rel;
            result.coarse_ngram_score = coarse_rel;
            result.tool_gram_score = tool_rel;
            result.miss_dispatch_score = miss_rel;
            result.vector_relevance_score = vector_rel;
            result.wcr_attention_score = wcr_gate;
            result.key_vector_x = static_cast<float>(key_vector.x);
            result.key_vector_y = static_cast<float>(key_vector.y);
            result.key_vector_z = static_cast<float>(key_vector.z);
            result.semantic_score = std::max(0.0f, sc.total);
        }
    }

    // Routing entropy over the eligible W·C·R distribution. Measurement only:
    // computed after eligibility and ranking, and never consulted by either.
    {
        const auto entropy = routing_entropy(eligible_wcr);
        result.eligible_count = entropy.eligible_count;
        result.routing_entropy_bits = entropy.bits;
        result.routing_entropy_normalized = entropy.normalized;
        result.omega_throttled = entropy.omega_throttled;
        const auto field = wcr_field_stats(eligible_wcr);
        result.wcr_top1 = field.top1;
        result.wcr_top2 = field.top2;
        result.wcr_margin = field.margin;
    }

    // Last-resort deterministic fallback when candidate map was empty or filtered out.
    // Under policy constraints, fallback is limited to policy-eligible candidates only.
    if (!picked && !result.capability_miss) {
        std::vector<std::string> ids;
        if (policy_constrained) {
            ids.reserve(candidates.size());
            for (const auto& [track_id, _] : candidates) ids.push_back(track_id);
            std::sort(ids.begin(), ids.end());
        } else {
            ids = mu_reg_.ids();
        }
        result.micronaut_id = scorer_.best(ids, 0.0f);
        if (!result.micronaut_id.empty()) {
            if (const auto* entry = mu_reg_.get(result.micronaut_id)) {
                result.dispatched_phase = entry->primary_phase;
                auto sc = scorer_.score(result.micronaut_id);
                const float static_w = std::clamp(entry->static_weight, 0.0f, 1.0f);
                const float fold_w = track_phase_fold_weight(*entry);
                const float driver_w = phase_driver_weight(req.phase);
                const float conf = std::clamp(sc.c_mu, 0.0f, 1.0f);
                const float health = std::clamp(entry->health, 0.0f, 1.0f);
                const auto key_vector = key_vector_for_entry(*entry);
                const float vector_rel = static_cast<float>(router_cosine_relevance(query_vector, key_vector));
                const float vector_gate = 0.75f + 0.25f * vector_rel;
                const float wcr_gate = static_cast<float>(router_wcr_score(static_w, conf, vector_rel));
                const float bow_rel = bow_relevance(*entry);
                const float dist_rel = distributional_similarity(result.micronaut_id);
                const float lex_rel = lexical_similarity(result.micronaut_id);
                const float nsm_rel = nsm_similarity(result.micronaut_id);
                const float fn_rel = std::clamp(
                    function_similarity(result.micronaut_id) * phase_function_channel_weight(req.phase),
                    0.0f, 1.0f);
                const float wsd_rel = std::clamp(
                    word_sense_similarity(result.micronaut_id) * phase_wsd_weight(req.phase),
                    0.0f, 1.0f);
                const float comp_rel = compositional_similarity(result.micronaut_id);
                const float coarse_rel = coarse_ngram_similarity(result.micronaut_id);
                const float tool_rel = std::clamp(
                    tool_gram_similarity(result.micronaut_id) * phase_tool_channel_weight(req.phase),
                    0.0f, 1.0f);
                const float miss_rel = std::clamp(
                    miss_dispatch_affinity(result.micronaut_id) * phase_miss_dispatch_weight(req.phase),
                    0.0f, 1.0f);
                const float rel = std::clamp(
                    0.22f * std::clamp(sc.r_mu, 0.0f, 1.0f) +
                    0.11f * bow_rel +
                    0.14f * dist_rel +
                    0.11f * lex_rel +
                    0.09f * nsm_rel +
                    0.07f * fn_rel +
                    0.08f * wsd_rel +
                    0.08f * comp_rel +
                    0.04f * coarse_rel +
                    0.06f * tool_rel +
                    0.05f * miss_rel,
                    0.0f, 1.0f);
                const float arc = static_cast<float>(
                    std::max(0.0, kuhul::arc_weight(entry->primary_phase, req.phase)));
                const float pattern = std::clamp(arc, 0.0f, 1.0f);
                result.fold_weight = fold_w;
                result.phase_driver_weight = driver_w;
                result.pattern_strength = pattern;
                result.static_weight = static_w;
                result.confidence_score = conf;
                result.health_score = health;
                result.relevance_score = rel;
                result.bow_score = bow_rel;
                result.distributional_score = dist_rel;
                result.lexical_semantic_score = lex_rel;
                result.nsm_prime_score = nsm_rel;
                result.function_word_score = fn_rel;
                result.word_sense_score = wsd_rel;
                result.compositional_phrase_score = comp_rel;
                result.coarse_ngram_score = coarse_rel;
                result.tool_gram_score = tool_rel;
                result.miss_dispatch_score = miss_rel;
                result.vector_relevance_score = vector_rel;
                result.wcr_attention_score = wcr_gate;
                result.key_vector_x = static_cast<float>(key_vector.x);
                result.key_vector_y = static_cast<float>(key_vector.y);
                result.key_vector_z = static_cast<float>(key_vector.z);
                result.semantic_score = std::max(0.0f, sc.total);
                result.score = static_w * pattern * fold_w * driver_w * conf * health * rel * vector_gate;
            }
        }
    }

    // Sek dispatch gate: revalidate capability/policy immediately before execution.
    // Policy constrains attention (eligibility) without becoming attention score.
    bool selected_capable = true;
    bool policy_satisfied = !policy_constrained;
    if (!result.micronaut_id.empty()) {
        if (const auto* entry = mu_reg_.get(result.micronaut_id)) {
            if (policy_constrained) {
                const auto check = evaluate_eligibility(*entry, policy_ctx);
                selected_capable = check.capable;
                policy_satisfied = check.capable;
            } else {
                selected_capable = true;
                policy_satisfied = true;
            }
        } else {
            selected_capable = false;
            policy_satisfied = false;
        }
    } else {
        selected_capable = !policy_constrained;
        policy_satisfied = !policy_constrained;
    }

    bool xcfe_authorized = false;
    try {
        xcfe_authorized = authorizer_ && authorizer_(req, result.micronaut_id);
    } catch (...) {
        xcfe_authorized = false; // Policy failures never dispatch.
    }
    if (req.payload && req.payload->is_object()) {
        for (const char* name : {"xcfe_authorized", "xcfe_authorize"}) {
            if (const auto* flag = req.payload->get(name))
                xcfe_authorized = xcfe_authorized && flag->is_bool() && flag->as_bool();
        }
    }

    result.selected_capable = selected_capable;
    result.policy_satisfied = policy_satisfied;
    result.xcfe_authorized = xcfe_authorized;
    result.observation_authority = result.xcfe_authorized && selected_capable && policy_satisfied;
    result.epistemic_state = result.observation_authority ? "SUPPORTED" : "UNKNOWN";

    // ── Ω gate: a still-flat eligible field refuses premature collapse ──────
    // Blocking observation only: entropy authorizes nothing here, and exhaustion
    // stays explicitly unresolved rather than falling through to a commit.
    {
        const RoutingEntropy entropy{result.eligible_count, result.routing_entropy_bits,
                                     result.routing_entropy_normalized, result.omega_throttled};
        // The budget key is the ambiguity episode, so every attempt at the same
        // routing question shares one identity and the retries are joinable.
        const std::string key = episode_id_for(canonical_routing_query(req.query));
        result.episode_id = key;
        const uint32_t prior = ambiguity_attempts_.count(key) ? ambiguity_attempts_[key] : 0;
        const auto gate = ambiguity_gate(entropy, prior);
        if (gate == AmbiguityGate::Allow) {
            // Preserve the episode's retry index: an allow after N blocks is the
            // observation that resolved at attempt N, which is what the ΔĤ / ΔM
            // joins need. Leaving it 0 would erase the retry index on success.
            result.ambiguity_attempt = prior;
            ambiguity_attempts_.erase(key);
        } else {
            result.omega_blocked = true;
            result.ambiguity_attempt = prior + 1;
            result.dispatch_block_reason = gate == AmbiguityGate::ExhaustedAmbiguity
                                               ? "MISS_ambiguity_exhausted"
                                               : "MISS_ambiguity";
            if (ambiguity_attempts_.size() >= kAmbiguityAttemptCap &&
                !ambiguity_attempts_.count(key))
                ambiguity_attempts_.erase(ambiguity_attempts_.begin());
            ambiguity_attempts_[key] = result.ambiguity_attempt;
        }
    }

    const bool has_dispatch_target = !result.micronaut_id.empty() || static_cast<bool>(default_handler_);
    result.dispatch_allowed = has_dispatch_target && selected_capable && policy_satisfied &&
                              xcfe_authorized && !result.omega_blocked;
    // Calibration tuple bucket: emitted for allowed events too, so the observed
    // distribution is not conditioned on being blocked.
    result.ambiguity_outcome =
        ambiguity_outcome_label(result.dispatch_block_reason, result.eligible_count);

    // ── GA phase-arc telemetry (Cl(2,0), measurement only — no routing authority) ──
    // Strictly post-decision: reads already-finalized result fields, never feeds back.
    // ga_phase_arc_scalar == arc_weight(req.phase→dispatched_phase) is the parity check.
    {
        using khanary::kuhul::statealgebra::cl20_phase_product;
        using khanary::kuhul::statealgebra::cl20_vector;
        using khanary::kuhul::statealgebra::cl20_outer_b;
        const int from_ph = static_cast<int>(req.phase);
        const int to_ph   = static_cast<int>(result.dispatched_phase);
        const auto arc    = cl20_phase_product(from_ph, to_ph);
        result.ga_phase_arc_scalar   = static_cast<float>(arc.s);
        result.ga_phase_arc_bivector = static_cast<float>(arc.b);
        const auto qv = cl20_vector(result.query_vector_x, result.query_vector_y);
        const auto kv = cl20_vector(result.key_vector_x,   result.key_vector_y);
        result.ga_orient_xy_query_key = static_cast<float>(cl20_outer_b(qv, kv));
    }

    // ── Episode observation (append-only, best effort) ──────────────────────
    // Persisted so H_Ω and A_max can be measured rather than guessed. The id is
    // pseudonymous (keyed digest, no raw query), logging failures never affect
    // routing, and the contract the observation was taken under is recorded.
    if (!calibration_path_.empty()) {
        util::JsonObject row;
        row["calibration_epoch"] = util::json_string(kCalibrationEpoch);
        row["lane"] = util::json_string(req.lane);
        row["episode_id"] = util::json_string(result.episode_id);
        row["ambiguity_attempt"] =
            util::json_number(static_cast<double>(result.ambiguity_attempt));
        row["eligible_count"] = util::json_number(static_cast<double>(result.eligible_count));
        row["routing_entropy_bits"] = util::json_number(result.routing_entropy_bits);
        row["routing_entropy_normalized"] =
            util::json_number(result.routing_entropy_normalized);
        row["wcr_top1"] = util::json_number(result.wcr_top1);
        row["wcr_top2"] = util::json_number(result.wcr_top2);
        row["wcr_margin"] = util::json_number(result.wcr_margin);
        row["omega_throttled"] = util::json_bool(result.omega_throttled);
        row["omega_blocked"] = util::json_bool(result.omega_blocked);
        row["dispatch_allowed"] = util::json_bool(result.dispatch_allowed);
        row["ambiguity_outcome"] = util::json_string(result.ambiguity_outcome);
        row["resolver_policy"] = util::json_string(result.resolver_policy);
        row["selected_task_op"] = util::json_string(result.selected_task_op);
        row["selected_micronaut"] = util::json_string(result.micronaut_id);
        row["entropy_authoritative"] = util::json_bool(false);
        row["h_omega"] = util::json_number(kOmegaEntropyThreshold);
        row["a_max"] = util::json_number(static_cast<double>(kAmbiguityMaxAttempts));
        row["ga_phase_arc_scalar"]    = util::json_number(result.ga_phase_arc_scalar);
        row["ga_phase_arc_bivector"]  = util::json_number(result.ga_phase_arc_bivector);
        row["ga_orient_xy_query_key"] = util::json_number(result.ga_orient_xy_query_key);
        try {
            std::lock_guard<std::mutex> lock(calibration_mutex_);
            const std::filesystem::path path(calibration_path_);
            if (!path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path());
            std::ofstream sink(calibration_path_, std::ios::app | std::ios::binary);
            if (sink) sink << util::stringify(util::json_object(std::move(row))) << "\n";
        } catch (...) {
            // Observation is diagnostic: it never affects the routing decision.
        }
    }

    if (!result.dispatch_allowed) {
        if (result.dispatch_block_reason.empty()) {
            if (!xcfe_authorized) result.dispatch_block_reason = "XCFE_DENIED";
            else if (policy_constrained && !selected_capable) result.dispatch_block_reason = "MISS_capability";
            else result.dispatch_block_reason = "NO_SELECTED_RESOLVER";
        }

        if (result.dispatch_block_reason == "MISS_capability") {
            result.epistemic_state = "MISS";
            result.observation_authority = false;
            result.epistemic_causes = {"local_miss", "insufficient_observability"};
        } else if (result.dispatch_block_reason == "MISS_ambiguity" ||
                   result.dispatch_block_reason == "MISS_ambiguity_exhausted") {
            result.epistemic_state = "UNKNOWN";
            result.observation_authority = false;
            result.epistemic_causes = {"local_miss", "insufficient_observability", "cross_track_evidence_not_checked"};
        } else if (result.dispatch_block_reason == "XCFE_DENIED") {
            result.epistemic_state = "UNKNOWN";
            result.observation_authority = false;
            result.epistemic_causes = {"host_policy_denied"};
        } else {
            result.epistemic_state = "UNKNOWN";
            result.observation_authority = false;
            result.epistemic_causes = {"dispatch_blocked"};
        }

        util::JsonObject blocked;
        blocked["ok"] = util::json_bool(false);
        const bool ambiguity = result.dispatch_block_reason == "MISS_ambiguity";
        const bool exhausted = result.dispatch_block_reason == "MISS_ambiguity_exhausted";
        blocked["status"] = util::json_string(
            result.dispatch_block_reason == "MISS_capability" ? "miss_capability"
            : ambiguity                                       ? "miss_ambiguity"
            : exhausted                                       ? "unresolved_ambiguity"
                                                              : "dispatch_blocked");
        blocked["epistemic_state"] = util::json_string(result.epistemic_state);
        blocked["observation_authority"] = util::json_bool(result.observation_authority);
        util::JsonArray causes;
        for (const auto& cause : result.epistemic_causes)
            causes.push_back(util::json_string(cause));
        blocked["epistemic_causes"] = util::json_array(std::move(causes));
        blocked["miss_kind"] = util::json_string(
            result.dispatch_block_reason == "MISS_capability" ? "capability"
            : ambiguity                                       ? "ambiguity"
            : exhausted                                       ? "ambiguity_exhausted"
                                                              : "dispatch");
        // Entropy is reported, never authoritative.
        blocked["eligible_count"] = util::json_number(static_cast<double>(result.eligible_count));
        blocked["routing_entropy_bits"] = util::json_number(result.routing_entropy_bits);
        blocked["routing_entropy_normalized"] =
            util::json_number(result.routing_entropy_normalized);
        blocked["omega_throttled"] = util::json_bool(result.omega_throttled);
        blocked["omega_blocked"] = util::json_bool(result.omega_blocked);
        blocked["ambiguity_attempt"] =
            util::json_number(static_cast<double>(result.ambiguity_attempt));
        blocked["entropy_authoritative"] = util::json_bool(false);
        blocked["wcr_top1"] = util::json_number(result.wcr_top1);
        blocked["wcr_top2"] = util::json_number(result.wcr_top2);
        blocked["wcr_margin"] = util::json_number(result.wcr_margin);
        blocked["ambiguity_outcome"] =
            util::json_string(ambiguity_outcome_label(result.dispatch_block_reason,
                                                      result.eligible_count));
        blocked["reason"] = util::json_string(result.dispatch_block_reason);
        blocked["selected_task_op"] = util::json_string(result.selected_task_op);
        blocked["resolver_policy"] = util::json_string(result.resolver_policy);
        blocked["required_primitive"] = util::json_string(result.required_primitive);
        blocked["selected_micronaut"] = util::json_string(result.micronaut_id);
        blocked["policy_satisfied"] = util::json_bool(result.policy_satisfied);
        blocked["selected_capable"] = util::json_bool(result.selected_capable);
        blocked["xcfe_authorized"] = util::json_bool(result.xcfe_authorized);

        util::JsonArray req_caps;
        for (const auto& cap : result.required_capabilities)
            req_caps.push_back(util::json_string(cap));
        blocked["required_capabilities"] = util::json_array(std::move(req_caps));

        util::JsonArray elig;
        for (const auto& id : result.eligible_resolvers)
            elig.push_back(util::json_string(id));
        blocked["eligible_resolvers"] = util::json_array(std::move(elig));

        util::JsonArray inelig;
        for (const auto& id : result.ineligible_resolvers)
            inelig.push_back(util::json_string(id));
        blocked["ineligible_resolvers"] = util::json_array(std::move(inelig));

        result.handler_id = "(blocked)";
        result.response = util::json_object(std::move(blocked));
        return result;
    }

    // Handler chain: walk blend in weight order, first registered handler wins
    util::JsonPtr response;
    for (const auto& [track_id, weight] : result.blend) {
        auto it = handlers_.find(track_id);
        if (it != handlers_.end()) {
            response = it->second(track_id, req);
            if (response) { result.handler_id = track_id; break; }
        }
    }
    // Fallback: primary track handler (covers scorer-selected track not in blend)
    if (!response && !result.micronaut_id.empty()) {
        auto it = handlers_.find(result.micronaut_id);
        if (it != handlers_.end()) {
            response = it->second(result.micronaut_id, req);
            if (response) result.handler_id = result.micronaut_id;
        }
    }
    if (!response && default_handler_) {
        response = default_handler_(result.micronaut_id, req);
        if (response) result.handler_id = "(default)";
    }
    if (!response)
        response = util::json_string("no handler registered");

    result.response = std::move(response);
    return result;
}

} // namespace khanary::io
