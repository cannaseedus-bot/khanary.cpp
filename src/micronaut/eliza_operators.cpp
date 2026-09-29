#include "eliza_operators.h"
#include "relation_isa.h"
#include "../kuhul/xcfe_node_class.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace khanary::micronaut {

namespace {

static std::string upper_ascii(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

static std::string lower_ascii(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

static std::string join_tokens(const std::vector<std::string>& toks, size_t start, size_t end_exclusive) {
    if (start >= end_exclusive || end_exclusive > toks.size()) return {};
    std::string out;
    for (size_t i = start; i < end_exclusive; ++i) {
        if (!out.empty()) out.push_back(' ');
        out += toks[i];
    }
    return out;
}

static std::unordered_set<std::string> ngram_set(
    const std::vector<std::string>& tokens,
    size_t n)
{
    std::unordered_set<std::string> out;
    if (n == 0 || tokens.size() < n) return out;
    for (size_t i = 0; i + n <= tokens.size(); ++i)
        out.insert(join_tokens(tokens, i, i + n));
    return out;
}

static std::string build_context(
    const std::vector<std::string>& tokens,
    const std::vector<std::string>& matched_cues)
{
    std::unordered_set<std::string> cue_words;
    for (const auto& cue : matched_cues) {
        for (const auto& part : split_words(cue))
            cue_words.insert(part);
    }
    std::string out;
    for (const auto& tok : tokens) {
        if (tok.empty()) continue;
        if (cue_words.count(tok)) continue;
        if (!out.empty()) out.push_back(' ');
        out += tok;
    }
    return out;
}

struct LocalVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool valid = false;
};

static LocalVec3 derive_vec(const std::vector<std::string>& tokens) {
    if (tokens.empty()) return {};
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    std::size_t kept = 0;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto tok = lower_ascii(tokens[i]);
        if (tok.empty()) continue;
        uint32_t h = 2166136261u;
        for (unsigned char c : tok) {
            h ^= static_cast<uint32_t>(c);
            h *= 16777619u;
        }
        sx += std::sin(static_cast<double>(h & 0x7ffu) * 0.009 + static_cast<double>(i) * 0.17);
        sy += std::cos(static_cast<double>((h >> 11) & 0x7ffu) * 0.011 + static_cast<double>(i) * 0.13);
        sz += std::sin(static_cast<double>((h >> 22) & 0x3ffu) * 0.013 + static_cast<double>(i) * 0.11);
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

static double cosine01(const LocalVec3& a, const LocalVec3& b) {
    if (!a.valid || !b.valid) return 0.5;
    const double dot = a.x * b.x + a.y * b.y + a.z * b.z;
    const double c = std::clamp(dot, -1.0, 1.0);
    return std::clamp(0.5 * (c + 1.0), 0.0, 1.0);
}

static double wcr(double w, double c, double r) {
    return std::clamp(std::clamp(w, 0.0, 1.0) *
                      std::clamp(c, 0.0, 1.0) *
                      std::clamp(r, 0.0, 1.0), 0.0, 1.0);
}

} // namespace

void ElizaOperatorBank::init() {
    name_index_.clear();
    ops_ = {
        { "CAUSE",       {"because","therefore","since","causes","result","due"},      "CAUSE[{context}] → effect" },
        { "CONTRAST",    {"but","however","although","yet","despite","whereas"},        "CONTRAST[{context}] ↔ opposition" },
        { "DEPENDENCY",  {"requires","depends","needs","prerequisite","if","then"},     "DEPEND[{context}] ← precondition" },
        { "CONSTRAINT",  {"must","cannot","forbidden","limit","restrict","bound"},      "CONSTRAIN[{context}] ∈ bounds" },
        { "EVIDENCE",    {"shows","proves","indicates","demonstrates","evidence","data"},"EVIDENCE[{context}] ⊢ claim" },
        { "SEQUENCE",    {"first","then","next","finally","after","before","step"},     "SEQ[{context}] → ordered" },
        { "BOUNDARY",    {"start","end","begin","stop","boundary","edge","limit"},      "BOUND[{context}] ∂domain" },
        { "STATE",       {"is","are","was","were","state","status","condition"},        "STATE[{context}] = value" },
        { "CONFLICT",    {"conflict","contradiction","inconsistent","oppose","versus"}, "CONFLICT[{context}] ⊗ resolution" },
        { "ALTERNATIVE", {"or","either","alternatively","instead","option","choice"},   "ALT[{context}] ∨ branch" },
        { "INVARIANT",   {"always","never","constant","invariant","unchanged","fixed"}, "INVAR[{context}] ∀ constant" },
        { "TEST",        {"test","check","verify","assert","validate","confirm"},       "TEST[{context}] ? assertion" },
        { "RELATION",    {"related","similar","like","connects","between","maps"},      "RELATE[{context}] ~ neighbor" },
        { "SUBSTITUTE",  {"replace","substitute","swap","instead","equivalent","equal"},"SUBST[{context}] ↦ replacement" },
        { "VERIFY",      {"true","false","correct","wrong","valid","invalid","confirm"},"VERIFY[{context}] ✓ truthvalue" },
        { "DIRECT",      {"go","run","execute","perform","do","apply","invoke"},        "DIRECT[{context}] → action" },
    };

    // Promoted into the XCFE Node ISA (src/kuhul/xcfe_node_class.h): operator ->
    // class now has a single definition, which also closes the three operators
    // that previously had no class (SUBSTITUTE -> TRANSFORM, BOUNDARY/RELATION ->
    // STRUCTURE).
    auto atom_for_operator = [](const std::string& upper_name) -> std::string {
        kuhul::XcfeNodeClass node_class{};
        if (!kuhul::xcfe_class_for_operator(upper_name, node_class)) return "";
        return kuhul::xcfe_node_class_name(node_class);
    };

    for (auto& op : ops_) {
        if (op.relation_hint.empty() || op.relation_hint == "@related_to")
            op.relation_hint = infer_relation_from_lexical(op.name);
        op.semantic_atom_hint = atom_for_operator(upper_ascii(op.name));
    }

    for (size_t i = 0; i < ops_.size(); ++i)
        name_index_[upper_ascii(ops_[i].name)] = i;
}

ElizaOperatorBank::ElizaOperatorBank() { init(); }

std::optional<OperatorMatch> ElizaOperatorBank::detect(
    const std::vector<std::string>& tokens) const
{
    auto ranked = detect_all(tokens, 0.0f, 1);
    if (ranked.empty()) return std::nullopt;
    return ranked.front();
}

std::vector<OperatorMatch> ElizaOperatorBank::detect_all(
    const std::vector<std::string>& tokens,
    float min_score,
    size_t top_k) const
{
    if (tokens.empty() || top_k == 0) return {};

    std::vector<std::string> normalized_tokens;
    normalized_tokens.reserve(tokens.size());
    for (const auto& tok : tokens) {
        const auto lower = lower_ascii(tok);
        if (!lower.empty()) normalized_tokens.push_back(lower);
    }
    if (normalized_tokens.empty()) return {};
    const auto query_vector = derive_vec(normalized_tokens);

    std::unordered_set<std::string> unigram(normalized_tokens.begin(), normalized_tokens.end());
    const auto bigrams = ngram_set(normalized_tokens, 2);
    const auto trigrams = ngram_set(normalized_tokens, 3);

    const float floor = std::isfinite(min_score) ? std::max(0.0f, min_score) : 0.3f;

    std::vector<OperatorMatch> out;
    for (const auto& op : ops_) {
        if (op.pop_governs.empty()) continue;
        float matched_count = 0.0f;
        std::vector<std::string> matched;

        for (const auto& raw_cue : op.pop_governs) {
            const std::string cue = lower_ascii(raw_cue);
            if (cue.empty()) continue;

            bool hit = false;
            if (cue.find(' ') != std::string::npos) {
                const size_t words = split_words(cue).size();
                if (words <= 2) hit = bigrams.count(cue) > 0;
                else if (words == 3) hit = trigrams.count(cue) > 0;
                else hit = false;
            } else {
                hit = unigram.count(cue) > 0;
            }
            if (hit) {
                matched_count += 1.0f;
                matched.push_back(cue);
            }
        }

        const float score = matched_count / static_cast<float>(op.pop_governs.size());
        if (score < floor || score <= 0.0f) continue;

        OperatorMatch m;
        m.op = &op;
        m.score = score;
        m.matched_cues = std::move(matched);
        m.context = build_context(normalized_tokens, m.matched_cues);
        m.canonical_relation = canonical_relation(op.relation_hint);
        if (m.canonical_relation.empty() || !is_supported_relation(m.canonical_relation))
            m.canonical_relation = infer_relation_from_lexical(op.name);
        m.relation_phase = relation_phase_hint(m.canonical_relation);
        m.semantic_atom = op.semantic_atom_hint;
        std::vector<std::string> key_tokens = split_words(lower_ascii(op.name));
        key_tokens.insert(key_tokens.end(), m.matched_cues.begin(), m.matched_cues.end());
        const auto key_vector = derive_vec(key_tokens);
        m.vector_relevance = static_cast<float>(cosine01(query_vector, key_vector));
        m.wcr_attention_score = static_cast<float>(wcr(1.0, score, m.vector_relevance));
        out.push_back(std::move(m));
    }

    std::sort(out.begin(), out.end(), [](const OperatorMatch& a, const OperatorMatch& b) {
        if (a.score != b.score) return a.score > b.score;
        const std::string an = a.op ? a.op->name : std::string{};
        const std::string bn = b.op ? b.op->name : std::string{};
        return an < bn;
    });
    if (out.size() > top_k) out.resize(top_k);
    return out;
}

std::string ElizaOperatorBank::emit(const ElizaOperator& op, const std::string& context) const {
    std::string tpl = op.sek_directive;
    size_t pos = tpl.find("{context}");
    if (pos != std::string::npos)
        tpl.replace(pos, 9, context.empty() ? "?" : context);
    return tpl;
}

const ElizaOperator* ElizaOperatorBank::find(const std::string& name) const {
    auto it = name_index_.find(upper_ascii(name));
    if (it == name_index_.end()) return nullptr;
    return &ops_[it->second];
}

} // namespace khanary::micronaut
