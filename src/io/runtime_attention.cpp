#include "runtime_attention.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace khanary::io {

namespace {

uint64_t fnv1a64(std::string_view text) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : text) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

SemanticVector3 normalized(double x, double y, double z) {
    const double n = std::sqrt(x * x + y * y + z * z);
    if (!(n > 1e-9)) return {};
    SemanticVector3 v;
    v.x = x / n;
    v.y = y / n;
    v.z = z / n;
    v.valid = true;
    return v;
}

bool contains_phrase(const std::string& text_lower, const std::string& phrase_lower) {
    return !phrase_lower.empty() && text_lower.find(phrase_lower) != std::string::npos;
}

double query_priority(const std::string& query_lower) {
    std::string low = query_lower;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });

    if (contains_phrase(low, "urgent") || contains_phrase(low, "asap") ||
        contains_phrase(low, "now") || contains_phrase(low, "immediately") ||
        contains_phrase(low, "blocker") || contains_phrase(low, "critical"))
        return 0.90;

    if (contains_phrase(low, "later") || contains_phrase(low, "optional") ||
        contains_phrase(low, "whenever") || contains_phrase(low, "nice to have"))
        return 0.30;

    return 0.50;
}

} // namespace

std::string fold_to_attention_kind(const std::string& fold) {
    if (fold == "TOPOLOGY") return "topology";
    if (fold == "MEMORY") return "memory";
    if (fold == "MATH") return "mathematical";
    if (fold == "CAPABILITY") return "capability";
    if (fold == "EVIDENCE") return "evidence";
    if (fold == "IMPLICATION") return "semantic";
    return "semantic";
}

std::string kind_to_attention_slot(const std::string& kind) {
    if (kind == "topology") return "DEPENDENCIES";
    if (kind == "memory") return "CONTEXT";
    if (kind == "mathematical") return "OPERATIONS";
    if (kind == "capability") return "DEPENDENCIES";
    if (kind == "semantic") return "CONTEXT";
    return "EVIDENCE";
}

RuntimeAttentionResult compute_runtime_attention(const RuntimeAttentionInputs& in) {
    RuntimeAttentionResult out;
    out.relevance_cold_start = in.relevance_cold_start;

    const auto required_count = std::max<std::size_t>(1, in.required_folds.size());
    out.D = std::clamp((double)in.missing_context_count / (double)required_count, 0.0, 1.0);

    out.lexical_prior = std::clamp(
        0.5 * (double)in.bow_score +
        0.5 * (double)in.distributional_score,
        0.0, 1.0);

    const double relevance_base = std::clamp((double)in.relevance_score, 0.0, 1.0);
    out.R = std::clamp(
        in.relevance_cold_start
            ? std::max(0.30, out.lexical_prior)
            : std::max(0.05, relevance_base),
        0.0, 1.0);

    // Continuous unresolvedness [0,1]; MISS keeps a floor to avoid zero-pressure dead zones.
    out.U = std::clamp(
        std::max(0.15, 1.0 - (double)in.confidence_score),
        0.0, 1.0);

    out.P_route = std::clamp(
        (double)in.static_weight * (double)in.health_score,
        0.0, 1.0);
    out.P_query = query_priority(in.query_lower);
    out.P = std::clamp(0.80 * out.P_route + 0.20 * out.P_query, 0.0, 1.0);

    out.research_pressure = std::clamp(out.D * out.R * out.U, 0.0, 1.0);

    out.candidates.reserve(std::max<std::size_t>(1, in.required_folds.size()));
    for (const auto& fold : in.required_folds) {
        const auto kind = fold_to_attention_kind(fold);
        const bool primary = (kind == in.miss_kind);
        const double local_R = std::clamp(primary ? out.R : (out.R * 0.90), 0.0, 1.0);
        const double local_D = std::clamp(primary ? std::min(1.0, out.D + 0.15) : (out.D * 0.75), 0.0, 1.0);

        RuntimeAttentionCandidate c;
        c.id = "fold:" + fold;
        c.kind = kind;
        c.slot = kind_to_attention_slot(kind);
        c.R = local_R;
        c.U = out.U;
        c.D = local_D;
        c.P = out.P;
        c.score = std::clamp(local_R * out.U * local_D * out.P, 0.0, 1.0);
        out.candidates.push_back(std::move(c));
    }

    if (out.candidates.empty()) {
        RuntimeAttentionCandidate c;
        c.id = "fold:EVIDENCE";
        c.kind = in.miss_kind.empty() ? "semantic" : in.miss_kind;
        c.slot = kind_to_attention_slot(c.kind);
        c.R = out.R;
        c.U = out.U;
        c.D = out.D;
        c.P = out.P;
        c.score = std::clamp(out.R * out.U * out.D * out.P, 0.0, 1.0);
        out.candidates.push_back(std::move(c));
    }

    std::sort(out.candidates.begin(), out.candidates.end(),
              [](const RuntimeAttentionCandidate& a, const RuntimeAttentionCandidate& b) {
                  if (std::fabs(a.score - b.score) > 1e-9) return a.score > b.score;
                  if (std::fabs(a.D - b.D) > 1e-9) return a.D > b.D;
                  if (std::fabs(a.R - b.R) > 1e-9) return a.R > b.R;
                  return a.id < b.id;
              });

    out.focus = out.candidates.front();
    out.score = out.focus.score;
    out.admitted = out.score >= out.admission_threshold;
    return out;
}

double compute_wcr_score(const WcrSelectionInputs& in) {
    const double W = std::clamp(in.static_weight, 0.0, 1.0);
    const double C = std::clamp(in.confidence, 0.0, 1.0);
    const double R = std::clamp(in.relevance, 0.0, 1.0);
    return std::clamp(W * C * R, 0.0, 1.0);
}

double compute_vector_relevance(const SemanticVector3& query_vector, const SemanticVector3& key_vector) {
    if (!query_vector.valid || !key_vector.valid) return 0.5;
    const double dot = query_vector.x * key_vector.x +
                       query_vector.y * key_vector.y +
                       query_vector.z * key_vector.z;
    const double cosine = std::clamp(dot, -1.0, 1.0);
    return std::clamp(0.5 * (cosine + 1.0), 0.0, 1.0);
}

double compute_wcr_qkv_score(const SemanticVector3& query_vector,
                             const SemanticVector3& key_vector,
                             double static_weight,
                             double confidence) {
    return compute_wcr_score({static_weight, confidence, compute_vector_relevance(query_vector, key_vector)});
}

SemanticVector3 derive_vector_from_tokens(const std::vector<std::string>& tokens_lower) {
    // Hash-based synthetic geometry for cold-start/unknown-query fallback paths.
    // Canonical Q/K geometry should come from declared semantic vector metadata.
    if (tokens_lower.empty()) return {};

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    std::size_t kept = 0;
    for (std::size_t i = 0; i < tokens_lower.size(); ++i) {
        const auto token = lower_ascii(tokens_lower[i]);
        if (token.empty()) continue;
        const uint64_t h = fnv1a64(token);
        const double a = static_cast<double>(h & 0xffffull) * 0.0009;
        const double b = static_cast<double>((h >> 16) & 0xffffull) * 0.0011;
        const double c = static_cast<double>((h >> 32) & 0xffffull) * 0.0013;
        sx += std::sin(a + static_cast<double>(i) * 0.17);
        sy += std::cos(b + static_cast<double>(i) * 0.13);
        sz += std::sin(c + static_cast<double>(i) * 0.11);
        ++kept;
    }
    if (kept == 0) return {};
    return normalized(sx / static_cast<double>(kept),
                      sy / static_cast<double>(kept),
                      sz / static_cast<double>(kept));
}

SemanticVector3 derive_vector_from_text(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    cur.reserve(32);
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '_') {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (!cur.empty()) {
            tokens.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return derive_vector_from_tokens(tokens);
}

} // namespace khanary::io
