#include "gravity_field.h"
#include "relation_isa.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <utility>

namespace khanary::micronaut {

namespace {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    bool valid = false;
};

static std::string lower_ascii(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static float normalize_threshold(float threshold) {
    if (!std::isfinite(threshold)) return 0.1f;
    return std::max(0.0f, threshold);
}

static float clamp01(float value, float fallback = 1.0f) {
    if (!std::isfinite(value)) return fallback;
    return std::clamp(value, 0.0f, 1.0f);
}

static float nonnegative(float value, float fallback = 0.0f) {
    if (!std::isfinite(value)) return fallback;
    return std::max(0.0f, value);
}

static Vec3 normalize_vec(float x, float y, float z) {
    const float n = std::sqrt(x * x + y * y + z * z);
    if (!(n > 1e-6f)) return {};
    return {x / n, y / n, z / n, true};
}

static Vec3 phase_unit_vector(kuhul::Phase phase) {
    constexpr float kPi = 3.14159265358979323846f;
    const float angle = (2.0f * kPi / 6.0f) * static_cast<int>(phase);
    return normalize_vec(std::cos(angle), std::sin(angle), 0.0f);
}

static float cosine01(const Vec3& a, const Vec3& b) {
    if (!a.valid || !b.valid) return 0.5f;
    const float dot = a.x * b.x + a.y * b.y + a.z * b.z;
    const float cosine = std::clamp(dot, -1.0f, 1.0f);
    return std::clamp(0.5f * (cosine + 1.0f), 0.0f, 1.0f);
}

static Vec3 query_vector_from_tokens(const std::vector<std::string>& tokens) {
    if (tokens.empty()) return {};
    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    size_t kept = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string t = lower_ascii(tokens[i]);
        if (t.empty()) continue;
        uint32_t h = 2166136261u;
        for (unsigned char c : t) {
            h ^= static_cast<uint32_t>(c);
            h *= 16777619u;
        }
        const float a = static_cast<float>(h & 0x3ffu) * 0.009f;
        const float b = static_cast<float>((h >> 10) & 0x3ffu) * 0.011f;
        const float c = static_cast<float>((h >> 20) & 0x3ffu) * 0.013f;
        sx += std::sin(a + static_cast<float>(i) * 0.17f);
        sy += std::cos(b + static_cast<float>(i) * 0.13f);
        sz += std::sin(c + static_cast<float>(i) * 0.11f);
        ++kept;
    }
    if (kept == 0) return {};
    return normalize_vec(sx / static_cast<float>(kept),
                         sy / static_cast<float>(kept),
                         sz / static_cast<float>(kept));
}

} // namespace

void GravityField::register_nodes(
    const std::string& micronaut_id,
    const std::unordered_set<std::string>& nodes,
    kuhul::Phase phase,
    float vector_x,
    float vector_y,
    float vector_z,
    bool vector_declared,
    float static_weight,
    float confidence,
    float health,
    float entropy_diffusion,
    bool policy_ok,
    bool capable,
    bool xcfe_ok)
{
    if (micronaut_id.empty()) return;

    std::unordered_set<std::string> normalized_nodes;
    normalized_nodes.reserve(nodes.size());
    for (const auto& node : nodes) {
        const auto lowered = lower_ascii(node);
        if (lowered.empty()) continue;
        normalized_nodes.insert(lowered);
        const auto inferred = infer_relation_from_lexical(lowered);
        if (!inferred.empty()) normalized_nodes.insert(lower_ascii(inferred));
    }

    std::lock_guard<std::mutex> lk(mtx_);
    registry_[micronaut_id] = {
        micronaut_id,
        std::move(normalized_nodes),
        phase,
        vector_x,
        vector_y,
        vector_z,
        vector_declared,
        clamp01(static_weight, 1.0f),
        clamp01(confidence, 1.0f),
        clamp01(health, 1.0f),
        nonnegative(entropy_diffusion, 0.0f),
        policy_ok,
        capable,
        xcfe_ok
    };
}

bool GravityField::unregister(const std::string& micronaut_id) {
    if (micronaut_id.empty()) return false;
    std::lock_guard<std::mutex> lk(mtx_);
    return registry_.erase(micronaut_id) > 0;
}

float GravityField::min_distance(
    const std::string& token,
    const std::unordered_set<std::string>& nodes) const
{
    if (token.empty() || nodes.empty()) return 4.0f;

    const std::string t = lower_ascii(token);
    if (t.empty()) return 4.0f;

    // Direct membership: distance = 0
    if (nodes.count(t)) return 0.0f;
    // Prefix/substring: distance = 1
    for (const auto& n : nodes) {
        if (n.empty()) continue;
        if (n.find(t) != std::string::npos || t.find(n) != std::string::npos)
            return 1.0f;
    }
    // Levenshtein-lite: first char match → distance = 2
    for (const auto& n : nodes) {
        if (!n.empty() && n[0] == t[0]) return 2.0f;
    }
    return 4.0f; // far
}

std::unordered_map<std::string, float> GravityField::score(
    const std::vector<std::string>& tokens,
    kuhul::Phase phase) const
{
    const auto captured = capture(tokens, phase, 0.0f);
    std::unordered_map<std::string, float> scores;
    scores.reserve(captured.size());
    for (const auto& row : captured) {
        scores[row.micronaut_id] = row.capture_score;
    }
    return scores;
}

std::vector<MicronautAttraction> GravityField::capture(
    const std::vector<std::string>& tokens,
    kuhul::Phase phase,
    float threshold) const
{
    std::unordered_map<std::string, MicronautNodes> snapshot;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        snapshot = registry_;
    }
    const Vec3 qvec = query_vector_from_tokens(tokens);
    const float capture_threshold = normalize_threshold(threshold);

    std::vector<MicronautAttraction> out;
    out.reserve(snapshot.size());
    for (const auto& [id, entry] : snapshot) {
        MicronautAttraction row;
        row.micronaut_id = id;
        row.static_weight = clamp01(entry.static_weight, 1.0f);
        row.confidence = clamp01(entry.confidence, 1.0f);
        row.health = clamp01(entry.health, 1.0f);
        row.entropy_diffusion = nonnegative(entry.entropy_diffusion, 0.0f);
        row.policy_ok = entry.policy_ok;
        row.capable = entry.capable;
        row.xcfe_ok = entry.xcfe_ok;
        row.legal = row.policy_ok && row.capable && row.xcfe_ok;

        if (id.empty() || entry.nodes.empty()) {
            out.push_back(std::move(row));
            continue;
        }

        float g = 0.0f;
        for (const auto& tok : tokens) {
            if (tok.empty()) continue;
            float d = min_distance(tok, entry.nodes);
            if (d < 4.0f) {
                g += (float)entry.nodes.size() / ((d + 1.0f) * (d + 1.0f));
            }
            const auto inferred = infer_relation_from_lexical(tok);
            if (!inferred.empty()) {
                const float rd = min_distance(inferred, entry.nodes);
                if (rd < 4.0f) {
                    g += 0.85f * (float)entry.nodes.size() / ((rd + 1.0f) * (rd + 1.0f));
                }
            }
        }
        // Phase arc weight supplies provider/phase relevance in WCR.
        const double w = std::max(0.0, kuhul::arc_weight(entry.phase, phase));
        const Vec3 key_vec = entry.vector_declared
            ? normalize_vec(entry.vector_x, entry.vector_y, entry.vector_z)
            : phase_unit_vector(entry.phase);
        row.semantic_mass = nonnegative(g, 0.0f);
        row.geometric_alignment = cosine01(qvec, key_vec);
        row.phase_relevance = clamp01(static_cast<float>(w), 0.0f);
        row.wcr_prior = std::clamp(
            row.static_weight * row.confidence * row.health * row.phase_relevance,
            0.0f, 1.0f);
        row.phi = row.semantic_mass * row.geometric_alignment * row.wcr_prior;
        row.capture_score = row.legal
            ? std::max(0.0f, row.phi - row.entropy_diffusion)
            : 0.0f;
        row.captured = row.legal && row.capture_score >= capture_threshold;
        out.push_back(std::move(row));
    }
    std::sort(out.begin(), out.end(), [](const MicronautAttraction& a,
                                         const MicronautAttraction& b) {
        constexpr float kEps = 1e-6f;
        if (std::fabs(a.capture_score - b.capture_score) > kEps)
            return a.capture_score > b.capture_score;
        if (std::fabs(a.phi - b.phi) > kEps)
            return a.phi > b.phi;
        return a.micronaut_id < b.micronaut_id;
    });
    return out;
}

std::string GravityField::best_match(
    const std::vector<std::string>& tokens,
    kuhul::Phase phase,
    float threshold) const
{
    auto scores = score(tokens, phase);
    std::string best_id;
    float best_score = normalize_threshold(threshold);
    constexpr float kEps = 1e-6f;
    for (const auto& [id, s] : scores) {
        if (s > best_score + kEps ||
            (std::fabs(s - best_score) <= kEps && !best_id.empty() && id < best_id)) {
            best_score = s;
            best_id = id;
        }
    }
    return best_id;
}

std::vector<std::string> GravityField::merge_phase(
    const std::string& micronaut_a,
    const std::string& micronaut_b,
    kuhul::Phase /*phase*/) const
{
    // Horizontal fold: union of node sets at same phase
    std::vector<std::string> merged;
    std::lock_guard<std::mutex> lk(mtx_);
    auto it_a = registry_.find(micronaut_a);
    auto it_b = registry_.find(micronaut_b);
    if (it_a != registry_.end())
        for (const auto& n : it_a->second.nodes) merged.push_back(n);
    if (it_b != registry_.end())
        for (const auto& n : it_b->second.nodes) merged.push_back(n);
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    return merged;
}

} // namespace khanary::micronaut
