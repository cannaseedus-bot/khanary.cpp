#include "gyroscope.h"
#include "../util/json.h"
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace khanary::world {

namespace {

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

} // namespace

const char* abi_dim_name(AbiDim d) {
    switch (d) {
        case AbiDim::Field:          return "⟁FIELD";
        case AbiDim::Evidence:       return "⟁EVIDENCE";
        case AbiDim::Infer:          return "⟁INFER";
        case AbiDim::Counterexample: return "⟁COUNTEREXAMPLE";
        case AbiDim::Verify:         return "⟁VERIFY";
        case AbiDim::Xul:            return "⟁XUL";
    }
    return "⟁UNKNOWN";
}

const char* track_dim_name(int idx) {
    static constexpr const char* names[TRACK_DIM_COUNT] = {
        "REASON-µ", "INFER-µ", "ORCH-µ", "MEM-µ", "CREATE-µ", "ELIZA"
    };
    if (idx < 0 || idx >= TRACK_DIM_COUNT) return "UNKNOWN";
    return names[idx];
}

// Hardcoded track name → column index.  Must match track_dim_name() ordering.
static int track_index(const std::string& name) {
    if (name == "REASON-µ") return 0;
    if (name == "INFER-µ")  return 1;
    if (name == "ORCH-µ")   return 2;
    if (name == "MEM-µ")    return 3;
    if (name == "CREATE-µ") return 4;
    if (name == "ELIZA")    return 5;
    return -1;
}

Gyroscope::Gyroscope(const std::string& abi_book_path, float alignment_tolerance)
    : tol_(alignment_tolerance)
{
    for (auto& row : a_v_) row.fill(0.0f);

    std::ifstream f(abi_book_path);
    if (!f.is_open())
        throw std::runtime_error("Gyroscope: cannot open ABI book: " + abi_book_path);

    std::string text(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>{});
    auto root = util::parse(text);
    if (!root || !root->is_object())
        throw std::runtime_error("Gyroscope: invalid JSON in ABI book: " + abi_book_path);

    // Find the first key starting with "⟁GYRO_ABI"
    const util::JsonValue* book_val = nullptr;
    for (const auto& [k, v] : root->as_object()) {
        if (k.rfind("⟁GYRO_ABI", 0) == 0) { book_val = v.get(); break; }
    }
    if (!book_val || !book_val->is_object())
        throw std::runtime_error("Gyroscope: no ⟁GYRO_ABI_vN key found in: " + abi_book_path);

    const auto& book = book_val->as_object();
    auto map_it = book.find("mapping");
    if (map_it == book.end() || !map_it->second->is_object())
        throw std::runtime_error("Gyroscope: 'mapping' not found in ABI book");

    const auto& mapping = map_it->second->as_object();

    for (int dim = 0; dim < ABI_DIM_COUNT; ++dim) {
        auto it = mapping.find(abi_dim_name(AbiDim(dim)));
        if (it == mapping.end() || !it->second->is_array()) continue;

        const auto& arr = it->second->as_array();
        if (arr.empty()) continue;
        float w = 1.0f / static_cast<float>(arr.size());  // uniform average
        for (const auto& tv : arr) {
            if (!tv->is_string()) continue;
            int col = track_index(tv->as_string());
            if (col >= 0) a_v_[dim][col] = w;
        }
    }

    // Load track_governs: natural-language vocabulary per µ-track.
    // Order must match measurement_tracks (hardcoded in track_index()).
    auto gov_it = book.find("track_governs");
    if (gov_it != book.end() && gov_it->second->is_object()) {
        for (const auto& [track_name, arr_val] : gov_it->second->as_object()) {
            int col = track_index(track_name);
            if (col < 0 || !arr_val->is_array()) continue;
            for (const auto& wv : arr_val->as_array())
                if (wv->is_string()) governs_sets_[col].insert(wv->as_string());
        }
    }
}

GyroResult Gyroscope::observe(
    const TrackScores& s_mu_projection,
    const TrackScores& s_mu_dds,
    const GyroInputs&  inputs) const
{
    GyroResult res{};

    // Δµ = s_mu_projection − s_mu_dds  (measurement-basis residual across neural boundary)
    for (int j = 0; j < TRACK_DIM_COUNT; ++j)
        res.track_residual[j] = s_mu_projection[j] - s_mu_dds[j];

    // ΔABI = A_v · Δµ  (epistemic-coordinate residual)
    float r_sq = 0.0f;
    for (int i = 0; i < ABI_DIM_COUNT; ++i) {
        float v = 0.0f;
        for (int j = 0; j < TRACK_DIM_COUNT; ++j)
            v += a_v_[i][j] * res.track_residual[j];
        res.abi_residual[i] = v;
        r_sq += v * v;
    }

    res.r = std::sqrt(r_sq);

    // Γ = m_dds · ω² / k  (r-independent; ratio of centrifugal to restoring magnitudes)
    res.gamma = (inputs.k > 0.0f)
        ? (inputs.m_dds * inputs.omega * inputs.omega) / inputs.k
        : std::numeric_limits<float>::infinity();

    res.dominant_anchor    = dominant_anchor_for(res.abi_residual);
    res.dynamically_stable = (res.gamma < 1.0f);
    res.aligned            = (res.r <= tol_);

    return res;
}

TrackScores Gyroscope::canonical_abi_scores() {
    TrackScores s;
    s.fill(1.0f);  // all tracks fully in-vocabulary — calibration / unit-test reference only
    return s;
}

TrackScores Gyroscope::score_tokens(
    const std::vector<std::string>& tokens,
    const std::array<std::unordered_set<std::string>, TRACK_DIM_COUNT>& governs_sets)
{
    return score_tokens(tokens, governs_sets, nullptr, 0.65f);
}

TrackScores Gyroscope::score_tokens(
    const std::vector<std::string>& tokens,
    const std::array<std::unordered_set<std::string>, TRACK_DIM_COUNT>& governs_sets,
    const LexicalEquivalenceMap* lexical_equivalence,
    float synonym_credit)
{
    std::unordered_set<std::string> tok_set(tokens.begin(), tokens.end());
    std::unordered_set<std::string> synonym_expanded;
    synonym_credit = std::clamp(synonym_credit, 0.0f, 1.0f);

    if (lexical_equivalence) {
        for (const auto& tok : tok_set) {
            auto hit = lexical_equivalence->find(tok);
            if (hit == lexical_equivalence->end()) continue;
            for (const auto& alias : hit->second) {
                const auto norm = lower_ascii(alias);
                if (!norm.empty()) synonym_expanded.insert(norm);
            }
        }
    }
    std::array<float, TRACK_DIM_COUNT> scores{};

    for (int i = 0; i < TRACK_DIM_COUNT; ++i) {
        const auto& gov = governs_sets[i];
        if (gov.empty()) { scores[i] = 0.0f; continue; }
        float hits = 0.0f;
        for (const auto& g_raw : gov) {
            const auto g = lower_ascii(g_raw);
            if (g.empty()) continue;
            if (tok_set.count(g)) {
                hits += 1.0f;
                continue;
            }

            bool synonym_hit = false;
            if (lexical_equivalence) {
                if (synonym_expanded.count(g)) {
                    synonym_hit = true;
                } else if (auto rev = lexical_equivalence->find(g); rev != lexical_equivalence->end()) {
                    for (const auto& alias : rev->second) {
                        if (tok_set.count(lower_ascii(alias))) {
                            synonym_hit = true;
                            break;
                        }
                    }
                }
            }
            if (synonym_hit) hits += synonym_credit;
        }
        scores[i] = hits / static_cast<float>(gov.size());
    }
    return scores;
}

float Gyroscope::cosine_relevance(const std::array<float, 3>& query_vector,
                                  const std::array<float, 3>& key_vector) {
    const float qn = std::sqrt(
        query_vector[0] * query_vector[0] +
        query_vector[1] * query_vector[1] +
        query_vector[2] * query_vector[2]);
    const float kn = std::sqrt(
        key_vector[0] * key_vector[0] +
        key_vector[1] * key_vector[1] +
        key_vector[2] * key_vector[2]);
    if (!(qn > 1e-6f) || !(kn > 1e-6f)) return 0.5f;
    const float cosine =
        (query_vector[0] * key_vector[0] +
         query_vector[1] * key_vector[1] +
         query_vector[2] * key_vector[2]) / (qn * kn);
    return std::clamp(0.5f * (cosine + 1.0f), 0.0f, 1.0f);
}

std::string Gyroscope::dominant_anchor_for(const std::array<float, ABI_DIM_COUNT>& delta) {
    int   best     = 0;
    float best_abs = std::fabs(delta[0]);
    for (int i = 1; i < ABI_DIM_COUNT; ++i) {
        float a = std::fabs(delta[i]);
        if (a > best_abs) { best_abs = a; best = i; }
    }
    return abi_dim_name(AbiDim(best));
}

} // namespace khanary::world
