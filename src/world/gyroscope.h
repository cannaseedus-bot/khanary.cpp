#pragma once
#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cctype>

namespace khanary::world {

// ── Stable external coordinates ──────────────────────────────────────────────
// Invariant to Micronaut topology changes.  The Gyroscope ABI version is a
// property of the Book file (gyro_abi_v1.json), not of the Micronaut set.
enum class AbiDim : uint8_t {
    Field          = 0,
    Evidence       = 1,
    Infer          = 2,
    Counterexample = 3,
    Verify         = 4,
    Xul            = 5,
};
constexpr int ABI_DIM_COUNT   = 6;

// ── Replaceable sensor basis ──────────────────────────────────────────────────
// µ-track column order: [REASON-µ, INFER-µ, ORCH-µ, MEM-µ, CREATE-µ, ELIZA]
// Must match the measurement_tracks array in gyro_abi_v1.json.
constexpr int TRACK_DIM_COUNT = 6;

// Convenience alias for a scored µ-track vector.
// Use s_mu_projection (from JROM_projection) and s_mu_dds (from DDS completion)
// as the two arguments to observe().  Do NOT pass the raw prompt or trace.
using TrackScores = std::array<float, TRACK_DIM_COUNT>;
using LexicalEquivalenceMap = std::unordered_map<std::string, std::vector<std::string>>;

const char* abi_dim_name(AbiDim d);    // "⟁FIELD" … "⟁XUL"
const char* track_dim_name(int idx);   // "REASON-µ" … "ELIZA"

// A_v[dim][track]: row-normalized projection from µ-space → ABI-space.
// Loaded from Book file; rows sum to 1.0 (uniform weight per contributing track).
using AbiMapMatrix = std::array<std::array<float, TRACK_DIM_COUNT>, ABI_DIM_COUNT>;

// ── Governor-supplied load state ──────────────────────────────────────────────
struct GyroInputs {
    float omega;  // cognitive load: ThinkingLevel × delegation_fan_out
    float m_dds;  // DDS inertia:    L_token_bar × (1 + σ_route)
    float k;      // ABI anchoring strength (policy knob); must be > 0
};

// ── Gyroscope output ──────────────────────────────────────────────────────────
struct GyroResult {
    // Stable projection-ABI coordinates (external interface — invariant to Micronaut changes)
    std::array<float, ABI_DIM_COUNT>   abi_residual;   // ΔABI = A_v · Δµ

    // Diagnostic measurement coordinates (telemetry / debugging)
    std::array<float, TRACK_DIM_COUNT> track_residual; // Δµ = s_mu_abi − s_mu_dds

    float r;      // ||abi_residual|| — scalar displacement from ABI center
    float gamma;  // Γ = m_dds·ω²/k  — load-dependent stability ratio (r-independent)

    // "⟁FIELD" | "⟁EVIDENCE" | "⟁INFER" | "⟁COUNTEREXAMPLE" | "⟁VERIFY" | "⟁XUL"
    // argmax of |abi_residual[i]| — tells Governor which anchor to strengthen
    std::string dominant_anchor;

    // ── Four diagnostic states (Γ × r) ───────────────────────────────────────
    // γ<1  r≤tol  → aligned + dynamically stable     (healthy)
    // γ<1  r>tol  → displaced but recoverable         (can restore)
    // γ>1  r≤tol  → aligned but regime unstable       (pre-failure warning)
    // γ>1  r>tol  → drifting and unstable             (reduce load / re-anchor)
    bool dynamically_stable;  // gamma < 1.0
    bool aligned;             // r <= alignment_tolerance
};

// ── Gyroscope ─────────────────────────────────────────────────────────────────
class Gyroscope {
public:
    // Load A_v from a Book JSON file.
    // Format: { "⟁GYRO_ABI_vN": { "mapping": { "⟁DIM": ["TRACK-µ", ...] } } }
    // alignment_tolerance: r threshold below which aligned=true (default 0.15)
    explicit Gyroscope(const std::string& abi_book_path,
                       float alignment_tolerance = 0.15f);

    // Compute GyroResult from pre-scored µ-track vectors.
    //   s_mu_projection: scores for the JROM_projection output of project_v()
    //   s_mu_dds:        scores for the DDS completion
    //   inputs:          Governor-supplied load state
    //
    // Measures displacement across the neural boundary:
    //   JROM_projection → DDS → completion
    // NOT the prompt and NOT a fixed canonical ideal (use canonical_abi_scores()
    // for calibration / startup tests instead).
    GyroResult observe(
        const TrackScores& s_mu_projection,
        const TrackScores& s_mu_dds,
        const GyroInputs&  inputs) const;

    // Fixed all-ones reference: every track fully in-vocabulary.
    // Use for calibration, unit tests, and startup self-checks only.
    // Pass as s_mu_projection to measure absolute ABI conformance of a DDS output.
    static TrackScores canonical_abi_scores();

    // Governs-vocabulary coverage scoring.
    // For each track i: score[i] = |governs(i) ∩ tokens| / |governs(i)|
    // This is a fraction in [0,1]; 1.0 = full governs vocabulary present in tokens.
    // NOTE: coverage-fraction is a design choice for this function.  Callers that
    // need weighted or TF-IDF scoring should compute TrackScores directly and pass to observe().
    static TrackScores score_tokens(
        const std::vector<std::string>& tokens,
        const std::array<std::unordered_set<std::string>, TRACK_DIM_COUNT>& governs_sets);

    // Lexical-equivalence scoring:
    // direct token match contributes 1.0, synonym/equivalent match contributes synonym_credit.
    static TrackScores score_tokens(
        const std::vector<std::string>& tokens,
        const std::array<std::unordered_set<std::string>, TRACK_DIM_COUNT>& governs_sets,
        const LexicalEquivalenceMap* lexical_equivalence,
        float synonym_credit = 0.65f);

    // Governs sets loaded from the ABI book's track_governs field.
    // Use with score_tokens() to measure semantic coverage of any text.
    const std::array<std::unordered_set<std::string>, TRACK_DIM_COUNT>& make_governs_sets() const {
        return governs_sets_;
    }

    // Tokenize text: extract all lowercase alphabetic words (len >= 3).
    // Strips JSON/structural punctuation so payload content scores correctly.
    static std::vector<std::string> tokenize(const std::string& text) {
        std::string normalized;
        normalized.reserve(text.size());
        for (unsigned char c : text)
            normalized += std::isalpha(c) ? static_cast<char>(std::tolower(c)) : ' ';
        std::vector<std::string> tokens;
        std::istringstream iss(normalized);
        for (std::string tok; iss >> tok;)
            if (tok.size() >= 3) tokens.push_back(std::move(tok));
        return tokens;
    }

    const AbiMapMatrix& abi_map()           const { return a_v_; }
    float               alignment_tolerance() const { return tol_; }

    // Compact vector-space helpers for KHANARY semantic-attention geometry.
    static float cosine_relevance(const std::array<float, 3>& query_vector,
                                  const std::array<float, 3>& key_vector);

private:
    AbiMapMatrix a_v_;
    float        tol_;
    std::array<std::unordered_set<std::string>, TRACK_DIM_COUNT> governs_sets_;

    static std::string dominant_anchor_for(const std::array<float, ABI_DIM_COUNT>& delta);
};

} // namespace khanary::world
