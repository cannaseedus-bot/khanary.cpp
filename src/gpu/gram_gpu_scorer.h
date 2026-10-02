#pragma once
#include "d3d11_device.h"
#include "stb_map.h"
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace khanary::kuhul { class GramStore; }

namespace khanary::gpu {

struct GramScoreResult {
    std::string key;
    float       semantic_score = 0.0f;
    float       base_score     = 0.0f;
    bool        admitted       = false;
};

struct LorentzBoostRows {
    std::array<float, 4> row0{1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> row1{0.0f, 1.0f, 0.0f, 0.0f};
    std::array<float, 4> row2{0.0f, 0.0f, 1.0f, 0.0f};
    std::array<float, 4> row3{0.0f, 0.0f, 0.0f, 1.0f};
    bool  enabled = false;
    float alignment_mix = 1.0f;
    float mass_scale = 1.0f;

    // Host-side constructor for a pure Minkowski(-,+,+,+) boost. Direction is
    // normalized deterministically; rapidity must be nonnegative finite.
    static LorentzBoostRows make_pure(float rapidity_rho,
                                      std::array<float, 3> direction,
                                      float alignment_mix_value = 1.0f,
                                      float mass_scale_value = 1.0f);
};

// GPU batch scorer for GramStore records via attention.hlsl.
// Packs gram channel/authority/relevance and gramVector geodesic coordinates
// into SemanticRecord[] and dispatches the semantic attention compute kernel.
//
// CPU/GPU split: call site is responsible for supplying relevance scores
// (e.g. from cosine similarity with the query vector). This class handles
// only the GPU scoring pass; the result is sorted and returned for use by
// whatever caller owns the retrieval loop.
class GramGpuScorer {
public:
    explicit GramGpuScorer(D3D11Device& dev);

    // Compile the attention shader. Returns false if D3DCompile unavailable.
    bool init();
    bool is_ready() const { return ready_; }

    // Score candidate gram keys from the store on the GPU.
    // relevance[i] is caller-supplied query relevance in [0,1] for candidate_keys[i].
    // Returns admitted entries first, all sorted descending by semantic_score.
    std::vector<GramScoreResult> score_batch(
        const kuhul::GramStore& store,
        const std::vector<std::string>& candidate_keys,
        const std::vector<float>&       relevance,
        float admission_threshold = 0.3f,
        float relation_mix        = 0.5f,
        float tensor_mix          = 0.0f,
        const std::vector<STBGeodesicRecord>* geodesic_metrics = nullptr,
        float geodesic_mix        = 0.0f,
        float beta_diffusion      = 0.0f,
        const LorentzBoostRows* lorentz_boost = nullptr,
        const std::unordered_map<std::string, std::array<float, 3>>* vector_overrides = nullptr);

private:
    D3D11Device&                dev_;
    bool                        ready_ = false;
    ComPtr<ID3D11ComputeShader> cs_;
    ComPtr<ID3D11Buffer>        cbuf_;
    ComPtr<ID3D11Buffer>        lorentz_cbuf_;

    // Channel → W (static_weight): reflects how authoritative the gram's channel is.
    static float channel_w(const std::string& ch);
    // Authority → C (confidence): reflects how trusted the gram's authority is.
    static float authority_c(const std::string& auth);
};

} // namespace khanary::gpu
