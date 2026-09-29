#pragma once
#include "d3d11_device.h"
#include <string>
#include <vector>

namespace khanary::kuhul { class GramStore; }

namespace khanary::gpu {

struct GramScoreResult {
    std::string key;
    float       semantic_score = 0.0f;
    float       base_score     = 0.0f;
    bool        admitted       = false;
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
        float relation_mix        = 0.5f);

private:
    D3D11Device&                dev_;
    bool                        ready_ = false;
    ComPtr<ID3D11ComputeShader> cs_;
    ComPtr<ID3D11Buffer>        cbuf_;

    // Channel → W (static_weight): reflects how authoritative the gram's channel is.
    static float channel_w(const std::string& ch);
    // Authority → C (confidence): reflects how trusted the gram's authority is.
    static float authority_c(const std::string& auth);
};

} // namespace khanary::gpu
