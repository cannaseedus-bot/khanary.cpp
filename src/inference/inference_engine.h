#pragma once
#include "tokenizer.h"
#include "moe_router.h"
#include "kv_cache.h"
#include "sampler.h"
#include "../gpu/dml_ops.h"
#include "../kuhul/phase_engine.h"
#include <string>
#include <vector>
#include <functional>

namespace khanary::inference {

struct InferenceConfig {
    uint32_t n_layers    = 12;
    uint32_t d_model     = 768;
    uint32_t d_head      = 64;
    uint32_t n_heads     = 12;
    uint32_t n_experts   = 8;
    uint32_t top_k       = 2;
    uint32_t max_tokens  = 256;
    float    temperature = 0.7f;
    float    top_p       = 0.9f;
};

struct InferenceResult {
    std::string text;
    std::vector<uint32_t> token_ids;
    kuhul::Phase final_phase;
    bool ok;
    std::string error;
};

class InferenceEngine {
public:
    InferenceEngine(gpu::D3D11Device& dev, InferenceConfig cfg = {});

    // Full pipeline: prompt → token ids → phase traverse → sample → text
    InferenceResult run(const std::string& prompt);

    // Set a token callback (called per generated token)
    using TokenCallback = std::function<void(uint32_t id, const std::string& tok)>;
    void set_token_callback(TokenCallback cb) { token_cb_ = std::move(cb); }

    GlyphTokenizer& tokenizer() { return tokenizer_; }
    MoERouter&      router()    { return *router_; }
    KVCache&        kv_cache()  { return *kv_cache_; }
    Sampler&        sampler()   { return sampler_; }

    void load_expert_weights(uint32_t expert_id, const float* data, size_t count);
    void clear_kv();

private:
    gpu::D3D11Device& dev_;
    InferenceConfig   cfg_;
    GlyphTokenizer    tokenizer_;
    std::unique_ptr<MoERouter> router_;
    std::unique_ptr<KVCache>   kv_cache_;
    Sampler           sampler_;
    gpu::DMLOps       dml_;
    kuhul::PhaseEngine pe_;
    TokenCallback     token_cb_;

    // Expert weight matrices [n_experts][d_model*d_model]
    std::vector<std::vector<float>> expert_weights_;

    // Single forward pass: tokens → logits
    std::vector<float> forward(const std::vector<uint32_t>& tokens);

    // Simple embedding lookup (hash-based, no pretrained weights)
    std::vector<float> embed(const std::vector<uint32_t>& tokens);

    // Apply one expert's FFN
    std::vector<float> apply_expert(uint32_t expert_id, const std::vector<float>& hidden);
};

} // namespace khanary::inference
