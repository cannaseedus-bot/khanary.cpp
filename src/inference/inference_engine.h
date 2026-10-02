#pragma once
#include "tokenizer.h"
#include "moe_router.h"
#include "kv_cache.h"
#include "sampler.h"
#include "fold_map.h"
#include "semantic_attention.h"
#include "../gpu/dml_ops.h"
#include "../gpu/gram_gpu_scorer.h"
#include "../gpu/hot_swap.h"
#include "../kuhul/phase_engine.h"
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <deque>
#include <optional>
#include <array>

namespace khanary::inference {

struct InferenceConfig {
    uint32_t n_layers    = 12;
    uint32_t d_model     = 768;
    uint32_t d_head      = 64;
    uint32_t n_heads     = 12;
    uint32_t n_kv_heads  = 12;    // KV heads for GQA; equals n_heads when no GQA
    uint32_t n_experts   = 8;
    uint32_t top_k       = 2;
    uint32_t vocab_size  = GLYPH_VOCAB_SIZE;
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

    // Load DDS weight shards and wire GPU forward path.
    //   kson_path     : gpt_oss_20b_folds.kson
    //   manifest_path : folds/model_dds_manifest.json
    //   shards_dir    : directory containing fold_N.dds files
    void load_dds_model(const std::string& kson_path,
                        const std::string& manifest_path,
                        const std::string& shards_dir);

    bool dds_loaded() const { return dds_loaded_; }
    size_t dds_resident_bytes() const;
    size_t dds_resident_ceiling_bytes() const;
    uint32_t dds_preload_total_tensors() const { return dds_preload_total_tensors_; }
    uint32_t dds_preload_resident_tensors() const { return dds_preload_resident_tensors_; }
    uint32_t dds_preload_failed_tensors() const { return dds_preload_failed_tensors_; }
    uint64_t dds_preload_total_bytes() const { return dds_preload_total_bytes_; }
    uint64_t dds_preload_resident_bytes() const { return dds_preload_resident_bytes_; }

    // max_tokens=0 uses cfg_.max_tokens
    InferenceResult run(const std::string& prompt, uint32_t max_tokens = 0);

    // Rolling mean of per-token generation latency (ms) over last n_recent tokens.
    // Returns 1000.0 (1 s/token placeholder) when ring buffer is empty.
    float mean_token_latency_ms(uint32_t n_recent = 32) const;

    using TokenCallback = std::function<void(uint32_t id, const std::string& tok)>;
    void set_token_callback(TokenCallback cb) { token_cb_ = std::move(cb); }

    GlyphTokenizer& tokenizer() { return tokenizer_; }
    MoERouter&      router()    { return *router_; }
    KVCache&        kv_cache()  { return *kv_cache_; }
    Sampler&        sampler()   { return sampler_; }

    void load_expert_weights(uint32_t expert_id, const float* data, size_t count);
    void clear_kv();

    // Optional hyperbolic frame transition for future GramGpuScorer dispatch calls.
    // This is projection geometry only; it does not alter semantic authority.
    void set_lorentz_transition(float rapidity_rho,
                                std::array<float, 3> direction,
                                float alignment_mix = 1.0f,
                                float mass_scale = 1.0f);
    void clear_lorentz_transition();
    const gpu::LorentzBoostRows* lorentz_transition() const;

    // Score bounded prompt GramStore candidates on the semantic-attention GPU
    // path.  The stored Lorentz transition is passed to the shader as a
    // projection-only frame transform.  Returns empty when GPU compile/dispatch
    // is unavailable.
    std::vector<gpu::GramScoreResult> score_prompt_attention(
        const kuhul::GramStore& store,
        const std::vector<PromptAttentionCandidate>& candidates,
        float admission_threshold = 0.3f,
        float tensor_mix = 0.25f);
    bool semantic_attention_ready() const;

private:
    gpu::D3D11Device& dev_;
    InferenceConfig   cfg_;
    GlyphTokenizer    tokenizer_;
    std::unique_ptr<MoERouter>         router_;
    std::unique_ptr<KVCache>           kv_cache_;
    Sampler                            sampler_;
    gpu::DMLOps                        dml_;
    kuhul::PhaseEngine                 pe_;
    TokenCallback                      token_cb_;
    std::optional<gpu::LorentzBoostRows> lorentz_transition_;
    std::unique_ptr<gpu::GramGpuScorer> semantic_scorer_;
    mutable std::mutex                  semantic_scorer_mtx_;

    // --- DDS model state ---
    std::unique_ptr<FoldMap>             fold_map_;
    std::unique_ptr<gpu::HotSwapManager> hot_swap_;
    bool                                 dds_loaded_ = false;

    // F32-dequantized weight cache: weak_ptr so HotSwapManager owns lifetime.
    // Key = "f32:<tensor_name>"; value locked → shared_ptr kept alive during use.
    std::unordered_map<std::string, std::weak_ptr<gpu::TensorBuffer>> f32_cache_;
    mutable std::mutex                                                  cache_mtx_;

    // Legacy CPU expert weights (used when !dds_loaded_)
    std::vector<std::vector<float>> expert_weights_;

    // Token generation latency ring buffer (ms per step, capped at 256 entries)
    std::deque<float> token_latencies_ms_;

    // DDS startup preload stats.
    uint32_t dds_preload_total_tensors_    = 0;
    uint32_t dds_preload_resident_tensors_ = 0;
    uint32_t dds_preload_failed_tensors_   = 0;
    uint64_t dds_preload_total_bytes_      = 0;
    uint64_t dds_preload_resident_bytes_   = 0;

    // --- DDS loading helpers ---

    // Load shard slices for a tensor, dequantize Q8_0/MXFP4 blocks → float vector.
    // Returns empty vector if tensor not found or dtype unsupported.
    std::vector<float> load_q8_0_to_f32(const std::string& tensor_name) const;

    // Dequantize one expert's MXFP4 weight slice → host float vector.
    // Physical layout: [n_experts, rows, pack(cols)] stored contiguously per expert.
    // rows = in_dim, cols = out_dim (kson [in, out] convention).
    std::vector<float> load_mxfp4_expert_f32(
        const std::string& tensor_name,
        uint32_t expert_id, uint32_t rows, uint32_t cols) const;

    // Allocate a F32 TensorBuffer and upload host data.
    std::shared_ptr<gpu::TensorBuffer> make_f32_buffer(const float* data, uint32_t count);

    // Allocate an empty F32 output TensorBuffer (for GEMM result).
    std::shared_ptr<gpu::TensorBuffer> alloc_f32_buffer(uint32_t count);

    // Return a GPU-resident F32 weight buffer, loading from disk when cache is cold.
    // Registered with hot_swap_ under "f32:<name>"; caller holds shared_ptr during use.
    std::shared_ptr<gpu::TensorBuffer> ensure_f32_weight(const std::string& name);

    // Readback a F32 TensorBuffer to host.
    static std::vector<float> readback_f32(gpu::TensorBuffer* buf, uint32_t count);

    // Raw bytes → GPU TensorBuffer for non-GEMM consumers (sandbox, JROM, etc.)
    std::shared_ptr<gpu::TensorBuffer> ensure_resident(const std::string& tensor_name);

    // Pre-load every tensor in a fold into hot_swap_ at startup.
    void preload_fold(const std::string& layer_id);
    void preload_all_folds();

    // Derive InferenceConfig fields from the loaded FoldMap.
    void update_config_from_folds();

    // --- Forward pass ---

    // Dispatcher: routes to forward_dds() or forward_stub() based on dds_loaded_.
    std::vector<float> forward(const std::vector<uint32_t>& tokens);

    // GPU DDS path — real transformer forward.
    std::vector<float> forward_dds(const std::vector<uint32_t>& tokens);

    // Legacy CPU stub (sinusoidal embed + placeholder MoE).
    std::vector<float> forward_stub(const std::vector<uint32_t>& tokens);

    // Per-step DDS sub-passes. hidden is flat [seq_len * d_model].
    std::vector<float> embed_dds(const std::vector<uint32_t>& tokens);
    std::vector<float> apply_attention_dds(uint32_t layer,
                                            const std::vector<float>& hidden,
                                            uint32_t seq_len);
    std::vector<float> apply_moe_dds(uint32_t layer,
                                      const std::vector<float>& hidden,
                                      uint32_t seq_len);

    // CPU helpers
    static std::vector<float> rms_norm(const std::vector<float>& x,
                                        const std::vector<float>& w,
                                        uint32_t d_model);
    static float fp16_to_f32(uint16_t h);

    // Output logits via chunked Q8_0 dot-product — avoids loading the full
    // 201088-vocab weight matrix into F32 (would be ~2.3 GB).
    std::vector<float> compute_output_logits_chunked(const float* last_hidden) const;

    // Legacy helpers (forward_stub)
    std::vector<float> embed(const std::vector<uint32_t>& tokens);
    std::vector<float> apply_expert(uint32_t expert_id, const std::vector<float>& hidden);
};

} // namespace khanary::inference
