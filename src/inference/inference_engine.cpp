#include "inference_engine.h"
#include "../config.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace khanary::inference {

InferenceEngine::InferenceEngine(gpu::D3D11Device& dev, InferenceConfig cfg)
    : dev_(dev), cfg_(cfg), dml_(dev)
{
    router_   = std::make_unique<MoERouter>(cfg.n_experts, cfg.d_model);
    kv_cache_ = std::make_unique<KVCache>(dev, cfg.n_layers, cfg.d_head, MAX_SEQ_LEN);
    expert_weights_.resize(cfg.n_experts);
    for (auto& w : expert_weights_)
        w.resize((size_t)cfg.d_model * cfg.d_model, 0.0f);
    dml_.init();
}

void InferenceEngine::load_expert_weights(uint32_t expert_id, const float* data, size_t count) {
    if (expert_id >= cfg_.n_experts) return;
    size_t cap = expert_weights_[expert_id].size();
    std::copy(data, data + std::min(count, cap), expert_weights_[expert_id].begin());
}

void InferenceEngine::clear_kv() { kv_cache_->clear(); }

std::vector<float> InferenceEngine::embed(const std::vector<uint32_t>& tokens) {
    // Simple sinusoidal embedding for bootstrapping
    size_t seq = tokens.size();
    std::vector<float> emb(seq * cfg_.d_model, 0.0f);
    for (size_t t = 0; t < seq; ++t) {
        uint32_t id = tokens[t];
        for (uint32_t d = 0; d < cfg_.d_model; ++d) {
            float pos = (float)t;
            float freq = 1.0f / std::pow(10000.0f, (float)(d / 2) / cfg_.d_model);
            emb[t * cfg_.d_model + d] = (d % 2 == 0)
                ? std::sin(pos * freq) + ((float)(id % 256) / 256.0f)
                : std::cos(pos * freq) + ((float)(id % 256) / 256.0f);
        }
    }
    return emb;
}

std::vector<float> InferenceEngine::apply_expert(uint32_t expert_id, const std::vector<float>& hidden) {
    // Simple linear transform: out = W * hidden
    uint32_t d = cfg_.d_model;
    std::vector<float> out(d, 0.0f);
    const auto& W = expert_weights_[expert_id];
    for (uint32_t i = 0; i < d; ++i) {
        float v = 0.0f;
        for (uint32_t j = 0; j < d; ++j)
            v += W[i * d + j] * hidden[j];
        out[i] = std::tanh(v); // activation
    }
    return out;
}

std::vector<float> InferenceEngine::forward(const std::vector<uint32_t>& tokens) {
    // Pop: embed
    pe_.set(kuhul::Phase::Pop);
    auto hidden = embed(tokens);

    // Take last token's hidden state for MoE routing
    size_t last_t = tokens.size() - 1;
    std::vector<float> last_h(hidden.begin() + last_t * cfg_.d_model,
                               hidden.begin() + (last_t + 1) * cfg_.d_model);

    // Wo: route to experts
    pe_.advance(); // → Wo
    auto allocs = router_->route(last_h, cfg_.top_k);

    // Yax: select
    pe_.advance(); // → Yax

    // Sek: compute via experts
    pe_.advance(); // → Sek
    std::vector<float> out(cfg_.d_model, 0.0f);
    for (const auto& alloc : allocs) {
        auto expert_out = apply_expert(alloc.expert_id, last_h);
        for (uint32_t d = 0; d < cfg_.d_model; ++d)
            out[d] += alloc.weight * expert_out[d];
        router_->record_dispatch(alloc.expert_id);
    }

    // Ch'en: verify (layer norm)
    pe_.advance(); // → Ch'en
    float mean = std::accumulate(out.begin(), out.end(), 0.0f) / out.size();
    float var  = 0.0f;
    for (float v : out) var += (v - mean) * (v - mean);
    var /= out.size();
    float std_inv = 1.0f / std::sqrt(var + 1e-6f);
    for (auto& v : out) v = (v - mean) * std_inv;

    // Project to vocab logits (d_model → vocab_size via simple projection)
    pe_.advance(); // → Xul
    std::vector<float> logits(GLYPH_VOCAB_SIZE, 0.0f);
    for (uint32_t v = 0; v < GLYPH_VOCAB_SIZE; ++v) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < cfg_.d_model && d < GLYPH_VOCAB_SIZE; ++d)
            dot += out[d] * std::sin((float)v * (float)d / GLYPH_VOCAB_SIZE);
        logits[v] = dot;
    }
    return logits;
}

InferenceResult InferenceEngine::run(const std::string& prompt) {
    InferenceResult result;
    result.ok = false;
    result.final_phase = kuhul::Phase::Pop;

    try {
        auto input_ids = tokenizer_.encode(prompt, true);
        std::vector<uint32_t> gen_ids;

        for (uint32_t step = 0; step < cfg_.max_tokens; ++step) {
            auto logits = forward(input_ids);
            uint32_t next_id = sampler_.sample(logits, cfg_.temperature, cfg_.top_p);

            if (next_id == GLYPH_EOS) break;

            gen_ids.push_back(next_id);
            input_ids.push_back(next_id);
            if (input_ids.size() > MAX_SEQ_LEN) {
                input_ids.erase(input_ids.begin()); // sliding window
            }

            if (token_cb_) {
                std::string tok = "[" + std::to_string(next_id) + "]";
                token_cb_(next_id, tok);
            }
        }

        result.token_ids   = gen_ids;
        result.text        = tokenizer_.decode(gen_ids);
        result.final_phase = pe_.current();
        result.ok          = true;
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

} // namespace khanary::inference
