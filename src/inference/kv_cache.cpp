#include "kv_cache.h"
#include <stdexcept>
#include <algorithm>

namespace khanary::inference {

using namespace khanary::gpu;

KVCache::KVCache(D3D11Device& dev, uint32_t n_layers, uint32_t d_head, uint32_t max_seq)
    : dev_(dev), n_layers_(n_layers), d_head_(d_head), max_seq_(max_seq)
{
    k_bufs_.resize(n_layers);
    v_bufs_.resize(n_layers);
    step_attention_scores_.assign(max_seq_, 0.0f);
    for (uint32_t l = 0; l < n_layers; ++l) {
        TensorDesc desc;
        desc.name       = "kv_k_" + std::to_string(l);
        desc.format     = TensorFormat::F32;
        desc.elem_count = max_seq * d_head;
        k_bufs_[l] = std::make_unique<TensorBuffer>(dev_, desc);
        desc.name       = "kv_v_" + std::to_string(l);
        v_bufs_[l] = std::make_unique<TensorBuffer>(dev_, desc);
    }
}

void KVCache::append(uint32_t layer,
                     const float* k_data, const float* v_data,
                     uint32_t seq_step, uint32_t d_head)
{
    if (layer >= n_layers_) return;
    if (seq_len_ >= max_seq_) return;

    size_t chunk_bytes  = (size_t)seq_step * d_head * sizeof(float);

    // We can't partial-upload into D3D11 easily without staging tricks.
    // For now, download full buffer, patch, re-upload.
    std::vector<float> buf(max_seq_ * d_head, 0.0f);
    k_bufs_[layer]->readback(buf.data(), buf.size() * 4);
    memcpy(buf.data() + seq_len_ * d_head, k_data, chunk_bytes);
    k_bufs_[layer]->upload(buf.data(), buf.size() * 4);

    v_bufs_[layer]->readback(buf.data(), buf.size() * 4);
    memcpy(buf.data() + seq_len_ * d_head, v_data, chunk_bytes);
    v_bufs_[layer]->upload(buf.data(), buf.size() * 4);

    for (uint32_t i = 0; i < seq_step && (seq_len_ + i) < max_seq_; ++i)
        step_attention_scores_[seq_len_ + i] = 1.0f;
    seq_len_ += seq_step;
}

TensorBuffer* KVCache::k_buf(uint32_t layer) {
    if (layer >= n_layers_) return nullptr;
    return k_bufs_[layer].get();
}

TensorBuffer* KVCache::v_buf(uint32_t layer) {
    if (layer >= n_layers_) return nullptr;
    return v_bufs_[layer].get();
}

std::optional<float> KVCache::attention_score(uint32_t step) const {
    if (step >= seq_len_ || step >= step_attention_scores_.size()) return std::nullopt;
    return step_attention_scores_[step];
}

void KVCache::record_attention_score(float score) {
    if (seq_len_ == 0 || step_attention_scores_.empty()) return;
    const uint32_t idx = seq_len_ - 1;
    if (idx >= step_attention_scores_.size()) return;
    step_attention_scores_[idx] = std::clamp(score, 0.0f, 1.0f);
}

void KVCache::clear() {
    seq_len_ = 0;
    std::fill(step_attention_scores_.begin(), step_attention_scores_.end(), 0.0f);
}

} // namespace khanary::inference
