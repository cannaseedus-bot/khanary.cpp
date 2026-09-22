#include "kv_cache.h"
#include <stdexcept>

namespace khanary::inference {

using namespace khanary::gpu;

KVCache::KVCache(D3D11Device& dev, uint32_t n_layers, uint32_t d_head, uint32_t max_seq)
    : dev_(dev), n_layers_(n_layers), d_head_(d_head), max_seq_(max_seq)
{
    k_bufs_.resize(n_layers);
    v_bufs_.resize(n_layers);
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

void KVCache::clear() {
    seq_len_ = 0;
}

} // namespace khanary::inference
