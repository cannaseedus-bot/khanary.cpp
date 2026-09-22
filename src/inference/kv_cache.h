#pragma once
#include "../gpu/tensor_buffer.h"
#include <vector>
#include <cstdint>
#include <memory>

namespace khanary::inference {

// K/V cache: per-layer key and value tensor buffers
// Appends new K,V pairs as sequence grows
class KVCache {
public:
    KVCache(gpu::D3D11Device& dev, uint32_t n_layers, uint32_t d_head, uint32_t max_seq);

    // Append K, V for a given layer (host float arrays)
    void append(uint32_t layer,
                const float* k_data, const float* v_data,
                uint32_t seq_step, uint32_t d_head);

    gpu::TensorBuffer* k_buf(uint32_t layer);
    gpu::TensorBuffer* v_buf(uint32_t layer);

    uint32_t seq_len()  const { return seq_len_; }
    uint32_t n_layers() const { return n_layers_; }

    void clear();

private:
    gpu::D3D11Device& dev_;
    uint32_t n_layers_, d_head_, max_seq_;
    uint32_t seq_len_ = 0;

    // [n_layers] each holding [max_seq * d_head] floats
    std::vector<std::unique_ptr<gpu::TensorBuffer>> k_bufs_, v_bufs_;
};

} // namespace khanary::inference
