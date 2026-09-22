#pragma once
#include "d3d11_device.h"
#include "tensor_buffer.h"
#include <string>
#include <vector>

namespace khanary::gpu {

// DirectML operator types
enum class DMLOpType { Gemm, Matmul, Conv, Attention, Softmax, Reduce };

struct DMLOpDesc {
    DMLOpType type;
    uint32_t  m = 0, n = 0, k = 0;  // for Gemm/Matmul
    uint32_t  heads = 0;             // for Attention
    float     alpha = 1.0f, beta = 0.0f;
};

// DMLOps wraps DirectML operator execution.
// When KHANARY_HAS_DIRECTML is not defined, falls back to CPU emulation.
class DMLOps {
public:
    explicit DMLOps(D3D11Device& dev);
    ~DMLOps();

    bool init();

    // GEMM: C = alpha * A*B + beta * C
    bool gemm(TensorBuffer* A, TensorBuffer* B, TensorBuffer* C,
              uint32_t M, uint32_t K, uint32_t N,
              float alpha = 1.0f, float beta = 0.0f);

    // Softmax along last dimension
    bool softmax(TensorBuffer* src, TensorBuffer* dst, uint32_t count);

    // Reduce (sum) along batch dimension
    bool reduce_sum(TensorBuffer* src, TensorBuffer* dst, uint32_t batch, uint32_t dim);

    // QKV Attention (simplified: QK^T/sqrt(d) → softmax → V)
    bool attention(
        TensorBuffer* Q, TensorBuffer* K, TensorBuffer* V, TensorBuffer* out,
        uint32_t seq_len, uint32_t d_head, uint32_t n_heads);

private:
    D3D11Device& dev_;

#ifdef KHANARY_HAS_DIRECTML
    // DirectML state would go here
    void* dml_device_ = nullptr;
#endif
};

} // namespace khanary::gpu
