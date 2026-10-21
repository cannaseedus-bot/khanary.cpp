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
    // GPU path: D3D11 HLSL_TILED (16×16 groupshared, validated 27.3 GFLOPS on HD 4600)
    // CPU fallback: used when alpha≠1, beta≠0, or shader not compiled.
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

    // Semantic-attention control gate used by KHANARY runtime routing.
    static float semantic_attention_gate(float static_weight, float confidence, float relevance);

private:
    D3D11Device& dev_;

    // D3D11 tiled GEMM compute shader state
    ComPtr<ID3D11ComputeShader> gemm_cs_;
    ComPtr<ID3D11Buffer>         gemm_cbuf_; // 16-byte cbuffer: {M, K, N, pad}
    bool gemm_cs_ready_ = false;
    bool init_gemm_shader();

    // D3D11 QKV attention compute shader state
    // GPU path: non-causal self-attention, seq_len ≤ kAttnMaxSeq, alpha=1/sqrt(d_head).
    // CPU fallback used when shader not compiled or seq_len exceeds the cap.
    // GPU parity run pending — no D3D11 device in test environment.
    ComPtr<ID3D11ComputeShader> attn_cs_;
    ComPtr<ID3D11Buffer>         attn_cbuf_; // 16-byte cbuffer: {seq_len, d_model, scale, pad}
    bool attn_cs_ready_ = false;
    bool init_attn_shader();
    static constexpr uint32_t kAttnMaxSeq = 512;

#ifdef KHANARY_HAS_OPENCL
    // OpenCL bridge (dynamically loaded from OpenCL.dll ICD)
    // Activated when KHANARY_HAS_OPENCL is defined at build time.
    void* ocl_platform_ = nullptr;
    void* ocl_device_   = nullptr;
    void* ocl_context_  = nullptr;
    void* ocl_queue_    = nullptr;
    bool  ocl_ready_    = false;
    bool  init_opencl();
#endif

#ifdef KHANARY_HAS_DIRECTML
    void* dml_device_ = nullptr;
#endif
};

} // namespace khanary::gpu
