#include "dml_ops.h"
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace khanary::gpu {

// ----------------------------------------------------------------
// HLSL_TILED — 16×16 groupshared tiled GEMM validated on HD 4600
// Computes C[M,N] = A[M,K] × B[K,N].  alpha=1, beta=0 only.
// ----------------------------------------------------------------
static const char* k_gemm_tiled_hlsl =
    "cbuffer P : register(b0) { uint M; uint K; uint N; uint pad; };\n"
    "StructuredBuffer<float>   A : register(t0);\n"
    "StructuredBuffer<float>   B : register(t1);\n"
    "RWStructuredBuffer<float> C : register(u0);\n"
    "#define TILE 16\n"
    "groupshared float As[TILE][TILE];\n"
    "groupshared float Bs[TILE][TILE];\n"
    "[numthreads(TILE, TILE, 1)]\n"
    "void CSMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {\n"
    "    uint row = gid.y * TILE + lid.y;\n"
    "    uint col = gid.x * TILE + lid.x;\n"
    "    float acc = 0.f;\n"
    "    uint nt = (K + TILE - 1) / TILE;\n"
    "    for (uint t = 0; t < nt; t++) {\n"
    "        uint ac = t * TILE + lid.x;\n"
    "        As[lid.y][lid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.f;\n"
    "        uint br = t * TILE + lid.y;\n"
    "        Bs[lid.y][lid.x] = (br < K && col < N) ? B[br * N + col] : 0.f;\n"
    "        GroupMemoryBarrierWithGroupSync();\n"
    "        [unroll] for (uint k = 0; k < TILE; k++) acc += As[lid.y][k] * Bs[k][lid.x];\n"
    "        GroupMemoryBarrierWithGroupSync();\n"
    "    }\n"
    "    if (row < M && col < N) C[row * N + col] = acc;\n"
    "}\n";

// ----------------------------------------------------------------
// Construction
// ----------------------------------------------------------------

DMLOps::DMLOps(D3D11Device& dev) : dev_(dev) {}
DMLOps::~DMLOps() {}

float DMLOps::semantic_attention_gate(float static_weight, float confidence, float relevance) {
    const float W = std::clamp(static_weight, 0.0f, 1.0f);
    const float C = std::clamp(confidence, 0.0f, 1.0f);
    const float R = std::clamp(relevance, 0.0f, 1.0f);
    return std::clamp(W * C * R, 0.0f, 1.0f);
}

bool DMLOps::init() {
    init_gemm_shader(); // non-fatal — falls back to CPU on failure
    return true;
}

// ----------------------------------------------------------------
// init_gemm_shader — compile HLSL_TILED and create cbuffer
// ----------------------------------------------------------------

bool DMLOps::init_gemm_shader() {
    if (!dev_.is_valid()) return false;
    ID3D11Device* d = dev_.GetDevice();
    if (!d) return false;

    ComPtr<ID3DBlob> blob, err_blob;
    HRESULT hr = D3DCompile(
        k_gemm_tiled_hlsl,
        strlen(k_gemm_tiled_hlsl),
        "gemm_tiled",
        nullptr, nullptr,
        "CSMain", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
        &blob, &err_blob);

    if (FAILED(hr)) return false;

    hr = d->CreateComputeShader(
        blob->GetBufferPointer(), blob->GetBufferSize(),
        nullptr, &gemm_cs_);
    if (FAILED(hr)) return false;

    // Constant buffer: { uint M, uint K, uint N, uint pad } = 16 bytes
    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth      = 16;
    cbd.Usage          = D3D11_USAGE_DEFAULT;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    hr = d->CreateBuffer(&cbd, nullptr, &gemm_cbuf_);
    if (FAILED(hr)) return false;

    gemm_cs_ready_ = true;
    return true;
}

// ----------------------------------------------------------------
// gemm — GPU tiled dispatch with CPU fallback
// ----------------------------------------------------------------

bool DMLOps::gemm(TensorBuffer* A, TensorBuffer* B, TensorBuffer* C,
                  uint32_t M, uint32_t K, uint32_t N,
                  float alpha, float beta)
{
    if (!A || !B || !C) return false;

    // GPU path: alpha=1, beta=0 only (kernel always writes acc, not alpha*acc + beta*C)
    if (gemm_cs_ready_ && alpha == 1.0f && beta == 0.0f) {
        ID3D11DeviceContext* ctx = dev_.GetContext();

        UINT params[4] = { static_cast<UINT>(M),
                           static_cast<UINT>(K),
                           static_cast<UINT>(N), 0u };
        ctx->UpdateSubresource(gemm_cbuf_.Get(), 0, nullptr, params, 0, 0);

        ctx->CSSetShader(gemm_cs_.Get(), nullptr, 0);
        ctx->CSSetConstantBuffers(0, 1, gemm_cbuf_.GetAddressOf());

        ID3D11ShaderResourceView* srvs[2] = { A->srv(), B->srv() };
        ctx->CSSetShaderResources(0, 2, srvs);

        ID3D11UnorderedAccessView* uavs[1] = { C->uav() };
        ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);

        UINT gx = (N + 15u) / 16u;
        UINT gy = (M + 15u) / 16u;
        ctx->Dispatch(gx, gy, 1u);

        // Unbind to avoid D3D11 warnings
        ID3D11ShaderResourceView*  null_srvs[2] = { nullptr, nullptr };
        ID3D11UnorderedAccessView* null_uavs[1] = { nullptr };
        ctx->CSSetShaderResources(0, 2, null_srvs);
        ctx->CSSetUnorderedAccessViews(0, 1, null_uavs, nullptr);
        ctx->CSSetShader(nullptr, nullptr, 0);

        return true;
    }

    // CPU fallback — also handles alpha≠1 or beta≠0
    std::vector<float> ha(static_cast<size_t>(M) * K);
    std::vector<float> hb(static_cast<size_t>(K) * N);
    std::vector<float> hc(static_cast<size_t>(M) * N, 0.0f);
    A->readback(ha.data(), ha.size() * 4);
    B->readback(hb.data(), hb.size() * 4);
    if (beta != 0.0f) C->readback(hc.data(), hc.size() * 4);

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                sum += ha[static_cast<size_t>(i) * K + k] * hb[static_cast<size_t>(k) * N + j];
            hc[static_cast<size_t>(i) * N + j] = alpha * sum + beta * hc[static_cast<size_t>(i) * N + j];
        }
    }
    C->upload(hc.data(), hc.size() * 4);
    return true;
}

// ----------------------------------------------------------------
// softmax / reduce_sum / attention — unchanged CPU implementations
// ----------------------------------------------------------------

bool DMLOps::softmax(TensorBuffer* src, TensorBuffer* dst, uint32_t count) {
    if (!src || !dst) return false;
    std::vector<float> h(count);
    src->readback(h.data(), count * 4);

    float max_v = *std::max_element(h.begin(), h.end());
    float sum = 0.0f;
    for (auto& v : h) { v = std::exp(v - max_v); sum += v; }
    for (auto& v : h) v /= sum;

    dst->upload(h.data(), count * 4);
    return true;
}

bool DMLOps::reduce_sum(TensorBuffer* src, TensorBuffer* dst, uint32_t batch, uint32_t dim) {
    if (!src || !dst) return false;
    std::vector<float> h(static_cast<size_t>(batch) * dim);
    std::vector<float> r(batch, 0.0f);
    src->readback(h.data(), h.size() * 4);
    for (uint32_t b = 0; b < batch; ++b)
        for (uint32_t d = 0; d < dim; ++d)
            r[b] += h[static_cast<size_t>(b) * dim + d];
    dst->upload(r.data(), batch * 4);
    return true;
}

bool DMLOps::attention(
    TensorBuffer* Q, TensorBuffer* K_buf, TensorBuffer* V, TensorBuffer* out,
    uint32_t seq_len, uint32_t d_head, uint32_t n_heads)
{
    if (!Q || !K_buf || !V || !out) return false;
    uint32_t d_model = d_head * n_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

    std::vector<float> hQ(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> hK(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> hV(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> hO(static_cast<size_t>(seq_len) * d_model, 0.0f);
    Q->readback(hQ.data(), hQ.size() * 4);
    K_buf->readback(hK.data(), hK.size() * 4);
    V->readback(hV.data(), hV.size() * 4);

    std::vector<float> scores(static_cast<size_t>(seq_len) * seq_len);
    for (uint32_t i = 0; i < seq_len; ++i)
        for (uint32_t j = 0; j < seq_len; ++j) {
            float s = 0.0f;
            for (uint32_t di = 0; di < d_model; ++di)
                s += hQ[static_cast<size_t>(i) * d_model + di]
                   * hK[static_cast<size_t>(j) * d_model + di];
            scores[static_cast<size_t>(i) * seq_len + j] = s * scale;
        }
    for (uint32_t i = 0; i < seq_len; ++i) {
        auto beg = scores.begin() + static_cast<ptrdiff_t>(i * seq_len);
        auto end = beg + static_cast<ptrdiff_t>(seq_len);
        float mx = *std::max_element(beg, end);
        float sm = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j) {
            scores[static_cast<size_t>(i) * seq_len + j] =
                std::exp(scores[static_cast<size_t>(i) * seq_len + j] - mx);
            sm += scores[static_cast<size_t>(i) * seq_len + j];
        }
        for (uint32_t j = 0; j < seq_len; ++j)
            scores[static_cast<size_t>(i) * seq_len + j] /= sm;
    }
    for (uint32_t i = 0; i < seq_len; ++i)
        for (uint32_t di = 0; di < d_model; ++di) {
            float v = 0.0f;
            for (uint32_t j = 0; j < seq_len; ++j)
                v += scores[static_cast<size_t>(i) * seq_len + j]
                   * hV[static_cast<size_t>(j) * d_model + di];
            hO[static_cast<size_t>(i) * d_model + di] = v;
        }
    out->upload(hO.data(), hO.size() * 4);
    return true;
}

} // namespace khanary::gpu
