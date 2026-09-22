#include "dml_ops.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace khanary::gpu {

DMLOps::DMLOps(D3D11Device& dev) : dev_(dev) {}
DMLOps::~DMLOps() {}

bool DMLOps::init() {
#ifdef KHANARY_HAS_DIRECTML
    // DirectML device init would go here
    return true;
#else
    return true; // CPU fallback always available
#endif
}

bool DMLOps::gemm(TensorBuffer* A, TensorBuffer* B, TensorBuffer* C,
                  uint32_t M, uint32_t K, uint32_t N,
                  float alpha, float beta)
{
    if (!A || !B || !C) return false;
    // CPU fallback: readback → compute → upload
    size_t szA = (size_t)M * K * 4;
    size_t szB = (size_t)K * N * 4;
    size_t szC = (size_t)M * N * 4;
    std::vector<float> ha(M * K), hb(K * N), hc(M * N);
    A->readback(ha.data(), szA);
    B->readback(hb.data(), szB);
    C->readback(hc.data(), szC);

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                sum += ha[i * K + k] * hb[k * N + j];
            hc[i * N + j] = alpha * sum + beta * hc[i * N + j];
        }
    }
    C->upload(hc.data(), szC);
    return true;
}

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
    std::vector<float> h(batch * dim), r(batch, 0.0f);
    src->readback(h.data(), batch * dim * 4);
    for (uint32_t b = 0; b < batch; ++b)
        for (uint32_t d = 0; d < dim; ++d)
            r[b] += h[b * dim + d];
    dst->upload(r.data(), batch * 4);
    return true;
}

bool DMLOps::attention(
    TensorBuffer* Q, TensorBuffer* K_buf, TensorBuffer* V, TensorBuffer* out,
    uint32_t seq_len, uint32_t d_head, uint32_t n_heads)
{
    if (!Q || !K_buf || !V || !out) return false;
    uint32_t d_model = d_head * n_heads;
    float scale = 1.0f / std::sqrt((float)d_head);

    std::vector<float> hQ(seq_len * d_model), hK(seq_len * d_model),
                       hV(seq_len * d_model), hO(seq_len * d_model, 0.0f);
    Q->readback(hQ.data(), hQ.size() * 4);
    K_buf->readback(hK.data(), hK.size() * 4);
    V->readback(hV.data(), hV.size() * 4);

    std::vector<float> scores(seq_len * seq_len);
    // QK^T * scale
    for (uint32_t i = 0; i < seq_len; ++i)
        for (uint32_t j = 0; j < seq_len; ++j) {
            float s = 0.0f;
            for (uint32_t d = 0; d < d_model; ++d)
                s += hQ[i * d_model + d] * hK[j * d_model + d];
            scores[i * seq_len + j] = s * scale;
        }
    // Softmax per row
    for (uint32_t i = 0; i < seq_len; ++i) {
        float mx = *std::max_element(scores.begin() + i * seq_len, scores.begin() + (i+1) * seq_len);
        float sm = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j) {
            scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - mx);
            sm += scores[i * seq_len + j];
        }
        for (uint32_t j = 0; j < seq_len; ++j) scores[i * seq_len + j] /= sm;
    }
    // scores @ V
    for (uint32_t i = 0; i < seq_len; ++i)
        for (uint32_t d = 0; d < d_model; ++d) {
            float v = 0.0f;
            for (uint32_t j = 0; j < seq_len; ++j)
                v += scores[i * seq_len + j] * hV[j * d_model + d];
            hO[i * d_model + d] = v;
        }
    out->upload(hO.data(), hO.size() * 4);
    return true;
}

} // namespace khanary::gpu
