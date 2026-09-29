#include "gram_gpu_scorer.h"
#include "tensor_buffer.h"
#include "../kuhul/gram_store.h"
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace khanary::gpu {

// Embedded attention shader — identical to data/shaders/attention.hlsl.
static const char* k_attention_hlsl = R"HLSL(
cbuffer SemanticAttentionConstants : register(b0)
{
    uint  record_count;
    float admission_threshold;
    float relation_mix;
    float pad0;
};

struct SemanticRecord
{
    float W;
    float C;
    float R;
    float relation_weight;

    float fold_weight;
    float phase_weight;
    float policy_ok;
    float capable;

    float xcfe_ok;
    float reserved0;
    float reserved1;
    float reserved2;
};

StructuredBuffer<SemanticRecord> Records : register(t0);
RWStructuredBuffer<float> BaseScore     : register(u0);
RWStructuredBuffer<float> SemanticScore : register(u1);
RWStructuredBuffer<uint>  LegalMask     : register(u2);
RWStructuredBuffer<uint>  AdmissionMask : register(u3);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= record_count) return;

    SemanticRecord rec = Records[idx];

    float w    = saturate(rec.W);
    float c    = saturate(rec.C);
    float r    = saturate(rec.R);
    float rel  = saturate(rec.relation_weight);
    float fold = saturate(rec.fold_weight);
    float phase = saturate(rec.phase_weight);

    float s = w * c * r;
    BaseScore[idx] = s;

    float semantic_modifier = lerp(1.0f, rel * fold * phase, saturate(relation_mix));
    float weighted = s * semantic_modifier;

    uint legal = (rec.policy_ok >= 0.5f && rec.capable >= 0.5f && rec.xcfe_ok >= 0.5f) ? 1u : 0u;
    LegalMask[idx] = legal;

    float score = (legal == 1u) ? weighted : 0.0f;
    SemanticScore[idx] = score;
    AdmissionMask[idx] = (legal == 1u && score >= admission_threshold) ? 1u : 0u;
}
)HLSL";

// ──────────────────────────────────────────────────────────────────────────────

GramGpuScorer::GramGpuScorer(D3D11Device& dev) : dev_(dev) {}

bool GramGpuScorer::init() {
    if (!dev_.is_valid()) return false;

    ComPtr<ID3DBlob> blob, errors;
    HRESULT hr = D3DCompile(
        k_attention_hlsl, strlen(k_attention_hlsl),
        "attention.hlsl", nullptr, nullptr,
        "main", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
        &blob, &errors);
    if (FAILED(hr)) return false;

    hr = dev_.GetDevice()->CreateComputeShader(
        blob->GetBufferPointer(), blob->GetBufferSize(),
        nullptr, &cs_);
    if (FAILED(hr)) return false;

    D3D11_BUFFER_DESC cbd{};
    cbd.ByteWidth      = 16;
    cbd.Usage          = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = dev_.GetDevice()->CreateBuffer(&cbd, nullptr, &cbuf_);
    if (FAILED(hr)) return false;

    ready_ = true;
    return true;
}

float GramGpuScorer::channel_w(const std::string& ch) {
    if (ch == "similarity_kernel")    return 0.90f;
    if (ch == "xcfe_control_program") return 0.85f;
    if (ch == "predication_sequence") return 0.75f;
    if (ch == "taxonomy_geometry")    return 0.65f;
    return 0.50f;
}

float GramGpuScorer::authority_c(const std::string& auth) {
    if (auth == "xcfe_control_program") return 0.90f;
    if (auth == "TOOL-\xc2\xb5")        return 0.80f;  // TOOL-µ (UTF-8)
    if (auth == "ELIZA")                 return 0.75f;
    if (auth == "Xul")                   return 0.70f;
    return 0.60f;
}

std::vector<GramScoreResult> GramGpuScorer::score_batch(
    const kuhul::GramStore& store,
    const std::vector<std::string>& candidate_keys,
    const std::vector<float>&       relevance,
    float admission_threshold,
    float relation_mix)
{
    std::vector<GramScoreResult> out;
    if (!ready_) return out;

    const uint32_t N = static_cast<uint32_t>(
        std::min(candidate_keys.size(), relevance.size()));
    if (N == 0) return out;

    // CPU-side struct matching SemanticRecord in the shader (12 floats = 48 bytes).
    struct alignas(4) SemanticRecordCPU {
        float W, C, R, relation_weight;
        float fold_weight, phase_weight, policy_ok, capable;
        float xcfe_ok, r0, r1, r2;
    };
    static_assert(sizeof(SemanticRecordCPU) == 48,
        "SemanticRecordCPU must match HLSL SemanticRecord layout");

    std::vector<SemanticRecordCPU> recs(N);
    for (uint32_t i = 0; i < N; ++i) {
        const auto& key = candidate_keys[i];
        auto g = store.gramRead(key);
        auto& r = recs[i];

        r.W = g ? channel_w(g->channel)     : 0.50f;
        r.C = g ? authority_c(g->authority)  : 0.60f;
        r.R = std::clamp(relevance[i], 0.0f, 1.0f);

        // Geodesic packing from gramVector() (x, y, z) = fold-manifold coordinates.
        //   fold_weight     = vector magnitude        (geodesic reach from manifold origin)
        //   relation_weight = equatorial component    sqrt(x²+y²)/mag  (in-plane phase alignment)
        //   phase_weight    = vertical component      0.5 + 0.5*(z/mag) mapped to [0,1]
        r.fold_weight = r.relation_weight = r.phase_weight = 0.5f;
        if (g) {
            const auto vec = store.gramVector(key);
            if (vec) {
                const float vx = (*vec)[0], vy = (*vec)[1], vz = (*vec)[2];
                const float mag = std::sqrt(vx*vx + vy*vy + vz*vz);
                if (mag > 1e-6f) {
                    r.fold_weight     = std::clamp(mag, 0.0f, 1.0f);
                    r.relation_weight = std::clamp(
                        std::sqrt(vx*vx + vy*vy) / mag, 0.0f, 1.0f);
                    r.phase_weight    = std::clamp(
                        0.5f + 0.5f * (vz / mag), 0.0f, 1.0f);
                }
            }
        }

        // Gates are open — caller controls admission via admission_threshold.
        r.policy_ok = 1.0f;
        r.capable   = 1.0f;
        r.xcfe_ok   = 1.0f;
        r.r0 = r.r1 = r.r2 = 0.0f;
    }

    // GPU buffers: input records (SRV t0), four output channels (UAV u0–u3).
    TensorBuffer buf_records { dev_, { "gram_records",  TensorFormat::F32, N, 48 } };
    TensorBuffer buf_base    { dev_, { "gram_base",     TensorFormat::F32, N,  4 } };
    TensorBuffer buf_sem     { dev_, { "gram_semantic", TensorFormat::F32, N,  4 } };
    TensorBuffer buf_legal   { dev_, { "gram_legal",    TensorFormat::F32, N,  4 } };
    TensorBuffer buf_admitted{ dev_, { "gram_admitted", TensorFormat::F32, N,  4 } };

    if (!buf_records.is_valid() || !buf_base.is_valid() ||
        !buf_sem.is_valid()     || !buf_legal.is_valid() || !buf_admitted.is_valid())
        return out;

    if (!buf_records.upload(recs.data(), N * sizeof(SemanticRecordCPU)))
        return out;

    // Update cbuffer (must be 16 bytes, DYNAMIC+WRITE_DISCARD).
    {
        struct CB { uint32_t record_count; float admission_threshold; float relation_mix; float pad0; };
        const CB cb{ N, admission_threshold, relation_mix, 0.0f };
        auto* ctx = dev_.GetContext();
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(ctx->Map(cbuf_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            std::memcpy(mapped.pData, &cb, sizeof(cb));
            ctx->Unmap(cbuf_.Get(), 0);
        }
    }

    // Bind, dispatch, unbind.
    auto* ctx = dev_.GetContext();
    ctx->CSSetShader(cs_.Get(), nullptr, 0);

    auto* raw_cb = cbuf_.Get();
    ctx->CSSetConstantBuffers(0, 1, &raw_cb);

    auto* rec_srv = buf_records.srv();
    ctx->CSSetShaderResources(0, 1, &rec_srv);

    ID3D11UnorderedAccessView* uavs[4] = {
        buf_base.uav(), buf_sem.uav(), buf_legal.uav(), buf_admitted.uav()
    };
    ctx->CSSetUnorderedAccessViews(0, 4, uavs, nullptr);

    ctx->Dispatch((N + 63u) / 64u, 1, 1);

    // Unbind in reverse order to avoid D3D debug warnings.
    static ID3D11UnorderedAccessView* const kNullUAVs[4] = {};
    static ID3D11ShaderResourceView*  const kNullSRVs[1] = {};
    ctx->CSSetUnorderedAccessViews(0, 4, kNullUAVs, nullptr);
    ctx->CSSetShaderResources(0, 1, kNullSRVs);
    ctx->CSSetShader(nullptr, nullptr, 0);

    // Readback all four output channels.
    std::vector<float>    base_scores(N), sem_scores(N);
    std::vector<uint32_t> legal_flags(N), admitted_flags(N);
    buf_base.readback(base_scores.data(),      N * sizeof(float));
    buf_sem.readback(sem_scores.data(),        N * sizeof(float));
    buf_legal.readback(legal_flags.data(),     N * sizeof(uint32_t));
    buf_admitted.readback(admitted_flags.data(), N * sizeof(uint32_t));

    out.reserve(N);
    for (uint32_t i = 0; i < N; ++i) {
        out.push_back({
            candidate_keys[i],
            sem_scores[i],
            base_scores[i],
            admitted_flags[i] != 0u
        });
    }

    // Admitted entries first; within each tier, descending by semantic_score.
    std::sort(out.begin(), out.end(), [](const GramScoreResult& a, const GramScoreResult& b) {
        if (a.admitted != b.admitted) return a.admitted > b.admitted;
        return a.semantic_score > b.semantic_score;
    });

    return out;
}

} // namespace khanary::gpu
