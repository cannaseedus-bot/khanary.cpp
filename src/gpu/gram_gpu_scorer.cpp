#include "gram_gpu_scorer.h"
#include "tensor_buffer.h"
#include "../kuhul/gram_store.h"
#include <d3dcompiler.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>

namespace khanary::gpu {

// Embedded attention shader — identical to data/shaders/attention.hlsl.
static const char* k_attention_hlsl = R"HLSL(
// KHANARY attention.hlsl
// Semantic attention kernel (Yax execution backend), not transformer QKV.
//
// Boundary:
//   Bot Semantics -> Shader Attention -> Attended Semantics
//
// Score law:
//   S_i = W_i * C_i * R_i
//
// Policy law (hard gate, not score shaping):
//   legal_i = capable_i AND policy_ok_i AND xcfe_ok_i
//
// Admission:
//   A_i = 1 when semantic_score_i >= threshold and legal_i == 1
//       = 0 otherwise

cbuffer SemanticAttentionConstants : register(b0)
{
    uint  record_count;
    float admission_threshold;
    float relation_mix;  // [0,1]: blend relation/fold/phase modifiers into S
    float tensor_mix;    // [0,1]: blend micronaut tensor-slot modifier into S

    float geodesic_mix;  // [0,1]: blend optional STB geodesic sidecar modifier into S
    float beta_diffusion; // β^-1: scales geodesic entropy diffusion when sidecar is active
    float reserved_c0;
    float reserved_c1;
};

// Optional frame-transition tensor.  This is a projection-only coordinate
// transform over the tensor slot four-vector:
//     u^i = L^i_j v^j
// The active WCR and policy/XCFE gates remain outside this contraction.
cbuffer LorentzTransitionTensor : register(b1)
{
    float4 boost_row0; // L^0_j
    float4 boost_row1; // L^1_j
    float4 boost_row2; // L^2_j
    float4 boost_row3; // L^3_j

    uint  lorentz_enabled;
    float boost_alignment_mix;
    float boost_mass_scale;
    float boost_reserved;
};

struct SemanticRecord
{
    float W;   // static/pattern-strength channel
    float C;   // confidence channel
    float R;   // relevance channel
    float relation_weight; // neighborhood/relation weighting

    float fold_weight;     // fold compatibility weighting
    float phase_weight;    // phase compatibility weighting
    float policy_ok;       // 1.0 if policy requirement satisfied, else 0.0
    float capable;         // 1.0 if resolver has required capabilities, else 0.0

    float xcfe_ok;         // 1.0 if XCFE authorization holds, else 0.0
    float tensor_mass;      // optional Micronaut slot/STB attraction mass
    float tensor_alignment; // optional Q-to-slot/fiber alignment in [0,1]
    float tensor_health;    // optional projection/runtime health in [0,1]

    float tensor_enabled;   // 1.0 when tensor_* channels are meaningful
    float reserved0;
    float reserved1;
    float reserved2;
};

struct GeodesicMetricRecord
{
    float x;
    float y;
    float z;
    float local_entropy;

    float geodesic_entropy;
    float unruh_temp;
    float divergence;
    float circumradius_R;

    float inradius_r;
};

StructuredBuffer<SemanticRecord> Records : register(t0);
StructuredBuffer<GeodesicMetricRecord> GeodesicMetrics : register(t1);

// BaseScore: raw S_i = W*C*R before relation/fold/phase modifiers.
RWStructuredBuffer<float> BaseScore : register(u0);
// SemanticScore: gated/weighted score used for ranking within legal candidates.
RWStructuredBuffer<float> SemanticScore : register(u1);
// LegalMask: 1 if capability+policy+XCFE all satisfied, else 0.
RWStructuredBuffer<uint> LegalMask : register(u2);
// AdmissionMask: 1 if admitted by threshold and legal gate, else 0.
RWStructuredBuffer<uint> AdmissionMask : register(u3);

float4 tensor_four_vector(SemanticRecord rec)
{
    // x0 is the timelike attraction mass.  reserved0..2 carry x1,x2,x3
    // from HyperbolicAttentionRecord64 without changing the 64-byte stride.
    return float4(rec.tensor_mass, rec.reserved0, rec.reserved1, rec.reserved2);
}

float4 lorentz_contract(float4 v)
{
    if (lorentz_enabled < 1u)
        return v;

    float4 u;
    // Einstein contraction: u^i = L^i_j v^j. Each dot product contracts j.
    u.x = dot(boost_row0, v);
    u.y = dot(boost_row1, v);
    u.z = dot(boost_row2, v);
    u.w = dot(boost_row3, v);
    return u;
}

float tensor_slot_modifier(SemanticRecord rec)
{
    if (rec.tensor_enabled < 0.5f)
        return 1.0f;

    // Tensor slots are projection/fiber state.  They may bias attention, but
    // they do not verify semantic truth and do not replace policy/XCFE gates.
    float mass = saturate(rec.tensor_mass);
    float alignment = saturate(rec.tensor_alignment);

    if (lorentz_enabled >= 1u) {
        float4 u = lorentz_contract(tensor_four_vector(rec));
        float spatial2 = dot(u.yzw, u.yzw);
        float interval = -u.x * u.x + spatial2;
        float boosted_alignment = (interval <= 0.0f) ? 1.0f : saturate(1.0f / (1.0f + interval));
        mass = saturate(abs(u.x) * max(boost_mass_scale, 0.0f));
        alignment = lerp(alignment, boosted_alignment, saturate(boost_alignment_mix));
    }

    return mass *
           alignment *
           saturate(rec.tensor_health);
}

float geodesic_sidecar_modifier(uint idx)
{
    if (geodesic_mix < 0.5f)
        return 1.0f;

    GeodesicMetricRecord geo = GeodesicMetrics[idx];

    // Entropic sink pull: divergence < 0 means the local field behaves like a
    // coordinate sink.  This is a projection modifier only; Ch'en/Xul still own
    // verification and persistence.
    float entropy_density = saturate(geo.local_entropy +
                                     saturate(beta_diffusion) * geo.geodesic_entropy);
    float entropic_pull = saturate(geo.unruh_temp * entropy_density);
    float sink_mix = saturate(-geo.divergence);
    float gravity_warp = lerp(1.0f, entropic_pull, sink_mix);

    // Euler triangle inequality.  Zero radii mean "not supplied" and remain
    // neutral.  If both radii are present and R < 2r, the local mesh is stressed.
    float mesh_supplied = (geo.circumradius_R > 0.0f && geo.inradius_r > 0.0f) ? 1.0f : 0.0f;
    float mesh_ok = (geo.circumradius_R >= 2.0f * geo.inradius_r) ? 1.0f : 0.1f;
    float mesh_stability = lerp(1.0f, mesh_ok, mesh_supplied);

    return saturate(gravity_warp * mesh_stability);
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= record_count) return;

    SemanticRecord rec = Records[idx];

    float w = saturate(rec.W);
    float c = saturate(rec.C);
    float r = saturate(rec.R);
    float rel = saturate(rec.relation_weight);
    float fold = saturate(rec.fold_weight);
    float phase = saturate(rec.phase_weight);

    float s = w * c * r;
    BaseScore[idx] = s;

    // Semantic weighting layer (still symbolic channels, not token tensors).
    float semantic_modifier = lerp(1.0f, rel * fold * phase, saturate(relation_mix));
    float tensor_modifier = tensor_slot_modifier(rec);
    float geodesic_modifier = geodesic_sidecar_modifier(idx);
    float weighted = s *
                     semantic_modifier *
                     lerp(1.0f, tensor_modifier, saturate(tensor_mix)) *
                     lerp(1.0f, geodesic_modifier, saturate(geodesic_mix));

    // Hard legality gate: policy constrains eligibility, not ranking algebra.
    uint legal = (rec.policy_ok >= 0.5f && rec.capable >= 0.5f && rec.xcfe_ok >= 0.5f) ? 1u : 0u;
    LegalMask[idx] = legal;

    float score = (legal == 1u) ? weighted : 0.0f;
    SemanticScore[idx] = score;
    AdmissionMask[idx] = (legal == 1u && score >= admission_threshold) ? 1u : 0u;
}

)HLSL";

// ──────────────────────────────────────────────────────────────────────────────

namespace {

float finite_or(float v, float fallback) {
    return std::isfinite(v) ? v : fallback;
}

float clamp01(float v) {
    return std::clamp(finite_or(v, 0.0f), 0.0f, 1.0f);
}

struct LorentzCB {
    float row0[4];
    float row1[4];
    float row2[4];
    float row3[4];
    uint32_t enabled;
    float alignment_mix;
    float mass_scale;
    float reserved;
};
static_assert(sizeof(LorentzCB) == 80,
              "LorentzCB must match LorentzTransitionTensor cbuffer");

LorentzCB make_lorentz_cb(const LorentzBoostRows* boost) {
    const LorentzBoostRows identity{};
    const LorentzBoostRows& src = boost ? *boost : identity;
    LorentzCB cb{};
    for (size_t i = 0; i < 4; ++i) {
        cb.row0[i] = finite_or(src.row0[i], identity.row0[i]);
        cb.row1[i] = finite_or(src.row1[i], identity.row1[i]);
        cb.row2[i] = finite_or(src.row2[i], identity.row2[i]);
        cb.row3[i] = finite_or(src.row3[i], identity.row3[i]);
    }
    cb.enabled = src.enabled ? 1u : 0u;
    cb.alignment_mix = clamp01(src.alignment_mix);
    cb.mass_scale = std::max(0.0f, finite_or(src.mass_scale, 1.0f));
    cb.reserved = 0.0f;
    return cb;
}

} // namespace

LorentzBoostRows LorentzBoostRows::make_pure(float rapidity_rho,
                                              std::array<float, 3> direction,
                                              float alignment_mix_value,
                                              float mass_scale_value)
{
    LorentzBoostRows rows;
    if (!std::isfinite(rapidity_rho) || rapidity_rho < 0.0f)
        return rows;

    float norm_sq = direction[0] * direction[0] +
                    direction[1] * direction[1] +
                    direction[2] * direction[2];
    if (!std::isfinite(norm_sq) || norm_sq <= 1e-12f)
        return rows;

    if (std::fabs(norm_sq - 1.0f) > 1e-6f) {
        const float inv = 1.0f / std::sqrt(norm_sq);
        direction[0] *= inv;
        direction[1] *= inv;
        direction[2] *= inv;
    }

    const float ch = std::cosh(rapidity_rho);
    const float sh = std::sinh(rapidity_rho);
    const float nx = direction[0];
    const float ny = direction[1];
    const float nz = direction[2];

    rows.row0 = {ch, sh * nx, sh * ny, sh * nz};
    rows.row1 = {sh * nx,
                 1.0f + (ch - 1.0f) * nx * nx,
                 (ch - 1.0f) * nx * ny,
                 (ch - 1.0f) * nx * nz};
    rows.row2 = {sh * ny,
                 (ch - 1.0f) * ny * nx,
                 1.0f + (ch - 1.0f) * ny * ny,
                 (ch - 1.0f) * ny * nz};
    rows.row3 = {sh * nz,
                 (ch - 1.0f) * nz * nx,
                 (ch - 1.0f) * nz * ny,
                 1.0f + (ch - 1.0f) * nz * nz};
    rows.enabled = true;
    rows.alignment_mix = clamp01(alignment_mix_value);
    rows.mass_scale = std::max(0.0f, finite_or(mass_scale_value, 1.0f));
    return rows;
}

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
    cbd.ByteWidth      = 32;
    cbd.Usage          = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = dev_.GetDevice()->CreateBuffer(&cbd, nullptr, &cbuf_);
    if (FAILED(hr)) return false;

    D3D11_BUFFER_DESC lbd{};
    lbd.ByteWidth      = sizeof(LorentzCB);
    lbd.Usage          = D3D11_USAGE_DYNAMIC;
    lbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    lbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = dev_.GetDevice()->CreateBuffer(&lbd, nullptr, &lorentz_cbuf_);
    if (FAILED(hr)) return false;

    ready_ = true;
    return true;
}

float GramGpuScorer::channel_w(const std::string& ch) {
    if (ch == "similarity_kernel")              return 0.90f;
    if (ch == "control_flow_operator")          return 0.88f;
    if (ch == "tool_dispatch_channel")          return 0.86f;
    if (ch == "xcfe_control_program")           return 0.85f;
    if (ch == "context_association_evidence")   return 0.78f;
    if (ch == "predication_sequence")           return 0.75f;
    if (ch == "coarse_ngram_structure")         return 0.70f;
    if (ch == "vector_projection")              return 0.70f;
    if (ch == "taxonomy_geometry")              return 0.65f;
    if (ch == "semantic_address_layer")         return 0.64f;
    if (ch == "lexicon_unigram")                return 0.62f;
    if (ch == "function_word_channel")          return 0.58f;
    return 0.50f;
}

float GramGpuScorer::authority_c(const std::string& auth) {
    if (auth == "xcfe_control_program")    return 0.90f;
    if (auth == "TOOL-\xc2\xb5")           return 0.80f;  // TOOL-µ (UTF-8)
    if (auth == "ELIZA")                    return 0.75f;
    if (auth == "Xul")                      return 0.70f;
    if (auth == "vector:projection-only")   return 0.55f;
    return 0.60f;
}

std::vector<GramScoreResult> GramGpuScorer::score_batch(
    const kuhul::GramStore& store,
    const std::vector<std::string>& candidate_keys,
    const std::vector<float>&       relevance,
    float admission_threshold,
    float relation_mix,
    float tensor_mix,
    const std::vector<STBGeodesicRecord>* geodesic_metrics,
    float geodesic_mix,
    float beta_diffusion,
    const LorentzBoostRows* lorentz_boost,
    const std::unordered_map<std::string, std::array<float, 3>>* vector_overrides)
{
    std::vector<GramScoreResult> out;
    if (!ready_) return out;

    const uint32_t N = static_cast<uint32_t>(
        std::min(candidate_keys.size(), relevance.size()));
    if (N == 0) return out;

    // CPU-side struct matching SemanticRecord in the shader (16 floats = 64 bytes).
    struct alignas(4) SemanticRecordCPU {
        float W, C, R, relation_weight;
        float fold_weight, phase_weight, policy_ok, capable;
        float xcfe_ok, tensor_mass, tensor_alignment, tensor_health;
        float tensor_enabled, r0, r1, r2;
    };
    static_assert(sizeof(SemanticRecordCPU) == 64,
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
        std::optional<std::array<float, 3>> vec;
        if (vector_overrides) {
            const auto it = vector_overrides->find(key);
            if (it != vector_overrides->end())
                vec = it->second;
        }
        if (!vec && g)
            vec = store.gramVector(key);

        bool has_tensor_geometry = false;
        float vx = 0.0f, vy = 0.0f, vz = 0.0f;
        if (vec) {
            vx = finite_or((*vec)[0], 0.0f);
            vy = finite_or((*vec)[1], 0.0f);
            vz = finite_or((*vec)[2], 0.0f);
            const float mag = std::sqrt(vx*vx + vy*vy + vz*vz);
            if (std::isfinite(mag)) {
                has_tensor_geometry = true;
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
        if (has_tensor_geometry) {
            const float spatial2 = vx*vx + vy*vy + vz*vz;
            r.tensor_mass = std::sqrt(1.0f + spatial2); // hyperboloid x0
            r.tensor_alignment = 1.0f;
            r.tensor_health = 1.0f;
            r.tensor_enabled = 1.0f;
            r.r0 = vx;
            r.r1 = vy;
            r.r2 = vz;
        } else {
            r.tensor_mass = 1.0f;
            r.tensor_alignment = 1.0f;
            r.tensor_health = 1.0f;
            r.tensor_enabled = 0.0f;
            r.r0 = r.r1 = r.r2 = 0.0f;
        }
    }

    std::vector<STBGeodesicRecord> neutral_geodesics;
    const std::vector<STBGeodesicRecord>* geodesics = geodesic_metrics;
    if (!geodesics || geodesics->size() < N) {
        neutral_geodesics.resize(N);
        geodesics = &neutral_geodesics;
        geodesic_mix = 0.0f;
    }

    // GPU buffers: input records (SRV t0), geodesic sidecar (SRV t1),
    // four output channels (UAV u0–u3).
    TensorBuffer buf_records { dev_, { "gram_records",  TensorFormat::F32, N, 64 } };
    TensorBuffer buf_geodesic{ dev_, { "gram_geodesic", TensorFormat::F32, N, 36 } };
    TensorBuffer buf_base    { dev_, { "gram_base",     TensorFormat::F32, N,  4 } };
    TensorBuffer buf_sem     { dev_, { "gram_semantic", TensorFormat::F32, N,  4 } };
    TensorBuffer buf_legal   { dev_, { "gram_legal",    TensorFormat::F32, N,  4 } };
    TensorBuffer buf_admitted{ dev_, { "gram_admitted", TensorFormat::F32, N,  4 } };

    if (!buf_records.is_valid() || !buf_geodesic.is_valid() || !buf_base.is_valid() ||
        !buf_sem.is_valid()     || !buf_legal.is_valid() || !buf_admitted.is_valid())
        return out;

    if (!buf_records.upload(recs.data(), N * sizeof(SemanticRecordCPU)))
        return out;
    if (!buf_geodesic.upload(geodesics->data(), N * sizeof(STBGeodesicRecord)))
        return out;

    // Update cbuffer (must be 32 bytes, DYNAMIC+WRITE_DISCARD).
    {
        struct CB {
            uint32_t record_count;
            float admission_threshold;
            float relation_mix;
            float tensor_mix;
            float geodesic_mix;
            float beta_diffusion;
            float reserved_c0;
            float reserved_c1;
        };
        const CB cb{ N, admission_threshold, relation_mix, tensor_mix,
                     geodesic_mix, beta_diffusion, 0.0f, 0.0f };
        auto* ctx = dev_.GetContext();
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(ctx->Map(cbuf_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            std::memcpy(mapped.pData, &cb, sizeof(cb));
            ctx->Unmap(cbuf_.Get(), 0);
        }
    }
    // Update Lorentz transition cbuffer.  Defaults to disabled identity rows so
    // existing callers keep their scalar tensor-slot behavior.
    {
        const LorentzCB cb = make_lorentz_cb(lorentz_boost);
        auto* ctx = dev_.GetContext();
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(ctx->Map(lorentz_cbuf_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            std::memcpy(mapped.pData, &cb, sizeof(cb));
            ctx->Unmap(lorentz_cbuf_.Get(), 0);
        }
    }

    // Bind, dispatch, unbind.
    auto* ctx = dev_.GetContext();
    ctx->CSSetShader(cs_.Get(), nullptr, 0);

    ID3D11Buffer* raw_cbs[2] = { cbuf_.Get(), lorentz_cbuf_.Get() };
    ctx->CSSetConstantBuffers(0, 2, raw_cbs);

    ID3D11ShaderResourceView* srvs[2] = {
        buf_records.srv(), buf_geodesic.srv()
    };
    ctx->CSSetShaderResources(0, 2, srvs);

    ID3D11UnorderedAccessView* uavs[4] = {
        buf_base.uav(), buf_sem.uav(), buf_legal.uav(), buf_admitted.uav()
    };
    ctx->CSSetUnorderedAccessViews(0, 4, uavs, nullptr);

    ctx->Dispatch((N + 63u) / 64u, 1, 1);

    // Unbind in reverse order to avoid D3D debug warnings.
    static ID3D11UnorderedAccessView* const kNullUAVs[4] = {};
    static ID3D11ShaderResourceView*  const kNullSRVs[2] = {};
    ctx->CSSetUnorderedAccessViews(0, 4, kNullUAVs, nullptr);
    ctx->CSSetShaderResources(0, 2, kNullSRVs);
    static ID3D11Buffer* const kNullCBs[2] = {};
    ctx->CSSetConstantBuffers(0, 2, kNullCBs);
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
