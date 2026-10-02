#include "inference_engine.h"
#include "../config.h"
#include <cmath>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <unordered_set>

namespace khanary::inference {

// ============================================================
// Construction
// ============================================================

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

void InferenceEngine::load_expert_weights(uint32_t expert_id,
                                           const float* data, size_t count) {
    if (expert_id >= cfg_.n_experts) return;
    size_t cap = expert_weights_[expert_id].size();
    std::copy(data, data + std::min(count, cap), expert_weights_[expert_id].begin());
}

void InferenceEngine::clear_kv() { kv_cache_->clear(); }

void InferenceEngine::set_lorentz_transition(float rapidity_rho,
                                             std::array<float, 3> direction,
                                             float alignment_mix,
                                             float mass_scale)
{
    auto rows = gpu::LorentzBoostRows::make_pure(
        rapidity_rho, direction, alignment_mix, mass_scale);
    if (!rows.enabled) {
        lorentz_transition_.reset();
        return;
    }
    lorentz_transition_ = rows;
}

void InferenceEngine::clear_lorentz_transition() {
    lorentz_transition_.reset();
}

const gpu::LorentzBoostRows* InferenceEngine::lorentz_transition() const {
    return lorentz_transition_ ? &*lorentz_transition_ : nullptr;
}

std::vector<gpu::GramScoreResult> InferenceEngine::score_prompt_attention(
    const kuhul::GramStore& store,
    const std::vector<PromptAttentionCandidate>& candidates,
    float admission_threshold,
    float tensor_mix)
{
    if (candidates.empty()) return {};

    std::lock_guard lock(semantic_scorer_mtx_);
    if (!semantic_scorer_) {
        semantic_scorer_ = std::make_unique<gpu::GramGpuScorer>(dev_);
        if (!semantic_scorer_->init()) {
            semantic_scorer_.reset();
            return {};
        }
    }

    std::vector<std::string> keys;
    std::vector<float> relevance;
    keys.reserve(candidates.size());
    relevance.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        keys.push_back(candidate.key);
        relevance.push_back(candidate.relevance);
    }
    const auto overrides = prompt_attention_vector_overrides(candidates);

    return semantic_scorer_->score_batch(
        store,
        keys,
        relevance,
        admission_threshold,
        0.5f,
        tensor_mix,
        nullptr,
        0.0f,
        0.0f,
        lorentz_transition(),
        &overrides);
}

bool InferenceEngine::semantic_attention_ready() const {
    std::lock_guard lock(semantic_scorer_mtx_);
    return semantic_scorer_ && semantic_scorer_->is_ready();
}

size_t InferenceEngine::dds_resident_bytes() const {
    return hot_swap_ ? hot_swap_->resident_bytes() : 0;
}

size_t InferenceEngine::dds_resident_ceiling_bytes() const {
    return hot_swap_ ? hot_swap_->ceiling() : 0;
}

// ============================================================
// load_dds_model
// ============================================================

void InferenceEngine::load_dds_model(const std::string& kson_path,
                                      const std::string& manifest_path,
                                      const std::string& shards_dir) {
    constexpr size_t MIN_GPU_RESIDENT_BUDGET_BYTES = 256ULL * 1024ULL * 1024ULL;
    const size_t dedicated_vram = dev_.adapter_info().dedicated_video_memory;
    const size_t resident_budget =
        dedicated_vram > 0
            ? std::max(MIN_GPU_RESIDENT_BUDGET_BYTES, (dedicated_vram * 85ULL) / 100ULL)
            : HD4600_RESIDENT_CEILING_BYTES;

    fold_map_  = std::make_unique<FoldMap>(FoldMap::load(kson_path, manifest_path, shards_dir));
    hot_swap_  = std::make_unique<gpu::HotSwapManager>(resident_budget);
    dds_loaded_ = false;
    dds_preload_total_tensors_ = 0;
    dds_preload_resident_tensors_ = 0;
    dds_preload_failed_tensors_ = 0;
    dds_preload_total_bytes_ = 0;
    dds_preload_resident_bytes_ = 0;

    update_config_from_folds();

    // Re-create router / kv_cache with real model dims
    router_   = std::make_unique<MoERouter>(cfg_.n_experts, cfg_.d_model);
    kv_cache_ = std::make_unique<KVCache>(dev_, cfg_.n_layers, cfg_.d_head, MAX_SEQ_LEN);

    // Preload DDS tensors into GPU residency cache at startup.
    preload_all_folds();

    dds_loaded_ = true;
}

// ============================================================
// update_config_from_folds
// ============================================================

void InferenceEngine::update_config_from_folds() {
    if (!fold_map_ || fold_map_->empty()) return;

    // n_layers = number of attention folds
    uint32_t n_attn = 0;
    for (uint32_t i = 0; i < fold_map_->fold_count(); ++i) {
        const auto* fe = fold_map_->get(i);
        if (fe && fe->layer_id.find("attention") != std::string::npos) ++n_attn;
    }
    if (n_attn > 0) cfg_.n_layers = n_attn;

    // d_model, vocab_size from token_embd.weight logical shape [d_model, vocab]
    const TensorRef* emb = fold_map_->get_tensor("token_embd.weight");
    if (emb && emb->logical_shape.size() >= 2) {
        cfg_.d_model    = emb->logical_shape[0]; // 2880
        cfg_.vocab_size = emb->logical_shape[1]; // 201088
    }

    // n_q_heads, n_kv_heads, d_head from blk.0 attention weights
    const TensorRef* wq = fold_map_->get_tensor("blk.0.attn_q.weight");
    const TensorRef* wk = fold_map_->get_tensor("blk.0.attn_k.weight");
    if (wq && wq->logical_shape.size() >= 2 &&
        wk && wk->logical_shape.size() >= 2) {
        // shape [d_model, total_q_dim] and [d_model, total_kv_dim]
        uint32_t d_q  = wq->logical_shape[1]; // 4096
        uint32_t d_kv = wk->logical_shape[1]; // 512
        cfg_.d_head     = 128; // canonical head dim for this architecture
        cfg_.n_heads    = d_q  / cfg_.d_head; // 32
        cfg_.n_kv_heads = d_kv / cfg_.d_head; // 4
    }

    // n_experts from MoE router weight shape [d_model, n_experts]
    const TensorRef* gate = fold_map_->get_tensor("blk.0.ffn_gate_inp.weight");
    if (gate && gate->logical_shape.size() >= 2)
        cfg_.n_experts = gate->logical_shape[1]; // 32
}

// ============================================================
// preload_fold — make a fold's tensors GPU-resident at startup
// ============================================================

void InferenceEngine::preload_fold(const std::string& layer_id) {
    const FoldEntry* fe = fold_map_->get(layer_id);
    if (!fe) return;
    for (const auto& tref : fe->tensors) {
        // Only pre-load reasonably-sized tensors (skip huge embedding/output matrices)
        uint64_t total = 0;
        for (const auto& s : tref.shards) total += s.size;
        if (total > 256ULL * 1024 * 1024) continue; // skip >256 MB at startup
        ensure_resident(tref.name);
    }
}

void InferenceEngine::preload_all_folds() {
    if (!fold_map_ || !hot_swap_) return;

    std::unordered_set<std::string> seen;
    for (const auto& fold : fold_map_->folds()) {
        for (const auto& tref : fold.tensors) {
            if (!seen.insert(tref.name).second) continue;

            uint64_t bytes = 0;
            for (const auto& s : tref.shards) bytes += s.size;
            dds_preload_total_tensors_ += 1;
            dds_preload_total_bytes_ += bytes;

            auto resident = ensure_resident(tref.name);
            if (resident && hot_swap_->is_resident(tref.name)) {
                dds_preload_resident_tensors_ += 1;
                dds_preload_resident_bytes_ += bytes;
            } else {
                dds_preload_failed_tensors_ += 1;
            }
        }
    }
}

// ============================================================
// MXFP4 / Q8_0 helpers
// ============================================================

// E8M0 shared exponent byte → float scale = 2^(e−127)
static inline float fp32_from_e8m0(uint8_t e) {
    uint32_t bits = static_cast<uint32_t>(e) << 23u;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// E2M1 dequantization lookup table (positive half 0..7, negative half 8..15)
static constexpr float k_mxfp4_e2m1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f,  6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f, -6.0f
};

float InferenceEngine::fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15u) & 1u;
    uint32_t exp  = (h >> 10u) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f32;
    if (exp == 0u) {
        if (mant == 0u) { f32 = sign << 31u; }
        else {
            // subnormal → normalise
            exp = 1u;
            while (!(mant & 0x400u)) { mant <<= 1u; --exp; }
            mant &= 0x3ffu;
            f32 = (sign << 31u) | ((exp + 127u - 15u) << 23u) | (mant << 13u);
        }
    } else if (exp == 31u) {
        f32 = (sign << 31u) | 0x7f800000u | (mant << 13u); // inf / NaN
    } else {
        f32 = (sign << 31u) | ((exp + 127u - 15u) << 23u) | (mant << 13u);
    }
    float result;
    std::memcpy(&result, &f32, 4);
    return result;
}

// Read shard slices for a tensor and dequantize Q8_0 → float32.
// Q8_0 block: [uint16 scale_f16][int8 qs[32]] = 34 bytes, 32 elements.
std::vector<float> InferenceEngine::load_q8_0_to_f32(const std::string& name) const {
    const TensorRef* tref = fold_map_->get_tensor(name);
    if (!tref || tref->shards.empty()) return {};

    // Accumulate raw bytes from all slices (ordered by offset_in_tensor)
    uint64_t total_bytes = 0;
    for (const auto& s : tref->shards) total_bytes += s.size;
    std::vector<uint8_t> raw(total_bytes);

    for (const auto& slice : tref->shards) {
        std::ifstream f(slice.path, std::ios::binary);
        if (!f) continue;
        f.seekg(static_cast<std::streamoff>(slice.offset_in_file));
        f.read(reinterpret_cast<char*>(raw.data() + slice.offset_in_tensor),
               static_cast<std::streamsize>(slice.size));
    }

    // Compute logical element count from shape
    uint64_t n_elems = 1;
    for (uint32_t d : tref->logical_shape) n_elems *= d;
    if (n_elems == 0) return {};

    std::vector<float> out(n_elems, 0.0f);

    if (tref->dtype == TensorDtype::Q8_0 || tref->dtype == TensorDtype::U8) {
        // Q8_0: 34 bytes per block of 32 elements
        constexpr uint64_t BLOCK_SIZE  = 34; // 2 (fp16 scale) + 32 (int8)
        constexpr uint64_t BLOCK_ELEMS = 32;
        uint64_t n_blocks = n_elems / BLOCK_ELEMS;
        for (uint64_t b = 0; b < n_blocks && (b + 1) * BLOCK_SIZE <= total_bytes; ++b) {
            const uint8_t* blk = raw.data() + b * BLOCK_SIZE;
            uint16_t scale_bits;
            std::memcpy(&scale_bits, blk, 2);
            float scale = fp16_to_f32(scale_bits);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            float* dst = out.data() + b * BLOCK_ELEMS;
            for (uint64_t i = 0; i < BLOCK_ELEMS; ++i)
                dst[i] = (float)qs[i] * scale;
        }
    } else if (tref->dtype == TensorDtype::F32) {
        if (total_bytes >= n_elems * 4)
            std::memcpy(out.data(), raw.data(), n_elems * 4);
    } else if (tref->dtype == TensorDtype::F16) {
        for (uint64_t i = 0; i < n_elems && (i + 1) * 2 <= total_bytes; ++i) {
            uint16_t h;
            std::memcpy(&h, raw.data() + i * 2, 2);
            out[i] = fp16_to_f32(h);
        }
    } else if (tref->dtype == TensorDtype::MXFP4) {
        // MXFP4 block: [uint8 e8m0_scale][uint8 qs[16]] = 17 bytes, 32 elements
        // ggml nibble order: qs[j] & 0xF → element j, qs[j] >> 4 → element j+16
        constexpr uint64_t MX_BYTES  = 17;
        constexpr uint64_t MX_ELEMS  = 32;
        const uint64_t n_blocks = n_elems / MX_ELEMS;
        for (uint64_t b = 0; b < n_blocks; ++b) {
            const uint64_t byte_off = b * MX_BYTES;
            if (byte_off + MX_BYTES > total_bytes) break;
            const uint8_t* blk   = raw.data() + byte_off;
            const float    scale = fp32_from_e8m0(blk[0]);
            const uint8_t* qs    = blk + 1;
            float*         dst   = out.data() + b * MX_ELEMS;
            for (uint64_t j = 0; j < 16; ++j) {
                dst[j]      = k_mxfp4_e2m1[qs[j] & 0xFu]  * scale;
                dst[j + 16] = k_mxfp4_e2m1[qs[j] >> 4u]   * scale;
            }
        }
    }

    return out;
}

// ============================================================
// TensorBuffer helpers
// ============================================================

std::shared_ptr<gpu::TensorBuffer> InferenceEngine::make_f32_buffer(
    const float* data, uint32_t count)
{
    gpu::TensorDesc desc;
    desc.format     = gpu::TensorFormat::F32;
    desc.elem_count = count;
    auto buf = std::make_shared<gpu::TensorBuffer>(dev_, desc);
    if (buf->is_valid())
        buf->upload(data, (size_t)count * 4);
    return buf;
}

std::shared_ptr<gpu::TensorBuffer> InferenceEngine::alloc_f32_buffer(uint32_t count) {
    gpu::TensorDesc desc;
    desc.format     = gpu::TensorFormat::F32;
    desc.elem_count = count;
    return std::make_shared<gpu::TensorBuffer>(dev_, desc);
}

std::vector<float> InferenceEngine::readback_f32(gpu::TensorBuffer* buf, uint32_t count) {
    std::vector<float> v(count, 0.0f);
    if (buf) buf->readback(v.data(), (size_t)count * 4);
    return v;
}

// Return a GPU-resident F32 weight buffer for GEMM.
// Cached under "f32:<name>" in hot_swap_.
std::shared_ptr<gpu::TensorBuffer> InferenceEngine::ensure_f32_weight(
    const std::string& name)
{
    if (!hot_swap_) return nullptr;
    const std::string key = "f32:" + name;
    {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        auto it = f32_cache_.find(key);
        if (it != f32_cache_.end()) {
            auto locked = it->second.lock();
            if (locked && hot_swap_->is_resident(key)) {
                hot_swap_->touch(key);
                return locked;
            }
        }
    }

    auto f32 = load_q8_0_to_f32(name);
    if (f32.empty()) return nullptr;

    auto buf = make_f32_buffer(f32.data(), static_cast<uint32_t>(f32.size()));
    if (!buf || !buf->is_valid()) return nullptr;

    if (!hot_swap_->make_resident(key, buf)) return nullptr;

    std::lock_guard<std::mutex> lk(cache_mtx_);
    f32_cache_[key] = buf;
    return buf;
}

// Ensure raw bytes for a tensor are GPU-resident (non-GEMM consumers).
std::shared_ptr<gpu::TensorBuffer> InferenceEngine::ensure_resident(
    const std::string& name)
{
    if (!hot_swap_ || !fold_map_) return nullptr;
    {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        auto it = f32_cache_.find(name);
        if (it != f32_cache_.end()) {
            auto locked = it->second.lock();
            if (locked && hot_swap_->is_resident(name)) {
                hot_swap_->touch(name);
                return locked;
            }
        }
    }

    const TensorRef* tref = fold_map_->get_tensor(name);
    if (!tref || tref->shards.empty()) return nullptr;

    uint64_t total_bytes = 0;
    for (const auto& s : tref->shards) total_bytes += s.size;
    if (total_bytes > hot_swap_->ceiling()) return nullptr;
    std::vector<uint8_t> raw(total_bytes);

    for (const auto& slice : tref->shards) {
        std::ifstream f(slice.path, std::ios::binary);
        if (!f) continue;
        f.seekg(static_cast<std::streamoff>(slice.offset_in_file));
        f.read(reinterpret_cast<char*>(raw.data() + slice.offset_in_tensor),
               static_cast<std::streamsize>(slice.size));
    }

    gpu::TensorDesc desc;
    desc.name       = name;
    desc.format     = (tref->dtype == TensorDtype::F32) ? gpu::TensorFormat::F32 : gpu::TensorFormat::Q8;
    desc.elem_count = static_cast<uint32_t>(total_bytes / 4);
    auto buf = std::make_shared<gpu::TensorBuffer>(dev_, desc);
    if (!buf->is_valid()) return nullptr;
    buf->upload(raw.data(), total_bytes);

    if (!hot_swap_->make_resident(name, buf)) return nullptr;
    std::lock_guard<std::mutex> lk(cache_mtx_);
    f32_cache_[name] = buf;
    return buf;
}

// ============================================================
// load_mxfp4_expert_f32 — per-expert MXFP4 slice dequantization
// ============================================================

// Reads the byte range for expert `expert_id` from a 3-D expert weight tensor
// with physical layout [n_experts, rows, pack(cols)] and dequantizes MXFP4 → F32.
// rows = in_dim (kson dim0), cols = out_dim (kson dim1).
std::vector<float> InferenceEngine::load_mxfp4_expert_f32(
    const std::string& tensor_name,
    uint32_t expert_id, uint32_t rows, uint32_t cols) const
{
    const TensorRef* tref = fold_map_->get_tensor(tensor_name);
    if (!tref || tref->shards.empty() || rows == 0 || cols == 0) return {};

    const bool is_f32_exp = (tref->dtype == TensorDtype::F32);
    const uint64_t row_bytes  = is_f32_exp ? static_cast<uint64_t>(cols) * 4u
                                           : (static_cast<uint64_t>(cols) / 32u) * 17u;
    const uint64_t slice_size = static_cast<uint64_t>(rows) * row_bytes;
    const uint64_t tensor_off = static_cast<uint64_t>(expert_id) * slice_size;
    if (slice_size == 0) return {};

    std::vector<uint8_t> raw(slice_size, 0u);

    for (const auto& slice : tref->shards) {
        const uint64_t slice_end = slice.offset_in_tensor + slice.size;
        if (tensor_off + slice_size <= slice.offset_in_tensor) break;
        if (tensor_off >= slice_end) continue;

        const uint64_t rd_start = std::max(tensor_off, slice.offset_in_tensor);
        const uint64_t rd_end   = std::min(tensor_off + slice_size, slice_end);
        const uint64_t rd_size  = rd_end - rd_start;
        const uint64_t buf_off  = rd_start - tensor_off;
        const uint64_t file_off = slice.offset_in_file + (rd_start - slice.offset_in_tensor);

        std::ifstream f(slice.path, std::ios::binary);
        if (f) {
            f.seekg(static_cast<std::streamoff>(file_off));
            f.read(reinterpret_cast<char*>(raw.data() + buf_off),
                   static_cast<std::streamsize>(rd_size));
        }
    }

    const uint64_t n_elems = static_cast<uint64_t>(rows) * cols;
    std::vector<float> out(n_elems, 0.0f);

    if (is_f32_exp) {
        if (slice_size >= n_elems * 4u)
            std::memcpy(out.data(), raw.data(), n_elems * 4u);
    } else {
        // MXFP4: 17-byte blocks, 32 elements each
        const uint64_t n_blocks = n_elems / 32u;
        for (uint64_t b = 0; b < n_blocks; ++b) {
            const uint64_t byte_off = b * 17u;
            if (byte_off + 17u > slice_size) break;
            const uint8_t* blk   = raw.data() + byte_off;
            const float    scale = fp32_from_e8m0(blk[0]);
            const uint8_t* qs    = blk + 1;
            float*         dst   = out.data() + b * 32u;
            for (uint64_t j = 0; j < 16; ++j) {
                dst[j]      = k_mxfp4_e2m1[qs[j] & 0xFu]  * scale;
                dst[j + 16] = k_mxfp4_e2m1[qs[j] >> 4u]   * scale;
            }
        }
    }
    return out;
}

// ============================================================
// CPU helpers
// ============================================================

// RMSNorm: out[t*d+i] = x[t*d+i] / RMS(x[t]) * w[i]
std::vector<float> InferenceEngine::rms_norm(const std::vector<float>& x,
                                              const std::vector<float>& w,
                                              uint32_t d) {
    std::vector<float> out(x.size(), 0.0f);
    uint32_t seq = static_cast<uint32_t>(x.size()) / d;
    for (uint32_t t = 0; t < seq; ++t) {
        const float* row = x.data() + t * d;
        float ss = 0.0f;
        for (uint32_t i = 0; i < d; ++i) ss += row[i] * row[i];
        float inv = 1.0f / std::sqrt(ss / d + 1e-5f);
        float* dst = out.data() + t * d;
        for (uint32_t i = 0; i < d; ++i)
            dst[i] = row[i] * inv * (i < w.size() ? w[i] : 1.0f);
    }
    return out;
}

// ============================================================
// embed_dds — token lookup via Q8_0 row extraction
// ============================================================

// Reads one embedding row per token directly from shard slices.
// token_embd.weight physical layout: [vocab_size, row_bytes]
// Q8_0 row: (d_model/32) * 34 bytes = 3060 bytes for d_model=2880
std::vector<float> InferenceEngine::embed_dds(const std::vector<uint32_t>& tokens) {
    const TensorRef* tref = fold_map_->get_tensor("token_embd.weight");
    if (!tref || tref->shards.empty()) return embed(tokens);

    uint32_t seq = static_cast<uint32_t>(tokens.size());
    uint32_t d   = cfg_.d_model;
    std::vector<float> emb(seq * d, 0.0f);

    const bool is_f32_emb = (tref->dtype == TensorDtype::F32);
    uint64_t row_bytes = is_f32_emb ? (uint64_t)d * 4u
                                    : (uint64_t)(d / 32u) * 34u;

    std::vector<uint8_t> row_buf(row_bytes);

    for (uint32_t t = 0; t < seq; ++t) {
        uint32_t tok = std::min(tokens[t], cfg_.vocab_size - 1);
        uint64_t tensor_off = (uint64_t)tok * row_bytes;

        // Find shard slices covering this row
        std::memset(row_buf.data(), 0, row_bytes);
        for (const auto& slice : tref->shards) {
            uint64_t slice_end = slice.offset_in_tensor + slice.size;
            if (tensor_off >= slice_end) continue;
            if (tensor_off + row_bytes <= slice.offset_in_tensor) break;

            uint64_t rd_start = std::max(tensor_off, slice.offset_in_tensor);
            uint64_t rd_end   = std::min(tensor_off + row_bytes, slice_end);
            uint64_t rd_size  = rd_end - rd_start;
            uint64_t buf_off  = rd_start - tensor_off;
            uint64_t file_off = slice.offset_in_file + (rd_start - slice.offset_in_tensor);

            std::ifstream f(slice.path, std::ios::binary);
            if (f) {
                f.seekg(static_cast<std::streamoff>(file_off));
                f.read(reinterpret_cast<char*>(row_buf.data() + buf_off),
                       static_cast<std::streamsize>(rd_size));
            }
        }

        float* dst = emb.data() + t * d;
        if (is_f32_emb) {
            std::memcpy(dst, row_buf.data(), (size_t)d * 4);
        } else {
            // Q8_0: (d / 32) blocks of 34 bytes
            uint64_t n_blocks = (uint64_t)(d / 32u);
            for (uint64_t b = 0; b < n_blocks; ++b) {
                const uint8_t* blk = row_buf.data() + b * 34u;
                uint16_t scale_bits;
                std::memcpy(&scale_bits, blk, 2);
                float scale = fp16_to_f32(scale_bits);
                const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
                for (uint64_t i = 0; i < 32u; ++i)
                    dst[b * 32u + i] = (float)qs[i] * scale;
            }
        }
    }

    return emb;
}

// ============================================================
// apply_attention_dds — one transformer attention layer
// ============================================================

std::vector<float> InferenceEngine::apply_attention_dds(
    uint32_t layer, const std::vector<float>& hidden, uint32_t seq_len)
{
    const uint32_t d  = cfg_.d_model;
    const uint32_t dq = cfg_.n_heads    * cfg_.d_head; // 4096
    const uint32_t dkv = cfg_.n_kv_heads * cfg_.d_head; // 512
    const std::string pfx = "blk." + std::to_string(layer) + ".";

    // 1. Pre-attention RMSNorm
    auto norm_w = load_q8_0_to_f32(pfx + "attn_norm.weight");
    auto hidden_n = rms_norm(hidden, norm_w, d);

    // 2. Upload normed hidden state to GPU [seq_len, d_model]
    auto h_buf = make_f32_buffer(hidden_n.data(), seq_len * d);

    // 3. Q projection: [seq, d_model] × [d_model, dq] → [seq, dq]
    auto wq = ensure_f32_weight(pfx + "attn_q.weight");
    auto q_buf = alloc_f32_buffer(seq_len * dq);
    if (wq) dml_.gemm(h_buf.get(), wq.get(), q_buf.get(), seq_len, d, dq);

    // 4. K projection: [seq, d_model] × [d_model, dkv] → [seq, dkv]
    auto wk = ensure_f32_weight(pfx + "attn_k.weight");
    auto k_buf = alloc_f32_buffer(seq_len * dkv);
    if (wk) dml_.gemm(h_buf.get(), wk.get(), k_buf.get(), seq_len, d, dkv);

    // 5. V projection: [seq, d_model] × [d_model, dkv] → [seq, dkv]
    auto wv = ensure_f32_weight(pfx + "attn_v.weight");
    auto v_buf = alloc_f32_buffer(seq_len * dkv);
    if (wv) dml_.gemm(h_buf.get(), wv.get(), v_buf.get(), seq_len, d, dkv);

    // 6. Scaled dot-product attention (GQA: repeat K/V n_heads/n_kv_heads times)
    // dml_.attention expects Q/K/V with matching head counts; expand K/V for GQA.
    // For simplicity: run per-head on CPU using readback (full GPU GQA is a later optimisation).
    auto q_host = readback_f32(q_buf.get(), seq_len * dq);
    auto k_host = readback_f32(k_buf.get(), seq_len * dkv);
    auto v_host = readback_f32(v_buf.get(), seq_len * dkv);

    const uint32_t head_dim   = cfg_.d_head;       // 128
    const uint32_t n_q        = cfg_.n_heads;       // 32
    const uint32_t n_kv       = cfg_.n_kv_heads;    // 4
    const uint32_t group      = n_q / n_kv;         // 8 Q heads per KV head
    const float    scale_val  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> attn_out(seq_len * dq, 0.0f);

    for (uint32_t h = 0; h < n_q; ++h) {
        uint32_t kv_h = h / group; // which KV head this Q head maps to

        // Compute attention scores: [seq, seq]
        std::vector<float> scores(seq_len * seq_len, 0.0f);
        for (uint32_t i = 0; i < seq_len; ++i) {
            for (uint32_t j = 0; j <= i; ++j) { // causal mask
                float dot = 0.0f;
                const float* qi = q_host.data() + i * dq + h * head_dim;
                const float* kj = k_host.data() + j * dkv + kv_h * head_dim;
                for (uint32_t d2 = 0; d2 < head_dim; ++d2) dot += qi[d2] * kj[d2];
                scores[i * seq_len + j] = dot * scale_val;
            }
            // Softmax over valid positions [0..i]
            float max_s = *std::max_element(scores.data() + i * seq_len,
                                             scores.data() + i * seq_len + i + 1);
            float sum = 0.0f;
            for (uint32_t j = 0; j <= i; ++j) {
                scores[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_s);
                sum += scores[i * seq_len + j];
            }
            if (sum > 0.0f)
                for (uint32_t j = 0; j <= i; ++j) scores[i * seq_len + j] /= sum;
        }

        // Weighted sum of V
        for (uint32_t i = 0; i < seq_len; ++i) {
            float* dst = attn_out.data() + i * dq + h * head_dim;
            for (uint32_t j = 0; j <= i; ++j) {
                const float* vj = v_host.data() + j * dkv + kv_h * head_dim;
                float w = scores[i * seq_len + j];
                for (uint32_t d2 = 0; d2 < head_dim; ++d2) dst[d2] += w * vj[d2];
            }
        }
    }

    // 7. Output projection: [seq, dq] × [dq, d_model] → [seq, d_model]
    auto attn_buf = make_f32_buffer(attn_out.data(), seq_len * dq);
    auto wo = ensure_f32_weight(pfx + "attn_output.weight");
    auto out_buf = alloc_f32_buffer(seq_len * d);
    if (wo) dml_.gemm(attn_buf.get(), wo.get(), out_buf.get(), seq_len, dq, d);

    auto out = readback_f32(out_buf.get(), seq_len * d);

    // Residual connection + optional output bias
    auto bias = load_q8_0_to_f32(pfx + "attn_output.bias");
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t i = 0; i < d; ++i) {
            out[t * d + i] += hidden[t * d + i];
            if (i < bias.size()) out[t * d + i] += bias[i];
        }
    }
    return out;
}

// ============================================================
// apply_moe_dds — one MoE feed-forward layer
// ============================================================

std::vector<float> InferenceEngine::apply_moe_dds(
    uint32_t layer, const std::vector<float>& hidden, uint32_t seq_len)
{
    const uint32_t d = cfg_.d_model;
    const uint32_t E = cfg_.n_experts;
    const uint32_t K = cfg_.top_k;
    const std::string pfx = "blk." + std::to_string(layer) + ".";

    // 1. Post-attention RMSNorm (uses MXFP4 weight; load_q8_0_to_f32 returns zeros)
    auto norm_w = load_q8_0_to_f32(pfx + "post_attention_norm.weight");
    auto hidden_n = norm_w.empty() ? hidden : rms_norm(hidden, norm_w, d);

    // 2. Router: [seq, d_model] × [d_model, n_experts] → [seq, n_experts] scores
    //    ffn_gate_inp.weight is MXFP4 — fall back to CPU zeros route if unavailable.
    auto h_buf     = make_f32_buffer(hidden_n.data(), seq_len * d);
    auto wgate_inp = ensure_f32_weight(pfx + "ffn_gate_inp.weight");

    std::vector<float> router_scores(seq_len * E, 0.0f);
    if (wgate_inp) {
        auto score_buf = alloc_f32_buffer(seq_len * E);
        dml_.gemm(h_buf.get(), wgate_inp.get(), score_buf.get(), seq_len, d, E);
        router_scores = readback_f32(score_buf.get(), seq_len * E);
    }

    // Derive d_ffn from gate weight logical_shape (kson dim1 = out_dim = d_ffn)
    uint32_t d_ffn = d; // symmetric default; correct for this model (d=d_ffn=2880)
    {
        const TensorRef* gate_ref = fold_map_->get_tensor(pfx + "ffn_gate_exps.weight");
        if (gate_ref && gate_ref->logical_shape.size() >= 2)
            d_ffn = gate_ref->logical_shape[1];
    }

    std::vector<float> out(seq_len * d, 0.0f);

    for (uint32_t t = 0; t < seq_len; ++t) {
        float* scores = router_scores.data() + t * E;

        // Softmax
        float max_s = *std::max_element(scores, scores + E);
        float sum = 0.0f;
        for (uint32_t e = 0; e < E; ++e) { scores[e] = std::exp(scores[e] - max_s); sum += scores[e]; }
        if (sum > 0.0f) for (uint32_t e = 0; e < E; ++e) scores[e] /= sum;

        // Top-K selection
        std::vector<std::pair<float, uint32_t>> ranked(E);
        for (uint32_t e = 0; e < E; ++e) ranked[e] = {scores[e], e};
        std::partial_sort(ranked.begin(), ranked.begin() + K, ranked.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });

        // Per-expert SwiGLU FFN through GPU GEMM
        // Expert weight layout: [n_experts, in_dim, pack(out_dim)] (kson [in, out] convention)
        // gate/up: in_dim=d_model, out_dim=d_ffn; down: in_dim=d_ffn, out_dim=d_model
        const float* h_tok = hidden_n.data() + t * d;
        float*       h_out = out.data() + t * d;
        auto h_gpu = make_f32_buffer(h_tok, d);

        for (uint32_t ki = 0; ki < K; ++ki) {
            const float    ew  = ranked[ki].first;
            const uint32_t eid = ranked[ki].second;

            // Load MXFP4 slices for this expert
            auto gate_f32 = load_mxfp4_expert_f32(pfx + "ffn_gate_exps.weight", eid, d, d_ffn);
            auto up_f32   = load_mxfp4_expert_f32(pfx + "ffn_up_exps.weight",   eid, d, d_ffn);
            auto down_f32 = load_mxfp4_expert_f32(pfx + "ffn_down_exps.weight", eid, d_ffn, d);

            if (gate_f32.empty() || up_f32.empty() || down_f32.empty()) {
                // Graceful fallback: identity so residual path still works
                for (uint32_t i = 0; i < d; ++i) h_out[i] += ew * h_tok[i];
                continue;
            }

            // Upload expert weights to GPU
            auto gate_gpu = make_f32_buffer(gate_f32.data(), static_cast<uint32_t>(gate_f32.size()));
            auto up_gpu   = make_f32_buffer(up_f32.data(),   static_cast<uint32_t>(up_f32.size()));

            // Gate GEMM: h[1,d] × gate_w[d,d_ffn] → gate_act[1,d_ffn]
            auto gate_act_buf = alloc_f32_buffer(d_ffn);
            dml_.gemm(h_gpu.get(), gate_gpu.get(), gate_act_buf.get(), 1u, d, d_ffn);

            // Up GEMM: h[1,d] × up_w[d,d_ffn] → up_act[1,d_ffn]
            auto up_act_buf = alloc_f32_buffer(d_ffn);
            dml_.gemm(h_gpu.get(), up_gpu.get(), up_act_buf.get(), 1u, d, d_ffn);

            // SwiGLU on CPU: ffn_act[j] = silu(gate[j]) * up[j]
            auto gate_h = readback_f32(gate_act_buf.get(), d_ffn);
            auto up_h   = readback_f32(up_act_buf.get(),   d_ffn);
            for (uint32_t i = 0; i < d_ffn; ++i) {
                const float g = gate_h[i];
                gate_h[i] = (g / (1.0f + std::exp(-g))) * up_h[i];
            }

            // Down GEMM: ffn_act[1,d_ffn] × down_w[d_ffn,d] → expert_out[1,d]
            auto ffn_gpu   = make_f32_buffer(gate_h.data(), d_ffn);
            auto down_gpu  = make_f32_buffer(down_f32.data(), static_cast<uint32_t>(down_f32.size()));
            auto d_out_buf = alloc_f32_buffer(d);
            dml_.gemm(ffn_gpu.get(), down_gpu.get(), d_out_buf.get(), 1u, d_ffn, d);

            auto d_out_h = readback_f32(d_out_buf.get(), d);
            for (uint32_t i = 0; i < d; ++i) h_out[i] += ew * d_out_h[i];
        }
    }

    // Residual
    for (uint32_t i = 0; i < seq_len * d; ++i) out[i] += hidden[i];
    return out;
}

// ============================================================
// forward_dds — full transformer forward pass
// ============================================================

std::vector<float> InferenceEngine::forward_dds(const std::vector<uint32_t>& tokens) {
    uint32_t seq = static_cast<uint32_t>(tokens.size());
    if (seq == 0) return std::vector<float>(cfg_.vocab_size, 0.0f);

    // Pop: embed tokens
    pe_.set(kuhul::Phase::Pop);
    auto hidden = embed_dds(tokens);

    // Wo → Sek: transformer layers
    pe_.advance(); // → Wo
    for (uint32_t layer = 0; layer < cfg_.n_layers; ++layer) {
        hidden = apply_attention_dds(layer, hidden, seq);
        hidden = apply_moe_dds(layer, hidden, seq);
    }

    // Ch'en: output norm
    pe_.advance(); // → Yax
    pe_.advance(); // → Sek
    pe_.advance(); // → Ch'en
    auto norm_w = load_q8_0_to_f32("output_norm.weight");
    auto normed = rms_norm(hidden, norm_w, cfg_.d_model);

    // Xul: unembed — last token only.
    // output.weight [d_model=2880, vocab=201088] is ~2.3 GB in F32; use chunked
    // Q8_0 dot-product to avoid that allocation.
    pe_.advance(); // → Xul
    uint32_t last_t = seq - 1;
    return compute_output_logits_chunked(normed.data() + last_t * cfg_.d_model);
}

// ============================================================
// compute_output_logits_chunked
// ============================================================

// Reads output.weight in 4096-vocab chunks and computes dot(row, last_hidden)
// without ever allocating the full matrix. Branches on tref->dtype (F32 or Q8_0).
std::vector<float> InferenceEngine::compute_output_logits_chunked(
    const float* last_hidden) const
{
    std::vector<float> logits(cfg_.vocab_size, 0.0f);
    if (!fold_map_) return logits;

    const TensorRef* tref = fold_map_->get_tensor("output.weight");
    if (!tref || tref->shards.empty()) return logits;

    const bool     is_f32_out   = (tref->dtype == TensorDtype::F32);
    const uint64_t row_bytes    = is_f32_out
        ? static_cast<uint64_t>(cfg_.d_model) * 4u
        : (static_cast<uint64_t>(cfg_.d_model) / 32u) * 34u;

    constexpr uint32_t CHUNK_VOCAB = 4096;
    std::vector<uint8_t> chunk_buf;

    for (uint32_t v_start = 0; v_start < cfg_.vocab_size; v_start += CHUNK_VOCAB) {
        const uint32_t v_end       = std::min(v_start + CHUNK_VOCAB, cfg_.vocab_size);
        const uint64_t chunk_rows  = v_end - v_start;
        const uint64_t chunk_bytes = chunk_rows * row_bytes;
        chunk_buf.assign(chunk_bytes, 0);

        const uint64_t tensor_off = (uint64_t)v_start * row_bytes;

        for (const auto& slice : tref->shards) {
            const uint64_t slice_end = slice.offset_in_tensor + slice.size;
            if (tensor_off + chunk_bytes <= slice.offset_in_tensor) break;
            if (tensor_off >= slice_end) continue;

            uint64_t rd_start = std::max(tensor_off, slice.offset_in_tensor);
            uint64_t rd_end   = std::min(tensor_off + chunk_bytes, slice_end);
            uint64_t rd_size  = rd_end - rd_start;
            uint64_t buf_off  = rd_start - tensor_off;
            uint64_t file_off = slice.offset_in_file + (rd_start - slice.offset_in_tensor);

            std::ifstream f(slice.path, std::ios::binary);
            if (f) {
                f.seekg(static_cast<std::streamoff>(file_off));
                f.read(reinterpret_cast<char*>(chunk_buf.data() + buf_off),
                       static_cast<std::streamsize>(rd_size));
            }
        }

        for (uint32_t vi = v_start; vi < v_end; ++vi) {
            float dot = 0.0f;
            const uint8_t* row = chunk_buf.data() + (vi - v_start) * row_bytes;
            if (is_f32_out) {
                const float* rf = reinterpret_cast<const float*>(row);
                for (uint32_t i = 0; i < cfg_.d_model; ++i)
                    dot += rf[i] * last_hidden[i];
            } else {
                // Q8_0 blocks
                const uint64_t blocks_per_row = cfg_.d_model / 32u;
                for (uint64_t b = 0; b < blocks_per_row; ++b) {
                    const uint8_t* blk = row + b * 34u;
                    uint16_t scale_bits;
                    std::memcpy(&scale_bits, blk, 2);
                    const float scale = fp16_to_f32(scale_bits);
                    const int8_t*  qs = reinterpret_cast<const int8_t*>(blk + 2);
                    const float*   h  = last_hidden + b * 32u;
                    for (uint64_t i = 0; i < 32u; ++i)
                        dot += static_cast<float>(qs[i]) * scale * h[i];
                }
            }
            logits[vi] = dot;
        }
    }
    return logits;
}

// ============================================================
// forward dispatcher
// ============================================================

std::vector<float> InferenceEngine::forward(const std::vector<uint32_t>& tokens) {
    if (dds_loaded_) return forward_dds(tokens);
    return forward_stub(tokens);
}

// ============================================================
// forward_stub — legacy CPU placeholder
// ============================================================

std::vector<float> InferenceEngine::forward_stub(const std::vector<uint32_t>& tokens) {
    pe_.set(kuhul::Phase::Pop);
    auto hidden = embed(tokens);

    size_t last_t = tokens.size() - 1;
    std::vector<float> last_h(hidden.begin() + last_t * cfg_.d_model,
                               hidden.begin() + (last_t + 1) * cfg_.d_model);

    pe_.advance(); // → Wo
    auto allocs = router_->route(last_h, cfg_.top_k);

    pe_.advance(); // → Yax
    pe_.advance(); // → Sek
    std::vector<float> out(cfg_.d_model, 0.0f);
    for (const auto& alloc : allocs) {
        auto expert_out = apply_expert(alloc.expert_id, last_h);
        for (uint32_t d = 0; d < cfg_.d_model; ++d)
            out[d] += alloc.weight * expert_out[d];
        router_->record_dispatch(alloc.expert_id);
    }

    pe_.advance(); // → Ch'en
    float mean = std::accumulate(out.begin(), out.end(), 0.0f) / out.size();
    float var  = 0.0f;
    for (float v : out) var += (v - mean) * (v - mean);
    var /= out.size();
    float std_inv = 1.0f / std::sqrt(var + 1e-6f);
    for (auto& v : out) v = (v - mean) * std_inv;

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

// ============================================================
// run
// ============================================================

InferenceResult InferenceEngine::run(const std::string& prompt, uint32_t max_tokens) {
    InferenceResult result;
    result.ok = false;
    result.final_phase = kuhul::Phase::Pop;

    const uint32_t limit = (max_tokens > 0) ? max_tokens : cfg_.max_tokens;

    try {
        auto input_ids = tokenizer_.encode(prompt, true);
        std::vector<uint32_t> gen_ids;

        for (uint32_t step = 0; step < limit; ++step) {
            auto t0 = std::chrono::steady_clock::now();
            auto logits = forward(input_ids);
            uint32_t next_id = sampler_.sample(logits, cfg_.temperature, cfg_.top_p);
            float step_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            token_latencies_ms_.push_back(step_ms);
            if (token_latencies_ms_.size() > 256) token_latencies_ms_.pop_front();

            if (next_id == GLYPH_EOS) break;

            gen_ids.push_back(next_id);
            input_ids.push_back(next_id);
            if (input_ids.size() > MAX_SEQ_LEN)
                input_ids.erase(input_ids.begin());

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

float InferenceEngine::mean_token_latency_ms(uint32_t n_recent) const {
    if (token_latencies_ms_.empty()) return 1000.0f;
    uint32_t n = static_cast<uint32_t>(
        std::min<size_t>(n_recent, token_latencies_ms_.size()));
    float sum = 0.0f;
    auto begin = token_latencies_ms_.end() - n;
    for (auto it = begin; it != token_latencies_ms_.end(); ++it) sum += *it;
    return sum / static_cast<float>(n);
}

// ============================================================
// Legacy helpers (forward_stub)
// ============================================================

std::vector<float> InferenceEngine::embed(const std::vector<uint32_t>& tokens) {
    size_t seq = tokens.size();
    std::vector<float> emb(seq * cfg_.d_model, 0.0f);
    for (size_t t = 0; t < seq; ++t) {
        uint32_t id = tokens[t];
        for (uint32_t d = 0; d < cfg_.d_model; ++d) {
            float pos  = (float)t;
            float freq = 1.0f / std::pow(10000.0f, (float)(d / 2) / cfg_.d_model);
            emb[t * cfg_.d_model + d] = (d % 2 == 0)
                ? std::sin(pos * freq) + ((float)(id % 256) / 256.0f)
                : std::cos(pos * freq) + ((float)(id % 256) / 256.0f);
        }
    }
    return emb;
}

std::vector<float> InferenceEngine::apply_expert(uint32_t expert_id,
                                                   const std::vector<float>& hidden) {
    uint32_t d = cfg_.d_model;
    std::vector<float> out(d, 0.0f);
    if (expert_id >= cfg_.n_experts) return out;
    const auto& W = expert_weights_[expert_id];
    for (uint32_t i = 0; i < d; ++i) {
        float v = 0.0f;
        for (uint32_t j = 0; j < d; ++j) v += W[i * d + j] * hidden[j];
        out[i] = std::tanh(v);
    }
    return out;
}

} // namespace khanary::inference
