#include "mou_router.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

namespace khanary::micronaut {

MouRouter::MouRouter(MicronautRegistry&      reg,
                     memory::ConfidenceStore& confidence)
    : reg_(reg)
    , confidence_(confidence)
    , moe_(0, 3)   // n_experts=0 placeholder; rebuilt in sync_index()
{}

uint32_t MouRouter::n_micronauts() const {
    std::lock_guard<std::mutex> lk(idx_mtx_);
    return static_cast<uint32_t>(index_to_id_.size());
}

// ── sync_index ────────────────────────────────────────────────────────────────

void MouRouter::sync_index() {
    auto ids = reg_.ids();
    std::lock_guard<std::mutex> lk(idx_mtx_);

    index_to_id_ = std::move(ids);
    id_to_index_.clear();
    for (uint32_t i = 0; i < static_cast<uint32_t>(index_to_id_.size()); ++i)
        id_to_index_[index_to_id_[i]] = i;

    const uint32_t n = static_cast<uint32_t>(index_to_id_.size());
    moe_ = inference::MoERouter(n, 3);

    // Expert weight row i = normalized TrackVectorMeta {x,y,z} for micronaut i.
    // Undeclared vectors default to {0,0,0} → normalized to uniform {0.577,0.577,0.577}.
    std::vector<float> W(n * 3, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        const auto* e = reg_.get(index_to_id_[i]);
        if (!e) continue;
        const auto v = normalize3({e->vector_meta.x,
                                   e->vector_meta.y,
                                   e->vector_meta.z});
        W[i * 3 + 0] = v[0];
        W[i * 3 + 1] = v[1];
        W[i * 3 + 2] = v[2];
    }
    moe_.load_weights(W.data(), W.size());
}

// ── sync_bias ─────────────────────────────────────────────────────────────────

void MouRouter::sync_bias() {
    // ConfidenceStore snapshot → expert index string key → float bias.
    // MoERouter::load_bias() expects map keyed by string(uint32_t).
    const auto snapshot = confidence_.snapshot();
    std::unordered_map<std::string, float> bias_map;
    std::lock_guard<std::mutex> lk(idx_mtx_);
    for (const auto& [id, conf] : snapshot) {
        auto it = id_to_index_.find(id);
        if (it != id_to_index_.end())
            bias_map[std::to_string(it->second)] = conf;
    }
    moe_.load_bias(bias_map);
}

// ── route ─────────────────────────────────────────────────────────────────────

std::vector<MouAlloc> MouRouter::route(
    kuhul::Phase                    phase,
    const std::vector<std::string>& tokens,
    uint32_t                        top_k) const
{
    std::lock_guard<std::mutex> lk(idx_mtx_);
    if (index_to_id_.empty()) return {};

    // F_Q: 3D query vector from tokens
    const auto qv = query_vec(tokens);
    const std::vector<float> hidden{qv[0], qv[1], qv[2]};

    // Request k_wide candidates to survive the phase filter downstream.
    const uint32_t k_wide = std::min<uint32_t>(
        top_k * 3, static_cast<uint32_t>(index_to_id_.size()));
    const auto allocs = moe_.route(hidden, k_wide);

    // Phase filter: keep only micronauts at the requested phase.
    std::vector<MouAlloc> out;
    out.reserve(top_k);
    for (const auto& a : allocs) {
        if (a.expert_id >= static_cast<uint32_t>(index_to_id_.size())) continue;
        const auto& id    = index_to_id_[a.expert_id];
        const auto* entry = reg_.get(id);
        if (!entry || entry->primary_phase != phase) continue;
        out.push_back({id, a.expert_id, a.weight});
        if (static_cast<uint32_t>(out.size()) >= top_k) break;
    }

    // Fallback: phase filter emptied output — return unfiltered top_k.
    if (out.empty()) {
        for (const auto& a : allocs) {
            if (a.expert_id >= static_cast<uint32_t>(index_to_id_.size())) continue;
            out.push_back({index_to_id_[a.expert_id], a.expert_id, a.weight});
            if (static_cast<uint32_t>(out.size()) >= top_k) break;
        }
    }

    // Renormalize weights to sum to 1.0.
    float sum = 0.0f;
    for (const auto& a : out) sum += a.weight;
    if (sum > 1e-6f)
        for (auto& a : out) a.weight /= sum;

    return out;
}

// ── record_dispatch / routing_variance ───────────────────────────────────────

void MouRouter::record_dispatch(const std::string& id) {
    std::lock_guard<std::mutex> lk(idx_mtx_);
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end())
        moe_.record_dispatch(it->second);
}

float MouRouter::routing_variance(uint32_t n_recent) const {
    return moe_.routing_variance(n_recent);
}

// ── Query vector helpers ──────────────────────────────────────────────────────
//
// Matches PerceptionStore::embed() approach: hash → sin projection → normalize.
// Using std::hash<string> + sin(bit_extract * 0.1 + dim_index) for dims 0,1,2.

std::array<float,3> MouRouter::query_vec(const std::vector<std::string>& tokens) {
    if (tokens.empty()) return normalize3({0.577f, 0.577f, 0.577f});

    // Aggregate over tokens: mean of per-token sin-hash projections.
    std::array<float,3> acc{0.0f, 0.0f, 0.0f};
    for (const auto& tok : tokens) {
        const size_t h = std::hash<std::string>{}(tok);
        constexpr uint32_t kBits = static_cast<uint32_t>(sizeof(size_t) * 8u);
        for (uint32_t d = 0; d < 3; ++d) {
            acc[d] += std::sin(
                static_cast<float>((h >> (d % kBits)) & 0xFFu) * 0.1f
                + static_cast<float>(d));
        }
    }
    const float n = static_cast<float>(tokens.size());
    return normalize3({acc[0] / n, acc[1] / n, acc[2] / n});
}

float MouRouter::norm3(const std::array<float,3>& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

std::array<float,3> MouRouter::normalize3(const std::array<float,3>& v) {
    const float n = norm3(v);
    if (n < 1e-6f) return {0.577f, 0.577f, 0.577f};  // uniform direction
    return {v[0]/n, v[1]/n, v[2]/n};
}

} // namespace khanary::micronaut
