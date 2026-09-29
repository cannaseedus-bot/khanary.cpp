#include "moe_router.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace khanary::inference {

MoERouter::MoERouter(uint32_t n_experts, uint32_t hidden_dim)
    : n_experts_(n_experts), hidden_dim_(hidden_dim)
{
    weights_.resize(n_experts * hidden_dim, 0.0f);
    bias_.resize(n_experts, 0.0f);
    dispatch_count_.resize(n_experts, 0);
}

void MoERouter::load_weights(const float* data, size_t count) {
    size_t expected = (size_t)n_experts_ * hidden_dim_;
    size_t copy = std::min(count, expected);
    std::copy(data, data + copy, weights_.begin());
}

void MoERouter::load_bias(const std::unordered_map<std::string, float>& bias) {
    for (auto& [k, v] : bias) {
        try {
            uint32_t id = (uint32_t)std::stoul(k);
            if (id < n_experts_) bias_[id] = v;
        } catch (...) {}
    }
}

std::vector<float> MoERouter::compute_logits(const std::vector<float>& hidden) const {
    std::vector<float> logits(n_experts_, 0.0f);
    for (uint32_t e = 0; e < n_experts_; ++e) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < hidden_dim_ && d < (uint32_t)hidden.size(); ++d)
            dot += weights_[e * hidden_dim_ + d] * hidden[d];
        logits[e] = dot + bias_[e];
    }
    return logits;
}

std::vector<float> MoERouter::softmax(const std::vector<float>& logits) {
    float max_l = *std::max_element(logits.begin(), logits.end());
    std::vector<float> p(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(logits[i] - max_l);
        sum += p[i];
    }
    for (auto& v : p) v /= sum;
    return p;
}

std::vector<ExpertAlloc> MoERouter::route(const std::vector<float>& hidden, uint32_t top_k) const {
    if (hidden.empty()) return {};
    auto logits = compute_logits(hidden);
    auto probs  = softmax(logits);

    // Top-k selection
    std::vector<uint32_t> indices(n_experts_);
    std::iota(indices.begin(), indices.end(), 0);
    uint32_t k = std::min(top_k, n_experts_);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&](uint32_t a, uint32_t b){ return probs[a] > probs[b]; });
    indices.resize(k);

    // Renormalize
    float sum = 0.0f;
    for (uint32_t i : indices) sum += probs[i];

    std::vector<ExpertAlloc> out;
    for (uint32_t i : indices)
        out.push_back({i, sum > 0 ? probs[i] / sum : 1.0f / k});

    // Record top-expert weight for routing_variance()
    if (!out.empty()) {
        recent_top_weights_.push_back(out.front().weight);
        if (recent_top_weights_.size() > 64) recent_top_weights_.pop_front();
    }
    return out;
}

float MoERouter::routing_variance(uint32_t n_recent) const {
    uint32_t n = static_cast<uint32_t>(
        std::min<size_t>(n_recent, recent_top_weights_.size()));
    if (n < 2) return 0.0f;
    float sum = 0.0f;
    auto begin = recent_top_weights_.end() - n;
    for (auto it = begin; it != recent_top_weights_.end(); ++it) sum += *it;
    float mean = sum / static_cast<float>(n);
    float var = 0.0f;
    for (auto it = begin; it != recent_top_weights_.end(); ++it) {
        float d = *it - mean;
        var += d * d;
    }
    return var / static_cast<float>(n - 1);
}

void MoERouter::record_dispatch(uint32_t expert_id) {
    if (expert_id < n_experts_) dispatch_count_[expert_id]++;
    // Load balancing: scale down bias for over-utilized experts
    uint64_t total = 0;
    for (auto c : dispatch_count_) total += c;
    if (total > 0 && dispatch_count_[expert_id] * n_experts_ > total * 2)
        bias_[expert_id] -= 0.01f;
}

} // namespace khanary::inference
