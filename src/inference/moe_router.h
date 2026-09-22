#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>

namespace khanary::inference {

struct ExpertAlloc {
    uint32_t expert_id;
    float    weight;
};

class MoERouter {
public:
    explicit MoERouter(uint32_t n_experts, uint32_t hidden_dim);

    // Route a hidden state to top-k experts
    // hidden: [hidden_dim] floats
    // Returns sorted (weight desc) expert allocations
    std::vector<ExpertAlloc> route(const std::vector<float>& hidden, uint32_t top_k = 2) const;

    // Load router weights from a flat float array [n_experts × hidden_dim]
    void load_weights(const float* data, size_t count);

    // Load routing_bias table from confidence.json style map
    void load_bias(const std::unordered_map<std::string, float>& bias);

    // Update utilization after dispatch (for load balancing)
    void record_dispatch(uint32_t expert_id);

    uint32_t n_experts() const { return n_experts_; }
    uint32_t hidden_dim() const { return hidden_dim_; }

private:
    uint32_t n_experts_;
    uint32_t hidden_dim_;
    std::vector<float> weights_;   // [n_experts * hidden_dim]
    std::vector<float> bias_;      // per-expert routing bias [n_experts]
    std::vector<uint64_t> dispatch_count_;

    std::vector<float> compute_logits(const std::vector<float>& hidden) const;
    static std::vector<float> softmax(const std::vector<float>& logits);
};

} // namespace khanary::inference
