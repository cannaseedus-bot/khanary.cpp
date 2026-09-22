#pragma once
#include <vector>
#include <cstdint>
#include <random>

namespace khanary::inference {

class Sampler {
public:
    explicit Sampler(uint64_t seed = 42);

    // Sample a token from logits using temperature + top-p + top-k
    uint32_t sample(
        const std::vector<float>& logits,
        float temperature = 0.7f,
        float top_p       = 0.9f,
        uint32_t top_k    = 40);

    // Greedy: argmax
    static uint32_t argmax(const std::vector<float>& logits);

    void set_seed(uint64_t seed);

private:
    std::mt19937_64 rng_;

    static std::vector<float> softmax(const std::vector<float>& logits, float temp);
};

} // namespace khanary::inference
