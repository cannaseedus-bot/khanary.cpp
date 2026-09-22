#include "sampler.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace khanary::inference {

Sampler::Sampler(uint64_t seed) : rng_(seed) {}

void Sampler::set_seed(uint64_t seed) { rng_.seed(seed); }

std::vector<float> Sampler::softmax(const std::vector<float>& logits, float temp) {
    std::vector<float> p(logits.size());
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp((logits[i] - max_l) / temp);
        sum += p[i];
    }
    for (auto& v : p) v /= sum;
    return p;
}

uint32_t Sampler::argmax(const std::vector<float>& logits) {
    return (uint32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

uint32_t Sampler::sample(
    const std::vector<float>& logits,
    float temperature,
    float top_p,
    uint32_t top_k)
{
    if (logits.empty()) throw std::runtime_error("Sampler: empty logits");

    if (temperature <= 0.0f) return argmax(logits);

    auto probs = softmax(logits, temperature);
    size_t n = probs.size();

    // Build sorted index array
    std::vector<uint32_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(),
        indices.begin() + std::min((size_t)top_k, n),
        indices.end(),
        [&](uint32_t a, uint32_t b){ return probs[a] > probs[b]; });

    // Apply top-k cutoff
    size_t k = std::min((size_t)top_k, n);
    indices.resize(k);

    // Apply top-p cutoff
    float cumsum = 0.0f;
    size_t p_cut = k;
    for (size_t i = 0; i < k; ++i) {
        cumsum += probs[indices[i]];
        if (cumsum >= top_p) { p_cut = i + 1; break; }
    }
    indices.resize(p_cut);

    // Renormalize
    float sum = 0.0f;
    for (uint32_t idx : indices) sum += probs[idx];
    std::vector<float> weights;
    for (uint32_t idx : indices) weights.push_back(probs[idx] / sum);

    // Sample
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
    return indices[dist(rng_)];
}

} // namespace khanary::inference
