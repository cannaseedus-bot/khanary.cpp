#include "perception.h"
#include <functional>
#include <cmath>

namespace khanary::memory {

PerceptionStore::PerceptionStore() {}

void PerceptionStore::embed(
    const std::string& query,
    const std::vector<float>& confidence_context,
    uint32_t field_dim)
{
    field_.assign(field_dim, 0.0f);
    // Hash the query into field positions
    std::hash<std::string> hasher;
    size_t h = hasher(query);
    for (uint32_t i = 0; i < field_dim; ++i) {
        float q_comp = std::sin((float)((h >> (i % 64)) & 0xFF) * 0.1f + (float)i);
        float c_comp = (i < confidence_context.size()) ? confidence_context[i] : 0.0f;
        field_[i] = q_comp * 0.7f + c_comp * 0.3f;
    }
    // L1 normalize
    float norm = 0.0f;
    for (float v : field_) norm += std::abs(v);
    if (norm > 1e-6f) for (auto& v : field_) v /= norm;
}

void PerceptionStore::reset() {
    field_.clear();
}

} // namespace khanary::memory
