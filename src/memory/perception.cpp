#include "perception.h"
#include <functional>
#include <cmath>
#include <cctype>
#include <algorithm>

namespace khanary::memory {

namespace {

static std::string canonicalize_query(const std::string& query) {
    std::string out;
    out.reserve(query.size());
    bool last_space = false;
    for (unsigned char ch : query) {
        if (std::isspace(ch)) {
            if (!out.empty() && !last_space) {
                out.push_back(' ');
                last_space = true;
            }
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(ch)));
        last_space = false;
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out.empty() ? std::string("__empty__") : out;
}

static float sanitize_confidence_component(float value) {
    if (!std::isfinite(value)) return 0.0f;
    return std::clamp(value, 0.0f, 1.0f);
}

} // namespace

PerceptionStore::PerceptionStore() {}

void PerceptionStore::embed(
    const std::string& query,
    const std::vector<float>& confidence_context,
    uint32_t field_dim)
{
    if (field_dim == 0) {
        field_.clear();
        return;
    }

    field_.assign(field_dim, 0.0f);
    // Hash the query into field positions
    std::hash<std::string> hasher;
    const std::string canonical_query = canonicalize_query(query);
    const size_t h = hasher(canonical_query);
    constexpr uint32_t kHashBits = static_cast<uint32_t>(sizeof(size_t) * 8u);
    for (uint32_t i = 0; i < field_dim; ++i) {
        const float q_comp = std::sin(
            static_cast<float>((h >> (i % kHashBits)) & 0xFFu) * 0.1f + static_cast<float>(i));
        const float c_comp = (i < confidence_context.size())
            ? sanitize_confidence_component(confidence_context[i])
            : 0.0f;
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
