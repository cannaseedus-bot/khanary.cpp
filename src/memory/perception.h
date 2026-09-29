#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace khanary::memory {

// F_Q: ephemeral perception field
// Created at Pop, reset at Xul. NOT written to L_t.
// embed(Q, L_t) produces a field vector from the query and confidence context.
class PerceptionStore {
public:
    PerceptionStore();

    // Embed a query string + confidence context into an ephemeral field vector
    // field_dim: size of the output vector
    void embed(const std::string& query,
               const std::vector<float>& confidence_context,
               uint32_t field_dim = 256);

    // Access current field
    const std::vector<float>& field() const { return field_; }

    // Reset at Xul phase end (ephemeral)
    void reset();

    bool is_empty() const { return field_.empty(); }

private:
    std::vector<float> field_;
};

} // namespace khanary::memory
