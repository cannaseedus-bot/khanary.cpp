#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace khanary::kuhul { class GramStore; }

namespace khanary::inference {

struct PromptAttentionCandidate {
    std::string key;
    float relevance = 0.0f;
    std::array<float, 3> vector{0.0f, 0.0f, 0.0f};
};

// Deterministic projection-only vector for a prompt gram row.  The vector
// supplies geometry for semantic attention; it does not carry semantic truth.
std::array<float, 3> prompt_attention_vector_for(
    const std::string& semantic_class,
    const std::string& gram_text);

// Collects bounded candidates from the GramStore rows written for one
// persisted prompt archive key.
std::vector<PromptAttentionCandidate> collect_prompt_attention_candidates(
    const kuhul::GramStore& store,
    const std::string& prompt_key,
    size_t max_candidates = 64);

std::unordered_map<std::string, std::array<float, 3>>
prompt_attention_vector_overrides(
    const std::vector<PromptAttentionCandidate>& candidates);

} // namespace khanary::inference
