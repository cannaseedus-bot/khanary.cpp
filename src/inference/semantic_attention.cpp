#include "semantic_attention.h"

#include "kuhul/gram_store.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>

namespace khanary::inference {
namespace {

uint64_t fnv1a64(const std::string& value) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : value) {
        h ^= static_cast<uint64_t>(ch);
        h *= 1099511628211ull;
    }
    return h;
}

float signed_component(uint64_t hash, unsigned shift) {
    const auto part = static_cast<uint32_t>((hash >> shift) & 0xffffu);
    return (static_cast<float>(part) / 32767.5f) - 1.0f;
}

float class_weight(const std::string& semantic_class) {
    if (semantic_class == "trigram")          return 0.95f;
    if (semantic_class == "bigram")           return 0.92f;
    if (semantic_class == "atom")             return 0.90f;
    if (semantic_class == "quadgram")         return 0.86f;
    if (semantic_class == "tool_gram")        return 0.85f;
    if (semantic_class == "control_gram")     return 0.84f;
    if (semantic_class == "coarse_gram")      return 0.80f;
    if (semantic_class == "function_gram")    return 0.78f;
    if (semantic_class == "unigram")          return 0.72f;
    if (semantic_class == "lexicon")          return 0.68f;
    if (semantic_class == "word_similarity")  return 0.64f;
    if (semantic_class == "atom_contract")    return 0.62f;
    return 0.0f;
}

bool parse_candidate_suffix(const std::string& suffix,
                            std::string& semantic_class,
                            uint32_t& ordinal,
                            bool& has_ordinal) {
    const auto pos = suffix.find(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= suffix.size())
        return false;

    semantic_class = suffix.substr(0, pos);
    const std::string tail = suffix.substr(pos + 1);
    if (semantic_class.rfind("file", 0) == 0)
        return false;
    if (class_weight(semantic_class) <= 0.0f)
        return false;

    has_ordinal = !tail.empty() &&
        std::all_of(tail.begin(), tail.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
    ordinal = 1;
    if (has_ordinal) {
        try {
            ordinal = static_cast<uint32_t>(std::stoul(tail));
        } catch (...) {
            has_ordinal = false;
            ordinal = 1;
        }
    }
    return true;
}

std::string gram_text_from_value(const std::string& semantic_class,
                                 const std::string& value,
                                 const std::string& fallback) {
    if (!value.empty() && value.front() != '{') {
        const auto first = value.find('|');
        const auto second = first == std::string::npos
            ? std::string::npos
            : value.find('|', first + 1);
        if (first != std::string::npos && second != std::string::npos &&
            value.substr(0, first) == semantic_class && second + 1 < value.size())
            return value.substr(second + 1);
        return value;
    }
    return fallback;
}

} // namespace

std::array<float, 3> prompt_attention_vector_for(
    const std::string& semantic_class,
    const std::string& gram_text)
{
    const std::string seed = semantic_class + ":" + gram_text;
    float x = signed_component(fnv1a64("x:" + seed), 0);
    float y = signed_component(fnv1a64("y:" + seed), 8);
    float z = signed_component(fnv1a64("z:" + seed), 16);
    const float norm = std::sqrt(x*x + y*y + z*z);
    if (!std::isfinite(norm) || norm <= 1e-6f)
        return {1.0f, 0.0f, 0.0f};
    return {x / norm, y / norm, z / norm};
}

std::vector<PromptAttentionCandidate> collect_prompt_attention_candidates(
    const kuhul::GramStore& store,
    const std::string& prompt_key,
    size_t max_candidates)
{
    std::vector<PromptAttentionCandidate> out;
    if (prompt_key.empty() || max_candidates == 0)
        return out;

    const std::string prefix = prompt_key + ":";
    auto keys = store.keys();
    std::sort(keys.begin(), keys.end());

    for (const auto& key : keys) {
        if (key.rfind(prefix, 0) != 0)
            continue;
        const std::string suffix = key.substr(prefix.size());
        if (suffix.rfind("vector:", 0) == 0)
            continue;

        std::string semantic_class;
        uint32_t ordinal = 1;
        bool has_ordinal = false;
        if (!parse_candidate_suffix(suffix, semantic_class, ordinal, has_ordinal))
            continue;

        const auto gram = store.gramRead(key);
        if (!gram)
            continue;

        const float base = class_weight(semantic_class);
        const float rank_decay = has_ordinal
            ? 1.0f / (1.0f + 0.05f * static_cast<float>(ordinal > 0 ? ordinal - 1 : 0))
            : 0.90f;
        const float relevance = std::clamp(base * rank_decay, 0.0f, 1.0f);
        const std::string gram_text = gram_text_from_value(semantic_class, gram->value, suffix);

        out.push_back(PromptAttentionCandidate{
            key,
            relevance,
            prompt_attention_vector_for(semantic_class, gram_text)
        });
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        if (a.relevance != b.relevance) return a.relevance > b.relevance;
        return a.key < b.key;
    });
    if (out.size() > max_candidates)
        out.resize(max_candidates);
    return out;
}

std::unordered_map<std::string, std::array<float, 3>>
prompt_attention_vector_overrides(
    const std::vector<PromptAttentionCandidate>& candidates)
{
    std::unordered_map<std::string, std::array<float, 3>> out;
    out.reserve(candidates.size());
    for (const auto& candidate : candidates)
        out.emplace(candidate.key, candidate.vector);
    return out;
}

} // namespace khanary::inference
