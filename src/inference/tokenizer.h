#pragma once
#include "../config.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace khanary::inference {

// 1024-token glyph vocabulary
// Sentinel IDs: PAD=0(⊘), BOS=1(⟨), EOS=2(⟩), TOOL_CALL=11(⟁)
// Language tokens: (std::hash(word) % GLYPH_LANG_SLOTS) + GLYPH_LANG_OFFSET
class GlyphTokenizer {
public:
    GlyphTokenizer();

    // Encode text to token ids
    std::vector<uint32_t> encode(const std::string& text, bool add_bos = true) const;

    // Decode ids back to text
    std::string decode(const std::vector<uint32_t>& ids) const;

    // Get the vocab size
    uint32_t vocab_size() const { return GLYPH_VOCAB_SIZE; }

    // Token id for a word (hash-based, deterministic)
    static uint32_t word_to_id(const std::string& word);

    // Check if id is a sentinel
    static bool is_sentinel(uint32_t id);

    // Sentinel token strings
    static const char* sentinel_str(uint32_t id);

private:
    // Pre-built reverse map for sentinels
    std::unordered_map<uint32_t, std::string> id_to_str_;

    void init_sentinels();
    std::vector<std::string> split_tokens(const std::string& text) const;
};

} // namespace khanary::inference
