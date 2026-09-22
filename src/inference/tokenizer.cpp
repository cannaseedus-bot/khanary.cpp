#include "tokenizer.h"
#include <sstream>
#include <cctype>
#include <functional>

namespace khanary::inference {

GlyphTokenizer::GlyphTokenizer() {
    init_sentinels();
}

void GlyphTokenizer::init_sentinels() {
    id_to_str_[GLYPH_PAD]       = "⊘";
    id_to_str_[GLYPH_BOS]       = "⟨";
    id_to_str_[GLYPH_EOS]       = "⟩";
    id_to_str_[3]                = "[SEP]";
    id_to_str_[4]                = "[MASK]";
    id_to_str_[5]                = "[UNK]";
    id_to_str_[6]                = "[CLS]";
    id_to_str_[7]                = "[PADDING]";
    id_to_str_[8]                = "[NEWLINE]";
    id_to_str_[9]                = "[TAB]";
    id_to_str_[10]               = "[SPACE]";
    id_to_str_[GLYPH_TOOL_CALL]  = "⟁";
    // 12–127: reserved for future system tokens
}

uint32_t GlyphTokenizer::word_to_id(const std::string& word) {
    std::hash<std::string> h;
    size_t hv = h(word);
    return (uint32_t)(hv % GLYPH_LANG_SLOTS) + GLYPH_LANG_OFFSET;
}

bool GlyphTokenizer::is_sentinel(uint32_t id) {
    return id < GLYPH_LANG_OFFSET;
}

const char* GlyphTokenizer::sentinel_str(uint32_t id) {
    switch (id) {
        case GLYPH_PAD:      return "⊘";
        case GLYPH_BOS:      return "⟨";
        case GLYPH_EOS:      return "⟩";
        case GLYPH_TOOL_CALL:return "⟁";
        default:             return "[SYS]";
    }
}

std::vector<std::string> GlyphTokenizer::split_tokens(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string cur;
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];
        if (std::isspace(c)) {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            ++i;
        } else if (std::ispunct(c) && c != '\'') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            tokens.push_back(std::string(1, (char)c));
            ++i;
        } else if (c >= 0x80) {
            // Multi-byte UTF-8: consume full codepoint
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            std::string glyph;
            glyph += (char)c; ++i;
            while (i < text.size() && (unsigned char)text[i] >= 0x80 && (unsigned char)text[i] < 0xC0) {
                glyph += text[i++];
            }
            tokens.push_back(glyph);
        } else {
            cur += (char)std::tolower(c);
            ++i;
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

std::vector<uint32_t> GlyphTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<uint32_t> ids;
    if (add_bos) ids.push_back(GLYPH_BOS);
    for (const auto& tok : split_tokens(text)) {
        ids.push_back(word_to_id(tok));
    }
    ids.push_back(GLYPH_EOS);
    return ids;
}

std::string GlyphTokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out;
    for (uint32_t id : ids) {
        if (id == GLYPH_BOS || id == GLYPH_EOS || id == GLYPH_PAD) continue;
        auto it = id_to_str_.find(id);
        if (it != id_to_str_.end()) {
            out += it->second;
        } else {
            // We can't perfectly reverse the hash, emit placeholder
            out += "[" + std::to_string(id) + "]";
        }
        out += ' ';
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

} // namespace khanary::inference
