#pragma once
#include <string>
#include <vector>

namespace khanary::klsl {

enum class TokenType {
    // Keywords
    KW_TENSOR, KW_CLUSTER, KW_MODEL, KW_PIPELINE,
    KW_PHASE, KW_FOLD, KW_GLYPH, KW_IMPORT, KW_EXPORT,
    // Phase names
    PHASE_POP, PHASE_WO, PHASE_YAX, PHASE_SEK, PHASE_CHEN, PHASE_XUL,
    // Glyph opcodes
    GLYPH_MATMUL, GLYPH_CONV, GLYPH_ATTN, GLYPH_SOFTMAX,
    GLYPH_AFFINE, GLYPH_COMPRESS, GLYPH_SPHERICAL,
    GLYPH_TORSION, GLYPH_RADIAL, GLYPH_WAVE,
    // Punctuation
    LBRACKET, RBRACKET, LBRACE, RBRACE, LPAREN, RPAREN,
    COMMA, COLON, SEMICOLON, ARROW, EQ,
    // Literals
    IDENT, NUMBER, STRING, PI_EXPR,  // "π/3" etc.
    // Special
    EOF_TOK, UNKNOWN,
};

struct Token {
    TokenType   type;
    std::string text;
    size_t      line;
    size_t      col;
};

class KLSLLexer {
public:
    explicit KLSLLexer(const std::string& source, const std::string& filename = "<input>");

    std::vector<Token> tokenize();
    const std::string& filename() const { return filename_; }

private:
    std::string source_;
    std::string filename_;
    size_t      pos_  = 0;
    size_t      line_ = 1;
    size_t      col_  = 1;

    char peek(int offset = 0) const;
    char consume();
    void skip_ws_and_comments();
    Token read_number();
    Token read_string();
    Token read_ident_or_kw();
    Token read_pi_expr();
    Token make(TokenType t, const std::string& text);
};

} // namespace khanary::klsl
