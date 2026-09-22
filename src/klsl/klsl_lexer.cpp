#include "klsl_lexer.h"
#include <cctype>
#include <unordered_map>

namespace khanary::klsl {

static const std::unordered_map<std::string, TokenType> KEYWORDS = {
    {"Tensor",   TokenType::KW_TENSOR},
    {"Cluster",  TokenType::KW_CLUSTER},
    {"Model",    TokenType::KW_MODEL},
    {"Pipeline", TokenType::KW_PIPELINE},
    {"phase",    TokenType::KW_PHASE},
    {"fold",     TokenType::KW_FOLD},
    {"glyph",    TokenType::KW_GLYPH},
    {"import",   TokenType::KW_IMPORT},
    {"export",   TokenType::KW_EXPORT},
    {"Pop",      TokenType::PHASE_POP},
    {"Wo",       TokenType::PHASE_WO},
    {"Yax",      TokenType::PHASE_YAX},
    {"Sek",      TokenType::PHASE_SEK},
    {"Ch'en",    TokenType::PHASE_CHEN},
    {"Chen",     TokenType::PHASE_CHEN},
    {"Xul",      TokenType::PHASE_XUL},
    // Glyph opcodes
    {"MATMUL",   TokenType::GLYPH_MATMUL},
    {"CONV",     TokenType::GLYPH_CONV},
    {"ATTN",     TokenType::GLYPH_ATTN},
    {"SOFTMAX",  TokenType::GLYPH_SOFTMAX},
    {"AFFINE",   TokenType::GLYPH_AFFINE},
    {"COMPRESS", TokenType::GLYPH_COMPRESS},
    {"SPHERICAL",TokenType::GLYPH_SPHERICAL},
    {"TORSION",  TokenType::GLYPH_TORSION},
    {"RADIAL",   TokenType::GLYPH_RADIAL},
    {"WAVE",     TokenType::GLYPH_WAVE},
};

KLSLLexer::KLSLLexer(const std::string& source, const std::string& filename)
    : source_(source), filename_(filename)
{}

char KLSLLexer::peek(int offset) const {
    size_t p = pos_ + offset;
    return (p < source_.size()) ? source_[p] : '\0';
}

char KLSLLexer::consume() {
    char c = source_[pos_++];
    if (c == '\n') { ++line_; col_ = 1; }
    else ++col_;
    return c;
}

Token KLSLLexer::make(TokenType t, const std::string& text) {
    return {t, text, line_, col_};
}

void KLSLLexer::skip_ws_and_comments() {
    while (pos_ < source_.size()) {
        char c = peek();
        if (std::isspace((unsigned char)c)) { consume(); continue; }
        // Line comment //
        if (c == '/' && peek(1) == '/') {
            while (pos_ < source_.size() && peek() != '\n') consume();
            continue;
        }
        // Block comment /* */
        if (c == '/' && peek(1) == '*') {
            consume(); consume();
            while (pos_ + 1 < source_.size()) {
                if (peek() == '*' && peek(1) == '/') { consume(); consume(); break; }
                consume();
            }
            continue;
        }
        break;
    }
}

Token KLSLLexer::read_number() {
    std::string s;
    while (pos_ < source_.size() &&
           (std::isdigit((unsigned char)peek()) || peek() == '.' || peek() == 'e' || peek() == 'E' ||
            (s.size() > 0 && (peek() == '+' || peek() == '-'))))
        s += consume();
    return make(TokenType::NUMBER, s);
}

Token KLSLLexer::read_string() {
    consume(); // opening "
    std::string s;
    while (pos_ < source_.size() && peek() != '"') {
        if (peek() == '\\') { consume(); s += consume(); }
        else s += consume();
    }
    if (pos_ < source_.size()) consume(); // closing "
    return make(TokenType::STRING, s);
}

Token KLSLLexer::read_ident_or_kw() {
    std::string s;
    while (pos_ < source_.size() &&
           (std::isalnum((unsigned char)peek()) || peek() == '_' || peek() == '\''))
        s += consume();

    auto it = KEYWORDS.find(s);
    if (it != KEYWORDS.end()) return make(it->second, s);
    return make(TokenType::IDENT, s);
}

Token KLSLLexer::read_pi_expr() {
    // Consume π (multi-byte UTF-8 0xCF 0x80) and following /N
    std::string s;
    s += consume(); // first byte of π
    if (pos_ < source_.size() && (unsigned char)peek() >= 0x80) s += consume(); // second byte
    while (pos_ < source_.size() && (peek() == '/' || std::isdigit((unsigned char)peek())))
        s += consume();
    return make(TokenType::PI_EXPR, s);
}

std::vector<Token> KLSLLexer::tokenize() {
    std::vector<Token> tokens;
    while (true) {
        skip_ws_and_comments();
        if (pos_ >= source_.size()) {
            tokens.push_back(make(TokenType::EOF_TOK, ""));
            break;
        }
        char c = peek();
        if (c == '[') { consume(); tokens.push_back(make(TokenType::LBRACKET, "[")); }
        else if (c == ']') { consume(); tokens.push_back(make(TokenType::RBRACKET, "]")); }
        else if (c == '{') { consume(); tokens.push_back(make(TokenType::LBRACE, "{")); }
        else if (c == '}') { consume(); tokens.push_back(make(TokenType::RBRACE, "}")); }
        else if (c == '(') { consume(); tokens.push_back(make(TokenType::LPAREN, "(")); }
        else if (c == ')') { consume(); tokens.push_back(make(TokenType::RPAREN, ")")); }
        else if (c == ',') { consume(); tokens.push_back(make(TokenType::COMMA, ",")); }
        else if (c == ':') { consume(); tokens.push_back(make(TokenType::COLON, ":")); }
        else if (c == ';') { consume(); tokens.push_back(make(TokenType::SEMICOLON, ";")); }
        else if (c == '=' && peek(1) != '>') { consume(); tokens.push_back(make(TokenType::EQ, "=")); }
        else if (c == '-' && peek(1) == '>') { consume(); consume(); tokens.push_back(make(TokenType::ARROW, "->")); }
        else if (c == '"') tokens.push_back(read_string());
        else if (std::isdigit((unsigned char)c)) tokens.push_back(read_number());
        else if (std::isalpha((unsigned char)c) || c == '_') tokens.push_back(read_ident_or_kw());
        else if ((unsigned char)c == 0xCF && pos_ + 1 < source_.size() && (unsigned char)source_[pos_+1] == 0x80)
            tokens.push_back(read_pi_expr()); // π
        else { tokens.push_back(make(TokenType::UNKNOWN, std::string(1, c))); consume(); }
    }
    return tokens;
}

} // namespace khanary::klsl
