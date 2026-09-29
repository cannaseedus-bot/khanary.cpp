#include "klsl_parser.h"
#include <stdexcept>

namespace khanary::klsl {

static Token EOF_TOKEN{TokenType::EOF_TOK, "", 0, 0};

KLSLParser::KLSLParser(std::vector<Token> tokens)
    : tokens_(std::move(tokens))
{}

const Token& KLSLParser::peek(int offset) const {
    size_t i = pos_ + offset;
    return (i < tokens_.size()) ? tokens_[i] : EOF_TOKEN;
}

const Token& KLSLParser::consume() {
    return (pos_ < tokens_.size()) ? tokens_[pos_++] : EOF_TOKEN;
}

bool KLSLParser::at(TokenType t) const {
    return peek().type == t;
}

bool KLSLParser::expect(TokenType t, const char* msg) {
    if (!at(t)) {
        errors_.push_back(std::string(msg) + " (got '" + peek().text + "')");
        return false;
    }
    consume();
    return true;
}

KLSLProgram KLSLParser::parse() {
    KLSLProgram prog;
    while (!at(TokenType::EOF_TOK)) {
        if (at(TokenType::KW_CLUSTER)) {
            prog.clusters.push_back(parse_cluster());
        } else if (at(TokenType::KW_MODEL)) {
            prog.models.push_back(parse_model());
        } else if (at(TokenType::KW_PIPELINE)) {
            prog.pipelines.push_back(parse_pipeline());
        } else {
            errors_.push_back("Unexpected token: " + peek().text);
            consume();
        }
    }
    return prog;
}

std::unique_ptr<TensorDecl> KLSLParser::parse_tensor_decl() {
    consume(); // KW_TENSOR
    auto t = std::make_unique<TensorDecl>();
    t->name = consume().text; // name
    // Optional dims: [M, N, ...]
    if (at(TokenType::LBRACKET)) {
        consume();
        while (!at(TokenType::RBRACKET) && !at(TokenType::EOF_TOK)) {
            if (at(TokenType::NUMBER)) {
                const auto text = consume().text;
                if (text.find_first_not_of("0123456789") != std::string::npos) {
                    errors_.push_back("tensor dimension must be an unsigned integer");
                } else {
                    try {
                        const auto value = std::stoull(text);
                        if (value == 0 || value > UINT32_MAX) errors_.push_back("tensor dimension out of range");
                        else t->dims.push_back(static_cast<uint32_t>(value));
                    } catch (const std::exception&) { errors_.push_back("tensor dimension out of range"); }
                }
            } else if (at(TokenType::COMMA)) consume();
            else {
                errors_.push_back("expected numeric tensor dimension");
                consume(); // Invalid dimensions must not stall the parser.
            }
        }
        expect(TokenType::RBRACKET, "expected ']'");
    }
    // Optional format: F32/Q8/F16
    if (at(TokenType::IDENT)) t->format = consume().text;
    return t;
}

std::unique_ptr<GlyphStatement> KLSLParser::parse_glyph_stmt() {
    auto stmt = std::make_unique<GlyphStatement>();
    // glyph opcode result = operand1 operand2 ...
    stmt->op = glyph_op_from_name(consume().text);
    stmt->result = consume().text;
    if (at(TokenType::EQ)) consume();
    while (at(TokenType::IDENT) || at(TokenType::NUMBER)) {
        stmt->operands.push_back(consume().text);
    }
    if (at(TokenType::SEMICOLON)) consume();
    return stmt;
}

std::unique_ptr<PhaseBlock> KLSLParser::parse_phase_block() {
    consume(); // KW_PHASE
    auto blk = std::make_unique<PhaseBlock>();
    blk->phase_name = consume().text; // phase name token
    expect(TokenType::LBRACE, "expected '{'");
    while (!at(TokenType::RBRACE) && !at(TokenType::EOF_TOK)) {
        if (at(TokenType::GLYPH_MATMUL) || at(TokenType::GLYPH_CONV) ||
            at(TokenType::GLYPH_ATTN) || at(TokenType::GLYPH_SOFTMAX) ||
            at(TokenType::GLYPH_AFFINE) || at(TokenType::GLYPH_COMPRESS) ||
            at(TokenType::GLYPH_SPHERICAL) || at(TokenType::GLYPH_TORSION) ||
            at(TokenType::GLYPH_RADIAL) || at(TokenType::GLYPH_WAVE)) {
            blk->statements.push_back(parse_glyph_stmt());
        } else {
            consume(); // skip unknown
        }
    }
    expect(TokenType::RBRACE, "expected '}'");
    return blk;
}

std::unique_ptr<ClusterDecl> KLSLParser::parse_cluster() {
    consume(); // KW_CLUSTER
    auto cl = std::make_unique<ClusterDecl>();
    cl->name = consume().text;
    expect(TokenType::LBRACE, "expected '{'");
    while (!at(TokenType::RBRACE) && !at(TokenType::EOF_TOK)) {
        if (at(TokenType::KW_TENSOR)) cl->tensors.push_back(parse_tensor_decl());
        else if (at(TokenType::KW_PHASE)) cl->phases.push_back(parse_phase_block());
        else consume();
    }
    expect(TokenType::RBRACE, "expected '}'");
    return cl;
}

std::unique_ptr<ModelDecl> KLSLParser::parse_model() {
    consume(); // KW_MODEL
    auto m = std::make_unique<ModelDecl>();
    m->name = consume().text;
    expect(TokenType::LBRACE, "expected '{'");
    while (!at(TokenType::RBRACE) && !at(TokenType::EOF_TOK)) {
        if (at(TokenType::IDENT)) m->clusters.push_back(consume().text);
        else if (at(TokenType::COMMA)) consume();
        else consume();
    }
    expect(TokenType::RBRACE, "expected '}'");
    return m;
}

std::unique_ptr<PipelineDecl> KLSLParser::parse_pipeline() {
    consume(); // KW_PIPELINE
    auto p = std::make_unique<PipelineDecl>();
    p->name = consume().text;
    expect(TokenType::LBRACE, "expected '{'");
    while (!at(TokenType::RBRACE) && !at(TokenType::EOF_TOK)) {
        if (at(TokenType::IDENT)) p->stages.push_back(consume().text);
        else if (at(TokenType::ARROW) || at(TokenType::COMMA)) consume();
        else consume();
    }
    expect(TokenType::RBRACE, "expected '}'");
    return p;
}

IRModule lower(const KLSLProgram& prog, const std::string& source_path) {
    IRModule mod;
    mod.source_path = source_path;
    for (const auto& cluster : prog.clusters) {
        IRFunction fn;
        fn.name = cluster->name;
        // Add entry block
        fn.add_block("entry");
        // Register tensors as values
        for (const auto& t : cluster->tensors) {
            IRValue v;
            v.name = t->name;
            v.is_param = true;
            for (size_t i = 0; i < t->dims.size() && i < 4; ++i) v.dim[i] = t->dims[i];
            fn.add_value(t->name, v);
            fn.params.push_back(t->name);
        }
        // Lower phase blocks
        for (const auto& phase : cluster->phases) {
            auto& blk = fn.add_block(phase->phase_name);
            for (const auto& stmt : phase->statements) {
                if (auto* gs = dynamic_cast<const GlyphStatement*>(stmt.get())) {
                    IRInstruction ins;
                    ins.op      = gs->op;
                    ins.result  = gs->result;
                    ins.operands= gs->operands;
                    ins.phase   = phase->phase_name;
                    blk.instructions.push_back(ins);
                    // Register result value
                    IRValue rv; rv.name = gs->result;
                    fn.add_value(gs->result, rv);
                }
            }
        }
        mod.functions.push_back(std::move(fn));
    }
    return mod;
}

} // namespace khanary::klsl
