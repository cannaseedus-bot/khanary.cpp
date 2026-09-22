#pragma once
#include "klsl_lexer.h"
#include "klsl_ir.h"
#include <string>
#include <memory>
#include <vector>

namespace khanary::klsl {

// AST nodes
struct ASTNode { virtual ~ASTNode() = default; };

struct TensorDecl : ASTNode {
    std::string name;
    std::vector<uint32_t> dims;
    std::string format; // "F32", "Q8", "F16"
};

struct GlyphStatement : ASTNode {
    GlyphOp op;
    std::string result;
    std::vector<std::string> operands;
    std::string phase;
};

struct PhaseBlock : ASTNode {
    std::string phase_name;
    std::vector<std::unique_ptr<ASTNode>> statements;
};

struct ClusterDecl : ASTNode {
    std::string name;
    std::vector<std::unique_ptr<TensorDecl>> tensors;
    std::vector<std::unique_ptr<PhaseBlock>> phases;
};

struct ModelDecl : ASTNode {
    std::string name;
    std::vector<std::string> clusters;
    std::vector<std::unique_ptr<GlyphStatement>> glyphs;
};

struct PipelineDecl : ASTNode {
    std::string name;
    std::vector<std::string> stages;
};

struct KLSLProgram {
    std::vector<std::unique_ptr<ClusterDecl>>  clusters;
    std::vector<std::unique_ptr<ModelDecl>>    models;
    std::vector<std::unique_ptr<PipelineDecl>> pipelines;
};

class KLSLParser {
public:
    explicit KLSLParser(std::vector<Token> tokens);

    KLSLProgram parse();
    const std::vector<std::string>& errors() const { return errors_; }

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;
    std::vector<std::string> errors_;

    const Token& peek(int offset = 0) const;
    const Token& consume();
    bool expect(TokenType t, const char* msg);
    bool at(TokenType t) const;

    std::unique_ptr<ClusterDecl>  parse_cluster();
    std::unique_ptr<ModelDecl>    parse_model();
    std::unique_ptr<PipelineDecl> parse_pipeline();
    std::unique_ptr<PhaseBlock>   parse_phase_block();
    std::unique_ptr<TensorDecl>   parse_tensor_decl();
    std::unique_ptr<GlyphStatement> parse_glyph_stmt();
};

// Lower an AST program into IR
IRModule lower(const KLSLProgram& prog, const std::string& source_path = "");

} // namespace khanary::klsl
