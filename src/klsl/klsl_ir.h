#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace khanary::klsl {

enum class GlyphOp {
    MATMUL, CONV, ATTN, SOFTMAX, AFFINE, COMPRESS,
    SPHERICAL, TORSION, RADIAL, WAVE, NOP,
};

struct IRValue {
    std::string name;        // SSA name
    uint32_t    dim[4]{};    // tensor dimensions
    bool        is_param = false;
};

struct IRInstruction {
    GlyphOp                  op;
    std::string              result;   // output SSA name
    std::vector<std::string> operands; // input SSA names
    std::vector<uint32_t>    attrs;    // numeric attributes
    std::string              phase;    // which phase this belongs to
};

struct IRBlock {
    std::string                      name;
    std::vector<IRInstruction>       instructions;
};

struct IRFunction {
    std::string                              name;
    std::vector<std::string>                 params;
    std::vector<IRBlock>                     blocks;
    std::unordered_map<std::string, IRValue> values;

    void add_value(const std::string& val_name, const IRValue& v) { values[val_name] = v; }
    IRBlock& add_block(const std::string& block_name) {
        blocks.push_back({block_name, {}});
        return blocks.back();
    }
    IRBlock& current_block() { return blocks.back(); }
};

struct IRModule {
    std::vector<IRFunction> functions;
    std::string             source_path;
};

const char* glyph_op_name(GlyphOp op);
GlyphOp glyph_op_from_name(const std::string& name);

} // namespace khanary::klsl
