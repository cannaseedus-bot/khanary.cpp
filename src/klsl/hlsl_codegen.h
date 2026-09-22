#pragma once
#include "klsl_ir.h"
#include <string>

namespace khanary::klsl {

// Generate HLSL cs_5_0 compute shader text from IR
class HLSLCodegen {
public:
    // Emit a complete HLSL cs_5_0 shader for one IRFunction
    std::string emit(const IRFunction& fn, uint32_t group_x = 64, uint32_t group_y = 1);

    // Emit all functions in a module
    std::string emit_module(const IRModule& mod);

private:
    std::string emit_header(const IRFunction& fn, uint32_t gx, uint32_t gy);
    std::string emit_body(const IRFunction& fn);
    std::string emit_instruction(const IRInstruction& ins);
    static std::string glyph_hlsl(GlyphOp op,
                                   const std::string& result,
                                   const std::vector<std::string>& operands);
};

} // namespace khanary::klsl
