#pragma once
#include "klsl_ir.h"
#include <string>

namespace khanary::klsl {

// Generate WGSL (WebGPU Shading Language) from IR
class WGSLCodegen {
public:
    std::string emit(const IRFunction& fn, uint32_t group_x = 64);
    std::string emit_module(const IRModule& mod);

private:
    static std::string glyph_wgsl(GlyphOp op,
                                   const std::string& result,
                                   const std::vector<std::string>& operands);
};

} // namespace khanary::klsl
