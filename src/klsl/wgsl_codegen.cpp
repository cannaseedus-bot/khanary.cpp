#include "wgsl_codegen.h"
#include <sstream>

namespace khanary::klsl {

std::string WGSLCodegen::glyph_wgsl(
    GlyphOp op,
    const std::string& result,
    const std::vector<std::string>& operands)
{
    auto op0 = operands.empty() ? "buf" : operands[0];
    auto op1 = operands.size() > 1 ? operands[1] : "buf";
    switch (op) {
        case GlyphOp::MATMUL:
            return "    let " + result + " = dot(" + op0 + "[id.x], " + op1 + "[id.x]);";
        case GlyphOp::SOFTMAX:
            return "    let " + result + " = exp(" + op0 + "[id.x]);";
        case GlyphOp::AFFINE:
            return "    let " + result + " = " + op0 + "[id.x] + " + op1 + "[0];";
        case GlyphOp::WAVE:
            return "    let " + result + " = sin(f32(id.x) * 0.1) * " + op0 + "[id.x];";
        default:
            return "    // " + std::string(glyph_op_name(op)) + " (wgsl)";
    }
}

std::string WGSLCodegen::emit(const IRFunction& fn, uint32_t group_x) {
    std::ostringstream ss;
    ss << "// KHANARY WGSL — " << fn.name << "\n";
    uint32_t slot = 0;
    for (const auto& p : fn.params)
        ss << "@group(0) @binding(" << slot++ << ") var<storage,read> " << p << " : array<f32>;\n";
    for (const auto& [name, val] : fn.values) {
        if (!val.is_param)
            ss << "@group(0) @binding(" << slot++ << ") var<storage,read_write> " << name << " : array<f32>;\n";
    }
    ss << "\n@compute @workgroup_size(" << group_x << ", 1, 1)\n";
    ss << "fn main(@builtin(global_invocation_id) id : vec3<u32>) {\n";
    for (const auto& blk : fn.blocks)
        for (const auto& ins : blk.instructions)
            ss << glyph_wgsl(ins.op, ins.result, ins.operands) << "\n";
    ss << "}\n";
    return ss.str();
}

std::string WGSLCodegen::emit_module(const IRModule& mod) {
    std::string out;
    for (const auto& fn : mod.functions) out += emit(fn) + "\n\n";
    return out;
}

} // namespace khanary::klsl
