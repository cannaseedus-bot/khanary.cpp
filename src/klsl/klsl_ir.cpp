#include "klsl_ir.h"

namespace khanary::klsl {

const char* glyph_op_name(GlyphOp op) {
    switch (op) {
        case GlyphOp::MATMUL:   return "MATMUL";
        case GlyphOp::CONV:     return "CONV";
        case GlyphOp::ATTN:     return "ATTN";
        case GlyphOp::SOFTMAX:  return "SOFTMAX";
        case GlyphOp::AFFINE:   return "AFFINE";
        case GlyphOp::COMPRESS: return "COMPRESS";
        case GlyphOp::SPHERICAL:return "SPHERICAL";
        case GlyphOp::TORSION:  return "TORSION";
        case GlyphOp::RADIAL:   return "RADIAL";
        case GlyphOp::WAVE:     return "WAVE";
        case GlyphOp::NOP:      return "NOP";
    }
    return "NOP";
}

GlyphOp glyph_op_from_name(const std::string& name) {
    if (name == "MATMUL")    return GlyphOp::MATMUL;
    if (name == "CONV")      return GlyphOp::CONV;
    if (name == "ATTN")      return GlyphOp::ATTN;
    if (name == "SOFTMAX")   return GlyphOp::SOFTMAX;
    if (name == "AFFINE")    return GlyphOp::AFFINE;
    if (name == "COMPRESS")  return GlyphOp::COMPRESS;
    if (name == "SPHERICAL") return GlyphOp::SPHERICAL;
    if (name == "TORSION")   return GlyphOp::TORSION;
    if (name == "RADIAL")    return GlyphOp::RADIAL;
    if (name == "WAVE")      return GlyphOp::WAVE;
    return GlyphOp::NOP;
}

} // namespace khanary::klsl
