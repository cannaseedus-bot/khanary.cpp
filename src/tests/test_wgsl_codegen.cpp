#include "conformance.h"
#include "../klsl/klsl_lexer.h"
#include "../klsl/klsl_parser.h"
#include "../klsl/wgsl_codegen.h"

using namespace khanary;

int main() {
    return test::run("wgsl-codegen", [](test::Suite& suite) {
        const std::string source =
            "Cluster sample { "
            "Tensor a [4, 8] F32 "
            "Tensor scale [1] F32 "
            "phase Sek { AFFINE out = a scale; } "
            "}";

        klsl::KLSLParser parser(klsl::KLSLLexer(source).tokenize());
        const auto program = parser.parse();
        suite.check("parse_ok", parser.errors().empty() && program.clusters.size() == 1);

        const auto module = klsl::lower(program, "wgsl_fixture.klsl");
        suite.check("lower_has_function", module.functions.size() == 1);

        klsl::WGSLCodegen wgsl;
        const auto emitted = wgsl.emit_module(module);
        suite.check("contains_compute_entry", emitted.find("@compute") != std::string::npos);
        suite.check("contains_binding_layout", emitted.find("@group(0) @binding(") != std::string::npos);
        suite.check("contains_affine_instruction", emitted.find("let out =") != std::string::npos);
    }, 5);
}
