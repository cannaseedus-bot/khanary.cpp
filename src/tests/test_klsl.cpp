#include "conformance.h"
#include "../klsl/klsl_parser.h"

using namespace khanary;
int main() {
    return test::run("klsl", [](test::Suite& suite) {
        const std::string source = "Cluster sample { Tensor a [4, 8] F32 phase Sek { AFFINE out = a scale bias; } }";
        klsl::KLSLParser parser(klsl::KLSLLexer(source).tokenize());
        const auto program = parser.parse();
        if (!parser.errors().empty()) for (const auto& error : parser.errors()) std::cerr << "KLSL: " << error << "\n";
        suite.check("valid_cluster", parser.errors().empty() && program.clusters.size() == 1);
        const auto module = klsl::lower(program, "fixture.klsl");
        suite.check("lowered_dimensions", module.functions.size() == 1 &&
            module.functions[0].values.at("a").dim[0] == 4 && module.functions[0].values.at("a").dim[1] == 8);
        const auto& ins = module.functions.at(0).blocks.at(1).instructions.at(0);
        suite.check("lowered_operation", ins.op == klsl::GlyphOp::AFFINE && ins.phase == "Sek" &&
            ins.result == "out" && ins.operands == std::vector<std::string>{"a", "scale", "bias"});
        for (const char* dimension : {"bad", "0", "1.5", "4294967296"}) {
            klsl::KLSLParser invalid(klsl::KLSLLexer(std::string("Cluster sample { Tensor a [") + dimension + "] F32 }").tokenize());
            (void)invalid.parse();
            suite.check(std::string("invalid_dimension/") + dimension, !invalid.errors().empty());
        }
        klsl::KLSLParser unclosed(klsl::KLSLLexer("Cluster sample {").tokenize());
        (void)unclosed.parse();
        suite.check("unclosed_cluster", !unclosed.errors().empty());
    }, 8);
}
