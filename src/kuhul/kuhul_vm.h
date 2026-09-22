#pragma once
#include "phase_engine.h"
#include "fold_engine.h"
#include "xcfe_executor.h"
#include "gram_store.h"
#include <string>
#include <vector>
#include <variant>
#include <stack>

namespace khanary::micronaut { class SandboxInspector; }

namespace khanary::kuhul {

// VM value types on the stack
struct TensorRef { std::string id; std::string type; };
struct StringVal { std::string s; };
struct NumberVal { double n; };
struct NullVal {};

using VMValue = std::variant<NullVal, StringVal, NumberVal, TensorRef>;

// Bracket program instruction
struct Instruction {
    Phase   phase;          // which bracket this is
    std::string opcode;     // e.g. "Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"
    std::string id;         // optional identifier
    std::string type;       // optional type (for [Wo id type])
    std::vector<std::string> args; // optional args (for [Sek glyph args...])
};

// Result of running a program
struct VMResult {
    bool ok;
    std::string error;
    std::vector<VMValue> stack;  // final stack state
};

class KuhulVM {
public:
    KuhulVM(PhaseEngine& pe, FoldEngine& fe, XCFEExecutor& xcfe, GramStore& gs);

    // Attach sandbox inspector — enforces gramRead/gramWrite at fold-op boundaries.
    // Call once after construction. Pass nullptr to disable enforcement.
    void set_sandbox(micronaut::SandboxInspector* si) { sandbox_ = si; }

    // Parse and execute a bracket program string
    // e.g. "[Pop] [Wo T1 Tensor] [Yax T1] [Sek MATMUL T1 T2 T3] [Ch'en T3 0.9] [Xul]"
    VMResult run(const std::string& program, const XCFEContext& ctx = {});

    // Execute a pre-parsed instruction sequence
    VMResult exec(const std::vector<Instruction>& instructions, const XCFEContext& ctx = {});

    // Parse a bracket program into instructions
    static std::vector<Instruction> parse_program(const std::string& program);

    PhaseEngine&  phase_engine() { return pe_; }
    FoldEngine&   fold_engine()  { return fe_; }
    GramStore&    gram_store()   { return gs_; }

private:
    PhaseEngine&  pe_;
    FoldEngine&   fe_;
    XCFEExecutor& xcfe_;
    GramStore&    gs_;

    micronaut::SandboxInspector* sandbox_ = nullptr;

    std::stack<VMValue> stack_;

    void exec_pop(const Instruction& ins, const XCFEContext& ctx);
    void exec_wo(const Instruction& ins, const XCFEContext& ctx);
    void exec_yax(const Instruction& ins, const XCFEContext& ctx);
    void exec_sek(const Instruction& ins, const XCFEContext& ctx);
    void exec_chen(const Instruction& ins, const XCFEContext& ctx);
    void exec_xul(const Instruction& ins, const XCFEContext& ctx);
};

} // namespace khanary::kuhul
