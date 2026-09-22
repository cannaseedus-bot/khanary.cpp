#include "kuhul_vm.h"
#include "../micronaut/sandbox_inspector.h"
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace khanary::kuhul {

KuhulVM::KuhulVM(PhaseEngine& pe, FoldEngine& fe, XCFEExecutor& xcfe, GramStore& gs)
    : pe_(pe), fe_(fe), xcfe_(xcfe), gs_(gs)
{
    // Wire fold engine's XCFE authorization to our executor
    fe_.set_xcfe_auth([this](Phase from, Phase to, const std::string& node) {
        XCFEContext ctx{{"node", node}};
        return xcfe_.authorize_phase_transition(from, to, ctx);
    });
}

// Parse: "[Pop] [Wo id type] [Yax id] [Sek glyph a b] [Ch'en id val] [Xul]"
std::vector<Instruction> KuhulVM::parse_program(const std::string& program) {
    std::vector<Instruction> instructions;
    size_t i = 0;
    while (i < program.size()) {
        while (i < program.size() && std::isspace((unsigned char)program[i])) ++i;
        if (i >= program.size()) break;
        if (program[i] != '[') { ++i; continue; }
        ++i; // skip [
        size_t end = program.find(']', i);
        if (end == std::string::npos) break;
        std::string tok = program.substr(i, end - i);
        i = end + 1;

        std::istringstream ss(tok);
        std::vector<std::string> words;
        std::string w;
        while (ss >> w) words.push_back(w);
        if (words.empty()) continue;

        Instruction ins;
        ins.opcode = words[0];
        if      (ins.opcode == "Pop")   ins.phase = Phase::Pop;
        else if (ins.opcode == "Wo")    ins.phase = Phase::Wo;
        else if (ins.opcode == "Yax")   ins.phase = Phase::Yax;
        else if (ins.opcode == "Sek")   ins.phase = Phase::Sek;
        else if (ins.opcode == "Ch'en" || ins.opcode == "Chen") { ins.opcode = "Ch'en"; ins.phase = Phase::Chen; }
        else if (ins.opcode == "Xul")   ins.phase = Phase::Xul;
        else ins.phase = Phase::Pop; // unknown

        if (words.size() > 1) ins.id = words[1];
        if (words.size() > 2) ins.type = words[2];
        for (size_t j = 3; j < words.size(); ++j) ins.args.push_back(words[j]);

        instructions.push_back(ins);
    }
    return instructions;
}

VMResult KuhulVM::run(const std::string& program, const XCFEContext& ctx) {
    auto instructions = parse_program(program);
    return exec(instructions, ctx);
}

VMResult KuhulVM::exec(const std::vector<Instruction>& instructions, const XCFEContext& ctx) {
    // Clear stack for this run
    while (!stack_.empty()) stack_.pop();
    pe_.set(Phase::Pop);

    try {
        for (const auto& ins : instructions) {
            switch (ins.phase) {
                case Phase::Pop:  exec_pop(ins, ctx);  break;
                case Phase::Wo:   exec_wo(ins, ctx);   break;
                case Phase::Yax:  exec_yax(ins, ctx);  break;
                case Phase::Sek:  exec_sek(ins, ctx);  break;
                case Phase::Chen: exec_chen(ins, ctx); break;
                case Phase::Xul:  exec_xul(ins, ctx);  break;
            }
        }
    } catch (const std::exception& e) {
        return VMResult{false, e.what(), {}};
    }

    VMResult r;
    r.ok = true;
    // Drain the stack
    while (!stack_.empty()) {
        r.stack.insert(r.stack.begin(), stack_.top());
        stack_.pop();
    }
    return r;
}

void KuhulVM::exec_pop(const Instruction& ins, const XCFEContext& ctx) {
    pe_.set(Phase::Pop);
    if (!ins.id.empty()) {
        // gramRead enforcement: check sandbox before accessing gram store
        if (sandbox_) {
            auto it = ctx.find("track_id");
            if (it != ctx.end() && !sandbox_->check_effect(it->second, "gramRead"))
                throw std::runtime_error("gramRead denied by sandbox for track: " + it->second);
        }
        auto g = gs_.gramRead(ins.id);
        if (g) stack_.push(StringVal{g->value});
        else   stack_.push(NullVal{});
    }
    (void)ctx;
}

void KuhulVM::exec_wo(const Instruction& ins, const XCFEContext& ctx) {
    auto r = fe_.vertical_fold(ins.id, Phase::Pop, Phase::Wo);
    if (!r.authorized) throw std::runtime_error("Wo: " + r.error);
    pe_.set(Phase::Wo);
    if (!ins.id.empty()) {
        // Allocate tensor reference
        stack_.push(TensorRef{ins.id, ins.type.empty() ? "Tensor" : ins.type});
    }
    (void)ctx;
}

void KuhulVM::exec_yax(const Instruction& ins, const XCFEContext& ctx) {
    auto r = fe_.vertical_fold(ins.id, Phase::Wo, Phase::Yax);
    if (!r.authorized) throw std::runtime_error("Yax: " + r.error);
    pe_.set(Phase::Yax);
    // Yax: select — peek top of stack if id matches
    if (!ins.id.empty() && !stack_.empty()) {
        // Horizontal fold — node selection at Yax phase
        fe_.horizontal_fold(ins.id, ins.id, Phase::Yax);
    }
    (void)ctx;
}

void KuhulVM::exec_sek(const Instruction& ins, const XCFEContext& ctx) {
    auto r = fe_.vertical_fold(ins.id, Phase::Yax, Phase::Sek);
    if (!r.authorized) throw std::runtime_error("Sek: " + r.error);
    pe_.set(Phase::Sek);
    // Sek: compute — push glyph dispatch result
    // ins.id = glyph name, ins.args = operand tensor ids
    StringVal op{ins.id};
    for (const auto& arg : ins.args) op.s += " " + arg;
    stack_.push(op);
    (void)ctx;
}

void KuhulVM::exec_chen(const Instruction& ins, const XCFEContext& ctx) {
    auto r = fe_.vertical_fold(ins.id, Phase::Sek, Phase::Chen);
    if (!r.authorized) throw std::runtime_error("Ch'en: " + r.error);
    pe_.set(Phase::Chen);
    // Ch'en: verify/evidence — store a confidence value
    if (!ins.id.empty() && !ins.type.empty()) {
        // ins.id = tensor/value id, ins.type = confidence score
        gs_.gramWrite("confidence:" + ins.id, ins.type, "Ch'en");
    }
    (void)ctx;
}

void KuhulVM::exec_xul(const Instruction& ins, const XCFEContext& ctx) {
    auto r = fe_.vertical_fold(ins.id, Phase::Chen, Phase::Xul);
    if (!r.authorized) throw std::runtime_error("Xul: " + r.error);
    pe_.set(Phase::Xul);
    if (!ins.id.empty() && !stack_.empty()) {
        // gramWrite enforcement: check sandbox before committing
        if (sandbox_) {
            auto it = ctx.find("track_id");
            if (it != ctx.end() && !sandbox_->check_effect(it->second, "gramWrite"))
                throw std::runtime_error("gramWrite denied by sandbox for track: " + it->second);
        }
        auto& top = stack_.top();
        if (auto* sv = std::get_if<StringVal>(&top)) {
            gs_.gramWrite(ins.id, sv->s, "Xul");
        }
        stack_.pop();
    }
    (void)ctx;
}

} // namespace khanary::kuhul
