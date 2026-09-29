#include "phase_engine.h"
#include "xcfe_node_class.h"
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace khanary::kuhul {

namespace {

std::string upper_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

} // namespace

const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Pop:  return "Pop";
        case Phase::Wo:   return "Wo";
        case Phase::Yax:  return "Yax";
        case Phase::Sek:  return "Sek";
        case Phase::Chen: return "Ch'en";
        case Phase::Xul:  return "Xul";
    }
    return "Unknown";
}

Phase phase_from_name(const std::string& name) {
    Phase out;
    if (!phase_from_name(name, out))
        throw std::runtime_error("PhaseEngine: unknown phase '" + name + "'");
    return out;
}

bool phase_from_name(const std::string& name, Phase& out) {
    if (name == "Pop")         { out = Phase::Pop;  return true; }
    if (name == "Wo")          { out = Phase::Wo;   return true; }
    if (name == "Yax")         { out = Phase::Yax;  return true; }
    if (name == "Sek")         { out = Phase::Sek;  return true; }
    if (name == "Ch'en" || name == "Chen") { out = Phase::Chen; return true; }
    if (name == "Xul")         { out = Phase::Xul;  return true; }
    return false;
}

bool canonical_phase_for_semantic_atom(const std::string& atom, Phase& out) {
    // Promoted into the XCFE Node ISA (src/kuhul/xcfe_node_class.h): the
    // semantic atom IS the node class, and this is now only its name-based
    // front door, so the class -> phase authority has a single definition.
    XcfeNodeClass node_class{};
    if (!xcfe_node_class_from_name(atom, node_class)) return false;
    return canonical_phase_for_xcfe_class(node_class, out);
}

std::string canonical_phase_name_for_semantic_atom(const std::string& atom) {
    Phase p = Phase::Pop;
    if (!canonical_phase_for_semantic_atom(atom, p)) return "";
    return phase_name(p);
}

PhaseEngine::PhaseEngine() : current_(Phase::Pop) {}

Phase PhaseEngine::advance() {
    current_ = next_phase(current_);
    return current_;
}

double PhaseEngine::effective_weight(Phase from, Phase to) const {
    return arc_weight(from, to);
}

bool PhaseEngine::is_vertical(Phase from, Phase to) {
    return phase_distance(from, to) == 1;
}

bool PhaseEngine::is_horizontal(Phase from, Phase to) {
    return from == to;
}

} // namespace khanary::kuhul
