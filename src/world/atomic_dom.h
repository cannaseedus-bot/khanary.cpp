#pragma once
#include "../kuhul/phase_engine.h"
#include <string_view>
#include <array>
#include <cstdint>

namespace khanary::world {

// Seven cognitive operations mapped onto the six K-CUBE faces.
// [memory] and [evolve] share the Xul/Entropy face — same phase, different authority:
//   [memory] = gramWrite (unconditional Xul)
//   [evolve] = ΔG_semantic (Xul + ⟁EVOLVE⟁ gate required)
enum class AtomicDOMBlock : uint8_t {
    Perceive   = 0,  // Pop  / Phi       — incoming observation
    Learn      = 1,  // Wo   / Fold      — allocate/admit learning candidate
    Think      = 2,  // Yax  / Gram      — select, horizontal fold
    Evaluate   = 3,  // Sek  / Geodesic  — compute, execute
    Confidence = 4,  // Ch'en/ Projection — verify, evidence
    Memory     = 5,  // Xul  / Entropy   — gramWrite / state collapse
    Evolve     = 6,  // Xul  / Entropy   — ΔG_semantic, requires ⟁EVOLVE⟁
    COUNT      = 7
};

// Full descriptor for one K-CUBE face.
// Xul carries two blocks (Memory primary, Evolve gated).
struct FaceDescriptor {
    std::string_view  geometry;     // "Phi","Fold","Gram","Geodesic","Projection","Entropy"
    kuhul::Phase      phase;
    std::string_view  block;        // primary block label
    std::string_view  extra_block;  // second block on this face (empty if none)
    float             angle_deg;    // position on C₆ circle
};

// Canonical 6-face table — immutable.
constexpr std::array<FaceDescriptor, 6> KCUBE_FACES = {{
    {"Phi",        kuhul::Phase::Pop,  "perceive",    "",       0.0f},
    {"Fold",       kuhul::Phase::Wo,   "learn",       "",      60.0f},
    {"Gram",       kuhul::Phase::Yax,  "think",       "",     120.0f},
    {"Geodesic",   kuhul::Phase::Sek,  "evaluate",    "",     180.0f},
    {"Projection", kuhul::Phase::Chen, "confidence",  "",     240.0f},
    {"Entropy",    kuhul::Phase::Xul,  "memory",      "evolve",300.0f},
}};

// Block → face index (0..5). Evolve shares Xul/face-5 with Memory.
constexpr uint8_t block_face_index(AtomicDOMBlock b) noexcept {
    return (b == AtomicDOMBlock::Evolve) ? 5u
                                         : static_cast<uint8_t>(b);
}

// Phase → primary block (Xul maps to Memory; Evolve is gate-selected, not default).
constexpr AtomicDOMBlock phase_to_block(kuhul::Phase p) noexcept {
    switch (p) {
        case kuhul::Phase::Pop:  return AtomicDOMBlock::Perceive;
        case kuhul::Phase::Wo:   return AtomicDOMBlock::Learn;
        case kuhul::Phase::Yax:  return AtomicDOMBlock::Think;
        case kuhul::Phase::Sek:  return AtomicDOMBlock::Evaluate;
        case kuhul::Phase::Chen: return AtomicDOMBlock::Confidence;
        case kuhul::Phase::Xul:  return AtomicDOMBlock::Memory;
    }
    return AtomicDOMBlock::Perceive;
}

constexpr std::string_view block_name(AtomicDOMBlock b) noexcept {
    constexpr std::string_view names[] = {
        "perceive","learn","think","evaluate","confidence","memory","evolve"
    };
    auto idx = static_cast<uint8_t>(b);
    return (idx < 7) ? names[idx] : "unknown";
}

} // namespace khanary::world
