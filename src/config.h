#pragma once
#include <numbers>
#include <cstdint>
#include <cstddef>

// C6 phase angles (radians) — π/3 steps around the unit circle
constexpr double PHASE_POP  = 0.0;
constexpr double PHASE_WO   = std::numbers::pi / 3.0;
constexpr double PHASE_YAX  = 2.0 * std::numbers::pi / 3.0;
constexpr double PHASE_SEK  = std::numbers::pi;
constexpr double PHASE_CHEN = 4.0 * std::numbers::pi / 3.0;
constexpr double PHASE_XUL  = 5.0 * std::numbers::pi / 3.0;

// arc_weight(from, to) = cos(phase_to - phase_from)
// same=+1.0, adjacent=+0.5, two-hop=-0.5, opposing=-1.0

// GPU resident ceiling for Intel HD 4600
constexpr size_t HD4600_RESIDENT_CEILING_BYTES = 1792ULL * 1024 * 1024;

// Network ports
constexpr uint16_t PORT_MCP      = 8764;
constexpr uint16_t PORT_JSON_RPC = 8787;

// Glyph vocab
constexpr uint32_t GLYPH_VOCAB_SIZE   = 1024;
constexpr uint32_t GLYPH_PAD          = 0;    // ⊘
constexpr uint32_t GLYPH_BOS          = 1;    // ⟨
constexpr uint32_t GLYPH_EOS          = 2;    // ⟩
constexpr uint32_t GLYPH_TOOL_CALL    = 11;   // ⟁
constexpr uint32_t GLYPH_LANG_OFFSET  = 128;
constexpr uint32_t GLYPH_LANG_SLOTS   = 896;  // [128, 1023]

// Inference
constexpr uint32_t MAX_SEQ_LEN        = 2048;
constexpr uint32_t MAX_BATCH_SIZE     = 8;
constexpr uint32_t MoE_TOP_K          = 2;
constexpr float    DEFAULT_TEMPERATURE = 0.7f;
constexpr float    DEFAULT_TOP_P       = 0.9f;
constexpr uint32_t DEFAULT_TOP_K       = 40;

// Manifest expansion
constexpr uint32_t SCX_MAX_DEPTH      = 64;   // cycle detection depth limit

// Confidence gate
constexpr float CONFIDENCE_DELETE_FLOOR = 0.7f;

// Retry policy
constexpr uint32_t RETRY_MAX_ATTEMPTS = 3;
constexpr uint32_t RETRY_DELAY_MS     = 200;

// JROM journal
constexpr char JROM_DEFAULT_PATH[]    = "data/journal/tool-events.jsonl";
