#pragma once
//
// Phase-supplied greeting style — the pure half of the adaptive salutation
// path. Everything here is deterministic and depends only on util/json, so the
// greeting-learning policy is unit-testable without a Factory, a GramStore, or
// the GPU runtime. This mirrors the inline implementation that previously lived
// in src/main.cpp; behavior is intentionally identical.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace khanary::micronaut::greeting {

// One phase-supplied greeting register from data/grammar/greeting-bank.json.
struct BankEntry {
    std::string phase;  // Pop | Wo | Yax | Sek | Ch'en | Xul | <phase>.<variant>
    std::string kind;   // canonical | learned | associative | status |
                        // confirmatory | recall | reactionary | synthesized
    std::vector<std::string> templates;
};

// Counters the templates interpolate ({turns}, {lingo_count}).
struct PlaceholderContext {
    uint32_t turns = 0;
    std::size_t lingo_count = 0;
};

// Parse the greeting bank. Entries missing a phase or templates are dropped,
// matching the runtime's tolerant loader (a malformed bank must never crash a
// greeting).
std::vector<BankEntry> parse_bank(const std::string& json_text);

// Built-in fallbacks for every phase key, used when the bank is absent or a
// supplied template cannot be filled.
const std::unordered_map<std::string, std::vector<std::string>>& builtin_templates();

// Learned lingo cues ranked by hit count (descending), ties broken
// lexicographically so the choice is stable across runs.
std::vector<std::string> rank_lingo_cues(const std::unordered_map<std::string, uint32_t>& hits);

std::string title_case(const std::string& text);

// Replace every occurrence of {key} in text with value.
void replace_placeholder(std::string& text, const std::string& key, const std::string& value);

// Fill a template. Returns an empty string when the template needs a value the
// runtime has not actually observed — the greeting surface never fakes lingo it
// has not learned.
std::string resolve_placeholders(const std::string& tmpl,
                                 const std::string& cue_display,
                                 const std::string& user_greeting,
                                 const std::string& last_greeting,
                                 const std::string& synthesized_word,
                                 const PlaceholderContext& ctx);

// Select a clause for a phase key: supplied bank entries first, built-in
// fallbacks second. Templates rotate by seed so repeated greetings vary, and
// unfillable templates are skipped rather than emitted half-substituted.
std::string select_clause(const std::vector<BankEntry>& bank,
                          const std::string& phase_key,
                          const std::string& cue_display,
                          const std::string& user_greeting,
                          const std::string& last_greeting,
                          const std::string& synthesized_word,
                          uint32_t seed,
                          const PlaceholderContext& ctx);

// Token signature of a salutation prompt (at most 12 tokens / 64 chars).
// Repeats of the same signature become reactionary mirrors.
std::string observed_form(const std::vector<std::string>& tokens);

// ── Adaptive salutation policy ──────────────────────────────────────────────

enum class SalutationRoute {
    LocaleChinese,        // locale greeting, terminal (not adaptable)
    ReactiveMirror,       // user repeated a form the runtime archived
    LearnedWaveSlang,     // learned slang register (wave)
    LearnedSlang,         // learned slang register (big dog)
    RandomChat,           // small-talk register
    SynthesizedGreeting,  // runtime's own blended word
    AssociativeGreeting,  // cue-shaped associative variant
    PhaseCycle,           // C6 phase-cycled register from the bank
    Neutral               // canonical neutral greeting
};

struct SalutationCandidate {
    SalutationRoute route = SalutationRoute::Neutral;
    std::string phase;       // decision label persisted with the style profile
    std::string bank_phase;  // greeting-bank key; empty means a literal clause
};

// Everything the policy reads: pure counters plus the signals the caller
// already derived from the prompt.
struct SalutationInputs {
    bool zh = false;
    uint32_t form_hits = 0;
    bool wave_now = false;
    uint32_t slang_wave_hits = 0;
    bool big_dog_now = false;
    uint32_t slang_big_dog_hits = 0;
    bool random_chat = false;
    uint32_t salutation_turns = 0;
    uint32_t associative_variant_hits = 0;
};

// The C6 phase cycle and the decision labels persisted per slot.
extern const char* const kPhaseCycle[6];
extern const char* const kPhaseCycleLabel[6];
constexpr std::size_t kSalutationWarmupTurns = 3;  // turns before cycling starts

// Ordered candidate routes. The caller walks the list and takes the first route
// that yields a clause; the list always ends with Neutral, so a greeting is
// never empty. The order *is* the learning policy: reactionary mirror before
// synthesis, synthesis before generic associative variants, phase cycling last.
std::vector<SalutationCandidate> salutation_candidates(const SalutationInputs& in);

}  // namespace khanary::micronaut::greeting
