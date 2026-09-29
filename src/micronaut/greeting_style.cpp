#include "greeting_style.h"

#include "../util/json.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

namespace khanary::micronaut::greeting {

// ── Bank parsing ────────────────────────────────────────────────────────────

std::vector<BankEntry> parse_bank(const std::string& json_text) {
    std::vector<BankEntry> out;
    // util::parse throws on malformed JSON: a broken bank must degrade to the
    // built-in fallbacks rather than break the greeting path.
    util::JsonPtr parsed;
    try {
        parsed = util::parse(json_text);
    } catch (const std::exception&) {
        return out;
    }
    if (!parsed || !parsed->is_object()) return out;
    const auto greetings = parsed->get("greetings");
    if (!greetings || !greetings->is_array()) return out;
    for (const auto& item : greetings->as_array()) {
        if (!item || !item->is_object()) continue;
        BankEntry entry;
        entry.phase = item->get_string("phase").value_or("");
        entry.kind = item->get_string("kind").value_or("");
        if (const auto* templates = item->get("templates");
            templates && templates->is_array()) {
            for (const auto& tpl : templates->as_array()) {
                if (!tpl || !tpl->is_string()) continue;
                const auto& text = tpl->as_string();
                if (!text.empty()) entry.templates.push_back(text);
            }
        }
        if (!entry.phase.empty() && !entry.templates.empty())
            out.push_back(std::move(entry));
    }
    return out;
}

// ── Built-in fallbacks ──────────────────────────────────────────────────────
// Kept in step with data/grammar/greeting-bank.json: if the bank is missing
// (for example a bundle without data/grammar) every phase key still answers.

const std::unordered_map<std::string, std::vector<std::string>>& builtin_templates() {
    static const std::unordered_map<std::string, std::vector<std::string>> kBuiltins = {
        {"Pop", {"Hello. I'm online and ready to help."}},
        {"Wo", {"Hey {cue} — picked that up from you; good to be talking again."}},
        {"Yax", {"Hey {cue}, good to see you again.",
                 "Yo {cue}, ready when you are.",
                 "What's good {cue}? I'm online and locked in."}},
        {"Sek", {"Reporting in: {turns} greetings so far and I'm adapting to how you talk. Ready when you are."}},
        {"Ch'en", {"Style check: {lingo_count} lingo cues learned so far. {Greeting}!"}},
        {"Xul", {"Last time you greeted me with \"{last}\". Memory holds — hello again."}},
        {"Pop.reactive_mirror",
         {"{Greeting} right back at you — I remembered that one.",
          "Back at you: {greeting}. I adapt to your lingo."}},
        {"Yax.synthesized_greeting",
         {"{Blend} vibes today — I made that word up from yours.",
          "Calling it {blend} now — your words, my spin."}}
    };
    return kBuiltins;
}

// ── Lingo ranking ───────────────────────────────────────────────────────────

std::vector<std::string> rank_lingo_cues(const std::unordered_map<std::string, uint32_t>& hits) {
    std::vector<std::pair<std::string, uint32_t>> ranked(hits.begin(), hits.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    std::vector<std::string> out;
    out.reserve(ranked.size());
    for (const auto& [cue, _] : ranked) out.push_back(cue);
    return out;
}

// ── Placeholder resolution ──────────────────────────────────────────────────

std::string title_case(const std::string& text) {
    if (text.empty()) return text;
    std::string out = text;
    out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

void replace_placeholder(std::string& text, const std::string& key, const std::string& value) {
    const std::string token = "{" + key + "}";
    std::size_t pos = 0;
    while ((pos = text.find(token, pos)) != std::string::npos) {
        text.replace(pos, token.size(), value);
        pos += value.size();
    }
}

std::string resolve_placeholders(const std::string& tmpl,
                                 const std::string& cue_display,
                                 const std::string& user_greeting,
                                 const std::string& last_greeting,
                                 const std::string& synthesized_word,
                                 const PlaceholderContext& ctx) {
    const bool needs_cue = tmpl.find("{cue}") != std::string::npos ||
                           tmpl.find("{Cue}") != std::string::npos;
    const bool needs_greet = tmpl.find("{greeting}") != std::string::npos ||
                             tmpl.find("{Greeting}") != std::string::npos;
    const bool needs_last = tmpl.find("{last}") != std::string::npos;
    const bool needs_blend = tmpl.find("{blend}") != std::string::npos ||
                             tmpl.find("{Blend}") != std::string::npos;
    if (needs_cue && cue_display.empty()) return {};
    if (needs_greet && user_greeting.empty()) return {};
    if (needs_last && last_greeting.empty()) return {};
    if (needs_blend && synthesized_word.empty()) return {};

    std::string out = tmpl;
    replace_placeholder(out, "cue", cue_display);
    replace_placeholder(out, "Cue", title_case(cue_display));
    replace_placeholder(out, "greeting", user_greeting);
    replace_placeholder(out, "Greeting", title_case(user_greeting));
    replace_placeholder(out, "last", last_greeting);
    replace_placeholder(out, "turns", std::to_string(ctx.turns));
    replace_placeholder(out, "lingo_count", std::to_string(ctx.lingo_count));
    replace_placeholder(out, "blend", synthesized_word);
    replace_placeholder(out, "Blend", title_case(synthesized_word));
    return out;
}

std::string select_clause(const std::vector<BankEntry>& bank,
                          const std::string& phase_key,
                          const std::string& cue_display,
                          const std::string& user_greeting,
                          const std::string& last_greeting,
                          const std::string& synthesized_word,
                          uint32_t seed,
                          const PlaceholderContext& ctx) {
    auto try_templates = [&](const std::vector<std::string>& templates) -> std::string {
        if (templates.empty()) return {};
        for (std::size_t offset = 0; offset < templates.size(); ++offset) {
            const auto& tmpl = templates[(seed + offset) % templates.size()];
            auto filled = resolve_placeholders(
                tmpl, cue_display, user_greeting, last_greeting, synthesized_word, ctx);
            if (!filled.empty()) return filled;
        }
        return {};
    };

    for (const auto& entry : bank) {
        if (entry.phase != phase_key) continue;
        auto clause = try_templates(entry.templates);
        if (!clause.empty()) return clause;
    }
    const auto& builtins = builtin_templates();
    auto it = builtins.find(phase_key);
    if (it != builtins.end()) {
        auto clause = try_templates(it->second);
        if (!clause.empty()) return clause;
    }
    return {};
}

// ── Observed greeting form ──────────────────────────────────────────────────

std::string observed_form(const std::vector<std::string>& tokens) {
    std::string out;
    std::size_t used = 0;
    for (const auto& tok : tokens) {
        if (used >= 12 || out.size() >= 64) break;
        if (!out.empty()) out.push_back(' ');
        out += tok;
        ++used;
    }
    if (out.size() > 64) out.resize(64);
    return out;
}

// ── Adaptive salutation policy ──────────────────────────────────────────────

const char* const kPhaseCycle[6] = {"Pop", "Wo", "Yax", "Sek", "Ch'en", "Xul"};
const char* const kPhaseCycleLabel[6] = {
    "Pop.canonical_greeting",
    "Wo.learned_lingo",
    "Yax.associative_greeting",
    "Sek.salutation_status",
    "Ch'en.style_confidence",
    "Xul.greeting_recall"
};

std::vector<SalutationCandidate> salutation_candidates(const SalutationInputs& in) {
    std::vector<SalutationCandidate> out;

    // A locale greeting is terminal: it is not adaptively re-registered.
    if (in.zh) {
        out.push_back({SalutationRoute::LocaleChinese, "Sek.locale_language", ""});
        return out;
    }

    // Reactionary mirror: the user repeated a greeting form the runtime already
    // archived — give the same type of greeting back, in their lingo.
    if (in.form_hits >= 2)
        out.push_back({SalutationRoute::ReactiveMirror, "Pop.reactive_mirror",
                       "Pop.reactive_mirror"});

    if (in.wave_now || in.slang_wave_hits >= 2)
        out.push_back({SalutationRoute::LearnedWaveSlang, "Yax.learned_wave_slang", ""});

    if (in.big_dog_now || in.slang_big_dog_hits >= 2)
        out.push_back({SalutationRoute::LearnedSlang, "Yax.learned_slang", ""});

    if (in.random_chat)
        out.push_back({SalutationRoute::RandomChat, "Wo.random_chat_context", ""});

    if (in.salutation_turns >= kSalutationWarmupTurns) {
        // Once enough associative variants have been emitted, start creating the
        // runtime's own words by blending the user's top lingo cues.
        if (in.associative_variant_hits >= 3)
            out.push_back({SalutationRoute::SynthesizedGreeting, "Yax.synthesized_greeting",
                           "Yax.synthesized_greeting"});
        out.push_back({SalutationRoute::AssociativeGreeting, "Yax.associative_greeting", ""});
        const std::size_t slot = (in.salutation_turns - kSalutationWarmupTurns) % 6;
        // The Yax slot is served by the associative path above.
        if (slot != 2) {
            out.push_back({SalutationRoute::PhaseCycle, kPhaseCycleLabel[slot],
                           kPhaseCycle[slot]});
        }
    } else {
        out.push_back({SalutationRoute::PhaseCycle, kPhaseCycleLabel[0], kPhaseCycle[0]});
    }

    out.push_back({SalutationRoute::Neutral, "Pop.neutral", ""});
    return out;
}

}  // namespace khanary::micronaut::greeting
