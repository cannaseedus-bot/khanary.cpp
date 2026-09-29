#include "conformance.h"
#include "../micronaut/greeting_style.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace khanary::micronaut::greeting;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

PlaceholderContext full_context() { return PlaceholderContext{7, 4}; }

// Values that satisfy every placeholder the bank uses.
const char* kCue = "friend";
const char* kGreeting = "hi there";
const char* kLast = "hey";
const char* kBlend = "frere";

bool resolves(const std::vector<BankEntry>& bank, const std::string& phase) {
    return !select_clause(bank, phase, kCue, kGreeting, kLast, kBlend, 0, full_context())
                .empty();
}

bool has_route(const std::vector<SalutationCandidate>& candidates, SalutationRoute route) {
    for (const auto& c : candidates)
        if (c.route == route) return true;
    return false;
}

int index_of(const std::vector<SalutationCandidate>& candidates, SalutationRoute route) {
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (candidates[i].route == route) return static_cast<int>(i);
    return -1;
}

std::string cycle_bank_phase(uint32_t turns) {
    SalutationInputs in;
    in.salutation_turns = turns;
    for (const auto& c : salutation_candidates(in))
        if (c.route == SalutationRoute::PhaseCycle) return c.bank_phase;
    return "<none>";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string bank_path = argc > 1 ? argv[1] : "";
    return khanary::test::run("greeting-style", [&](khanary::test::Suite& suite) {
        // ── parse_bank ──────────────────────────────────────────────────────
        const std::string good = R"({"greetings":[
            {"phase":"Pop","kind":"canonical","templates":["A","B"]},
            {"phase":"Wo","kind":"learned","templates":["C"]}
        ]})";
        const auto parsed = parse_bank(good);
        suite.check("parse_bank_count", parsed.size() == 2);
        suite.check("parse_bank_phase", parsed.size() == 2 && parsed[0].phase == "Pop");
        suite.check("parse_bank_kind", parsed.size() == 2 && parsed[0].kind == "canonical");
        suite.check("parse_bank_templates", parsed.size() == 2 && parsed[0].templates.size() == 2);
        suite.check("parse_bank_second_entry", parsed.size() == 2 && parsed[1].phase == "Wo");

        suite.check("parse_bank_rejects_malformed", parse_bank("{not json").empty());
        suite.check("parse_bank_rejects_non_object", parse_bank("[1,2,3]").empty());
        suite.check("parse_bank_rejects_missing_array", parse_bank(R"({"other":1})").empty());

        // Entries missing a phase, missing templates, or offering only empty
        // templates are dropped — a malformed bank must never break a greeting.
        const auto partial = parse_bank(R"({"greetings":[
            {"kind":"canonical","templates":["A"]},
            {"phase":"Pop","templates":[]},
            {"phase":"Pop","kind":"canonical","templates":[""]},
            "string-entry"
        ]})");
        suite.check("parse_bank_drops_incomplete_entries", partial.empty());

        // ── built-in fallbacks ──────────────────────────────────────────────
        const auto& builtins = builtin_templates();
        for (const char* phase : kPhaseCycle)
            suite.check(std::string("builtin_covers_") + phase, builtins.count(phase) == 1);
        suite.check("builtin_covers_reactive_mirror", builtins.count("Pop.reactive_mirror") == 1);
        suite.check("builtin_covers_synthesized", builtins.count("Yax.synthesized_greeting") == 1);

        // ── select_clause: bank first, builtins second ──────────────────────
        const std::vector<BankEntry> bank_only = {BankEntry{"Pop", "canonical", {"BANK-POP"}}};
        suite.check("select_prefers_bank",
                    select_clause(bank_only, "Pop", "", "", "", "", 0, full_context()) ==
                        "BANK-POP");
        suite.check("select_falls_back_to_builtin",
                    select_clause(bank_only, "Wo", kCue, kGreeting, kLast, kBlend, 0,
                                  full_context())
                            .find("picked that up") != std::string::npos);
        suite.check("select_unknown_phase_empty",
                    select_clause(bank_only, "NoSuchPhase", kCue, kGreeting, kLast, kBlend, 0,
                                  full_context())
                        .empty());

        // A bank template needing lingo the runtime has not observed is skipped
        // rather than emitted half-substituted.
        const std::vector<BankEntry> greedy = {BankEntry{"Pop", "canonical", {"{cue} hello"}}};
        suite.check("select_skips_unfillable_bank_template",
                    select_clause(greedy, "Pop", "", "", "", "", 0, full_context()) ==
                        "Hello. I'm online and ready to help.");

        // ── rotation ────────────────────────────────────────────────────────
        const std::vector<BankEntry> three = {BankEntry{"Pop", "canonical", {"T0", "T1", "T2"}}};
        suite.check("rotation_seed0", select_clause(three, "Pop", "", "", "", "", 0,
                                                    full_context()) == "T0");
        suite.check("rotation_seed1", select_clause(three, "Pop", "", "", "", "", 1,
                                                    full_context()) == "T1");
        suite.check("rotation_seed2", select_clause(three, "Pop", "", "", "", "", 2,
                                                    full_context()) == "T2");
        suite.check("rotation_wraps", select_clause(three, "Pop", "", "", "", "", 3,
                                                    full_context()) == "T0");

        // ── placeholders ────────────────────────────────────────────────────
        const PlaceholderContext ctx{5, 2};
        suite.check("placeholder_rejects_missing_cue",
                    resolve_placeholders("Hey {cue}", "", kGreeting, kLast, kBlend, ctx).empty());
        suite.check("placeholder_rejects_missing_greeting",
                    resolve_placeholders("{Greeting}!", kCue, "", kLast, kBlend, ctx).empty());
        suite.check("placeholder_rejects_missing_last",
                    resolve_placeholders("Last {last}", kCue, kGreeting, "", kBlend, ctx).empty());
        suite.check("placeholder_rejects_missing_blend",
                    resolve_placeholders("{blend}!", kCue, kGreeting, kLast, "", ctx).empty());
        suite.check("placeholder_title_case_cue",
                    resolve_placeholders("{Cue}!", "friend", kGreeting, kLast, kBlend, ctx) ==
                        "Friend!");
        suite.check("placeholder_title_case_greeting",
                    resolve_placeholders("{Greeting} back", kCue, "hi there", kLast, kBlend,
                                         ctx) == "Hi there back");
        suite.check("placeholder_counters",
                    resolve_placeholders("{turns}/{lingo_count}", kCue, kGreeting, kLast, kBlend,
                                         ctx) == "5/2");
        suite.check("placeholder_all_filled",
                    resolve_placeholders("{cue}|{greeting}|{last}|{blend}", kCue, kGreeting, kLast,
                                         kBlend, ctx) == "friend|hi there|hey|frere");

        // ── observed_form ───────────────────────────────────────────────────
        std::vector<std::string> many;
        for (int i = 0; i < 20; ++i) many.push_back("t" + std::to_string(i));
        const auto form = observed_form(many);
        suite.check("observed_form_caps_tokens",
                    std::count(form.begin(), form.end(), ' ') == 11);
        suite.check("observed_form_caps_chars",
                    observed_form({std::string(200, 'x')}).size() == 64);
        suite.check("observed_form_empty", observed_form({}).empty());

        // ── adaptive salutation policy ──────────────────────────────────────
        const SalutationInputs fresh;  // first salutation, nothing learned yet
        const auto c0 = salutation_candidates(fresh);
        suite.check("policy_always_ends_neutral",
                    !c0.empty() && c0.back().route == SalutationRoute::Neutral);
        suite.check("policy_fresh_starts_phase_cycle",
                    c0.front().route == SalutationRoute::PhaseCycle);
        suite.check("policy_fresh_uses_pop", c0.front().bank_phase == "Pop");
        suite.check("policy_fresh_label", c0.front().phase == "Pop.canonical_greeting");

        SalutationInputs zh;
        zh.zh = true;
        const auto cz = salutation_candidates(zh);
        suite.check("policy_zh_is_terminal",
                    cz.size() == 1 && cz[0].route == SalutationRoute::LocaleChinese);
        suite.check("policy_zh_phase", cz.size() == 1 && cz[0].phase == "Sek.locale_language");

        SalutationInputs mirror;
        mirror.form_hits = 2;
        const auto cm = salutation_candidates(mirror);
        suite.check("policy_mirror_present", has_route(cm, SalutationRoute::ReactiveMirror));
        suite.check("policy_mirror_first",
                    cm.front().route == SalutationRoute::ReactiveMirror);
        suite.check("policy_mirror_bank_key",
                    cm.front().bank_phase == "Pop.reactive_mirror");
        suite.check("policy_mirror_before_phase_cycle",
                    index_of(cm, SalutationRoute::ReactiveMirror) <
                        index_of(cm, SalutationRoute::PhaseCycle));
        SalutationInputs single;
        single.form_hits = 1;
        suite.check("policy_mirror_needs_repeat",
                    !has_route(salutation_candidates(single), SalutationRoute::ReactiveMirror));

        SalutationInputs syn;
        syn.salutation_turns = 3;
        syn.associative_variant_hits = 3;
        const auto cs = salutation_candidates(syn);
        suite.check("policy_synthesis_after_warmup",
                    has_route(cs, SalutationRoute::SynthesizedGreeting));
        suite.check("policy_synthesis_before_associative",
                    index_of(cs, SalutationRoute::SynthesizedGreeting) <
                        index_of(cs, SalutationRoute::AssociativeGreeting));
        suite.check("policy_synthesis_bank_key",
                    cs[index_of(cs, SalutationRoute::SynthesizedGreeting)].bank_phase ==
                        "Yax.synthesized_greeting");
        SalutationInputs early = syn;
        early.salutation_turns = 2;
        suite.check("policy_synthesis_requires_warmup",
                    !has_route(salutation_candidates(early),
                               SalutationRoute::SynthesizedGreeting));
        SalutationInputs no_variants = syn;
        no_variants.associative_variant_hits = 2;
        suite.check("policy_synthesis_requires_variants",
                    !has_route(salutation_candidates(no_variants),
                               SalutationRoute::SynthesizedGreeting));

        // Phase cycling; the Yax slot is served by the associative path.
        suite.check("policy_cycle_turn3_pop", cycle_bank_phase(3) == "Pop");
        suite.check("policy_cycle_turn4_wo", cycle_bank_phase(4) == "Wo");
        suite.check("policy_cycle_turn5_yax_skipped", cycle_bank_phase(5) == "<none>");
        suite.check("policy_cycle_turn6_sek", cycle_bank_phase(6) == "Sek");
        suite.check("policy_cycle_turn7_chen", cycle_bank_phase(7) == "Ch'en");
        suite.check("policy_cycle_turn8_xul", cycle_bank_phase(8) == "Xul");
        suite.check("policy_cycle_turn9_pop", cycle_bank_phase(9) == "Pop");

        SalutationInputs slang;
        slang.slang_wave_hits = 2;
        slang.slang_big_dog_hits = 2;
        slang.random_chat = true;
        const auto csl = salutation_candidates(slang);
        suite.check("policy_slang_order",
                    index_of(csl, SalutationRoute::LearnedWaveSlang) <
                            index_of(csl, SalutationRoute::LearnedSlang) &&
                        index_of(csl, SalutationRoute::LearnedSlang) <
                            index_of(csl, SalutationRoute::RandomChat));
        SalutationInputs no_slang;
        suite.check("policy_slang_needs_signal",
                    !has_route(salutation_candidates(no_slang),
                               SalutationRoute::LearnedWaveSlang));

        // Every candidate carries a decision label; bank-driven routes carry a key.
        SalutationInputs busy;
        busy.form_hits = 2;
        busy.wave_now = true;
        busy.big_dog_now = true;
        busy.random_chat = true;
        busy.salutation_turns = 4;
        busy.associative_variant_hits = 3;
        bool labels_present = true;
        bool keys_match_routes = true;
        for (const auto& c : salutation_candidates(busy)) {
            if (c.phase.empty()) labels_present = false;
            const bool needs_key = c.route == SalutationRoute::ReactiveMirror ||
                                   c.route == SalutationRoute::SynthesizedGreeting ||
                                   c.route == SalutationRoute::PhaseCycle;
            if (needs_key != !c.bank_phase.empty()) keys_match_routes = false;
        }
        suite.check("policy_labels_present", labels_present);
        suite.check("policy_bank_keys_match_routes", keys_match_routes);

        // ── builtins answer every phase with no bank at all ─────────────────
        const std::vector<BankEntry> no_bank;
        for (const char* phase : kPhaseCycle)
            suite.check(std::string("no_bank_answers_") + phase, resolves(no_bank, phase));
        suite.check("no_bank_answers_reactive_mirror",
                    resolves(no_bank, "Pop.reactive_mirror"));
        suite.check("no_bank_answers_synthesized",
                    resolves(no_bank, "Yax.synthesized_greeting"));

        // ── shipped bank (path supplied by CMake) ───────────────────────────
        if (!bank_path.empty()) {
            const auto text = read_file(bank_path);
            suite.check("shipped_bank_readable", !text.empty());
            const auto bank = parse_bank(text);
            suite.check("shipped_bank_entry_count", bank.size() >= 8);
            const char* expected[] = {"Pop",           "Wo",
                                      "Yax",           "Sek",
                                      "Ch'en",         "Xul",
                                      "Pop.reactive_mirror", "Yax.synthesized_greeting"};
            for (const char* phase : expected) {
                bool found = false;
                for (const auto& e : bank)
                    if (e.phase == phase) found = true;
                suite.check(std::string("shipped_bank_has_") + phase, found);
            }
            bool kinds_present = true;
            bool templates_present = true;
            bool every_phase_resolves = true;
            for (const auto& e : bank) {
                if (e.kind.empty()) kinds_present = false;
                if (e.templates.empty()) templates_present = false;
                if (select_clause(bank, e.phase, kCue, kGreeting, kLast, kBlend, 0,
                                  full_context())
                        .empty())
                    every_phase_resolves = false;
            }
            suite.check("shipped_bank_kinds_present", kinds_present);
            suite.check("shipped_bank_templates_present", templates_present);
            suite.check("shipped_bank_every_phase_resolves", every_phase_resolves);

            const auto pop_clause =
                select_clause(bank, "Pop", kCue, kGreeting, kLast, kBlend, 0, full_context());
            bool phases_distinct = true;
            for (const auto& e : bank) {
                if (e.phase == "Pop") continue;
                if (select_clause(bank, e.phase, kCue, kGreeting, kLast, kBlend, 0,
                                  full_context()) == pop_clause)
                    phases_distinct = false;
            }
            suite.check("shipped_bank_phases_distinct", phases_distinct);
        }
    });
}
