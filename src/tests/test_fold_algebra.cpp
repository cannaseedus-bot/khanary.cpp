//
// KHANARY_FOLD_ALGEBRA_TEST — K'UHUL Fold Algebra conformance suite.
//
// Asserts that a fold is a typed state transition, that composition is a monoid
// with FOLD_IDENTITY as its unit, that the six-fold cycle composes, and that the
// composition law rejects exactly the plans the registry says it must:
//
//   * TRAIN fed straight from SCORE — types chain, only the verification rule
//     saves it (REQUIRES_VERIFICATION);
//   * Xul before Sek — the cycle only composes forwards (PHASE_OUT_OF_ORDER);
//   * a typed identity carrying the wrong carrier (TYPE_MISMATCH).
//
// It also asserts the negative claims that keep KFA honest: the composition
// monoid is NOT commutative and is NOT an algebra over a field, and a plan's
// rewrite licence is never wider than its weakest stage.
//
// Usage: KHANARY_FOLD_ALGEBRA_TEST <data-root>   (default: "data")
//
#include "conformance.h"
#include "../kuhul/algebra_contract.h"
#include "../kuhul/fold_algebra.h"

#include <fstream>
#include <sstream>

namespace {

using khanary::util::JsonPtr;
using khanary::util::JsonValue;
namespace kfa = khanary::kuhul::foldalgebra;
namespace aoi = khanary::kuhul::algebra;

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

bool contains(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

std::vector<std::string> codes_of(const std::vector<kfa::Violation>& violations) {
    std::vector<std::string> codes;
    codes.reserve(violations.size());
    for (const kfa::Violation& violation : violations) codes.push_back(violation.code);
    return codes;
}

struct Fixture {
    kfa::FoldRegistry registry;
    std::vector<kfa::FoldOp> identity_ops;
};

kfa::FoldOp require_fold(const kfa::FoldRegistry& registry, const std::string& id) {
    const kfa::FoldOp* fold = registry.find(id);
    if (fold == nullptr) throw std::runtime_error("missing fold " + id);
    return *fold;
}

kfa::ComposedPlan plan_of(const kfa::FoldRegistry& registry, const std::vector<std::string>& ids) {
    std::vector<kfa::FoldOp> ops;
    ops.reserve(ids.size());
    for (const std::string& id : ids) ops.push_back(require_fold(registry, id));
    return kfa::plan_sequence(ops);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_root = argc > 1 ? argv[1] : "data";

    return khanary::test::run("fold_algebra", [&](khanary::test::Suite& suite) {
        // ── 1. The registry parses and is internally consistent ────────────
        JsonPtr document = khanary::util::parse(
            read_file(data_root + "/grammar/kuhul-fold-algebra.contracts.json"));
        std::vector<kfa::Violation> errors;
        auto parsed = kfa::parse_fold_registry(*document, errors);
        if (!parsed.has_value()) throw std::runtime_error("fold registry did not parse");

        std::string error_text;
        for (const kfa::Violation& violation : errors) {
            error_text += violation.code + ":" + violation.detail + ";";
        }
        suite.check("registry/parses-cleanly", errors.empty());
        if (!errors.empty()) std::cerr << "registry errors: " << error_text << '\n';

        const kfa::FoldRegistry& registry = *parsed;
        suite.check("registry/phase-order", registry.phase_order.size() == std::size_t{6});
        suite.check("registry/phase-order-starts-at-pop",
                    !registry.phase_order.empty() && registry.phase_order.front() == "Pop");
        suite.check("registry/phase-order-ends-at-xul",
                    !registry.phase_order.empty() && registry.phase_order.back() == "Xul");

        for (const char* id : {"F_Pop", "F_Wo", "F_Yax", "F_Sek", "F_Ch'en", "F_Xul",
                               "MAP", "FILTER", "REDUCE", "FOLD_IDENTITY"}) {
            suite.check(std::string("registry/declares/") + id, registry.find(id) != nullptr);
        }
        for (std::size_t i = 0; i < 6; ++i) {
            const auto phase = static_cast<khanary::kuhul::Phase>(i);
            const std::string id = "F_" + std::string(khanary::kuhul::phase_name(phase));
            suite.check("registry/declares/" + id, registry.find(id) != nullptr);
        }

        // ── 2. The cycle composes ───────────────────────────────────────────
        const std::vector<std::string> cycle = {"F_Pop", "F_Wo", "F_Yax", "F_Sek", "F_Ch'en", "F_Xul"};
        const kfa::ComposedPlan cycle_plan = plan_of(registry, cycle);
        suite.check("cycle/is-legal", cycle_plan.legal);
        suite.check("cycle/crosses-five-boundaries", cycle_plan.phase_crossings == std::size_t{5});
        suite.check("cycle/requires-admission", cycle_plan.requires_admission);
        suite.check("cycle/is-not-pure", !cycle_plan.pure);
        suite.check("cycle/starts-at-pop", cycle_plan.first_phase == khanary::kuhul::Phase::Pop);
        suite.check("cycle/ends-at-xul", cycle_plan.last_phase == khanary::kuhul::Phase::Xul);
        suite.check("cycle/execution-order-preserved", cycle_plan.execution_order == cycle);

        // ── 3. The unit law ─────────────────────────────────────────────────
        const kfa::FoldOp identity = require_fold(registry, "FOLD_IDENTITY");
        suite.check("unit/is-a-wildcard", identity.is_identity && identity.output_type == "*");

        for (const std::string& id : {"F_Sek", "SCORE", "TRAIN", "MAP"}) {
            const kfa::FoldOp fold = require_fold(registry, id);
            const kfa::CompositionDecision left = kfa::compose(fold, identity);
            const kfa::CompositionDecision right = kfa::compose(identity, fold);
            suite.check("unit/right-elides/" + id, left.legal && left.violations.empty());
            suite.check("unit/left-elides/" + id, right.legal && right.violations.empty());
        }

        const kfa::FoldOp state_identity = require_fold(registry, "IDENTITY_STATE");
        suite.check("unit/typed-identity-does-not-carry-candidates",
                    !kfa::compose(require_fold(registry, "SCORE"), state_identity).legal);

        // ── 4. Composition is not commutative ──────────────────────────────
        const kfa::FoldOp score = require_fold(registry, "SCORE");
        const kfa::FoldOp rank = require_fold(registry, "RANK");
        const kfa::CompositionDecision score_then_rank = kfa::compose(rank, score);
        const kfa::CompositionDecision rank_then_score = kfa::compose(score, rank);
        suite.check("monoid/score-then-rank-legal", score_then_rank.legal);
        suite.check("monoid/rank-then-score-illegal", !rank_then_score.legal);
        suite.check("monoid/not-commutative",
                    score_then_rank.legal != rank_then_score.legal);
        suite.check("monoid/same-phase-needs-no-admission",
                    !score_then_rank.requires_admission);

        // ── 5. A monoid of transitions is not an algebra over a field ──────
        // Nothing in the composition block claims addition, scalar multiplication
        // or a bilinear product, and the plan's rewrite licence is only ever
        // narrower than its members'.
        suite.check("algebra/is-not-over-field",
                    document->get("composition")->get_bool("is_algebra_over_field") ==
                        std::optional<bool>(false));
        suite.check("algebra/composition-is-associative",
                    document->get("composition")->get_bool("associative") == std::optional<bool>(true));
        suite.check("algebra/composition-is-not-commutative",
                    document->get("composition")->get_bool("commutative") == std::optional<bool>(false));

        // ── 6. The verification rule is structural ──────────────────────────
        const kfa::ComposedPlan shortcut = plan_of(registry, {"SCORE", "TRAIN"});
        suite.check("verification/shortcut-is-illegal", !shortcut.legal);
        suite.check("verification/shortcut-code",
                    contains(codes_of(shortcut.violations), "REQUIRES_VERIFICATION"));

        const kfa::ComposedPlan safe = plan_of(registry, {"VERIFY", "TRAIN"});
        suite.check("verification/through-verify-is-legal", safe.legal);
        suite.check("verification/through-verify-crosses-one", safe.phase_crossings == std::size_t{1});

        const kfa::ComposedPlan full_pipeline =
            plan_of(registry, {"SCORE", "RANK", "COLLAPSE", "VERIFY", "TRAIN"});
        suite.check("verification/srai-pipeline-is-legal", full_pipeline.legal);
        suite.check("verification/srai-pipeline-crossings",
                    full_pipeline.phase_crossings == std::size_t{2});

        // ── 7. The cycle only composes forwards ─────────────────────────────
        const kfa::ComposedPlan backwards = plan_of(registry, {"TRAIN", "SCORE"});
        suite.check("order/backwards-is-illegal", !backwards.legal);
        suite.check("order/backwards-code",
                    contains(codes_of(backwards.violations), "PHASE_OUT_OF_ORDER"));

        const kfa::ComposedPlan wrapped = plan_of(registry, {"F_Xul", "F_Pop"});
        suite.check("order/wrap-around-is-a-new-plan", !wrapped.legal);
        suite.check("order/wrap-around-code",
                    contains(codes_of(wrapped.violations), "PHASE_OUT_OF_ORDER"));

        // ── 8. Pairwise locality is what makes the decision associative ─────
        const std::vector<std::string> sample = {"F_Pop", "F_Wo", "F_Yax", "F_Sek"};
        const kfa::ComposedPlan whole = plan_of(registry, sample);
        std::vector<std::string> pairwise;
        for (std::size_t i = 1; i < sample.size(); ++i) {
            const kfa::CompositionDecision decision =
                kfa::compose(require_fold(registry, sample[i]), require_fold(registry, sample[i - 1]));
            for (const kfa::Violation& violation : decision.violations) {
                pairwise.push_back(violation.code);
            }
        }
        suite.check("locality/plan-equals-pairwise", codes_of(whole.violations) == pairwise);
        suite.check("locality/regrouping-cannot-change-the-answer", whole.legal && pairwise.empty());

        // ── 9. A plan is never more algebraic than its weakest stage ────────
        JsonPtr aoi_document = khanary::util::parse(
            read_file(data_root + "/grammar/algebra-isa.contracts.json"));
        std::vector<aoi::Violation> aoi_errors;
        auto aoi_table = aoi::parse_structure_table(*aoi_document, aoi_errors);
        if (!aoi_table.has_value() || !aoi_errors.empty()) {
            throw std::runtime_error("algebra ISA table did not parse");
        }

        std::vector<aoi::AlgebraContract> contracts;
        for (const auto& entry : aoi_document->get("operators")->as_array()) {
            if (entry == nullptr) continue;
            aoi::ParsedContract parsed_contract = aoi::parse_contract(*entry);
            contracts.push_back(std::move(parsed_contract.contract));
        }
        const auto lookup = [contracts](const std::string& ref)
            -> const aoi::AlgebraContract* {
            const std::size_t hash = ref.find('#');
            const std::string id = hash == std::string::npos ? ref : ref.substr(hash + 1);
            for (const aoi::AlgebraContract& contract : contracts) {
                if (contract.id == id) return &contract;
            }
            return nullptr;
        };

        const std::vector<kfa::FoldOp> score_only = {require_fold(registry, "SCORE")};
        const aoi::RewriteSet score_licence = kfa::plan_rewrites(score_only, lookup);
        suite.check("licence/score-does-not-regroup", !score_licence.regroup);

        const std::vector<kfa::FoldOp> mixed = {require_fold(registry, "MAP"),
                                                require_fold(registry, "SCORE")};
        const aoi::RewriteSet mixed_licence = kfa::plan_rewrites(mixed, lookup);
        suite.check("licence/unresolved-stage-withholds-everything",
                    !mixed_licence.regroup && !mixed_licence.distribute);

        const std::vector<kfa::FoldOp> no_lookup = {require_fold(registry, "SCORE")};
        const aoi::RewriteSet bare = kfa::plan_rewrites(no_lookup, nullptr);
        suite.check("licence/no-lookup-withholds-everything", !bare.regroup);

        // ── 10. Fixtures: the same file the Python checker reads ────────────
        JsonPtr fixtures = khanary::util::parse(read_file(data_root + "/conformance/fold_algebra.json"));

        for (const auto& case_entry : fixtures->get("legal")->as_array()) {
            if (case_entry == nullptr) continue;
            const std::string id = case_entry->get_string("id").value_or("?");
            std::vector<std::string> sequence;
            for (const auto& name : case_entry->get("sequence")->as_array()) {
                if (name != nullptr && name->is_string()) sequence.push_back(name->as_string());
            }
            const kfa::ComposedPlan plan = plan_of(registry, sequence);
            bool ok = plan.legal;
            if (case_entry->get_number("expect_phase_crossings")) {
                const double expected = *case_entry->get_number("expect_phase_crossings");
                ok = ok && plan.phase_crossings == static_cast<std::size_t>(expected);
            }
            if (const auto expected = case_entry->get_bool("expect_admission")) {
                ok = ok && plan.requires_admission == *expected;
            }
            if (const auto expected = case_entry->get_bool("expect_pure")) {
                ok = ok && plan.pure == *expected;
            }
            suite.check("fixtures/legal/" + id, ok);
        }

        for (const auto& case_entry : fixtures->get("illegal")->as_array()) {
            if (case_entry == nullptr) continue;
            const std::string id = case_entry->get_string("id").value_or("?");
            const std::string expected = case_entry->get_string("expect_code").value_or("");
            std::vector<std::string> sequence;
            for (const auto& name : case_entry->get("sequence")->as_array()) {
                if (name != nullptr && name->is_string()) sequence.push_back(name->as_string());
            }
            const kfa::ComposedPlan plan = plan_of(registry, sequence);
            suite.check("fixtures/illegal/" + id, contains(codes_of(plan.violations), expected));
        }

        // ── 11. Every SRAI operator is a KFA fold with matching facts ───────
        JsonPtr srai = khanary::util::parse(
            read_file(data_root + "/grammar/srai-operators.contracts.json"));
        std::size_t mapped = 0;
        for (const auto& frame : srai->get("operators")->as_array()) {
            if (frame == nullptr) continue;
            const std::string name = frame->get_string("operator").value_or("");
            const kfa::FoldOp* fold = registry.find(name);
            if (fold == nullptr) continue;
            ++mapped;
            const std::string phase = frame->get_string("phase").value_or("");
            const std::string normalised = (phase == "Ch_en" || phase == "Chen") ? "Ch'en" : phase;
            const bool phase_ok =
                fold->phase.has_value() &&
                khanary::kuhul::phase_name(*fold->phase) == normalised;
            const std::string algebra_ref = frame->get_string("algebra_ref").value_or("");
            suite.check("cross/phase/" + name, phase_ok);
            suite.check("cross/algebra-ref/" + name,
                        fold->algebra_ref.has_value() && *fold->algebra_ref == algebra_ref);
        }
        suite.check("cross/all-operators-mapped", mapped == std::size_t{12});
    });
}
