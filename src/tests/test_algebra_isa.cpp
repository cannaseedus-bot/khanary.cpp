//
// KHANARY_ALGEBRA_ISA_TEST — the Algebraic Operator ISA conformance suite.
//
// Asserts four separate things:
//
//   1. the axiom table compiled into src/kuhul/algebra_contract.cpp is byte-
//      identical to data/grammar/algebra-isa.contracts.json (no drift between
//      authority and implementation),
//   2. every registry contract validates against its own binding law table,
//   3. the derived permission vector matches the one the Python checker
//      derives for the same fixtures (no drift between the two checkers), and
//   4. regrouping is only ever performed when the contract licenses it — with
//      a numerical demonstration of what happens when it is not.
//
// Usage: KHANARY_ALGEBRA_ISA_TEST [data-root]   (default: "data")
//
#include "conformance.h"
#include "../graphscript/graph_algebra.h"
#include "../kuhul/algebra_contract.h"
#include "../kuhul/state_algebra.h"

#include <cmath>
#include <map>
#include <numbers>
#include <fstream>
#include <sstream>

namespace {

using khanary::util::JsonPtr;
using khanary::util::JsonValue;
namespace aoi = khanary::kuhul::algebra;
namespace szero = khanary::kuhul::statealgebra;
namespace graphscript = khanary::graphscript;

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

struct Registry {
    JsonPtr document;
    aoi::StructureTable table;
    std::vector<aoi::AlgebraContract> contracts;
    std::vector<std::string> operator_ids;
    std::vector<std::string> reducer_ids;

    const aoi::AlgebraContract* find(const std::string& id) const {
        for (const aoi::AlgebraContract& contract : contracts) {
            if (contract.id == id) return &contract;
        }
        return nullptr;
    }
};

Registry load_registry(const std::string& data_root) {
    Registry registry;
    registry.document = khanary::util::parse(
        read_file(data_root + "/grammar/algebra-isa.contracts.json"));
    std::vector<aoi::Violation> errors;
    auto table = aoi::parse_structure_table(*registry.document, errors);
    if (!table.has_value() || !errors.empty()) {
        std::string detail;
        for (const aoi::Violation& violation : errors) {
            detail += violation.code + ": " + violation.detail + "; ";
        }
        throw std::runtime_error("structure table failed to parse: " + detail);
    }
    registry.table = std::move(*table);
    for (const char* group : {"operators", "reducers"}) {
        const JsonValue* list = registry.document->get(group);
        if (list == nullptr || !list->is_array()) continue;
        for (const auto& entry : list->as_array()) {
            if (entry == nullptr) continue;
            aoi::ParsedContract parsed = aoi::parse_contract(*entry);
            if (!parsed.errors.empty()) {
                throw std::runtime_error("registry entry failed to parse: " +
                                         parsed.errors.front().code + " " +
                                         parsed.errors.front().detail);
            }
            if (std::string(group) == "operators") {
                registry.operator_ids.push_back(parsed.contract.id);
            } else {
                registry.reducer_ids.push_back(parsed.contract.id);
            }
            registry.contracts.push_back(std::move(parsed.contract));
        }
    }
    return registry;
}

// left fold seeded with `items.front()` — the regrouping baseline
template <class Op>
double left_fold(const std::vector<double>& items, Op op) {
    double accumulator = items.front();
    for (std::size_t i = 1; i < items.size(); ++i) accumulator = op(accumulator, items[i]);
    return accumulator;
}

// pairwise tree over the same sequence — the regrouped baseline
template <class Op>
double tree_fold(const std::vector<double>& items, std::size_t begin, std::size_t end, Op op) {
    if (end - begin == 1) return items[begin];
    const std::size_t middle = begin + (end - begin) / 2;
    return op(tree_fold(items, begin, middle, op), tree_fold(items, middle, end, op));
}

bool contains(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

// ── Small exact carriers ─────────────────────────────────────────────────────
// Small integers so every equality below is exact in binary floating point: the
// laws are being tested, not the summation order.

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vec3 add(const Vec3& a, const Vec3& b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 negate(const Vec3& a) { return Vec3{-a.x, -a.y, -a.z}; }
bool same(const Vec3& a, const Vec3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

struct Mat2 {
    double a = 0.0;  // [a b]
    double b = 0.0;  // [c d]
    double c = 0.0;
    double d = 0.0;
};

Mat2 matmul(const Mat2& x, const Mat2& y) {
    return Mat2{x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d,
                x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d};
}
Mat2 matadd(const Mat2& x, const Mat2& y) {
    return Mat2{x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d};
}
bool same(const Mat2& x, const Mat2& y) {
    return x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d;
}

const Mat2 kIdentity{1.0, 0.0, 0.0, 1.0};

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double total = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) total += a[i] * b[i];
    return total;
}

double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    const double denominator = norm(a) * norm(b);
    return denominator == 0.0 ? 0.0 : dot(a, b) / denominator;
}

std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out = a;
    for (std::size_t i = 0; i < b.size() && i < out.size(); ++i) out[i] += b[i];
    return out;
}

bool has_code(const std::vector<aoi::Violation>& violations, const std::string& code) {
    for (const aoi::Violation& violation : violations) {
        if (violation.code == code) return true;
    }
    return false;
}

bool matches_expected_policy(const aoi::ReductionPolicy& policy, const aoi::RewriteSet& rewrites,                             const aoi::AlgebraContract& contract, const JsonValue& expected) {
    if (!expected.is_object()) return false;
    for (const auto& [key, value] : expected.as_object()) {
        if (value == nullptr) return false;

        if (key == "grouping") {
            if (!value->is_string() || value->as_string() != aoi::grouping_name(policy.grouping)) {
                return false;
            }
            continue;
        }
        if (key == "rewrites") {
            if (!value->is_object()) return false;
            for (const auto& [rewrite, wanted] : value->as_object()) {
                if (wanted == nullptr || !wanted->is_bool()) return false;
                bool actual = false;
                if (rewrite == "regroup") actual = rewrites.regroup;
                else if (rewrite == "reorder") actual = rewrites.reorder;
                else if (rewrite == "swap_with_sign") actual = rewrites.swap_with_sign;
                else if (rewrite == "distribute") actual = rewrites.distribute;
                else if (rewrite == "elide_unit") actual = rewrites.elide_unit;
                else if (rewrite == "repeat") actual = rewrites.repeat;
                else if (rewrite == "invert") actual = rewrites.invert;
                else return false;
                if (actual != wanted->as_bool()) return false;
            }
            continue;
        }
        if (key == "phase_cross" || key == "phase_same") {
            if (!value->is_object()) return false;
            const aoi::PhaseDecision decision =
                aoi::check_phase_transition(contract, key == "phase_same");
            if (const JsonValue* allowed = value->get("allowed")) {
                if (!allowed->is_bool() || allowed->as_bool() != decision.allowed) return false;
            }
            if (const JsonValue* code = value->get("code")) {
                if (!code->is_string() || code->as_string() != decision.code) return false;
            }
            continue;
        }

        if (!value->is_bool()) return false;
        bool actual = false;
        if (key == "mathematically_associative") actual = policy.mathematically_associative;
        else if (key == "executable_associative") actual = policy.executable_associative;
        else if (key == "commutative") actual = policy.commutative;
        else if (key == "unital") actual = policy.unital;
        else if (key == "regroup") actual = policy.regroup;
        else if (key == "reorder") actual = policy.reorder;
        else if (key == "parallel") actual = policy.parallel;
        else if (key == "repeat") actual = policy.repeat;
        else if (key == "invert") actual = policy.invert;
        else if (key == "empty_fold") actual = policy.empty_fold;
        else if (key == "carrier_exact") actual = policy.carrier_exact;
        else return false;
        if (actual != value->as_bool()) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_root = argc > 1 ? argv[1] : "data";

    return khanary::test::run("KHANARY_ALGEBRA_ISA_TEST", [&](khanary::test::Suite& suite) {
        Registry registry = load_registry(data_root);

        // ── 1. Three-valued laws ────────────────────────────────────────────
        suite.check("laws/null-is-not-applicable",
                    aoi::law_state_from_token("null") == aoi::LawState::NotApplicable);
        suite.check("laws/true-is-yes", aoi::law_state_from_token("true") == aoi::LawState::Yes);
        suite.check("laws/false-is-no", aoi::law_state_from_token("false") == aoi::LawState::No);
        suite.check("laws/unknown-token-rejected",
                    !aoi::law_state_from_token("probably").has_value());
        suite.check("laws/token-roundtrip",
                    std::string(aoi::law_state_token(aoi::LawState::NotApplicable)) == "null");
        suite.check("laws/not-applicable-is-not-yes",
                    aoi::LawState::NotApplicable != aoi::LawState::Yes);

        // ── 2. Axiom table parity: compiled-in == registry ──────────────────
        std::string why;
        const bool parity =
            aoi::same_structure_table(aoi::builtin_structure_table(), registry.table, &why);
        suite.check("table/parity-with-registry", parity);
        if (!parity) std::cerr << "structure table drift: " << why << '\n';

        for (std::size_t i = 0; i < aoi::kStructureCount; ++i) {
            const auto structure = static_cast<aoi::Structure>(i);
            suite.check(std::string("table/declares/") + aoi::structure_name(structure),
                        registry.table.find(structure) != nullptr);
        }

        // ── 3. Registry contracts validate ─────────────────────────────────
        suite.check("registry/operator-count", registry.operator_ids.size() == std::size_t{12});
        suite.check("registry/reducer-count", registry.reducer_ids.size() == std::size_t{14});

        for (const aoi::AlgebraContract& contract : registry.contracts) {
            const std::vector<aoi::Violation> violations = aoi::validate(contract, registry.table);
            suite.check("registry/valid/" + contract.id, violations.empty());
        }

        // ── 4. Derived policy per contract ─────────────────────────────────
        const auto require = [&](const std::string& id) -> const aoi::AlgebraContract& {
            const aoi::AlgebraContract* contract = registry.find(id);
            if (contract == nullptr) throw std::runtime_error("missing contract " + id);
            return *contract;
        };
        const auto policy_of = [&](const std::string& id) {
            return aoi::derive_policy(require(id), registry.table);
        };

        const aoi::ReductionPolicy sum = policy_of("SUM_SCORES");
        suite.check("policy/SUM_SCORES/mathematically-associative", sum.mathematically_associative);
        suite.check("policy/SUM_SCORES/regroup", sum.regroup);
        suite.check("policy/SUM_SCORES/not-reorderable", !sum.reorder);
        suite.check("policy/SUM_SCORES/grouping-balanced", sum.grouping == aoi::Grouping::Balanced);
        suite.check("policy/SUM_SCORES/carrier-inexact", !sum.carrier_exact);

        const aoi::ReductionPolicy max_scores = policy_of("MAX_SCORES");
        suite.check("policy/MAX_SCORES/reorderable", max_scores.reorder);
        suite.check("policy/MAX_SCORES/repeatable", max_scores.repeat);
        suite.check("policy/MAX_SCORES/grouping-free", max_scores.grouping == aoi::Grouping::Free);

        const aoi::ReductionPolicy gram = policy_of("GRAM_SEQUENCE");
        suite.check("policy/GRAM_SEQUENCE/regroup", gram.regroup);
        suite.check("policy/GRAM_SEQUENCE/not-reorderable", !gram.reorder);
        suite.check("policy/GRAM_SEQUENCE/grouping-balanced", gram.grouping == aoi::Grouping::Balanced);

        const aoi::ReductionPolicy cross_policy = policy_of("CROSS_PRODUCT");
        suite.check("policy/CROSS_PRODUCT/not-associative", !cross_policy.mathematically_associative);
        suite.check("policy/CROSS_PRODUCT/not-regroupable", !cross_policy.regroup);

        const aoi::ReductionPolicy cosine_policy = policy_of("COSINE_SIMILARITY");
        suite.check("policy/COSINE/not-regroupable", !cosine_policy.regroup);
        suite.check("policy/COSINE/product-leaves-carrier",
                    require("COSINE_SIMILARITY").product_image == aoi::ProductImage::Scalar);

        const aoi::ReductionPolicy update = policy_of("LEARNING_STATE_UPDATE");
        suite.check("policy/LEARNING_STATE_UPDATE/associative-mathematically",
                    update.mathematically_associative);
        suite.check("policy/LEARNING_STATE_UPDATE/regroup-withheld", !update.regroup);
        suite.check("policy/LEARNING_STATE_UPDATE/impurity-is-the-reason",
                    contains(update.reasons, "impure"));

        const aoi::ReductionPolicy score = policy_of("SCORE");
        suite.check("policy/SCORE/no-binary-law", !score.mathematically_associative);
        suite.check("policy/SCORE/not-regroupable", !score.regroup);
        suite.check("policy/SCORE/laws-not-applicable",
                    require("SCORE").laws.is(aoi::Law::Associative,
                                             aoi::LawState::NotApplicable));

        suite.check("policy/COLLAPSE/not-regroupable", !policy_of("COLLAPSE").regroup);
        suite.check("policy/TRAIN/not-regroupable", !policy_of("TRAIN").regroup);
        suite.check("policy/PROJECT/repeatable", policy_of("PROJECT").repeat);
        suite.check("policy/EVIDENCE_MEET/repeatable", policy_of("EVIDENCE_MEET").repeat);

        // ── 5. Fold planning follows the contract, never the call site ─────
        const aoi::AlgebraContract& sum_contract = require("SUM_SCORES");
        const aoi::AlgebraContract& score_contract = require("SCORE");
        const aoi::AlgebraContract& max_contract = require("MAX_SCORES");

        const aoi::FoldPlan small = aoi::plan_fold(sum_contract, 1, registry.table);
        suite.check("plan/single-element-is-sequential",
                    small.grouping == aoi::Grouping::Sequential);

        const aoi::FoldPlan ungated = aoi::plan_fold(score_contract, 512, registry.table);
        suite.check("plan/ungated-is-sequential", ungated.grouping == aoi::Grouping::Sequential);
        suite.check("plan/ungated-chunk-one", ungated.chunk == 1);

        const aoi::FoldPlan canonical = aoi::plan_fold(sum_contract, 512, registry.table);
        suite.check("plan/canonical-balanced", canonical.grouping == aoi::Grouping::Balanced);
        suite.check("plan/canonical-chunk", canonical.chunk == 64);

        const aoi::FoldPlan free_plan = aoi::plan_fold(max_contract, 512, registry.table);
        suite.check("plan/commutative-exact-is-free", free_plan.grouping == aoi::Grouping::Free);

        // ── 6. Regrouping changes the answer when the law does not hold ────
        // Subtraction is not associative. This is the failure the contract
        // exists to prevent: the same inputs, two legal-looking tree shapes,
        // two different answers.
        const std::vector<double> numbers = {1.0, 2.0, 3.0, 4.0};
        const auto difference = [](double a, double b) { return a - b; };
        const auto addition = [](double a, double b) { return a + b; };
        suite.check("regroup/nonassociative-left-fold",
                    left_fold(numbers, difference) == -8.0);
        suite.check("regroup/nonassociative-tree-fold-differs",
                    tree_fold(numbers, 0, numbers.size(), difference) == 0.0);
        suite.check("regroup/nonassociative-differs",
                    left_fold(numbers, difference) != tree_fold(numbers, 0, numbers.size(), difference));
        suite.check("regroup/associative-is-equal",
                    left_fold(numbers, addition) == tree_fold(numbers, 0, numbers.size(), addition));

        // ── 7. Float accumulation: mathematically associative, not executable
        // 1e16 + 1 == 1e16 in float64, so the seed order decides the answer.
        const std::vector<double> absorbing = {1e16, 1.0, -1e16, 1.0};
        const double left = left_fold(absorbing, addition);
        const double tree = tree_fold(absorbing, 0, absorbing.size(), addition);
        suite.check("float/left-fold-drops-the-unit", left == 1.0);
        suite.check("float/tree-fold-keeps-the-unit", tree == 0.0);
        suite.check("float/grouping-is-not-free", left != tree);

        // The declared canonical tree is reproducible: chunking at the same
        // width always yields the same bytes.
        const aoi::FoldPlan canonical_plan{aoi::Grouping::Balanced, 64, {}};
        std::vector<double> wide(200, 0.1);
        const double a = aoi::fold_with_plan(wide, 0.0, addition, canonical_plan);
        const double b = aoi::fold_with_plan(wide, 0.0, addition, canonical_plan);
        suite.check("float/canonical-tree-is-reproducible", a == b);
        suite.check("float/empty-fold-returns-identity",
                    aoi::fold_with_plan(std::vector<double>{}, 7.0, addition, canonical_plan) == 7.0);

        // ── 8. The effect lattice cannot widen authority ───────────────────
        const aoi::EffectSet phase = {"read", "measure", "order"};
        const aoi::EffectSet track = {"read", "measure"};
        const aoi::EffectSet executor = {"read", "measure", "gate"};
        const std::vector<aoi::EffectSet> authorities = {phase, track, executor};
        const aoi::EffectSet effective = aoi::effective_permissions(authorities);
        suite.check("lattice/meet-is-intersection", effective == (aoi::EffectSet{"read", "measure"}));
        suite.check("lattice/meet-cannot-widen-phase", aoi::effect_meet(phase, track) == track);
        suite.check("lattice/no-effect-beyond-every-authority",
                    aoi::grants_only_attested(effective, authorities));
        suite.check("lattice/join-is-union",
                    aoi::joined_permissions(authorities) ==
                        (aoi::EffectSet{"read", "measure", "order", "gate"}));
        suite.check("lattice/meet-idempotent", aoi::effect_meet(phase, phase) == phase);
        suite.check("lattice/meet-associative",
                    aoi::effect_meet(aoi::effect_meet(phase, track), executor) ==
                        aoi::effect_meet(phase, aoi::effect_meet(track, executor)));
        suite.check("lattice/meet-commutative",
                    aoi::effect_meet(phase, track) == aoi::effect_meet(track, phase));
        suite.check("lattice/unattested-effect-is-rejected",
                    !aoi::grants_only_attested(aoi::EffectSet{"read", "collapse"}, authorities));

        // ── 9. Algebraic rewrites licensed by the contract ──────────────────
        const aoi::RewriteSet matmul_rewrites = aoi::rewrites_for(require("MATMUL"), registry.table);
        suite.check("rewrites/MATMUL/regroup", matmul_rewrites.regroup);
        suite.check("rewrites/MATMUL/not-reorder", !matmul_rewrites.reorder);
        suite.check("rewrites/MATMUL/distribute", matmul_rewrites.distribute);
        suite.check("rewrites/MATMUL/elide-unit", matmul_rewrites.elide_unit);
        suite.check("rewrites/MATMUL/not-anticonmutative-swap", !matmul_rewrites.swap_with_sign);

        const aoi::RewriteSet cross_rewrites = aoi::rewrites_for(require("CROSS_PRODUCT"), registry.table);
        suite.check("rewrites/CROSS/not-regroup", !cross_rewrites.regroup);
        suite.check("rewrites/CROSS/swap-with-sign", cross_rewrites.swap_with_sign);
        suite.check("rewrites/CROSS/distribute", cross_rewrites.distribute);
        suite.check("rewrites/CROSS/no-unit-to-elide", !cross_rewrites.elide_unit);

        const aoi::RewriteSet cosine_rewrites =
            aoi::rewrites_for(require("COSINE_SIMILARITY"), registry.table);
        const aoi::RewriteSet dot_rewrites = aoi::rewrites_for(require("DOT_PRODUCT"), registry.table);
        suite.check("rewrites/DOT/distribute", dot_rewrites.distribute);
        suite.check("rewrites/COSINE/does-not-distribute", !cosine_rewrites.distribute);

        // ── 10. Unitality and bilinearity, checked on the numbers ───────────
        // Matrix multiplication is associative, unital and not commutative.
        const Mat2 A{1.0, 2.0, 3.0, 4.0};
        const Mat2 B{5.0, 6.0, 7.0, 8.0};
        const Mat2 C{9.0, 10.0, 11.0, 12.0};
        suite.check("algebra/matmul-associative",
                    same(matmul(matmul(A, B), C), matmul(A, matmul(B, C))));
        suite.check("algebra/matmul-unital-left", same(matmul(kIdentity, A), A));
        suite.check("algebra/matmul-unital-right", same(matmul(A, kIdentity), A));
        suite.check("algebra/matmul-not-commutative", !same(matmul(A, B), matmul(B, A)));
        suite.check("algebra/matmul-distributes", same(matmul(A, matadd(B, C)), matadd(matmul(A, B), matmul(A, C))));

        // The cross product is bilinear, distributive, anticommutative, and
        // neither associative nor unital.
        const Vec3 p{1.0, 2.0, 3.0};
        const Vec3 q{4.0, 5.0, 6.0};
        const Vec3 r{7.0, 8.0, 9.0};
        suite.check("algebra/cross-not-associative",
                    !same(cross(cross(p, q), r), cross(p, cross(q, r))));
        suite.check("algebra/cross-anticommutative", same(cross(q, p), negate(cross(p, q))));
        suite.check("algebra/cross-not-commutative", !same(cross(p, q), cross(q, p)));
        suite.check("algebra/cross-distributes",
                    same(cross(p, add(q, r)), add(cross(p, q), cross(p, r))));

        // The inner product is bilinear; cosine similarity is the same operation
        // after normalization and is not. A derived metric does not inherit the
        // algebra of what it normalizes.
        const std::vector<double> u{1.0, 0.0};
        const std::vector<double> v{1.0, 1.0};
        const std::vector<double> w{1.0, -1.0};
        suite.check("algebra/dot-bilinear", dot(u, add(v, w)) == dot(u, v) + dot(u, w));
        suite.check("algebra/cosine-not-bilinear",
                    cosine(u, add(v, w)) != cosine(u, v) + cosine(u, w));

        // ── 11. The unit is the seed, the phase is not the data ─────────────
        suite.check("algebra/unital-seeds-empty-fold", sum.empty_fold);
        suite.check("algebra/nonunital-has-no-empty-seed", !cross_policy.empty_fold);
        suite.check("algebra/empty-fold-of-unital-returns-unit",
                    aoi::fold_with_plan(std::vector<double>{}, 0.0, addition,
                                        aoi::FoldPlan{aoi::Grouping::Balanced, 64, {}}) == 0.0);

        const aoi::PhaseDecision identity_same = aoi::check_phase_transition(require("IDENTITY_STATE"), true);
        const aoi::PhaseDecision identity_cross = aoi::check_phase_transition(require("IDENTITY_STATE"), false);
        suite.check("phase/identity/allowed-in-phase", identity_same.allowed);
        suite.check("phase/identity/cannot-cross", !identity_cross.allowed);
        suite.check("phase/identity/code", identity_cross.code == "identity_cannot_advance_phase");

        const aoi::PhaseDecision transform_cross =
            aoi::check_phase_transition(require("IDENTITY_TRANSFORM"), false);
        suite.check("phase/identity-transform/cannot-cross", !transform_cross.allowed);

        const aoi::PhaseDecision advance_cross = aoi::check_phase_transition(require("ADVANCE_PHASE"), false);
        suite.check("phase/advance/allowed", advance_cross.allowed);
        suite.check("phase/advance/still-needs-xcfe",
                    advance_cross.detail.find("XCFE") != std::string::npos);

        const aoi::PhaseDecision reducer_cross = aoi::check_phase_transition(require("SUM_SCORES"), false);
        suite.check("phase/reducer/no-authority", !reducer_cross.allowed);
        suite.check("phase/reducer/code", reducer_cross.code == "no_phase_authority");

        // ── 12. A = K ⊕ V: the square-zero state algebra ────────────────────
        // (a,v)(b,w) = (ab, aw + bv) with v.w = 0. Exact integers, so every
        // equality below is a statement about the algebra.
        const szero::Pair alpha = szero::pair(2.0, {1.0, 0.0, 0.0});
        const szero::Pair beta = szero::pair(3.0, {0.0, 1.0, 2.0});
        const szero::Pair gamma = szero::pair(-1.0, {4.0, 0.0, 1.0});
        const szero::Pair one = szero::one(3);
        const szero::Pair zero_pair = szero::zero(3);

        suite.check("squarezero/unit-is-one-zero", one.scalar == 1.0 && one.vector[0] == 0.0);
        suite.check("squarezero/left-unit", szero::mul(one, alpha) == alpha);
        suite.check("squarezero/right-unit", szero::mul(alpha, one) == alpha);
        suite.check("squarezero/zero-absorbs", szero::mul(alpha, zero_pair) == zero_pair);

        suite.check("squarezero/associative",
                    szero::mul(szero::mul(alpha, beta), gamma) == szero::mul(alpha, szero::mul(beta, gamma)));
        suite.check("squarezero/commutative", szero::mul(alpha, beta) == szero::mul(beta, alpha));

        // Bilinearity: a.(b+c) = a.b + a.c, on both sides.
        suite.check("squarezero/distributes-right",
                    szero::mul(alpha, szero::add(beta, gamma)) ==
                        szero::add(szero::mul(alpha, beta), szero::mul(alpha, gamma)));
        suite.check("squarezero/distributes-left",
                    szero::mul(szero::add(beta, gamma), alpha) ==
                        szero::add(szero::mul(beta, alpha), szero::mul(gamma, alpha)));

        // V is square-zero: the internal product of two vectors vanishes.
        const std::vector<double> v1{1.0, 2.0, 3.0};
        const std::vector<double> v2{4.0, 5.0, 6.0};
        suite.check("squarezero/vector-product-is-zero",
                    szero::square_is_zero(v1, v2));
        suite.check("squarezero/nilpotent-element-is-not-a-unit",
                    !szero::inverse(szero::pair(0.0, v1)).has_value());

        // (a,v) inverts to (1/a, -v/a^2) exactly when a != 0.
        const auto inverse = szero::inverse(szero::pair(2.0, {2.0, 0.0, 0.0}));
        suite.check("squarezero/scalar-elements-invert",
                    inverse.has_value() &&
                        szero::mul(szero::pair(2.0, {2.0, 0.0, 0.0}), *inverse) == one);

        // Compartments: submodules of V are ideals of A contained in V.
        const szero::Subspace evidence = szero::span_of({0});
        const szero::Subspace candidate_span = szero::span_of({1});
        const szero::Subspace uncertainty = szero::span_of({2});
        const szero::Subspace pair_span = szero::sum(evidence, candidate_span);
        const szero::Subspace whole = szero::span_of({0, 1, 2});

        const std::vector<szero::Pair> probes = {alpha, beta, gamma, one};
        suite.check("compartments/evidence-is-an-ideal",
                    szero::is_ideal(evidence, probes, 3));
        suite.check("compartments/pair-is-an-ideal", szero::is_ideal(pair_span, probes, 3));
        suite.check("compartments/v-itself-is-an-ideal", szero::is_ideal(whole, probes, 3));
        suite.check("compartments/inclusion-is-preserved",
                    szero::includes(pair_span, evidence) && szero::includes(whole, pair_span));
        suite.check("compartments/sum-is-union",
                    szero::sum(evidence, uncertainty) == szero::span_of({0, 2}));
        suite.check("compartments/intersection-is-intersection",
                    szero::intersection(pair_span, candidate_span) == candidate_span);
        suite.check("compartments/embedding-has-no-scalar-part",
                    szero::embed(evidence, 3).scalar == 0.0);
        suite.check("compartments/projection-drops-the-scalar",
                    szero::project(alpha, evidence, 3).scalar == 0.0 &&
                        szero::project(alpha, evidence, 3).vector[0] == 1.0);

        // The headline distinction: zero multiplication is not zero computation.
        // The internal product of v1 and v2 is zero; the external bilinear form is
        // not, and 0 and 32 are different answers to different questions.
        suite.check("external/inner-product-is-not-the-algebra-product",
                    szero::external_inner_product(v1, v2) == 32.0);
        suite.check("external/zero-product-does-not-imply-zero-form",
                    szero::square_is_zero(v1, v2) &&
                        szero::external_inner_product(v1, v2) != 0.0);

        // The contract itself validates and is representational, not executable.
        const aoi::AlgebraContract& state_algebra = require("MICRONAUT_STATE_ALGEBRA");
        suite.check("squarezero/contract-validates",
                    aoi::validate(state_algebra, registry.table).empty());
        suite.check("squarezero/contract-is-unital",
                    state_algebra.laws.is(aoi::Law::Unital, aoi::LawState::Yes));
        suite.check("squarezero/contract-names-the-unit",
                    state_algebra.identity.has_value() && *state_algebra.identity == "1_A");
        suite.check("squarezero/contract-is-not-invertible",
                    state_algebra.laws.is(aoi::Law::Invertible, aoi::LawState::No));
        suite.check("squarezero/contract-is-representational",
                    state_algebra.is_reducer == false &&
                        state_algebra.phase_policy == aoi::PhasePolicy::None);

        // ── 13. Fixtures: the same file the Python checker reads ────────────
        JsonPtr fixtures = khanary::util::parse(read_file(data_root + "/conformance/algebra_isa.json"));

        for (const auto& entry : fixtures->get("positive")->as_array()) {
            if (entry == nullptr) continue;
            const std::string id = entry->get_string("id").value_or("?");
            const JsonValue* block = entry->get("algebra");
            if (block == nullptr) {
                suite.check("fixtures/positive/" + id, false);
                continue;
            }
            khanary::util::JsonObject wrapper = {{"id", std::make_shared<JsonValue>(id)},
                                                 {"algebra", std::make_shared<JsonValue>(*block)}};
            JsonValue record(wrapper);
            const std::vector<aoi::Violation> violations =
                aoi::validate_entry(record, registry.table);
            suite.check("fixtures/positive/" + id, violations.empty());

            if (const JsonValue* expected = entry->get("expect_policy")) {
                aoi::ParsedContract parsed = aoi::parse_contract(record);
                const aoi::ReductionPolicy policy = aoi::derive_policy(parsed.contract, registry.table);
                const aoi::RewriteSet rewrites = aoi::rewrites_for(parsed.contract, registry.table);
                suite.check("fixtures/policy/" + id,
                            matches_expected_policy(policy, rewrites, parsed.contract, *expected));
            }
        }

        for (const auto& entry : fixtures->get("negative")->as_array()) {
            if (entry == nullptr) continue;
            const std::string id = entry->get_string("id").value_or("?");
            const std::string expected = entry->get_string("expect_code").value_or("");
            const JsonValue* block = entry->get("algebra");
            std::vector<aoi::Violation> violations;
            if (block == nullptr) {
                violations.push_back(aoi::Violation{"MISSING_ALGEBRA_BLOCK", id});
            } else {
                khanary::util::JsonObject wrapper = {{"id", std::make_shared<JsonValue>(id)},
                                                     {"algebra", std::make_shared<JsonValue>(*block)}};
                JsonValue record(wrapper);
                violations = aoi::validate_entry(record, registry.table);
            }
            suite.check("fixtures/negative/" + id, has_code(violations, expected));
        }

        // ── 14. The lattice fixture agrees with the runtime lattice ─────────
        if (const JsonValue* lattice = fixtures->get("lattice")) {
            std::vector<aoi::EffectSet> declared;
            const JsonValue* authorities_json = lattice->get("authorities");
            if (authorities_json != nullptr && authorities_json->is_object()) {
                for (const auto& [name, values] : authorities_json->as_object()) {
                    aoi::EffectSet set;
                    if (values != nullptr && values->is_array()) {
                        for (const auto& value : values->as_array()) {
                            if (value != nullptr && value->is_string()) {
                                set.insert(value->as_string());
                            }
                        }
                    }
                    declared.push_back(std::move(set));
                }
            }
            const aoi::EffectSet fixture_effective = aoi::effective_permissions(declared);
            aoi::EffectSet expected;
            const JsonValue* expect_effective = lattice->get("expect_effective");
            if (expect_effective != nullptr && expect_effective->is_array()) {
                for (const auto& value : expect_effective->as_array()) {
                    if (value != nullptr && value->is_string()) expected.insert(value->as_string());
                }
            }
            suite.check("fixtures/lattice/effective", fixture_effective == expected);
            suite.check("fixtures/lattice/never-widens",
                        aoi::grants_only_attested(fixture_effective, declared));
        }

        // ── 15. The square-zero fixture agrees with the state algebra ───────
        if (const JsonValue* square_zero = fixtures->get("square_zero")) {
            const std::size_t dimension =
                static_cast<std::size_t>(square_zero->get_number("dimension").value_or(0.0));
            suite.check("fixtures/squarezero/dimension", dimension == std::size_t{3});

            std::vector<szero::Pair> samples;
            if (const JsonValue* list = square_zero->get("sample_pairs")) {
                for (const auto& entry : list->as_array()) {
                    if (entry == nullptr) continue;
                    std::vector<double> vector;
                    if (const JsonValue* values = entry->get("vector")) {
                        for (const auto& value : values->as_array()) {
                            if (value != nullptr) vector.push_back(value->as_number());
                        }
                    }
                    samples.push_back(szero::pair(entry->get_number("scalar").value_or(0.0),
                                                  std::move(vector)));
                }
            }
            suite.check("fixtures/squarezero/samples", samples.size() == std::size_t{3});

            const szero::Pair unit = szero::one(dimension);
            bool unit_ok = !samples.empty();
            for (const szero::Pair& sample : samples) {
                unit_ok = unit_ok && szero::mul(unit, sample) == sample &&
                          szero::mul(sample, unit) == sample;
            }
            suite.check("fixtures/squarezero/unit-law", unit_ok);

            std::vector<std::vector<double>> zero_vectors;
            if (const JsonValue* list = square_zero->get("zero_vectors")) {
                for (const auto& entry : list->as_array()) {
                    if (entry == nullptr) continue;
                    std::vector<double> vector;
                    for (const auto& value : entry->as_array()) {
                        if (value != nullptr) vector.push_back(value->as_number());
                    }
                    zero_vectors.push_back(std::move(vector));
                }
            }
            if (zero_vectors.size() >= 2) {
                const double expected =
                    square_zero->get_number("expect_external_inner_product").value_or(0.0);
                suite.check("fixtures/squarezero/product-is-zero",
                            szero::square_is_zero(zero_vectors[0], zero_vectors[1]));
                suite.check("fixtures/squarezero/external-form-matches",
                            szero::external_inner_product(zero_vectors[0], zero_vectors[1]) ==
                                expected);
                suite.check("fixtures/squarezero/external-form-is-not-zero", expected != 0.0);
            }

            // Ideals: (0,W) closed under multiplication, and the lattice
            // operations preserved by the embedding.
            std::map<std::string, szero::Subspace> ideals;
            if (const JsonValue* list = square_zero->get("ideals")) {
                for (const auto& entry : list->as_array()) {
                    if (entry == nullptr) continue;
                    const std::string id = entry->get_string("id").value_or("");
                    szero::Subspace coordinates;
                    if (const JsonValue* values = entry->get("coordinates")) {
                        for (const auto& value : values->as_array()) {
                            if (value == nullptr) continue;
                            coordinates.insert(static_cast<std::size_t>(value->as_number()));
                        }
                    }
                    ideals.emplace(id, std::move(coordinates));
                }
            }

            bool every_ideal_closed = !ideals.empty();
            for (const auto& [id, coordinates] : ideals) {
                every_ideal_closed =
                    every_ideal_closed && szero::is_ideal(coordinates, samples, dimension);
            }
            suite.check("fixtures/squarezero/every-compartment-is-an-ideal",
                        every_ideal_closed);

            bool inclusions_ok = true;
            if (const JsonValue* list = square_zero->get("inclusions")) {
                for (const auto& entry : list->as_array()) {
                    if (entry == nullptr) continue;
                    const auto smaller = ideals.find(entry->get_string("smaller").value_or(""));
                    const auto larger = ideals.find(entry->get_string("larger").value_or(""));
                    if (smaller == ideals.end() || larger == ideals.end()) {
                        inclusions_ok = false;
                        continue;
                    }
                    inclusions_ok =
                        inclusions_ok && szero::includes(larger->second, smaller->second);
                }
            }
            suite.check("fixtures/squarezero/inclusions-preserved", inclusions_ok);

            bool operations_ok = true;
            if (const JsonValue* list = square_zero->get("operations")) {
                for (const auto& entry : list->as_array()) {
                    if (entry == nullptr) continue;
                    const auto lhs = ideals.find(entry->get_string("a").value_or(""));
                    const auto rhs = ideals.find(entry->get_string("b").value_or(""));
                    if (lhs == ideals.end() || rhs == ideals.end()) {
                        operations_ok = false;
                        continue;
                    }
                    szero::Subspace expected_coordinates;
                    if (const JsonValue* values = entry->get("expect_coordinates")) {
                        for (const auto& value : values->as_array()) {
                            if (value == nullptr) continue;
                            expected_coordinates.insert(
                                static_cast<std::size_t>(value->as_number()));
                        }
                    }
                    const std::string op = entry->get_string("op").value_or("");
                    const szero::Subspace actual =
                        op == "sum" ? szero::sum(lhs->second, rhs->second)
                                    : szero::intersection(lhs->second, rhs->second);
                    operations_ok = operations_ok && actual == expected_coordinates;
                }
            }
            suite.check("fixtures/squarezero/operations-preserved", operations_ok);
        }

        // ── 16. The pipeline consumes the contract, not the call site ──────
        // Same inputs, same result, different legal grouping — and the report
        // names which grouping ran and why.
        graphscript::ActiveField field;
        field.vertices = {{1.0f}, {2.0f}, {3.0f}, {4.0f}};
        field.adjacency.assign(4, std::vector<float>(4, 0.0f));

        const auto identity_map = [](const graphscript::VertexValues& values) { return values; };
        const auto keep_all = [](const graphscript::VertexValues&) { return true; };
        const auto sum_values = [](const graphscript::VertexValues& a,
                                   const graphscript::VertexValues& b) {
            graphscript::VertexValues out = a;
            if (out.empty()) return b;
            if (!b.empty()) out[0] += b[0];
            return out;
        };
        const graphscript::VertexValues zero = {0.0f};

        graphscript::PipelineReport regrouped;
        const graphscript::VertexValues balanced =
            graphscript::execute_graph_pipeline(field, identity_map, keep_all, sum_values, zero,
                                                sum_contract, &regrouped);
        suite.check("pipeline/regrouped-value", !balanced.empty() && balanced[0] == 10.0f);
        suite.check("pipeline/regrouped-grouping", regrouped.fold_grouping == "balanced");

        graphscript::PipelineReport withheld;
        const graphscript::VertexValues sequential =
            graphscript::execute_graph_pipeline(field, identity_map, keep_all, sum_values, zero,
                                                require("CROSS_PRODUCT"), &withheld);
        suite.check("pipeline/withheld-value", !sequential.empty() && sequential[0] == 10.0f);
        suite.check("pipeline/withheld-grouping", withheld.fold_grouping == "sequential");
        suite.check("pipeline/withheld-reason-recorded",
                    contains(withheld.fold_reasons, "not_associative"));

        graphscript::PipelineReport legacy;
        const graphscript::VertexValues left_fold_values =
            graphscript::execute_graph_pipeline(field, identity_map, keep_all, sum_values, zero,
                                                &legacy);
        suite.check("pipeline/legacy-is-sequential", legacy.fold_grouping == "sequential");
        suite.check("pipeline/legacy-matches-regrouped", left_fold_values == balanced);

        // ── 17. Cl(2,0): geometric algebra laws and phase correspondences ──────
        // e1²=+1, e2²=+1, e12²=-1, e1*e2=e12, e2*e1=-e12
        namespace cl = khanary::kuhul::statealgebra;
        const cl::Cl20 e1 = cl::cl20_vector(1.0, 0.0);
        const cl::Cl20 e2 = cl::cl20_vector(0.0, 1.0);
        const cl::Cl20 e12 = cl::cl20_bivector(1.0);
        const cl::Cl20 one_cl = cl::cl20_scalar(1.0);

        // Basis products — the signature contract
        suite.check("cl20/e1-sq-is-scalar-1",     cl::cl20_mul(e1, e1)   == one_cl);
        suite.check("cl20/e2-sq-is-scalar-1",     cl::cl20_mul(e2, e2)   == one_cl);
        suite.check("cl20/e12-sq-is-scalar-neg1", cl::cl20_mul(e12, e12) == cl::cl20_scalar(-1.0));
        suite.check("cl20/e1e2-is-e12",           cl::cl20_mul(e1, e2)   == e12);
        suite.check("cl20/e2e1-is-neg-e12",          cl::cl20_mul(e2, e1)   == cl::cl20_bivector(-1.0));

        // Unitality: 1*e1 = e1
        suite.check("cl20/unit-left",  cl::cl20_mul(one_cl, e1) == e1);
        suite.check("cl20/unit-right", cl::cl20_mul(e1, one_cl) == e1);

        // Non-commutativity: e1*e2 != e2*e1
        suite.check("cl20/not-commutative", cl::cl20_mul(e1, e2) != cl::cl20_mul(e2, e1));

        // Associativity: (e1*e2)*e1 == e1*(e2*e1) == (0,0,-1,0)
        const cl::Cl20 expect_neg_e2 = cl::cl20_vector(0.0, -1.0);
        suite.check("cl20/associative-left",
                    cl::cl20_mul(cl::cl20_mul(e1, e2), e1) == expect_neg_e2);
        suite.check("cl20/associative-right",
                    cl::cl20_mul(e1, cl::cl20_mul(e2, e1)) == expect_neg_e2);
        suite.check("cl20/associative-both-paths-agree",
                    cl::cl20_mul(cl::cl20_mul(e1, e2), e1) ==
                        cl::cl20_mul(e1, cl::cl20_mul(e2, e1)));

        // Reverse: negates grade-2 only
        const cl::Cl20 mixed = cl::cl20_multivector(1.0, 2.0, 3.0, 4.0);
        const cl::Cl20 rev = cl::cl20_reverse(mixed);
        suite.check("cl20/reverse-grade0-unchanged", rev.s == mixed.s);
        suite.check("cl20/reverse-grade1-unchanged", rev.x == mixed.x && rev.y == mixed.y);
        suite.check("cl20/reverse-grade2-negated", rev.b == -mixed.b);

        // Phase product: scalar == arc_weight, full-angle (NOT a rotation rotor)
        {
            const cl::Cl20 pp = cl::cl20_phase_product(0, 1);  // Pop→Wo, Δθ=π/3
            const double arc_weight_pop_wo = std::cos(std::numbers::pi / 3.0);
            suite.check("cl20/phase-product-scalar-is-arc-weight",
                        std::abs(pp.s - arc_weight_pop_wo) < 1e-12);
            suite.check("cl20/phase-product-vector-zero", pp.x == 0.0 && pp.y == 0.0);
        }

        // Phase rotor: half-angle, sandwich rotates by Δθ, not 2Δθ
        {
            // Pop(0)→Wo(1): rotate e0=(1,0) by π/3. Expect (cos π/3, sin π/3) = (0.5, 0.866…)
            const cl::Cl20 rotor = cl::cl20_phase_rotor(0, 1);
            const cl::Cl20 e0 = cl::cl20_phase_vector(0);
            const cl::Cl20 rotated = cl::cl20_rotate(e0, rotor);
            const cl::Cl20 e1_expect = cl::cl20_phase_vector(1);
            suite.check("cl20/rotor-rotates-to-e1",
                        std::abs(rotated.x - e1_expect.x) < 1e-10 &&
                        std::abs(rotated.y - e1_expect.y) < 1e-10);
        }

        // GA_PHASE_PRODUCT != GA_PHASE_ROTATION: using the full-angle product in a
        // sandwich would rotate by 2Δθ=2π/3, not π/3.
        {
            const cl::Cl20 full_angle = cl::cl20_phase_product(0, 1);
            const cl::Cl20 e0 = cl::cl20_phase_vector(0);
            const cl::Cl20 rotated_double = cl::cl20_rotate(e0, full_angle);
            const cl::Cl20 e2_expect = cl::cl20_phase_vector(2);  // Yax = 2π/3
            suite.check("cl20/full-angle-product-rotates-double",
                        std::abs(rotated_double.x - e2_expect.x) < 1e-10 &&
                        std::abs(rotated_double.y - e2_expect.y) < 1e-10);
        }

        // Orientation bivector: a∧b = a.x*b.y - a.y*b.x (telemetry, not authority)
        suite.check("cl20/orient-e1-e2-ccw", cl::cl20_outer_b(e1, e2) == 1.0);
        suite.check("cl20/orient-e2-e1-cw",  cl::cl20_outer_b(e2, e1) == -1.0);

        // cl20_vector_norm_sq: valid only for pure grade-1
        suite.check("cl20/norm-sq-e1",  cl::cl20_vector_norm_sq(e1)  == 1.0);
        suite.check("cl20/norm-sq-sum", cl::cl20_vector_norm_sq(cl::cl20_add(e1, e2)) == 2.0);

        // encode_twist: opt-in, NOT isometric for mixed-grade
        {
            const cl::Cl20 twist = cl::cl20_encode_twist(1.0, 0.0, 1.0);
            suite.check("cl20/encode-twist-maps-z-to-bivector",
                        twist.x == 1.0 && twist.y == 0.0 && twist.b == 1.0);
            // M² = x²+y²-z² for the scalar part: 1²+0²-1² = 0, so NOT squared-magnitude
            const cl::Cl20 sq = cl::cl20_mul(twist, twist);
            suite.check("cl20/encode-twist-not-isometric", sq.s == 0.0 && sq != cl::cl20_scalar(1.0));
        }

        // GA_PRODUCT contract validates
        const aoi::AlgebraContract& ga_contract = require("GA_PRODUCT");
        suite.check("cl20/ga-product-contract-validates",
                    aoi::validate(ga_contract, registry.table).empty());
        suite.check("cl20/ga-product-associative",
                    ga_contract.laws.is(aoi::Law::Associative, aoi::LawState::Yes));
        suite.check("cl20/ga-product-not-commutative",
                    ga_contract.laws.is(aoi::Law::Commutative, aoi::LawState::No));
        suite.check("cl20/ga-product-no-phase-authority",
                    ga_contract.phase_policy == aoi::PhasePolicy::None);

        // Clifford fixtures
        if (const JsonValue* clifford = fixtures->get("clifford")) {
            if (const JsonValue* products = clifford->get("basis_products")) {
                auto read_cl20 = [](const JsonValue& v) -> cl::Cl20 {
                    return {v.get_number("s").value_or(0.0),
                            v.get_number("x").value_or(0.0),
                            v.get_number("y").value_or(0.0),
                            v.get_number("b").value_or(0.0)};
                };
                for (const auto& entry : products->as_array()) {
                    if (entry == nullptr) continue;
                    const std::string fid = entry->get_string("id").value_or("?");
                    const std::string op = entry->get_string("op").value_or("product");
                    if (op == "associativity_check") {
                        const cl::Cl20 fa = read_cl20(*entry->get("a"));
                        const cl::Cl20 fb = read_cl20(*entry->get("b"));
                        const cl::Cl20 fc = read_cl20(*entry->get("c"));
                        const cl::Cl20 fab_c = read_cl20(*entry->get("expect_ab_c"));
                        const cl::Cl20 fa_bc = read_cl20(*entry->get("expect_a_bc"));
                        suite.check("fixtures/clifford/" + fid,
                                    cl::cl20_mul(cl::cl20_mul(fa, fb), fc) == fab_c &&
                                    cl::cl20_mul(fa, cl::cl20_mul(fb, fc)) == fa_bc);
                    } else {
                        const cl::Cl20 fa = read_cl20(*entry->get("a"));
                        const cl::Cl20 fb = read_cl20(*entry->get("b"));
                        const cl::Cl20 fexpect = read_cl20(*entry->get("expect"));
                        suite.check("fixtures/clifford/" + fid, cl::cl20_mul(fa, fb) == fexpect);
                    }
                }
            }
        }
    });
}
