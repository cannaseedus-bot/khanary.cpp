#pragma once
//
// Algebraic Operator ISA — the law surface of an executable operator.
//
// Every XCFE/SRAI operator and every Fold reducer declares the algebraic
// structure it inhabits plus the individual laws that structure licenses.
// Execution decisions (regroup, reorder, swap-with-sign, distribute, elide a
// unit, parallelise, repeat, invert, empty fold, cross a phase) are *derived*
// from that declaration. Nothing is inferred from the operator's name, and
// nothing is assumed to be permitted because it was merely not forbidden.
//
// Three-valued laws are the whole point:
//
//     Yes            the law holds
//     No             the law provably fails
//     NotApplicable  the law does not apply (a unary operator has no
//                    associativity; the combiner does)
//
// `NotApplicable` never means "unchecked" and never means "permitted".
// Collapsing it into "probably fine" is the defect this layer exists to
// prevent.
//
// Two distinctions carry most of the weight:
//
//   * Unitality is a law, not a flag. 1_A . x = x . 1_A = x, and unital does not
//     imply commutative: matrix multiplication is associative, unital and
//     not commutative, which is exactly the shape of ordered transformation
//     composition. The unit is also the seed that lets an empty fold return a
//     defined value without inventing a candidate.
//
//   * Mathematical associativity is not executable associativity. Real addition
//     is associative; IEEE-754 float addition is not. A contract therefore
//     carries a reproducibility policy next to its laws, and regrouping is
//     licensed only when the law AND the carrier agree.
//
// A no-op inside a phase is not a phase transition: an identity operator
// preserves the phase, and only PhasePolicy::Advance may request a crossing
// (which XCFE still has to admit).
//
// Authority: data/grammar/algebra-isa.contracts.json (axiom table, contract
// registry) and data/grammar/algebra-isa.ebnf (law algebra).
// Registry/implementation parity is asserted by KHANARY_ALGEBRA_ISA_TEST.
//
// This header has no dependency on the phase engine: a caller may classify an
// operator without linking the K'UHUL runtime.
//
#include "../util/json.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace khanary::kuhul::algebra {

// ── Laws ─────────────────────────────────────────────────────────────────────

enum class LawState : std::uint8_t {
    No = 0,             // the law provably fails
    NotApplicable = 1,  // the law does not apply
    Yes = 2,            // the law holds
};

inline const char* law_state_token(LawState state) {
    switch (state) {
        case LawState::No:            return "false";
        case LawState::NotApplicable: return "null";
        case LawState::Yes:           return "true";
    }
    return "null";
}

inline std::optional<LawState> law_state_from_token(const std::string& token) {
    if (token == "true") return LawState::Yes;
    if (token == "false") return LawState::No;
    if (token == "null") return LawState::NotApplicable;
    return std::nullopt;
}

enum class Law : std::uint8_t {
    Associative = 0,
    Commutative = 1,
    Idempotent = 2,
    Invertible = 3,
    Distributive = 4,
    Unital = 5,
    Bilinear = 6,
    Anticommutative = 7,
};

inline constexpr std::size_t kLawCount = 8;

inline const char* law_name(Law law) {
    switch (law) {
        case Law::Associative:     return "associative";
        case Law::Commutative:     return "commutative";
        case Law::Idempotent:      return "idempotent";
        case Law::Invertible:      return "invertible";
        case Law::Distributive:    return "distributive";
        case Law::Unital:          return "unital";
        case Law::Bilinear:        return "bilinear";
        case Law::Anticommutative: return "anticommutative";
    }
    return "";
}

inline std::optional<Law> law_from_name(const std::string& name) {
    for (std::size_t i = 0; i < kLawCount; ++i) {
        const Law law = static_cast<Law>(i);
        if (name == law_name(law)) return law;
    }
    return std::nullopt;
}

struct LawSet {
    std::array<LawState, kLawCount> states{};

    static LawSet all_not_applicable() {
        LawSet set;
        set.states.fill(LawState::NotApplicable);
        return set;
    }
    LawState get(Law law) const { return states[static_cast<std::size_t>(law)]; }
    void set(Law law, LawState state) { states[static_cast<std::size_t>(law)] = state; }
    bool is(Law law, LawState state) const { return get(law) == state; }
};

// ── Structures ───────────────────────────────────────────────────────────────

enum class Structure : std::uint8_t {
    None = 0,
    ScalarArithmetic,
    Monoid,
    CommutativeMonoid,
    Group,
    AbelianGroup,
    SemilatticeMeet,
    SemilatticeJoin,
    Lattice,
    Ring,
    Module,
    Algebra,
    AssociativeAlgebra,
    UnitalAssociativeAlgebra,
    NonassociativeAlgebra,
    InnerProductSpace,
    SquareZeroExtension,
    CliffordAlgebra,
};

inline constexpr std::size_t kStructureCount = 18;

inline const char* structure_name(Structure structure) {
    switch (structure) {
        case Structure::None:                     return "none";
        case Structure::ScalarArithmetic:         return "scalar_arithmetic";
        case Structure::Monoid:                   return "monoid";
        case Structure::CommutativeMonoid:        return "commutative_monoid";
        case Structure::Group:                    return "group";
        case Structure::AbelianGroup:             return "abelian_group";
        case Structure::SemilatticeMeet:          return "semilattice_meet";
        case Structure::SemilatticeJoin:          return "semilattice_join";
        case Structure::Lattice:                  return "lattice";
        case Structure::Ring:                     return "ring";
        case Structure::Module:                   return "module";
        case Structure::Algebra:                  return "algebra";
        case Structure::AssociativeAlgebra:       return "associative_algebra";
        case Structure::UnitalAssociativeAlgebra: return "unital_associative_algebra";
        case Structure::NonassociativeAlgebra:    return "nonassociative_algebra";
        case Structure::InnerProductSpace:        return "inner_product_space";
        case Structure::SquareZeroExtension:      return "square_zero_extension";
        case Structure::CliffordAlgebra:          return "clifford_algebra";
    }
    return "none";
}

inline std::optional<Structure> structure_from_name(const std::string& name) {
    for (std::size_t i = 0; i < kStructureCount; ++i) {
        const Structure structure = static_cast<Structure>(i);
        if (name == structure_name(structure)) return structure;
    }
    return std::nullopt;
}

// ── Product image, reproducibility, grouping ─────────────────────────────────

enum class ProductImage : std::uint8_t {
    None = 0,     // no binary product declared
    Carrier = 1,  // V × V → V  (composable as a multiplication)
    Scalar = 2,   // V × V → R  (inner product; NOT an algebra multiplication)
};

inline const char* product_image_name(ProductImage image) {
    switch (image) {
        case ProductImage::None:    return "none";
        case ProductImage::Carrier: return "carrier";
        case ProductImage::Scalar:  return "scalar";
    }
    return "none";
}

inline std::optional<ProductImage> product_image_from_name(const std::string& name) {
    if (name == "none") return ProductImage::None;
    if (name == "carrier") return ProductImage::Carrier;
    if (name == "scalar") return ProductImage::Scalar;
    return std::nullopt;
}

enum class Reproducibility : std::uint8_t {
    Unspecified = 0,
    Exact = 1,                  // every grouping returns the identical value
    DeterministicGrouping = 2,  // one canonical tree is byte-reproducible
    ImplementationDefined = 3,
};

inline const char* reproducibility_name(Reproducibility value) {
    switch (value) {
        case Reproducibility::Unspecified:          return "unspecified";
        case Reproducibility::Exact:                return "exact";
        case Reproducibility::DeterministicGrouping: return "deterministic_grouping";
        case Reproducibility::ImplementationDefined: return "implementation_defined";
    }
    return "unspecified";
}

inline std::optional<Reproducibility> reproducibility_from_name(const std::string& name) {
    if (name == "exact") return Reproducibility::Exact;
    if (name == "deterministic_grouping") return Reproducibility::DeterministicGrouping;
    if (name == "implementation_defined") return Reproducibility::ImplementationDefined;
    if (name == "unspecified") return Reproducibility::Unspecified;
    return std::nullopt;
}

enum class Grouping : std::uint8_t {
    Sequential = 0,  // left fold, canonical order
    Chunked = 1,     // contiguous chunks reduced then combined left-to-right
    Balanced = 2,    // contiguous chunks reduced then combined as a tree
    Free = 3,        // grouping and operand order both unconstrained
};

inline const char* grouping_name(Grouping grouping) {
    switch (grouping) {
        case Grouping::Sequential: return "sequential";
        case Grouping::Chunked:    return "chunked";
        case Grouping::Balanced:   return "balanced";
        case Grouping::Free:       return "free";
    }
    return "sequential";
}

inline std::optional<Grouping> grouping_from_name(const std::string& name) {
    if (name == "sequential") return Grouping::Sequential;
    if (name == "chunked") return Grouping::Chunked;
    if (name == "balanced") return Grouping::Balanced;
    if (name == "free") return Grouping::Free;
    return std::nullopt;
}

struct GroupingSpec {
    Grouping kind = Grouping::Sequential;
    std::size_t chunk = 1;
};

// ── Phase authority ──────────────────────────────────────────────────────────
//
// A no-op inside a phase is not a phase transition. Wo → Yax changes control
// state even when no candidate data changes, so an identity operator must never
// be usable to cross a phase boundary. Only an operator that declares
// PhasePolicy::Advance may request the transition — and the request still goes
// through XCFE admission; declaring the policy is not the authorization.

enum class PhasePolicy : std::uint8_t {
    None = 0,      // no phase authority at all
    Preserve = 1,  // leaves the phase untouched; may never cross a boundary
    Advance = 2,   // may request a phase transition (subject to XCFE admission)
};

inline const char* phase_policy_name(PhasePolicy policy) {
    switch (policy) {
        case PhasePolicy::None:     return "none";
        case PhasePolicy::Preserve: return "preserve";
        case PhasePolicy::Advance:  return "advance";
    }
    return "none";
}

inline std::optional<PhasePolicy> phase_policy_from_name(const std::string& name) {
    if (name == "none") return PhasePolicy::None;
    if (name == "preserve") return PhasePolicy::Preserve;
    if (name == "advance") return PhasePolicy::Advance;
    return std::nullopt;
}

// ── The contract ─────────────────────────────────────────────────────────────

struct AlgebraContract {
    std::string id;
    Structure structure = Structure::None;
    std::string carrier;
    std::string binary_operation;
    LawSet laws = LawSet::all_not_applicable();
    bool closed = false;
    // `identity` names the element that makes the operation unital. It is the
    // fact behind Law::Unital, not a second flag: unital == Yes requires a name
    // here, and a name here requires unital == Yes.
    std::optional<std::string> identity;
    std::optional<std::string> scalar_field;  // the field scalars come from, when relevant
    bool pure = true;
    bool deterministic = true;
    bool order_sensitive = false;
    ProductImage product_image = ProductImage::None;
    Reproducibility reproducibility = Reproducibility::Unspecified;
    PhasePolicy phase_policy = PhasePolicy::None;
    std::optional<GroupingSpec> canonical_grouping;
    std::string description;
    std::string note;
    bool is_reducer = false;
};

// ── Violations ───────────────────────────────────────────────────────────────

struct Violation {
    std::string code;
    std::string detail;
};

bool has_violations(const std::vector<Violation>& violations);
std::vector<std::string> violation_codes(const std::vector<Violation>& violations);

// ── The axiom table ──────────────────────────────────────────────────────────

enum class LawRequirementKind : std::uint8_t {
    MustHold = 0,       // declared value must equal `value`
    MustBeApplicable = 1,  // declared value must be Yes or No — null is rejected
};

struct LawRequirement {
    Law law = Law::Associative;
    LawRequirementKind kind = LawRequirementKind::MustHold;
    LawState value = LawState::NotApplicable;
};

struct StructureRule {
    Structure structure = Structure::None;
    std::string family;
    std::string description;
    std::vector<LawRequirement> laws;              // absent law = unconstrained
    std::vector<std::pair<std::string, bool>> structural;
    std::optional<ProductImage> product_image;
    std::vector<std::string> binary_operations;    // empty = any operation
    bool requires_scalar_field = false;
};

struct CarrierSpec {
    std::string name;
    bool exact = false;
};

struct StructureTable {
    std::vector<StructureRule> rules;
    std::vector<CarrierSpec> carriers;

    const StructureRule* find(Structure structure) const;
    bool has_carrier(const std::string& carrier) const;
    // Conservative: an unknown carrier is never assumed exact.
    bool carrier_exact(const std::string& carrier) const;
};

const StructureTable& builtin_structure_table();

// Parses the `structures` + `vocabulary.carriers` tables out of the registry
// document. Returns std::nullopt when the document cannot even be read as a
// registry; per-entry problems are reported in `errors`.
std::optional<StructureTable> parse_structure_table(const util::JsonValue& registry,
                                                    std::vector<Violation>& errors);

bool same_structure_rule(const StructureRule& a, const StructureRule& b);
bool same_structure_table(const StructureTable& a, const StructureTable& b, std::string* why);

// ── Parsing and validation ───────────────────────────────────────────────────

struct ParsedContract {
    AlgebraContract contract;
    std::vector<Violation> errors;  // shape errors; semantic checks are validate()
};

// `entry` is one record from `operators` / `reducers`: it must carry `operator`
// and `algebra`.
ParsedContract parse_contract(const util::JsonValue& entry);

// Semantic validation against the binding axiom table.
std::vector<Violation> validate(const AlgebraContract& contract,
                                const StructureTable& table = builtin_structure_table());
std::vector<Violation> validate_entry(const util::JsonValue& entry,
                                      const StructureTable& table = builtin_structure_table());

// ── Derived execution permissions ────────────────────────────────────────────

struct ReductionPolicy {
    bool mathematically_associative = false;
    bool executable_associative = false;
    bool commutative = false;
    bool unital = false;
    bool regroup = false;   // may the fold be regrouped / tree-reduced?
    bool reorder = false;   // may operands be permuted?
    bool parallel = false;  // may independent chunks run concurrently?
    bool repeat = false;    // may the operation be re-applied idempotently?
    bool invert = false;    // may a completed operation be undone?
    bool empty_fold = false;// is the unit available as the seed for an empty sequence?
    bool carrier_exact = false;
    Grouping grouping = Grouping::Sequential;
    std::vector<std::string> reasons;  // why any permission was withheld
};

ReductionPolicy derive_policy(const AlgebraContract& contract,
                              const StructureTable& table = builtin_structure_table());

// Algebraic rewrites licensed by the contract.
//
// This is the answer to "which algebraic rewrites may XCFE perform?". Each flag
// is a theorem over the declared laws; none of them is inferred from the
// operator's name or from the shape of the data it happens to accept.
struct RewriteSet {
    bool regroup = false;         // (a.b).c  ⇄  a.(b.c)      -- associativity
    bool reorder = false;         // a.b      ⇄  b.a          -- commutativity
    bool swap_with_sign = false;  // a.b      ⇄  -(b.a)       -- anticommutativity
    bool distribute = false;      // a.(b+c)  ⇄  a.b + a.c    -- bilinearity
    bool elide_unit = false;      // 1.a      ⇄  a   and  a.1 ⇄ a
    bool repeat = false;          // a.a      ⇄  a            -- idempotence
    bool invert = false;          // a.a^-1   ⇄  1
    std::vector<std::string> reasons;  // why a rewrite was withheld
};

RewriteSet rewrites_for(const AlgebraContract& contract,
                        const StructureTable& table = builtin_structure_table());

// May this operator be used to cross a phase boundary?
struct PhaseDecision {
    bool allowed = false;
    std::string code;
    std::string detail;
};

// `same_phase` means the operator runs inside one phase and no transition is
// requested. A cross-phase step is allowed only for PhasePolicy::Advance, and
// even then the caller still owes XCFE admission.
PhaseDecision check_phase_transition(const AlgebraContract& contract, bool same_phase);

struct FoldPlan {
    Grouping grouping = Grouping::Sequential;
    std::size_t chunk = 1;
    std::vector<std::string> reasons;
};

FoldPlan plan_fold(const AlgebraContract& contract, std::size_t element_count,
                   const StructureTable& table = builtin_structure_table());

// Executes the fold under the plan. Regrouping is the caller's contract: this
// function does not re-check the laws, it obeys the plan it was handed.
template <class T, class Op>
T fold_with_plan(const std::vector<T>& items, const T& identity, Op op, const FoldPlan& plan) {
    if (items.empty()) return identity;
    const std::size_t count = items.size();
    const std::size_t chunk = plan.chunk == 0 ? count : plan.chunk;

    auto sequential_range = [&](std::size_t begin, std::size_t end) {
        T accumulator = identity;
        for (std::size_t i = begin; i < end; ++i) accumulator = op(accumulator, items[i]);
        return accumulator;
    };

    if (plan.grouping == Grouping::Sequential) return sequential_range(0, count);

    std::vector<T> partials;
    partials.reserve((count + chunk - 1) / chunk);
    for (std::size_t begin = 0; begin < count; begin += chunk) {
        partials.push_back(sequential_range(begin, std::min(begin + chunk, count)));
    }
    if (partials.empty()) return identity;

    if (plan.grouping == Grouping::Chunked) {
        T accumulator = identity;
        for (const T& partial : partials) accumulator = op(accumulator, partial);
        return accumulator;
    }

    // Balanced / free: pairwise tree over the chunk results. Free additionally
    // permits permutation, which the caller performs before calling (see
    // ReductionPolicy::reorder); the tree shape itself is identical.
    while (partials.size() > 1) {
        std::vector<T> next;
        next.reserve((partials.size() + 1) / 2);
        for (std::size_t i = 0; i + 1 < partials.size(); i += 2) {
            next.push_back(op(partials[i], partials[i + 1]));
        }
        if (partials.size() % 2 != 0) next.push_back(partials.back());
        partials = std::move(next);
    }
    return partials.front();
}

// ── The effect lattice (XCFE authority aggregation) ──────────────────────────
//
// meet = intersection (combine constraints), join = union (combine
// alternatives). Combining authorities can never grant an effect that none of
// them permitted: that is the whole reason this is a lattice and not a sum.

using EffectSet = std::set<std::string>;

EffectSet effect_meet(const EffectSet& a, const EffectSet& b);
EffectSet effect_join(const EffectSet& a, const EffectSet& b);
EffectSet effective_permissions(const std::vector<EffectSet>& authorities);
EffectSet joined_permissions(const std::vector<EffectSet>& alternatives);
bool grants_only_attested(const EffectSet& combined, const std::vector<EffectSet>& sources);

}  // namespace khanary::kuhul::algebra
