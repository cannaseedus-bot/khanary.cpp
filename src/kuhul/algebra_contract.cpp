#include "algebra_contract.h"

#include <cmath>
#include <cstdio>

namespace khanary::kuhul::algebra {

namespace {

using util::JsonValue;

const JsonValue* member(const JsonValue& object, const char* key) {
    if (!object.is_object()) return nullptr;
    return object.get(key);
}

std::string label_of(const JsonValue& entry) {
    if (const auto id = entry.get_string("operator")) return *id;
    if (const auto id = entry.get_string("id")) return *id;
    return "<unnamed>";
}

void push(std::vector<Violation>& out, const char* code, std::string detail) {
    out.push_back(Violation{code, std::move(detail)});
}

bool read_bool(const JsonValue& block, const char* key, const std::string& label,
               std::vector<Violation>& errors, bool* out) {
    const JsonValue* value = member(block, key);
    if (value == nullptr) {
        push(errors, "MISSING_FIELD", label + "." + key);
        return false;
    }
    if (!value->is_bool()) {
        push(errors, "INVALID_TYPE", label + "." + key + " must be boolean");
        return false;
    }
    *out = value->as_bool();
    return true;
}

bool read_law(const JsonValue& block, Law law, const std::string& label,
              std::vector<Violation>& errors, LawSet* laws) {
    const char* key = law_name(law);
    const JsonValue* value = member(block, key);
    if (value == nullptr) {
        push(errors, "MISSING_FIELD", label + "." + key);
        return false;
    }
    if (value->is_null()) {
        laws->set(law, LawState::NotApplicable);
        return true;
    }
    if (value->is_bool()) {
        laws->set(law, value->as_bool() ? LawState::Yes : LawState::No);
        return true;
    }
    push(errors, "UNKNOWN_LAW_STATE", label + "." + key + " must be true, false, or null");
    return false;
}

std::string number_text(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%g", value);
    return std::string(buffer);
}

std::string identity_text(const JsonValue& value) {
    if (value.is_string()) return value.as_string();
    if (value.is_number()) return number_text(value.as_number());
    if (value.is_bool()) return value.as_bool() ? "true" : "false";
    return util::stringify(value);
}

}  // namespace

// ── Violations ───────────────────────────────────────────────────────────────

bool has_violations(const std::vector<Violation>& violations) {
    return !violations.empty();
}

std::vector<std::string> violation_codes(const std::vector<Violation>& violations) {
    std::vector<std::string> codes;
    codes.reserve(violations.size());
    for (const Violation& violation : violations) codes.push_back(violation.code);
    return codes;
}

// ── Structure table ──────────────────────────────────────────────────────────

const StructureRule* StructureTable::find(Structure structure) const {
    for (const StructureRule& rule : rules) {
        if (rule.structure == structure) return &rule;
    }
    return nullptr;
}

bool StructureTable::has_carrier(const std::string& carrier) const {
    for (const CarrierSpec& spec : carriers) {
        if (spec.name == carrier) return true;
    }
    return false;
}

bool StructureTable::carrier_exact(const std::string& carrier) const {
    for (const CarrierSpec& spec : carriers) {
        if (spec.name == carrier) return spec.exact;
    }
    return false;  // an unknown carrier never gets the benefit of the doubt
}

namespace {

LawRequirement must_hold(Law law, LawState value) {
    return LawRequirement{law, LawRequirementKind::MustHold, value};
}

LawRequirement must_apply(Law law) {
    return LawRequirement{law, LawRequirementKind::MustBeApplicable, LawState::NotApplicable};
}

StructureRule rule_none() {
    StructureRule rule;
    rule.structure = Structure::None;
    rule.family = "atomic";
    rule.structural.emplace_back("closed", false);
    rule.product_image = ProductImage::None;
    return rule;
}

StructureRule rule_scalar_arithmetic() {
    StructureRule rule;
    rule.structure = Structure::ScalarArithmetic;
    rule.family = "atomic";
    rule.laws = {must_hold(Law::Associative, LawState::NotApplicable),
                 must_hold(Law::Commutative, LawState::NotApplicable),
                 must_hold(Law::Idempotent, LawState::NotApplicable),
                 must_hold(Law::Invertible, LawState::NotApplicable),
                 must_hold(Law::Distributive, LawState::NotApplicable),
                 must_hold(Law::Unital, LawState::NotApplicable),
                 must_hold(Law::Bilinear, LawState::NotApplicable),
                 must_hold(Law::Anticommutative, LawState::NotApplicable)};
    rule.structural.emplace_back("closed", false);
    rule.product_image = ProductImage::Scalar;
    return rule;
}

StructureRule rule_monoid() {
    StructureRule rule;
    rule.structure = Structure::Monoid;
    rule.family = "group_like";
    rule.laws = {must_hold(Law::Associative, LawState::Yes), must_hold(Law::Unital, LawState::Yes)};
    rule.structural = {{"closed", true}};
    rule.product_image = ProductImage::Carrier;
    return rule;
}

StructureRule rule_commutative_monoid() {
    StructureRule rule = rule_monoid();
    rule.structure = Structure::CommutativeMonoid;
    rule.laws.push_back(must_hold(Law::Commutative, LawState::Yes));
    return rule;
}

StructureRule rule_group() {
    StructureRule rule = rule_monoid();
    rule.structure = Structure::Group;
    rule.laws.push_back(must_hold(Law::Invertible, LawState::Yes));
    return rule;
}

StructureRule rule_abelian_group() {
    StructureRule rule = rule_group();
    rule.structure = Structure::AbelianGroup;
    rule.laws.push_back(must_hold(Law::Commutative, LawState::Yes));
    return rule;
}

StructureRule rule_semilattice(Structure structure, std::vector<std::string> operations) {
    StructureRule rule;
    rule.structure = structure;
    rule.family = "lattice_like";
    rule.laws = {must_hold(Law::Associative, LawState::Yes),
                 must_hold(Law::Commutative, LawState::Yes),
                 must_hold(Law::Idempotent, LawState::Yes),
                 must_hold(Law::Unital, LawState::Yes)};
    rule.structural = {{"closed", true}};
    rule.product_image = ProductImage::Carrier;
    rule.binary_operations = std::move(operations);
    return rule;
}

StructureRule rule_lattice() {
    StructureRule rule = rule_semilattice(Structure::Lattice, {"meet_join_pair"});
    return rule;
}

StructureRule rule_ring() {
    StructureRule rule;
    rule.structure = Structure::Ring;
    rule.family = "ring_like";
    rule.laws = {must_hold(Law::Associative, LawState::Yes),
                 must_hold(Law::Invertible, LawState::Yes),
                 must_hold(Law::Distributive, LawState::Yes),
                 must_hold(Law::Unital, LawState::Yes)};
    rule.structural = {{"closed", true}};
    rule.product_image = ProductImage::Carrier;
    rule.requires_scalar_field = true;
    return rule;
}

StructureRule rule_module() {
    StructureRule rule;
    rule.structure = Structure::Module;
    rule.family = "module_like";
    rule.laws = {must_hold(Law::Associative, LawState::Yes),
                 must_hold(Law::Invertible, LawState::Yes),
                 must_hold(Law::Distributive, LawState::Yes),
                 must_hold(Law::Unital, LawState::Yes)};
    rule.structural = {{"closed", false}};
    rule.product_image = ProductImage::Carrier;
    rule.requires_scalar_field = true;
    return rule;
}

StructureRule rule_algebra(Structure structure) {
    StructureRule rule;
    rule.structure = structure;
    rule.family = "algebra_like";
    rule.laws = {must_apply(Law::Associative), must_hold(Law::Bilinear, LawState::Yes)};
    rule.structural = {{"closed", true}};
    rule.product_image = ProductImage::Carrier;
    rule.requires_scalar_field = true;
    return rule;
}

StructureRule rule_unital_associative_algebra() {
    StructureRule rule = rule_algebra(Structure::UnitalAssociativeAlgebra);
    rule.laws = {must_hold(Law::Associative, LawState::Yes),
                 must_hold(Law::Bilinear, LawState::Yes),
                 must_hold(Law::Unital, LawState::Yes)};
    return rule;
}

StructureRule rule_inner_product_space() {
    StructureRule rule;
    rule.structure = Structure::InnerProductSpace;
    rule.family = "algebra_like";
    rule.laws = {must_hold(Law::Associative, LawState::NotApplicable),
                 must_hold(Law::Idempotent, LawState::NotApplicable),
                 must_apply(Law::Bilinear)};
    rule.structural = {{"closed", false}};
    rule.product_image = ProductImage::Scalar;
    rule.requires_scalar_field = true;
    return rule;
}

// A = K + V with v.w = 0. Associative, commutative, bilinear and unital, with
// 1_A = (1,0) taken entirely from K; V is an ideal whose square is zero, so no
// element (0,v) is invertible.
StructureRule rule_square_zero_extension() {
    StructureRule rule;
    rule.structure = Structure::SquareZeroExtension;
    rule.family = "algebra_like";
    rule.laws = {must_hold(Law::Associative, LawState::Yes),
                 must_hold(Law::Commutative, LawState::Yes),
                 must_hold(Law::Unital, LawState::Yes),
                 must_hold(Law::Bilinear, LawState::Yes),
                 must_hold(Law::Invertible, LawState::No)};
    rule.structural = {{"closed", true}};
    rule.product_image = ProductImage::Carrier;
    rule.requires_scalar_field = true;
    return rule;
}

// Cl(2,0): associative, bilinear, unital, NOT commutative (e1*e2 = e12 ≠ e2*e1 = -e12),
// and NOT universally invertible (per-value precondition, like singular matrices).
StructureRule rule_clifford_algebra() {
    StructureRule rule;
    rule.structure = Structure::CliffordAlgebra;
    rule.family = "algebra_like";
    rule.laws = {must_hold(Law::Associative, LawState::Yes),
                 must_hold(Law::Commutative, LawState::No),
                 must_hold(Law::Unital, LawState::Yes),
                 must_hold(Law::Bilinear, LawState::Yes),
                 must_hold(Law::Invertible, LawState::No)};
    rule.structural = {{"closed", true}};
    rule.product_image = ProductImage::Carrier;
    rule.requires_scalar_field = true;
    return rule;
}

}  // namespace

const StructureTable& builtin_structure_table() {
    static const StructureTable table = [] {
        StructureTable value;
        value.carriers = {
            {"real", false},          {"float32", false},     {"float64", false},
            {"integer", true},        {"fixed_point", true},  {"effect_set", true},
            {"gram_event_list", true},{"boolean", true},      {"record", true},
            {"vector", false},        {"vector3", false},     {"square_matrix_n", false},
            {"micronaut_state", false},{"state", false},
            {"cl20_multivector", false},
        };
        value.rules = {
            rule_none(),
            rule_scalar_arithmetic(),
            rule_monoid(),
            rule_commutative_monoid(),
            rule_group(),
            rule_abelian_group(),
            rule_semilattice(Structure::SemilatticeMeet, {"intersect", "min", "and", "gcd", "meet"}),
            rule_semilattice(Structure::SemilatticeJoin, {"union", "max", "or", "lcm", "join"}),
            rule_lattice(),
            rule_ring(),
            rule_module(),
            rule_algebra(Structure::Algebra),
            rule_algebra(Structure::AssociativeAlgebra),
            rule_algebra(Structure::NonassociativeAlgebra),
            rule_unital_associative_algebra(),
            rule_inner_product_space(),
            rule_square_zero_extension(),
            rule_clifford_algebra(),
        };
        // Structure-specific overrides of the generic algebra rule.
        for (StructureRule& rule : value.rules) {
            if (rule.structure == Structure::AssociativeAlgebra) {
                rule.laws = {must_hold(Law::Associative, LawState::Yes),
                             must_hold(Law::Bilinear, LawState::Yes)};
            } else if (rule.structure == Structure::NonassociativeAlgebra) {
                rule.laws = {must_hold(Law::Associative, LawState::No),
                             must_hold(Law::Bilinear, LawState::Yes)};
            }
        }
        return value;
    }();
    return table;
}

namespace {

std::string canonical_rule_text(const StructureRule& rule) {
    // The registry is JSON, so its key order is not authoritative. Canonicalise
    // before comparing, or a reordered table would look like a drift.
    std::vector<LawRequirement> laws = rule.laws;
    std::sort(laws.begin(), laws.end(), [](const LawRequirement& a, const LawRequirement& b) {
        return static_cast<int>(a.law) < static_cast<int>(b.law);
    });
    std::vector<std::pair<std::string, bool>> structural = rule.structural;
    std::sort(structural.begin(), structural.end(),
              [](const std::pair<std::string, bool>& a, const std::pair<std::string, bool>& b) {
                  return a.first < b.first;
              });

    std::string text = std::string(structure_name(rule.structure)) + "|" + rule.family + "|";
    for (const LawRequirement& requirement : laws) {
        text += law_name(requirement.law);
        text += ':';
        text += requirement.kind == LawRequirementKind::MustBeApplicable
                    ? "*"
                    : law_state_token(requirement.value);
        text += ',';
    }
    text += '|';
    for (const auto& [name, value] : structural) {
        text += name;
        text += '=';
        text += value ? "true" : "false";
        text += ',';
    }
    text += '|';
    text += rule.product_image ? product_image_name(*rule.product_image) : "-";
    text += '|';
    std::vector<std::string> operations = rule.binary_operations;
    std::sort(operations.begin(), operations.end());
    for (const std::string& operation : operations) {
        text += operation;
        text += ',';
    }
    text += '|';
    text += rule.requires_scalar_field ? "scalar_field" : "-";
    return text;
}

}  // namespace

bool same_structure_rule(const StructureRule& a, const StructureRule& b) {
    return canonical_rule_text(a) == canonical_rule_text(b);
}

bool same_structure_table(const StructureTable& a, const StructureTable& b, std::string* why) {
    for (const CarrierSpec& spec : a.carriers) {
        bool found = false;
        for (const CarrierSpec& other : b.carriers) {
            if (other.name != spec.name) continue;
            found = true;
            if (other.exact != spec.exact) {
                if (why) *why = "carrier exactness differs for '" + spec.name + "'";
                return false;
            }
        }
        if (!found) {
            if (why) *why = "carrier '" + spec.name + "' missing from the parsed table";
            return false;
        }
    }
    for (const CarrierSpec& spec : b.carriers) {
        if (a.carriers.end() == std::find_if(a.carriers.begin(), a.carriers.end(),
                                             [&](const CarrierSpec& other) {
                                                 return other.name == spec.name;
                                             })) {
            if (why) *why = "parsed table declares unknown carrier '" + spec.name + "'";
            return false;
        }
    }
    for (const StructureRule& rule : a.rules) {
        const StructureRule* other = b.find(rule.structure);
        if (other == nullptr) {
            if (why) *why = std::string("structure '") + structure_name(rule.structure) +
                            "' missing from the parsed table";
            return false;
        }
        if (!same_structure_rule(rule, *other)) {
            if (why) {
                *why = std::string("structure '") + structure_name(rule.structure) +
                       "' differs: builtin=" + canonical_rule_text(rule) +
                       " parsed=" + canonical_rule_text(*other);
            }
            return false;
        }
    }
    for (const StructureRule& rule : b.rules) {
        if (a.find(rule.structure) == nullptr) {
            if (why) *why = std::string("parsed table declares unknown structure '") +
                            structure_name(rule.structure) + "'";
            return false;
        }
    }
    return true;
}

// ── Parsing the structure table ──────────────────────────────────────────────

std::optional<StructureTable> parse_structure_table(const JsonValue& registry,
                                                    std::vector<Violation>& errors) {
    if (!registry.is_object()) {
        push(errors, "INVALID_TYPE", "registry must be a JSON object");
        return std::nullopt;
    }
    StructureTable table;

    if (const JsonValue* vocabulary = member(registry, "vocabulary");
        vocabulary != nullptr && vocabulary->is_object()) {
        if (const JsonValue* carriers = member(*vocabulary, "carriers");
            carriers != nullptr && carriers->is_object()) {
            for (const auto& [name, spec] : carriers->as_object()) {
                if (spec == nullptr || !spec->is_object()) {
                    push(errors, "INVALID_TYPE", "carrier '" + name + "' must be an object");
                    continue;
                }
                const JsonValue* exact = member(*spec, "exact");
                if (exact == nullptr || !exact->is_bool()) {
                    push(errors, "INVALID_TYPE", "carrier '" + name + ".exact must be boolean");
                    continue;
                }
                table.carriers.push_back(CarrierSpec{name, exact->as_bool()});
            }
        }
    }

    const JsonValue* structures = member(registry, "structures");
    if (structures == nullptr || !structures->is_array()) {
        push(errors, "INVALID_TYPE", "registry.structures must be an array");
        return std::nullopt;
    }

    for (const auto& entry : structures->as_array()) {
        if (entry == nullptr || !entry->is_object()) {
            push(errors, "INVALID_TYPE", "structure entry must be an object");
            continue;
        }
        const auto id = entry->get_string("id");
        if (!id) {
            push(errors, "STRUCTURE_ID_MISSING", "structure entry without id");
            continue;
        }
        const auto structure = structure_from_name(*id);
        if (!structure) {
            push(errors, "UNKNOWN_STRUCTURE", "unknown structure '" + *id + "'");
            continue;
        }
        StructureRule rule;
        rule.structure = *structure;
        if (const auto family = entry->get_string("family")) rule.family = *family;
        if (const auto description = entry->get_string("description")) rule.description = *description;
        if (const JsonValue* laws = member(*entry, "laws"); laws != nullptr) {
            if (!laws->is_object()) {
                push(errors, "STRUCTURE_LAWS_INVALID", *id + ".laws must be an object");
            } else {
                for (const auto& [name, requirement] : laws->as_object()) {
                    const auto law = law_from_name(name);
                    if (!law) {
                        push(errors, "STRUCTURE_LAW_UNKNOWN", *id + "." + name);
                        continue;
                    }
                    if (requirement == nullptr) continue;
                    if (requirement->is_string() && requirement->as_string() == "*") {
                        rule.laws.push_back(must_apply(*law));
                    } else if (requirement->is_null()) {
                        rule.laws.push_back(must_hold(*law, LawState::NotApplicable));
                    } else if (requirement->is_bool()) {
                        rule.laws.push_back(
                            must_hold(*law, requirement->as_bool() ? LawState::Yes : LawState::No));
                    } else {
                        push(errors, "STRUCTURE_LAW_REQUIREMENT_INVALID", *id + "." + name);
                    }
                }
            }
        }
        if (const JsonValue* structural = member(*entry, "structural");
            structural != nullptr && structural->is_object()) {
            for (const auto& [name, requirement] : structural->as_object()) {
                if (requirement == nullptr || !requirement->is_bool()) {
                    push(errors, "STRUCTURE_STRUCTURAL_REQUIREMENT_INVALID", *id + "." + name);
                    continue;
                }
                rule.structural.emplace_back(name, requirement->as_bool());
            }
        }
        if (const auto image = entry->get_string("product_image")) {
            const auto parsed = product_image_from_name(*image);
            if (!parsed) {
                push(errors, "STRUCTURE_PRODUCT_IMAGE_INVALID", *id + ".product_image");
            } else {
                rule.product_image = *parsed;
            }
        }
        if (const JsonValue* operations = member(*entry, "binary_operations");
            operations != nullptr && operations->is_array()) {
            for (const auto& operation : operations->as_array()) {
                if (operation != nullptr && operation->is_string()) {
                    rule.binary_operations.push_back(operation->as_string());
                }
            }
        }
        if (const JsonValue* scalar_field = member(*entry, "scalar_field");
            scalar_field != nullptr) {
            if (!scalar_field->is_bool()) {
                push(errors, "STRUCTURE_SCALAR_FIELD_INVALID", *id + ".scalar_field");
            } else {
                rule.requires_scalar_field = scalar_field->as_bool();
            }
        }
        table.rules.push_back(std::move(rule));
    }

    // The axiom table must be complete: a missing structure would silently
    // un-bind every contract that names it.
    for (std::size_t i = 0; i < kStructureCount; ++i) {
        const auto structure = static_cast<Structure>(i);
        if (table.find(structure) == nullptr) {
            push(errors, "STRUCTURE_MISSING", structure_name(structure));
        }
    }
    return table;
}

// ── Parsing a contract ───────────────────────────────────────────────────────

ParsedContract parse_contract(const JsonValue& entry) {
    ParsedContract parsed;
    const std::string label = label_of(entry);
    parsed.contract.id = label;

    if (!entry.is_object()) {
        push(parsed.errors, "INVALID_TYPE", label + " must be a JSON object");
        return parsed;
    }
    if (const auto kind = entry.get_string("kind")) parsed.contract.is_reducer = (*kind == "reducer");
    if (const auto description = entry.get_string("description")) {
        parsed.contract.description = *description;
    }

    const JsonValue* block = member(entry, "algebra");
    if (block == nullptr) {
        push(parsed.errors, "MISSING_ALGEBRA_BLOCK", label);
        return parsed;
    }
    if (!block->is_object()) {
        push(parsed.errors, "INVALID_TYPE", label + ".algebra must be an object");
        return parsed;
    }

    if (const auto structure = block->get_string("structure")) {
        const auto parsed_structure = structure_from_name(*structure);
        if (!parsed_structure) {
            push(parsed.errors, "UNKNOWN_STRUCTURE", label + "." + *structure);
        } else {
            parsed.contract.structure = *parsed_structure;
        }
    } else {
        push(parsed.errors, "MISSING_FIELD", label + ".structure");
    }

    if (const auto carrier = block->get_string("carrier")) {
        parsed.contract.carrier = *carrier;
    } else {
        push(parsed.errors, "MISSING_FIELD", label + ".carrier");
    }

    if (const auto operation = block->get_string("binary_operation")) {
        parsed.contract.binary_operation = *operation;
    } else {
        push(parsed.errors, "MISSING_FIELD", label + ".binary_operation");
    }

    if (const auto image = block->get_string("product_image")) {
        const auto parsed_image = product_image_from_name(*image);
        if (!parsed_image) {
            push(parsed.errors, "UNKNOWN_PRODUCT_IMAGE", label + "." + *image);
        } else {
            parsed.contract.product_image = *parsed_image;
        }
    } else {
        push(parsed.errors, "MISSING_FIELD", label + ".product_image");
    }

    if (const auto policy = block->get_string("reproducibility")) {
        const auto parsed_policy = reproducibility_from_name(*policy);
        if (!parsed_policy) {
            push(parsed.errors, "UNKNOWN_REPRODUCIBILITY", label + "." + *policy);
        } else {
            parsed.contract.reproducibility = *parsed_policy;
        }
    } else {
        push(parsed.errors, "MISSING_FIELD", label + ".reproducibility");
    }

    if (const auto policy = block->get_string("phase_policy")) {
        const auto parsed_policy = phase_policy_from_name(*policy);
        if (!parsed_policy) {
            push(parsed.errors, "UNKNOWN_PHASE_POLICY", label + "." + *policy);
        } else {
            parsed.contract.phase_policy = *parsed_policy;
        }
    } else {
        push(parsed.errors, "MISSING_FIELD", label + ".phase_policy");
    }

    if (const auto field = block->get_string("scalar_field")) {
        parsed.contract.scalar_field = *field;
    } else if (const JsonValue* raw = member(*block, "scalar_field"); raw == nullptr) {
        push(parsed.errors, "MISSING_FIELD", label + ".scalar_field");
    } else if (!raw->is_null()) {
        push(parsed.errors, "INVALID_TYPE", label + ".scalar_field must be a string or null");
    }

    read_bool(*block, "closed", label, parsed.errors, &parsed.contract.closed);
    read_bool(*block, "pure", label, parsed.errors, &parsed.contract.pure);
    read_bool(*block, "deterministic", label, parsed.errors, &parsed.contract.deterministic);
    read_bool(*block, "order_sensitive", label, parsed.errors, &parsed.contract.order_sensitive);

    for (std::size_t i = 0; i < kLawCount; ++i) {
        read_law(*block, static_cast<Law>(i), label, parsed.errors, &parsed.contract.laws);
    }

    // `identity` names the element behind Law::Unital. Its relationship with
    // the law is a semantic check, so both halves of a contradiction are named.
    if (const JsonValue* identity = member(*block, "identity")) {
        if (!identity->is_null()) parsed.contract.identity = identity_text(*identity);
    }

    if (const JsonValue* grouping = member(*block, "canonical_grouping")) {
        if (!grouping->is_object()) {
            push(parsed.errors, "INVALID_TYPE", label + ".canonical_grouping must be an object");
        } else {
            GroupingSpec spec;
            const auto kind = grouping->get_string("kind");
            const auto parsed_kind = kind ? grouping_from_name(*kind) : std::nullopt;
            if (!parsed_kind) {
                push(parsed.errors, "UNKNOWN_GROUPING", label + ".canonical_grouping.kind");
            } else {
                spec.kind = *parsed_kind;
            }
            const auto chunk = grouping->get_number("chunk");
            if (!chunk || *chunk < 1.0 || std::floor(*chunk) != *chunk) {
                push(parsed.errors, "INVALID_TYPE", label + ".canonical_grouping.chunk");
            } else {
                spec.chunk = static_cast<std::size_t>(*chunk);
            }
            parsed.contract.canonical_grouping = spec;
        }
    }

    if (const auto note = block->get_string("note")) parsed.contract.note = *note;
    return parsed;
}

// ── Semantic validation ──────────────────────────────────────────────────────

std::vector<Violation> validate(const AlgebraContract& contract, const StructureTable& table) {
    std::vector<Violation> violations;
    const std::string& label = contract.id;
    const LawSet& laws = contract.laws;

    // Carrier exactness decides whether regrouping preserves the value, so an
    // unrecognised carrier is a contract error rather than a default.
    if (!table.carriers.empty() && !table.has_carrier(contract.carrier)) {
        push(violations, "UNKNOWN_CARRIER", label + "." + contract.carrier);
    }

    const bool commutative = laws.is(Law::Commutative, LawState::Yes);

    // A5 / A3: commutativity makes operand order irrelevant.
    if (commutative && contract.order_sensitive) {
        push(violations, "COMMUTATIVE_IMPLIES_ORDER_FREE", label);
    }

    // A14: a product that closes on the carrier makes associativity applicable.
    if (contract.product_image == ProductImage::Carrier &&
        laws.is(Law::Associative, LawState::NotApplicable)) {
        push(violations, "CARRIER_PRODUCT_NEEDS_ASSOCIATIVITY", label);
    }

    // A13: an inner product leaves the carrier and is never an algebra product.
    if (contract.structure == Structure::InnerProductSpace &&
        contract.product_image != ProductImage::Scalar) {
        push(violations, "INNER_PRODUCT_MUST_NOT_BE_ALGEBRA", label);
    }

    // A16: a nonassociative algebra declares the law false, not null.
    if (contract.structure == Structure::NonassociativeAlgebra &&
        !laws.is(Law::Associative, LawState::No)) {
        push(violations, "NONASSOCIATIVE_MUST_DECLARE_FALSE", label);
    }

    // A19: unital and the named identity are one fact, declared twice. They must
    // agree, and unital == false means no element exists to name.
    const bool unital = laws.is(Law::Unital, LawState::Yes);
    if (unital && !contract.identity.has_value()) {
        push(violations, "IDENTITY_MISSING", label);
    }
    if (!unital && contract.identity.has_value()) {
        push(violations, "IDENTITY_UNEXPECTED", label);
    }

    // A20: a no-op inside a phase is not a phase transition. An operator that is
    // pure, deterministic and idempotent is a fixed point, and a fixed point may
    // not also be the thing that moves the phase — that is the silent bypass.
    if (contract.phase_policy == PhasePolicy::Advance) {
        const bool idempotent = laws.is(Law::Idempotent, LawState::Yes);
        if (idempotent && contract.pure && contract.deterministic) {
            push(violations, "IDENTITY_OPERATOR_MUST_PRESERVE_PHASE", label);
        }
    }

    // Binding law table of the declared structure.
    if (const StructureRule* rule = table.find(contract.structure)) {
        for (const LawRequirement& requirement : rule->laws) {
            const LawState declared = laws.get(requirement.law);
            const std::string detail = label + "." + law_name(requirement.law) + " for structure '" +
                                       structure_name(contract.structure) + "'";
            if (requirement.kind == LawRequirementKind::MustBeApplicable) {
                if (declared == LawState::NotApplicable) {
                    push(violations, "LAW_APPLICABILITY_REQUIRED", detail);
                }
                continue;
            }
            if (requirement.value == LawState::NotApplicable) {
                if (declared != LawState::NotApplicable) {
                    push(violations, "LAW_NOT_APPLICABLE", detail);
                }
            } else if (declared == LawState::NotApplicable) {
                push(violations, "LAW_DECLARED_NOT_APPLICABLE", detail);
            } else if (declared != requirement.value) {
                push(violations, requirement.value == LawState::Yes ? "LAW_REQUIRED_TRUE"
                                                                   : "LAW_REQUIRED_FALSE",
                     detail);
            }
        }
        for (const auto& [name, required] : rule->structural) {
            bool found = false;
            bool actual = false;
            if (name == "closed") {
                found = true;
                actual = contract.closed;
            }
            if (!found) {
                push(violations, "STRUCTURE_STRUCTURAL_UNKNOWN", label + "." + name);
            } else if (actual != required) {
                push(violations, "STRUCTURAL_REQUIRED",
                     label + "." + name + " must be " + (required ? "true" : "false") +
                         " for structure '" + structure_name(contract.structure) + "'");
            }
        }
        if (rule->product_image && contract.product_image != *rule->product_image) {
            push(violations, "PRODUCT_IMAGE_REQUIRED",
                 label + ".product_image must be '" +
                     product_image_name(*rule->product_image) + "' for structure '" +
                     structure_name(contract.structure) + "'");
        }
        if (rule->requires_scalar_field && !contract.scalar_field.has_value()) {
            push(violations, "SCALAR_FIELD_REQUIRED",
                 label + ".scalar_field is required by structure '" +
                     structure_name(contract.structure) + "'");
        }
        if (!rule->binary_operations.empty() &&
            std::find(rule->binary_operations.begin(), rule->binary_operations.end(),
                      contract.binary_operation) == rule->binary_operations.end()) {
            push(violations, "BINARY_OPERATION_NOT_ALLOWED",
                 label + ".binary_operation '" + contract.binary_operation + "' for structure '" +
                     structure_name(contract.structure) + "'");
        }
    }

    // A7-A9, A17: a declared execution permission must be derivable.
    const ReductionPolicy policy = derive_policy(contract, table);
    if (contract.canonical_grouping.has_value() && !policy.regroup) {
        push(violations, "REGROUP_NOT_LICENSED", label);
    }

    return violations;
}

std::vector<Violation> validate_entry(const JsonValue& entry, const StructureTable& table) {
    ParsedContract parsed = parse_contract(entry);
    std::vector<Violation> violations = std::move(parsed.errors);
    for (Violation& violation : validate(parsed.contract, table)) {
        violations.push_back(std::move(violation));
    }
    return violations;
}

// ── Derived permissions ──────────────────────────────────────────────────────

ReductionPolicy derive_policy(const AlgebraContract& contract, const StructureTable& table) {
    ReductionPolicy policy;
    const bool associative = contract.laws.is(Law::Associative, LawState::Yes);
    const bool commutative = contract.laws.is(Law::Commutative, LawState::Yes);
    const bool idempotent = contract.laws.is(Law::Idempotent, LawState::Yes);
    const bool invertible = contract.laws.is(Law::Invertible, LawState::Yes);
    const bool unital = contract.laws.is(Law::Unital, LawState::Yes);

    policy.commutative = commutative;
    policy.unital = unital;
    policy.carrier_exact = table.carrier_exact(contract.carrier);
    const bool exact_operation = contract.reproducibility == Reproducibility::Exact;
    const bool exact_reproducible =
        contract.reproducibility == Reproducibility::Exact ||
        contract.reproducibility == Reproducibility::DeterministicGrouping;

    if (!associative) policy.reasons.emplace_back("not_associative");
    if (!contract.pure) policy.reasons.emplace_back("impure");
    if (!contract.deterministic) policy.reasons.emplace_back("nondeterministic");
    if (associative && !policy.carrier_exact && !exact_reproducible) {
        policy.reasons.emplace_back("carrier_not_grouping_exact");
    }

    policy.mathematically_associative = associative;
    policy.executable_associative = associative && contract.pure && contract.deterministic &&
                                    (policy.carrier_exact || exact_reproducible);
    policy.regroup = policy.executable_associative;
    policy.parallel = policy.regroup;

    // Permuting operands is only value-identical when grouping is order-safe.
    const bool order_safe = policy.carrier_exact || exact_operation;
    policy.reorder = policy.regroup && commutative && order_safe;
    if (policy.regroup && commutative && !order_safe) {
        policy.reasons.emplace_back("reorder_requires_order_safe_carrier");
    }

    policy.repeat = idempotent && contract.pure && contract.deterministic;
    policy.invert = invertible && contract.pure && contract.deterministic;
    // A8: the unit is the seed. Without it, an empty fold has no defined value
    // and the runtime must not invent one.
    policy.empty_fold = unital && contract.pure;
    if (!unital) policy.reasons.emplace_back("no_unit_for_empty_fold");

    if (policy.reorder) {
        policy.grouping = Grouping::Free;
    } else if (policy.regroup) {
        policy.grouping = Grouping::Balanced;
    } else {
        policy.grouping = Grouping::Sequential;
    }
    return policy;
}

RewriteSet rewrites_for(const AlgebraContract& contract, const StructureTable& table) {
    const ReductionPolicy policy = derive_policy(contract, table);
    RewriteSet rewrites;
    rewrites.reasons = policy.reasons;

    rewrites.regroup = policy.regroup;
    rewrites.reorder = policy.reorder;
    rewrites.repeat = policy.repeat;
    rewrites.invert = policy.invert;

    // A22: anticommutativity licenses a swap only together with a sign flip.
    // It is not commutativity: a.b is not b.a, it is -(b.a).
    const bool anticommutative = contract.laws.is(Law::Anticommutative, LawState::Yes);
    rewrites.swap_with_sign = anticommutative && contract.pure && contract.deterministic;
    if (anticommutative && !rewrites.swap_with_sign) {
        rewrites.reasons.emplace_back("swap_needs_purity");
    }

    // A21: bilinearity licenses distribution over addition. It does not license
    // regrouping (that is associativity), and it does not survive normalization
    // — which is why COSINE_SIMILARITY declares bilinear = false while
    // DOT_PRODUCT declares bilinear = true.
    const bool bilinear = contract.laws.is(Law::Bilinear, LawState::Yes);
    rewrites.distribute = bilinear && contract.pure && contract.deterministic;
    if (bilinear && !rewrites.distribute) {
        rewrites.reasons.emplace_back("distribution_needs_purity");
    }

    // A19: the unit may be inserted or elided only when it exists.
    rewrites.elide_unit = policy.unital && contract.pure && contract.deterministic;
    if (policy.unital && !rewrites.elide_unit) {
        rewrites.reasons.emplace_back("unit_elision_needs_purity");
    }

    return rewrites;
}

PhaseDecision check_phase_transition(const AlgebraContract& contract, bool same_phase) {
    PhaseDecision decision;
    if (same_phase) {
        decision.allowed = true;
        decision.code = "phase_preserved";
        decision.detail = contract.id + " runs inside one phase";
        return decision;
    }
    switch (contract.phase_policy) {
        case PhasePolicy::Advance:
            decision.allowed = true;
            decision.code = "phase_advance_declared";
            decision.detail = contract.id +
                              " may request a transition; XCFE admission is still required";
            return decision;
        case PhasePolicy::Preserve:
            decision.allowed = false;
            decision.code = "identity_cannot_advance_phase";
            decision.detail = contract.id +
                              " preserves the phase and may not be used to cross a boundary";
            return decision;
        case PhasePolicy::None:
            decision.allowed = false;
            decision.code = "no_phase_authority";
            decision.detail = contract.id + " declares no phase authority";
            return decision;
    }
    decision.allowed = false;
    decision.code = "no_phase_authority";
    decision.detail = contract.id;
    return decision;
}

FoldPlan plan_fold(const AlgebraContract& contract, std::size_t element_count,
                   const StructureTable& table) {
    const ReductionPolicy policy = derive_policy(contract, table);
    FoldPlan plan;
    plan.reasons = policy.reasons;

    if (element_count <= 1 || !policy.regroup) {
        plan.grouping = Grouping::Sequential;
        plan.chunk = 1;
        if (element_count <= 1) plan.reasons.emplace_back("too_small_to_regroup");
        return plan;
    }

    if (contract.canonical_grouping.has_value()) {
        plan.grouping = contract.canonical_grouping->kind;
        plan.chunk = contract.canonical_grouping->chunk;
        plan.reasons.emplace_back("canonical_grouping");
    } else {
        plan.grouping = policy.grouping;
        plan.chunk = 64;
        plan.reasons.emplace_back("derived_grouping");
    }
    if (plan.chunk == 0) plan.chunk = 1;
    return plan;
}

// ── The effect lattice ───────────────────────────────────────────────────────

EffectSet effect_meet(const EffectSet& a, const EffectSet& b) {
    EffectSet result;
    for (const std::string& effect : a) {
        if (b.count(effect) != 0) result.insert(effect);
    }
    return result;
}

EffectSet effect_join(const EffectSet& a, const EffectSet& b) {
    EffectSet result = a;
    result.insert(b.begin(), b.end());
    return result;
}

EffectSet effective_permissions(const std::vector<EffectSet>& authorities) {
    if (authorities.empty()) return EffectSet{};
    EffectSet result = authorities.front();
    for (std::size_t i = 1; i < authorities.size(); ++i) result = effect_meet(result, authorities[i]);
    return result;
}

EffectSet joined_permissions(const std::vector<EffectSet>& alternatives) {
    EffectSet result;
    for (const EffectSet& alternative : alternatives) result = effect_join(result, alternative);
    return result;
}

bool grants_only_attested(const EffectSet& combined, const std::vector<EffectSet>& sources) {
    for (const std::string& effect : combined) {
        bool attested = false;
        for (const EffectSet& source : sources) {
            if (source.count(effect) != 0) {
                attested = true;
                break;
            }
        }
        if (!attested) return false;
    }
    return true;
}

}  // namespace khanary::kuhul::algebra
