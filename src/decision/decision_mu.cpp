#include "decision_mu.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace khanary::decision {

namespace {

struct Branch {
    std::string id;
    std::string epistemic_state = "UNKNOWN";
    bool        probability_known = false;
    double      probability = 0.0;
};

struct RawOutcome {
    std::optional<double> utility;
    std::optional<double> loss;
};

struct Action {
    std::string id;
    bool        allowed = true;
    std::vector<std::string> rejection_causes;
    std::unordered_map<std::string, RawOutcome> outcomes;
};

struct ScoredAction {
    const Action* action = nullptr;
    double        score = 0.0;
};

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static std::string upper_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

static std::string normalize_token(std::string value) {
    value = lower_copy(std::move(value));
    for (char& c : value) {
        if (c == '-' || c == ' ' || c == ':') c = '_';
    }
    return value;
}

static bool finite_number(double value) {
    return std::isfinite(value) != 0;
}

static std::optional<std::string> string_field(const util::JsonValue& object,
                                               std::initializer_list<const char*> keys) {
    if (!object.is_object()) return std::nullopt;
    for (const char* key : keys) {
        if (auto value = object.get_string(key)) return *value;
    }
    return std::nullopt;
}

static std::optional<double> number_field(const util::JsonValue& object,
                                          std::initializer_list<const char*> keys) {
    if (!object.is_object()) return std::nullopt;
    for (const char* key : keys) {
        if (auto value = object.get_number(key)) {
            if (finite_number(*value)) return *value;
        }
    }
    return std::nullopt;
}

static bool bool_field(const util::JsonValue& object,
                       std::initializer_list<const char*> keys,
                       bool fallback) {
    if (!object.is_object()) return fallback;
    for (const char* key : keys) {
        if (auto value = object.get_bool(key)) return *value;
    }
    return fallback;
}

static void append_string_array(const util::JsonValue* node, std::vector<std::string>& out) {
    if (!node || !node->is_array()) return;
    for (const auto& item : node->as_array()) {
        if (item && item->is_string() && !item->as_string().empty())
            out.push_back(item->as_string());
    }
}

static util::JsonArray strings_to_json(const std::vector<std::string>& values) {
    util::JsonArray out;
    out.reserve(values.size());
    for (const auto& value : values) out.push_back(util::json_string(value));
    return out;
}

static bool valid_epistemic_state(const std::string& state) {
    return state == "SUPPORTED" || state == "CONTRADICTED" ||
           state == "UNKNOWN" || state == "MISS";
}

static RawOutcome parse_outcome(const util::JsonValue& value) {
    RawOutcome out;
    if (value.is_number()) {
        const double number = value.as_number();
        if (finite_number(number)) out.utility = number;
        return out;
    }
    if (!value.is_object()) return out;
    out.utility = number_field(value, {"utility", "u", "payoff"});
    out.loss = number_field(value, {"loss", "cost", "regret"});
    return out;
}

static bool outcome_value(const RawOutcome& outcome,
                          const std::string& objective,
                          double& value) {
    if (objective == "maximize_utility") {
        if (outcome.utility) {
            value = *outcome.utility;
            return true;
        }
        if (outcome.loss) {
            value = -*outcome.loss;
            return true;
        }
        return false;
    }

    if (outcome.loss) {
        value = *outcome.loss;
        return true;
    }
    if (outcome.utility) {
        value = -*outcome.utility;
        return true;
    }
    return false;
}

static void add_unique(std::vector<std::string>& values, const std::string& value) {
    if (value.empty()) return;
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

static util::JsonPtr build_response(const std::string& status,
                                    const std::string& objective,
                                    const std::string& model,
                                    const std::vector<Branch>& branches,
                                    const std::vector<Action>& actions,
                                    const std::vector<ScoredAction>& scores,
                                    const std::string& selected_action,
                                    const std::vector<std::string>& causes,
                                    const std::vector<std::string>& required_evidence) {
    const bool action_selected = !selected_action.empty();

    std::vector<std::string> unresolved;
    std::vector<std::string> missing_probabilities;
    for (const auto& branch : branches) {
        if (branch.epistemic_state == "UNKNOWN" || branch.epistemic_state == "MISS")
            add_unique(unresolved, branch.id);
        if (branch.epistemic_state != "CONTRADICTED" && !branch.probability_known)
            add_unique(missing_probabilities, branch.id);
    }

    const bool observation_authority = unresolved.empty();
    const bool probabilities_specified = missing_probabilities.empty();
    const std::string epistemic_state = observation_authority ? "SUPPORTED" : "UNKNOWN";
    const std::string probability_state = probabilities_specified ? "SPECIFIED" : "UNKNOWN";

    util::JsonObject root;
    root["@kind"] = util::json_string("khanary.decision.result.v1");
    root["ok"] = util::json_bool(action_selected);
    root["status"] = util::json_string(status);
    root["decision_model"] = util::json_string(model);
    root["objective"] = util::json_string(objective);
    root["action"] = util::json_string(selected_action);
    root["action_selected"] = util::json_bool(action_selected);
    root["decision_collapse"] = util::json_bool(action_selected);
    root["fact_collapsed"] = util::json_bool(false);
    root["fact_commit"] = util::json_string("denied");
    root["xul_scope"] = util::json_string("action_only");
    root["epistemic_state"] = util::json_string(epistemic_state);
    root["observation_authority"] = util::json_bool(observation_authority);
    root["probabilities_specified"] = util::json_bool(probabilities_specified);
    root["probability_state"] = util::json_string(probability_state);
    root["causes"] = util::json_array(strings_to_json(causes));
    root["required_evidence"] = util::json_array(strings_to_json(required_evidence));
    root["unresolved_branches"] = util::json_array(strings_to_json(unresolved));

    util::JsonArray branch_rows;
    for (const auto& branch : branches) {
        util::JsonObject row;
        row["id"] = util::json_string(branch.id);
        row["epistemic_state"] = util::json_string(branch.epistemic_state);
        row["probability_known"] = util::json_bool(branch.probability_known);
        if (branch.probability_known)
            row["probability"] = util::json_number(branch.probability);
        row["active"] = util::json_bool(branch.epistemic_state != "CONTRADICTED");
        branch_rows.push_back(util::json_object(std::move(row)));
    }
    root["branches"] = util::json_array(std::move(branch_rows));

    util::JsonArray action_rows;
    for (const auto& action : actions) {
        util::JsonObject row;
        row["id"] = util::json_string(action.id);
        row["allowed"] = util::json_bool(action.allowed);
        row["eligible"] = util::json_bool(action.allowed && action.rejection_causes.empty());
        row["selected"] = util::json_bool(action.id == selected_action);
        row["causes"] = util::json_array(strings_to_json(action.rejection_causes));
        action_rows.push_back(util::json_object(std::move(row)));
    }
    root["actions"] = util::json_array(std::move(action_rows));

    util::JsonArray score_rows;
    for (const auto& scored : scores) {
        util::JsonObject row;
        row["action"] = util::json_string(scored.action->id);
        row["score"] = util::json_number(scored.score);
        row["basis"] = util::json_string(model);
        score_rows.push_back(util::json_object(std::move(row)));
    }
    root["scores"] = util::json_array(std::move(score_rows));
    return util::json_object(std::move(root));
}

static util::JsonPtr decision_error(const std::string& cause,
                                    const std::vector<std::string>& required_evidence = {}) {
    return build_response("decision_unknown", "", "", {}, {}, {}, "", {cause},
                          required_evidence);
}

} // namespace

util::JsonPtr evaluate_spec(const util::JsonValue& spec) {
    if (!spec.is_object()) return decision_error("decision_spec_not_object");

    std::vector<Branch> branches;
    std::unordered_set<std::string> branch_ids;
    const util::JsonValue* branch_node = spec.get("branches");
    if (!branch_node || !branch_node->is_array() || branch_node->as_array().empty())
        return decision_error("missing_branches", {"branches"});

    for (const auto& item : branch_node->as_array()) {
        if (!item) continue;
        Branch branch;
        if (item->is_string()) {
            branch.id = item->as_string();
        } else if (item->is_object()) {
            branch.id = string_field(*item, {"id", "branch", "name"}).value_or("");
            branch.epistemic_state = upper_copy(
                string_field(*item, {"epistemic_state", "state", "status"}).value_or("UNKNOWN"));
            if (auto probability = number_field(*item, {"probability", "p"})) {
                if (*probability < 0.0 || *probability > 1.0)
                    return decision_error("probability_out_of_range:" + branch.id,
                                          {"branch_probability:" + branch.id});
                branch.probability_known = true;
                branch.probability = *probability;
            }
        }
        if (branch.id.empty()) return decision_error("branch_missing_id", {"branches.id"});
        if (!valid_epistemic_state(branch.epistemic_state))
            return decision_error("invalid_epistemic_state:" + branch.id,
                                  {"branch_state:" + branch.id});
        if (!branch_ids.insert(branch.id).second)
            return decision_error("duplicate_branch:" + branch.id, {"branches.id"});
        branches.push_back(std::move(branch));
    }

    std::vector<std::string> forbidden_actions;
    append_string_array(spec.get("forbidden_actions"), forbidden_actions);
    if (const util::JsonValue* constraints = spec.get("constraints")) {
        if (constraints->is_object())
            append_string_array(constraints->get("forbidden_actions"), forbidden_actions);
        if (constraints->is_array()) {
            std::vector<std::string> constraint_rows;
            append_string_array(constraints, constraint_rows);
            for (const auto& row : constraint_rows) {
                const std::string normalized = lower_copy(row);
                if (normalized.rfind("forbid:", 0) == 0)
                    forbidden_actions.push_back(row.substr(7));
            }
        }
    }

    std::vector<Action> actions;
    const util::JsonValue* action_node = spec.get("actions");
    if (!action_node || !action_node->is_array() || action_node->as_array().empty())
        return decision_error("missing_actions", {"actions"});

    for (const auto& item : action_node->as_array()) {
        if (!item) continue;
        Action action;
        if (item->is_string()) {
            action.id = item->as_string();
        } else if (item->is_object()) {
            action.id = string_field(*item, {"id", "action", "name"}).value_or("");
            action.allowed = bool_field(*item, {"allowed"}, true);

            if (const util::JsonValue* outcomes = item->get("outcomes")) {
                if (outcomes->is_object()) {
                    for (const auto& [branch_id, outcome_value_node] : outcomes->as_object()) {
                        if (!outcome_value_node) continue;
                        action.outcomes[branch_id] = parse_outcome(*outcome_value_node);
                    }
                } else if (outcomes->is_array()) {
                    for (const auto& outcome_row : outcomes->as_array()) {
                        if (!outcome_row || !outcome_row->is_object()) continue;
                        const std::string branch_id =
                            string_field(*outcome_row, {"branch", "branch_id", "id"}).value_or("");
                        if (branch_id.empty()) continue;
                        action.outcomes[branch_id] = parse_outcome(*outcome_row);
                    }
                }
            }
        }
        if (action.id.empty()) return decision_error("action_missing_id", {"actions.id"});
        if (std::find(forbidden_actions.begin(), forbidden_actions.end(), action.id) !=
            forbidden_actions.end()) {
            action.allowed = false;
            action.rejection_causes.push_back("forbidden_by_constraint");
        }
        actions.push_back(std::move(action));
    }

    std::string objective = normalize_token(
        string_field(spec, {"objective", "goal"}).value_or(""));
    if (objective == "maximize" || objective == "max" || objective == "utility" ||
        objective == "max_utility")
        objective = "maximize_utility";
    if (objective == "minimize" || objective == "min" || objective == "loss" ||
        objective == "min_loss")
        objective = "minimize_loss";

    if (objective.empty()) {
        bool saw_utility = false;
        bool saw_loss = false;
        for (const auto& action : actions) {
            for (const auto& outcome_entry : action.outcomes) {
                const auto& outcome = outcome_entry.second;
                saw_utility = saw_utility || outcome.utility.has_value();
                saw_loss = saw_loss || outcome.loss.has_value();
            }
        }
        if (saw_utility) objective = "maximize_utility";
        else if (saw_loss) objective = "minimize_loss";
    }
    if (objective != "maximize_utility" && objective != "minimize_loss")
        return decision_error("missing_or_unknown_objective", {"objective"});

    std::vector<size_t> active_branches;
    std::vector<std::string> unresolved_branches;
    for (size_t i = 0; i < branches.size(); ++i) {
        if (branches[i].epistemic_state != "CONTRADICTED") active_branches.push_back(i);
        if (branches[i].epistemic_state == "UNKNOWN" || branches[i].epistemic_state == "MISS")
            unresolved_branches.push_back(branches[i].id);
    }
    if (active_branches.empty())
        return decision_error("no_active_branches", {"branch_state:*"});

    bool all_probabilities_known = true;
    for (size_t index : active_branches) {
        if (!branches[index].probability_known) {
            all_probabilities_known = false;
            break;
        }
    }

    std::string model = normalize_token(
        string_field(spec, {"decision_model", "model", "provider"}).value_or(""));
    const bool explicit_model = !model.empty();
    if (model == "eu" || model == "bayesian" || model == "bayesian_decision")
        model = "expected_utility";
    if (model == "robust")
        model = objective == "minimize_loss" ? "minimax" : "maximin";
    if (model == "prospect" || model == "prospect_theory" || model == "observed_policy")
        return decision_error("descriptive_model_not_authoritative:" + model,
                              {"normative_decision_model"});
    if (model.empty())
        model = all_probabilities_known ? "expected_utility"
                                        : (objective == "minimize_loss" ? "minimax" : "maximin");

    if (model != "expected_utility" && model != "maximin" && model != "minimax")
        return decision_error("unknown_decision_model:" + model, {"decision_model"});
    if (model == "maximin" && objective != "maximize_utility")
        return decision_error("model_objective_mismatch:maximin", {"objective:maximize_utility"});
    if (model == "minimax" && objective != "minimize_loss")
        return decision_error("model_objective_mismatch:minimax", {"objective:minimize_loss"});

    std::vector<std::string> causes;
    std::vector<std::string> required_evidence;
    for (const auto& branch_id : unresolved_branches) {
        add_unique(causes, "unknown_branch_state:" + branch_id);
        add_unique(required_evidence, "branch_state:" + branch_id);
    }
    if (!explicit_model && !all_probabilities_known)
        add_unique(causes, "robust_model_selected_under_unresolved_probability");

    std::vector<double> probabilities(active_branches.size(), 0.0);
    if (model == "expected_utility") {
        if (!all_probabilities_known) {
            for (size_t index : active_branches) {
                if (!branches[index].probability_known)
                    add_unique(required_evidence, "branch_probability:" + branches[index].id);
            }
            add_unique(causes, "probability_unresolved");
            return build_response("decision_unknown", objective, model, branches, actions, {},
                                  "", causes, required_evidence);
        }

        double sum = 0.0;
        for (size_t i = 0; i < active_branches.size(); ++i)
            sum += branches[active_branches[i]].probability;
        if (!(sum > 0.0))
            return decision_error("probability_mass_invalid", {"branch_probability:*"});
        if (std::fabs(sum - 1.0) > 1e-6)
            add_unique(causes, "probability_mass_normalized");
        for (size_t i = 0; i < active_branches.size(); ++i)
            probabilities[i] = branches[active_branches[i]].probability / sum;
    }

    std::vector<ScoredAction> scores;
    for (auto& action : actions) {
        if (!action.allowed) continue;

        std::vector<double> values;
        values.reserve(active_branches.size());
        bool eligible = true;
        for (size_t branch_index : active_branches) {
            const auto& branch = branches[branch_index];
            const auto outcome_it = action.outcomes.find(branch.id);
            if (outcome_it == action.outcomes.end()) {
                action.rejection_causes.push_back("missing_outcome:" + branch.id);
                add_unique(required_evidence, "outcome:" + action.id + ":" + branch.id);
                eligible = false;
                continue;
            }
            double value = 0.0;
            if (!outcome_value(outcome_it->second, objective, value)) {
                action.rejection_causes.push_back("missing_value:" + branch.id);
                add_unique(required_evidence, "outcome_value:" + action.id + ":" + branch.id);
                eligible = false;
                continue;
            }
            values.push_back(value);
        }
        if (!eligible || values.empty()) continue;

        double score = 0.0;
        if (model == "expected_utility") {
            for (size_t i = 0; i < values.size(); ++i) score += probabilities[i] * values[i];
        } else if (model == "maximin") {
            score = *std::min_element(values.begin(), values.end());
        } else {
            score = *std::max_element(values.begin(), values.end());
        }
        scores.push_back(ScoredAction{&action, score});
    }

    if (scores.empty()) {
        add_unique(causes, "no_eligible_actions");
        return build_response("decision_unknown", objective, model, branches, actions, {}, "",
                              causes, required_evidence);
    }

    const bool maximize = objective == "maximize_utility";
    std::sort(scores.begin(), scores.end(), [maximize](const ScoredAction& a,
                                                       const ScoredAction& b) {
        if (std::fabs(a.score - b.score) > 1e-12)
            return maximize ? a.score > b.score : a.score < b.score;
        return a.action->id < b.action->id;
    });

    return build_response("decision_selected", objective, model, branches, actions, scores,
                          scores.front().action->id, causes, required_evidence);
}

util::JsonPtr evaluate_request(const std::string& query,
                               const util::JsonValue* payload) {
    const util::JsonValue* spec = nullptr;
    if (payload && payload->is_object()) {
        if (const util::JsonValue* decision = payload->get("decision")) {
            if (decision->is_object()) spec = decision;
        }
        if (!spec && payload->get("branches") && payload->get("actions")) spec = payload;
    }

    util::JsonPtr parsed;
    if (!spec) {
        const size_t begin = query.find('{');
        if (begin != std::string::npos) {
            try {
                parsed = util::parse(query.substr(begin));
            } catch (const std::exception&) {
                return decision_error("invalid_decision_json", {"decision"});
            }
            if (parsed && parsed->is_object()) spec = parsed.get();
        }
    }

    if (!spec) return decision_error("missing_decision_spec", {"decision"});
    return evaluate_spec(*spec);
}

} // namespace khanary::decision
