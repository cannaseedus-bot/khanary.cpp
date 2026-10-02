#pragma once

#include "../util/json.h"
#include <string>

namespace khanary::decision {

// Evaluate a typed decision problem without collapsing unresolved facts.
//
// Accepted locations for the decision spec, in order:
//   1. payload.decision
//   2. payload itself when it contains branches + actions
//   3. the first JSON object embedded in the query string
//
// The evaluator may select an action under UNKNOWN branch probabilities, but it
// never fabricates a prior and never marks the underlying facts collapsed.
util::JsonPtr evaluate_request(const std::string& query,
                               const util::JsonValue* payload);
util::JsonPtr evaluate_spec(const util::JsonValue& spec);

} // namespace khanary::decision
