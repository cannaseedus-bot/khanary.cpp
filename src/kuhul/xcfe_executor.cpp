#include "xcfe_executor.h"
#include <sstream>
#include <stdexcept>

namespace khanary::kuhul {

using namespace khanary::util;

XCFEExecutor::XCFEExecutor() {
    // Built-in: @op identity
    register_op("identity", [](const JsonPtr& args, const XCFEContext&) { return args; });
}

void XCFEExecutor::register_op(const std::string& name, OpFn fn) {
    ops_[name] = std::move(fn);
}

std::string XCFEExecutor::lookup(const std::string& path, const XCFEContext& ctx) {
    auto it = ctx.find(path);
    if (it != ctx.end()) return it->second;
    // Dotted path: try splitting on first dot
    auto dot = path.find('.');
    if (dot != std::string::npos) {
        // Flatten to "a.b" style — ctx key may have dots
        return "";
    }
    return "";
}

bool XCFEExecutor::eval_condition(const std::string& condition, const XCFEContext& ctx) {
    // Simple "lhs == rhs" or "lhs != rhs" patterns
    auto eval_side = [&](const std::string& s) -> std::string {
        // If wrapped in quotes → literal
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
            return s.substr(1, s.size() - 2);
        std::string v = lookup(s, ctx);
        return v.empty() ? s : v;
    };

    for (auto& op : std::vector<std::string>{"==", "!=", "<=", ">="}) {
        auto pos = condition.find(op);
        if (pos != std::string::npos) {
            std::string lhs = eval_side(condition.substr(0, pos));
            std::string rhs = eval_side(condition.substr(pos + op.size()));
            // Trim whitespace
            auto trim = [](std::string s) {
                size_t a = s.find_first_not_of(" \t");
                size_t b = s.find_last_not_of(" \t");
                return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            lhs = trim(lhs); rhs = trim(rhs);
            if (op == "==")  return lhs == rhs;
            if (op == "!=")  return lhs != rhs;
            if (op == "<=")  return lhs <= rhs;
            if (op == ">=")  return lhs >= rhs;
        }
    }
    // Plain boolean
    std::string v = eval_side(condition);
    return v == "true" || v == "1" || v == "yes";
}

JsonPtr XCFEExecutor::eval(const JsonPtr& node, const XCFEContext& ctx) {
    if (!node) return json_null();

    if (node->is_string()) {
        return eval_token(node->as_string(), ctx);
    }

    if (node->is_object()) {
        const auto& obj = node->as_object();

        // @if / @then / @else
        auto if_it = obj.find("@if");
        if (if_it != obj.end()) {
            bool cond = false;
            if (if_it->second->is_string())
                cond = eval_condition(if_it->second->as_string(), ctx);
            else if (if_it->second->is_bool())
                cond = if_it->second->as_bool();

            const char* branch = cond ? "@then" : "@else";
            auto b_it = obj.find(branch);
            if (b_it != obj.end()) return eval(b_it->second, ctx);
            return json_null();
        }

        // @op / @fn / @args
        auto op_it = obj.find("@op");
        if (op_it == obj.end()) op_it = obj.find("@fn");
        if (op_it != obj.end()) {
            std::string name = op_it->second->as_string();
            JsonPtr args = json_null();
            auto args_it = obj.find("@args");
            if (args_it != obj.end()) args = args_it->second;

            auto fn_it = ops_.find(name);
            if (fn_it != ops_.end())
                return fn_it->second(args, ctx);
            return json_null();
        }

        // @node:Name in object context
        auto node_it = obj.find("@node");
        if (node_it != obj.end() && node_it->second->is_string()) {
            return eval_token("@node:" + node_it->second->as_string(), ctx);
        }

        return node; // pass through unknown objects
    }

    if (node->is_array()) {
        JsonArray result;
        for (const auto& elem : node->as_array()) {
            auto r = eval(elem, ctx);
            if (r && !r->is_null()) result.push_back(r);
        }
        return json_array(std::move(result));
    }

    return node;
}

JsonPtr XCFEExecutor::eval_token(const std::string& token, const XCFEContext& ctx) {
    if (token.rfind("@node:", 0) == 0) {
        // Referenced by name — caller resolves via ManifestRegistry
        return json_string(token); // pass the ref through
    }
    if (token.rfind("@op:", 0) == 0) {
        std::string name = token.substr(4);
        auto it = ops_.find(name);
        if (it != ops_.end()) return it->second(json_null(), ctx);
        return json_null();
    }
    // Other @ tokens — pass through as operative strings
    return json_string(token);
}

bool XCFEExecutor::authorize_phase_transition(Phase from, Phase to, const XCFEContext& ctx) {
    if (PhaseEngine::is_horizontal(from, to)) return true; // always authorized
    // Default policy: allow all vertical folds unless explicitly forbidden
    // Subclasses / policy rules can override via registered @op handlers
    (void)ctx;
    return true;
}

} // namespace khanary::kuhul
