#include "schema.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <set>

namespace khanary::util {
namespace {
class Validator {
    std::filesystem::path root_;
    std::map<std::string, JsonPtr> documents_;
    static bool fail(std::string* error, const std::string& path, const std::string& reason) {
        if (error) *error = path + ": " + reason;
        return false;
    }
    const JsonValue& document(const std::string& name) {
        auto it = documents_.find(name);
        if (it != documents_.end()) return *it->second;
        // Schemas may reference only a sibling file; no network or traversal.
        if (name.empty() || name.find_first_of("/\\:") != std::string::npos || name == "..")
            throw std::runtime_error("schema: non-local reference");
        std::ifstream f(root_ / name, std::ios::binary);
        if (!f) throw std::runtime_error("schema: cannot open " + name);
        std::string bytes((std::istreambuf_iterator<char>(f)), {});
        auto parsed = parse(bytes);
        return *documents_.emplace(name, std::move(parsed)).first->second;
    }
    bool check(const JsonValue& v, const JsonValue& s, const std::string& file,
               const std::string& path, unsigned depth, std::string* error) {
        if (depth > 256) throw std::runtime_error("schema: recursion limit");
        if (s.is_bool()) return s.as_bool() || fail(error, path, "false schema");
        if (!s.is_object()) throw std::runtime_error("schema: expected object");
        static const std::set<std::string> supported = {
            "$schema", "$id", "$defs", "$ref", "title", "description", "default",
            "type", "const", "enum", "required", "properties", "additionalProperties",
            "propertyNames", "items", "minItems", "minLength", "minimum", "maximum", "anyOf"
        };
        for (const auto& entry : s.as_object())
            if (!supported.count(entry.first)) throw std::runtime_error("schema: unsupported keyword " + entry.first);
        if (auto ref = s.get_string("$ref")) {
            const auto hash = ref->find('#');
            const auto name = ref->substr(0, hash);
            const std::string target_file = name.empty() ? file : name;
            const JsonValue* target = &document(target_file);
            if (hash != std::string::npos && hash + 1 < ref->size()) {
                const auto fragment = ref->substr(hash + 1);
                if (fragment.rfind("/$defs/", 0) != 0) throw std::runtime_error("schema: unsupported fragment");
                target = target->get("$defs");
                if (target) target = target->get(fragment.substr(7));
                if (!target) throw std::runtime_error("schema: unresolved reference");
            }
            if (!check(v, *target, target_file, path, depth + 1, error)) return false;
        }
        if (const auto* choices = s.get("anyOf")) {
            bool accepted = false;
            for (const auto& choice : choices->as_array())
                if (check(v, *choice, file, path, depth + 1, nullptr)) accepted = true;
            if (!accepted) return fail(error, path, "no anyOf branch matched");
        }
        if (auto type = s.get_string("type")) {
            const bool ok = (*type == "object" && v.is_object()) || (*type == "array" && v.is_array()) ||
                (*type == "string" && v.is_string()) || (*type == "boolean" && v.is_bool()) ||
                (*type == "null" && v.is_null()) || (*type == "number" && v.is_number() && std::isfinite(v.as_number())) ||
                (*type == "integer" && v.is_number() && std::isfinite(v.as_number()) && std::floor(v.as_number()) == v.as_number());
            if (!ok) return fail(error, path, "expected " + *type);
        }
        if (const auto* c = s.get("const"))
            if (stringify(v) != stringify(*c)) return fail(error, path, "const mismatch");
        if (const auto* values = s.get("enum")) {
            const auto text = stringify(v);
            bool found = false;
            for (const auto& item : values->as_array()) if (text == stringify(item)) found = true;
            if (!found) return fail(error, path, "enum mismatch");
        }
        if (v.is_number()) {
            if (!std::isfinite(v.as_number())) return fail(error, path, "non-finite number");
            if (auto n = s.get_number("minimum")) if (v.as_number() < *n) return fail(error, path, "below minimum");
            if (auto n = s.get_number("maximum")) if (v.as_number() > *n) return fail(error, path, "above maximum");
        }
        if (v.is_string()) {
            size_t length = 0;
            for (unsigned char c : v.as_string()) if ((c & 0xc0) != 0x80) ++length;
            if (auto n = s.get_number("minLength")) if (static_cast<double>(length) < *n) return fail(error, path, "short string");
        }
        if (v.is_array()) {
            if (auto n = s.get_number("minItems"))
                if (static_cast<double>(v.as_array().size()) < *n) return fail(error, path, "short array");
            if (const auto* item_schema = s.get("items")) {
                for (size_t i = 0; i < v.as_array().size(); ++i)
                    if (!check(*v.as_array()[i], *item_schema, file, path + "/" + std::to_string(i), depth + 1, error)) return false;
            }
        }
        if (v.is_object()) {
            if (const auto* required = s.get("required"))
                for (const auto& key : required->as_array())
                    if (!v.get(key->as_string())) return fail(error, path, "missing " + key->as_string());
            const auto* properties = s.get("properties");
            std::vector<std::string> keys;
            for (const auto& entry : v.as_object()) keys.push_back(entry.first);
            std::sort(keys.begin(), keys.end());
            for (const auto& key : keys) {
                if (const auto* names = s.get("propertyNames"))
                    if (!check(*json_string(key), *names, file, path + "/" + key, depth + 1, error)) return false;
                const auto* rule = properties ? properties->get(key) : nullptr;
                if (!rule) rule = s.get("additionalProperties");
                if (rule && !check(*v.get(key), *rule, file, path + "/" + key, depth + 1, error)) return false;
            }
        }
        return true;
    }
public:
    explicit Validator(std::filesystem::path root) : root_(std::move(root)) {}
    bool validate(const JsonValue& v, const std::string& file, std::string* error) {
        if (error) error->clear();
        return check(v, document(file), file, "$", 0, error);
    }
};
}
bool validate_schema_file(const JsonValue& instance, const std::filesystem::path& schema, std::string* error) {
    return Validator(schema.parent_path()).validate(instance, schema.filename().string(), error);
}
}
