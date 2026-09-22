#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <stdexcept>
#include <memory>
#include <optional>

namespace khanary::util {

struct JsonValue;
using JsonObject = std::unordered_map<std::string, std::shared_ptr<JsonValue>>;
using JsonArray  = std::vector<std::shared_ptr<JsonValue>>;

struct JsonValue {
    using Variant = std::variant<
        std::monostate,  // null
        bool,
        double,
        std::string,
        JsonArray,
        JsonObject
    >;
    Variant value;

    JsonValue() : value(std::monostate{}) {}
    explicit JsonValue(bool v)        : value(v) {}
    explicit JsonValue(double v)      : value(v) {}
    explicit JsonValue(std::string v) : value(std::move(v)) {}
    explicit JsonValue(JsonArray v)   : value(std::move(v)) {}
    explicit JsonValue(JsonObject v)  : value(std::move(v)) {}

    bool is_null()   const { return std::holds_alternative<std::monostate>(value); }
    bool is_bool()   const { return std::holds_alternative<bool>(value); }
    bool is_number() const { return std::holds_alternative<double>(value); }
    bool is_string() const { return std::holds_alternative<std::string>(value); }
    bool is_array()  const { return std::holds_alternative<JsonArray>(value); }
    bool is_object() const { return std::holds_alternative<JsonObject>(value); }

    bool        as_bool()   const { return std::get<bool>(value); }
    double      as_number() const { return std::get<double>(value); }
    int64_t     as_int()    const { return (int64_t)std::get<double>(value); }
    const std::string&   as_string() const { return std::get<std::string>(value); }
    const JsonArray&     as_array()  const { return std::get<JsonArray>(value); }
    const JsonObject&    as_object() const { return std::get<JsonObject>(value); }
    JsonArray&     mut_array()  { return std::get<JsonArray>(value); }
    JsonObject&    mut_object() { return std::get<JsonObject>(value); }

    const JsonValue* get(const std::string& key) const;
    std::optional<std::string> get_string(const std::string& key) const;
    std::optional<double>      get_number(const std::string& key) const;
    std::optional<bool>        get_bool(const std::string& key) const;
};

using JsonPtr = std::shared_ptr<JsonValue>;

JsonPtr json_null();
JsonPtr json_bool(bool v);
JsonPtr json_number(double v);
JsonPtr json_string(const std::string& v);
JsonPtr json_array(JsonArray v = {});
JsonPtr json_object(JsonObject v = {});

JsonPtr parse(const std::string& text);
JsonPtr parse(const char* text, size_t len);
std::string stringify(const JsonValue& v, bool pretty = false, int indent = 0);
std::string stringify(const JsonPtr& v, bool pretty = false, int indent = 0);

} // namespace khanary::util
