#include "json.h"
#include <sstream>
#include <cctype>
#include <cstdio>
#include <stdexcept>

namespace khanary::util {

const JsonValue* JsonValue::get(const std::string& key) const {
    if (!is_object()) return nullptr;
    auto it = as_object().find(key);
    if (it == as_object().end()) return nullptr;
    return it->second.get();
}

std::optional<std::string> JsonValue::get_string(const std::string& key) const {
    auto* v = get(key);
    if (v && v->is_string()) return v->as_string();
    return std::nullopt;
}

std::optional<double> JsonValue::get_number(const std::string& key) const {
    auto* v = get(key);
    if (v && v->is_number()) return v->as_number();
    return std::nullopt;
}

std::optional<bool> JsonValue::get_bool(const std::string& key) const {
    auto* v = get(key);
    if (v && v->is_bool()) return v->as_bool();
    return std::nullopt;
}

JsonPtr json_null()                    { return std::make_shared<JsonValue>(); }
JsonPtr json_bool(bool v)              { return std::make_shared<JsonValue>(v); }
JsonPtr json_number(double v)          { return std::make_shared<JsonValue>(v); }
JsonPtr json_string(const std::string& v) { return std::make_shared<JsonValue>(v); }
JsonPtr json_array(JsonArray v)        { return std::make_shared<JsonValue>(std::move(v)); }
JsonPtr json_object(JsonObject v)      { return std::make_shared<JsonValue>(std::move(v)); }

namespace {

struct Parser {
    const char* p;
    const char* end;

    void skip_ws() {
        while (p < end && std::isspace((unsigned char)*p)) ++p;
    }

    char peek() { return (p < end) ? *p : '\0'; }
    char consume() { return (p < end) ? *p++ : '\0'; }

    JsonPtr parse_value();
    JsonPtr parse_string_val();
    JsonPtr parse_number();
    JsonPtr parse_array();
    JsonPtr parse_object();
    std::string parse_raw_string();

    [[noreturn]] void error(const char* msg) {
        throw std::runtime_error(std::string("JSON parse error: ") + msg);
    }
};

std::string Parser::parse_raw_string() {
    if (peek() != '"') error("expected '\"'");
    ++p;
    std::string s;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (p >= end) error("unexpected end in string escape");
            switch (*p++) {
                case '"': s += '"'; break;
                case '\\': s += '\\'; break;
                case '/': s += '/'; break;
                case 'n': s += '\n'; break;
                case 'r': s += '\r'; break;
                case 't': s += '\t'; break;
                case 'b': s += '\b'; break;
                case 'f': s += '\f'; break;
                case 'u': {
                    if (p + 4 > end) error("short \\u escape");
                    char hex[5] = {};
                    memcpy(hex, p, 4); p += 4;
                    unsigned cp = (unsigned)strtoul(hex, nullptr, 16);
                    if (cp < 0x80)      s += (char)cp;
                    else if (cp < 0x800){ s += (char)(0xC0|(cp>>6)); s += (char)(0x80|(cp&0x3F)); }
                    else { s += (char)(0xE0|(cp>>12)); s += (char)(0x80|((cp>>6)&0x3F)); s += (char)(0x80|(cp&0x3F)); }
                    break;
                }
                default: s += *(p-1); break;
            }
        } else {
            s += *p++;
        }
    }
    if (p >= end) error("unterminated string");
    ++p; // closing "
    return s;
}

JsonPtr Parser::parse_string_val() {
    return json_string(parse_raw_string());
}

JsonPtr Parser::parse_number() {
    const char* start = p;
    if (*p == '-') ++p;
    while (p < end && std::isdigit((unsigned char)*p)) ++p;
    if (p < end && *p == '.') { ++p; while (p < end && std::isdigit((unsigned char)*p)) ++p; }
    if (p < end && (*p == 'e' || *p == 'E')) {
        ++p;
        if (p < end && (*p == '+' || *p == '-')) ++p;
        while (p < end && std::isdigit((unsigned char)*p)) ++p;
    }
    std::string s(start, p);
    return json_number(std::stod(s));
}

JsonPtr Parser::parse_array() {
    ++p; // [
    JsonArray arr;
    skip_ws();
    if (peek() == ']') { ++p; return json_array(std::move(arr)); }
    while (true) {
        skip_ws();
        arr.push_back(parse_value());
        skip_ws();
        char c = peek();
        if (c == ']') { ++p; break; }
        if (c != ',') error("expected ',' or ']'");
        ++p;
    }
    return json_array(std::move(arr));
}

JsonPtr Parser::parse_object() {
    ++p; // {
    JsonObject obj;
    skip_ws();
    if (peek() == '}') { ++p; return json_object(std::move(obj)); }
    while (true) {
        skip_ws();
        if (peek() != '"') error("expected string key");
        std::string key = parse_raw_string();
        skip_ws();
        if (consume() != ':') error("expected ':'");
        skip_ws();
        obj[key] = parse_value();
        skip_ws();
        char c = peek();
        if (c == '}') { ++p; break; }
        if (c != ',') error("expected ',' or '}'");
        ++p;
    }
    return json_object(std::move(obj));
}

JsonPtr Parser::parse_value() {
    skip_ws();
    char c = peek();
    if (c == '"') return parse_string_val();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '-' || std::isdigit((unsigned char)c)) return parse_number();
    if (p + 4 <= end && strncmp(p, "true", 4) == 0)  { p += 4; return json_bool(true); }
    if (p + 5 <= end && strncmp(p, "false", 5) == 0) { p += 5; return json_bool(false); }
    if (p + 4 <= end && strncmp(p, "null", 4) == 0)  { p += 4; return json_null(); }
    error("unexpected character");
}

} // anon

JsonPtr parse(const char* text, size_t len) {
    Parser p{text, text + len};
    return p.parse_value();
}

JsonPtr parse(const std::string& text) {
    return parse(text.data(), text.size());
}

std::string stringify(const JsonValue& v, bool pretty, int indent) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return v.as_bool() ? "true" : "false";
    if (v.is_number()) {
        char buf[64];
        double d = v.as_number();
        if (d == (int64_t)d) snprintf(buf, sizeof(buf), "%lld", (long long)d);
        else snprintf(buf, sizeof(buf), "%g", d);
        return buf;
    }
    if (v.is_string()) {
        std::string s = "\"";
        for (char c : v.as_string()) {
            switch (c) {
                case '"': s += "\\\""; break;
                case '\\': s += "\\\\"; break;
                case '\n': s += "\\n"; break;
                case '\r': s += "\\r"; break;
                case '\t': s += "\\t"; break;
                default: s += c;
            }
        }
        s += '"';
        return s;
    }
    if (v.is_array()) {
        std::string s = "[";
        const auto& arr = v.as_array();
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i) s += ',';
            if (pretty) { s += '\n'; for (int j=0;j<indent+2;++j) s+=' '; }
            s += stringify(*arr[i], pretty, indent + 2);
        }
        if (pretty && !arr.empty()) { s += '\n'; for (int j=0;j<indent;++j) s+=' '; }
        s += ']';
        return s;
    }
    if (v.is_object()) {
        std::string s = "{";
        const auto& obj = v.as_object();
        bool first = true;
        for (const auto& [k, val] : obj) {
            if (!first) s += ',';
            first = false;
            if (pretty) { s += '\n'; for (int j=0;j<indent+2;++j) s+=' '; }
            s += '"'; s += k; s += "\":";
            if (pretty) s += ' ';
            s += stringify(*val, pretty, indent + 2);
        }
        if (pretty && !obj.empty()) { s += '\n'; for (int j=0;j<indent;++j) s+=' '; }
        s += '}';
        return s;
    }
    return "null";
}

std::string stringify(const JsonPtr& v, bool pretty, int indent) {
    if (!v) return "null";
    return stringify(*v, pretty, indent);
}

} // namespace khanary::util
