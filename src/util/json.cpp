#include "json.h"
#include <sstream>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>

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

// Reject malformed UTF-8, including overlong encodings and surrogate scalars.
void check_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        unsigned c = static_cast<unsigned char>(s[i++]);
        if (c < 0x80) continue;
        unsigned cp = 0, count = 0, minimum = 0;
        if (c >= 0xc2 && c <= 0xdf) { cp = c & 31; count = 1; minimum = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { cp = c & 15; count = 2; minimum = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { cp = c & 7; count = 3; minimum = 0x10000; }
        else throw std::runtime_error("JSON: invalid UTF-8");
        if (count > s.size() - i) throw std::runtime_error("JSON: short UTF-8");
        while (count--) {
            c = static_cast<unsigned char>(s[i++]);
            if ((c & 0xc0) != 0x80) throw std::runtime_error("JSON: invalid UTF-8 continuation");
            cp = (cp << 6) | (c & 63);
        }
        if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
            throw std::runtime_error("JSON: invalid Unicode scalar");
    }
}

std::string quote(const std::string& value) {
    check_utf8(value);
    const char* hex = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
                else out += static_cast<char>(c);
        }
    }
    return out + '"';
}

struct Parser {
    const char* p;
    const char* end;
    unsigned depth = 0;

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    }

    char peek() { return (p < end) ? *p : '\0'; }
    char consume() { return (p < end) ? *p++ : '\0'; }

    JsonPtr parse_value();
    JsonPtr parse_string_val();
    JsonPtr parse_number();
    JsonPtr parse_array();
    JsonPtr parse_object();
    std::string parse_raw_string();
    unsigned hex4() {
        if (end - p < 4) error("short Unicode escape");
        unsigned result = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = *p++;
            unsigned digit;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else error("invalid Unicode escape");
            result = (result << 4) | digit;
        }
        return result;
    }

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
                    unsigned cp = hex4();
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        if (end - p < 2 || p[0] != '\\' || p[1] != 'u') error("missing low surrogate");
                        p += 2;
                        const unsigned low = hex4();
                        if (low < 0xdc00 || low > 0xdfff) error("invalid low surrogate");
                        cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                    } else if (cp >= 0xdc00 && cp <= 0xdfff) error("unpaired low surrogate");
                    if (cp < 0x80)      s += (char)cp;
                    else if (cp < 0x800){ s += (char)(0xC0|(cp>>6)); s += (char)(0x80|(cp&0x3F)); }
                    else if (cp < 0x10000) { s += (char)(0xE0|(cp>>12)); s += (char)(0x80|((cp>>6)&0x3F)); s += (char)(0x80|(cp&0x3F)); }
                    else { s += (char)(0xf0|(cp>>18)); s += (char)(0x80|((cp>>12)&63)); s += (char)(0x80|((cp>>6)&63)); s += (char)(0x80|(cp&63)); }
                    break;
                }
                default: error("unknown string escape");
            }
        } else {
            if (static_cast<unsigned char>(*p) < 0x20) error("unescaped control character");
            s += *p++;
        }
    }
    if (p >= end) error("unterminated string");
    ++p; // closing "
    check_utf8(s);
    return s;
}

JsonPtr Parser::parse_string_val() {
    return json_string(parse_raw_string());
}

JsonPtr Parser::parse_number() {
    const char* start = p;
    if (*p == '-') ++p;
    auto digit = [&]() { return p < end && *p >= '0' && *p <= '9'; };
    if (!digit()) error("expected number digit");
    if (*p == '0') { ++p; if (digit()) error("leading zero"); }
    else while (digit()) ++p;
    if (p < end && *p == '.') { ++p; if (!digit()) error("missing fraction"); while (digit()) ++p; }
    if (p < end && (*p == 'e' || *p == 'E')) {
        ++p;
        if (p < end && (*p == '+' || *p == '-')) ++p;
        if (!digit()) error("missing exponent");
        while (digit()) ++p;
    }
    double value = 0;
    const auto result = std::from_chars(start, p, value);
    if (result.ec != std::errc() || result.ptr != p || !std::isfinite(value)) error("number outside binary64 range");
    return json_number(value);
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
        auto value = parse_value();
        if (!obj.emplace(key, std::move(value)).second) error("duplicate object key");
        skip_ws();
        char c = peek();
        if (c == '}') { ++p; break; }
        if (c != ',') error("expected ',' or '}'");
        ++p;
    }
    return json_object(std::move(obj));
}

JsonPtr Parser::parse_value() {
    if (++depth > 256) error("nesting limit exceeded");
    struct DepthGuard { unsigned& value; ~DepthGuard() { --value; } } guard{depth};
    skip_ws();
    char c = peek();
    if (c == '"') return parse_string_val();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '-' || std::isdigit((unsigned char)c)) return parse_number();
    if (end - p >= 4 && strncmp(p, "true", 4) == 0)  { p += 4; return json_bool(true); }
    if (end - p >= 5 && strncmp(p, "false", 5) == 0) { p += 5; return json_bool(false); }
    if (end - p >= 4 && strncmp(p, "null", 4) == 0)  { p += 4; return json_null(); }
    error("unexpected character");
}

} // anon

JsonPtr parse(const char* text, size_t len) {
    if (!text || len == 0) throw std::runtime_error("JSON parse error: empty input");
    Parser p{text, text + len};
    auto value = p.parse_value();
    p.skip_ws();
    if (p.p != p.end) p.error("trailing content");
    return value;
}

JsonPtr parse(const std::string& text) {
    return parse(text.data(), text.size());
}

std::string stringify(const JsonValue& v, bool pretty, int indent) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return v.as_bool() ? "true" : "false";
    if (v.is_number()) {
        char buf[64];
        const double d = v.as_number();
        if (!std::isfinite(d)) throw std::runtime_error("JSON: non-finite number");
        if (d == 0) return "0";
        const auto result = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::general);
        if (result.ec != std::errc()) throw std::runtime_error("JSON: number serialization failed");
        std::string number(buf, result.ptr);
        const auto e = number.find('e');
        if (e != std::string::npos) {
            size_t digits = e + 1;
            if (number[digits] == '+') number.erase(digits, 1);
            else if (number[digits] == '-') ++digits;
            while (number.size() - digits > 1 && number[digits] == '0') number.erase(digits, 1);
        }
        return number;
    }
    if (v.is_string()) {
        return quote(v.as_string());
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
        std::vector<std::string> keys;
        for (const auto& entry : obj) keys.push_back(entry.first);
        std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                [](unsigned char x, unsigned char y) { return x < y; });
        });
        bool first = true;
        for (const auto& k : keys) {
            const auto& val = obj.at(k);
            if (!first) s += ',';
            first = false;
            if (pretty) { s += '\n'; for (int j=0;j<indent+2;++j) s+=' '; }
            s += quote(k); s += ':';
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
