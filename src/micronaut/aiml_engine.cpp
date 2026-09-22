#include "aiml_engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <functional>
#include <regex>
#include <filesystem>

namespace khanary::micronaut {

namespace fs = std::filesystem;

AIMLEngine::AIMLEngine() {
    vars_["topic"] = "*";
}

static std::string xml_text_between(const std::string& s, const std::string& tag) {
    std::string open  = "<" + tag + ">";
    std::string close = "</" + tag + ">";
    size_t a = s.find(open);
    if (a == std::string::npos) return "";
    a += open.size();
    size_t b = s.find(close, a);
    if (b == std::string::npos) return s.substr(a);
    return s.substr(a, b - a);
}

std::string AIMLEngine::normalize(const std::string& s) {
    std::string r;
    for (char c : s) r += (char)toupper((unsigned char)c);
    // Collapse whitespace
    std::string out;
    bool in_ws = false;
    for (char c : r) {
        if (std::isspace((unsigned char)c)) {
            if (!in_ws) { out += ' '; in_ws = true; }
        } else {
            out += c; in_ws = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    return out;
}

bool AIMLEngine::pattern_match(
    const std::string& pattern,
    const std::string& input,
    std::vector<std::string>& captures)
{
    if (pattern == "*" || pattern == "_") { captures.push_back(input); return true; }

    std::vector<std::string> ptoks, itoks;
    {
        std::istringstream ps(pattern), is(input);
        std::string t;
        while (ps >> t) ptoks.push_back(t);
        while (is >> t) itoks.push_back(t);
    }

    // Simple recursive match
    std::function<bool(size_t, size_t)> rec = [&](size_t pi, size_t ii) -> bool {
        if (pi == ptoks.size() && ii == itoks.size()) return true;
        if (pi == ptoks.size()) return false;
        if (ptoks[pi] == "*" || ptoks[pi] == "_") {
            // Greedy: try consuming 1..N input tokens
            std::string cap;
            for (size_t j = ii; j <= itoks.size(); ++j) {
                if (rec(pi + 1, j)) {
                    captures.push_back(cap);
                    return true;
                }
                if (j < itoks.size()) {
                    if (!cap.empty()) cap += ' ';
                    cap += itoks[j];
                }
            }
            return false;
        }
        if (ii >= itoks.size()) return false;
        if (ptoks[pi] != itoks[ii]) return false;
        return rec(pi + 1, ii + 1);
    };
    return rec(0, 0);
}

void AIMLEngine::parse_aiml(const std::string& content) {
    // Very light XML parser: find <category>...</category> blocks
    size_t pos = 0;
    while (true) {
        size_t cat_start = content.find("<category>", pos);
        if (cat_start == std::string::npos) break;
        size_t cat_end = content.find("</category>", cat_start);
        if (cat_end == std::string::npos) break;
        std::string cat_body = content.substr(cat_start + 10, cat_end - cat_start - 10);
        pos = cat_end + 11;

        AIMLCategory cat;
        cat.pattern = normalize(xml_text_between(cat_body, "pattern"));
        cat.that    = xml_text_between(cat_body, "that");
        if (cat.that.empty()) cat.that = "*";
        cat.tmpl    = xml_text_between(cat_body, "template");
        if (!cat.pattern.empty() && !cat.tmpl.empty())
            categories_.push_back(std::move(cat));
    }
}

bool AIMLEngine::load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    parse_aiml(ss.str());
    return true;
}

int AIMLEngine::load_dir(const std::string& dir) {
    int count = 0;
    if (!fs::is_directory(dir)) return 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".aiml") {
            if (load_file(entry.path().string())) ++count;
        }
    }
    return count;
}

std::string AIMLEngine::match(const std::string& input, const std::string& that) const {
    std::string norm_in   = normalize(input);
    std::string norm_that = that.empty() ? "*" : normalize(that);

    for (const auto& cat : categories_) {
        std::vector<std::string> caps;
        if (!pattern_match(cat.pattern, norm_in, caps)) continue;
        if (cat.that != "*") {
            std::vector<std::string> that_caps;
            if (!pattern_match(cat.that, norm_that, that_caps)) continue;
        }
        // Expand <star/> in template
        std::string tpl = cat.tmpl;
        size_t star_pos = tpl.find("<star/>");
        if (star_pos != std::string::npos && !caps.empty())
            tpl.replace(star_pos, 7, caps[0]);
        return expand(tpl, input);
    }
    return "";
}

std::string AIMLEngine::expand(const std::string& tmpl, const std::string& /*input*/) const {
    std::string result = tmpl;
    // <srai>...</srai> → recursive match
    size_t pos = 0;
    while (true) {
        size_t s = result.find("<srai>", pos);
        if (s == std::string::npos) break;
        size_t e = result.find("</srai>", s);
        if (e == std::string::npos) break;
        std::string inner = result.substr(s + 6, e - s - 6);
        std::string sub = match(inner);
        result.replace(s, e - s + 7, sub.empty() ? inner : sub);
        pos = s + (sub.empty() ? inner.size() : sub.size());
    }
    // <get name="..."/>
    {
        std::regex get_re("<get name=\"([^\"]+)\"\\s*/>");
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), get_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            out += get(m[1].str());
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }
    // <set name="...">...</set>
    {
        std::regex set_re("<set name=\"([^\"]+)\">([^<]*)</set>");
        std::string out;
        auto beg = std::sregex_iterator(result.begin(), result.end(), set_re);
        auto end = std::sregex_iterator{};
        size_t last = 0;
        for (auto it = beg; it != end; ++it) {
            const auto& m = *it;
            out += result.substr(last, m.position() - last);
            const_cast<AIMLEngine*>(this)->set(m[1].str(), m[2].str());
            out += m[2].str();
            last = m.position() + m.length();
        }
        out += result.substr(last);
        result = std::move(out);
    }
    return result;
}

void AIMLEngine::set(const std::string& name, const std::string& value) {
    vars_[name] = value;
}

std::string AIMLEngine::get(const std::string& name) const {
    auto it = vars_.find(name);
    return it != vars_.end() ? it->second : "";
}

} // namespace khanary::micronaut
