#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

namespace khanary::micronaut {

struct AIMLCategory {
    std::string pattern;
    std::string that;    // context pattern (empty = *)
    std::string topic;
    std::string tmpl;    // template (response)
};

class AIMLEngine {
public:
    AIMLEngine();

    // Load a .aiml file
    bool load_file(const std::string& path);

    // Load all .aiml files in a directory
    int load_dir(const std::string& dir);

    // Match input against loaded patterns
    // Returns the template or "" if no match
    std::string match(const std::string& input, const std::string& that = "") const;

    // Expand template: <srai>, <get>/<set>
    std::string expand(const std::string& tmpl, const std::string& input = "") const;

    size_t category_count() const { return categories_.size(); }

    void set(const std::string& name, const std::string& value);
    std::string get(const std::string& name) const;

private:
    std::vector<AIMLCategory>             categories_;
    mutable std::unordered_map<std::string, std::string> vars_;

    // Wildcard match: * matches anything, _ matches single token
    static bool pattern_match(const std::string& pattern,
                               const std::string& input,
                               std::vector<std::string>& captures);

    // Normalize input
    static std::string normalize(const std::string& s);

    // Parse XML-style AIML from string buffer
    void parse_aiml(const std::string& content);
    std::string parse_template(const std::string& xml_content) const;
};

} // namespace khanary::micronaut
