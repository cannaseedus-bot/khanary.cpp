#pragma once
#include "../kuhul/xul_gate.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <functional>
#include <cstddef>

namespace khanary::micronaut {

struct AIMLCategory {
    std::string pattern;
    std::string that;    // context pattern (empty = *)
    std::string topic;
    std::string tmpl;    // template (response)
};

struct AIMLGraphEdge {
    std::string source;
    std::string target;
    std::string verb;
    bool        directed = true;
};

class AIMLEngine {
public:
    using ToolFn = std::function<std::string(const std::string&)>;
    using SemanticFallbackFn = std::function<std::string(const std::string&)>;

    AIMLEngine();

    // Load a .aiml file
    bool load_file(const std::string& path);

    // Load all .aiml files in a directory
    int load_dir(const std::string& dir);

    // Match input against loaded patterns
    // Returns the template or "" if no match
    std::string match(const std::string& input, const std::string& that = "") const;

    // Expand template: <srai>, <get>/<set>, graph tags (<edges>/<getedge>/<traverse>/...)
    std::string expand(const std::string& tmpl, const std::string& input = "") const;

    size_t category_count() const { return categories_.size(); }

    void set(const std::string& name, const std::string& value);
    std::string get(const std::string& name) const;

    // Register a template tool callable from <tool name="...">arg</tool>.
    void register_tool(const std::string& name, ToolFn fn);
    bool has_tool(const std::string& name) const;

    // Fallback when no AIML category matches normalized input.
    void set_semantic_fallback(SemanticFallbackFn fn);

    // Load variable state (<set>/<get>) from JSON object.
    bool load_state(const std::string& path);

    // Persist variable state — CommitAuthority is consumed (move-only).
    bool persist(kuhul::CommitAuthority authority, const std::string& path) const;

    // Graph surfaces for graph-native AIML tags.
    void clear_graph();
    void set_graph_root(const std::string& node_id);
    void add_graph_node(const std::string& node_id);
    void add_graph_node_attr(const std::string& node_id,
                             const std::string& field,
                             const std::string& value);
    void add_graph_edge(const std::string& source,
                        const std::string& target,
                        const std::string& verb,
                        bool directed = true);
    bool load_graphml(const std::string& path);
    size_t graph_node_count() const;
    size_t graph_edge_count() const;

private:
    bool save_state(const std::string& path) const;

    std::vector<AIMLCategory>             categories_;
    mutable std::unordered_map<std::string, std::string> vars_;
    mutable std::mutex vars_mtx_;
    mutable std::mutex tools_mtx_;
    std::unordered_map<std::string, ToolFn> tools_;
    mutable std::mutex fallback_mtx_;
    SemanticFallbackFn semantic_fallback_;
    mutable std::mutex graph_mtx_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> graph_nodes_;
    std::vector<AIMLGraphEdge> graph_edges_;
    std::string graph_root_;

    // Wildcard match: * matches anything, _ matches single token
    static bool pattern_match(const std::string& pattern,
                               const std::string& input,
                               std::vector<std::string>& captures);

    // Normalize input
    static std::string normalize(const std::string& s);

    // Parse XML-style AIML from string buffer
    void parse_aiml(const std::string& content);
    std::string expand_impl(const std::string& tmpl, const std::string& input) const;
};

} // namespace khanary::micronaut
