#pragma once
#include "../util/json.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace khanary::io {

struct CodeNode {
    std::string id;       // "n0", "n1", ...
    std::string type;     // grammar token type, e.g. "function_def"
    std::string text;     // matched source text
    size_t      line = 0; // 1-based
    size_t      col  = 0; // 0-based byte offset from line start
    std::unordered_map<std::string, std::string> captures; // named capture groups
};

struct CodeEdge {
    std::string from;       // CodeNode::id
    std::string to;         // CodeNode::id
    std::string kind;       // edge label from grammar rules
    float       weight = 1.0f;
};

struct CodeGraph {
    std::string           language;
    std::vector<CodeNode> nodes;
    std::vector<CodeEdge> edges;
};

// Parse source text against a loaded grammar JSON.
// This is the AST adapter stage (Abstract Syntax Transpiler): it normalizes
// surface syntax into typed nodes/edges before K'UHUL phase execution.
// grammar_json: parsed content of a .code-grammar.json file
// source:       the source text to scan
CodeGraph parse_with_grammar(const util::JsonPtr& grammar_json,
                              const std::string& source);

// Semantic grounding evidence for one canonical atom.
struct SemanticAtomEvidence {
    std::string atom;
    std::vector<std::string> cues;
};

// Resolve lexical/morpheme cues into canonical semantic atoms.
std::vector<SemanticAtomEvidence> derive_semantic_atom_grounding(
    const std::vector<std::string>& tokens_lower);

// Build a deterministic Fold Graph Parser payload:
// Text -> grams -> associations -> books -> semantic atoms -> typed relations.
util::JsonPtr fold_graph_parse(const std::string& text);
util::JsonPtr semantic_vector_contract();

} // namespace khanary::io
