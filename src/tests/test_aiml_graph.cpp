#include "conformance.h"
#include "../micronaut/aiml_engine.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace khanary::micronaut;

namespace {

bool expect_eq(const std::string& label, const std::string& got, const std::string& want) {
    if (got == want) return true;
    std::cerr << "[FAIL] " << label << ": expected '" << want << "' got '" << got << "'\n";
    return false;
}

bool expect_true(const std::string& label, bool value) {
    if (value) return true;
    std::cerr << "[FAIL] " << label << ": expected true\n";
    return false;
}

bool expect_contains(const std::string& label, const std::string& got, const std::string& needle) {
    if (got.find(needle) != std::string::npos) return true;
    std::cerr << "[FAIL] " << label << ": expected substring '" << needle << "' in '" << got << "'\n";
    return false;
}

} // namespace

static int acceptance_main() {
    AIMLEngine aiml;
    aiml.clear_graph();

    aiml.add_graph_node("root");
    aiml.add_graph_node("childA");
    aiml.add_graph_node("childB");
    aiml.add_graph_node("claim");
    aiml.add_graph_node("evidence");
    aiml.add_graph_edge("root", "childA", "@delegate", true);
    aiml.add_graph_edge("root", "childB", "@delegate", true);
    aiml.add_graph_edge("root", "claim", "@related_to", true);
    aiml.add_graph_edge("evidence", "claim", "@supports", true);
    aiml.set_graph_root("root");

    if (!expect_eq("node bind", aiml.expand("<node id=\"root\"/>"), "root")) return 1;
    if (!expect_eq("getroot", aiml.expand("<getroot/>"), "root")) return 1;
    if (!expect_eq("edges", aiml.expand("<edges source=\"root\" verb=\"@delegate\" join=\",\"/>"), "childA,childB")) return 1;
    if (!expect_eq("edges canonical alias", aiml.expand("<edges source=\"root\" relation=\"@delegates_to\" join=\",\"/>"), "childA,childB")) return 1;
    if (!expect_eq("getchildren", aiml.expand("<getchildren id=\"root\" verb=\"@delegate\" join=\"|\"/>"), "childA|childB")) return 1;
    if (!expect_eq("getedge index=2", aiml.expand("<getedge source=\"root\" verb=\"@delegate\" index=\"2\"/>"), "childB")) return 1;
    if (!expect_eq("getindex childB", aiml.expand("<getindex within=\"root\" id=\"childB\" verb=\"@delegate\"/>"), "2")) return 1;
    if (!expect_eq("getlevel childB", aiml.expand("<getlevel id=\"childB\"/>"), "1")) return 1;
    if (!expect_eq("traverse relation", aiml.expand("<traverse start=\"root\" relation=\"@delegates_to\" direction=\"out\" depth=\"1\" join=\",\"/>"), "childA,childB")) return 1;
    if (!expect_eq("traverse target hit", aiml.expand("<traverse start=\"evidence\" relation=\"@supports\" direction=\"out\" depth=\"1\" target=\"claim\"/>"), "claim")) return 1;
    if (!expect_contains("traverse miss_relation", aiml.expand("<traverse start=\"root\" relation=\"@causes\" direction=\"out\" depth=\"1\" target=\"claim\"/>"), "MISS_relation")) return 1;

    if (!expect_eq("node-write", aiml.expand("<node-write id=\"childA\" field=\"response\">ready</node-write>"), "ready")) return 1;
    if (!expect_eq("node-attr", aiml.expand("<node-attr id=\"childA\" field=\"response\"/>"), "ready")) return 1;

    const auto graphml_path = std::filesystem::temp_directory_path() / "khanary_aiml_graph_test.graphml";
    {
        std::ofstream g(graphml_path);
        g << "<graphml>\n";
        g << "  <graph edgedefault=\"directed\">\n";
        g << "    <node id=\"g0\"/>\n";
        g << "    <node id=\"g1\"/>\n";
        g << "    <edge source=\"g0\" target=\"g1\" verb=\"@relate\"/>\n";
        g << "  </graph>\n";
        g << "</graphml>\n";
    }

    aiml.clear_graph();
    const bool loaded = aiml.load_graphml(graphml_path.string());
    std::error_code ec;
    std::filesystem::remove(graphml_path, ec);

    if (!expect_true("graphml loaded", loaded)) return 1;
    if (!expect_true("graphml node count", aiml.graph_node_count() >= 2)) return 1;
    if (!expect_true("graphml edge count", aiml.graph_edge_count() >= 1)) return 1;
    if (!expect_eq("graphml edge traversal alias", aiml.expand("<edges source=\"g0\" verb=\"@relate\" join=\",\"/>"), "g1")) return 1;
    if (!expect_eq("graphml edge traversal canonical", aiml.expand("<edges source=\"g0\" relation=\"@related_to\" join=\",\"/>"), "g1")) return 1;

    std::cout << "AIML graph tests passed.\n";
    return 0;
}

int main() { return khanary::test::legacy("aiml", [] { return acceptance_main(); }); }
