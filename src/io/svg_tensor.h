#pragma once
#include <memory>
#include <string>
#include <vector>

namespace khanary::io {

// Node loaded from an SVG-Tensor file via k:* namespace attributes.
// Fields default to safe values; only what the source provides is set.
struct SVGTensorNode {
    std::string track_id;
    std::string phase_name;
    std::string kind;           // "track" | "tensor" | "pipeline_stage"
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float entropy    = 0.0f;
    float geodesic_entropy = 0.0f;
    float unruh_temp       = 0.0f;
    float divergence       = 0.0f;
    float circumradius_R   = 0.0f;
    float inradius_r       = 0.0f;
    float weight     = 1.0f;
    float confidence = 1.0f;
    std::vector<float> q_vec;   // query lane (comma-separated in k:q_vec)
    std::vector<float> k_vec;   // key lane
    std::vector<float> v_vec;   // value lane
};

// Edge loaded from an SVG-Tensor file via k:* namespace attributes.
struct SVGTensorEdge {
    std::string from;           // k:edge_from
    std::string to;             // k:edge_to
    std::string verb;           // k:verb (e.g. "@delegates_to")
    std::string fold_type;      // "horizontal" | "vertical"
    float weight      = 1.0f;
    float delta_theta = 0.0f;
};

// Full graph parsed from one SVG-Tensor document.
struct SVGTensorGraph {
    std::vector<SVGTensorNode> nodes;
    std::vector<SVGTensorEdge> edges;
};

// Parse SVG-Tensor content from a string.
// Elements are classified as nodes (k:track_id present) or edges
// (k:edge_from + k:edge_to present).  Non-k:* SVG elements are ignored.
std::unique_ptr<SVGTensorGraph> parse_svg_tensor(const std::string& text);

// Load an SVG-Tensor file.  Returns nullptr if the file cannot be opened.
std::unique_ptr<SVGTensorGraph> load_svg_tensor(const std::string& path);

} // namespace khanary::io
