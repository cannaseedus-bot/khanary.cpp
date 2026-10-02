#include "conformance.h"
#include "../io/svg_tensor.h"
#include <cmath>

using namespace khanary;

// Minimal SVG-Tensor fixture.  k:* attributes carry semantic state;
// non-k:* SVG attributes (cx, cy, r, ...) are ignored by the loader.
static const char* kMinimal = R"svg(<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:k="https://khanary.io/tensor">
  <!-- field nodes -->
  <circle cx="100" cy="200" r="10"
          k:track_id="ELIZA" k:phase="Wo" k:entropy="0.34"
          k:geodesic_entropy="0.55" k:unruh_temp="0.02" k:divergence="-0.4"
          k:circumradius_R="2.5" k:inradius_r="1.0"
          k:x="1.0" k:y="2.0" k:z="0.5"
          k:weight="0.9" k:confidence="0.75"
          k:q_vec="0.5, 0.0, 0.0" k:k_vec="0.0, 0.5, 0.0" k:v_vec="0.34, 0.9, 0.75"/>
  <circle cx="300" cy="200" r="10"
          k:track_id="Librarian" k:phase="Pop"
          k:kind="tensor"/>
  <!-- fold edge -->
  <line x1="100" y1="200" x2="300" y2="200"
        k:edge_from="ELIZA" k:edge_to="Librarian"
        k:verb="@delegates_to" k:fold_type="vertical"
        k:weight="0.5" k:delta_theta="60.0"/>
</svg>)svg";

int main() {
    return test::run("svg_tensor", [](test::Suite& suite) {

        // ── Empty input ───────────────────────────────────────────────────────
        suite.check("empty_svg_empty_graph", []{
            const auto g = io::parse_svg_tensor("<svg/>");
            return g && g->nodes.empty() && g->edges.empty();
        }());

        // ── xmlns:k attribute must NOT be extracted as a node/edge ────────────
        suite.check("xmlns_k_not_extracted", []{
            const auto g = io::parse_svg_tensor(
                R"(<svg xmlns:k="https://khanary.io/tensor"/>)");
            return g && g->nodes.empty() && g->edges.empty();
        }());

        // ── Non-k:* elements are ignored ─────────────────────────────────────
        suite.check("ignore_non_k_elements", []{
            const auto g = io::parse_svg_tensor(
                R"(<svg><circle cx="100" cy="200" r="10"/></svg>)");
            return g && g->nodes.empty() && g->edges.empty();
        }());

        // ── Node parsing ──────────────────────────────────────────────────────
        const auto g = io::parse_svg_tensor(kMinimal);
        suite.check("parse_returns_nonnull", g != nullptr);
        if (!g) return;

        suite.check("mixed_graph_two_nodes_one_edge",
            g->nodes.size() == 2 && g->edges.size() == 1);

        const auto& n0 = g->nodes[0];
        suite.check("node_track_id",   n0.track_id   == "ELIZA");
        suite.check("node_phase_parsed", n0.phase_name == "Wo");
        suite.check("node_entropy_parsed",
            std::abs(n0.entropy - 0.34f) < 1e-5f);
        suite.check("node_sidecar_fields_parsed",
            std::abs(n0.geodesic_entropy - 0.55f) < 1e-5f &&
            std::abs(n0.unruh_temp - 0.02f) < 1e-5f &&
            std::abs(n0.divergence - (-0.4f)) < 1e-5f &&
            std::abs(n0.circumradius_R - 2.5f) < 1e-5f &&
            std::abs(n0.inradius_r - 1.0f) < 1e-5f);
        suite.check("node_xyz_parsed",
            std::abs(n0.x - 1.0f) < 1e-5f &&
            std::abs(n0.y - 2.0f) < 1e-5f &&
            std::abs(n0.z - 0.5f) < 1e-5f);
        suite.check("node_weight_parsed",
            std::abs(n0.weight - 0.9f) < 1e-5f);
        suite.check("node_confidence_parsed",
            std::abs(n0.confidence - 0.75f) < 1e-5f);

        // Q/K/V vector parsing
        suite.check("q_vec_parsed",
            n0.q_vec.size() == 3 &&
            std::abs(n0.q_vec[0] - 0.5f) < 1e-5f);
        suite.check("k_vec_parsed",
            n0.k_vec.size() == 3 &&
            std::abs(n0.k_vec[1] - 0.5f) < 1e-5f);
        suite.check("v_vec_parsed",
            n0.v_vec.size() == 3 &&
            std::abs(n0.v_vec[0] - 0.34f) < 1e-5f);

        // Second node defaults
        const auto& n1 = g->nodes[1];
        suite.check("default_kind_is_tensor", n1.kind == "tensor");
        suite.check("default_weight_is_1",
            std::abs(n1.weight - 1.0f) < 1e-5f);
        suite.check("default_confidence_is_1",
            std::abs(n1.confidence - 1.0f) < 1e-5f);

        // ── Edge parsing ──────────────────────────────────────────────────────
        const auto& e0 = g->edges[0];
        suite.check("edge_from_to",
            e0.from == "ELIZA" && e0.to == "Librarian");
        suite.check("edge_fields_correct",
            e0.verb      == "@delegates_to" &&
            e0.fold_type == "vertical"      &&
            std::abs(e0.weight      - 0.5f ) < 1e-5f &&
            std::abs(e0.delta_theta - 60.0f) < 1e-4f);

    }, 20);
}
