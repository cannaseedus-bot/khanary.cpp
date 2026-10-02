#include "conformance.h"
#include "../io/stb_map_convert.h"
#include <cmath>

using namespace khanary;

// Minimal SVG-Tensor fixture (same as test_svg_tensor.cpp).
static const char* kMinimal = R"svg(<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:k="https://khanary.io/tensor">
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
  <line x1="100" y1="200" x2="300" y2="200"
        k:edge_from="ELIZA" k:edge_to="Librarian"
        k:verb="@delegates_to" k:fold_type="vertical"
        k:weight="0.5" k:delta_theta="60.0"/>
</svg>)svg";

int main() {
    return test::run("stb_map_convert", [](test::Suite& suite) {

        // ── Empty graph ───────────────────────────────────────────────────────
        suite.check("empty_graph_empty_stbmap", []{
            io::SVGTensorGraph empty;
            const auto m = io::svg_tensor_to_stb_map(empty);
            return m.sections.empty();
        }());

        // ── SVG-Tensor → STBMap ───────────────────────────────────────────────
        const auto g = io::parse_svg_tensor(kMinimal);
        suite.check("parse_nonnull", g != nullptr);
        if (!g) return;

        const auto m = io::svg_tensor_to_stb_map(*g);

        // kMinimal: ELIZA has q/k/v → Field; both nodes → Geodesic; 1 edge → Fold
        suite.check("section_count_three", m.sections.size() == 3);
        if (m.sections.size() != 3) return;

        // Field section: node[0]=ELIZA only (Librarian has no q/k/v)
        const auto field_recs = gpu::get_field_records(m.sections[0]);
        suite.check("field_record_count_one", field_recs.size() == 1);
        if (!field_recs.empty()) {
            suite.check("field_q_x",  std::abs(field_recs[0].q[0] - 0.5f)  < 1e-5f);
            suite.check("field_v_x",  std::abs(field_recs[0].v[0] - 0.34f) < 1e-5f);
        } else {
            suite.check("field_q_x", false);
            suite.check("field_v_x", false);
        }

        // Geodesic section: both nodes
        const auto geo_recs = gpu::get_geodesic_records(m.sections[1]);
        suite.check("geodesic_record_count", geo_recs.size() == 2);
        if (geo_recs.size() >= 2) {
            suite.check("geodesic_eliza_entropy",
                std::abs(geo_recs[0].local_entropy - 0.34f) < 1e-5f);
            suite.check("geodesic_eliza_xyz",
                std::abs(geo_recs[0].x - 1.0f) < 1e-5f &&
                std::abs(geo_recs[0].y - 2.0f) < 1e-5f &&
                std::abs(geo_recs[0].z - 0.5f) < 1e-5f);
            suite.check("geodesic_eliza_sidecar_mesh",
                std::abs(geo_recs[0].geodesic_entropy - 0.55f) < 1e-5f &&
                std::abs(geo_recs[0].unruh_temp - 0.02f) < 1e-5f &&
                std::abs(geo_recs[0].divergence - (-0.4f)) < 1e-5f &&
                std::abs(geo_recs[0].circumradius_R - 2.5f) < 1e-5f &&
                std::abs(geo_recs[0].inradius_r - 1.0f) < 1e-5f);
            suite.check("geodesic_librarian_entropy_zero",
                geo_recs[1].local_entropy == 0.0f);
        } else {
            suite.check("geodesic_eliza_entropy",        false);
            suite.check("geodesic_eliza_xyz",            false);
            suite.check("geodesic_eliza_sidecar_mesh",   false);
            suite.check("geodesic_librarian_entropy_zero", false);
        }

        // Fold section: one edge ELIZA→Librarian, fold_type="vertical"
        const auto fold_recs = gpu::get_fold_records(m.sections[2]);
        suite.check("fold_record_count_one", fold_recs.size() == 1);
        if (!fold_recs.empty()) {
            suite.check("fold_delta_theta",
                std::abs(fold_recs[0].delta_theta - 60.0f) < 1e-4f);
            suite.check("fold_arc_weight",
                std::abs(fold_recs[0].arc_weight - 0.5f)  < 1e-5f);
            suite.check("fold_type_vertical", fold_recs[0].fold_type == 0u);
            suite.check("fold_from_phase_zero", fold_recs[0].from_phase == 0u);
            suite.check("fold_to_phase_one",    fold_recs[0].to_phase   == 1u);
        } else {
            suite.check("fold_delta_theta",     false);
            suite.check("fold_arc_weight",      false);
            suite.check("fold_type_vertical",   false);
            suite.check("fold_from_phase_zero", false);
            suite.check("fold_to_phase_one",    false);
        }

        // ── FieldNode/FieldEdge vector path ───────────────────────────────────
        {
            world::FieldNode n{};
            n.track_id         = "Alpha";
            n.x = 1.0f; n.y = 0.5f; n.z = 0.25f;
            n.local_entropy    = 0.7f;
            n.geodesic_entropy = 0.8f;
            n.unruh_temp       = 0.03f;
            n.divergence       = -0.2f;
            n.circumradius_R   = 3.0f;
            n.inradius_r       = 1.25f;

            const auto mv = io::field_vectors_to_stb_map({n}, {});
            if (!mv.sections.empty()) {
                const auto gr = gpu::get_geodesic_records(mv.sections[0]);
                suite.check("field_vecs_geodesic_count", gr.size() == 1);
                suite.check("field_vecs_geodesic_entropy",
                    !gr.empty() &&
                    std::abs(gr[0].local_entropy - 0.7f) < 1e-5f &&
                    std::abs(gr[0].geodesic_entropy - 0.8f) < 1e-5f &&
                    std::abs(gr[0].circumradius_R - 3.0f) < 1e-5f &&
                    std::abs(gr[0].inradius_r - 1.25f) < 1e-5f);
            } else {
                suite.check("field_vecs_geodesic_count",   false);
                suite.check("field_vecs_geodesic_entropy", false);
            }
        }
        {
            world::FieldNode na{}, nb{};
            na.track_id = "A"; nb.track_id = "B";
            world::FieldEdge e{};
            e.from = "A"; e.to = "B"; e.type = "delegate";
            e.weight = 0.75f; e.delta_theta = 45.0f;

            const auto mv = io::field_vectors_to_stb_map({na, nb}, {e});
            // geodesic section[0] + fold section[1]
            suite.check("field_vecs_fold_count", mv.sections.size() == 2);
            if (mv.sections.size() == 2) {
                const auto fr = gpu::get_fold_records(mv.sections[1]);
                suite.check("field_vecs_fold_delegate_type",
                    !fr.empty() && fr[0].fold_type == 2u);
            } else {
                suite.check("field_vecs_fold_delegate_type", false);
            }
        }

    }, 21);
}
