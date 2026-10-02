#include "conformance.h"
#include "../world/gml_parser.h"

using namespace khanary::world;

int main() {
    return khanary::test::run("gml-parser-native", [](khanary::test::Suite& suite) {
        const char* gml = R"GML(
graph [
  directed 1
  id 501
  label "Decision_Space_Alpha"
  node [ id 10 label "hot" x -1.0 y 0.5 z 0.0 prior_mass 0.65 risk_bound 0.12 mutation "modify" base_version 1 ]
  node [ id 20 label "cold" x 1.0 y -0.5 z 0.0 prior_mass 0.35 risk_bound 0.08 mutation "unchanged" base_version 1 ]
  edge [ id 101 source 10 target 20 label "hot_to_cold" verb "@contradicts" weight 1.0 dual_twin 102 loss_cost 0.95 mutation "modify" ]
  edge [ id 102 source 20 target 10 label "cold_to_hot" verb "@contradicts" weight 1.0 dual_twin 101 loss_cost 0.95 mutation "modify" ]
]
)GML";
        auto graph = parse_gml(gml);
        suite.check("graph_parsed", static_cast<bool>(graph));
        suite.check("graph_directed", graph && graph->directed && graph->id == 501);
        suite.check("node_decision_fields", graph && graph->nodes.size() == 2 &&
            graph->nodes[0].has_prior_mass && graph->nodes[0].prior_mass == 0.65 &&
            graph->nodes[0].has_risk_bound && graph->nodes[0].risk_bound == 0.12 &&
            graph->nodes[0].has_base_version && graph->nodes[0].base_version == 1 &&
            graph->nodes[0].mutation == "modify");
        suite.check("edge_dual_loss_fields", graph && graph->edges.size() == 2 &&
            graph->edges[0].has_id && graph->edges[0].id == 101 &&
            graph->edges[0].has_dual_twin && graph->edges[0].dual_twin == 102 &&
            graph->edges[0].has_loss_cost && graph->edges[0].loss_cost == 0.95 &&
            graph->edges[0].mutation == "modify");
        suite.check("edge_twin_roundtrip", graph && graph->edges.size() == 2 &&
            graph->edges[0].dual_twin == graph->edges[1].id &&
            graph->edges[1].dual_twin == graph->edges[0].id);

        // ── Vector field: divergence / singularity / vx / vy / vz ──────────
        const char* gml_vf = R"GML(
graph [
  directed 1
  id 1205
  label "Semantic_Vortex_Alpha"
  node [ id 100 label "Root_Context_Invariance" x 0.0 y 0.0 z 0.0 divergence -2.45 singularity "sink" ]
  node [ id 101 label "Peripheral_Evidence_A"   x -1.5 y 1.0 z 0.2 divergence 0.0  singularity "ordinary" ]
  edge [ source 101 target 100 label "gradient_descent_line" verb "@requires" weight 1.0 vx 0.75 vy -0.50 vz -0.10 ]
]
)GML";
        auto gvf = parse_gml(gml_vf);
        suite.check("vf_graph_parsed", static_cast<bool>(gvf));
        suite.check("vf_sink_divergence", gvf && gvf->nodes.size() == 2 &&
            gvf->nodes[0].has_divergence && gvf->nodes[0].divergence == -2.45 &&
            gvf->nodes[0].singularity == "sink");
        suite.check("vf_ordinary_node", gvf && gvf->nodes.size() == 2 &&
            gvf->nodes[1].has_divergence && gvf->nodes[1].divergence == 0.0 &&
            gvf->nodes[1].singularity == "ordinary");
        suite.check("vf_edge_velocity", gvf && gvf->edges.size() == 1 &&
            gvf->edges[0].has_vx && gvf->edges[0].vx == 0.75 &&
            gvf->edges[0].has_vy && gvf->edges[0].vy == -0.50 &&
            gvf->edges[0].has_vz && gvf->edges[0].vz == -0.10);

        // ── Entropic gravity: local_entropy / unruh_temp / bit_density / eh_x/y/z ──
        const char* gml_eg = R"GML(
graph [
  directed 1
  id 1420
  label "Entropic_Well_Test"
  node [ id 200 label "REASON_sink"
         x 0.0 y 0.0 z 0.0
         divergence -4.82 singularity "sink"
         local_entropy 78.5398 unruh_temp 0.1591 bit_density 314.1592 ]
  node [ id 201 label "ELIZA_source"
         x 2.5 y 1.0 z 0.0
         divergence 3.91 singularity "source"
         local_entropy 18.0955 unruh_temp 0.3314 bit_density 72.3822 ]
  edge [ source 201 target 200 verb "@requires" weight 1.0
         vx -0.80 vy -0.40 vz 0.0
         eh_x -1.25 eh_y -0.63 eh_z 0.0 ]
]
)GML";
        auto geg = parse_gml(gml_eg);
        suite.check("eg_graph_parsed", static_cast<bool>(geg));
        suite.check("eg_sink_entropy", geg && geg->nodes.size() == 2 &&
            geg->nodes[0].has_local_entropy && geg->nodes[0].local_entropy == 78.5398 &&
            geg->nodes[0].has_unruh_temp    && geg->nodes[0].unruh_temp    == 0.1591  &&
            geg->nodes[0].has_bit_density   && geg->nodes[0].bit_density   == 314.1592);
        suite.check("eg_source_entropy", geg && geg->nodes.size() == 2 &&
            geg->nodes[1].has_local_entropy && geg->nodes[1].local_entropy == 18.0955 &&
            geg->nodes[1].has_unruh_temp    && geg->nodes[1].unruh_temp    == 0.3314  &&
            geg->nodes[1].has_bit_density   && geg->nodes[1].bit_density   == 72.3822);
        suite.check("eg_edge_eh_gradient", geg && geg->edges.size() == 1 &&
            geg->edges[0].has_eh_x && geg->edges[0].eh_x == -1.25 &&
            geg->edges[0].has_eh_y && geg->edges[0].eh_y == -0.63 &&
            geg->edges[0].has_eh_z && geg->edges[0].eh_z == 0.0);

        // ── Training gradient telemetry: provider_id / parameter_id / training_step / loss_value / grad_norm ──
        const char* gml_tg = R"GML(
graph [
  directed 1
  id 1421
  label "Training_Gradient_Telemetry"
  node [ id 300 label "phi_0_checkpoint"
         x 0.0 y 0.0 z 0.0
         provider_id "VECTOR-u" parameter_id "phi.layer3.attn.W"
         parameter_generation 7 training_step 1024
         loss_value 0.3142 grad_norm 1.4142
         learning_rate 0.0003 gradient_clip 1.0 ]
  node [ id 301 label "phi_1_checkpoint"
         x 1.0 y 0.0 z 0.0
         provider_id "VECTOR-u" parameter_id "phi.layer3.attn.W"
         parameter_generation 8 training_step 1025
         loss_value 0.2718 grad_norm 0.9871
         learning_rate 0.0003 gradient_clip 1.0 ]
  edge [ source 300 target 301
         verb "@delegates_to" weight 1.0
         provider_id "VECTOR-u" parameter_id "phi.layer3.attn.W"
         training_step 1025 loss_value 0.2718 grad_norm 0.9871
         learning_rate 0.0003 gradient_clip 1.0
         field_kind "trajectory" ]
]
)GML";
        auto gtg = parse_gml(gml_tg);
        suite.check("tg_graph_parsed", static_cast<bool>(gtg));
        suite.check("tg_node_provider", gtg && gtg->nodes.size() == 2 &&
            gtg->nodes[0].provider_id  == "VECTOR-u" &&
            gtg->nodes[0].parameter_id == "phi.layer3.attn.W" &&
            gtg->nodes[0].has_parameter_generation && gtg->nodes[0].parameter_generation == 7 &&
            gtg->nodes[0].has_training_step         && gtg->nodes[0].training_step         == 1024);
        suite.check("tg_node_gradient", gtg && gtg->nodes.size() == 2 &&
            gtg->nodes[0].has_loss_value    && gtg->nodes[0].loss_value    == 0.3142 &&
            gtg->nodes[0].has_grad_norm     && gtg->nodes[0].grad_norm     == 1.4142 &&
            gtg->nodes[0].has_learning_rate && gtg->nodes[0].learning_rate == 0.0003 &&
            gtg->nodes[0].has_gradient_clip && gtg->nodes[0].gradient_clip == 1.0);
        suite.check("tg_edge_telemetry", gtg && gtg->edges.size() == 1 &&
            gtg->edges[0].provider_id  == "VECTOR-u" &&
            gtg->edges[0].has_training_step  && gtg->edges[0].training_step  == 1025 &&
            gtg->edges[0].has_loss_value     && gtg->edges[0].loss_value     == 0.2718 &&
            gtg->edges[0].has_grad_norm      && gtg->edges[0].grad_norm      == 0.9871 &&
            gtg->edges[0].field_kind == "trajectory");

        // ── Entropy gradient projection: entropy_spike / entropy_bound / diffusion_beta_inv / entropy_delta ──
        const char* gml_ep = R"GML(
graph [
  directed 1
  id 1402
  label "Entropy_Gradient_Manifold"
  node [ id 10 label "The verified user authenticated successfully."
         x -1.0 y 0.5 z 0.0
         local_entropy 0.15 entropy_spike 0.00
         entropy_bound 4.00 diffusion_beta_inv 0.25 ]
  node [ id 20 label "Initiate conditional multi-task system response pathways."
         x 1.5 y -0.5 z 0.5
         local_entropy 3.82 entropy_spike 3.67
         entropy_bound 4.00 diffusion_beta_inv 0.25 ]
  edge [ source 10 target 20
         verb "@delegates_to" weight 0.75
         field_kind "entropy_gradient"
         eh_x 2.50 eh_y -1.00 eh_z 0.50
         entropy_delta 3.67 diffusion_beta_inv 0.25 ]
]
)GML";
        auto gep = parse_gml(gml_ep);
        suite.check("ep_graph_parsed", static_cast<bool>(gep));
        suite.check("ep_low_entropy_node", gep && gep->nodes.size() == 2 &&
            gep->nodes[0].has_local_entropy     && gep->nodes[0].local_entropy     == 0.15 &&
            gep->nodes[0].has_entropy_spike      && gep->nodes[0].entropy_spike      == 0.00 &&
            gep->nodes[0].has_entropy_bound      && gep->nodes[0].entropy_bound      == 4.00 &&
            gep->nodes[0].has_diffusion_beta_inv && gep->nodes[0].diffusion_beta_inv == 0.25);
        suite.check("ep_high_entropy_node", gep && gep->nodes.size() == 2 &&
            gep->nodes[1].has_local_entropy     && gep->nodes[1].local_entropy     == 3.82 &&
            gep->nodes[1].has_entropy_spike      && gep->nodes[1].entropy_spike      == 3.67 &&
            gep->nodes[1].has_entropy_bound      && gep->nodes[1].entropy_bound      == 4.00);
        suite.check("ep_edge_gradient", gep && gep->edges.size() == 1 &&
            gep->edges[0].field_kind == "entropy_gradient" &&
            gep->edges[0].has_eh_x          && gep->edges[0].eh_x          == 2.50  &&
            gep->edges[0].has_eh_y          && gep->edges[0].eh_y          == -1.00 &&
            gep->edges[0].has_entropy_delta      && gep->edges[0].entropy_delta      == 3.67 &&
            gep->edges[0].has_diffusion_beta_inv && gep->edges[0].diffusion_beta_inv == 0.25);

        // ── Geodesic entropy + Triangular entropy + simplicial edge fields ────
        const char* gml_se = R"GML(
graph [
  directed 1
  id 1422
  label "Simplicial_Entropy_Test"
  node [ id 400 label "REASON_hub"
         x 0.0 y 0.0 z 0.0
         geodesic_entropy 1.142 triangular_scale 0.085
         local_entropy 78.5398 divergence -4.82 singularity "sink" ]
  node [ id 401 label "ELIZA_branch"
         x 2.5 y 1.5 z 0.0
         geodesic_entropy 2.718 triangular_scale 0.391
         local_entropy 18.0955 divergence 3.91 singularity "source" ]
  node [ id 402 label "ADVISOR_vertex"
         x -1.0 y 1.5 z 0.0
         geodesic_entropy 1.886 triangular_scale 0.152
         local_entropy 32.1750 divergence 0.0 singularity "ordinary" ]
  edge [ source 401 target 400 verb "@requires" weight 1.0
         triangle_id 1001 simplex_weight 0.75 vx -0.80 vy -0.40 vz 0.0 ]
  edge [ source 402 target 400 verb "@defines"  weight 0.9
         triangle_id 1001 simplex_weight 0.65 vx 0.30 vy -0.60 vz 0.0 ]
  edge [ source 401 target 402 verb "@part_of"  weight 0.8
         triangle_id 1001 simplex_weight 0.55 vx -0.50 vy 0.10 vz 0.0 ]
]
)GML";
        auto gse = parse_gml(gml_se);
        suite.check("se_graph_parsed", static_cast<bool>(gse));
        suite.check("se_hub_geodesic", gse && gse->nodes.size() == 3 &&
            gse->nodes[0].has_geodesic_entropy && gse->nodes[0].geodesic_entropy == 1.142 &&
            gse->nodes[0].has_triangular_scale  && gse->nodes[0].triangular_scale  == 0.085);
        suite.check("se_branch_geodesic", gse && gse->nodes.size() == 3 &&
            gse->nodes[1].has_geodesic_entropy && gse->nodes[1].geodesic_entropy == 2.718 &&
            gse->nodes[1].has_triangular_scale  && gse->nodes[1].triangular_scale  == 0.391);
        suite.check("se_simplex_edges", gse && gse->edges.size() == 3 &&
            gse->edges[0].has_triangle_id   && gse->edges[0].triangle_id   == 1001 &&
            gse->edges[0].has_simplex_weight && gse->edges[0].simplex_weight == 0.75 &&
            gse->edges[1].has_triangle_id   && gse->edges[1].triangle_id   == 1001 &&
            gse->edges[1].has_simplex_weight && gse->edges[1].simplex_weight == 0.65 &&
            gse->edges[2].has_triangle_id   && gse->edges[2].triangle_id   == 1001 &&
            gse->edges[2].has_simplex_weight && gse->edges[2].simplex_weight == 0.55);

        // ── Tensor bundle projection: rank(p,q) / smooth section / g_ij / Jacobian ──
        const char* gml_tb = R"GML(
graph [
  directed 1
  id 1501
  label "Tensor_Bundle_Projection_Field"
  node [ id 1 label "Tensor_Source"
         x 0.0 y 0.0 z 0.0
         tensor_rank_p 1 tensor_rank_q 1 is_smooth_section 1
         jacobian_determinant 1.0834 riemann_curvature_scalar -0.412 ]
  node [ id 2 label "Tensor_Target"
         x 1.0 y 0.0 z 0.0
         tensor_rank_p 1 tensor_rank_q 1 is_smooth_section 1 ]
  edge [ source 1 target 2
         label "coordinate_transition"
         verb "@requires" field_kind "tensor_field" weight 1.0
         g_00 -0.8524 g_01 0.0125 g_10 0.0125 g_11 1.1486
         jacobian_w 1.0834 vx 0.1 vy 0.0 vz 0.0 ]
]
)GML";
        auto gtb = parse_gml(gml_tb);
        suite.check("tb_graph_parsed", static_cast<bool>(gtb));
        suite.check("tb_node_rank", gtb && gtb->nodes.size() == 2 &&
            gtb->nodes[0].has_tensor_rank_p && gtb->nodes[0].tensor_rank_p == 1 &&
            gtb->nodes[0].has_tensor_rank_q && gtb->nodes[0].tensor_rank_q == 1 &&
            gtb->nodes[0].has_is_smooth_section && gtb->nodes[0].is_smooth_section == 1);
        suite.check("tb_node_curvature_jacobian", gtb && gtb->nodes.size() == 2 &&
            gtb->nodes[0].has_jacobian_determinant && gtb->nodes[0].jacobian_determinant == 1.0834 &&
            gtb->nodes[0].has_riemann_curvature_scalar && gtb->nodes[0].riemann_curvature_scalar == -0.412);
        suite.check("tb_edge_metric_tensor", gtb && gtb->edges.size() == 1 &&
            gtb->edges[0].field_kind == "tensor_field" &&
            gtb->edges[0].has_g_00 && gtb->edges[0].g_00 == -0.8524 &&
            gtb->edges[0].has_g_01 && gtb->edges[0].g_01 == 0.0125 &&
            gtb->edges[0].has_g_10 && gtb->edges[0].g_10 == 0.0125 &&
            gtb->edges[0].has_g_11 && gtb->edges[0].g_11 == 1.1486 &&
            gtb->edges[0].has_jacobian_w && gtb->edges[0].jacobian_w == 1.0834);
    });
}
