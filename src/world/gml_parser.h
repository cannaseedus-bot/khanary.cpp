#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace khanary::world {

struct GmlNode {
    int64_t id    = 0;
    std::string label;
    double x = 0.0, y = 0.0, z = 0.0;
    double confidence = 1.0;
    double static_weight = 1.0;
    double health = 1.0;
    double prior_mass = 0.0;
    double risk_bound = 0.0;
    int64_t base_version = 0;
    std::string mutation;
    // vector field
    double divergence = 0.0;
    std::string singularity;        // "source" | "sink" | "saddle" | "ordinary"
    // entropic gravity (Verlinde-Jacobson)
    double local_entropy    = 0.0;  // H(x): Shannon uncertainty; feeds ∇H entropic force
    double unruh_temp       = 0.0;  // T: Unruh surface temperature; F = T∇H
    double bit_density      = 0.0;  // N: holographic degrees of freedom (Bekenstein-Hawking)
    double geodesic_entropy = 0.0;  // H_G: shortest-path distribution entropy
    double triangular_scale = 0.0;  // H_Δ: 3-node simplicial complex error metric
    double circumradius_R   = 0.0;  // R: circumcircle horizon radius for local triangle
    double inradius_r       = 0.0;  // r: incircle compression radius for local triangle
    // quantum graph
    std::string boundary_type;      // "kirchhoff" | "dirichlet" | "robin"
    double vertex_pot  = 0.0;       // localized vertex scattering potential
    double phase_shift = 0.0;       // localized initial phase angle φ
    // text structure
    std::string text_boundary;      // "sentence" | "paragraph" | "article" | "section" | "none"
    std::string boundary_token;
    // curve / trajectory
    std::string curve_id;
    double parameter_t      = 0.0;  // curve parameter t for γ(t)
    double arc_length       = 0.0;  // accumulated arc length s along γ
    double curvature        = 0.0;  // local curvature κ(t)
    double scalar_potential = 0.0;  // projection scalar f(x)
    // training gradient telemetry
    std::string provider_id;
    std::string parameter_id;
    int64_t parameter_generation = 0;
    int64_t training_step        = 0;
    double loss_value    = 0.0;
    double grad_norm     = 0.0;
    double learning_rate = 0.0;
    double gradient_clip = 0.0;
    // entropy gradient telemetry
    double entropy_spike      = 0.0; // ΔH: structural uncertainty jump
    double entropy_bound      = 0.0; // maximum admitted entropy
    double diffusion_beta_inv = 0.0; // β⁻¹ entropy diffusion scale
    // tensor bundle telemetry
    int64_t tensor_rank_p = 0;        // contravariant rank p
    int64_t tensor_rank_q = 0;        // covariant rank q
    int64_t is_smooth_section = 0;    // 1 when this node denotes a smooth section
    double riemann_curvature_scalar = 0.0;
    double jacobian_determinant = 1.0;

    bool has_x = false, has_y = false, has_z = false;
    bool has_prior_mass     = false;
    bool has_risk_bound     = false;
    bool has_base_version   = false;
    bool has_divergence     = false;
    bool has_local_entropy    = false;
    bool has_unruh_temp       = false;
    bool has_bit_density      = false;
    bool has_geodesic_entropy = false;
    bool has_triangular_scale = false;
    bool has_circumradius_R   = false;
    bool has_inradius_r       = false;
    bool has_vertex_pot     = false;
    bool has_phase_shift    = false;
    bool has_parameter_t          = false;
    bool has_arc_length           = false;
    bool has_curvature            = false;
    bool has_scalar_potential     = false;
    bool has_parameter_generation = false;
    bool has_training_step        = false;
    bool has_loss_value           = false;
    bool has_grad_norm            = false;
    bool has_learning_rate        = false;
    bool has_gradient_clip        = false;
    bool has_entropy_spike        = false;
    bool has_entropy_bound        = false;
    bool has_diffusion_beta_inv   = false;
    bool has_tensor_rank_p        = false;
    bool has_tensor_rank_q        = false;
    bool has_is_smooth_section    = false;
    bool has_riemann_curvature_scalar = false;
    bool has_jacobian_determinant = false;
};

struct GmlEdge {
    int64_t id = 0;
    int64_t source = 0;
    int64_t target = 0;
    std::string label;
    std::string verb;
    double weight   = 1.0;
    int64_t dual_twin = 0;
    double loss_cost  = 0.0;
    std::string mutation;
    // vector field
    double vx = 0.0, vy = 0.0, vz = 0.0;
    // entropic gradient
    double eh_x = 0.0, eh_y = 0.0, eh_z = 0.0;
    // metric / wave
    double length       = 0.0;
    double potential    = 0.0;
    double refractive   = 1.0;      // propagation slowdown; 1.0 = free
    double wave_number  = 0.0;
    double frequency    = 0.0;
    double amplitude_re = 0.0;
    double amplitude_im = 0.0;
    // curve / trajectory
    std::string curve_id;
    double t0 = 0.0;
    double t1 = 0.0;
    std::string field_kind;         // "flow" | "trajectory" | "euclidean_gradient" | "entropy_gradient" | ...
    double curl_x = 0.0, curl_y = 0.0, curl_z = 0.0;
    double potential_delta = 0.0;   // f(γ(b)) - f(γ(a)) for conservative fields
    // training gradient telemetry
    std::string provider_id;
    std::string parameter_id;
    int64_t training_step = 0;
    double loss_value     = 0.0;
    double grad_norm      = 0.0;
    double learning_rate  = 0.0;
    double gradient_clip  = 0.0;
    // entropy gradient telemetry
    double entropy_delta      = 0.0; // H(target) - H(source), signed
    double diffusion_beta_inv = 0.0; // β⁻¹ entropy diffusion scale
    // simplicial complex
    int64_t triangle_id    = 0;      // verified 3-node loop token
    double  simplex_weight = 0.0;    // local geometric thickness of the triangular face
    // tensor bundle local coordinate transition
    double g_00 = 0.0, g_01 = 0.0, g_10 = 0.0, g_11 = 0.0;
    double jacobian_w = 1.0;         // density/weight factor for Jacobian transport

    bool has_id        = false;
    bool has_weight    = false;
    bool has_dual_twin = false;
    bool has_loss_cost = false;
    bool has_vx = false, has_vy = false, has_vz = false;
    bool has_eh_x = false, has_eh_y = false, has_eh_z = false;
    bool has_length       = false;
    bool has_potential    = false;
    bool has_refractive   = false;
    bool has_wave_number  = false;
    bool has_frequency    = false;
    bool has_amplitude_re = false;
    bool has_amplitude_im = false;
    bool has_t0 = false, has_t1 = false;
    bool has_curl_x = false, has_curl_y = false, has_curl_z = false;
    bool has_potential_delta  = false;
    bool has_training_step    = false;
    bool has_loss_value       = false;
    bool has_grad_norm        = false;
    bool has_learning_rate    = false;
    bool has_gradient_clip    = false;
    bool has_entropy_delta      = false;
    bool has_diffusion_beta_inv = false;
    bool has_triangle_id        = false;
    bool has_simplex_weight     = false;
    bool has_g_00 = false, has_g_01 = false, has_g_10 = false, has_g_11 = false;
    bool has_jacobian_w = false;
};

struct GmlGraph {
    bool directed = false;
    int64_t id = 0;
    std::string label;
    std::string comment;
    std::vector<GmlNode> nodes;
    std::vector<GmlEdge> edges;
};

// Returns nullptr on parse failure.
std::unique_ptr<GmlGraph> parse_gml(const std::string& text);
std::unique_ptr<GmlGraph> parse_gml_file(const std::string& path);

} // namespace khanary::world
