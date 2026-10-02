#pragma once
#include "../kuhul/phase_engine.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace khanary::world {

// MetricTensor — explicit (0,2) tensor field on the SVG-3D base manifold.
//
// A symmetric, positive-definite bilinear form g: V × V → ℝ that defines
// inner products on tangent vectors at each base point.
//
// The C6 arc_weight is a derived quantity under this metric:
//
//   arc_weight(θ₁, θ₂) = g(e_θ₁, e_θ₂)
//
// where e_θ = (cos θ, sin θ, 0) is the unit phase direction on the C6 circle
// embedded in the xy-plane of the SVG-3D manifold.  With g = I (Euclidean)
// this recovers cos(θ₂ − θ₁) exactly.
struct MetricTensor {
    // g[i][j] — 3×3 symmetric matrix.  Row-major; only upper triangle is
    // canonical; g[j][i] must equal g[i][j] at all times.
    float g[3][3]{};

    // Bilinear form: g(v, w) = Σ_{i,j} v_i g_{ij} w_j
    float inner_product(float vx, float vy, float vz,
                        float wx, float wy, float wz) const noexcept {
        return vx * (g[0][0]*wx + g[0][1]*wy + g[0][2]*wz)
             + vy * (g[1][0]*wx + g[1][1]*wy + g[1][2]*wz)
             + vz * (g[2][0]*wx + g[2][1]*wy + g[2][2]*wz);
    }

    // Squared norm: g(v, v)
    float norm_sq(float vx, float vy, float vz) const noexcept {
        return inner_product(vx, vy, vz, vx, vy, vz);
    }

    // arc_weight derived from metric — replaces the magic cos(Δθ) formula.
    // Phase directions are unit vectors on the C6 circle in the xy-plane.
    double arc_weight(kuhul::Phase from, kuhul::Phase to) const {
        const double af = kuhul::phase_angle(from);
        const double at = kuhul::phase_angle(to);
        const float fx = static_cast<float>(std::cos(af));
        const float fy = static_cast<float>(std::sin(af));
        const float tx = static_cast<float>(std::cos(at));
        const float ty = static_cast<float>(std::sin(at));
        return static_cast<double>(inner_product(fx, fy, 0.0f, tx, ty, 0.0f));
    }

    // Verify the metric is symmetric (g[i][j] == g[j][i]) and positive-
    // definite (all 3×3 leading principal minors > 0, Sylvester's criterion).
    // Returns true if both conditions hold within tolerance eps.
    bool is_valid(float eps = 1e-6f) const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (std::abs(g[i][j] - g[j][i]) > eps)
                    return false;
        // Sylvester: all principal minors must be positive
        if (g[0][0] <= eps) return false;
        const float m2 = g[0][0]*g[1][1] - g[0][1]*g[1][0];
        if (m2 <= eps) return false;
        const float det = g[0][0]*(g[1][1]*g[2][2] - g[1][2]*g[2][1])
                        - g[0][1]*(g[1][0]*g[2][2] - g[1][2]*g[2][0])
                        + g[0][2]*(g[1][0]*g[2][1] - g[1][1]*g[2][0]);
        return det > eps;
    }

    // ── Factory methods ────────────────────────────────────────────────────────

    // Euclidean (identity) metric — the C6 phase circle metric on ℝ³.
    // Recovers arc_weight = cos(Δθ) exactly.  This is the Phase 1 metric;
    // non-Euclidean deformations come later.
    static MetricTensor euclidean() {
        MetricTensor m;
        m.g[0][0] = 1.0f;
        m.g[1][1] = 1.0f;
        m.g[2][2] = 1.0f;
        return m;
    }

    // Diagonal metric with per-axis scale factors sx, sy, sz.
    // Useful when the semantic axes have different physical units
    // (e.g., time in seconds vs. frequency in Hz after denormalisation).
    static MetricTensor diagonal(float sx, float sy, float sz) {
        MetricTensor m;
        m.g[0][0] = sx * sx;
        m.g[1][1] = sy * sy;
        m.g[2][2] = sz * sz;
        return m;
    }

    // Conformal deformation from a geodesic entropy scalar.
    //
    // g_ij(x) = e^{2Φ} δ_ij   where Φ = geodesic_entropy at node x.
    //
    // A conformal metric preserves angles but varies scale.  High entropy
    // means large local scale (geodesics spread); low entropy means small
    // scale (geodesics converge).  The conformal factor e^{2Φ} is uniform
    // across all three axes, so the metric remains diagonal.
    //
    // Christoffel symbols for this metric are non-zero when ∇Φ ≠ 0:
    //   Γ^k_ij = δ^k_j ∂_i Φ + δ^k_i ∂_j Φ − δ_ij ∂_k Φ
    // (computed by ChristoffelSymbols::from_conformal_gradient in christoffel.h)
    static MetricTensor conformal_from_entropy(float phi) {
        const float e_phi = std::exp(phi);   // e^Φ
        return diagonal(e_phi, e_phi, e_phi); // g[i][i] = e^{2Φ}
    }
};

} // namespace khanary::world
