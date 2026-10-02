#pragma once
#include "metric_tensor.h"
#include <cmath>

namespace khanary::world {

// ChristoffelSymbols — explicit (1,2) tensor of connection coefficients.
//
// Γ^k_ij are the Christoffel symbols of the second kind for the Levi-Civita
// connection on the SVG-3D manifold.  They encode how basis vectors change
// as you move through the manifold; the geodesic equation is:
//
//   d²x^k/dτ² + Γ^k_ij (dx^i/dτ)(dx^j/dτ) = 0
//
// For a flat (Euclidean) metric, all Γ = 0 (constant metric, no curvature).
// For the conformal metric g_ij = e^{2Φ} δ_ij the closed-form formula is:
//
//   Γ^k_ij = δ^k_j ∂_i Φ + δ^k_i ∂_j Φ − δ_ij ∂_k Φ
//
// This only depends on ∇Φ, not on the conformal factor itself — the factor
// cancels with the inverse metric.
struct ChristoffelSymbols {
    // Gamma[k][i][j] = Γ^k_ij.  Upper index first, matching standard notation.
    float Gamma[3][3][3]{};

    // Compute Γ^k_ij from the entropy gradient ∇Φ = (∂_x Φ, ∂_y Φ, ∂_z Φ).
    //
    // Formula: Γ^k_ij = δ^k_j ∂_i Φ + δ^k_i ∂_j Φ − δ_ij ∂_k Φ
    //
    // This is exact for any conformal metric g = e^{2Φ} δ regardless of Φ.
    // Pass the finite-difference gradient of geodesic_entropy across
    // neighbouring FieldNodes.
    static ChristoffelSymbols from_conformal_gradient(
        float dPhi_dx, float dPhi_dy, float dPhi_dz) noexcept
    {
        ChristoffelSymbols cs;
        const float dPhi[3] = { dPhi_dx, dPhi_dy, dPhi_dz };
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    cs.Gamma[k][i][j] =
                        (k == j ? 1.0f : 0.0f) * dPhi[i]
                      + (k == i ? 1.0f : 0.0f) * dPhi[j]
                      - (i == j ? 1.0f : 0.0f) * dPhi[k];
        return cs;
    }

    // Convenience: zero symbols for a flat (Euclidean) manifold.
    // Equivalent to from_conformal_gradient(0, 0, 0).
    static ChristoffelSymbols flat() noexcept { return ChristoffelSymbols{}; }

    // True when all 27 coefficients are finite IEEE-754 values.
    bool is_finite() const noexcept {
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (!std::isfinite(Gamma[k][i][j])) return false;
        return true;
    }

    // True when all 27 coefficients are within eps of zero.
    bool is_zero(float eps = 1e-6f) const noexcept {
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (std::abs(Gamma[k][i][j]) > eps) return false;
        return true;
    }

    // Torsion-free (Levi-Civita): Γ^k_ij == Γ^k_ji for all k, i, j.
    bool is_torsion_free(float eps = 1e-6f) const noexcept {
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (std::abs(Gamma[k][i][j] - Gamma[k][j][i]) > eps)
                        return false;
        return true;
    }
};

} // namespace khanary::world
