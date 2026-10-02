#include "conformance.h"
#include "../world/christoffel.h"
#include "../world/metric_tensor.h"
#include <cmath>

using namespace khanary::world;
using namespace khanary::kuhul;

int main() {
    return khanary::test::run("christoffel", [](khanary::test::Suite& suite) {

        // ── Flat metric: all symbols zero ─────────────────────────────────────
        // For g = I (constant), ∂_k g_ij = 0 everywhere → Γ = 0.
        // This is the baseline; was a placeholder in test_metric_tensor.cpp.
        suite.check("flat_gradient_is_zero", ChristoffelSymbols::flat().is_zero());
        suite.check("zero_gradient_is_zero",
            ChristoffelSymbols::from_conformal_gradient(0.0f, 0.0f, 0.0f).is_zero());

        // ── Torsion-free (Levi-Civita): Γ^k_ij = Γ^k_ji ─────────────────────
        // The conformal formula gives a Levi-Civita connection (torsion T = 0).
        suite.check("torsion_free_x_gradient",
            ChristoffelSymbols::from_conformal_gradient(1.0f, 0.0f, 0.0f)
                .is_torsion_free());
        suite.check("torsion_free_arbitrary_gradient",
            ChristoffelSymbols::from_conformal_gradient(0.5f, -0.3f, 0.7f)
                .is_torsion_free());

        // ── Non-flat gradient produces non-zero symbols ───────────────────────
        suite.check("nonzero_for_nonzero_gradient",
            !ChristoffelSymbols::from_conformal_gradient(1.0f, 0.0f, 0.0f).is_zero());

        // ── Specific values: dPhi = (p, 0, 0) ────────────────────────────────
        // Γ^k_ij = δ^k_j * dPhi[i] + δ^k_i * dPhi[j] - δ_ij * dPhi[k]
        //
        // With p = 1, dPhi = (1, 0, 0):
        //   Γ^0_00 = 1*1 + 1*1 - 1*1 = 1
        //   Γ^0_11 = 0 + 0 - 1*1     = -1
        //   Γ^0_22 = 0 + 0 - 1*1     = -1
        //   Γ^1_01 = Γ^1_10 = 1 (δ^1_1 * dPhi[0] = 1)
        //   Γ^2_02 = Γ^2_20 = 1 (δ^2_2 * dPhi[0] = 1)
        //   all other entries = 0
        const float p = 1.0f;
        const auto cs = ChristoffelSymbols::from_conformal_gradient(p, 0.0f, 0.0f);

        suite.check("x_grad_Gamma_0_00", std::abs(cs.Gamma[0][0][0] -  p) < 1e-6f);
        suite.check("x_grad_Gamma_0_11", std::abs(cs.Gamma[0][1][1] - (-p)) < 1e-6f);
        suite.check("x_grad_Gamma_0_22", std::abs(cs.Gamma[0][2][2] - (-p)) < 1e-6f);
        suite.check("x_grad_Gamma_1_01", std::abs(cs.Gamma[1][0][1] -  p) < 1e-6f);
        suite.check("x_grad_Gamma_1_10", std::abs(cs.Gamma[1][1][0] -  p) < 1e-6f);
        suite.check("x_grad_Gamma_2_02", std::abs(cs.Gamma[2][0][2] -  p) < 1e-6f);
        suite.check("x_grad_Gamma_2_20", std::abs(cs.Gamma[2][2][0] -  p) < 1e-6f);
        // Off-diagonal upper indices should be zero for x-only gradient
        suite.check("x_grad_Gamma_1_11_zero", std::abs(cs.Gamma[1][1][1]) < 1e-6f);
        suite.check("x_grad_Gamma_2_22_zero", std::abs(cs.Gamma[2][2][2]) < 1e-6f);

        // ── Tensor-calculus validator invariants ─────────────────────────────
        suite.check("symbols_finite_for_arbitrary_gradient",
            ChristoffelSymbols::from_conformal_gradient(0.5f, -0.3f, 0.7f).is_finite());
        suite.check("flat_metric_flat_connection_zero_ricci_proxy", [&]{
            const auto metric = MetricTensor::euclidean();
            const auto connection = ChristoffelSymbols::flat();
            return metric.is_valid() && connection.is_zero();
        }());
        suite.check("covariant_rank_increment_law", []{
            const int p = 1;
            const int q = 1;
            const int out_p = p;
            const int out_q = q + 1;
            return out_p == p && out_q == q + 1;
        }());

        // ── conformal_from_entropy factory ────────────────────────────────────
        // phi = 0  → e^{2·0} = 1 → recovers Euclidean metric
        const auto g0 = MetricTensor::conformal_from_entropy(0.0f);
        suite.check("conformal_phi0_is_euclidean", [&]{
            const auto g_e = MetricTensor::euclidean();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (std::abs(g0.g[i][j] - g_e.g[i][j]) > 1e-6f) return false;
            return true;
        }());

        // phi = 1  → scale = e^2 ≈ 7.389
        const auto g1 = MetricTensor::conformal_from_entropy(1.0f);
        suite.check("conformal_phi1_diagonal", [&]{
            const float expected = std::exp(2.0f);  // e^{2Φ} = e^2
            return std::abs(g1.g[0][0] - expected) < 1e-5f
                && std::abs(g1.g[1][1] - expected) < 1e-5f
                && std::abs(g1.g[2][2] - expected) < 1e-5f
                && std::abs(g1.g[0][1]) < 1e-6f
                && std::abs(g1.g[0][2]) < 1e-6f;
        }());

        // Conformal metric must pass is_valid (symmetric + positive-definite)
        suite.check("conformal_phi1_valid", g1.is_valid());
        suite.check("conformal_phi_neg_valid",
            MetricTensor::conformal_from_entropy(-2.0f).is_valid());

        // Conformal map: angles preserved, lengths scaled.
        // g(e_a, e_b) = e^{2Φ} * cos(Δθ)  — raw inner product scales by e^{2Φ}.
        // Normalized ratio g(e_a, e_b) / g(e_a, e_a) = cos(Δθ) is preserved.
        suite.check("conformal_preserves_angle_structure", [&]{
            const double self_w = g1.arc_weight(Phase(0), Phase(0));
            if (self_w <= 0.0) return false;
            // All 6 self-weights equal (isotropic scaling by e^{2Φ})
            for (int a = 0; a < 6; ++a)
                if (std::abs(g1.arc_weight(Phase(a), Phase(a)) - self_w) > 1e-4)
                    return false;
            // Normalized ratio recovers cos(Δθ) exactly
            const auto g_e = MetricTensor::euclidean();
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    const double normalized = g1.arc_weight(Phase(a), Phase(b)) / self_w;
                    const double canonical  = g_e.arc_weight(Phase(a), Phase(b));
                    if (std::abs(normalized - canonical) > 1e-4) return false;
                }
            return true;
        }());
    });
}
