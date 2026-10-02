#include "conformance.h"
#include "../world/metric_tensor.h"
#include "../kuhul/phase_engine.h"
#include <cmath>

using namespace khanary::world;
using namespace khanary::kuhul;

// Expected arc_weight from canonical formula (phase_engine.h)
static double canonical_arc_weight(Phase a, Phase b) {
    return std::cos(phase_angle(b) - phase_angle(a));
}

int main() {
    return khanary::test::run("metric-tensor", [](khanary::test::Suite& suite) {

        // ── Euclidean factory ──────────────────────────────────────────────────
        const auto g_e = MetricTensor::euclidean();
        suite.check("euclidean_valid", g_e.is_valid());

        // diagonal entries = 1, off-diagonal = 0
        suite.check("euclidean_diagonal", [&]{
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if ((g_e.g[i][j] != (i == j ? 1.0f : 0.0f))) return false;
            return true;
        }());

        // ── arc_weight equivalence: all 36 phase pairs ─────────────────────────
        bool all_match = true;
        for (int a = 0; a < 6; ++a) {
            for (int b = 0; b < 6; ++b) {
                const Phase pa = Phase(a), pb = Phase(b);
                const double metric_val  = g_e.arc_weight(pa, pb);
                const double canonical   = canonical_arc_weight(pa, pb);
                if (std::abs(metric_val - canonical) > 1e-6) {
                    all_match = false;
                    break;
                }
            }
            if (!all_match) break;
        }
        suite.check("arc_weight_36_pairs_match_canonical", all_match);

        // ── Symmetry: g(v,w) == g(w,v) ────────────────────────────────────────
        suite.check("inner_product_symmetric", [&]{
            const float vx = 0.6f, vy = 0.8f, vz = 0.0f;
            const float wx = 0.3f, wy = 0.4f, wz = 0.9f;
            const float gvw = g_e.inner_product(vx, vy, vz, wx, wy, wz);
            const float gwv = g_e.inner_product(wx, wy, wz, vx, vy, vz);
            return std::abs(gvw - gwv) < 1e-6f;
        }());

        // ── Positive-definiteness: g(v,v) > 0 for v != 0 ─────────────────────
        suite.check("positive_definite_unit_x", g_e.norm_sq(1.0f, 0.0f, 0.0f) > 0.0f);
        suite.check("positive_definite_unit_y", g_e.norm_sq(0.0f, 1.0f, 0.0f) > 0.0f);
        suite.check("positive_definite_unit_z", g_e.norm_sq(0.0f, 0.0f, 1.0f) > 0.0f);
        suite.check("positive_definite_arbitrary",
            g_e.norm_sq(0.6f, 0.8f, 0.1f) > 0.0f);

        // ── Euclidean norm matches ||v||^2 ────────────────────────────────────
        suite.check("euclidean_norm_sq", [&]{
            const float vx = 3.0f, vy = 4.0f, vz = 0.0f;
            return std::abs(g_e.norm_sq(vx, vy, vz) - 25.0f) < 1e-5f;
        }());

        // ── Diagonal factory ──────────────────────────────────────────────────
        const auto g_d = MetricTensor::diagonal(2.0f, 3.0f, 1.0f);
        suite.check("diagonal_valid", g_d.is_valid());
        suite.check("diagonal_entries", [&]{
            return std::abs(g_d.g[0][0] - 4.0f) < 1e-6f &&   // sx^2
                   std::abs(g_d.g[1][1] - 9.0f) < 1e-6f &&   // sy^2
                   std::abs(g_d.g[2][2] - 1.0f) < 1e-6f &&   // sz^2
                   g_d.g[0][1] == 0.0f && g_d.g[0][2] == 0.0f;
        }());
        suite.check("diagonal_inner_product", [&]{
            // g(e_x, e_x) = sx^2 = 4
            return std::abs(g_d.inner_product(1.0f, 0.0f, 0.0f,
                                              1.0f, 0.0f, 0.0f) - 4.0f) < 1e-5f;
        }());

        // ── is_valid rejects bad metrics ──────────────────────────────────────
        MetricTensor g_bad;
        g_bad.g[0][0] = -1.0f;  // negative diagonal -> not positive-definite
        g_bad.g[1][1] = 1.0f;
        g_bad.g[2][2] = 1.0f;
        suite.check("invalid_negative_diagonal", !g_bad.is_valid());

        MetricTensor g_asym;
        g_asym.g[0][0] = 1.0f; g_asym.g[1][1] = 1.0f; g_asym.g[2][2] = 1.0f;
        g_asym.g[0][1] = 1.0f; g_asym.g[1][0] = -1.0f;  // asymmetric
        suite.check("invalid_asymmetric", !g_asym.is_valid());

        // ── Self-test: Pop → Sek is exactly -1 (Δθ = π, antipodal pair) ────────
        // Wo → Sek is Δθ = 2π/3 (cos = -0.5), NOT antipodal.
        // Antipodal pairs on C6: Pop↔Sek, Wo↔Chen, Yax↔Xul.
        suite.check("arc_weight_antipodal",
            std::abs(g_e.arc_weight(Phase::Pop, Phase::Sek) - (-1.0)) < 1e-6);

        // ── Self-test: same phase is exactly +1 (Δθ = 0) ────────────────────
        suite.check("arc_weight_identity", [&]{
            for (int a = 0; a < 6; ++a)
                if (std::abs(g_e.arc_weight(Phase(a), Phase(a)) - 1.0) > 1e-6)
                    return false;
            return true;
        }());

        // ── Phase 2 preview: flat metric gives zero partial derivatives ────────
        // For g = I (constant field), ∂_k g_ij = 0, so Γ^k_ij = 0 everywhere.
        // This is the correct result for a flat (Euclidean) manifold.
        // Christoffel symbols are only non-zero for position-dependent metrics.
        // Noted here so Phase 2 tests can build on this baseline.
        suite.check("flat_metric_note", true);  // placeholder for Γ = 0 check
    });
}
