#pragma once
//
// Micronaut State Algebra — A_mu = K ⊕ V_mu, the square-zero extension.
//
// A Micronaut's local representational state is a module V_mu: features,
// evidence, candidate coordinates. V_mu has no internal product of its own. The
// trivial (square-zero) extension adds the scalar field back:
//
//     A = K ⊕ V,      (a,v)(b,w) = (ab, aw + bv),      v.w = 0
//
// There is no vw term. The result is nevertheless a unital associative
// commutative algebra, because the identity comes entirely from K:
//
//     1_A = (1, 0),   0_A = (0, 0)
//     (1,0)(a,v) = (a, v) = (a,v)(1,0)
//
// Two consequences this header exists to make checkable:
//
//   1. V is an ideal, and its square is zero. Every subspace W ⊆ V is closed
//      under scalar multiplication, so (0,W) is an ideal of A contained in V —
//      the correspondence Sub_K(V) ≅ {I ⊴ A | I ⊆ V} preserves inclusion, sums
//      and intersections. That is the formal classification of a Micronaut's
//      state compartments: evidence, candidate features and uncertainty are
//      submodules, hence ideals.
//
//   2. ZERO MULTIPLICATION IS NOT ZERO COMPUTATION. (0,v1)(0,v2) = 0 says
//      nothing about <v1,v2>, cos(v1,v2) or M·v1. Those are EXTERNAL bilinear
//      forms, similarity functions and module actions, owned by different
//      contracts (DOT_PRODUCT is an inner product space, not this algebra).
//      Conflating the two is the error this layer is shaped to prevent.
//
// A_mu is a REPRESENTATIONAL algebra. It organizes state; it grants no execution
// authority and it is not a fold. Execution authority is KFA + XCFE's business.
//
#include "algebra_contract.h"

#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace khanary::kuhul::statealgebra {

// An element (a, v) of A = K ⊕ V.
struct Pair {
    double scalar = 0.0;               // the K component: coefficients, identity
    std::vector<double> vector;        // the V component: local features

    bool operator==(const Pair& other) const {
        return scalar == other.scalar && vector == other.vector;
    }
    bool operator!=(const Pair& other) const { return !(*this == other); }
};

// ── The algebra ──────────────────────────────────────────────────────────────

Pair pair(double scalar, std::vector<double> vector);
Pair zero(std::size_t dimension);
Pair one(std::size_t dimension);  // 1_A = (1, 0)

Pair add(const Pair& a, const Pair& b);
Pair negate(const Pair& a);
Pair mul(const Pair& a, const Pair& b);  // (ab, aw + bv) — no vw term

// (a,v) is invertible exactly when a != 0, with inverse (1/a, -v/a^2).
// (0,v) squares to zero and has no inverse, which is why the structure declares
// invertible = false.
std::optional<Pair> inverse(const Pair& a);

// True when the V-component product vanishes: mul((0,v),(0,w)) == 0.
bool square_is_zero(const std::vector<double>& v, const std::vector<double>& w);

// ── Compartments: submodules of V are ideals of A ────────────────────────────
//
// A compartment subspace is modelled as a set of coordinate indices. That is
// enough to exercise the correspondence exactly: coordinate subspaces are
// closed under scalar multiplication, sums are unions and intersections are
// intersections, so inclusion, sum and intersection are preserved by the
// v |-> (0,v) embedding.

using Subspace = std::set<std::size_t>;

Subspace span_of(std::initializer_list<std::size_t> coordinates);
Subspace sum(const Subspace& a, const Subspace& b);
Subspace intersection(const Subspace& a, const Subspace& b);
bool includes(const Subspace& a, const Subspace& b);  // b ⊆ a

// Embeds W into A as the ideal (0, W).
Pair embed(const Subspace& w, std::size_t dimension);

// True when (0,W) is an ideal of A: closed under multiplication by every element
// sampled from `probes` on both sides.
bool is_ideal(const Subspace& w, const std::vector<Pair>& probes, std::size_t dimension);

// Projects an element onto the compartment W (the K component drops out).
Pair project(const Pair& a, const Subspace& w, std::size_t dimension);

// The external bilinear form <v,w>. This is NOT the algebra product: it is
// DOT_PRODUCT's inner product, and it is generally nonzero exactly where the
// square-zero product is zero.
double external_inner_product(const std::vector<double>& v, const std::vector<double>& w);

// ── Cl(2,0) — 2D Euclidean Geometric Algebra ─────────────────────────────────
//
// Basis: {1, e1, e2, e12}   e1²=+1, e2²=+1, e12²=-1
//        e1*e2=e12, e2*e1=-e12
//
// Three important distinctions:
//
// 1. GA_PHASE_PRODUCT (e_j*e_i) returns the full-angle product whose scalar
//    part equals arc_weight(from→to)=cos(Δθ). GA_PHASE_ROTATION returns the
//    half-angle rotor for sandwich rotation. Using the full-angle product in a
//    sandwich rotates by 2Δθ, not Δθ.
//
// 2. The fold_engine vertical-fold adjacency constraint (dist==1) cannot be
//    enforced by norm(rotor)==1 — every phase-vector product has unit norm.
//    The existing dist!=1 gate stays authoritative.
//
// 3. cl20_encode_twist maps (x,y,z) → (0,x,y,z) as an opt-in experimental
//    embedding. M²= x²+y²-z² + 2z(xe2-ye1) is NOT a scalar for mixed-grade
//    input, so cl20_vector_norm_sq is valid ONLY on pure grade-1 vectors.

struct Cl20 {
    double s = 0.0;  // grade 0 — scalar
    double x = 0.0;  // grade 1 — e1 coefficient
    double y = 0.0;  // grade 1 — e2 coefficient
    double b = 0.0;  // grade 2 — e12 coefficient
    bool operator==(const Cl20& other) const {
        return s == other.s && x == other.x && y == other.y && b == other.b;
    }
    bool operator!=(const Cl20& other) const { return !(*this == other); }
};

// Construction
Cl20 cl20_scalar(double s);
Cl20 cl20_vector(double vx, double vy);
Cl20 cl20_bivector(double bv);
Cl20 cl20_multivector(double s, double vx, double vy, double bv);

// Grade extraction
double cl20_scalar_part(const Cl20& a);
double cl20_vector_part_x(const Cl20& a);
double cl20_vector_part_y(const Cl20& a);
double cl20_bivector_part(const Cl20& a);

// Algebra operations
Cl20 cl20_add(const Cl20& a, const Cl20& b);
Cl20 cl20_negate(const Cl20& a);
Cl20 cl20_scale(double s, const Cl20& a);
Cl20 cl20_mul(const Cl20& a, const Cl20& b);   // full geometric product
Cl20 cl20_reverse(const Cl20& a);              // negates grade-2 component

// Derived measurements
// Symmetric scalar part: (A*B + B*A)/2 — equals a·b for pure grade-1 vectors.
double cl20_inner(const Cl20& a, const Cl20& b);
// Bivector coefficient of (A*B - B*A)/2 = a.x*b.y - a.y*b.x (orientation).
// Positive = counterclockwise. Telemetry only — not ranking authority.
double cl20_outer_b(const Cl20& a, const Cl20& b);
// x²+y² for pure grade-1 (b==0) vectors only. Do NOT apply to encode_twist output.
double cl20_vector_norm_sq(const Cl20& a);

// Phase geometry — C6 representability in Cl(2,0)
// k in {0..5} maps to Pop/Wo/Yax/Sek/Ch'en/Xul at angle k*π/3.
Cl20 cl20_phase_vector(int phase_k);  // (0, cos(kπ/3), sin(kπ/3), 0)

// GA_PHASE_PRODUCT: e_to * e_from = (cos(Δθ), 0, 0, -sin(Δθ)), Δθ=(to-from)*π/3.
// scalar part == arc_weight(from→to). Full-angle — NOT a sandwich rotation rotor.
Cl20 cl20_phase_product(int from_phase, int to_phase);

// GA_PHASE_ROTATION: half-angle rotor U = (cos(Δθ/2), 0, 0, -sin(Δθ/2)).
// Use with cl20_rotate for vector rotation by Δθ: v' = U*v*reverse(U).
Cl20 cl20_phase_rotor(int from_phase, int to_phase);

// Sandwich product: rotor * v * reverse(rotor). Rotates grade-1 part of v by Δθ.
Cl20 cl20_rotate(const Cl20& v, const Cl20& rotor);

// GA_ENCODE_TWIST: (x,y,z) → (0,x,y,z). Opt-in mixed-grade embedding.
// cl20_vector_norm_sq does NOT apply to the result.
Cl20 cl20_encode_twist(double vx, double vy, double vz);

}  // namespace khanary::kuhul::statealgebra
