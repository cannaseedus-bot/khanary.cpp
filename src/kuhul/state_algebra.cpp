#include "state_algebra.h"
#include "../config.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numbers>

namespace khanary::kuhul::statealgebra {

Pair pair(double scalar, std::vector<double> vector) {
    Pair result;
    result.scalar = scalar;
    result.vector = std::move(vector);
    return result;
}

Pair zero(std::size_t dimension) { return pair(0.0, std::vector<double>(dimension, 0.0)); }

Pair one(std::size_t dimension) { return pair(1.0, std::vector<double>(dimension, 0.0)); }

Pair add(const Pair& a, const Pair& b) {
    Pair result;
    result.scalar = a.scalar + b.scalar;
    result.vector = a.vector;
    const std::size_t n = std::min(result.vector.size(), b.vector.size());
    for (std::size_t i = 0; i < n; ++i) result.vector[i] += b.vector[i];
    return result;
}

Pair negate(const Pair& a) {
    Pair result;
    result.scalar = -a.scalar;
    result.vector.reserve(a.vector.size());
    for (double value : a.vector) result.vector.push_back(-value);
    return result;
}

Pair mul(const Pair& a, const Pair& b) {
    // (a,v)(b,w) = (ab, a.w + b.v). The vw term is absent because V is
    // square-zero: this line is where that shows up.
    Pair result;
    result.scalar = a.scalar * b.scalar;
    const std::size_t n = std::max(a.vector.size(), b.vector.size());
    result.vector.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double left = i < a.vector.size() ? a.vector[i] : 0.0;   // v
        const double right = i < b.vector.size() ? b.vector[i] : 0.0;  // w
        result.vector[i] = a.scalar * right + b.scalar * left;         // a.w + b.v
    }
    return result;
}

std::optional<Pair> inverse(const Pair& a) {
    if (a.scalar == 0.0) return std::nullopt;
    Pair result;
    result.scalar = 1.0 / a.scalar;
    result.vector.reserve(a.vector.size());
    const double factor = -1.0 / (a.scalar * a.scalar);
    for (double value : a.vector) result.vector.push_back(factor * value);
    return result;
}

bool square_is_zero(const std::vector<double>& v, const std::vector<double>& w) {
    const Pair left = pair(0.0, v);
    const Pair right = pair(0.0, w);
    const Pair product = mul(left, right);
    if (product.scalar != 0.0) return false;
    for (double value : product.vector) {
        if (value != 0.0) return false;
    }
    return true;
}

// ── Compartments ─────────────────────────────────────────────────────────────

Subspace span_of(std::initializer_list<std::size_t> coordinates) {
    return Subspace(coordinates.begin(), coordinates.end());
}

Subspace sum(const Subspace& a, const Subspace& b) {
    Subspace result = a;
    result.insert(b.begin(), b.end());
    return result;
}

Subspace intersection(const Subspace& a, const Subspace& b) {
    Subspace result;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(result, result.end()));
    return result;
}

bool includes(const Subspace& a, const Subspace& b) {
    return std::includes(a.begin(), a.end(), b.begin(), b.end());
}

Pair embed(const Subspace& w, std::size_t dimension) {
    Pair result = zero(dimension);
    for (std::size_t coordinate : w) {
        if (coordinate < dimension) result.vector[coordinate] = 1.0;
    }
    return result;
}

bool is_ideal(const Subspace& w, const std::vector<Pair>& probes, std::size_t dimension) {
    const Pair embedded = embed(w, dimension);
    for (const Pair& probe : probes) {
        // Both sides, because the algebra is commutative here but the check must
        // not rely on that: an ideal is two-sided by definition.
        for (const Pair& product : {mul(probe, embedded), mul(embedded, probe)}) {
            if (product.scalar != 0.0) return false;  // an ideal inside V has no K part
            for (std::size_t i = 0; i < dimension; ++i) {
                const double value = i < product.vector.size() ? product.vector[i] : 0.0;
                if (value == 0.0) continue;
                if (w.count(i) == 0) return false;  // leaked outside the compartment
            }
        }
    }
    return true;
}

Pair project(const Pair& a, const Subspace& w, std::size_t dimension) {
    Pair result = zero(dimension);
    for (std::size_t i = 0; i < dimension && i < a.vector.size(); ++i) {
        if (w.count(i) != 0) result.vector[i] = a.vector[i];
    }
    return result;
}

double external_inner_product(const std::vector<double>& v, const std::vector<double>& w) {
    double total = 0.0;
    const std::size_t n = std::min(v.size(), w.size());
    for (std::size_t i = 0; i < n; ++i) total += v[i] * w[i];
    return total;
}

// ── Cl(2,0) — 2D Euclidean Geometric Algebra ─────────────────────────────────
//
// Geometric product formula for M=s+xe1+ye2+be12:
//   (A*B).s  = sa*sb + xa*xb + ya*yb - ba*bb
//   (A*B).x  = sa*xb + xa*sb + ba*yb - ya*bb
//   (A*B).y  = sa*yb + ya*sb + xa*bb - ba*xb
//   (A*B).b  = sa*bb + ba*sb + xa*yb - ya*xb
//
// Derived from the basis multiplication table:
//   e1*e1=1, e2*e2=1, e12*e12=-1, e1*e2=e12, e2*e1=-e12

Cl20 cl20_scalar(double s)                               { return {s, 0.0, 0.0, 0.0}; }
Cl20 cl20_vector(double vx, double vy)                   { return {0.0, vx, vy, 0.0}; }
Cl20 cl20_bivector(double bv)                            { return {0.0, 0.0, 0.0, bv}; }
Cl20 cl20_multivector(double s, double vx, double vy, double bv) { return {s, vx, vy, bv}; }

double cl20_scalar_part(const Cl20& a)    { return a.s; }
double cl20_vector_part_x(const Cl20& a) { return a.x; }
double cl20_vector_part_y(const Cl20& a) { return a.y; }
double cl20_bivector_part(const Cl20& a) { return a.b; }

Cl20 cl20_add(const Cl20& a, const Cl20& b) {
    return {a.s + b.s, a.x + b.x, a.y + b.y, a.b + b.b};
}

Cl20 cl20_negate(const Cl20& a) { return {-a.s, -a.x, -a.y, -a.b}; }

Cl20 cl20_scale(double s, const Cl20& a) { return {s * a.s, s * a.x, s * a.y, s * a.b}; }

Cl20 cl20_mul(const Cl20& a, const Cl20& b) {
    return {
        a.s * b.s + a.x * b.x + a.y * b.y - a.b * b.b,
        a.s * b.x + a.x * b.s + a.b * b.y - a.y * b.b,
        a.s * b.y + a.y * b.s + a.x * b.b - a.b * b.x,
        a.s * b.b + a.b * b.s + a.x * b.y - a.y * b.x
    };
}

// reverse: grade-0 and grade-1 unchanged, grade-2 negated.
Cl20 cl20_reverse(const Cl20& a) { return {a.s, a.x, a.y, -a.b}; }

double cl20_inner(const Cl20& a, const Cl20& b) {
    // scalar part of (A*B + B*A)/2
    return 0.5 * (cl20_mul(a, b).s + cl20_mul(b, a).s);
}

double cl20_outer_b(const Cl20& a, const Cl20& b) {
    // bivector coefficient of (A*B - B*A)/2 = a.x*b.y - a.y*b.x
    return a.x * b.y - a.y * b.x;
}

double cl20_vector_norm_sq(const Cl20& a) {
    // Valid ONLY for pure grade-1 vectors (a.b == 0). Returns x²+y².
    return a.x * a.x + a.y * a.y;
}

Cl20 cl20_phase_vector(int phase_k) {
    const double theta = phase_k * (std::numbers::pi / 3.0);
    return cl20_vector(std::cos(theta), std::sin(theta));
}

Cl20 cl20_phase_product(int from_phase, int to_phase) {
    // e_to * e_from: scalar = cos(Δθ), bivector = -sin(Δθ)
    const double delta = (to_phase - from_phase) * (std::numbers::pi / 3.0);
    return {std::cos(delta), 0.0, 0.0, -std::sin(delta)};
}

Cl20 cl20_phase_rotor(int from_phase, int to_phase) {
    // Half-angle rotor for sandwich rotation by Δθ: U = cos(Δθ/2) - sin(Δθ/2)*e12
    const double half_delta = (to_phase - from_phase) * (std::numbers::pi / 6.0);
    return {std::cos(half_delta), 0.0, 0.0, -std::sin(half_delta)};
}

Cl20 cl20_rotate(const Cl20& v, const Cl20& rotor) {
    return cl20_mul(cl20_mul(rotor, v), cl20_reverse(rotor));
}

Cl20 cl20_encode_twist(double vx, double vy, double vz) {
    // Opt-in embedding: (x,y,z) → (0, x, y, z). Mixed-grade — NOT norm-preserving.
    return {0.0, vx, vy, vz};
}

}  // namespace khanary::kuhul::statealgebra
