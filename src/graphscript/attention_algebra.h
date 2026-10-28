#pragma once
#include <stdexcept>
#include <vector>

namespace khanary::graphscript {

// ── Attention algebra: Attention(Q,K,V) over the Map-Filter-Fold triple ────
// The mechanism from the box, exactly — softmax(QKᵀ/√d)·V — decomposed so the
// map/filter/fold algebra and the attention mechanism are THE SAME machinery:
//
//   1. MAP       scaled_dot_scores(Q, K)   = Q·Kᵀ/√d_k   (per query row)
//   2. FILTER    softmax_row(...)           — the soft selection; plus the
//                OPTIONAL crisp mask (true = keep) — normally the DECLARED
//                graph's edges: attention is admitted only over explicitly
//                retained relationships.
//   3. FOLD      weighted_fold(w_row, V)    = Σ_j w_j·V_j (the aggregation)
//
// Masked entries are excluded from each row's softmax and the weights
// re-normalize over survivors; a row with NO surviving key is a named
// AttentionViolation (the crisp filter never silently fabricates mass).
// The cosine/WCR scoring variant (W·C·cos(Q,K) — the grammar's own kernel)
// is exposed as an alternative score map: on unit-scaled rows it agrees
// with the dot form.

using RowMatrix = std::vector<std::vector<double>>;

struct AttentionViolation : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ── The decomposition pieces ───────────────────────────────────────────────
// MAP: scaled dot-product score matrix, shape |Q|×|K|.
RowMatrix scaled_dot_scores(const RowMatrix& queries, const RowMatrix& keys);

// The WCR-family variant: W·C·cos(Q,K) — grammar's kernel as scores.
RowMatrix wcr_scores(const RowMatrix& queries, const RowMatrix& keys,
                     const std::vector<double>& weights,     // W per query
                     const std::vector<double>& confidences); // C per key

// FILTER: stable softmax over a row; masked (keep=false) entries excluded
// and the remaining weights renormalized. All-masked → AttentionViolation.
std::vector<double> softmax_row(const std::vector<double>& row,
                                const std::vector<bool>& keep = {});

// FOLD: the weighted aggregation of value rows under one weight row.
std::vector<double> weighted_fold(const std::vector<double>& weights,
                                  const RowMatrix& values);

// The composed mechanism: Attention(Q,K,V) = softmax(QKᵀ/√d)·V with the
// optional keep-mask (n_q×n_k; empty vector = all admitted) and the optional
// causal policy (query i sees keys j ≤ i). Output: |Q|×d_v, shape preserved.
RowMatrix attention(const RowMatrix& queries,
                    const RowMatrix& keys,
                    const RowMatrix& values,
                    const std::vector<std::vector<bool>>& keep_mask = {},
                    bool causal = false);

// The field tie: SELF-attention over a declared graph — the vertices'
// payloads are Q, K, and V; the mask IS the declared adjacency (weight > 0
// = an admitted relationship). This is the explicit-retained-relationships
// principle applied to attention.
RowMatrix self_attention_over_field(
    const std::vector<std::vector<float>>& vertex_payloads,
    const std::vector<std::vector<float>>& adjacency);

// ── INCIDENCE_PRODUCT: A = B · W_E · Bᵀ ─────────────────────────────────────
// The Active Fold Graph attention matrix (SEMANTIC_FOLDS.md / FOLD_GRAPH.md):
//   A_h_active = B_h_active · W_E · B_h_activeᵀ
//
// B:   |V|×|C| incidence fingerprint (rows = vertices, cols = contexts/edges)
// W_E: |C|×|C| weight matrix (diagonal for per-edge scalar weights)
// A:   |V|×|V|; A = Aᵀ when W_E = W_Eᵀ
//
// Entry-wise: A[i][j] = B_i · W_E · B_jᵀ — the W_E-Gram bilinear form on
// vertex-row pairs. The FULL map B → BWBᵀ is quadratic in B (not linear),
// so INCIDENCE_PRODUCT is NOT bilinear in the whole-matrix sense; it is linear
// in W_E for fixed B.
//
// Violation: W_E dimension != number of columns in B.
RowMatrix incidence_product(const RowMatrix& B, const RowMatrix& W_E);

// Build an n×n diagonal matrix from per-context weight vector w.
// Convenience for the common case where W_E encodes a scalar weight per edge.
RowMatrix diagonal_weights(const std::vector<double>& w);

} // namespace khanary::graphscript