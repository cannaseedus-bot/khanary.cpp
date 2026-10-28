#include "attention_algebra.h"
#include <algorithm>
#include <cmath>
#include <iterator>

namespace khanary::graphscript {

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

} // namespace

// ── MAP: the scaled dot-product score matrix (Q·Kᵀ/√d_k) ───────────────────
RowMatrix scaled_dot_scores(const RowMatrix& queries, const RowMatrix& keys) {
    if (!keys.empty() && keys[0].empty())
        throw AttentionViolation("attention: zero-dimension key rows");
    const double scale = keys.empty() || keys[0].empty()
        ? 1.0 : 1.0 / std::sqrt(static_cast<double>(keys[0].size()));
    RowMatrix scores;
    scores.reserve(queries.size());
    for (const auto& q : queries) {
        std::vector<double> row;
        row.reserve(keys.size());
        for (const auto& k : keys) row.push_back(dot(q, k) * scale);
        scores.push_back(std::move(row));
    }
    return scores;
}

// ── MAP (WCR variant): W·C·cos(Q,K) — the grammar's kernel as scores ───────
RowMatrix wcr_scores(const RowMatrix& queries, const RowMatrix& keys,
                     const std::vector<double>& weights,
                     const std::vector<double>& confidences) {
    RowMatrix scores;
    scores.reserve(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        const double w = i < weights.size() ? weights[i] : 1.0;
        std::vector<double> row;
        row.reserve(keys.size());
        for (size_t j = 0; j < keys.size(); ++j) {
            const double c = j < confidences.size() ? confidences[j] : 1.0;
            // cos(q, k) with zero-norm guarded to 0 (no direction = no mass)
            const double nq = std::sqrt(dot(queries[i], queries[i]));
            const double nk = std::sqrt(dot(keys[j], keys[j]));
            const double cos_qk = (nq == 0.0 || nk == 0.0)
                ? 0.0 : dot(queries[i], keys[j]) / (nq * nk);
            row.push_back(w * c * cos_qk);
        }
        scores.push_back(std::move(row));
    }
    return scores;
}

// ── FILTER: stable softmax, the soft selection ─────────────────────────────
std::vector<double> softmax_row(const std::vector<double>& row,
                                const std::vector<bool>& keep) {
    if (!keep.empty() && keep.size() != row.size())
        throw AttentionViolation("attention: mask width != key count");
    bool any = false;
    double max_logit = row.empty() ? 0.0 : row[0];
    if (!keep.empty()) {
        for (size_t j = 0; j < row.size(); ++j) {
            if (keep[j]) { any = true; max_logit = std::max(max_logit, row[j]); }
        }
        if (!any)
            throw AttentionViolation(
                "attention: query row has no surviving key (all-masked)");
    } else {
        any = !row.empty();
        for (double v : row) max_logit = std::max(max_logit, v);
    }
    if (!any) throw AttentionViolation("attention: empty score row");

    std::vector<double> weights(row.size(), 0.0);
    double total = 0.0;
    for (size_t j = 0; j < row.size(); ++j) {
        if (!keep.empty() && !keep[j]) continue;   // excluded from the mass
        weights[j] = std::exp(row[j] - max_logit);
        total += weights[j];
    }
    if (total > 0.0)
        for (double& w : weights) w /= total;
    return weights;
}

// ── FOLD: the weighted aggregation Σ w_j·V_j ───────────────────────────────
std::vector<double> weighted_fold(const std::vector<double>& weights,
                                  const RowMatrix& values) {
    std::vector<double> out;
    for (size_t j = 0; j < values.size() && j < weights.size(); ++j) {
        const double w = weights[j];
        if (w == 0.0) continue;
        out.resize(std::max(out.size(), values[j].size()), 0.0);
        for (size_t d = 0; d < values[j].size(); ++d)
            out[d] += w * values[j][d];
    }
    return out;
}

// ── The composed mechanism ─────────────────────────────────────────────────
RowMatrix attention(const RowMatrix& queries,
                    const RowMatrix& keys,
                    const RowMatrix& values,
                    const std::vector<std::vector<bool>>& keep_mask,
                    bool causal) {
    RowMatrix out;
    out.reserve(queries.size());
    const RowMatrix scores = scaled_dot_scores(queries, keys);

    for (size_t i = 0; i < scores.size(); ++i) {
        std::vector<bool> keep;
        if (i < keep_mask.size()) keep = keep_mask[i];
        if (causal && keys.size() == scores.size()) {
            keep.resize(keys.size(), true);
            for (size_t j = i + 1; j < keep.size(); ++j) keep[j] = false;
        }
        const auto weights = softmax_row(scores[i], keep);
        out.push_back(weighted_fold(weights, values));
    }
    return out;
}

// ── INCIDENCE PRODUCT: A = B · W_E · Bᵀ ──────────────────────────────────────
RowMatrix incidence_product(const RowMatrix& B, const RowMatrix& W_E) {
    if (B.empty()) return {};
    const size_t nv = B.size();
    const size_t nc = B[0].size();
    if (W_E.size() != nc)
        throw AttentionViolation("incidence_product: W_E rows != B columns");
    for (size_t r = 0; r < nc; ++r)
        if (W_E[r].size() != nc)
            throw AttentionViolation("incidence_product: W_E must be square C×C");

    // Step 1: BW[i][d] = Σ_c B[i][c] * W_E[c][d]
    RowMatrix BW(nv, std::vector<double>(nc, 0.0));
    for (size_t i = 0; i < nv; ++i)
        for (size_t c = 0; c < nc; ++c)
            if (B[i][c] != 0.0)
                for (size_t d = 0; d < nc; ++d)
                    BW[i][d] += B[i][c] * W_E[c][d];

    // Step 2: A[i][j] = BW[i] · B[j]ᵀ = Σ_d BW[i][d] * B[j][d]
    RowMatrix A(nv, std::vector<double>(nv, 0.0));
    for (size_t i = 0; i < nv; ++i)
        for (size_t j = 0; j < nv; ++j)
            A[i][j] = dot(BW[i], B[j]);
    return A;
}

RowMatrix diagonal_weights(const std::vector<double>& w) {
    const size_t n = w.size();
    RowMatrix W(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) W[i][i] = w[i];
    return W;
}

// ── The field tie: self-attention over a declared graph ────────────────────
RowMatrix self_attention_over_field(
    const std::vector<std::vector<float>>& vertex_payloads,
    const std::vector<std::vector<float>>& adjacency) {
    if (vertex_payloads.size() != adjacency.size())
        throw AttentionViolation(
            "self attention: adjacency rows != vertex count");
    RowMatrix q, k, v;
    for (const auto& payload : vertex_payloads) {
        q.push_back(std::vector<double>(payload.begin(), payload.end()));
        k.push_back(std::vector<double>(payload.begin(), payload.end()));
        v.push_back(std::vector<double>(payload.begin(), payload.end()));
    }
    std::vector<std::vector<bool>> keep;
    keep.reserve(adjacency.size());
    for (const auto& row : adjacency) {
        std::vector<bool> admitted;
        admitted.reserve(row.size());
        for (float weight : row) admitted.push_back(weight > 0.0f);
        keep.push_back(std::move(admitted));
    }
    return attention(q, k, v, keep);
}

} // namespace khanary::graphscript