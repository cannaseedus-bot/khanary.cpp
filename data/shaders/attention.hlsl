// KHANARY attention.hlsl
// QK-norm + scaled dot-product attention
// Optimized for Intel HD 4600: group size 64, R32_FLOAT buffers

cbuffer AttentionConstants : register(b0)
{
    uint  seq_len;
    uint  d_head;
    uint  n_heads;
    float scale;  // 1/sqrt(d_head)
};

StructuredBuffer<float>   Q   : register(t0);  // [seq_len * d_head * n_heads]
StructuredBuffer<float>   K   : register(t1);  // [seq_len * d_head * n_heads]
StructuredBuffer<float>   V   : register(t2);  // [seq_len * d_head * n_heads]
RWStructuredBuffer<float> Out : register(u0);  // [seq_len * d_head * n_heads]

groupshared float g_scores[64];  // per-row softmax accumulator

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint q_pos = tid.x;       // query position in sequence
    uint head  = tid.y;       // head index (not used here, simplified to head 0)

    if (q_pos >= seq_len) return;

    uint d = d_head;
    uint stride = seq_len * d;

    // QK^T scaled dot product for query q_pos
    float max_score = -1e9f;
    float scores[64]; // local score buffer (max seq = 64 in this shader)
    uint n = min(seq_len, 64u);

    for (uint k_pos = 0; k_pos < n; k_pos++) {
        float dot = 0.0f;
        for (uint di = 0; di < d; di++)
            dot += Q[q_pos * d + di] * K[k_pos * d + di];
        scores[k_pos] = dot * scale;
        if (scores[k_pos] > max_score) max_score = scores[k_pos];
    }

    // Softmax
    float sum = 0.0f;
    for (uint k_pos = 0; k_pos < n; k_pos++) {
        scores[k_pos] = exp(scores[k_pos] - max_score);
        sum += scores[k_pos];
    }
    for (uint k_pos = 0; k_pos < n; k_pos++) scores[k_pos] /= sum;

    // Weighted sum of V
    for (uint di = 0; di < d; di++) {
        float v_sum = 0.0f;
        for (uint k_pos = 0; k_pos < n; k_pos++)
            v_sum += scores[k_pos] * V[k_pos * d + di];
        Out[q_pos * d + di] = v_sum;
    }
}
