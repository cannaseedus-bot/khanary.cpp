// KHANARY moe_router.hlsl
// MoE router: softmax over expert logits + top-k selection via bitonic sort

cbuffer RouterConstants : register(b0)
{
    uint  n_experts;
    uint  hidden_dim;
    uint  top_k;
    uint  pad;
};

StructuredBuffer<float>   hidden      : register(t0);  // [hidden_dim]
StructuredBuffer<float>   router_w    : register(t1);  // [n_experts * hidden_dim]
StructuredBuffer<float>   router_bias : register(t2);  // [n_experts]
RWStructuredBuffer<float> logits      : register(u0);  // [n_experts]
RWStructuredBuffer<uint>  top_k_ids   : register(u1);  // [top_k]
RWStructuredBuffer<float> top_k_w     : register(u2);  // [top_k]

groupshared float g_logits[256];
groupshared uint  g_ids[256];

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint e = tid.x;
    if (e >= n_experts) return;

    // Compute logit: dot(router_w[e], hidden) + bias[e]
    float dot_val = 0.0f;
    for (uint d = 0; d < hidden_dim; d++)
        dot_val += router_w[e * hidden_dim + d] * hidden[d];
    dot_val += router_bias[e];
    logits[e] = dot_val;
    g_logits[e] = dot_val;
    g_ids[e] = e;

    GroupMemoryBarrierWithGroupSync();

    // Softmax normalization (single-pass approximation for HD 4600)
    if (gtid.x == 0) {
        float max_l = g_logits[0];
        for (uint i = 1; i < n_experts; i++)
            if (g_logits[i] > max_l) max_l = g_logits[i];
        float sum = 0.0f;
        for (uint i = 0; i < n_experts; i++) {
            g_logits[i] = exp(g_logits[i] - max_l);
            sum += g_logits[i];
        }
        for (uint i = 0; i < n_experts; i++) g_logits[i] /= sum;

        // Bitonic sort: descending by logit (simple O(N²) for N <= 16)
        for (uint i = 0; i < n_experts; i++) {
            for (uint j = i + 1; j < n_experts; j++) {
                if (g_logits[j] > g_logits[i]) {
                    float tmp_l = g_logits[i]; g_logits[i] = g_logits[j]; g_logits[j] = tmp_l;
                    uint  tmp_i = g_ids[i];    g_ids[i]    = g_ids[j];    g_ids[j]    = tmp_i;
                }
            }
        }

        // Write top-k
        for (uint k = 0; k < top_k && k < n_experts; k++) {
            top_k_ids[k] = g_ids[k];
            top_k_w[k]   = g_logits[k];
        }
    }
}
