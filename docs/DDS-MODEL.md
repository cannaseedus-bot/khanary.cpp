# DDS Model Format — KHANARY.CPP Reference

## Revelation — K'UHUL Native Fold Math

This stack now treats graph math as native runtime law, not external analysis:

- **Token is address; meaning unfolds from context** (`Book(token)` / fold neighborhood).
- **Fold is distributed semantic context**, not only taxonomy (`Hypernym Fold ⊂ Semantic Folds`).
- **Incidence topology and weights are separated** (`B` for participation, `W_E` for pattern strength).
- **Distributional similarity is its own channel** (context overlap signal, not node meaning or edge weight).

The GPT-OSS 20B model is stored as a **DDS shard set**: a collection of 128 MB binary shard files indexed by two JSON manifests and a KAST-1 driver descriptor (`.kson`). KHANARY loads it through the `FoldMap → HotSwapManager → InferenceEngine` pipeline.

---

## File Layout

```
E:/models/GPT-DDS/GPT-OSS/
  gpt_oss_20b_folds.kson              KAST/1 driver manifest — fold → tensor mapping
  gpt-oss-20b-MXFP4.tensor_map.json  tensor name → safetensors offset index
  gpt-oss-20b-MXFP4.safetensors      source weights (11.3 GB, not used at runtime)

  folds/
    model_dds_manifest.json           physical shard index — 91 entries × 128 MB
    fold_0.dds … fold_90.dds          physical shard files (11.27 GB total payload)

  dds_shards/                         hot-swap shard staging area
  model.scxqdds                       SCXQDDS micro-model (643 KB, SDDS format)
```

---

## Gemma → DDS Runtime Contract

`infer` in KHANARY is DDS-driven. A `.gguf` path (for example Gemma toolcall builds) is
preloaded into system RAM as a base-model sidecar and used for model-selection hooks; it is
**not** a direct execution weight source for the `InferenceEngine` path. To run Gemma through
this runtime, Gemma weights must be converted to the same DDS fold artifact family.

### Required runtime artifacts

```
<model-root>\
  <model>_folds.kson                  required
  folds\model_dds_manifest.json       required (dds_manifest.json may exist as an alias/report output)
  folds\dds_shards\fold_*.dds         required
```

### Supporting artifacts (recommended)

```
folds\model_fold_manifest.json        fold-domain grouping / residency metadata
folds\fold_manifest.json              alternate manifest naming
folds\pipeline_results.json           export summary / shard statistics
folds\model.scxqdds                   compact SDDS bootstrap payload
```

### Current launch defaults used in this project

```
KHANARY_DDS_KSON=E:\models\GPT-DDS\GPT-OSS\gpt_oss_20b_folds.kson
KHANARY_DDS_MANIFEST=E:\models\GPT-DDS\GPT-OSS\folds\model_dds_manifest.json
KHANARY_DDS_SHARDS=E:\models\GPT-DDS\GPT-OSS\folds\dds_shards
KHANARY_GGUF_MODEL=E:\models\GPT2\gemma_toolcall\gemma_chat_toolcall_q8.gguf
```

If/when Gemma DDS exports are generated, swap `KHANARY_DDS_*` to Gemma’s DDS set.

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-OSS (decoder-only MoE) |
| Parameters | ~20B |
| d_model | 2880 |
| n_layers | 24 |
| n_heads (Q) | 32 |
| n_kv_heads (GQA) | 4 |
| d_head | 128 |
| n_experts | 32 |
| top_k | 2 |
| d_ffn | 2880 (symmetric) |
| vocab_size | 201088 |
| max_seq_len | 256 (config default) |

GQA group ratio: 8 Q heads per KV head. All attention projections are Q8_0; all expert FFN weights are MXFP4.

---

## Quantization Formats

### Q8_0 — Grouped INT8

Used by: attention Q/K/V/output projections, embeddings, output unembed.

```
Block: 34 bytes → 32 float elements
  [uint16  scale_f16]   2 bytes  — IEEE 754 half-precision block scale
  [int8    qs[32]]      32 bytes — quantized values

Dequant: f[i] = qs[i] × fp16_to_f32(scale_f16)
Row bytes (d_model=2880): (2880/32) × 34 = 3060 bytes/row
```

### MXFP4 — Microscaled FP4 (E2M1 + E8M0)

Used by: all expert gate/up/down weight matrices, token embedding.

```
Block: 17 bytes → 32 float elements
  [uint8   e8m0_scale]  1 byte   — shared exponent, bias=127
  [uint8   qs[16]]      16 bytes — 32 × 4-bit E2M1 nibbles

Scale: scale = 2^(e8m0 - 127)   [fp32_from_e8m0: bits = e << 23u; memcpy]

Nibble order (ggml):
  element j     (0 ≤ j < 16) = k_mxfp4_e2m1[ qs[j] & 0xF ] × scale
  element j+16  (0 ≤ j < 16) = k_mxfp4_e2m1[ qs[j] >> 4  ] × scale

E2M1 lookup table (index 0..15):
  0.0  0.5  1.0  1.5  2.0  3.0  4.0   6.0
 -0.0 -0.5 -1.0 -1.5 -2.0 -3.0 -4.0  -6.0

Expert weight row bytes (cols=2880): (2880/32) × 17 = 1530 bytes/row
```

---

## DDS Shard Format

Each `fold_N.dds` file:

```
Offset 0  : [128 bytes] DDS format header
Offset 128: [payload]   raw tensor byte stream
```

The payload is a flat byte array. Tensor slices within it are indexed by `offset_in_file` and `size` from `model_dds_manifest.json`. A single tensor may span multiple shard files; `offset_in_tensor` gives the position within the logical tensor byte stream.

### model_dds_manifest.json — Per-Shard Entry

```json
{
  "id": 9,
  "dds_file": "fold_9.dds",
  "size": 134217852,
  "payload_bytes": 134217728,
  "tensors": [
    {
      "name": "blk.0.ffn_down_exps.weight",
      "offset_in_file": 51324032,
      "offset_in_payload": 51323904,
      "offset_in_tensor": 0,
      "size": 82893824,
      "dtype": "U8",
      "shape": [32, 2880, 1530],
      "logical_shape": [2880, 2880, 32],
      "ggml_type": 39,
      "ggml_type_name": "MXFP4"
    }
  ]
}
```

- **`shape`** — physical/quantized shape (raw U8 array dimensions)
- **`logical_shape`** — pre-quantization float shape, kson `[in_dim, out_dim, ...]` convention
- **`dtype`** — physical: always `"U8"` (raw bytes)
- **`ggml_type_name`** — logical: `"F32"`, `"Q8_0"`, or `"MXFP4"`

### Shard statistics

| Type | Entries | Payload |
|------|---------|---------|
| F32 | 289 | 35 MB |
| Q8_0 | 110 | 1.78 GB |
| MXFP4 | 150 | 9.46 GB |
| **Total** | **549** | **11.27 GB** (91 × 128 MB shards) |

---

## gpt_oss_20b_folds.kson — KAST/1 Driver

Describes the fold-to-tensor logical mapping consumed by `FoldMap::load()`.

```json
{
  "protocol": "kast/1",
  "kind": "driver-only",
  "@driver": {
    "@resources": {
      "fold_pool": { "total_folds": 50, "n_experts": 32 },
      "folds": [
        { "fold": 0, "id": "embedding",         "nodes": [ ... ] },
        { "fold": 1, "id": "layer_00.attention", "nodes": [ ... ] },
        { "fold": 2, "id": "layer_00.moe",       "nodes": [ ... ] },
        ...
        { "fold": 49, "id": "output",            "nodes": [ ... ] }
      ]
    }
  }
}
```

Fold IDs: `embedding`, `layer_NN.attention`, `layer_NN.moe`, `output`. Each node lists `name`, `shape` (logical), `scxq2` (dtype: `"q8_0"` or `"mxfp4"`), `offset`, `length`.

**kson shape convention**: `[in_dim, out_dim]` (NOT ggml's `[out, in]`). All GEMM calls in KHANARY use this directly: `A[M, K] × W[K, N] = C[M, N]` with no transpose.

---

## Tensor Inventory by Layer

### Global (fold 0 / fold 49)

| Tensor | Shape | Dtype | Note |
|--------|-------|-------|------|
| `token_embd.weight` | [2880, 201088] | Q8_0 | embedding lookup |
| `output_norm.weight` | [2880] | F32 | final RMSNorm scale |
| `output.weight` | [2880, 201088] | Q8_0 | unembed (shared with embd) |

### Per Layer `blk.N.*` (N = 0..23)

**Attention (layer_NN.attention fold)**

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `blk.N.attn_norm.weight` | [2880] | F32 |
| `blk.N.attn_q.weight` | [2880, 4096] | Q8_0 |
| `blk.N.attn_k.weight` | [2880, 512] | Q8_0 |
| `blk.N.attn_v.weight` | [2880, 512] | Q8_0 |
| `blk.N.attn_q.bias` | [4096] | F32 |
| `blk.N.attn_k.bias` | [512] | F32 |
| `blk.N.attn_v.bias` | [512] | F32 |
| `blk.N.attn_output.weight` | [4096, 2880] | Q8_0 |
| `blk.N.attn_output.bias` | [2880] | F32 |
| `blk.N.attn_sinks.weight` | [64] | F32 |

**MoE FFN (layer_NN.moe fold)**

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `blk.N.post_attention_norm.weight` | [2880] | F32 |
| `blk.N.ffn_gate_inp.weight` | [2880, 32] | Q8_0 | router |
| `blk.N.ffn_gate_inp.bias` | [32] | F32 | |
| `blk.N.ffn_gate_exps.weight` | [2880, 2880, 32] | MXFP4 | gate proj |
| `blk.N.ffn_up_exps.weight` | [2880, 2880, 32] | MXFP4 | up proj |
| `blk.N.ffn_down_exps.weight` | [2880, 2880, 32] | MXFP4 | down proj |
| `blk.N.ffn_gate_exps.bias` | [2880, 32] | F32 | |
| `blk.N.ffn_down_exps.bias` | [2880, 32] | F32 | |

---

## Expert Weight Physical Layout

Expert weights are 3-D: `[n_experts=32, rows, packed_cols]`.

```
Physical shard bytes for expert eid:
  byte_offset = eid × rows × row_bytes
  row_bytes   = (cols / 32) × 17   [MXFP4]

For gate/up (rows=d_model=2880, cols=d_ffn=2880):
  row_bytes  = 1530
  expert_slice = 2880 × 1530 = 4,406,400 bytes (~4.2 MB per expert)

For down (rows=d_ffn=2880, cols=d_model=2880):
  same dimensions — symmetric model
```

`load_mxfp4_expert_f32(tensor_name, eid, rows, cols)` reads only the slice for `eid`, not the full 135 MB matrix.

---

## Forward Pass (DDS Path)

```
forward_dds(tokens):
  Phase Pop:   embed_dds(tokens)
               → Q8_0 row lookup: token × 3060 bytes → 2880 floats/token

  Phase Wo→Sek (24 layers):
    apply_attention_dds(layer, hidden, seq):
      1. RMSNorm (attn_norm.weight, F32)
      2. GPU GEMM → Q [seq, 4096], K [seq, 512], V [seq, 512]   [Q8_0]
      3. CPU GQA: causal softmax, 32 Q heads / 4 KV heads
      4. GPU GEMM → output proj [seq, 2880]   [Q8_0]
      5. Residual + output bias

    apply_moe_dds(layer, hidden, seq):
      1. RMSNorm (post_attention_norm.weight, F32)
      2. GPU GEMM → router scores [seq, 32]   [Q8_0]
      3. Softmax + top-2 selection per token
      4. For each top-2 expert eid per token:
           load_mxfp4_expert_f32(gate, eid, 2880, 2880) → GPU
           load_mxfp4_expert_f32(up,   eid, 2880, 2880) → GPU
           load_mxfp4_expert_f32(down, eid, 2880, 2880) → GPU
           GPU GEMM: h × gate_w → gate_act [1, 2880]
           GPU GEMM: h × up_w   → up_act   [1, 2880]
           CPU SwiGLU: ffn[i] = silu(gate[i]) × up[i]
           GPU GEMM: ffn × down_w → expert_out [1, 2880]
           Accumulate: out += router_weight × expert_out
      5. Residual

  Phase Ch'en:  RMSNorm (output_norm.weight, F32)

  Phase Xul:    compute_output_logits_chunked(last_hidden)
                → Q8_0 dot-product in 4096-vocab chunks (avoids 2.3 GB F32 alloc)
                → logits [201088]
```

---

## HotSwap Residency

`HotSwapManager` maintains an LRU pool within the **1792 MB** HD 4600 ceiling.

- Keys: `"f32:<tensor_name>"` (fully dequantized F32 TensorBuffers)
- Attention weights are cached across tokens; expert weights are loaded per-call
- Embedding and output folds are pre-warmed at `load_dds_model()` startup
- `make_resident(key, buf)` / `touch(key)` / `evict()` as needed

---

## SCXQDDS Mini-Model (model.scxqdds)

A compact SDDS wire-format model used for fast bootstrapping and sandboxed inference.

```
Header: magic "SDDS" (4 bytes)
Chunks: [name: char[64]][elem_count: uint32][format: uint32]
        [data_size: uint32][crc32: uint32][data: uint8[data_size]]
Formats: 0=F32, 1=Q8 (INT8 packed), 2=F16
```

---

## GPU GEMM Kernel

All GEMM calls in the DDS path dispatch through `DMLOps::gemm()`.

```hlsl
// HLSL_TILED — 16×16 groupshared, compiled via D3DCompile in DMLOps::init()
// Validated: 27.3 GFLOPS on Intel HD 4600
cbuffer P : register(b0) { uint M; uint K; uint N; uint pad; };
StructuredBuffer<float>   A : register(t0);   // [M, K]
StructuredBuffer<float>   B : register(t1);   // [K, N]
RWStructuredBuffer<float> C : register(u0);   // [M, N]
// Dispatch: groups = (ceil(N/16), ceil(M/16), 1)
```

Falls back to CPU when `alpha ≠ 1` or `beta ≠ 0`.
TensorBuffers use `D3D11_RESOURCE_MISC_BUFFER_STRUCTURED` (stride=4), compatible with `StructuredBuffer<float>` / `RWStructuredBuffer<float>`.

---

## Loading Sequence

```cpp
InferenceEngine::load_dds_model(kson_path, manifest_path, shards_dir):
  1. FoldMap::load(kson, manifest, shards_dir)
       — parse manifest → tensor_name → vector<ShardSlice> (sorted by offset_in_tensor)
       — parse kson → fold entries → TensorRefs with dtype + logical_shape + shards
       — build_indices(): by_layer_id_, by_fold_index_, by_tensor_name_

  2. HotSwapManager()  [1792 MB ceiling]

  3. update_config_from_folds()
       — d_model, vocab_size from token_embd.weight logical_shape
       — n_heads, n_kv_heads from blk.0.attn_q/k shape
       — n_experts from blk.0.ffn_gate_inp.weight shape[1]

  4. preload_fold("embedding") + preload_fold("output")
       [skips tensors >256 MB at startup]

  5. dds_loaded_ = true
```
