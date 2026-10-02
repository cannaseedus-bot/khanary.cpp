# KHANARY Vector Attention (Q/K + W-C-R)

KHANARY uses semantic-field geometry for runtime attention. It does not treat n-gram co-occurrence as attention by itself.

Core mapping:

- `Q` = query semantic vector
- `K_i` = candidate key vector (semantic field position for node `i`)
- `V_i` = Book/Fold payload addressed by key `i`
- `R_i(Q,K_i)` = normalized cosine relevance
- `S_i` = W-C-R admission score

Formulas:

```text
R_i(Q,K_i) = (cos(Q, K_i) + 1) / 2
S_i        = W_i * C_i * R_i(Q,K_i)
```

Where:

- `W_i` is static/learned track weight
- `C_i` is runtime confidence
- `R_i` is query-dependent geometric relevance

Operationally:

```text
Q/K locate relevance
W/C qualify relevance
Key addresses V
Yax admits V into the active field
```

## Runtime contract

1. Query text/tokens activate candidate tracks (lexical, distributional, n-gram, fold compatibility channels).
2. Query vector is resolved from semantic key vectors when possible (weighted centroid across admitted candidates).
3. Hash-derived token vector is used only as cold-start fallback when semantic vectors are unavailable.
4. Candidate attention score is computed from W-C-R and contributes to dispatch ranking.

This keeps declared geometry authoritative while retaining deterministic behavior for unknown/cold-start queries.

## Data and serialization surfaces

Field nodes carry the geometry and W-C-R inputs:

- `x`, `y`, `z` (semantic vector geometry)
- `static_weight` (`W`)
- `confidence` (`C`)
- `health`
- `wcr_prior`
- `vector_source` (declared metadata vs phase fallback)

`field.xml` emits these values per semantic node, including nested `vector_meta`:

```xml
<node id="@node:..." confidence="..." static_weight="..." health="..." wcr_prior="..."
      x="..." y="..." z="...">
  <vector_meta source="declared|phase-default" qkv_role="K" x="..." y="..." z="..."/>
</node>
```

`wcr_prior` is derived at DAG build time as:

```text
wcr_prior = static_weight * confidence * health
```

## Layer separation

- N-grams: activation/co-occurrence evidence.
- Geometry field: semantic positions and typed neighborhood structure.
- Yax/runtime attention: admission and dispatch over the active field using W-C-R.

This preserves deterministic control semantics while allowing geometric relevance scoring and GPU-oriented projection paths.
