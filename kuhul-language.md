# K'UHUL v8.0.0 Language Reference

K'UHUL is the semantic field algebra for KHANARY execution. This reference adds tensor-bundle grounding to the existing K'UHUL π runtime grammar without changing the authority boundary: grammar and projection surfaces may describe geometry, but Ch'en/Xul remain the verification and topology commit gates.

## Tensor Bundle Grounding

K'UHUL's runtime is a computational implementation of tensor-bundle calculus over the SVG-3D/FoldGraph projection manifold.

The core distinction is:

```text
Phase θ is the CONNECTION: the parallel-transport prescription.
The geodesic propagator finds GEODESICS under that connection.

Connection = rule.
Geodesic = path the rule selects.
```

## Precise mappings

| Tensor bundle object | K'UHUL object | Status |
| --- | --- | --- |
| Base manifold `M` | SVG-3D tensor plane / FoldGraph projection manifold | precise projection substrate |
| Phase `θ` | discrete connection on C6 with `arc_weight = cos(Δθ)` | precise |
| TensorSubstrate | fiber bundle descriptor `{base point, fiber}` | precise runtime substrate |
| Gram `G` | tensor product rank, multi-token keys as `V^⊗p` | precise symbolic basis |

## Structural mappings

| Tensor bundle object | K'UHUL object | Boundary |
| --- | --- | --- |
| Cotangent `V*` | Fold `F` | algebra-level only |
| Geodesic curve `γ` | `dispatch_geodesic_propagator()` | path under connection, not the connection |
| Interior product `ι` | Projection `Π` | projection-shaped contraction, not exterior derivative `d` |

## Type system additions

- `bundle<M,p,q>`: tensor bundle over manifold `M` with type `(p,q)` rank.
- `fiber<p,q>`: fiber at a point, with `p` contravariant and `q` covariant rank.
- `connection`: parallel-transport rule on a bundle.
- `geodesic`: curve selected under a connection.
- `parallel_transport`: operator that moves a section along `γ` under `∇`.

## Operators

- `⊗`: tensor product.
- `∇`: connection / covariant derivative.
- `d`: exterior derivative, `Ω^k -> Ω^(k+1)`.
- `ι`: interior product, `Ω^k -> Ω^(k-1)`.
- `γ`: geodesic path under a connection.
- `Π`: projection / observation, closer to `ι` than `d`.

## Phase cycle as connection

```text
Pop -> Wo -> Yax -> Sek -> Ch'en -> Xul
```

The phase cycle is path-ordered. Reordering phases changes the holonomy, so the cycle is a connection constraint rather than a scheduling preference. Each adjacent transition uses `arc_weight = cos(Δθ)`.

## Example

```kuhul
universe TensorBundleExample : "8.0.0" {
  manifold M : spherical(2)

  bundle TB over M {
    fiber { rank: (1,0), dimension: 3, stride: 3 }
    connection { type: discrete_circle, arc_weight: cos(delta_theta), holonomy: path_ordered }
    geodesic { propagator: dispatch_geodesic_propagator, minimizes: entropy }
  }

  func transport_vector(v: vector, from: point, to: point) -> vector {
    let gamma = geodesic(from, to, TB.connection)
    return parallel_transport(v, gamma, TB.connection)
  }
}
```

## Authority boundary

TENSOR-µ owns derived tensor-bundle projection geometry. VECTOR-µ remains the rank-1/vector provider under that contract. Neither TENSOR-µ nor VECTOR-µ can verify semantic truth, rewrite RelationISA, advance phase authority, or commit topology. Those remain Ch'en/Xul responsibilities.

See `docs/TENSOR_BUNDLE.md` for the full mathematical grounding and `data/grammar/kuhul.runtime-grammar.ebnf` for the runtime grammar surface.
