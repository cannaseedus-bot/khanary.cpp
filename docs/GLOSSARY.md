# KHANARY Glossary — Fold, Tuple, Arity, Phase

Human-readable authority for the fold / phase vocabulary. The **machine-readable**
authority is the `formal_model` exposed by the MCP surface
(`src/io/mcp_server.cpp`), and
`tools/grammar/mini_transpilers.py fold-geometry-check` asserts that this
document, those strings, and the executable implementation
(`src/config.h`, `src/kuhul/phase_engine.h`, `src/kuhul/fold_engine.*`,
`src/world/field_dag.cpp`) still agree. Change one, and the check fails until the
others follow.

## The durable law

> **Fold is structure; `SHF(h,τ)` is its phase-indexed state; K'UHUL phases operate over it.**

Refinement: the bounded neighborhood is the structure being interpreted. A
K'UHUL semantic fold is the algebraic interpretation of that neighborhood under
an active target algebra. Phase selects the interpretive position; Xul collapses
the interpreted result.

Fold vocabulary is namespace-scoped. A K'UHUL semantic fold is not the same
object as a KLSL, KXC, or KHLC fold:

| namespace | fold means |
| --- | --- |
| K'UHUL semantic fold | algebraic interpretation/catamorphism over a semantic neighborhood |
| KLSL fold | shader/lane lowering or kernel-side reduction/projection surface |
| KXC fold | execution/control lowering surface unless explicitly bound to K'UHUL semantics |
| KHLC fold | host/compiler lowering surface unless explicitly bound to K'UHUL semantics |

Only the K'UHUL semantic fold carries the `N(k) --[α, θ]--> B --Xul--> k'`
contract. Backend/compiler folds may preserve or project that contract, but they
do not define it.

## Fold Binding

**Fold Binding = (α, θ, backend, entrypoint, contract).** It is the
contract-carrying link between a K'UHUL semantic fold at a phase and a concrete
executor. K'UHUL owns the fold and phase; the backend supplies the entrypoint,
dispatch shape, and evidence that it satisfies the declared contract.

```
Fold → Phase Binding → Backend Binding → Executor
K'UHUL Fold → Fold Binding → KXC → HLSL → GPU
Bind_backend(α_θ) = K_θ^backend
valid iff K_θ^backend ⊨ Contract(α_θ)
```

This is a binding, not an identity. An HLSL kernel may implement a phase
contract for a semantic fold; it does not become the fold or the phase. The
registry at `data/grammar/fold-bindings.contracts.json` is checked by
`tools/check_fold_bindings.py`.

## Fold Kernel

A **Fold Kernel** is the JSON lifecycle record for a fold as it moves from
unbound observation to autonomous routing. It is state-dependent: `@learned`
appears when the N-Gram Burner owns a reusable pattern, `@binding` appears when
the Binding Resolver attaches an executor contract, `@evidence` appears when
the Observation System has proof counts, and `@policy`/`@route` appear when
XCFE can choose a consequence.

```
FoldKernel =
  core(@fold,@phase,@semantic,@state)
  + learned?
  + binding?
  + evidence?
  + policy?
  + route?
  + ui?
```

Ownership stays separated:

| field | owner |
| --- | --- |
| `@fold`, `@phase`, `@semantic` | K'UHUL semantic layer |
| `@learned` | N-Gram Burner |
| `@binding` | Binding Resolver |
| `@evidence` | Observation System |
| `@policy` | Policy Registry |
| `@route` | XCFE |
| `@ui` | projection/UI only |

XCFE owns the route; `@ui` is non-normative. The grammar is
`data/grammar/fold-kernel.ebnf`, the lifecycle contract is
`data/grammar/fold-kernel.contracts.json`, and validation is performed by
`tools/check_fold_kernel.py`.

## Fold Lattice

The **Fold Lattice** is the product-graph slice that separates semantic position from evolutionary competence while preserving the full runtime coordinate:

```
K_t = (F_h, phi, T, s, sigma, X_t)
K = (theta, sigma)
G_fold_lattice = G_phase □ G_competence
```

`F_h` is the fold/hypergraph structure, `T` is the active track, `s` is the folded/unfolded navigation state, and `X_t` is mutable runtime state. `theta` is the K'UHUL phase axis: `Pop`, `Wo`, `Yax`, `Sek`, `Ch'en`, `Xul`. `sigma` is the competence axis: `unbound`, `observed`, `learned`, `bound`, `validated`, `autonomous`. This makes learning phase-granular: Pop can become autonomous while Sek still calls a teacher callback.

The lattice is advisory, not authoritative. Metrics recommend competence movement; XCFE Policy authorizes state/route movement. `@route_hint` is only a hint; XCFE remains route authority. **Fold != Xul**: `@fold_state` changes bounded/expanded navigation, while Xul remains the commit phase. The proof boundary stays hard: Sek proposes, Ch'en proves, XCFE authorizes, Xul commits. The grammar is `data/grammar/fold-lattice.ebnf`, the contract is `data/grammar/fold-lattice.contracts.json`, fixtures live in `data/conformance/fold_lattice.json`, and validation is performed by `tools/check_fold_lattice.py`.

## XCFE Policy

An **XCFE Policy** is the declarative control object for Fold Kernel lifecycle movement. A policy is data; XCFE is the evaluator. The policy declares evidence inputs, a total confidence expression, thresholds, routes, and optional guards. It does not execute folds, store state, or create semantic truth.

```
Pi(Evidence, State, Event) -> (State, Route)
Policy != Control != Execution
```

The grammar is `data/grammar/xcfe-policy.ebnf`, the machine contract is `data/grammar/xcfe-policy.contracts.json`, fixtures live in `data/conformance/xcfe_policy.json`, and validation is performed by `tools/check_xcfe_policy.py`. Fold Kernel `@policy` fields point at this XCFE-owned grammar instead of defining an inline policy subset.

## Terms

### Fold — structure plus interpretation (not a tuple)

```
Fold:{contexts, relations, nodes} -> h
N(k) --α,θ--> B --Xul--> k'
```

A bounded semantic context around a handle `h`, built from three inputs
(contexts, relations, nodes). The *operation* has declared arity 3. A fold is
structure; it is not itself a tuple.

At the semantic layer, a fold also names the active interpretation algebra:
`α : F(B) -> B`. The same structural signature `F` can be interpreted as
reasoning, retrieval, compression, or an execution route without rewriting the
underlying neighborhood. This keeps the invariant split explicit:

| layer | meaning |
| --- | --- |
| Signature `F` | What semantic constructors and relation verbs exist |
| Algebra `α` | What those structures mean in the target domain |
| Phase `θ` | Where that interpretation sits in the K'UHUL cycle |

Compressed contract:

```
N(k) --[α, θ]--> B --Xul--> k'
```

### SHF — the phase-indexed state

```
SHF(h,τ) = <h, B_τ[h,*], N_τ(h), W_τ, State(h)>
```

A 5-tuple: the handle, the Book row for `h` at phase τ, the neighbourhood at τ,
the weights at τ, and the node state. Incidence is preserved by *indexing* it
with τ rather than flattening it. A tuple describes the ordered state a fold
carries — it is **not** the fold's whole semantics: the typed relations live in
the graph outside the tuple (a tuple `⟨A,B,C⟩` does not say
`A @requires B`, `B @supports C`).

### n-ary — arity of an operation/relation

`n-tuple` = n-part **state**; `n-ary` = n-part **operation/relation**.
`Fold` is 3-ary. `TRAVERSE` is 6-ary. Neither of those is the fold engine.

### FoldEngine — binary node fold + explicit phase

```
horizontal_fold(node_a, node_b, phase)   Δθ = 0,    weight = cos(0)   = 1.0   (no XCFE authorization)
vertical_fold(node, from_phase, to_phase) Δθ = π/3, weight = cos(π/3) = 0.5   (XCFE-gated)
```

`vertical_fold` accepts **exactly one** phase step; non-primitive steps are
rejected with `dist != 1`.

> **The engine's fold is binary + phase. It is not the semantic `Fold` arity.**
> Conflating the two is the vocabulary drift this glossary exists to prevent.

### Unfold — addressable members, bounded by construction

```
Unfold(h,τ) = { e | B_τ[h,e] ≠ 0 }
```

Boundedness is part of the **semantics**, not merely the implementation: cyclic
Books make unrestricted recursive expansion non-terminating. Operational
traversal is constrained by

```
TRAVERSE(start, relation_type, direction, depth, limit, predicate)
```

and the runtime resolves misses with `bounded_recursive_unfold_on_miss`.

### Phase — deterministic position on C6

`PHASE_COUNT = 6`, at angles `0, π/3, 2π/3, π, 4π/3, 5π/3` (`src/config.h`), with

```
arc_weight(from, to) = cos(angle(to) − angle(from))
```

| Step | Δθ | weight |
| --- | --- | --- |
| vertical (adjacent phase) | π/3 | cos(π/3) = 0.5 |
| horizontal (same phase) | 0 | cos(0) = 1.0 |

A K'UHUL phase is a **deterministic geometric/control position**. It is not a
Markov state.

### XCFE Node ISA — the class coordinate

An XCFE node separates four things that must not be collapsed:

> **Class ≠ Operator ≠ Phase** — and class is not the Relation ISA (`RelationFamily` is a separate ISA).

| coordinate | meaning |
| --- | --- |
| **class** | what kind of control responsibility the node has |
| **operator** | what control operation it requests |
| **flow_phase** | **where it appears** — where the construct was encountered |
| **authority_phase** | **where its semantics are governed** |

`authority_phase = CanonicalPhase(class)` — **always**. But

```
flow_phase ≠ authority_phase      is legal
```

and must never be forced equal: the construct is encountered in one phase and governed
in another.

The closed **14-class** set, with each class's canonical phase
(`src/kuhul/xcfe_node_class.h` is the single definition):

| class | canonical phase |
| --- | --- |
| CONDITION, ROUTING, CAUSAL, STRUCTURE | Wo |
| SELECTION | Yax |
| EXECUTION, CAPABILITY, TRANSFORM | Sek |
| VERIFICATION, CONSEQUENCE | Ch'en |
| COMMIT | Xul |
| MEMORY, NEGATION, KUHUL_RUNTIME | Pop |

Worked examples of the two coordinates legitimately differing:

```
if        class CONDITION    flow Pop   authority Wo
why       class CAUSAL       flow Yax   authority Wo
then      class CONSEQUENCE  flow Wo    authority Ch'en
```

There is deliberately **no `Policy` class**: operation authority is XCFE's own gate
(`authorize_phase_transition` + the `xcfe_required` edge flag), so a Policy class would
double-encode it. That is why class is **not the Relation ISA** — relations keep their own
`RelationFamily`, and authority keeps its own mechanism.

### The dispatch chain: class, micronaut class, executor

A Micronaut is a resolver/executor that XCFE **calls** — it is never an `XcfeNodeClass`.
There are two different meanings of "class", and they must stay distinct:

```
XCFE
  class: VERIFICATION          <- WHAT kind of control work is needed
  operator: VERIFY
  authority_phase: Ch'en       <- WHERE it is governed
        |
        +- requires: verification
               |
               v
        MicronautClass: VERIFY-µ   <- WHO is qualified (capability family)
               |
             Yax / W·C·R           <- WHICH µ instance gets it
               |
             selected µ
               |
        deterministic | shader/GPU | parser | database | tool | LLM   <- HOW it runs
```

> **XCFE class** says WHAT kind of control operation is needed.
> **Micronaut class** says WHO is qualified to resolve it.
> **W·C·R** says WHICH µ gets it.
> **Executor** says HOW it actually runs.
> **K'UHUL phase** says WHERE in the cycle it is governed.

A class is a capability family, not an individual micronaut: several µ instances compete
within one family, and eligibility is

```
Eligible_µ(N) = { µ | capabilities(µ) ⊇ requirements(N) }
µ* = argmax over Eligible_µ of  W_µ C_µ R_µ(Q)
```

`XcfeNodeClass` names a *kind of control work*; a micronaut/track id names a *specialist*.
The drift guard fails if any node class collides with a micronaut id, and reports any
requirement that no track's capabilities can satisfy.

The Ω gate turns the entropy measurement into a **blocking observation**:

```
|Eligible| = 0                    -> MISS_capability          -> Factory
|Eligible| > 1 and Ĥ_route > 0.90 -> MISS_ambiguity           -> research/unfold -> requery
                                     MISS_ambiguity_exhausted -> unresolved (never a commit)
```

Entropy is **not a seventh phase, a node class, or an authority class**. It is an observer
over runtime state — `Entropy: F_τ → ℝ`, not `F_τ → F_{τ+1}`. It can refuse premature
collapse, and it can never grant it. The retry/evidence budget is bounded
(`kAmbiguityMaxAttempts`) and exhaustion stays explicitly unresolved rather than bypassing
the gate and committing anyway.

An operator record keeps **two separate namespaces**:

```json
{"operator": "VERIFY", "class": "VERIFICATION",
 "role": "verification-gate", "requires": ["VERIFY"],
 "flow_phase": "Ch'en", "authority_phase": "Ch'en"}
```

| field | job |
| --- | --- |
| `class` | WHAT kind of XCFE node |
| `role` | WHAT semantic/control role it plays (descriptive metadata) |
| `requires` | WHAT capabilities execution requires (capability ids) |
| `flow_phase` | WHERE encountered/activated |
| `authority_phase` | WHERE governed |

> **role ≠ requires.** Eligibility SHALL be derived only from explicit
> `requires(op)` versus declared `capabilities(µ)`:

```
Eligible_µ(op) = { µ | requires(op) ⊆ capabilities(µ) }
```

---

## SVG-Tensor Format

An SVG file that carries semantic field state as `k:*` namespace attributes on its elements, making it both a valid SVG document and a loadable source for `parse_svg_tensor()`.

**Namespace declaration** (required on `<svg>` root):
```xml
xmlns:k="https://khanary.io/tensor"
```

**Node element** (any SVG element with `k:track_id` or `k:kind`):
```xml
<circle k:track_id="ELIZA" k:phase="Wo" k:kind="track"
        k:entropy="0.34" k:geodesic_entropy="0.55"
        k:unruh_temp="0.02" k:divergence="-0.4"
        k:circumradius_R="2.5" k:inradius_r="1.0"
        k:x="1.0" k:y="2.0" k:z="0.5"
        k:weight="0.9" k:confidence="0.75"
        k:q_vec="0.5,0.0,0.0" k:k_vec="0.0,0.5,0.0" k:v_vec="0.34,0.9,0.75"/>
```

**Edge element** (any SVG element with both `k:edge_from` and `k:edge_to`):
```xml
<path k:edge_from="ELIZA" k:edge_to="Librarian"
      k:verb="@delegates_to" k:fold_type="vertical"
      k:weight="0.5" k:delta_theta="60.0"/>
```

**Disambiguation rule**: the parser requires a whitespace character (space, tab, newline, CR) immediately before `k:` to exclude `xmlns:k="..."` declarations and `<k:elem>` tag names from attribute extraction.

**Round-trip**: `FieldDag::to_svg()` emits `xmlns:k` on the `<svg>` element and full `k:*` attrs on every non-ontology node circle and edge path, so its output is valid SVG-Tensor input for `parse_svg_tensor()`. Source: `src/io/svg_tensor.h/.cpp`.

---

## GAS MeshNet Control Plane

A Google Apps Script web app that acts as an HTTPS control-plane node for MeshNet peers. Source: `tools/meshnet_gas_control_plane.gs`; deployment metadata: `tools/meshnet_gas_control_plane.url.json`.

The preferred deployment executes as the **user accessing the web app**. Google sign-in is therefore the outer authority boundary: user-owned Drive, Sheets, and table data remain in that user's Google session, and KHANARY receives only routed outputs after Google admits the request. Anonymous health probes are expected to return Google Accounts sign-in HTML rather than MeshNet JSON.

Peer registration may link to the Google session through `Session.getTemporaryActiveUserKey()`, a script-local temporary user key. Email storage is opt-in (`MESHNET_STORE_GOOGLE_EMAIL=true`) because Apps Script can return blank email strings depending on authorization context. The linked Google session is provenance/routing identity only; it is not semantic or fact authority.

After Google session admission, the script's MeshNet layer still applies its own register handshake, HMAC/shared-secret checks, session tokens, nonce replay guard, and optional code-hash allowlist.

---

## Relativistic Event Projection for Key-Grammar Memory

A relativity event is the model for a key-grammar memory event: an observed occurrence with a track/phase/time address. In KHANARY this is serialized as `key.grammar.memory_event.v1` and may carry a `key_grammar.hyperbolic_event.v1` projection.

| relativity concept | KHANARY projection |
| --- | --- |
| event at a definite place and time | `memory:event:<kind>:<epoch>-<seq>` with `track`, `phase`, and source |
| four-vector coordinate | hyperbolic event `coords = (x0,x1,x2,x3)` or `hyperbolic_metrics` |
| causal interval class | `timelike_candidate`, `lightlike_boundary`, `spacelike_candidate`, or `unclassified` |
| causal past/future relation | prompt/model_response/plan parentage and candidate continuation branches |
| frame of reference | micronaut track + K'UHUL phase + optional Lorentz boost projection |

This is still projection evidence. A prompt, model response, plan, or hyperbolic event can guide routing and recall, but it is not a fact commit by itself. `fact_authority=false` and `semantic_authority=false` remain the default until Ch'en verifies and Xul commits the result.

---

## KQL — User-Scoped Mesh Database

KQL is the planned KHANARY Query Language for localizing the online GAS mesh into `KHANARY.CPP`. It is SQLite-like in role, but its first backend is KHANARY-native append-only event tables plus the existing vector/gram engine.

The key scope is `user_mesh_id`, derived from provider/session identity plus a local salt. Every table read/write is user-scoped by default. Cross-user reads require an explicit shared-mesh grant. See `docs/KQL.md`.

KQL tables separate evidence from authority: prompts, model responses, plans, peer rows, vector candidates, and hyperbolic events are memory/event evidence; only Ch'en-verified and Xul-committed rows enter fact/topology authority tables.

KQL → GAS also defines the lightweight online deployment pattern: a static landing page distributes the install/connect button, GAS supplies the signed-in user orchestration plane, and KQL maps the session into local user-scoped tables. Static API model manifests can be contributed through the mesh as `static_api_model.contribute.v1` provider candidates; they remain routing evidence until verified.

---

## Tensor bundle grounding

The `TensorSubstrate` map is a **fiber bundle over the SVG-3D base manifold**.

| role | K'UHUL object |
| --- | --- |
| Base point (x ∈ M) | `FieldNode.x, y, z` |
| Fiber at x | `TensorSubstrate.shape = {N, stride_floats}` |
| Fiber type | rank-(p,q) tensor space V^⊗p ⊗ (V\*)^⊗q |
| Connection ∇ | Phase θ — C6 cycle, `arc_weight = cos(Δθ)` |
| Geodesic | `dispatch_geodesic_propagator()` finds curves minimising entropy growth under ∇ |
| Gram G | tensor product rank — multi-token keys of rank p ARE V^⊗p (exact) |
| Fold F | structurally cotangent-shaped at the algebra level; NOT a continuous dual space |
| Projection Π | closer to interior product ι_v: Ω^k → Ω^(k-1) than to exterior derivative d |

**Critical split:** Phase θ is the **connection** (the rule for parallel transport).
`dispatch_geodesic_propagator()` finds **geodesics** (paths selected by that rule).
Connection ≠ geodesic; they operate at different levels.

See `docs/TENSOR_BUNDLE.md` for full mathematical grounding.

---

## Tensor Substrate Binding Layer

### TensorSubstrate — passive GPU buffer descriptor

`src/world/tensor_substrate.h`

```
TensorSubstrate { tensor_id, shape, strides, dtype, residency,
                  provider_identity, physical_buffer_allocation }
```

A **descriptor**, not an allocator. It records the shape, dtype, and device
residency of a tensor buffer without owning the memory.
`physical_buffer_allocation` is an opaque backend handle (raw `ID3D11Buffer*`
for D3D11; `void*` for backend agnosticism); it stays **null** until the
substrate is explicitly promoted from `HOST_RAM` to a device.

| field | job |
| --- | --- |
| `tensor_id` | monotonically increasing id across all substrates in a FieldDag |
| `shape` | e.g. `{N, stride_floats}` — populated by first dispatch |
| `dtype` | FLOAT32 / FLOAT16 / INT8 / BINARY_SDR |
| `residency` | HOST_RAM → GPU_VRAM_D3D11 after first geodesic propagator run |
| `provider_identity` | matches `provider_id` on the GML node that created this substrate |
| `physical_buffer_allocation` | null until buffers are made persistent (future work) |

**Authority boundary:** a `TensorSubstrate` with a geometry-source `provider_identity`
(e.g. `"brain4-expert-34"`) is **projection-only**. It MUST NOT be promoted to a
`MicronautEntry` in the registry. Only `TrackLoader` (JSON track files) and
`Factory::propose_create()` may create `MicronautEntry` objects.

### VectorMuMicronaut — active projection operator

`src/micronaut/vector_mu.h`

```
VectorMuMicronaut { micronaut_id, capability_verb, static_weight W,
                    confidence C, numerical_projection_vector }
  ExecuteProjection(target_substrate, context) → result_substrate
```

An **executable specialist** that projects state streamlines across a
`TensorSubstrate` field. Holds WCR epistemic weights (W, C) and a numerical
projection vector. `ExecuteProjection` returns a new bounded result substrate
without mutating the target — zero pattern inflation.

The concrete GPU implementation wires `ExecuteProjection` to
`FieldDag::dispatch_geodesic_propagator()`. `VectorMuMicronaut` itself is
geometry-layer only; it does not dispatch, commit, or gate.

### MicronautContext — execution environment snapshot

```
MicronautContext { active_scope, environmental_temperature, trajectory_generation_id }
```

Passed to `ExecuteProjection` at call time. `environmental_temperature` maps to
`kb_boltzmann` in the geodesic propagator cbuffer.
`trajectory_generation_id` identifies the CSO / shader variant to select.

### provider_id — GML binding token

A string attribute on a GML node (and edge) that names the
specialist or substrate family responsible for that position.

```
node [ id 0 provider_id "brain4-expert-34" ... ]
```

At `load_gml()` time, each unique `provider_id` value creates one
`TensorSubstrate` entry in `FieldDag::substrates_` (residency=HOST_RAM, shape
unpopulated). After `dispatch_geodesic_propagator()` completes, all substrates
are promoted to `GPU_VRAM_D3D11` with shape filled.

Canonical format: `"{run_name}-expert-{id:02d}"` for geometry brain nodes;
`"VECTOR-u"` for semantic track telemetry nodes (see `test_gml_parser.cpp`).

### Binding chain

```
export_brain_gml.py
  → brain4/brain.gml  (30,628 nodes, each with provider_id "brain4-expert-NN")

FieldDag::load_gml("brain4/brain.gml")
  → FieldNode::provider_id populated
  → substrates_["brain4-expert-NN"] created (HOST_RAM, 61 entries for 61 experts)

FieldDag::dispatch_geodesic_propagator(dev, cso, beta_inv)
  → geodesic_entropy computed on GPU for all 30,628 nodes
  → substrates_[*].residency = GPU_VRAM_D3D11
  → substrates_[*].shape = {N, sizeof(GpuFieldNode)/4}

VectorMuMicronaut::ExecuteProjection(substrate, ctx)
  → result_substrate (bounded projection, no mutation of source)
```

Semantic similarity may *recommend*; capability declarations *authorize* eligibility;
W-C-R ranks only within the authorized candidate set. A `requires` of `[]` means the
operator lowers to a native deterministic XCFE op and needs no Micronaut at all —
inventing one merely to satisfy the architecture would make the system less deterministic.

> **XCFE does not call a Micronaut by name. XCFE declares required control capability;
> eligible Micronauts satisfy that contract; W-C-R chooses among them; the selected
> Micronaut chooses or invokes the executor under XCFE authority.**

Eligibility gates *before* scoring (`request_router.cpp` admits a track to the candidate
set only when `evaluate_eligibility(...).capable`), and an empty eligible set is the
correct, named outcome:

```
W·C·R never promotes an unqualified specialist.
Eligible_µ(op) = ∅  ->  capability_miss = true, dispatch_block_reason = "MISS_capability"
                        status = "miss_capability"
```

Capability gaps are recorded as a formal artifact
(`data/grammar/control-capability-gaps.jsonl`, generated by
`mini_transpilers.py capability-gap`) with

```
Gap(op) = Req(op) − ⋃_µ Cap(µ)
```

and a `resolution` per gap: `satisfied`, `deterministic-executor` (the lowering is a
native XCFE op, so no specialist is needed), or `micronaut-required`. `advisory_near_tracks`
is a **hint for a human decision** (promote a capability vs. add a Micronaut vs. correct
the requirement declaration) and never a routing decision.

### PH model — measurement, not definition

A phase-type distribution `PH(α,S)` is the time-to-absorption of a continuous-time
Markov process with *m* transient phases and one absorbing state. It is an
**optional observational model over timestamped runtime events**:

```
K'UHUL execution → timestamped journal → latency observations → PH(α,S)
```

It would answer *"how long does a candidate stay unresolved before a terminal
outcome?"* — never *"what is a phase?"*.

Status: **not applicable yet.** Journal events record
`{channel, seq, key, value, phase, authority}` with **no timestamp**, so `α`/`S`
cannot be fitted. PH is also light-tailed, so commit latency would likely need a
Coxian chain with many phases.

### Commit model (adjacent to PH — not PH)

The confidence contract is `Ch'en_pending` → `Xul_commit`, with
`BAD_ANSWER` / `GOOD_ANSWER`
(`data/manifest/micronaut.manifest.json`). Absorption here is a **gated
decision** — feedback-gated and XCFE-authorized — not a probabilistic absorption.

## Authority hierarchy

```
GLOSSARY.md             human semantic contract        (this file)
      ↓
formal-model strings    machine-readable definition    (src/io/mcp_server.cpp)
      ↓
fold-geometry-check     invariant / drift verification (tools/grammar/mini_transpilers.py)
      ↓
phase_engine · fold_engine · FieldDag · config.h       executable implementation
      ↓
tests                   KHANARY_*_TEST
```

```powershell
py -3 tools\grammar\mini_transpilers.py fold-geometry-check
```
