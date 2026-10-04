# KHANARY.CPP — Single-Executable Specification

Single native Windows executable: GPU inference + manifest runtime + micronaut engine + MCP gateway.
Target: HD 4600 (DirectML, D3D11), 1792 MB resident ceiling, Windows 10 x64.

## Revelation — Native Semantic Fold Runtime

K'UHUL now declares what the math was already doing:

- **Meaning is context-distributed**: a token points to `Book(token)` / semantic neighborhood `N(token)`.
- **Fold is structure plus interpretation**: the neighborhood is the bounded structure; the semantic fold is the active algebra `α : F(B) -> B` that assigns meaning to that structure under the current phase.
- **Signature, algebra, and phase stay separate**: `F` declares constructors/relations, `α` interprets them, and `θ` positions that interpretation in the Pop→Wo→Yax→Sek→Ch'en→Xul cycle.
- **Participation topology is first-class** (`B` incidence), with **pattern strength isolated** (`W_E`).
- **Distributional semantics remains independent** as a context-similarity channel.
- **Grammar boundary is explicit** in `data/schema/kuhul.xsd` (structure contract; runtime authorization remains in XCFE/Xul).

---

## Build Target

```
KHANARY.exe
  — no installer, no service, no node_modules
  — ships with a /data directory (manifests, shaders, tracks)
  — listens on 8764 (MCP) + 8787 (JSON-RPC) at startup
  — GPU device selected at launch via DXGI adapter enumeration
```

Toolchain: MSVC 2022+, C++20, CMake 3.28+, Windows SDK 10.0.26100, DirectML 1.15+.

---

## Source File List

### Entry & Config
| File | Purpose |
|---|---|
| `src/main.cpp` | CLI args, device init, server start, shutdown |
| `src/config.cpp/h` | Port map, paths, GPU limits, phase constants |

### Manifest System
| File | Purpose |
|---|---|
| `src/manifest/manifest_registry.cpp/h` | key→array O(1) registry; load/resolve/expand |
| `src/manifest/manifest_loader.cpp/h` | Parse css/kuhul/words/micronaut .manifest.json |
| `src/manifest/scx_resolver.cpp/h` | Recursive key expansion; cycle detection; lazy depth limit |

### K'uhul Runtime (Phase Engine)
| File | Purpose |
|---|---|
| `src/kuhul/phase_engine.cpp/h` | C₆ phase cycle (Pop→Wo→Yax→Sek→Ch'en→Xul), arc weights |
| `src/kuhul/fold_engine.cpp/h` | Horizontal fold (Δθ=0), vertical fold (Δθ=π/3), cost table |
| `src/kuhul/kuhul_vm.cpp/h` | Bracket program executor: [Pop]/[Wo]/[Yax]/[Sek]/[Ch'en]/[Xul] |
| `src/kuhul/xcfe_executor.cpp/h` | @if/@then/@else, @for, @from, @as, @cause, @direct, @op, @fn, phase-gate enforcement (the op library receives a recursive evaluator) |
| `src/kuhul/gram_store.cpp/h` | KXML gram nodes; Pop=gramRead, Xul=gramWrite, checkpoint; append log at `data/journal/gram-events.jsonl`; prompts/model responses/plans are key-grammar memory events |

### Grammar Contracts
| File | Purpose |
|---|---|
| `kuhul-grammar.ebnf` | K'UHUL v8 standalone grammar: tensor bundles, Minkowski/hyperbolic events, Einstein contraction, holographic mesh boundaries, and CSO/CDATA kernel dispatch |
| `data/grammar/kuhul.runtime-grammar.ebnf` | Canonical runtime grammar surface; mirrors the v8 event/CSO/contraction constructs while preserving the full phase/fold/XCFE contract |
| `data/grammar/fold-bindings.contracts.json` | Fold Binding registry: `(α, θ, backend, entrypoint, contract)` records that bind K'UHUL semantic folds/phases to backend executors without transferring semantic authority |
| `data/grammar/fold-space.ebnf` | Fold Space grammar: umbrella runtime positions `p_t = (h, phi, T, s, X_t)` / `p_t = (h, phi, T, s, sigma, X_t)` plus horizontal/vertical/radial movement and typed distance projections |
| `data/grammar/fold-space.contracts.json` | Machine-readable Fold Space contract: C6 authority vs S1 projection, Book(h)==N(h), movement-axis rules, typed distance metrics, and navigation pipeline ordering |
| `data/schema/fold-event.schema.json` | Draft 2020-12 JSON Schema for Fold Event rows; events log proposed/observed movement and carry XCFE decision state without owning transition authority |
| `data/grammar/fold-event.contracts.json` | Machine-readable Fold Theory/Field/Event contract: `FrakturF=(F_K,M_F,E_F,delta_Pi,B,Exe)`, Fold Field projection laws, event-axis mapping, and topology commit requirements |
| `data/grammar/fold-kernel.ebnf` | Fold Kernel concrete/abstract grammar: state-dependent JSON lifecycle records for learned folds |
| `data/grammar/fold-kernel.contracts.json` | Machine-readable Fold Kernel lifecycle contract: state requirements, evidence invariants, XCFE policy linkage, Fold Lattice linkage, and XCFE route authority |
| `data/grammar/fold-lattice.ebnf` | Fold Lattice grammar: full coordinate `K_t = (F_h, phi, T, s, sigma, X_t)` with checkable `K = (theta, sigma)` phase/competence slice |
| `data/grammar/fold-lattice.contracts.json` | Machine-readable Fold Lattice contract: θ×σ axes, track/fold-state/runtime-state fields, callback/substrate rules, bounded metrics, and transition recommendations that cannot bypass XCFE Policy |
| `data/grammar/xcfe-policy.ebnf` | XCFE Policy grammar: policy is data; XCFE is the evaluator for `Pi(Evidence, State, Event) -> (State, Route)` and `Policy != Control != Execution` |
| `data/grammar/xcfe-policy.contracts.json` | Machine-readable XCFE Policy contract: evidence fields, total confidence expression language, weight/threshold invariants, routes, guards, and transition cases |
| `data/conformance/fold_space.json` | Positive/negative Fold Space fixtures covering valid positions, movement-axis constraints, Book boundaries, typed distances, and projection authority rules |
| `data/conformance/fold_event.json` | Positive/negative Fold Event fixtures covering track/phase/fold-state moves, competence candidates/upgrades, field gradients, topology candidates, topology commits, and schema failures |
| `data/conformance/fold_kernel.json` | Positive/negative Fold Kernel fixtures covering unbound→autonomous progression and invalid evidence/policy/route/binding cases |
| `data/conformance/fold_lattice.json` | Positive/negative Fold Lattice fixtures covering valid phase-granular runtime lattices, callback requirements, Xul policy gates, and bounded metric rules |
| `data/conformance/xcfe_policy.json` | Positive/negative XCFE Policy fixtures covering default scoring, explicit derive expressions, guards, route transitions, and malformed policies |
| `tools/check_kuhul_grammar.py` | Conformance checker for standalone/runtime grammar drift and the fold namespace boundary |
| `tools/check_fold_bindings.py` | Conformance checker for Fold Binding phase contracts, HLSL entrypoints, dispatch shape, and the law `K_θ^backend ⊨ Contract(α_θ)` |
| `tools/check_fold_space.py` | Conformance checker for Fold Space records; validates full navigation coordinates, horizontal/vertical/radial movement, typed distance declarations, and C6/S1 authority boundaries |
| `tools/check_fold_event.py` | Conformance checker for Fold Event records; validates the JSON Schema plus event-axis laws, XCFE transition boundary, field-gradient boundary, and topology commit preconditions |
| `tools/check_fold_kernel.py` | Conformance checker for Fold Kernel lifecycle records; verifies state-dependent field growth, XCFE confidence derivation, and binding-registry compatibility |
| `tools/check_fold_lattice.py` | Conformance checker for Fold Lattice product-graph records; validates θ×σ coordinates, callback/substrate rules, metric bounds, and transition recommendations |
| `tools/check_xcfe_policy.py` | Conformance checker for XCFE Policy records; validates total expression parsing, policy invariants, guard typing, and lifecycle transition output |
| `docs/SEMANTIC_FOLDS.md` | Declares that K'UHUL semantic folds are algebraic/catamorphic folds; KLSL/KXC/KHLC folds are backend/compiler lowering folds unless explicitly bound to the semantic contract |

### KLSL Compiler
| File | Purpose |
|---|---|
| `src/klsl/klsl_lexer.cpp/h` | Tokenize .kuhul/.khl source |
| `src/klsl/klsl_parser.cpp/h` | AST: Cluster/Model/Pipeline/Tensor nodes |
| `src/klsl/klsl_ir.cpp/h` | SSA geometric IR |
| `src/klsl/hlsl_codegen.cpp/h` | IR → HLSL cs_5_0 compute shader text |
| `src/klsl/wgsl_codegen.cpp/h` | IR → WGSL (fallback/WebGPU path) |
| `src/klsl/d3dcompile_bridge.cpp/h` | D3DCompile HLSL text → ID3DBlob at runtime |

### GPU Layer (D3D11 / DirectML)
| File | Purpose |
|---|---|
| `src/gpu/d3d11_device.cpp/h` | DXGI adapter enum, D3D11 device + context, HD 4600 quirks |
| `src/gpu/dml_ops.cpp/h` | DirectML: GEMM, Conv, Attention, Softmax, Reduce |
| `src/gpu/tensor_buffer.cpp/h` | R32_UINT/R32_FLOAT buffer alloc, staging, upload/readback |
| `src/gpu/compute_dispatch.cpp/h` | CS dispatch, barrier management, UAV binding |
| `src/gpu/dds_loader.cpp/h` | DDS shard streaming: SDDS magic, INT8 tensor records, CRC |
| `src/gpu/sdds_format.cpp/h` | SDDS chunk reader + `sdds_crc32`; shared by STBMap |
| `src/gpu/hot_swap.cpp/h` | Resident weight eviction within 1792 MB ceiling |
| `src/gpu/gram_gpu_scorer.cpp/h` | GPU batch scoring of GramStore via `attention.hlsl`; geodesic gramVector packing |
| `src/gpu/stb_map.cpp/h` | STBMap binary format (magic "STBM"): Field/Fold/Gram/Geodesic sections; CRC32 per section; parallel to SDDS |

### Inference Engine
| File | Purpose |
|---|---|
| `src/inference/tokenizer.cpp/h` | 1024-token glyph vocab; hash→id; BOS/EOS/TOOL_CALL sentinels |
| `src/inference/moe_router.cpp/h` | Router logits → top-k expert selection; routing_bias table |
| `src/inference/inference_engine.cpp/h` | Token→embedding→fold→expert→logits→sample pipeline |
| `src/inference/kv_cache.cpp/h` | KV-cache alloc, GPU QK-norm, sequence management |
| `src/inference/sampler.cpp/h` | Temperature, top-p, top-k sampling |

### Micronaut Runtime
| File | Purpose |
|---|---|
| `src/micronaut/micronaut_registry.cpp/h` | Register/dispatch micronauts by phase + governs |
| `src/micronaut/aiml_engine.cpp/h` | ELIZA/ALICE AIML 1.0 pattern match; template expansion |
| `src/micronaut/eliza_operators.cpp/h` | 16 reasoning operators (CAUSE/CONTRAST/.../DIRECT) Pop+Sek pairs |
| `src/micronaut/track_loader.cpp/h` | Load micronaut.semantic-tracks.v1 JSON; bind folds |
| `src/micronaut/gravity_field.cpp/h` | Gravity well scoring: established_nodes / distance² |
| `src/micronaut/factory.cpp/h` | Create/evolve/delete micronauts; confidence_floor gate |

### Cognitive Memory (Three-Store)
| File | Purpose |
|---|---|
| `src/memory/perception.cpp/h` | F_Q ephemeral field; reset at Xul; embed(Q, L_t) |
| `src/memory/confidence.cpp/h` | L_t persistent; Ch'en verify → Xul commit gate only |
| `src/memory/semantic_graph.cpp/h` | G_semantic topology; (⟁EVOLVE⟁)-gated node/arc mutation |
| `src/memory/scoring.cpp/h` | S_µ(Q) = W_µ · C_µ · R_µ(Q); ΔZ = ΔG_semantic ⊕ ΔC_earned |

### User-Scoped Mesh Database / KQL
| File | Purpose |
|---|---|
| `docs/KQL.md` | KHANARY Query Language contract for a local, user-scoped mesh database; SQLite-like query intent over KHANARY-native JSONL/JROM tables backed by the existing vector/gram engine |
| `src/io/kql.cpp/h` | Native KQL SELECT parser/evaluator surface for user tables, memory events, hyperbolic events, peer rows, vector candidates, static API model manifests, and Xul-committed fact/topology rows |
| `src/tests/test_kql.cpp` | KQL conformance: user isolation, vector-candidate filtering, hyperbolic event lookup, and authority table boundaries |

### World Graph
| File | Purpose |
|---|---|
| `src/world/field_dag.cpp/h` | Semantic field DAG: `FieldNode`/`FieldEdge`/`FieldDag`; GML overlay; `to_svg()` (emits `k:*` attrs); geodesic propagator dispatch |
| `src/world/gml_parser.cpp/h` | Hand-rolled GML text parser → `GmlGraph`; `provider_id` + geometry fields |
| `src/world/vector_projection.cpp/h` | `VectorProjectionContainer` gen1: GML topology → overlay JSON at `data/world/vector_projection.json`; hyperbolic key-grammar events as future-sheet four-vector routing projections |
| `src/world/tensor_substrate.h` | `TensorSubstrate` descriptor: shape, dtype, residency, `provider_identity`; `VectorMuMicronaut` projection operator |

### I/O & Servers
| File | Purpose |
|---|---|
| `src/io/json_rpc_server.cpp/h` | HTTP/JSON-RPC on 8787; XJSON/XCFE object surface |
| `src/io/mcp_server.cpp/h` | MCP gateway on 8764; 20 tools incl. kuhul_forge/code_parse/semantic_fold |
| `src/io/jrom.cpp/h` | JROM audit journal; append log at `data/journal/tool-events.jsonl`; event memory evidence for prompt/model_response/plan/hyperbolic_event rows |
| `src/io/request_router.cpp/h` | Route incoming requests → phase dispatch → micronaut |
| `src/io/svg_tensor.cpp/h` | SVG-Tensor source loader: `parse_svg_tensor()` / `load_svg_tensor()` → `SVGTensorGraph`; scans `k:*` namespace attributes without XML library |
| `src/io/stb_map_convert.cpp/h` | Converter chain: `svg_tensor_to_stb_map()` + `field_vectors_to_stb_map()` + `field_dag_to_stb_map()` inline wrapper |

### External Control Planes
| File | Purpose |
|---|---|
| `tools/meshnet_gas_control_plane.gs` | Google Apps Script MeshNet control plane for register/poll/send/peers/transpile; deployed as a user-session web app so Drive/Sheets authority remains with the signed-in Google user; peer registration links to a script-local Google session key with email storage opt-in |
| `tools/meshnet_gas_control_plane.url.json` | Deployment metadata for the GAS endpoint; records that anonymous probes are expected to receive Google Accounts sign-in HTML before Apps Script JSON endpoints run |
| `tools/meshnet_gas_landing.html` | Static landing/install page that links users to the GAS control plane without requiring a hosted app backend |
| `tools/meshnet_static_api_model.example.json` | Example provider manifest for `static_api_model.contribute.v1`; mesh-published API models are provider/routing evidence, not semantic authority |

### Utilities
| File | Purpose |
|---|---|
| `src/util/sha256.cpp/h` | SHA-256 for manifest CRC and SCXQ2 checksums |
| `src/util/json.cpp/h` | Minimal JSON parse/emit (no external dep) |
| `src/util/thread_pool.cpp/h` | Work-stealing pool for async fold execution |

---

## Data Directory (`/data`)

```
data/
  manifest/
    css.manifest.json         — ⟁CSS symbol → CSS declaration array
    kuhul.manifest.json       — ⟁K'uhul opcode → phase opcode strings
    words.manifest.json       — word/vocab → intents, phase refs, XCFE objects
    micronaut.manifest.json   — micronaut name → capability strings, phase refs

  drivers/phases/
    pop.json                  — Pop phase-driver op/emission contract
    wo.json                   — Wo phase-driver op/emission contract
    yax.json                  — Yax phase-driver op/emission contract
    sek.json                  — Sek phase-driver op/emission contract
    chen.json                 — Ch'en phase-driver op/emission contract
    xul.json                  — Xul phase-driver op/emission contract

  shaders/
    fold_compute.hlsl         — horizontal/vertical fold compute kernel
    svg_compute.hlsl          — SVG-3D tensor geometric projection kernel
    moe_router.hlsl           — expert router softmax + top-k
    attention.hlsl            — QK-norm + scaled dot-product attention
    dds_dequant.hlsl          — INT8 DDS weight dequantization

  tracks/
    gravity.semantic-tracks.v1.json
    ELIZA.semantic-tracks.v1.json
    ELIZA-1.semantic-tracks.v1.json
    intent-unresolved.semantic-tracks.v1.json
    Librarian.semantic-tracks.v1.json
    Factory.semantic-tracks.v1.json
    Coordinator.semantic-tracks.v1.json
    KNOWLED-1.semantic-tracks.v1.json

  alice/
    mp0.aiml … mp9.aiml       — ALICE AIML corpus

  weights/                    — user-supplied .scxqdds / .gguf shards (not shipped)
```

---

## Module Dependency Graph

```
main
  ├── config
  ├── d3d11_device → dml_ops → compute_dispatch → tensor_buffer
  ├── dds_loader → hot_swap
  ├── manifest_registry → manifest_loader → scx_resolver
  ├── phase_engine → fold_engine → kuhul_vm → xcfe_executor → gram_store
  ├── klsl_lexer → klsl_parser → klsl_ir → hlsl_codegen → d3dcompile_bridge
  ├── tokenizer → moe_router → inference_engine → kv_cache → sampler
  ├── micronaut_registry
  │     ├── aiml_engine → eliza_operators
  │     ├── track_loader
  │     ├── gravity_field
  │     └── factory
  ├── perception → confidence → semantic_graph → scoring
  └── json_rpc_server + mcp_server → request_router → jrom
```

---

## Phase Constants (compile-time)

```cpp
// C6 phase cycle angles (radians)
constexpr double PHASE_POP   = 0.0;
constexpr double PHASE_WO    = M_PI / 3.0;
constexpr double PHASE_YAX   = 2.0 * M_PI / 3.0;
constexpr double PHASE_SEK   = M_PI;
constexpr double PHASE_CHEN  = 4.0 * M_PI / 3.0;
constexpr double PHASE_XUL   = 5.0 * M_PI / 3.0;

// Arc weight: arc_w * cos(phase_j - phase_i)
// same-phase=+1.0, adjacent=+0.5, two-hop=-0.5, opposing=-1.0

// GPU ceiling
constexpr size_t HD4600_RESIDENT_CEILING_BYTES = 1792ULL * 1024 * 1024;

// Ports
constexpr uint16_t PORT_MCP      = 8764;
constexpr uint16_t PORT_JSON_RPC = 8787;
```

---

## Manifest Resolution (pseudocode)

```cpp
// O(1) key lookup, lazy recursive expansion
Value resolve(const std::string& key, const Context& ctx) {
    auto& node = registry.at(key);           // O(1) hash map
    if (node.is_cached()) return node.cache;

    Value result;
    for (const auto& elem : node.array) {
        if (registry.has(elem))              // expansion key
            result.merge(resolve(elem, ctx));
        else if (elem.starts_with('@'))      // declarative operator
            result.push(xcfe_executor.eval(elem, ctx));
        else
            result.push(elem);               // raw string
    }
    node.cache = result;
    return result;
}
```

---

## Inference Pipeline (pseudocode)

```cpp
// Token → phase → expert → output
Output run(const std::string& prompt) {
    auto tokens  = tokenizer.encode(prompt);          // 1024-glyph vocab
    auto phase   = phase_engine.enter(PHASE_POP);     // perceive
    auto gravity = gravity_field.score(tokens, phase);
    auto expert  = moe_router.select(gravity, top_k); // Wo: allocate
    auto hidden  = inference_engine.forward(tokens, expert, kv_cache);
    auto gate    = xcfe_executor.eval(verify_node, phase_ctx);  // Ch'en: XCFE gate over the fold
    return sampler.sample(logits);                      // Xul: emit
}
```

---

## CMake Sketch

```cmake
cmake_minimum_required(VERSION 3.28)
project(KHANARY CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(directx-headers CONFIG REQUIRED)
find_package(DirectML 1.15 REQUIRED)

add_executable(KHANARY
  src/main.cpp
  src/manifest/manifest_registry.cpp
  src/manifest/manifest_loader.cpp
  src/manifest/scx_resolver.cpp
  src/kuhul/phase_engine.cpp
  src/kuhul/fold_engine.cpp
  src/kuhul/kuhul_vm.cpp
  src/kuhul/xcfe_executor.cpp
  src/kuhul/gram_store.cpp
  src/klsl/klsl_lexer.cpp src/klsl/klsl_parser.cpp
  src/klsl/klsl_ir.cpp
  src/klsl/hlsl_codegen.cpp src/klsl/wgsl_codegen.cpp
  src/klsl/d3dcompile_bridge.cpp
  src/gpu/d3d11_device.cpp src/gpu/dml_ops.cpp
  src/gpu/tensor_buffer.cpp src/gpu/compute_dispatch.cpp
  src/gpu/dds_loader.cpp src/gpu/hot_swap.cpp
  src/inference/tokenizer.cpp src/inference/moe_router.cpp
  src/inference/inference_engine.cpp
  src/inference/kv_cache.cpp src/inference/sampler.cpp
  src/micronaut/micronaut_registry.cpp
  src/micronaut/aiml_engine.cpp src/micronaut/eliza_operators.cpp
  src/micronaut/track_loader.cpp
  src/micronaut/gravity_field.cpp src/micronaut/factory.cpp
  src/memory/perception.cpp src/memory/confidence.cpp
  src/memory/semantic_graph.cpp src/memory/scoring.cpp
  src/io/json_rpc_server.cpp src/io/mcp_server.cpp
  src/io/jrom.cpp src/io/request_router.cpp
  src/util/sha256.cpp src/util/json.cpp src/util/thread_pool.cpp
)

target_link_libraries(KHANARY PRIVATE
  d3d11 dxgi d3dcompiler
  DirectML
  directx-headers::directx-headers
)

target_compile_definitions(KHANARY PRIVATE
  UNICODE _UNICODE
  KHANARY_HD4600_CEILING=1879048192   # 1792 MiB
)
```

---

## What This Replaces

| Old (experimental) | New (KHANARY.exe) |
|---|---|
| Node.js kuhul-es runtime | `kuhul_vm.cpp` + `phase_engine.cpp` |
| PowerShell micronaut-v4 | `micronaut_registry.cpp` + `track_loader.cpp` |
| Python tools/gen.py | `inference_engine.cpp` |
| kuhul-server.js MCP | `mcp_server.cpp` |
| json_runtime (port 8787) | `json_rpc_server.cpp` |
| KLSL JS compiler | `klsl_*` modules |
| Loose AIML files | `aiml_engine.cpp` + `data/alice/` |

Total: ~35 .cpp files, ~8,000 lines estimated, 1 executable, 1 `/data` directory.
