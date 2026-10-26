# KHANARY JSON Runtime

The JSON runtime is the apps-host lane — a loopback-only C++ service that runs KXML programs, dispatches micronaut drivers, and routes tool calls. It deliberately cannot commit topology or claim semantic authority; everything requiring real Ch'en/Xul authority routes through the JSON-RPC (port 8787) or MCP (port 8764) lanes.

---

## Boot

The host boots from `data/host/app-host.manifest.json` via the C FFI surface:

```c
json_runtime_create()
json_runtime_boot(path_to_manifest)
json_runtime_run_inline(program_json)
json_runtime_run_file(path)
```

Four sub-systems initialise at boot:

| Sub-system | Role |
|---|---|
| `SCORegistry` | Alias → `{endpoint, path, sha256, data}` identity map. Loads from JSONL; resolves names; verifies SHA-256 before any dispatch. |
| `SidecarTable` | Format-directive router. Maps `@CDATA / @SVG / @XML` to declared CONVERT targets for `MicroXcfe::CONVERT` ops. |
| `MicroXcfe` | Deterministic projection executor. Runs `@state / @import / @control` programs. No phase gates, no topology commits. |
| `ModelCatalog` | Parses `drivers/runtime/models.toml`, selects the single model with `gguf_link = true`, and loads that GGUF into the process through the linked llama.cpp core before the listener starts. |

---

## REST surface

Loopback-only (`127.0.0.1:17885`). DNS-rebinding guard on `Host` header. CORS restricted to loopback. No auth token yet — non-loopback exposure requires one before any deployment outside localhost.

| Route | What it does |
|---|---|
| `POST /api/kxml/exec` | Execute a compiled KXML tape (magic `KXML` + CBOR payload) |
| `POST /api/kxml/lower` | Validate and lower an admitted `semantic.xcfe-node-graph.v1` into a projection-only JSON program |
| `POST /api/kxml/compile` | Encode a projection JSON program as a KXML+CBOR tape; raw XCFE graphs are rejected |
| `POST /api/kxml/parse` | Decode a KXML+CBOR tape back to its JSON projection program |
| `POST /api/kxml/write` | Write a KXML tape to disk (repo-rooted path) |
| `POST /api/program/{verb}` | Typed program dispatcher with defer-shape preservation |
| `GET /api/program/registry` | Read-only route registry plus the derived Micronaut program catalog |
| `POST /api/program/registry/register` | Atomically register a repository-rooted program/Micronaut manifest after complete existence validation |
| `GET /api/model/status` | Report the linked llama.cpp model and real startup readiness |
| `GET /v1/models` | OpenAI-compatible list containing only the in-process linked model |
| `POST /v1/chat/completions` | Run non-streaming chat inference in the app-host process |
| `GET /.well-known/sco` | Capability advertisement: `mx.run` (not `xcfe.run`) |

---

## In-process GGUF hosting

`app-host.manifest.json` points to `drivers/runtime/models.toml` with
`model_required:true`. Exactly one `[[models]]` row must opt in:

```toml
[registry]
default_model = "qwen3-1.7b-chat"
gguf_link_required = true

[[models]]
id = "qwen3-1.7b-chat"
path = "C:/.../Qwen3-1.7B-Q4_K_M.gguf"
path_env = "QWEN3_CHAT_MODEL_PATH"
gguf_link = true
chat_template = "chatml"
ctx_size = 4096
max_tokens = 512
```

`gguf_link` is separate from `auto_launch`: it links the declared GGUF to the
app host's own llama.cpp core rather than starting or proxying an external
server. The environment variable named by `path_env` may replace the configured
absolute path. Multiple linked rows, a linked non-default row, a missing file,
or invalid llama parameters fail boot when the model is required.

The completion route is deliberately non-streaming and serialized per model
context. Its `x_khanary` block marks output as an unpromoted generated finding
with `semantic_authority:false` and `truth_authority:false`; model loading does
not grant XCFE or fact authority.

---

## MicroXcfe — the projection executor

`MicroXcfe` runs deterministic programs. It is **not** the XCFE authority layer — it has no access to the phase engine, gram store, or topology mutation. Capability string: `mx.run`.

### Op set

| Op | Description |
|---|---|
| `ADD`, `MUL` | Numeric projection |
| `CALL` | Invoke stdlib function (allow-listed) |
| `LOG` | Append to journal |
| `CONVERT` | Format conversion via `SidecarTable` (`@CDATA/@SVG/@XML`) |
| `FOLD_ENTER`, `FOLD_EXIT` | Annotation only — marks a fold boundary in the program trace; no phase authorization |
| `SCO_RESOLVE` | Resolve a name through `SCORegistry` |
| `SYMBOL_RESOLVE` | Resolve one typed KXML symbol to its address descriptor without dereferencing it |
| `PATTERN_BANK` | Match a declared wildcard-pattern bank and retain ordered captures |
| `SET` | Set a literal or copy one state key to another state key |
| `APPEND` | Append one state value to an existing state array |

Programs that need real XCFE control flow (phase gates, `@cause / @direct / @for`) must go through port 8787.

### Trusted lowering boundary

`/api/kxml/compile` is an encoder, not a semantic compiler. XCFE authority is
established before tape encoding through `/api/kxml/lower`:

`LowerToKXML(n) iff Admitted(n) && ProjectionSafe(n) && SupportedByMicroXcfe(n)`.

The three gates fail independently. The 14-class supported-node matrix and
rejection taxonomy are defined in
`data/grammar/xcfe-kxml-lowering.contracts.json`; the executable boundary is
`src/host/kxml_lowering.cpp`. Commit, Ch'en verification, semantic phase
transitions, and persistent memory never lower. `FOLD_ENTER` and `FOLD_EXIT`
remain trace annotations with `@phase_authority:false`.

---

## Research corpus builder

`research_run.h/.cpp` provides a corpus-building sub-command:

```powershell
# Scrape Wikipedia + arXiv topics, append bigrams to data/gram/research.jsonl
json_runtime --scrape --topics "fold theory" "K'UHUL phase"

# Inject a URL directly into sys_ctx (no corpus write)
json_runtime --url https://...
```

Output: `data/gram/research.jsonl` (same format as CODEGEN expert corpora — loadable by `GramStore::load_corpus()`).

---

## Micronaut program catalog

At boot the host loads `drivers/micronauts/registry.json` before deriving program bindings from each registered driver. The registry remains authoritative for Micronaut identity; individual driver files remain authoritative for program bodies. The derived catalog records references only.

The `micronaut.program` route accepts three observation-only operations through `POST /api/program/POST`:

| Operation | Required fields | Result |
|---|---|---|
| `list` | none | Every registered Micronaut and its derived program bindings |
| `inspect` | `@micronaut` | One registered Micronaut, driver status, capabilities, and programs |
| `run` | `@micronaut`, `@program` | Executes only the resolved registered binding through MicroXcfe |

Example:

```json
{
  "@kind": "semantic.json-program.v1",
  "@route": "micronaut.program",
  "@verb": "POST",
  "@operation": "inspect",
  "@authority": "observation_only",
  "@micronaut": "coder"
}
```

Program discovery recognizes embedded `@control`, declared `programs`, `@import`, template/program references, external executors, and named `@execute.operations`. The registered `programs` Micronaut also owns JSON files under the manifest-declared `programs_root`, so the rooted program store is visible without copying each body into the registry.

Before `run`, the catalog audits the control graph against MicroXcfe's implemented op set and its loaded stdlib call names. Compatible bindings carry `projection_executable:true`; unsupported bindings remain inspectable with `unsupported_ops` and `unresolved_calls` and are rejected before dispatch. External executors and declarative operations are always visible but cannot be run by MicroXcfe. Ordinary program requests cannot add a Micronaut, expand the registry, or supply an arbitrary execution path.

The manifest binds `programs_root` to `data/programs`, `micronaut_registry` to the master registry, and `micronaut_drivers_root` to the driver directory. Imports are resolved under that program root.

### Manifest registration

The host control plane accepts a repository-relative registration manifest at
`POST /api/program/registry/register`:

```json
{
  "manifest": "data/host/program-registration.manifest.json"
}
```

Registration paths must fall under a `program_registration_roots` entry from
`app-host.manifest.json`. The registration document names the JSON route
registry, authoritative Micronaut registry, driver root, and program root. The
two execution roots must equal the roots established at boot; an HTTP request
cannot replace them.

The host loads candidate catalogs first. Registration fails if a Micronaut has
no driver manifest, a driver is invalid, the program store is absent, or any
executable program reference cannot be loaded. Only a completely valid catalog
replaces the live one. The replacement is in-memory (`persisted:false`), and
`GET /api/program/registry` exposes the accepted manifest list plus
`catalog_validation`.

### Eliza question and intent programs

`drivers/micronauts/semantic/eliza_distil.json` is a declarative ten-track program bank (`intent`, `question`, `think`, `tools`, `reason`, `needs_plan`, `memory`, `micronaut-grams`, `compare`, `clarification`). Its 32 wildcard programs use `PATTERN_BANK`; captures remain strings until a registered resolver binds them. The compiled tape is `semantic/eliza_distil.kxml` (`KXML` magic + CBOR). Rebuild or check it with:

```powershell
python tools\compile_eliza_distil.py
python tools\compile_eliza_distil.py --check
```

The tape remains observation-only and carries `promote_generated_facts:false`.

## ASX DOM and mini-app routes

Two registered JSON routes expose the UI layer without adding a browser-side execution engine:

| Route | Operations | Contract |
|---|---|---|
| `json.ui` | `compile`, `inspect`, `call` | Compiles AtomicDOM keys, expands declared CSS key grammar, and lowers keyed actions into registered JSON programs. SVG compute widgets resolve through the hash-pinned widget registry and return a `projection_request`; visual edges never mutate semantic topology. |
| `json.app` | `compile`, `inspect`, `call` | Loads a rooted `asx.mini-app.v1`, scopes every component address under its `⟁APP:*` key, and dispatches an event to a declared app-local program. |

Mini-apps must declare `@mount`, `@state`, `@components`, `@programs`, `@events`, `@imports`, `@permissions`, and `@lifecycle`. `semantic.json-program.v1` bodies re-enter the shared dispatcher. `asx.state-program.v1` bodies lower to MicroXcfe and may use `ADD`, `MUL`, `CALL`, `LOG`, `SET`, or `APPEND`. Caller state is merge-patched for that request and discarded afterward.

App-scoped addresses are also published in `data/grammar/asx-mini-app.key-grammar.jsonl`. Those rows are constrained to `executor:"mx.run"`, `authority:"projection_only"`, and `semantic_authority:false`. The source grammar is `E:\models\_khanary_inspect\key-grammar-kit\key-grammar-json.runtime.v4.ebnf`, where `asx_app_key_row` is a first-class row type.

The JSONL rows are bindings. Runtime vocabulary comes from the manifest books declared by `key_grammar_manifests` in `data/host/app-host.manifest.json`: `gml`, `joke`, `kuhul`, `words`, `aiml`, `css`, `eliza`, `folds`, and `tool`. Host boot parses all nine books, requires their bootstrap keys, indexes every declared key, and records a SHA-256 identity for each book. App compilation fails closed if the books are absent or malformed and returns `manifest_books` plus `manifest_key_count` so the exact bootstrap set is auditable. `GET /api/program/registry` exposes the same state as `key_grammar_bootstrap` before a client invokes an app.

`tool.manifest.json` supplies the verb layer over the existing intent vocabulary. Intent answers *why* a tool family is being requested; the verb answers *which registered operation* is proposed. `GET`, `POST`, `READ`, `WRITE`, and `PATCH` therefore expand to an intent, a verb, and a command candidate. The corresponding `verb:*` entry must name a semantic command track, a declared program under `data/programs/`, `authority:host_registry`, and `boundary:InvocationValidationBoundary`. The book can make the runtime tool-aware, but it cannot execute a path or grant authority from text. The portable kit exposes micronauts at `data/micronauts/`; KHANARY projects that same catalog at `data/programs/micronauts/`. Both catalogs are declared roots, never wildcard file-execution roots.

`template:micronaut-control` binds those verbs to typed wildcard patterns. A template's declared C6 phase set is the hard legality gate. After that gate passes, `DirectionStore.query_at_phase` may rank or narrow candidates using phase positions learned only from verified `GOOD` `FoldDelta` records. Fold evidence is observational: it cannot add a phase to the template, authorize an invocation, or replace `InvocationValidationBoundary`. An unresolved capture remains a wildcard with `on_unresolved:"defer"`.

Shipped examples:

- `social-media-animal-farm.asx-app.json`: Rebellion → Revolution → Restoration, with a hidden Pig Index revealed only at the mirror.
- `pig-kingdom.asx-app.json`: civilian protection, evidence, public trust, and independent accountability across faction lines.
- `war-pigs.asx-app.json`: treasury, communications, civilian safety, and accountability, with five callable specialist programs plus the scene dispatcher.

`data/host/examples/asx-story-app.html` runs Pig Kingdom (`?app=kingdom`) and War Pigs (`?app=war`) against `json.app`. `social-media-animal-farm.html` is the dedicated Animal Farm renderer. Both expose `render_game_to_text()` and `advanceTime(ms)` for deterministic browser validation.

## Micronaut drivers

The files in `drivers/micronauts/` are **declarative JSON records**, not executable code. The runtime loads them to configure sampling parameters, route to models or sidecars, and declare capability strings. All drivers extend the base contract defined in `micronaut_core.json`.

### Base contract — `micronaut_core.json`

Every micronaut inherits:

| Field | Value |
|---|---|
| 4-tier memory | working → `SharedMemoryBridge`; episodic → `asx_memory.events.jsonl`; semantic → kuhul_engine:17474; procedural → `micronaut.cso` XCFE bytecode |
| Cognitive physics | `temperature ↔ attention level`; `confidence ↔ enthusiasm` |
| Transport | `@communicate` → kuhul_engine:17474 (not port 8080) |

---

## Driver categories

### 1. Phase fold drivers

One sampling profile per C6 phase. The runtime selects the active phase driver to bias the LLM toward the correct epistemic posture.

| File | Phase | Temperature | Character |
|---|---|---|---|
| `pop.json` | Pop | 0.35 | Observe, long context, seed |
| `wo.json` | Wo | 0.55 | Plan, schedule, bind |
| `yax.json` | Yax | 0.85 | Branch, explore, expand |
| `sek.json` | Sek | 0.20 | Execute, deterministic |
| `chen.json` | Ch'en | 0.45 | Verify, structured proof |
| `xul.json` | Xul | 0.65 | Emit, commit, collapse |

### 2. Specialist LLM micronauts

| Driver | Fold | T | What it does |
|---|---|---|---|
| `eliza.json` | Ch'en | 0.45 | Metacognitive layer — questions → compare → plan; outputs `intent/next/works/wrong/right/questions`; invariant: **never executes** |
| `alice.json` | Yax | 0.75 | Gram engine — bigrams/trigrams, domain classification; sub-µ at :3210 |
| `think.json` | Wo | 0.55 | 9-branch thinker: greeting / howto / explain / compare / command / reflect / debug / hypothesize / summarise |
| `reason.json` | Yax | 0.70 | Multi-step reasoning chain + contradiction check via eliza |
| `expand.json` | Yax | 0.80 | 6 expansion modes: extrapolate / hypothesize / synthesize / elaborate / metaphor / analogy |
| `coder.json` | Sek | 0.15 | Code generation, low temperature |
| `ts_coder.json` | Sek | 0.15 | TypeScript/JS specialist trained on TS error→fix pairs |
| `ast.json` | Sek | 0.00 | Deterministic grammar → KastNode parse; grammars: c / cpp / python / typescript / kuhul / kast / kson / scxq2 |
| `tool.json` | Sek | 0.20 | Tool selection and dispatch; validates against `actions.manifest.json`; routes to kuhul_task_boss :8764 |
| `tool_call.json` | Sek | 0.20 | Tool call envelope formatter |
| `tool_dispatch_proof.json` | Ch'en | 0.00 | Hash-chained dispatch proof record for Ch'en verification |
| `memory.json` | Pop | 0.30 | Long-context memory retrieval |
| `librarian.json` | Yax | 0.60 | Knowledge library: fiction / science / theory / tech / politics |
| `regex.json` | — | 0.00 | Deterministic pattern matching for KXML token streams, schema validation, AIML guards |
| `ui.json` | Xul | 0.50 | UI output driver — short context, concise emission |
| `chat.json` | Wo | 0.60 | Conversational surface |
| `codex.json` | Sek | 0.15 | Codex-style code completion |
| `domain_seed.json` | Pop | 0.35 | Domain seeding for new fold contexts |

### 3. Native / sidecar micronauts

These route to DLLs or sidecar URIs, not LLMs. Temperature is 0.0.

| Driver | Backend | What it does |
|---|---|---|
| `native_factory` | `micronaut_factory_core.dll` | Spawn and evolve micronauts |
| `nnck` | `NeuralGrammar.Core.dll` (.NET 8) | Grammar parsing, pattern classification, CLR-hosted inference via NETFXSDK |
| `quantum` | `sidecar://quantum/dispatch` | Routes to quantum_trinity / quantum_personality / asx_ram_v2 (D3D11 xshard) / asx_gemm |
| `adam.json` | SMCA / kxc pipeline | Adam optimizer: gradient accumulation + parameter update kernel |
| `jrom` | port 3217 | GPU adapter registry, kernel classification, BLAKE2b hash-linked replay log |
| `cline_jshell.sidecar.json` | JShell sidecar | Java/JVM code execution via JShell process |
| `java_micronaut_agent.json` | JVM agent | Full Java agent lifecycle (attach/detach/instrument) |

### 4. Orchestration and meta

| Driver | T | What it does |
|---|---|---|
| `factory.json` | 0.70 | Auto-factory — spawns and evolves new micronauts based on capability gaps |
| `evolution.json` | 1.00 | High-entropy pattern discovery; produces candidate micronaut definitions |
| `programs.json` | 0.40 | JSON program dispatcher: loads control programs, resolves stdlib, `GPU_DISPATCH`, routes to semantic_cube |
| `atomic_micronaut_v10.json` | 0.45 | Event-sourced orchestrator: Fisher-Rao trust decoding, BLAKE2b hash-chained event log, SLERP geodesic step; `CanonicalState = Replay(EventLog)`; OpenCL 1.2 + D3D11 CS5 |
| `khanary.json` | 0.50 | Top-level KHANARY runtime persona |
| `default.json` | 0.55 | Fallback driver when no specialist matches |

### 5. Entropy-collapse pair

These two work as a matched pair — one generates, the other collapses:

| Driver | T | Role |
|---|---|---|
| `sheogorath.json` | 0.97 | Generates candidate states via maximum entropy / divergence — "the Madgod" |
| `jyggalag.json` | 0.00 | Collapses candidates to a single surviving state — "Greymarch = total-order collapse event" |

### 6. Stack guides

High-confidence (T≈0.3) reference micronauts — system-prompt documents for answering questions about specific runtime subsystems. No generation; retrieval only.

`stack_doc.json` · `primeos_guide.json` · `scx_guide.json` · `asx_guide.json` · `distillation_guide.json` · `compiled_model.json` · `codex.json`

### 7. Geometry / routing layer

| Driver | What it does |
|---|---|
| `semantic_cube.json` | 6-face K-CUBE tensor (Phi / Fold / Gram / Geodesic / Projection / Entropy); spherical-harmonic projection selects the dominant µ for routing; used by all fold-µ routing and `tool_dispatch_proof` replay |
| `moe_seed.json` | MoE seeding record for expert weight initialisation |
| `mx2_brains.registry.json` | Registry of brain-layout experts for the MX2LM vector field |

### 8. User / session drivers

| Driver | What it does |
|---|---|
| `user-PRIME.micronaut.json` | Primary user session identity — authority scope, persona, preference bindings |
| `user-template.micronaut.json` | Blank template for provisioning new user session micronauts |
| `mini_residency.micronaut.json` | Short-session residency contract (learning window) |
| `mini_residency.v1.micronaut.json` | V1 residency with explicit evidence-collection fields |
| `adam.json` | Also used as the user-gradient accumulation driver in RLHF loops |

### 9. Domain specialists

| Driver | Fold | What it does |
|---|---|---|
| `dolphin.json` | Yax | Unconstrained exploration (no system-prompt guard) |
| `sheogorath.json` | Xul | Entropy maximiser (see collapse pair above) |
| `jyggalag.json` | Ch'en | Order enforcer (see collapse pair above) |
| `distillation_guide.json` | Pop | Guides model distillation runs |
| `MicronautDatabase.json` | — | Persistent micronaut state database schema |
| `micronaut_core.json` | — | Base contract (see above) |

---

## Master registry

`drivers/micronauts/registry.json` is the authoritative index of all 48+ micronauts. Each entry carries:

```json
{
  "name": "eliza",
  "fold": "Ch'en",
  "category": "metacognitive",
  "temperature": 0.45,
  "capabilities": ["compare", "question", "plan"],
  "system_ref": "drivers/micronauts/eliza.json",
  "sidecar_uri": null,
  "native_dll": null
}
```

The runtime resolves `sidecar_uri` and `native_dll` through `SCORegistry` before dispatch. Fields left null route to the LLM path.

### Daemon Bot AIML pattern bank

`daemon_bot` is a deterministic, observation-only Micronaut backed by 86 reviewed AIML 2.0 categories and a matching JSON knowledge book:

- `data/host/bots/daemon-bot-expanded.aiml` is the recognition/template source.
- `data/host/bots/daemon-bot-knowledge.json` records group, provenance, feed status, and anti-fabrication rules.
- `data/programs/micronauts/daemon-bot.json` is the reviewed JSON projection.
- `data/programs/micronauts/daemon-bot.kxml` is its canonical CBOR execution tape.
- `drivers/micronauts/daemon_bot.json` registers the `chat` program.

The compiler validates AIML/knowledge parity, group counts, and ordered wildcard captures before emitting either derived artifact:

```powershell
python tools\compile_daemon_bot.py
python tools\compile_daemon_bot.py --check
```

Static categories emit `finding`; blog categories emit `gap` and retain their `BLOG_FEED_*` action marker. The tape never fetches a feed, invokes a tool, persists state, or promotes a finding.

Run it through the registered control-plane route:

```json
{
  "@kind": "semantic.json-program.v1",
  "@route": "micronaut.program",
  "@verb": "POST",
  "@operation": "run",
  "@authority": "observation_only",
  "@micronaut": "daemon_bot",
  "@program": "chat",
  "@state": { "input": "COMPARE BLUE DREAM AND SOUR DIESEL" }
}
```

The ASX integration example is `data/host/examples/asx-compression-white.v2.html`. Its symbolic keys call the same route and display the returned phase, output kind, captures, and promotion state.

### Self Hosted Forum app-key integration

The forum UI is represented by `data/host/apps/self-hosted-forum.asx-app.json` and rendered by `data/host/examples/self-hosted-forum.html`. The manifest is the app contract: `⟁APP:SELF_HOSTED_FORUM` scopes its components, state, schema tables, events, and projection programs. Its companion rows are registered in `data/grammar/asx-mini-app.key-grammar.jsonl`.

The page uses `data-asx-key` on the root, toolbar, search input, post table, modals, and actions. Events such as `search.input`, `auth.submit`, and `post.submit` lower through the existing `json.app` route. IndexedDB or another storage adapter may render the data, but storage is request-scoped in the app contract; the browser page cannot grant semantic commit authority.

`data/host/apps/socialsphere.asx-app.json` applies the same contract to the SocialSphere feed. `data/host/examples/socialsphere.html` is deliberately a declarative keyed page; React, ReactDOM, CDN loading, localStorage logic, and page-level event executors are not part of the runtime. Feed, authentication, compose, like, upvote, and search behavior are declared as ASX state programs and dispatched through the JSON app route.

---

## Authority boundary

The JSON runtime enforces a hard boundary — it cannot cross into the XCFE/Ch'en/Xul commit path:

| Capability | JSON runtime | JSON-RPC :8787 | MCP :8764 |
|---|---|---|---|
| Run KXML programs | ✓ | — | — |
| Dispatch micronaut drivers | ✓ | ✓ | ✓ |
| Phase gate enforcement | ✗ | ✓ | ✓ |
| Topology mutation | ✗ | ✗ (proposes only) | ✗ (proposes only) |
| Xul commit | ✗ | ✓ (Ch'en-gated) | ✓ (Ch'en-gated) |
| Write files outside repo root | ✗ | ✗ | ✗ |
| Non-loopback connections | ✗ (no auth token) | loopback | loopback |

`FOLD_ENTER` / `FOLD_EXIT` in MicroXcfe programs are **annotations** — they mark boundaries in the program trace for observability but do not authorise phase transitions.

---

## Regeneration commands

```powershell
# K'UHUL expert corpus (kuhul + khl + klsl + kxml + json)
py -3 tools\scrape.py --domains kuhul --split

# JSON expert scoped to project only (avoids 460 MB drive-wide noise)
py -3 tools\scrape.py --domains json --paths "E:\models\_khanary_inspect\KHANARY.CPP" --split

# Research corpus (Wikipedia / arXiv)
bin\KHANARY.exe --scrape --topics "fold theory" "K'UHUL phase cycle"
```
