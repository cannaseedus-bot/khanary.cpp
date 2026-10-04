# ASX / Micronaut Unification Plan

## Mission
Make the stack explicitly micronaut-first: ASX is the full micronaut system, tapes are early micronaut bootstrap artifacts, JROM is the tape player/provenance journal, and K'UHUL/XJSON are execution and contract forms of the same system.

## Governing Architecture Rule
- **ASX is the map**: canonical semantic/specification layer describing intended system behavior and surfaces.
- **KHANARY is the implementation**: modern native runtime that executes the map in deterministic, enforceable contracts.
- **Interpretation rule**: if legacy narrative and runtime behavior differ, resolve by mapping legacy terms to explicit KHANARY contracts, then implement or downgrade claims.

## Canonical Source Anchor
- User canonical reference artifact: `E:\important\json-asx-language-support-0.1.0.vsix`
- User canonical guide artifact: `E:\important\Asx Scripting Language — Ultimate Guide (based On Our Chats).pdf`
- User canonical installer lineage artifact: `E:\important\ASX Micronaut (Mx2LM) Installer Setup.pdf`
- Planning rule: treat this VSIX as a canonical language/spec input and mirror extracted, versioned contract snapshots into repo-tracked docs/contracts for reproducible audits.
- Planning rule (guide): treat the Ultimate Guide as a canonical **map-layer** narrative/spec source (chat-derived canon), then reconcile each claimed behavior with KHANARY runtime contracts before marking it `active`.
- Planning rule (installer lineage): treat the Mx2LM installer doc as micronaut lineage + active Mx2LM profile input (2-gram/3-gram + persona CLI + ASX-RAM versioning), where Mx2LM-owned n-grams are local authority for that micronaut and cross-stack claims must be explicitly reconciled.
- Packaging target note: this VSIX declares `Microsoft.VisualStudio.Code` support, so install with VS Code/Cursor extension flow (Install from VSIX or `code --install-extension`), not Visual Studio `VSIXInstaller.exe`.
- Known installer signature: `VSIXInstaller.NoApplicableSKUsException` indicates wrong installer/SKU path, not a missing payload file.

## Source Hierarchy (Conflict Resolution)
1. **KHANARY executable contracts/tests** (implementation truth).
2. **ASX map artifacts** (VSIX + Ultimate Guide) as intended behavior.
3. **Legacy/playground docs** as lineage unless promoted by tests/contracts.

If (2) claims behavior not in (1): add mapping task, implement deterministically, or downgrade claim to `legacy`/`experimental`.

## Canonical Model (Authoritative Terms)
- **ASX**: the full system surface (contracts, execution, runtime behavior), not a separate side format.
- **ASX scope intent**: "everything-and-anything OS" target surface (games, apps, websites, tools), with staged delivery rather than claiming all surfaces are production-ready today.
- **Micronauts**: the primary runtime abstraction; all orchestration and capability routing is framed through them.
- **Mx2LM**: an explicit micronaut in that runtime family, with its own n-gram assets/behavior profile (not a detached side tool).
- **Tapes**: legacy bootstrap/game-ROM style artifacts that predate current micronaut structure.
- **JROM**: the runtime tape player + provenance journal for replay/audit; not the semantic authority itself.
- **K'UHUL**: phase/grammar execution surface for micronaut behavior and transitions.
- **XJSON**: contract/data surface for deterministic exchange and validation.
- **KLH**: legacy hive/atlas framing now interpreted through the semantic cube + graphics API sandbox used by micronauts.
- **CSS compression**: part of ASX symbolic compression law, not a disconnected UI concern.

## Legacy Interpretation Policy
- **Active core**: semantics currently enforced in runtime/contracts/tests.
- **Legacy lineage**: prior system ideas kept as historical lineage and mapped to modern equivalents.
- **Experimental/playground**: hypotheses or prototype tracks that are not runtime guarantees.
- **Rule**: every legacy term in docs must be tagged as `active`, `legacy`, or `experimental` and mapped to present implementation status.

## Deep Gap Audit — 2026-10-02

### Current Verification Snapshot
- Local Release suite: `ctest --test-dir build -C Release --output-on-failure` passes **48/48** after the live semantic-attention dispatch wiring.
- GPU semantic attention is now wired: prompt GramStore candidates -> `InferenceEngine::score_prompt_attention()` -> `GramGpuScorer::score_batch()` -> Lorentz `b1` cbuffer. This closes the previous “no live dispatch call site” gap.
- Root conformance summary is stale: `build/conformance/conformance-summary.json` was last written before the current semantic-attention work and reports 31 suites, while the active build tree runs 48 tests.
- Several old audit items are now closed or narrowed:
  - Phase driver mode is `enforce`, not `audit`, and `KHANARY_RUNTIME_SCHEMA_STACK_TEST` passes.
  - `data/drivers/phases/{pop,wo,yax,sek,chen,xul}.json` exist and hash checks pass.
  - Boundary split checks exist and pass, but the check is narrower than the real repository boundary problem.
  - GPU/WGSL lanes now report `pass`, but mostly as shape/probe lanes, not numerical backend parity. Supernaut is an external future addon with its own repository and is intentionally excluded from root KHANARY conformance.
  - Compile-time verdict stamps were removed from the current C++ verdict header; stale summaries may still contain old `build_stamp` fields.

### Gap State Legend
- **REGRESSION RISK**: can invalidate trusted execution or produce false PASS evidence.
- **SECURITY / EXPOSURE**: can expose mutation or model surfaces beyond intended local control.
- **AUTHORITY**: can blur projection/routing bias with semantic truth, verification, or commit.
- **DURABILITY**: can lose or corrupt replay/persistence guarantees.
- **OPERABILITY**: can make local startup/API behavior brittle or misleading.
- **COVERAGE**: implementation exists, but the executed proof surface is incomplete.

### Frozen Semantic Core
The semantic core is frozen as:

```text
K = (G_semantic, Θ, 𝒜, Π, X)
Θ = (P, W, Y, S, C, X)
𝒜 = (α_P, α_W, α_Y, α_S, α_C)
```

Execution law:

```text
k -> N_G(k) -> α_P -> B_P -> α_W -> B_W -> α_Y -> B_Y
  -> α_S -> B_S -> α_C -> B_C -> X_Π -> k'
k'_t -> P_(t+1)
```

Xul is the sixth phase, **not** a sixth fold. Five semantic fold algebras transform state; Xul is the policy-governed boundary/canonicalization operation that closes the cycle. Representations and executors are optional, heterogeneous realizations. The common executor surface is the envelope, not the computation:

```text
E_θ(contract, input, context) -> (result, evidence, status)
```

Invariant: **Executor implements contract; contract does not implement executor.**

---

## Gap Register

| ID | Severity | State | Gap | Evidence |
|---|---:|---|---|---|
| GAP-001 | REGRESSION RISK | Open | Conformance summary is stale and does not prove the current 48-test tree. | `build/conformance/conformance-summary.json` timestamp older than `build/Release/KHANARY.exe`; `tools/run_conformance.py` |
| GAP-002 | COVERAGE | Open | Required-suite inventory is manually maintained and omits newer/native lanes such as semantic attention, GML parser, STB map, SVG tensor, control-flow, metric tensor, Christoffel, and others. | `tools/run_conformance.py` required set vs `CMakeLists.txt` test registrations |
| GAP-003 | REGRESSION RISK | Open | Verdict aggregation validates summary hashes but does not normalize/sort case rows, reject duplicate case IDs, or validate every verdict shape before hashing. | `tools/run_conformance.py`, `src/tests/conformance.h`, `tools/conformance_common.py` |
| GAP-004 | COVERAGE | Open | Boundary split checks ignore rules and `#include` references only; it does not detect tracked/untracked generated artifacts under `data/`, `artifacts/`, root docs, or generated runtime outputs. | `tools/check_boundary_split.py`, `.gitignore`, `.rgignore`, `git status` |
| GAP-010 | AUTHORITY | Reframed | Driver artifacts are semantic µkernel/SMART-DB payloads, not presumed native DLLs. The contract should explicitly version JSON/BSON/bytecode payload kinds and host-driver packaging instead of implying native DLL loading. | User clarification; `data/schema/fold.runtime.json`, `src/kuhul/phase_driver_lock.cpp` |
| GAP-011 | OPERABILITY | Open | Phase artifact root resolution differs between startup preflight and `PhaseDriverLock::load()` for non-repo `--data-dir` or changed CWD. | `src/main.cpp`, `src/kuhul/phase_driver_lock.cpp` |
| GAP-012 | AUTHORITY | Open | Startup validates the runtime stack core, but not the complete internal schema/module link graph before subsystem initialization. | `src/main.cpp::validate_runtime_schema_stack`, `data/schema/*.json` |
| GAP-013 | REGRESSION RISK | Open | Native runtime-stack validation and Python `check_runtime_schema_stack.py` are not fixture-equivalent. | `src/main.cpp`, `tools/check_runtime_schema_stack.py` |
| GAP-014 | OPERABILITY | Open | Native startup can warn on RPC/MCP failure but still print ready. | `src/main.cpp` server startup block |
| GAP-020 | SECURITY / EXPOSURE | Open | Native JSON-RPC binds `INADDR_ANY` and has no server-side authentication/authorization. | `src/io/json_rpc_server.cpp` |
| GAP-021 | SECURITY / EXPOSURE | Open | MCP tool surface lacks a server-side authorization boundary; bot-side `openai.yaml` allowlist is not enforcement. | `src/io/mcp_server.cpp`, `openai.yaml` |
| GAP-022 | OPERABILITY | Open | JSON-RPC returns HTTP 200 for parse/method/internal errors. | `src/io/json_rpc_server.cpp` |
| GAP-023 | SECURITY / EXPOSURE | Open | JSON-RPC HTTP handling performs a single fixed `recv`, without content-length framing, request limits, or fragmented-request handling. | `src/io/json_rpc_server.cpp` |
| GAP-024 | REGRESSION RISK | Open | JSON-RPC exception text is interpolated into JSON without escaping. | `src/io/json_rpc_server.cpp` |
| GAP-025 | SECURITY / EXPOSURE | Open | Completion gateway accepts all CORS origins and ignores `Authorization`. | `tools/khanary_completion_api.py` |
| GAP-026 | OPERABILITY | Open | Completion streaming is buffered synthetic SSE, not incremental streaming, and cancellation does not propagate upstream. | `tools/khanary_completion_api.py` |
| GAP-027 | OPERABILITY | Open | No real `/healthz` or `/readyz`; `/v1/models` only proves the gateway is listening. | `tools/khanary_completion_api.py`, `KHANARY.bat` |
| GAP-028 | OPERABILITY | Open | Launcher can skip/fail completion API startup and still succeed. | `KHANARY.bat` |
| GAP-029 | OPERABILITY | Open | Completion port is not collision-resolved and `ports.json` can disagree with the actual endpoint. | `KHANARY.bat`, `tools/khanary_completion_api.py` |
| GAP-030 | OPERABILITY | Open | If KHANARY and both model contributors fail, gateway can return a successful completion containing `No response available.` | `tools/khanary_completion_api.py` |
| GAP-031 | OBSERVABILITY | Open | Completion IDs are timestamp-second based; usage is always zero; structured latency/error metrics are absent. | `tools/khanary_completion_api.py` |
| GAP-032 | COVERAGE | Open | Gateway extracts only the latest user message; system/assistant context and structured content are not normalized into KHANARY. | `tools/khanary_completion_api.py` |
| GAP-033 | OPERABILITY | Open | Contributor capability state (`disabled`, `unavailable`, `failed`, `used`) is not advertised as a first-class readiness/response contract. | `tools/khanary_completion_api.py`, `KHANARY.bat` |
| GAP-034 | OPERABILITY | Open | `bot.py` still presents separate advisor/chat profiles as primary, competing with the single completion gateway. | `bot.py`, `openai.yaml`, `README.md` |
| GAP-040 | AUTHORITY | Open | Semantic attention dispatches from prompt archive rows without a parser/policy/XCFE eligibility gate. | `src/main.cpp`, `src/inference/semantic_attention.cpp` |
| GAP-041 | AUTHORITY | Open | GPU shader hard gates exist, but host packing forces `policy_ok`, `capable`, and `xcfe_ok` to `1.0f`. | `src/gpu/gram_gpu_scorer.cpp`, `data/shaders/attention.hlsl` |
| GAP-042 | OPERABILITY | Open | GPU failure and “no results” both collapse to `{}` at the `InferenceEngine` caller. | `src/inference/inference_engine.cpp`, `src/gpu/gram_gpu_scorer.cpp` |
| GAP-043 | REGRESSION RISK | Open | GPU readback return values are ignored before scores are consumed. | `src/gpu/gram_gpu_scorer.cpp` |
| GAP-044 | REGRESSION RISK | Open | D3D11 immediate context use is not protected by one cross-subsystem execution lock. | `src/gpu/gram_gpu_scorer.cpp`, `src/world/field_dag.cpp`, `src/inference/inference_engine.cpp` |
| GAP-045 | OPERABILITY | Open | Semantic attention allocates and synchronously reads back six buffers per dispatch; no reusable capacity/fence/generation model exists. | `src/gpu/gram_gpu_scorer.cpp`, `src/gpu/tensor_buffer.cpp` |
| GAP-046 | AUTHORITY | Open | Enabled Lorentz matrices are not checked for `LᵀηL ≈ η`; arbitrary transforms can act like a boost. | `tools/check_gml_validator.py`, `data/grammar/gml.validator.json`, `src/gpu/gram_gpu_scorer.cpp` |
| GAP-047 | REGRESSION RISK | Open | Lorentz state is engine-global and can leak into later prompts unless explicitly cleared/replaced. | `src/inference/inference_engine.cpp`, `src/main.cpp` |
| GAP-048 | DURABILITY | Open | Hyperbolic GPU results are transient; there is no Ch’en -> Xul persistence path for accepted observations. | `src/world/field_dag.cpp`, `src/io/jrom.cpp` |
| GAP-049 | AUTHORITY | Open | Hyperbolic event validation permits some partial event records as warnings instead of dispatch rejection. | `tools/check_gml_validator.py`, `src/world/vector_projection.cpp` |
| GAP-050 | AUTHORITY | Open | Semantic-attention inputs do not share the GML validator’s geometry/authority admission contract. | `src/main.cpp`, `data/grammar/gml.validator.json` |
| GAP-051 | COVERAGE | Open | No CPU reference vs GPU numerical parity suite for attention, geodesic sidecar, and Lorentz contraction. | `src/tests/test_semantic_attention.cpp`, `tools/check_attention_shader.py` |
| GAP-052 | COVERAGE | Open | Metric/Christoffel math tests are not connected to runtime geodesic propagation or admission outputs. | `src/tests/test_metric_tensor.cpp`, `src/tests/test_christoffel.cpp`, `src/world/field_dag.cpp` |
| GAP-060 | AUTHORITY | Open | DECISION-µ can select an action while aggregate epistemic state remains `UNKNOWN`; policy semantics for action-under-uncertainty are not explicit. | `src/decision/decision_mu.cpp` |
| GAP-061 | AUTHORITY | Open | Branch-level `MISS` can be reduced to aggregate `UNKNOWN`, losing retrieval/capability cause. | `src/decision/decision_mu.cpp` |
| GAP-062 | DURABILITY | Open | Factory commits are in-memory until a separate persistence call; commit success is not atomic with durable provenance. | `src/micronaut/factory.cpp` |
| GAP-063 | REGRESSION RISK | Open | Factory registry and vector-frame state are locked/snapshotted separately, permitting stale validation under concurrency. | `src/micronaut/factory.cpp` |
| GAP-064 | DURABILITY | Open | JROM writer interpolates event/payload strings and replay uses substring parsing instead of JSON line validation. | `src/io/jrom.cpp` |
| GAP-065 | DURABILITY | Open | JROM replay lacks schema/version/run-boundary/hash-chain integrity and tamper detection. | `src/io/jrom.cpp`, `src/io/mcp_server.cpp` |
| GAP-066 | AUTHORITY | Open | Scheme URI pattern/template calls can over-admit an operation if any pattern/template exists. | `src/io/request_router.cpp` |
| GAP-067 | OPERABILITY | Open | Scheme aliases are derived from IDs, creating collision/addressability drift unless explicit schemes are required. | `src/io/request_router.cpp`, `data/tracks/*.json` |
| GAP-068 | REGRESSION RISK | Open | URI detection searches for `://` in prose and lacks strict URI/percent-encoding semantics. | `src/io/request_router.cpp` |
| GAP-069 | AUTHORITY | Open | Micronaut profile projection is documented as non-authoritative but not enforced by a typed boundary at every consumer. | `src/micronaut/track_loader.cpp`, `src/micronaut/micronaut_registry.cpp` |
| GAP-070 | REGRESSION RISK | Open | Malformed profile projection data is silently discarded. | `src/micronaut/track_loader.cpp` |
| GAP-071 | COVERAGE | Open | AIML/SCFML/key-grammar alignment is lexical/token-based, not a parsed semantic parity proof. | `tools/check_micronaut_grammar.py`, `tools/grammar/mini_transpilers.py` |
| GAP-072 | AUTHORITY | Open | Semantic envelopes route/decision calls while journaling remains best-effort and not correlation-bound. | `bot.py`, `openai.yaml`, `src/main.cpp` |
| GAP-073 | COVERAGE | Open | VECTOR-µ projection execution returns metadata without applying a documented numerical operation. | `src/micronaut/vector_mu.h` |
| GAP-074 | COVERAGE | Open | Routable track contract coverage is sparse across shipped tracks. | `data/tracks/*.json`, `src/micronaut/track_loader.cpp` |
| GAP-080 | COVERAGE | Open | SCXQ2 remains explicitly not root-conformant. | `docs/CONFORMANCE.md`, `tools/run_conformance.py` |
| GAP-081 | COVERAGE | Open | HLSL compiler/execution lane is not a required root conformance lane. | `CMakeLists.txt`, `tools/run_conformance.py` |
| GAP-082 | COVERAGE | Open | WGSL lane validates emitted shape, not browser/native WGSL compilation or numerical behavior. | `src/tests/test_wgsl_codegen.cpp`, `docs/CONFORMANCE.md` |
| GAP-083 | COVERAGE | Open | DirectML availability/fallback is compile-time configured but not reported as a conformance lane with explicit PASS/SKIP/UNAVAILABLE semantics. | `CMakeLists.txt`, `src/gpu/dml_ops.cpp` |
| GAP-090 | OPERABILITY | Open | Canonical language/docs plan is still broader than implemented runtime truth; active/legacy/experimental tags need continuous enforcement. | `README.md`, `docs/`, old PLAN content |
| GAP-091 | AUTHORITY | Open | The frozen semantic core and Fold Binding Grammar are documented, but executor request/result envelopes are not yet schema-backed or phase-gated in runtime. | `kuhul-grammar-kit/docs/FOLD_ALGEBRA.md`, `kuhul-grammar-kit/docs/FOLD_BINDING.md`, `src/io/request_router.cpp`, `src/micronaut/*` |
| GAP-092 | AUTHORITY | Open | The full runtime coordinate `𝒦_t=(F_h, φ, T, s, σ, X_t)` is documented, but learning/binding state `σ` and Burner observations are not yet schema-backed or separated from fold/phase state in runtime. | `docs/KUHUL_PHASE_ALGEBRA.md`, `kuhul-grammar-kit/docs/FOLD_ALGEBRA.md`, `src/micronaut/*`, `src/io/jrom.cpp` |
| GAP-093 | AUTHORITY | Open | Fold Graph / Active Fold Graph are documented, but there is no schema-backed `𝒢_F(h)` / `𝒢_F^Q(h)` runtime contract proving unfold reveals structure without topology mutation. | `docs/FOLD_GRAPH.md`, `src/main.cpp`, `src/world/field_dag.cpp`, `src/io/request_router.cpp` |

---

## Execution Plan

### Phase 0 — Audit Truth, Boundary, and Evidence Integrity
**Goal:** make the audit surface itself incapable of false confidence.

- [ ] **T0.1 / GAP-001 — Refresh root conformance after every contract-affecting change.**
  - Files: `tools/run_conformance.py`, `docs/CONFORMANCE.md`
  - Work: rerun `python tools/run_conformance.py`; document that `build/conformance/conformance-summary.json` is generated and ephemeral unless explicitly published.
  - Validate: summary suite count equals the CTest verdict-record count for the same tree.

- [ ] **T0.2 / GAP-002 — Derive conformance inventory from CTest.**
  - Files: `tools/run_conformance.py`, `CMakeLists.txt`, `tools/conformance_common.py`
  - Work: enumerate registered tests via `ctest -N`, classify verdict-capable suites, fail on missing verdict JSON, and make the required set data-driven.
  - Validate: add a temporary non-verdict test and prove conformance errors instead of silently omitting it.

- [ ] **T0.3 / GAP-003 — Normalize and harden verdict aggregation.**
  - Files: `tools/run_conformance.py`, `tools/conformance_common.py`, `src/tests/conformance.h`
  - Work: sort case rows by ID before hashing, reject duplicate suite/case IDs, require verdict vocabulary, reject malformed rows, and keep timing/paths/build stamps out of hashes.
  - Validate: fixtures for reordered cases, duplicate cases, duplicate suites, invalid verdicts, and stale build stamps.

- [ ] **T0.4 / GAP-004 — Replace line-presence boundary checks with repository evidence.**
  - Files: `tools/check_boundary_split.py`, `.gitignore`, `.rgignore`
  - Work: use `git ls-files`, pathspec checks, generated-output classification, and source/reference scanning across `src`, `tools`, `data`, docs, and scripts.
  - Validate: fixtures for nested generated paths, backslash paths, tracked generated artifacts, and runtime outputs under `data/world`.

- [ ] **T0.5 — Establish an artifact ownership manifest.**
  - Files: new `data/manifest/artifact-ownership.manifest.json` or equivalent
  - Work: classify source, contract, generated, cache, runtime-journal, model, and fixture paths; include regeneration commands and allowed commit state.
  - Validate: boundary checker fails when a generated artifact is committed without `generated: true` metadata.

### Phase 1 — Runtime Contract Boot Gate and Phase Driver Truth
**Goal:** startup either proves the full contract graph or fails with deterministic diagnostics.

- [ ] **T1.1 / GAP-010 — Formalize semantic µkernel driver payloads.**
  - Files: `data/schema/fold.runtime.json`, `data/schema/kuhul.fold-runtime.json`, `src/kuhul/phase_driver_lock.cpp`, docs
  - Work: replace DLL-only assumptions with a typed payload contract: `payload_kind: json|bson|bytecode`, `semantic_microkernel: true`, `smart_db_role`, optional compiled-driver wrapper metadata for PowerShell/C#, and hash-bound artifact bytes. Keep host driver loading outside phase-authority semantics unless a native DLL is explicitly declared.
  - Validate: JSON, BSON, bytecode, missing artifact, tampered artifact, wrong payload kind, unknown wrapper target, and audit/enforce fixtures.

- [ ] **T1.2 / GAP-011 — Unify phase-root resolution.**
  - Files: `src/main.cpp`, `src/kuhul/phase_driver_lock.cpp`
  - Work: resolve `artifact_root` once relative to the selected data root/repository root, keep `dll_root` as a compatibility alias for existing contracts, and pass the resolved path into the lock loader.
  - Validate: launch from a different CWD with external `--data-dir`; startup and `check_runtime_schema_stack.py` agree for both `artifact_root` and legacy `dll_root`.

- [ ] **T1.3 / GAP-012 — Validate the complete internal schema link graph.**
  - Files: `src/main.cpp`, `src/util/schema.cpp`, `data/schema/*.json`
  - Work: recursively resolve declared internal links/imports, reject out-of-root traversal, validate parseability and required roles before subsystem initialization.
  - Validate: missing link, malformed JSON, path traversal, duplicate role, and unknown schema module fixtures.

- [ ] **T1.4 / GAP-013 — Make native and Python runtime-stack gates fixture-equivalent.**
  - Files: `src/main.cpp`, `tools/check_runtime_schema_stack.py`, `data/conformance/*runtime*`
  - Work: shared fixture matrix for exact phase set, duplicate phases, malformed rows, missing hashes, pending hashes, wrong artifact type, and hash mismatch.
  - Validate: both validators return equivalent PASS/FAIL for every fixture.

- [ ] **T1.5 / GAP-014 — Make service readiness explicit.**
  - Files: `src/main.cpp`, `src/io/json_rpc_server.*`, `src/io/mcp_server.*`
  - Work: required RPC/MCP startup failure is fatal unless an explicit degraded mode flag is set; expose startup status in a typed readiness object.
  - Validate: occupied RPC/MCP ports produce deterministic failure or explicit `degraded:true`, never silent readiness.

### Phase 2 — Network, RPC, MCP, and Single Completion API Hardening
**Goal:** the user-facing API is local-first, explicit, testable, and does not confuse transport success with semantic success.

- [ ] **T2.1 / GAP-020 — Bind JSON-RPC to loopback by default.**
  - Files: `src/io/json_rpc_server.*`, `src/main.cpp`, config flags
  - Work: default to `127.0.0.1`; require an explicit `--rpc-bind` for broader exposure.
  - Validate: remote-interface connection rejected in default mode; explicit bind works when configured.

- [ ] **T2.2 / GAP-021 — Add server-side authorization.**
  - Files: `src/io/json_rpc_server.*`, `src/io/mcp_server.*`, `openai.yaml`
  - Work: local token or loopback-only policy, server-side method/tool allowlists, and separate read/write/mutation scopes.
  - Validate: missing/invalid token, read-only token attempting mutation, and bot-side allowlist bypass attempts.

- [ ] **T2.3 / GAP-022 — Map JSON-RPC failures to correct HTTP statuses.**
  - Files: `src/io/json_rpc_server.cpp`
  - Work: parse/invalid request -> 400, method not found -> 404, internal -> 500; preserve JSON-RPC error body.
  - Validate: integration tests assert both HTTP status and JSON-RPC code.

- [ ] **T2.4 / GAP-023 — Implement bounded HTTP request framing.**
  - Files: `src/io/json_rpc_server.cpp`
  - Work: incremental reads, `Content-Length`, max body size, timeout, malformed request rejection, and no silent truncation.
  - Validate: fragmented, oversized, missing-length, and pipelined request fixtures.

- [ ] **T2.5 / GAP-024 — Serialize RPC error messages safely.**
  - Files: `src/io/json_rpc_server.cpp`
  - Work: use `util::json_string` or equivalent for exception text.
  - Validate: exception strings containing quotes, backslashes, newlines, and Unicode produce valid JSON.

- [ ] **T2.6 / GAP-025 — Add gateway auth and CORS policy.**
  - Files: `tools/khanary_completion_api.py`, config/docs
  - Work: optional bearer token, explicit allowed origins, secure defaults for non-loopback bind.
  - Validate: unauthorized request, wrong origin, preflight, and loopback no-auth development mode.

- [ ] **T2.7 / GAP-026 — Make streaming honest.**
  - Files: `tools/khanary_completion_api.py`, native inference surfaces
  - Work: either document buffered SSE clearly or implement incremental token/event streaming with cancellation propagation.
  - Validate: `stream:true` emits multiple chunks for long output or a capability flag says `buffered_sse:true`; client disconnect stops upstream work where possible.

- [ ] **T2.8 / GAP-027 — Add health and readiness endpoints.**
  - Files: `tools/khanary_completion_api.py`, `KHANARY.bat`
  - Work: `/healthz` for process liveness; `/readyz` checks RPC, required model identity, optional CHAT-µ/TOOL-µ availability, and degraded state.
  - Validate: RPC down, model missing, optional contributors absent, and all-ready cases.

- [ ] **T2.9 / GAP-028 — Make completion API startup required by default.**
  - Files: `KHANARY.bat`
  - Work: fail launch when gateway cannot start unless `KHANARY_ALLOW_NO_COMPLETION_API=1`.
  - Validate: missing Python, occupied port, gateway crash, and explicit opt-out.

- [ ] **T2.10 / GAP-029 — Resolve completion-port collisions consistently.**
  - Files: `KHANARY.bat`, `data/ports.json`, `tools/khanary_completion_api.py`
  - Work: scan for a free port or fail explicitly; update `ports.json`, logs, and downstream launch arguments with the resolved port.
  - Validate: occupy 17892 and verify all consumers follow the resolved port.

- [ ] **T2.11 / GAP-030 — Return typed degraded completion errors.**
  - Files: `tools/khanary_completion_api.py`
  - Work: if KHANARY and contributors fail, return a typed error/degraded response rather than text that looks like an answer.
  - Validate: RPC down, both contributors down, empty model output, and `no_fact_found` cases.

- [ ] **T2.12 / GAP-031 — Add unique request IDs and observability.**
  - Files: `tools/khanary_completion_api.py`
  - Work: UUID completion IDs, optional usage fields when known, structured phase/source/latency/error metrics without prompt-content logging.
  - Validate: concurrent same-second requests have unique IDs; metrics exclude prompt secrets.

- [ ] **T2.13 / GAP-032 — Normalize full chat history.**
  - Files: `tools/khanary_completion_api.py`
  - Work: support system/assistant/user message history and structured content parts or explicitly reject unsupported subsets with typed errors.
  - Validate: multi-turn history, structured content, missing user content, system-only request.

- [ ] **T2.14 / GAP-033 — Advertise contributor capabilities.**
  - Files: `tools/khanary_completion_api.py`, `KHANARY.bat`, `data/ports.json`
  - Work: expose `chat_mu`, `tool_mu`, `khanary_rpc`, `dds`, and `gyro` as `disabled/unavailable/ready/degraded`.
  - Validate: zero, one, and both contributor configurations across Pop/Wo/Yax/Sek.

- [ ] **T2.15 / GAP-034 — Make `bot.py` use the single gateway by default.**
  - Files: `bot.py`, `openai.yaml`, `README.md`
  - Work: default profile targets the gateway; legacy advisor/chat profiles become explicit advanced modes.
  - Validate: `bot.py` prompt-mode integration test reaches `/v1/chat/completions` on the gateway.

### Phase 3 — Semantic Authority, JROM, and Micronaut Contract Closure
**Goal:** projections can bias routing, but only Ch’en verifies and only Xul commits.

- [ ] **T3.1 / GAP-060 — Separate action selection from fact collapse.**
  - Files: `src/decision/decision_mu.cpp`, router tests
  - Work: add explicit `action_selection` policy (`allowed_under_unknown`, `blocked`, `requires_confirmation`) and keep `fact_collapsed:false` for UNKNOWN/MISS facts.
  - Validate: expected-utility, robust maximin/minimax, all-MISS, mixed UNKNOWN/MISS, and no-commit tests.

- [ ] **T3.2 / GAP-061 — Preserve typed aggregate MISS.**
  - Files: `src/decision/decision_mu.cpp`, schema/docs
  - Work: aggregate epistemic state should distinguish `MISS`, `UNKNOWN`, and mixed causes via `epistemic_causes`.
  - Validate: all-MISS and mixed-branch fixtures never degrade to plain UNKNOWN.

- [ ] **T3.3 / GAP-062 — Make Factory commits atomic with persistence.**
  - Files: `src/micronaut/factory.cpp`, JROM/provenance surfaces
  - Work: commit includes before/after hashes, authority, proposal identity, persistence result, and rollback/failure status.
  - Validate: restart after Create/Edit/Patch/Evolve/TrackVector; failed persistence cannot report committed.

- [ ] **T3.4 / GAP-063 — Transaction-lock Factory registry and vector frames.**
  - Files: `src/micronaut/factory.cpp`
  - Work: one transaction/version check for registry mutation plus vector frame commit.
  - Validate: concurrent evolve/vector commits cannot validate from stale snapshots.

- [ ] **T3.5 / GAP-064 — Rewrite JROM write/replay as strict JSONL.**
  - Files: `src/io/jrom.cpp`, `src/tests/test_jrom.cpp`
  - Work: serialize with JSON utilities, validate payload JSON before append, parse lines with the JSON parser, preserve nested payloads exactly, report malformed lines without throwing.
  - Validate: escaped strings, nested objects, truncated lines, invalid sequence, legacy rows, and corrupted-line replay.

- [ ] **T3.6 / GAP-065 — Add JROM integrity and replay modes.**
  - Files: `src/io/jrom.*`, `src/io/mcp_server.cpp`, schema/docs
  - Work: event schema version, run ID, optional payload hash/prev hash, inspect vs reconstruct modes, and non-authority assertion.
  - Validate: deterministic replay, tamper detection, sequence gaps, and replay-cannot-mutate tests.

- [ ] **T3.7 / GAP-066 — Match URI operations against declared pattern/template contracts.**
  - Files: `src/io/request_router.cpp`, `src/micronaut/micronaut_registry.*`, track schema
  - Work: normalize operation names and reject undeclared `scheme://pattern/<op>` or `scheme://template/<op>`.
  - Validate: positive and negative operation-specific scheme tests.

- [ ] **T3.8 / GAP-067 — Require explicit schemes for externally addressable micronauts.**
  - Files: `src/io/request_router.cpp`, `data/tracks/*.json`, contract fixtures
  - Work: derived aliases become compatibility-only with collision diagnostics; primary tracks declare schemes/verbs explicitly.
  - Validate: duplicate alias, undeclared scheme, and collision fixtures.

- [ ] **T3.9 / GAP-068 — Implement strict URI parsing.**
  - Files: `src/io/request_router.cpp`, router tests
  - Work: leading URI grammar, authority/path/query split, percent-decoding policy, malformed URI rejection, and prose URL non-routing behavior.
  - Validate: natural-language URL mention, malformed URI, encoded component, and compound URI cases.

- [ ] **T3.10 / GAP-069 — Enforce profile projections by type.**
  - Files: `src/micronaut/track_loader.cpp`, `src/micronaut/micronaut_registry.*`, consumers
  - Work: wrap profile/personality/progression in a `ProjectionOnly<T>` or equivalent compile-time/runtime boundary.
  - Validate: profile cannot mutate RelationISA, semantic authority, Factory commit, or Xul persistence.

- [ ] **T3.11 / GAP-070 — Surface malformed profile errors.**
  - Files: `src/micronaut/track_loader.cpp`, schema/fixtures
  - Work: schema or loader returns structured diagnostics; no silent empty projection for malformed data.
  - Validate: malformed profile fixture fails with deterministic error code.

- [ ] **T3.12 / GAP-071 — Build semantic AIML/SCFML/key-grammar parity checks.**
  - Files: `tools/check_micronaut_grammar.py`, `tools/grammar/mini_transpilers.py`, manifests
  - Work: parse all three surfaces and compare IDs, operations, phases, authorities, profile boundaries, and generated artifact freshness.
  - Validate: missing pattern, mismatched template, orphan SCFML row, stale generated artifact.

- [ ] **T3.13 / GAP-072 — Give semantic envelopes a schema and correlation contract.**
  - Files: `bot.py`, `openai.yaml`, `src/main.cpp`, JROM schema
  - Work: route/decision envelopes require correlation ID, epistemic state, authority, phase, and result schema; journal failure appears in response metadata.
  - Validate: malformed envelope, contradictory metadata, journal failure, and replay correlation tests.

- [ ] **T3.14 / GAP-073 — Define VECTOR-µ execution semantics.**
  - Files: `src/micronaut/vector_mu.h`, tests/docs
  - Work: either implement the documented projection operation or return explicit unsupported status with required evidence.
  - Validate: valid projection, invalid observation, partial observation, bounded drift, persistence, and non-commit tests.

- [ ] **T3.15 / GAP-074 — Complete routable track contract coverage.**
  - Files: `data/tracks/*.json`, `tools/check_micronaut_grammar.py`, contract fixtures
  - Work: every externally addressable track declares schemes, verbs, key patterns/templates/grammar, capabilities, and authority boundary.
  - Validate: repository-wide track contract validator fails on missing/ambiguous route surfaces.

- [ ] **T3.16 / GAP-091 — Schema-back the Fold Binding and executor envelope.**
  - Files: `data/schema/fold-binding.schema.json`, `data/schema/executor-contract.schema.json`, `src/io/request_router.*`, `src/micronaut/*`, `src/decision/*`, conformance fixtures
  - Work: implement the documented Fold Binding record (`@fold`, `@phase`, `@semantic`, `@backend`, `@entry`, `@dispatch`, `@contract`, `@resources`, `@fallback`) and the `EXECUTOR_REQUEST` / `EXECUTOR_RESULT` envelope. Let tracks/micronauts declare `realizes` capabilities such as `Yax.candidate`, `Sek.COMPARE`, or `Ch'en.observation` without making any executor the semantic law.
  - Validate: registry resolves `(fold, phase, backend) -> binding`; invalid bindings fail V1–V7; missing capability returns typed `MISS_capability`; fallback bindings stay contract-bound; deterministic and model-backed executors satisfy the same phase contract; no executor can bypass Ch'en/Xul.

- [ ] **T3.17 / GAP-092 — Schema-back full fold-phase-learning coordinates.**
  - Files: `data/schema/fold-runtime-coordinate.schema.json`, `src/io/jrom.*`, `src/micronaut/*`, `src/inference/*`, conformance fixtures
  - Work: persist runtime observations as `(F_h, φ, T, s, σ, X_t)` with explicit separators between fold identity, phase, track, folded/unfolded state, learning/binding state, and mutable execution state. Burner observations use `τ=(F_h, φ, T, s, Q, input, operation, output, evidence)` and remain proposal evidence until Ch'en/Xul.
  - Validate: folding at non-Xul phases does not imply commit; Xul-only topology mutation still holds; learning state cannot alter `F_h`; replay can distinguish fold state, phase state, and learned binding state.

- [ ] **T3.18 / GAP-093 — Schema-back Fold Graph and Active Fold Graph contracts.**
  - Files: `data/schema/fold-graph.schema.json`, `data/grammar/fold-graph.contracts.json`, `data/conformance/fold_graph.json`, `tools/check_fold_graph.py`, runtime router/FieldDag surfaces
  - Work: define `𝒢_F(h,τ,Q)` and `𝒢_F^Q(h)` records with source fold/book, projection `τ`, query gate, active mask, node/edge/incidence payload, and `semantic_authority:false`. Prove Yax derives Active Fold Graph without mutating persistent topology.
  - Validate: unfold reveals only existing `B_τ` structure; query gating is masked/ephemeral; Sek consumes Active Fold Graph only; Fold contracts working graph without topology delta; Xul-only commit rule remains enforced.

### Phase 4 — Semantic Attention, GPU Geometry, and Lorentz Execution Hardening
**Goal:** GPU geometry becomes a verified execution surface, not an advisory black box.

- [ ] **T4.1 / GAP-040 — Gate semantic attention on parser/policy eligibility.**
  - Files: `src/main.cpp`, `src/inference/semantic_attention.*`, router/policy surfaces
  - Work: require a validated prompt semantic envelope, candidate policy status, capability status, and XCFE authorization before dispatch.
  - Validate: malformed, empty, ambiguous, unauthorized, and incapable prompts do not dispatch and return typed fallback reasons.

- [ ] **T4.2 / GAP-041 — Populate GPU legality gates from authoritative inputs.**
  - Files: `src/gpu/gram_gpu_scorer.cpp`, `src/inference/semantic_attention.*`
  - Work: extend candidate metadata with `policy_ok`, `capable`, and `xcfe_ok`; remove unconditional `1.0f` gates.
  - Validate: CPU/GPU parity cases with each gate independently false produce zero score and `admitted:false`.

- [ ] **T4.3 / GAP-042 — Return typed semantic-attention results.**
  - Files: `src/inference/inference_engine.*`, `src/gpu/gram_gpu_scorer.*`, `src/main.cpp`
  - Work: replace `{}` failure ambiguity with `{status,error,candidate_count,results}` covering unavailable, compile failed, upload failed, dispatch failed, and readback failed.
  - Validate: each failure mode produces deterministic status and JROM diagnostic.

- [ ] **T4.4 / GAP-043 — Check every GPU readback.**
  - Files: `src/gpu/gram_gpu_scorer.cpp`, `src/gpu/tensor_buffer.*`
  - Work: propagate readback failure, clear partial outputs, and fail the batch before ranking.
  - Validate: forced staging/readback failure test cannot emit zero-shaped success.

- [ ] **T4.5 / GAP-044 — Introduce a shared D3D11 execution lock.**
  - Files: `src/gpu/d3d11_device.*`, `src/gpu/gram_gpu_scorer.cpp`, `src/world/field_dag.cpp`, other GPU users
  - Work: centralize immediate-context locking or move work to owned deferred contexts with explicit synchronization.
  - Validate: concurrent attention + field upload + geodesic dispatch stress test with debug device state validation.

- [ ] **T4.6 / GAP-045 — Reuse buffers and add completion/generation synchronization.**
  - Files: `src/gpu/gram_gpu_scorer.*`, `src/gpu/tensor_buffer.*`
  - Work: reusable per-scorer buffers with capacity growth, fence/event or equivalent completion tracking, and generation IDs.
  - Validate: repeated batches, capacity growth, concurrent dispatch rejection/serialization, and no stale-buffer reads.

- [ ] **T4.7 / GAP-046 — Validate Lorentz metric preservation.**
  - Files: `tools/check_gml_validator.py`, `data/grammar/gml.validator.json`, `src/gpu/gram_gpu_scorer.*`, tests
  - Work: for enabled transitions, verify `LᵀηL ≈ η` within explicit epsilon and preserve interval class for known four-vectors.
  - Validate: valid x/y/z boosts, identity, non-preserving matrix, NaN/Inf, and negative interval fixtures.

- [ ] **T4.8 / GAP-047 — Scope Lorentz transitions to prompt/world generations.**
  - Files: `src/inference/inference_engine.*`, `src/main.cpp`, JROM metadata
  - Work: attach source, world/prompt generation, sequence, and expiry; clear or reject stale transitions.
  - Validate: sequential prompts prove no previous boost affects a new prompt unless explicitly reapplied.

- [ ] **T4.9 / GAP-048 — Persist accepted hyperbolic observations through authority phases.**
  - Files: `src/world/field_dag.cpp`, `src/io/jrom.cpp`, Factory/Xul surfaces
  - Work: GPU result -> observation proposal -> Ch’en verification -> Xul commit or rejection; replayable JROM provenance.
  - Validate: restart/replay, rejected observation, no projection-only commit, and hash-linked provenance.

- [ ] **T4.10 / GAP-049 — Require complete dispatchable hyperbolic events.**
  - Files: `tools/check_gml_validator.py`, `data/grammar/gml.validator.json`, `src/world/vector_projection.cpp`
  - Work: dispatchable events require finite x0..x3, direction, rapidity, interval, interval class, and epistemic state.
  - Validate: missing interval, partial coordinates, zero direction, NaN/Inf, and mismatched interval class reject.

- [ ] **T4.11 / GAP-050 — Align attention input envelopes with GML validation.**
  - Files: `src/main.cpp`, `src/inference/semantic_attention.*`, `data/grammar/gml.validator.json`
  - Work: either route candidate envelopes through equivalent geometry/authority checks or define a separate validated attention schema.
  - Validate: invalid geometry/authority metadata produces no dispatch and typed rejection.

- [ ] **T4.12 / GAP-051 — Add CPU reference vs GPU numerical parity.**
  - Files: `src/tests/test_semantic_attention.cpp`, new reference implementation
  - Work: implement the `S=W*C*R`, relation/fold/phase, tensor, geodesic, legality, and Lorentz laws on CPU and compare GPU outputs within tolerance.
  - Validate: tensor mix on/off, geodesic sidecar on/off, policy gates, Lorentz boosts, and unavailable-hardware explicit skip/unavailable reporting.

- [ ] **T4.13 / GAP-052 — Connect metric/Christoffel math to runtime admission.**
  - Files: `src/tests/test_metric_tensor.cpp`, `src/tests/test_christoffel.cpp`, `src/world/field_dag.cpp`, attention tests
  - Work: non-flat metric -> connection -> geodesic sidecar -> attention modifier end-to-end fixture.
  - Validate: torsion-free and non-torsion inputs, Euler mesh stress, entropy diffusion thresholds, and admission output changes.

### Phase 5 — Backend, Model, and Compression Coverage
**Goal:** conformance claims match executed backend behavior.

- [ ] **T5.1 / GAP-080 — Promote SCXQ2 with fixtures and parity.**
  - Files: SCXQ2 loader/runner, `docs/CONFORMANCE.md`, conformance fixtures
  - Work: bytecode fixture validation plus SCXQ2 output parity against SDDS/reference vectors.
  - Validate: root `KHANARY_SCXQ2_TEST` and summary scope changes from `not_root_conformant` to executed PASS/FAIL.

- [ ] **T5.2 / GAP-081 — Add required HLSL compile/execution lane.**
  - Files: `CMakeLists.txt`, `tools/run_conformance.py`, shader tests
  - Work: compile all shipped HLSL shaders and run deterministic dispatch fixtures where hardware exists.
  - Validate: compile failure, invalid signature, unavailable hardware, and numerical fixture cases.

- [ ] **T5.3 / GAP-082 — Extend WGSL beyond shape checks.**
  - Files: `src/klsl/wgsl_codegen.cpp`, tests, optional Dawn/wgpu tooling
  - Work: compile WGSL where tooling exists and compare simple kernel outputs to CPU reference.
  - Validate: explicit `PASS`, `FAIL`, or `UNAVAILABLE`; never implicit PASS from text shape alone.

- [ ] **T5.4 / GAP-083 — Add DirectML availability/fallback conformance lane.**
  - Files: `src/gpu/dml_ops.cpp`, `CMakeLists.txt`, conformance runner
  - Work: report DirectML compiled/enabled/available/used/fallback separately.
  - Validate: with and without DirectML SDK, forced fallback, and GEMM reference output.

- [ ] **T5.5 — Add model artifact readiness contract.**
  - Files: `KHANARY.bat`, `src/main.cpp`, `tools/khanary_completion_api.py`, docs
  - Work: declare required/optional DDS, GGUF, CHAT-µ, TOOL-µ, gyro, and projection assets with readiness states.
  - Validate: clean machine with no model, partial model, full model, and identity mismatch cases.

### Phase 6 — Documentation, Canonical Sources, and Operator Runbooks
**Goal:** docs no longer claim more than tests prove.

- [ ] **T6.1 / GAP-090 — Maintain a live implementation-status matrix.**
  - Files: `README.md`, `docs/KHANARY-CPP-SPEC.md`, `docs/CONFORMANCE.md`, `PLAN.md`
  - Work: every major claim gets `active`, `partial`, `experimental`, or `legacy`, plus proof command.
  - Validate: docs CI grep/table checker fails on untagged runtime claims.

- [ ] **T6.2 — Preserve canonical source ingestion plan.**
  - Files: docs/contracts snapshots
  - Work: retain VSIX, Ultimate Guide, and Mx2LM installer extraction inventories; map claims to runtime proof.
  - Validate: source snapshot hashes, extraction notes, and claim-status matrix.

- [ ] **T6.3 — Write the single-completion operator runbook.**
  - Files: `README.md`, `docs/DDS-MODEL.md`, `KHANARY.bat` comments
  - Work: document startup order, ports, readiness, auth/CORS, model identity, degraded states, and troubleshooting.
  - Validate: a clean operator can start, verify `/readyz`, prompt through the gateway, and diagnose every failed dependency from documented outputs.

- [ ] **T6.4 — Add security boundaries to user docs.**
  - Files: `README.md`, `docs/KHANARY-CPP-SPEC.md`, `openai.yaml` comments
  - Work: state loopback default, token/auth model, mutation scopes, browser CORS, and non-authority of model contributions.
  - Validate: docs examples never recommend unauthenticated non-loopback deployment.

### Phase 7 — Final Acceptance Gate
**Goal:** close the plan only when runtime proof, live API behavior, and documentation agree.

- [ ] Fresh `python tools/run_conformance.py` passes and includes every verdict-capable registered lane.
- [ ] Full Release CTest passes with no `Not Run` tests unless explicitly classified as `UNAVAILABLE`.
- [ ] Clean startup passes with local data; occupied RPC/MCP/completion ports produce deterministic readiness failures or explicit degraded mode.
- [ ] Single completion API returns a KHANARY answer with correct contributor metadata for Pop/Wo/Yax/Sek scenarios.
- [ ] Unauthorized local/remote mutation attempts fail closed.
- [ ] Semantic attention dispatches only after parser/policy/XCFE eligibility, and GPU failure returns typed fallback diagnostics.
- [ ] Lorentz-enabled attention preserves interval behavior in CPU/GPU parity fixtures.
- [ ] JROM survives malformed-line, nested-payload, restart, replay, and tamper-detection suites.
- [ ] DECISION-µ never reports fact collapse for UNKNOWN/MISS branch facts.
- [ ] Factory commits are durable or explicitly typed as in-memory proposals.
- [ ] Docs/status matrix, README, CONFORMANCE, and PLAN.md agree on what is active, partial, experimental, and legacy.

## Immediate Next Seven Tasks
1. **GAP-041 + GAP-040:** populate semantic-attention legality gates and gate dispatch on parser/policy eligibility.
2. **GAP-042 + GAP-043:** typed GPU result/status plus checked readbacks.
3. **GAP-020 through GAP-024:** loopback/auth, HTTP statuses, request framing, safe RPC errors.
4. **GAP-064 + GAP-065:** strict JROM JSONL writer/replay and integrity checks.
5. **GAP-001 through GAP-003:** refresh and harden conformance aggregation.
6. **GAP-060 through GAP-063:** decision epistemics and Factory transaction durability.
7. **GAP-027 through GAP-030:** readiness and degraded single-completion behavior.

## Working Rule
A gap is closed only when its contract, implementation, negative fixture, and executed validation command all agree. Proposal-only geometry, profile metadata, and model contributions must remain incapable of semantic verification or Xul commit.
