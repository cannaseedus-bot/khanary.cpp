# KHANARY stack audit

Audit of the current working tree, including existing modifications and untracked runtime assets. Source/runtime behavior was not changed. Native build succeeded with a normalized child environment and isolated executable output; all four rebuilt CTest tests passed. Local JSON Schema validation checked 41 shipped track/manifest documents and found one failure. Native GPU inference, live RPC/MCP, external services and nested Supernaut builds were not executed.

## Stack Map

- Native host: `src/main.cpp`, `CMakeLists.txt`; MSVC C++20/D3D11 executable. Validate: `cmake --build build --config Release` (normal output locked during audit; isolated output succeeded).
- Manifest/SCX expansion: `src/manifest/manifest_registry.cpp`, `src/manifest/scx_resolver.cpp`, `data/manifest/`; validate: `python artifacts/stack-intel/probe.py` (one specimen fails).
- Phase/fold authority: `src/kuhul/phase_engine.cpp`, `src/kuhul/phase_driver_lock.cpp`, `data/schema/fold.runtime.json`; inspect: `rg -n "allow_phase|strict_|verified" src/kuhul/phase_driver_lock.cpp`. DLL hash lock defaults to audit mode, not strict enforcement.
- Track/AIML semantics: `src/micronaut/track_loader.cpp`, `src/micronaut/aiml_engine.cpp`, `data/tracks/`, `data/alice/`; validate: `ctest --test-dir build -C Release -R KHANARY_AIML_TEST --output-on-failure` (pass; does not validate every track).
- Routing/attention: `src/io/request_router.cpp`, `src/io/runtime_attention.cpp`, `src/memory/`; validate: `ctest --test-dir build -C Release -R "KHANARY_(ROUTER|ATTENTION)_TEST" --output-on-failure` (pass).
- Projection/replay: `src/io/projection.cpp`, `src/io/jrom.cpp`, `src/world/gyroscope.cpp`, `data/projection_abi_v1.json`; validate: `ctest --test-dir build -C Release -R KHANARY_PTEST --output-on-failure` (pass).
- GPU/inference: `src/gpu/dds_loader.cpp`, `src/inference/inference_engine.cpp`, `data/shaders/`; inspect: `rg -n "forward_stub|dds_loaded_|validate_crc" src/inference/inference_engine.cpp src/gpu/dds_loader.cpp`. DDS forward path exists; no-DDS forward explicitly uses a CPU placeholder.
- KLSL lowering: `src/klsl/klsl_parser.cpp`, `src/klsl/hlsl_codegen.cpp`, `src/klsl/wgsl_codegen.cpp`; build validation: `cmake --build build --config Release --target KHANARY`. Compilation is not backend numerical parity.
- CLI/RPC/MCP and Python control: `src/io/json_rpc_server.cpp`, `src/io/mcp_server.cpp`, `bot.py`; safe CLI check: `artifacts/stack-intel/bin/KHANARY.exe --help` (pass); allowlist inspection: `rg -n "allowlist" bot.py`.
- WPF projection/packaging: `KHANARY.bat`, `tools/micronaut/build_mx2lm_wpf_bundle.ps1`, `.ui-shell/`, `artifacts/micronaut-bundles/`; inspect: `Get-Content tools/micronaut/build_mx2lm_wpf_bundle.ps1`. Generator is the maintenance surface; bundle and UI shell are derived copies. Not launched during audit.
- Parallel native/Canvas runtime: `tools/supernaut/README.md`, `tools/supernaut/supernaut-cpp/CMakeLists.txt`, `tools/supernaut/external-build/`; inspect: `Get-Content tools/supernaut/supernaut-cpp/CMakeLists.txt`. Independent build tree and nested Git checkout, outside root CTest coverage.
- Versioned geometry model: `models/khanary-geometry-v0.3.0/MODEL.json`, `kernels/`, `knu/`; reproduction check: `Test-Path tools/build_geometry_model.py` (False); coverage claim is Partial for this checkout.

## Folder Intel

Sorted physical folders at path depth 1 and 2, including hidden folders. Classification is the primary role, not an enforcement guarantee. Evidence: `Get-ChildItem -Force -Directory -Recurse -Depth 1 | Sort-Object FullName`. Audit-generated `artifacts/stack-intel` is excluded from the source topology count. Git internals and build folders are inventoried but not treated as canonical code. The deeper PowerShell `-Depth 2` inventory was also inspected.

| Path | Role |
|---|---|
| `.claude` | contracts |
| `.codex` | contracts |
| `.codex/skills` | contracts |
| `.git` | artifacts |
| `.git/hooks` | artifacts |
| `.git/info` | artifacts |
| `.git/logs` | artifacts |
| `.git/objects` | artifacts |
| `.git/refs` | artifacts |
| `.ui-shell` | projection |
| `.ui-shell/assets` | projection |
| `.ui-shell/scripts` | projection |
| `__pycache__` | artifacts |
| `artifacts` | artifacts |
| `artifacts/micronaut-bundles` | artifacts |
| `artifacts/smoke-data` | artifacts |
| `build` | artifacts |
| `build/ALL_BUILD.dir` | artifacts |
| `build/CMakeFiles` | artifacts |
| `build/Debug` | artifacts |
| `build/INSTALL.dir` | artifacts |
| `build/KHANARY.dir` | artifacts |
| `build/KHANARY_AIML_TEST.dir` | artifacts |
| `build/KHANARY_ATTENTION_TEST.dir` | artifacts |
| `build/KHANARY_PTEST.dir` | artifacts |
| `build/KHANARY_ROUTER_TEST.dir` | artifacts |
| `build/RUN_TESTS.dir` | artifacts |
| `build/Release` | artifacts |
| `build/Testing` | artifacts |
| `build/ZERO_CHECK.dir` | artifacts |
| `build/data` | artifacts |
| `build/x64` | artifacts |
| `data` | contracts |
| `data/alice` | examples |
| `data/drivers` | contracts |
| `data/gram` | contracts |
| `data/grammar` | contracts |
| `data/journal` | artifacts |
| `data/manifest` | contracts |
| `data/mx2lm` | contracts |
| `data/schema` | contracts |
| `data/shaders` | execution |
| `data/tracks` | examples |
| `data/world` | artifacts |
| `docs` | contracts |
| `models` | contracts |
| `models/khanary-geometry-v0.3.0` | contracts |
| `src` | execution |
| `src/gpu` | execution |
| `src/inference` | execution |
| `src/io` | execution |
| `src/klsl` | execution |
| `src/kuhul` | execution |
| `src/manifest` | execution |
| `src/memory` | execution |
| `src/micronaut` | execution |
| `src/tests` | execution |
| `src/util` | execution |
| `src/world` | execution |
| `tools` | execution |
| `tools/__pycache__` | artifacts |
| `tools/micronaut` | execution |
| `tools/supernaut` | execution |

Key deeper boundaries: `.codex/skills/khanary-micronaut-runtime` (contracts), `data/drivers/phases` (execution artifacts), `models/khanary-geometry-v0.3.0/{data,kernels,knu}` (contracts/execution), and `tools/supernaut/{.git,bin,external-build,supernaut-cpp}` (artifacts/execution). Evidence: the same inventory with `-Depth 2`. No root package.json, codex/, schemas/, native/, gpu/, fpga/ or .asx specimens were located; this is a different concrete layout from the skill's generic AS-XCFE template.

## Contracts & Specimens

| Authority candidate / contract | Specimens / consumer | Coverage and status |
|---|---|---|
| `data/schema/micronaut-track.schema.json` + `micronaut-track-types.schema.json` | `data/tracks/*.semantic-tracks.v1.json`; `src/micronaut/track_loader.cpp` | Local Draft 2020-12 probe passes current tracks, but false-no-eval counterexample exposes schema/loader drift. |
| `data/schema/manifest-map.schema.json` | `data/manifest/*.manifest.json`; `src/manifest/manifest_registry.cpp` | Probe finds `words.manifest.json` help[2] invalid; runtime accepts/stringifies it. |
| `data/schema/{cycle-hash.spec,fold.runtime,kuhul.fold-runtime}.json`, `kuhul.xsd` | phase engine, phase driver lock, runtime metadata | Specifications/configuration, not interchangeable JSON Schemas. Driver bytes are SHA-256 verified in enforce mode; default configuration is audit. Full cycle-certificate conformance not located. |
| `data/projection_abi_v1.json`, `data/gyro_abi_v1.json` | `src/io/projection.cpp`, `src/tests/test_projection.cpp` | Two deterministic projection/gyro acceptance cases pass; payloads are preserved verbatim. |
| `data/schema/{math-isa,resolver.primitives}.json` | AIML lowering and `src/io/request_router.cpp` | Two router policy scenarios pass; schema-wide and authorization-provenance vectors missing. |
| `data/grammar/*.json`, `cpp.code-grammar.ebnf` | `src/io/code_parser.cpp`, KLSL parser | Grammar surfaces present; no .asx/MX2LEX family located, so their absence is not itself a defect. |
| `src/gpu/dds_loader.h` SDDS v1 wire description; `docs/DDS-MODEL.md` | `.scxqdds` loader; separate `.dds` inference path | Native decode exists; full-container rejection and cross-layer vectors are Partial. SCXQ2 mention does not establish format equivalence. |
| `models/khanary-geometry-v0.3.0/data/*schema.json`, `MODEL.json` | birdsong graph and HLSL/WGSL assets | Model-local schema/spec assets; no local reproduction harness at documented paths. Not included in the 41-file schema execution count. |

Evidence commands: `Get-ChildItem data/schema -Name`; `python artifacts/stack-intel/probe.py`; `Get-Content CMakeLists.txt`; `Get-Content src/gpu/dds_loader.h`; `Get-Content models/khanary-geometry-v0.3.0/README.md`. Canonical candidates above are the files runtime/docs point to; the audit does not invent a precedence decision where schema and loader disagree.

## Gaps

- **TRACK_CONTRACT_DRIFT** — Schema and loader accept different track languages. Verified schema counterexample: `@no-eval: false` is accepted, whereas `src/micronaut/track_loader.cpp` rejects it. Conversely, schema-required id/phase/governs are not all required by the loader; missing id is derived and unknown phase defaults to Pop. Evidence: `python artifacts/stack-intel/probe.py`; `Get-Content src/micronaut/track_loader.cpp`.

- **MANIFEST_SPECIMEN_DRIFT** — Shipped manifest fails its declared string-array contract. Verified failure at `help[2]`: an object containing @if/@then/@else violates `data/schema/manifest-map.schema.json`. `src/manifest/manifest_registry.cpp` stringifies non-string array entries instead of rejecting them. Evidence: `python artifacts/stack-intel/probe.py`; `Get-Content src/manifest/manifest_registry.cpp`.

- **JSON_PARSE_BOUNDARY** — Invalid trailing input is outside parser enforcement. Source-confirmed: `parse(text,len)` returns `parse_value()` without checking end-of-input; unknown string escapes are accepted by the default switch arm. Nested arrays/objects exist, but strict JSON acceptance is Partial. Evidence: `Get-Content src/util/json.cpp`; no dedicated parser CTest in `CMakeLists.txt`. This was not dynamically fuzzed.

- **CANONICAL_HASH_GAP** — Declared canonical cycle identity lacks a located native implementation. The spec requires json-c14n-v1 and four hash-bound identity fields. `rg -n "json-c14n|canonical_json|canonicalize|cycle_hash|candidate_delta_hash" src` found no implementation of that cycle contract. `src/util/json.cpp` serializes unordered-map iteration and uses %g numeric formatting. `src/io/projection.cpp` hashes verbatim payload strings: valid byte-level identity, not semantic JSON canonicalization.

- **SDDS_PARTIAL_ACCEPT** — Truncated containers can yield partial model residency. Source-confirmed: load() breaks and returns accumulated shards on short chunk/header reads. validate_crc() does not check version, unlike load(). `src/main.cpp` installs individually CRC-valid returned shards. Evidence: `Get-Content src/gpu/dds_loader.cpp`; `rg -n "crc_ok|validate_crc|DDSLoader" src`. Bad CRC is filtered by the caller; the gap is container completeness and consistent format validation, not unchecked residency of every corrupt shard.

- **CONFORMANCE_COVERAGE** — Four passing tests do not verify loader and backend contracts. CTest registers projection, AIML graph, attention and routing only. No registered JSON/track/manifest rejection, SDDS, KLSL or GPU parity suite. `VECTORS.md` describes attention rather than a golden-vector runner. Evidence: `Get-Content CMakeLists.txt`; `ctest --test-dir build -C Release --output-on-failure` (4/4 pass).

- **MODEL_REPRO_DRIFT** — Published geometry verification cannot be reproduced with documented local commands. README prescribes `python tools/build_geometry_model.py` and `python -m pytest tests/ -q`; `Test-Path tools/build_geometry_model.py` and `Test-Path tests` both return False from repo root. No such model-local reproduction tree was found either. Hardware parity remains a documented historical claim, not reproduced audit evidence; WGSL is explicitly unexecuted in the model README.

- **AUTHORITY_PROVENANCE** — Dispatch authorization provenance is not enforced at the router boundary. Source-confirmed: xcfe_authorized starts true and can be overwritten from req.payload. The final dispatch predicate checks it but does not itself establish trusted authorization. `src/tests/test_router_policy.cpp` covers capability filtering and MISS, not forged/missing authorization. Evidence: `rg -n "xcfe_authorized|dispatch_allowed" src/io/request_router.cpp`. This identifies a contract boundary needing explicit caller assumptions, not a demonstrated remote exploit.

- **VERIFIER_OUTPUT_ABI** — Native checks lack a shared machine-readable verdict and error contract. Projection emits [PASS]/[FAIL]; other native tests emit prose and 0/1 exits. There is no common @kind result, distinction between invalid input and failed assertion, or golden-summary hash. Evidence: `Get-Content src/tests/test_projection.cpp`; `Get-Content src/tests/test_router_policy.cpp`; rebuilt CTest output. Human CLI --help is intentionally textual and is not itself a defect.

## Next Path

1. **Reconcile track and manifest contracts.** Top Blocker: TRACK_CONTRACT_DRIFT together with MANIFEST_SPECIMEN_DRIFT. The concrete conflict spans `data/schema/micronaut-track.schema.json`, `src/micronaut/track_loader.cpp`, `data/manifest/words.manifest.json`, and `src/manifest/manifest_registry.cpp`. Establish one accepted language before expanding execution. Branch: if manifest objects are intended XCFE data, give them a typed union and explicit lowering; otherwise rewrite the specimen as legal references. Require @no-eval const true and decide whether missing id/phase is legal. Expected: zero specimen failures and false-no-eval rejected. Verify: `python artifacts/stack-intel/probe.py`.

2. **Add strict native contract conformance target.** Proposed target, not present today. Minimal Slice Plan: touch the two JSON schemas, their loaders, `src/util/json.cpp`, a new native contract test/fixture set and `CMakeLists.txt`. Reject trailing JSON and invalid escapes before admission. Add a legal Pop track and the identical track with @no-eval false as a golden pair; add a missing-id case that both layers classify identically. Use @kind khanary.contract.verdict.v1 with exit 0 pass, 1 conformance mismatch, 2 invalid fixture/harness error. No inference/trainer edits needed. Verify: `ctest --test-dir build -C Release -R KHANARY_CONTRACT_TEST --no-tests=error --output-on-failure`.

3. **Define and test canonical hash identity.** Proposed target. Follow-on A extends verification: resolve whether `data/schema/cycle-hash.spec.json` means canonical semantic JSON or exact bytes; implement its number, Unicode and key-order rules. Branch: if byte identity is intended, narrow the spec explicitly. Test reordered keys, numeric forms and fixed SHA-256 vectors; keep existing projection bytes stable unless an ABI version changes. Verify: `ctest --test-dir build -C Release -R KHANARY_CANONICAL_TEST --no-tests=error --output-on-failure`.

4. **Gate routing with trusted authorization context.** Follow-on B integrates validated contracts into `src/io/request_router.cpp`. Separate user payload from trusted dispatch context or document and enforce the trusted-caller boundary. Add forged true, explicit denial and absent-authorization cases; preserve capability MISS semantics. Depend on steps 1-2. Verify: `ctest --test-dir build -C Release -R KHANARY_ROUTER_TEST --no-tests=error --output-on-failure`.

5. **Make SDDS validation atomic and add vectors.** Proposed target. Validate all headers, version, sizes, name termination, formats and CRCs before GPU allocation/residency. Reject truncation as a whole container. Branch: if partial loading is a desired feature, expose an explicit partial contract rather than successful complete load. Add truncated-final-chunk and unsupported-version vectors alongside a valid SDDS v1 file. Verify: `ctest --test-dir build -C Release -R KHANARY_SDDS_TEST --no-tests=error --output-on-failure`.

6. **Restore geometry reproduction and bounded backend claims.** These documented commands are currently unavailable. Restore the harness or replace the model README commands with the actual maintained source location. Branch: if hardware is unavailable, keep CPU/static tests mandatory and label GPU runs as skipped. Do not claim WGSL execution or SCXQ2 interoperability from SDDS-only evidence. Verify: `python tools/build_geometry_model.py; python -m pytest tests/ -q`.

7. **Integrate conformance outputs and isolated build workflow.** Integrate steps 2-5 into CTest and emit stable sorted vector summaries alongside human output. Document the normalized child environment and isolated OutDir procedure used in this audit, or introduce a dedicated build tree. A normal output build requires the executable to be unlocked; this audit did not stop running services. Keep run timing, absolute paths and journal sequence metadata outside canonical summary seeds. Verify: `cmake --build build --config Release; ctest --test-dir build -C Release --no-tests=error --output-on-failure`.

## Todo (Future)

- [ ] Extend GPU parity for `data/shaders/` and `models/khanary-geometry-v0.3.0/kernels/`; revisit when step 6 restores a reproducible CPU reference (risk: high). Evidence: `Get-Content models/khanary-geometry-v0.3.0/README.md`.
- [ ] Add cross-language SCXQ2 interoperability only when an actual SCXQ2 codec/spec is connected to this checkout; current native SDDS v1 must not be relabeled by assumption (risk: high). Evidence: `rg -n "SCXQ2|SDDS" src docs/DDS-MODEL.md`.
- [ ] Extend inference/training after valid containers and numerical goldens are available; `forward_stub` remains a placeholder fallback, not model inference evidence (risk: high). Evidence: `rg -n "forward_stub" src/inference/inference_engine.cpp`.
- [ ] Harmonize Supernaut/Canvas and WPF interfaces when their runtime contracts actually converge; keep their independent builds and narrow claims meanwhile (risk: med). Evidence: `Get-Content tools/supernaut/README.md`; `Get-Content README.md`.

## Stack Comparison Notes

- Root C++ is the execution implementation; `bot.py` is a control client and `tools/micronaut/build_keycache.py` is a cache producer. No root Node package provides an equivalent contract gate. Evidence: `Get-Content CMakeLists.txt`; `rg -n "allowlist" bot.py`.
- `src/io/projection.cpp` provides stable ordering and byte hashing; `src/util/json.cpp` provides generic unordered serialization. Do not generalize the projection test to semantic JSON canonicalization. Evidence: `rg -n "sort|sha256|stringify" src/io/projection.cpp src/util/json.cpp`.
- `src/gpu/dds_loader.h` describes SDDS v1; `src/inference/inference_engine.cpp` also has a DDS model path. `src/io/mcp_server.cpp` lists SCXQ2 among representations, but that label is not a verified JS/native/GPU codec equivalence. Evidence: `rg -n "SCXQ2|SDDS|forward_dds" src`.

## Appendix: Evidence

The phase log below preserves all 20 questions. Good signals and branches are scoped to this checkout. Evidence files: `artifacts/stack-intel/schema-results.json`, `summary-seeds.txt`, `summary.json`, and `build/Testing/Temporary/LastTest.log` (volatile test log).

### Phase 1 — topology

1. **What is the complete depth-two topology?** Inspect sorted folder inventory above. Good: distinguish source/contracts/generated assets. Found those boundaries, with runtime data and contracts co-located under data/. Branch: classify paths explicitly; do not infer a boundary violation merely from co-location.
2. **Which commands are authoritative?** Inspect `CMakeLists.txt`, README build section and ancestor AGENTS instructions. Good: local commands correspond to the actual project. CMake/CTest apply; inherited generic npm instructions do not have a root package to operate on. Branch: use native commands; model-specific stale commands become MODEL_REPRO_DRIFT.
3. **Where are contracts anchored?** Inspect `data/schema/`, `data/grammar/`, ABI books and model schemas. Good: each has a specimen and checker. Partial: some consumers are ad hoc loaders without a contract runner. Branch: TRACK_CONTRACT_DRIFT and CONFORMANCE_COVERAGE.
4. **What verification exists?** Inspect four CMake add_test entries and src/tests/. Good: deterministic failures/outputs. Four tests rebuilt and passed, but loader/backend coverage absent. Branch: add minimal conformance gates before extending capability.

### Phase 2 — law/contracts

5. **Where is authority enforced?** Inspect track no-eval rejection, router dispatch predicate, phase driver lock, and bot allowlist. Good: hard checks with trusted input. Checks exist; router authorization defaults true and driver lock is configured audit. Branch: AUTHORITY_PROVENANCE; do not describe audit mode as enforce mode.
6. **Are canonical hashes reproducible at the intended layer?** Inspect cycle-hash.spec, util/json and projection. Good: a defined serialization plus matching vectors. Projection byte identity is covered; json-c14n-v1 is not located in source. Branch: CANONICAL_HASH_GAP.
7. **Do shipped specimens validate?** Run `python artifacts/stack-intel/probe.py` using installed jsonschema 4.26.0 and local references only. Good: all pass. Actual: 41 checked, one failing manifest; false-no-eval accepted by schema. Branch: reconcile contracts/specimens before adding fields. This is schema validation, not native round-trip coverage.
8. **Is a golden-vector runner present?** Inspect VECTORS.md, src/tests and CMake. Good: batch verdicts and stable summary hash. Current tests use in-code cases, no located MX2LEX or unified golden runner. Branch: add a KHANARY-native runner; do not introduce MX2LEX solely to match the template.

### Phase 3 — native/toolchain

9. **Is the toolchain environment-correct?** Inspect MSVC-only CMake and build/CMakeCache.txt (Visual Studio 17 2022). Initial build failed on duplicate Path/PATH. Normalizing environment keys in the child process allowed compilation; link then failed LNK1104 because normal KHANARY.exe was running. An isolated OutDir link succeeded. Good: reproducible build without stopping services. Branch: document isolated workflow, not source portability changes to an explicitly Windows project.
10. **Are native verdicts machine-readable?** Inspect source success/error paths and execute CTest plus `artifacts/stack-intel/bin/KHANARY.exe --help`. Good: stable typed pass/fail/error ABI. Existing checks return text and 0/1, no shared @kind. Branch: VERIFIER_OUTPUT_ABI. Arbitrary invalid runtime flags were not tried because they can start a service; full runtime error ABI remains unverified.
11. **Does the parser cover current specimens?** Inventory *.asx (none located), inspect JSON and KLSL parsers. Good: supported grammar plus strict invalid-input rejection. Native JSON handles nested objects/arrays/scalars but omits a trailing-input check. Branch: JSON_PARSE_BOUNDARY, not an invented .asx parser requirement.
12. **Are critical native loaders present?** Inspect src/gpu/dds_loader and manifest loaders. Good: complete validation before execution/residency. SDDS loader exists but permits partial returns. Branch: SDDS_PARTIAL_ACCEPT before new decoder frontends.

### Phase 4 — runtime/orchestration

13. **Which backends are supported by evidence?** Inspect inference dispatcher, shaders, model README and tests. Good: claims backed by reproducible tests. Root builds; four CPU-oriented tests pass; GPU inference not executed; WGSL parity is explicitly structural only. Branch: restore model reproduction and backend vectors.
14. **Are data and execution separated end to end?** Inspect registry expansion, track gate, XCFE registered ops and router predicate. Good: explicit registered dispatch and trusted authority. Explicit mechanisms exist, but payload authorization and permissive manifest coercion leave boundary contracts Partial. Branch: steps 1 and 4; not a claim of observed arbitrary code evaluation.
15. **Is compression consistent across layers?** Inspect SDDS header, docs/DDS-MODEL and SCXQ2 search results. Good: common magic/version/spec/vectors. SDDS v1 is concrete; JS/native/GPU SCXQ2 equivalence is not demonstrated. Branch: repair SDDS consistency first and defer cross-format interoperability.
16. **Is provenance separate from canonical output?** Inspect JROM sequence logging and projection filtering/order/hash. Good: relative order affects output, absolute sequence does not. Projection acceptance test verifies that narrow property. Payload strings remain verbatim, so caller-inserted volatile fields are not stripped. Branch: define canonical payload boundary before extending hash guarantees.

### Phase 5 — next path

17. **What blocker unlocks the most work?** Contract admission parity, evidenced by the track schema/loader contradiction and the failing words manifest. Good: one accepted language across tools and runtime. Branch: Next Path 1-2.
18. **What is the minimal lawful slice?** Two schema/loader pairs, strict JSON consumption, one native conformance target and positive/negative track/manifest fixtures. Good: explicit 0/1/2 exit contract and deterministic verdicts. Branch: preserve intended XCFE data via a typed schema union or legal references; no blind specimen deletion. See Next Path 2.
19. **Which follow-ons preserve direction?** A: canonical identity vectors; B: trusted routing integration. Good: coverage precedes expanded capability. Dependencies and verification commands are in Next Path 3-4; SDDS atomic verification follows.
20. **What is deferred and when reconsidered?** GPU parity expansion, SCXQ2 interoperability, training and UI convergence. Good: each has a concrete evidence trigger. See the four Todo (Future) items; no runtime implementation was attempted during the audit.

### Reproduce verification

Run from the repository root. `cmake`/`ctest` aliases below assume the installed CMake bin directory is on PATH; the actual audit used its absolute path. The Python child environment normalization is an environment workaround, not a permission bypass.

```powershell
Get-ChildItem -Force -Directory -Recurse -Depth 2 | Sort-Object FullName
Get-Content CMakeLists.txt
git status --short
python artifacts/stack-intel/probe.py > artifacts/stack-intel/schema-results.json
python -c "import os,subprocess; e={k.upper():v for k,v in os.environ.items()}; raise SystemExit(subprocess.call(['C:/Program Files/CMake/bin/cmake.exe','--build','build','--config','Release','--target','KHANARY','--','/p:OutDir='+os.path.abspath('artifacts/stack-intel/bin')+os.sep],env=e))"
& 'C:\Program Files\CMake\bin\ctest.exe' --test-dir build -C Release --output-on-failure
& .\artifacts\stack-intel\bin\KHANARY.exe --help
Test-Path tools/build_geometry_model.py
Test-Path tests
python artifacts/stack-intel/write_report.py
```

## Deterministic Summary

Counts: folders are inventoried physical paths at depth 1-2 excluding the audit output; contracts counts the three root `data/schema/*.schema.json` documents, not every configuration/specification or model copy. Hash input is the lexicographically sorted GAP/NEXT seed lines, joined by LF with no terminal newline, UTF-8. No timestamps or absolute machine paths occur in seeds/summary. This is a findings hash, not a digest of every repository file. Re-rendering preserves the same summary for the same inventory and audit inputs; re-rendering alone does not re-run native tests.

```json
{
  "@kind": "as-xcfe.stack-intel.summary.v1",
  "@counts": {
    "folders": 63,
    "contracts": 3,
    "gaps": 9,
    "next_steps": 7,
    "specimens_checked": 41,
    "specimen_failures": 1,
    "native_tests_passed": 4
  },
  "@top_blocker": "TRACK_CONTRACT_DRIFT",
  "@summary_hash": "sha256:29d1802475c8f587585229de3df523cd22c00917fe820e7009e9a9b1951d6b16"
}
```

## Decisions

[]
