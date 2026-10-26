# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

**MSVC 2022+ is required** — gcc and clang will not work (`CMakeLists.txt` enforces this with a FATAL_ERROR).

Configure and build using the VS 2022 BuildTools cmake:

```powershell
# Configure (first time or after CMakeLists changes)
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake

# Build all targets
cmake --build build --config Release

# Build a single target
cmake --build build --config Release --target KHANARY
```

The canonical cmake executable lives at:
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`

MSBuild alternative:
```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe" `
  build\KHANARY.sln /p:Configuration=Release /m /nologo /v:minimal
```

### Bat shims (use these on this machine)

`build.bat`, `test.bat`, and related shims in the project root handle E: → C: drive restrictions automatically. **Do not invoke cmake, MSBuild, or test binaries directly** — route through the bat tools. If the bat files themselves need to run in a restricted context, stage them to `C:\Temp\` first.

```powershell
.\build.bat           # configure (if needed) + build all Release
.\build.bat KHANARY   # build a single target
```

See `AGENTS.md` at the repo root for the full bat-tool vocabulary (`read.bat`, `write.bat`, `list.bat`, `hash.bat`, `patch.bat`, `run.bat`). **Never write directly to E: paths from a subprocess context** — write to `%TEMP%\<name>` first, then use `write.bat <dest> %TEMP%\<name>`.

### Output location

All executables land in `bin/` regardless of config — build trees (`build/`, `_build/`, `.build-copilot/`) are disposable. Launcher scripts and conformance tests reference `bin\KHANARY.exe`, not a build-tree path.

Override with `-DKHANARY_BIN_DIR=<path>` on the cmake configure line.

### Optional dependencies

- **DirectML**: searched via `$DIRECTML_PATH`, vcpkg, or `extern/DirectML/`. Missing → `KHANARY_HAS_DIRECTML` not defined; DML ops disabled at compile time (warning only, not an error).
- **llama-haswell**: pre-built static libs at `llama-haswell/` — linked as `kh_llama`. No E: drive access required at build time. LNK2038 on `/MT` vs `/MD` mismatch is the known failure mode here.
- **Python 3**: required for `KHANARY_SCHEMA_TEST` and many grammar/conformance tests (`tools/check_*.py`, `tools/grammar/mini_transpilers.py`).

## Testing

Run the full suite via ctest (requires tests already built):
```powershell
ctest --test-dir build --config Release
```

**E: drive execute restriction** — `bin\*.exe` cannot be run directly from E:. Use `test.bat` to stage binaries to C: before execution:

```powershell
test.bat KHANARY_CONTRACT_TEST                              # default args auto-filled
test.bat KHANARY_PTEST data\projection_abi_v1.json data\gyro_abi_v1.json
test.bat KHANARY_CANONICAL_TEST data\conformance\canonical.json
test.bat KHANARY_GREETING_TEST data\grammar\greeting-bank.json
test.bat KHANARY_PHASE_DRIVER_LOCK_TEST . build
test.bat KHANARY_LEXICON_TEST .
test.bat KHANARY_SEMANTIC_TEST .
test.bat KHANARY_AIML_PHASE_BANK_TEST data
test.bat KHANARY_QUESTION_SELECTOR_TEST data
```

Tests with D3D11 deps (`KHANARY_FOLD_SCORE_DISPATCH_TEST`, `KHANARY_GPU_DEVICE_TEST`, `KHANARY_SEMANTIC_ATTENTION_TEST`, etc.) skip gracefully when no D3D11 device is available.

Run all Python grammar/conformance checks in one step:
```powershell
py -3 tools\run_conformance.py
```

CMakeLists.txt defines 50+ additional test targets (CFPC pipeline stages, tensor/metric, LLM integration, app-host, MCP client, Python grammar suites). List them all with:
```powershell
ctest --test-dir build --config Release -N
```

## Running KHANARY

```powershell
# Standalone (no model)
bin\KHANARY.exe --data-dir data --port-mcp 8764 --port-rpc 8787

# With GGUF sidecar
bin\KHANARY.exe --gguf-model <path.gguf> --data-dir data

# With DDS shards (GPU-resident expert tensors)
bin\KHANARY.exe --dds-kson <folds.kson> --dds-manifest <manifest.json> --dds-shards <dir> --port-rpc 8787
```

Default ports: MCP `8764`, JSON-RPC `8787`.

## Architecture — key lane boundaries

**Keep these two lanes separate:**

| Lane | Owner | Purpose |
|---|---|---|
| Apps host | `src/host/` | User-facing app shells, REST API, JROM/CBOR ROM tools, KQL/GAS bridges, Atomic DOM widgets. Loopback-only (127.0.0.1). |
| MCP | `src/io/mcp_server.cpp` | Model/tool protocol. Do not merge app-surface logic here. |

`KHANARY_APP_HOST` is a separate CMake target from `KHANARY`.

**AGENTS.md** at the repo root is the canonical bat-tool and agent contract. Read it before issuing build, file, or patch operations — it defines the full shim vocabulary and E: drive write rules.

**Epistemic states** — KHANARY never fabricates. Responses carry one of `SUPPORTED`, `MISS`, `UNKNOWN`, or `CONTRADICTED`. A miss returns `MISS` with a research fallback plan, not a generated answer. Do not write code that collapses these states to a guess.

## Code style

- C++20, MSVC `/W4 /WX /utf-8 /permissive- /Zc:__cplusplus`
- Compile definitions: `UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN`
- `KHANARY_HD4600_CEILING=1879048192` is the 1792 MB GPU resident ceiling for the Intel HD 4600 target
- No formatter config in the repo — match surrounding style when editing

## Key domain vocabulary

- **K'UHUL C₆ phase cycle**: Pop → Wo → Yax → Sek → Ch'en → Xul
- **XCFE**: policy evaluator; owns `@if/@then/@else/@for/@from/@cause/@direct` operator execution and phase-gate enforcement
- **Micronaut / track**: expert-dispatch unit; routed by W·C·R gravity scores
- **CFPC**: context-field pattern classifier — deterministic, no model required
- **Fold**: horizontal (Δθ=0) or vertical (Δθ=π/3) structure; `fold_engine` owns cost table
- **gram_store**: KXML gram nodes; prompts/responses/plans are key-grammar memory events logged to `data/journal/gram-events.jsonl`
- **JROM**: JSON ROM — serialized memory format used by the apps host lane

See @docs/GLOSSARY.md for the full term reference and @docs/KHANARY-CPP-SPEC.md for the full source file map.
