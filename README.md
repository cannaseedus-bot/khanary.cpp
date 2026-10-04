# KHANARY.CPP

Single native Windows executable — GPU inference + manifest runtime + micronaut engine + MCP gateway.

## Requirements

- Windows 10 x64, MSVC 2022+, CMake 3.28+
- Windows SDK 10.0.26100
- [vcpkg](https://vcpkg.io): `directx-headers` package
- Intel HD 4600 or compatible D3D11 GPU

## Build

```powershell
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## Validate

The conformance runner builds an isolated `build/conformance` tree and runs all
contract and schema checks. This is the primary validation path:

```powershell
python -m pip install -r tools/requirements-conformance.txt
python tools/run_conformance.py
```

See [conformance contracts and verification](docs/CONFORMANCE.md) and
[audit remediation status](docs/STACK-INTEL-REMEDIATION.md). The runner builds the
runtime and checks without replacing an executable held open by an existing server.

```
KHANARY.exe [options]
  --port-mcp <n>         MCP server port (default: 8764)
  --port-rpc <n>         JSON-RPC server port (default: 8787)
  --data-dir <path>      Data directory (default: ./data)
  --weights-dir <path>   SCXQDDS weight shards directory
  --gguf-model <path>    GGUF file preloaded into system RAM (Gemma hook metadata also enabled)
  --dds-kson <path>      KAST/1 .kson driver file
  --dds-manifest <path>  DDS shard manifest JSON
  --dds-shards <path>    Directory containing fold_N.dds files
  --gpu <hint>           GPU adapter hint (default: dedicated)
  --proj-book <path>     Projection ABI book JSON (default: data/projection_abi_v1.json)
  --gyro-book <path>     Gyroscope ABI book JSON (default: data/gyro_abi_v1.json)
  --build-keycache      Build key-store binary cache and exit
  --build-lexicon-grammar Build dictionary key-grammar JSONL files and exit
  --verbose             Verbose logging
  --help                Print help and exit
```

## Launch everything (runtime + WPF shell)

From repo root:

```bat
KHANARY.bat
```

This launcher:
- ensures the `mx2lm-wpf-cli-v1` bundle exists,
- refreshes `.\.ui-shell` as an example/template workspace (not a host),
- builds `data\factory_keys.bin` + `data\factory_keys.cachemeta.json` on first run (one-time compile),
- starts `KHANARY.exe` on RPC/MCP ports if it is not already ready (passing `--gguf-model` from `KHANARY_GGUF_MODEL` and DDS args from `KHANARY_DDS_KSON` / `KHANARY_DDS_MANIFEST` / `KHANARY_DDS_SHARDS`),
- starts CHAT-µ (LFM2.5-1.2B) and TOOL-µ (gemma-3-1B) llama-server instances — each checks if its default port is free, scans the next 20 if not, and resolves to the first available port,
- starts a KHANARY single-completion OpenAI-compatible gateway (`tools\khanary_completion_api.py`) that fronts `key_query` and allows phase-gated CHAT-µ/TOOL-µ contributions on one endpoint,
- writes `data\ports.json` with all resolved live ports so `lfm_factory.py` and `bot.py` connect automatically,
- serves the custom WebUI from `build\webui` via `--path` when `index.html` is present there,
- opens the WPF terminal/console shell.

In the WPF shell, bot output is rendered as chat text first (with track + ngram lines), rather than dumping raw JSON as the primary response view.
Direct UI-side ngram/action/concept display customization is available in `artifacts\micronaut-bundles\mx2lm-wpf-cli-v1\cli\wpf\Ngrams.cs` (regenerated from `tools\micronaut\build_mx2lm_wpf_bundle.ps1`).
That bundle build script also writes `bundle_duplicate_files.json` to track same-content duplicate files inside the bundle by SHA-256 hash.
Timer responses from `utility-runtime` now render a live countdown line in the WPF output pane after `timer_started` / `timer_status`.
Calendar prompts like `What is today?` and `What is the first day of the week?` are handled locally by `utility-runtime` (no web-research fallback).
Bot responses also render as interactive cards with context-driven static VAULT bot pose selection (from position-locked cells in `asx-atomic-blocks.png`, mapped by JSON source/status/intents/actions) plus chat-bubble metadata.
Common definition prompts like `What is the sun?` / `What is the moon?` / `What is K'UHUL?` are now answered locally by `utility-runtime` with deterministic fact snapshots (local-file utility path for K'UHUL definitions).
Every submitted `key_query`/`infer` prompt is now archived into runtime memory/key-store with typed gram channels (`unigram`, `bigram`, `trigram`, `quadgram`, `coarse_gram`, `function_gram`, `tool_gram`) plus word associations; when prompts include local file paths (`.md`, `.html`, `.txt`, `.json`, `.grammar`, `.ebnf`, etc.) the file contents are ingested into the same pipeline. `.pdf` paths are tracked and flagged as scraper-required.
Lexical cues and morphemes are also grounded into canonical semantic atoms (`word_to_atom:*`, `atom:*`) with fixed phase contracts (`atom_phase_contract:*`) for recognition only—this does **not** grant learned associations execution authority over K'UHUL phase algebra.
The dictionary feed (`data\gram\english-dictionary.jsonl`) now seeds lexical similarity keys (`word_similarity:*`, `synonym:*`) and prompt-time semantic association links (`assoc:semantic:*`) for meaning-preserving alternate wording.
The runtime now exposes a deterministic **Fold Graph Parser** surface via JSON-RPC (`fold_graph_parse`) and `key_query` triggers (`fgp: ...`, `@fgp ...`, `fold graph parse ...`), producing `atoms/relations/books/candidates/unresolved/provenance` from lexical evidence, with canonical relation typing (`infer_relation_from_lexical`) + relation phase hints, while keeping phase binding canonical.

The semantic core is frozen as `K = (G_semantic, Θ, 𝒜, Π, X)`: Pop through Ch'en are five fold algebras (`α_P`, `α_W`, `α_Y`, `α_S`, `α_C`), and Xul is the sixth phase but **not** a sixth fold — it is the policy-governed canonicalization/commit boundary. Fold Phases are C6 operations over a Fold, not fold kinds or learning states; the full runtime coordinate is `𝒦_t = (F_h, φ, T, s, σ, X_t)`. A Fold Graph `𝒢_F(h)` is the directed typed relational graph recoverable by unfolding a Fold Handle; Yax derives the query-local Active Fold Graph `𝒢_F^Q(h)` without mutating persistent topology, and legal traversal must satisfy both semantic edge orientation and C6 phase-forward law. Fold contraction must preserve recoverable local structure. Executors are heterogeneous behind the envelope `E_θ(contract,input,context) -> (result,evidence,status)`; CPU/GPU/model/SCXQ2 paths are replaceable realizations, not semantic law.

## JOKE-µ delivery-frame routing

`JOKE-µ` is now a first-class reusable µ-track like `PLAN-µ` and `ADVISOR-µ`. It is not a freeform joke authoring module; it is a humor-frame routing layer that permits ambiguity, pun bridges, metaphor routes, and punchline-compatible sense collapse while preserving the selected semantic key separately from rendered answer text.

The delivery-frame fold is declared in `data/manifest/joke.manifest.json` and governed by `data/tracks/JOKE-u.semantic-tracks.v1.json`:

```text
wind-up -> frame:joke_setup
pitch   -> frame:timed_delivery
swing   -> frame:collapse
```

The baseball words are surface aliases, not the selected domain. Boundary rules keep the fold structural: frames are declared, beats are phase-typed, roles are schema-bound, collapse is verified, subversion is a declared variant, and surface/key-word co-occurrence cannot promote itself into a frame.
`key_query` also supports phase-governed lexical rewrites (`synonymize: ...`, `synonym: ...`, `syn: ...`, `@syn ...`) that apply dictionary alternates with a Ch'en semantic-invariance gate before emit.
When a question still misses local facts, `web_research` dispatch now returns explicit routed-status text instead of mirroring the user query.
Both utility and web-research payloads carry runtime authority metadata (`governing_track`, `activation`, and `response_phase`) so WPF consumes track/phase state directly from runtime JSON.
If the sprite sheet is unavailable, the UI falls back to the SVG-C vector badge renderer.

Use `KHANARY.bat --dry-run` to print the launch commands without executing them.
Any extra arguments are passed through to WPF mode (for example: `KHANARY.bat -Configuration Release`).

You can precompile the cache directly (preferred: Python packer):

```powershell
py -3 tools\micronaut\build_keycache.py --data-dir .\data --repo-root .
```

Fallback via runtime:

```powershell
.\build\Release\KHANARY.exe --data-dir .\data --build-keycache
```

Compile dictionary lexicon + definition-relation key grammar artifacts:

```powershell
.\build\Release\KHANARY.exe --data-dir .\data --build-lexicon-grammar
```

This also emits runtime EBNF specs:
- `data\grammar\aiml.runtime-grammar.ebnf`
- `data\grammar\key-grammar-json.runtime.ebnf`

The key-grammar EBNF (v2) is a structured contract, not a bag-of-words list: every typed gram channel (`unigram`, `bigram`, `trigram`, `quadgram`, `coarse_gram`, `function_gram`, `tool_gram`) declares an ordered slot width, first-occurrence position, a semantic role, a K'UHUL phase binding, provenance, a key-address form, and a consumption rule. Regenerate the checked-in EBNF artifacts after editing the runtime strings with the mini transpiler:

```powershell
py -3 tools\grammar\mini_transpilers.py ebnf-sync
```

## Adaptive greeting learning (phase-supplied bank)

Salutation prompts are handled by the adaptive style system: every submitted prompt already archives its typed n-grams (all seven channels) into the key-store, and the greeting layer additionally records observed greeting forms, adapts to the user's lingo over time, mirrors greetings the user repeats (reactionary, `Pop.reactive_mirror`), rotates greeting registers through the C6 phases, and synthesizes its own associative words by blending the user's top lingo cues (`Yax.synthesized_greeting`). Greeting registers are supplied per phase in `data\grammar\greeting-bank.json` and persist under `mx2lm:style_profile:v1`, `greeting:observed:*`, `greeting:variant:*`, and `mx2lm:lingo*` keys. The `GREETING-µ` micronaut track (`data\tracks\GREETING-u.semantic-tracks.v1.json`) governs this surface, routing through ELIZA's gyroscope measurement basis.

Mini transpilers over the key-grammar surfaces:

```powershell
py -3 tools\grammar\mini_transpilers.py greeting-bank-compile   # bank -> greetings.aiml + key-grammar JSONL
py -3 tools\grammar\mini_transpilers.py advisor-bootstrap-compile # authored advisor contract -> advisor.aiml + key-grammar JSONL
py -3 tools\grammar\mini_transpilers.py advisor-check            # replay routes and verify ADVISOR-µ/card binding
py -3 tools\grammar\mini_transpilers.py plan-bootstrap-compile   # authored plan contract -> plan.aiml + key-grammar JSONL
py -3 tools\grammar\mini_transpilers.py plan-check                # replay routes and verify PLAN-µ/tasks-card binding
py -3 tools\grammar\mini_transpilers.py validate                 # check JSONL rows against the EBNF contract
py -3 tools\grammar\mini_transpilers.py gyro-align               # gyro book <-> tracks <-> field.xml alignment
```

## Control flow as syntax sugar

The key grammar expresses control flow directly: control words are first-class *flow* operators, not a bag of operator tokens. Each one declares a deterministic operator authority (`ELIZA-1:<OP>`), the XCFE op it lowers to, the K'UHUL phase its construct runs in, its block role and its arity:

| Sugar | Lowers to | Phase | Role |
| --- | --- | --- | --- |
| `if <condition>` | `@if` | Pop | conditional-branch |
| `then <consequence>` | `@then` | Wo | ordered-transition |
| `else <alternative>` | `@else` | Yax | branch-alternative |
| `when <trigger>` | `@if` | Pop | temporal-condition |
| `how <method>` | `@direct` | Wo | method-execution |
| `who <actor>` | `@node` | Pop | entity-binding |
| `why <reason>` | `@cause` | Yax | causal-link |
| `for each <item> in <collection>` | `@for` | Sek | iteration-or-scope |
| `from <source>` | `@from` | Pop | origin-extraction |
| `as <role>` | `@as` | Sek | role-cast-alias |
| `<lhs> equals <rhs>` | condition `==` | Ch'en | comparison-equivalence |

A control clause therefore reads like English and lowers to an executable XCFE control node:

```
if mode equals debug then verbose else quiet
  -> control plan: form=if/then/else, condition="mode == debug", 4 phase-bound steps
  -> XCFE: {"@if": "mode == debug", "@then": "verbose", "@else": "quiet"}

for each item in list        -> {"@for": "list", "@as": "item"}
from archive.deep as source  -> {"@from": "archive.deep", "@as": "source"}
why did it fail              -> {"@cause": "did it fail"}
how do i deploy              -> {"@direct": "deploy"}
name equals khanary          -> {"@if": "name == khanary", "@then": true, "@else": false}
```

**The ops execute, not just lower.** `src\kuhul\xcfe_executor.{h,cpp}` registers the control-flow op library — `@if`/`@then`/`@else` natively plus `for`, `from`, `as`, `cause` and `direct` — and hands every op a recursive evaluator, so an op can evaluate its own body. A native control key dispatches to the op of the same name with the whole object as args (precedence `@if`, `@for`, `@from`, `@as`, `@cause`, `@direct`), and `@op`/`@fn` names may carry the leading `@`:

| Op | Executes | Reads |
| --- | --- | --- |
| `for` | a JSON array or comma/space-separated collection from the context, evaluating `@do` per element | `@for`/`each`, `@as`, `@index`, `@do` |
| `from` | origin extraction (URI origin, first namespace segment, or first delimited element), binding it for `@do` | `@from`, `@as`, `@do` |
| `as` | alias/cast (`string`, `upper`, `lower`, `key`, `number`, `bool`), binding it for `@do` | `@as`, `@value`, `@cast`, `@do` |
| `cause` | the causal link record (`@causes`, `ELIZA-1:CAUSE`, phase `Yax`) | `@cause`, `@because` |
| `direct` | resolves a method to the host-provided action `op.<method>` | `@direct` |

An unbound collection is empty rather than the variable name read as a one-element list, and an unregistered op evaluates to `null` — XCFE never invents authority or side effects.

The single authority is `src\micronaut\control_flow.{h,cpp}` (`control_operators()`, `plan_control_flow()`, `lower_to_xcfe()`). It drives:

- the runtime's typed control bindings (`Mx2LmNGram::resolve_control_bindings` — phase-bound `detect_phase`/`emit_phase` per operator, C6 successor),
- prompt archiving: every submitted prompt records its control flow under `control:*` keys (`control:<word>`, `control:<word>:phase:<phase>`, `control:xcfe:<op>`, `control:form:*`, `control:condition:*`) and writes `control_gram` rows into the gram store,
- the EBNF v2 control-flow contract (`control_flow_row`, `control_plan_row`, the sugar clause productions, and the declared vocabulary/bounds),
- the checked-in artifact `data\grammar\control-flow.key-grammar.jsonl` (one row per operator, emitted by `--build-lexicon-grammar`),
- the `CONTROL-µ` micronaut track (`data\tracks\CONTROL-u.semantic-tracks.v1.json`) and the `words.manifest.json` routing for the control words.

```powershell
py -3 tools\grammar\mini_transpilers.py control-check            # EBNF contract <-> artifact <-> Python mirror
```

`control-check` fails if the EBNF vocabulary, the runtime-emitted operator rows, or the transpiler's `CONTROL_RULES` mirror drift apart, so the sugar cannot silently diverge from the operator authority. `KHANARY_CONTROLFLOW_TEST` executes every sugar form through the real executor, so lowering and execution cannot drift apart either.

## Micronaut command wrapper (agent-factory)

Thin Codex-callable wrappers for `src\micronaut` are included without changing runtime behavior:

- Skill wrapper: `.codex\skills\khanary-micronaut-runtime\SKILL.md`
- Command shim: `tools\micronaut\kmu.ps1`
- Tool-call dataset rows: `data\mx2lm\micronaut_wrapper.toolcalls.jsonl`

Examples:

```powershell
powershell -ExecutionPolicy Bypass -File tools\micronaut\kmu.ps1 status
powershell -ExecutionPolicy Bypass -File tools\micronaut\kmu.ps1 dispatch "@math solve 2x + 4 = 0"
powershell -ExecutionPolicy Bypass -File tools\micronaut\kmu.ps1 semfold "Coordinator|Librarian"
```

## LFM distillation factory

`tools/lfm_factory.py` feeds spec prompts to CHAT-µ (LFM2.5-1.2B) and writes KHANARY data files.

```powershell
# start CHAT-µ server first (finds a free port automatically)
tools\start-lfm.bat

# run a spec file
python tools/lfm_factory.py run specs/sample.jsonl

# interactive ask
python tools/lfm_factory.py ask "list 5 Pop-phase greetings" --format greeting_bank
python tools/lfm_factory.py ask "scaffold a todo app" --format scaffold --name todo-app --out build/scaffold

# list supported output formats
python tools/lfm_factory.py formats
```

Spec file — one JSON object per line:

```jsonc
{"id":"greet-pop-1","format":"greeting_bank","phase":"Pop","count":5}
{"id":"ctrl-if-1","format":"control_flow","capability":"if/then/else"}
{"id":"scfml-1","format":"scfml_pattern_learning","namespace":"plan","track_id":"PLAN-µ","topic":"semantic branch learning"}
{"id":"klsl-1","format":"code_grammar","language":"klsl","class_name":"KLSL"}
{"id":"tmpl-1","format":"template_fill","template":"vite-react-ts","name":"MyApp","features":"dark mode,rpc"}
```

Supported formats: `greeting_bank`, `key_grammar`, `track`, `capability_gap`, `aiml`,
`control_flow`, `scfml_pattern_learning`, `code_grammar`, `raw_jsonl`, `scaffold`, `template_fill`.
SCFML rows are emitted as `scfml.branch.v1` plus `scfml.pattern_learning.v1`; generated candidates default
to `UNKNOWN` and require `SUPPORTED_or_user_verified` before promotion.

The factory reads `data/ports.json` (written by `KHANARY.bat` or `start-lfm.bat`) to auto-connect
to the live CHAT-µ port — no hard-coded port needed after the server starts.

### Standalone LFM launcher

```bat
tools\start-lfm.bat          # CHAT-µ (LFM2.5-1.2B, default port 17889)
tools\start-lfm.bat --tool   # TOOL-µ (gemma-3-1B, default port 17890)
```

Both check if the server is already ours (skip), then test if the target port is free — if not,
scan the next 20 ports via `TcpListener`, start the server on the resolved port, and merge the
live port into `data/ports.json` so the factory and bot pick it up.

## TEMPLATE-µ and ASX design kits

23 pre-built UI kits live in `data/templates/registry.json` (22 HTML + 1 Vite):

| Category | Kits |
|---|---|
| Atomic / component | asx-atomic, asx-atomic-guide, asx-components |
| Block layouts | asx-blocks, asx-blocks-ui |
| Chat / console | asx-chat, asx-terminal, asx-terminal-adv, asx-console-html, asx-guix-chat |
| Control / cockpit | asx-control, asx-ghost, asx-cockpit, asx-trading |
| Compression / tunnel | asx-tunnel, asx-compression, asx-compression-white |
| Specialty | asx-mathml, asx-klsl-compiler, asx-terminaa, asx-scx-spec, asx-studio |
| Vite React+TS | vite-react-ts |

All HTML kits share house tokens: `#16f2aa` accent teal · `#38bdf8` sky blue · `#020617` dark bg.
Slots use `{{slot_name}}` syntax; LFM expansion markers use `{{lfm:expand — instruction}}`.

Apply a kit via the factory:

```powershell
# fill slots + expand LFM markers, copy brand.js/chat.js alongside
python tools/lfm_factory.py ask "brand an asx-chat kit for MyApp" ^
  --format template_fill --template asx-chat --name "MyApp" ^
  --set brand_accent=#ff6b35 --out build/webui

# scaffold a Vite project
python tools/lfm_factory.py ask "scaffold vite-react-ts todo app" ^
  --format scaffold --template vite-react-ts --name todo-app ^
  --features "dark mode,rpc client" --out build/todo-app
```

`TEMPLATE-µ` micronaut track (`data/tracks/TEMPLATE-u.semantic-tracks.v1.json`) governs kit
selection, slot filling, and LFM expansion — phase Yax (associative).

## WebUI personalization

When `llama-server.exe` is started with `--path <dir>`, it serves files from that directory
instead of the embedded WebUI. KHANARY.bat uses `--path %KHANARY_WEBUI_PATH%` (default:
`build/webui`) when `index.html` exists there.

Two scripts in `data/webui/` wire any ASX kit for live use:

- **`khanary-brand.js`** — floating gear button → settings drawer: title, logo URL, accent/secondary/bg/surface color pickers, template switcher. Reads `/khanary-registry.json` to populate the switcher. Saves to `localStorage` key `khanary_brand`; applies all CSS var name variants simultaneously so any ASX kit responds without modification.
- **`khanary-chat.js`** — wires `#chat-thread` / `#chat-input` / `#send-btn` in the kit; falls back to an injected floating chat bubble. Targets `http://127.0.0.1:17892/v1/chat/completions` by default (override via `localStorage.khanary_completion_api`) and supports both SSE streaming and regular JSON completion responses.

`apply_template_kit()` in the factory injects a `:root` override block for brand tokens,
appends both scripts before `</body>`, and copies `khanary-registry.json` to the output dir.

## Ports

| Port | Service |
|------|---------|
| 8764 | MCP gateway (20 tools) |
| 8787 | JSON-RPC 2.0 |
| 17892 | KHANARY single completion API (`/v1/chat/completions`) |
| 17889 | CHAT-µ — LFM2.5-1.2B (default; auto-reassigned if taken) |
| 17890 | TOOL-µ — gemma-3-1B (default; auto-reassigned if taken) |

`data/ports.json` holds the live resolved ports written at server startup:

```json
{
  "chat_mu": 17889,
  "tool_mu": 17890,
  "completion": 17892,
  "mcp": 8764,
  "rpc": 8787,
  "abstract_ports": {
    "chat": { "port_ref": "chat_mu" },
    "tool": { "port_ref": "tool_mu" },
    "tensor": { "port_ref": "rpc" },
    "memory": { "port_ref": "rpc" },
    "control": { "port_ref": "mcp" }
  },
  "known_micronauts": {
    "ADVISOR-µ": { "abstract_port": "chat" },
    "TOOL-µ": { "abstract_port": "tool" },
    "TENSOR-µ": { "abstract_port": "tensor" }
  }
}
```

Consumers should keep using the legacy top-level keys when they only need the raw socket,
but micronaut-aware code should bind through `known_micronauts[*].abstract_port` and
`abstract_ports[*].port_ref`. Do not allocate one concrete port per micronaut; the known
micronauts share abstract lanes such as `chat`, `tool`, `tensor`, `memory`, `control`,
`completion`, `mcp`, and `rpc`.

## Bot Command Center (`bot.py`)

`bot.py` reads `openai.yaml` and can run a command-center flow that lets the model emit:

```xml
<tools>{"tool_calls":[...]}</tools>
<tools><![CDATA[{"tool_calls":[{"verb":"fold","query":"decision://choose","$agent":"advisor"}]}]]></tools>
```

The script unwraps standard CDATA, literal `<CDATA>...</CDATA>`, and `<agent>...</agent>` envelopes; the inner payload must still be strict JSON. Optional `$agent`/`agent` fields resolve only against profiles declared in `openai.yaml` (`advisor`, `chat`, `chat_mu`, `tool_mu`) and carry `authority: "config_reference_only"`. The script then validates the call against `command_center.allowlist` before executing local KHANARY MCP/RPC calls.

Typed semantic envelopes are recognized in the same pass. `<route>` and `<decision>` become `micronaut_dispatch` calls, and `<commands>` contributes ordinary allowlisted tool calls. `<think>`, `<reason>`, `<plan>`, `<task>`, `<grammar>`/`<keygrammar>`, `<sql>`, `<memory>`, and `<verify>` return proposal-only metadata; `<commit>` is explicitly blocked without XulGate.

JROM is the runtime tape player / provenance journal for these envelopes: it records `semantic_envelope` events with type, `epistemic_type`, disposition, status, content, and result. Epistemic types are constrained to observation/preference/plan/generated_output/verified_fact/failed_retrieval/capability_metric/execution_trace. JROM does **not** decide truth, execute SQL, or commit state; it preserves routed observations so replay, Ch'en review, and later learning can inspect what was proposed versus what was authorized.

Examples:

```powershell
python bot.py center "Build a dog walking app and show the blueprint"
python bot.py center "Show app options" --dry-run
python bot.py rpc key_query "{\"query\":\"APP: Dog Walking App\"}"
python bot.py rpc key_query "{\"query\":\"organic compounds in space\"}"
python bot.py tui --mode center
python bot.py tui --mode center --resume
python bot.py khanary-flags
python bot.py khanary-run --dry-run -- --data-dir data --port-rpc 8787 --verbose
```

`bot.py tui` provides an interactive terminal chat feed with unrestricted text input plus slash commands:
`/model`, `/tools`, `/resume`, `/new`, `/save`, `/fork`, `/login`, `/launch`, `/mcp`, `/rpc`, `/memory`,
runtime verb actions like `/<verb>` (from `command_center.verbs`), and generic `/<skill-name>` fallback
to `key_query` (`@<skill-name> ...`). It also supports direct file loading for TS/markup/text:
`/load path\\to\\file.ts [optional prompt]`.
Replies are text-first (not JSON by default); use `/raw on` when you want full structured JSON output.
`/load` now prints the exact input payload block sent to the model/runtime so markup is visible in the feed.
When ANSI colors are supported, the input form uses a distinct full-width dark background with white bold text and user chat rows render with contrasting styling (`NO_COLOR=1` disables color).
The TUI keeps the input prompt docked at the bottom, redraws the message window above it so conversation scrolls upward, and renders user/bot chat boxes in reversed alignment (user right, bot left).
When RPC fallback is offline, TUI attempts to auto-launch `KHANARY.exe` once (disable with `--no-auto-launch-khanary`).
The docked compose area is a multi-line input region (5-row height target, auto-clamped by terminal size).
Multiline compose is supported: in Windows interactive TTY, use Shift+Enter for newline and Enter to send; compatibility fallback also supports trailing `\` continuation.
If your terminal has repaint/ANSI quirks, set `KHANARY_TUI_SIMPLE_INPUT=1` to force classic line input fallback.

`key_query` now has a utility-intent lane before app/ngram fallback:
- live clock/calendar snapshots (`what time/date/day/month/year is it`)
- deterministic arithmetic (`calculate 5 times 500`, `what is 12 divided by 3`)
- in-memory timers (`start timer for 5 minutes`, `timer status`)

Utility responses return `source: "utility-runtime"` with explicit statuses (`clock_snapshot`,
`calculation_complete`, `timer_started`, `timer_status`, `knowledge_snapshot`, `style_profile_updated`) so repeated prompts do not replay stale fallback text.

Scheme-addressable micronaut calls are also recognized in `key_query` routing (`math://...`, `time://...`, `date://...`, `plan://...`, `decision://...`, `mx2lm://...`, etc.). Track contracts may declare `schemes`, `verbs`, `aiml_patterns`, `aiml_templates`, `key_patterns`, `key_templates`, `key_grammar`, and `tokens` (top-level or under `route_contract` / `address` / `uri` / `callable`) to seed fast-path resolver selection before normal blend ranking. `key_*` fields are callable/learning surfaces, not fact authority.

Micronaut route contracts have matching transpiler surfaces: `python tools\grammar\mini_transpilers.py micronaut-transpile` writes `data\grammar\micronaut-route-contracts.key-grammar.jsonl`, while the C++ `MicronautTranspiler` projects live registry entries through RPC `micronaut_transpile`. Both normalize semantic data into runtime classes/artifacts; neither executes a micronaut nor grants semantic authority. Each row also carries a `micronaut.class_projection.example.v1` block with identity/cognitive/memory/execution/method examples; that block is explicitly `example_only` and must not be treated as schema law. Optional track fields `brain_profile`, `personality`, `capability_grams`, and `progression` are preserved as `micronaut_profile` projections; they may bias routing/presentation but cannot claim semantic, topology, verification, or commit authority.

`data/grammar/micronaut.runtime-grammar.ebnf` is the standalone Micronaut grammar surface for declarations, profile projections, tensor slots, route contracts, known AIML/SCFML pattern/template declarations, key-grammar bundles, and `semantic_envelope` JROM rows. `tools/check_micronaut_grammar.py` keeps that EBNF aligned with the track schema, JROM `epistemic_type` set, and manifest anchors. Pattern/template declarations are callable/projection surfaces and carry the same non-authority boundary.

Key-grammar rows may now carry an optional `hyperbolic_event` projection (`key_grammar.hyperbolic_event.v1`) with `origin`, `event_role`, `rapidity`/`rapidity_rho`, `direction`/`direction_vector`, `minkowski_interval`, `interval_class`, and `epistemic_state`; standalone rows use `$schema: urn:kuhul:khanary:schema:key-grammar-event:v1` with nested `event_record`, `hyperbolic_metrics`, and `bounds`. These fields describe a candidate continuation on the future sheet; interval classes must match the Minkowski interval sign, and `semantic_authority` must remain `false`. RPC `journal_hyperbolic_event` writes a typed JROM `hyperbolic_event` row with epistemic type `observation`, preserving replay evidence without committing facts.

`decision://choose`, `decision://select`, and `decision://evaluate` route to DECISION-µ. Pass a `decision` object with `branches`, `actions`, `outcomes`, and an explicit `objective` (`maximize_utility` or `minimize_loss`). `expected_utility` requires supplied branch probabilities; if probabilities are unresolved, DECISION-µ either selects the explicit robust model (`maximin`/`minimax`) or returns `decision_unknown` with `required_evidence`. A selected action sets `decision_collapse: true` but always keeps `fact_collapsed: false` when branch facts remain `UNKNOWN`.

Advisor/LFM uncertainty triggers now have dedicated key-grammar + AIML routes:
- fact-gap phrasing like `I don't know the answer to that question...` triggers `advisor_web_research_trigger` and routes to ADVISOR-µ / WEB-RESEARCH-µ.
- tool-uncertainty phrasing like `I'm not sure what tool to use...` triggers `lfm_tool_help_trigger` and routes to CHAT-u/LFM assist through advisor templates.
- both emit `status: "sandbox_candidate"` with `authority: "none"`, `commit_phase: "Ch'en_pending"`, and `learnable_aiml_candidates` so mutation proposals stay sandbox-only until Ch'en/Xul promotion.

Adaptive style profile controls are available through key-query text commands:
- `set language chinese` / `set language english`
- `set chat mode random` / `set chat mode neutral`
- `reset style profile`

On `key_query` misses, KHANARY now returns `status: "no_fact_found"` plus a
`research_fallback` plan (`delegate_track: "WEB-RESEARCH-µ"`, `retrieval_track: "Librarian"`) with query variants,
manifest-term expansions, source providers, and a phase watch stream for replay/evolve.
MISS responses also include `miss_deficit`, `runtime_attention`, and Ch'en classification/transition
surfaces so research routing is explicit (`MISS != WRONG`).
`runtime_attention` now emits the executable contract
`A(v)=R*U*D*P` with `R` cold-start fallback (lexical prior floor),
continuous `U` (`max(0.15,1-confidence)`), blended `P` (`0.8*route + 0.2*query-priority`),
admission threshold (`0.35`), deterministic tie-break (`score,D,R,node-id`), and
`resolver_selection` metadata (`S_mu(v)` route score surface) so WHY/WHO decisions are inspectable.
Resolver policy is split by authority: Yax computes eligibility (policy/capability filter), W-C-R ranks
only within that eligible set, and Sek revalidates `Selected && Capable && PolicySatisfied && XCFEAuthorized`
before dispatch. Empty eligibility now maps to `MISS_capability` (not WRONG).

Domain-specific AIML extensions are now declared through schema-bound ISA files:
`data\alice\math.aiml` + `data\schema\math-isa.json` define math-domain ops (e.g., `SOLVE`, `SIMPLIFY`)
as GraphPlan ISA extensions under the same `L_A` lowering path. Resolver execution backends are declared in
`data\schema\resolver.primitives.json`; math remains symbolic by default, with matrix-buffer primitives
used only when task shape requires them (linear systems/eigen/decomposition).

`memory:` / `@memory` are first-class key-query controls:
- `memory: <note>` appends a scratch-memory note (`<memory> @memory`) for fast retrieval.
- `@memory recall` (or `memory: list`) returns stored memory notes.

Key storage is hash-accelerated. Dictionary rows may include `hash` (`hash_hex`/`hash64` also accepted);
`record_key` also accepts optional `hash`, and lookups support explicit `hash:<hex>` addressing.
When exact/hash/word hits miss, key lookup falls back to a bag-of-words vector-space pass
(`tf-idf` weighted cosine over stored keys) before returning a miss.
Key-store state is persisted to `data/factory_keys.json` at shutdown and restored at startup.
Successful lookups now increment per-key hit counters and derive a reinforced key confidence score.
Factory vector tracking frames are persisted in the same state files under `vector_frames` (and binary format v2). `Factory::propose_vector_track()` computes a bounded spherical update to a micronaut's semantic key vector (`vector_meta`), requires monotonic frame ids, caps per-frame drift, and commits only through XulGate scope `micronaut-vector-tracking`. These frames update routing geometry/semantic gravity only; they do not confer semantic or factual authority.

AIML runtime state (`<set>/<get>` variables) is persisted to `data/aiml_state.json`.
Template expansion now supports `<tool name="...">...</tool>` dispatch for registered host tools
(`manifest_resolve`, `key_lookup`), plus `<star index="n"/>` and `<thatstar index="n"/>`.
AIML also has a graph-native surface (seeded from micronaut tracks/relations, optional GraphML overlay at
`data\alice\aiml.graphml`) with deterministic traversal tags:
`<node id="..."/>`, `<edges .../>`, `<getchildren .../>`, `<getedge .../>`, `<getroot/>`,
`<getlevel .../>`, `<getindex .../>`, `<node-attr .../>`, `<node-write ...>...</node-write>`, and
generic relation traversal:
`<traverse start="..." relation="@supports" direction="out|in|both" depth="2" limit="16" target="..." />`.
`TRAVERSE` is the operation; `relation` is an operand. Relation filters use a closed typed Relation ISA
(semantic/causal/epistemic/dependency/context/control/runtime) with compatibility aliases
(`@relate -> @related_to`, `@delegate/@call/@invoke -> @delegates_to`, etc.).
If a stronger relation is requested but only weak `@related_to` links exist, traversal emits `MISS_relation`
instead of silently upgrading association to causation/support/dependency.
Execution is single-pass template expansion with explicit tag resolution (no dynamic eval).

`fold_query` and `key_query.route_hint` now expose channelized mesh-routing scores with
`score_policy: "mesh_static_pattern_fold_phase_driver_confidence_health_distributional_lexical_nsm_function_wsd_compositional_coarse_ngram_tool_gram_vector_gate_prior"`:
`static_weight × pattern_strength × fold_weight × phase_driver_weight × confidence_score × health_score × relevance_score`
(`relevance_score` blends semantic scorer signal + `bow_score` lexical overlap +
`distributional_score` context-distribution similarity + `lexical_semantic_score`
lemma/synonym-normalized overlap + `nsm_prime_score` NSM semantic-prime overlap +
`function_word_score` function-word structural overlap + `word_sense_score`
contextual word-sense disambiguation overlap for polysemous terms +
`compositional_phrase_score` multiword compositional overlap +
`coarse_ngram_score` low-weight coarse n-gram structural prior +
`tool_gram_score` tool-invocation phrase overlap channel).
Route output also includes lexical-class diagnostics:
`content_word_ratio`, `function_word_ratio`, and `content_function_binding_ratio`.
Track JSON may optionally declare `static_weight` and `health` in `[0,1]`; both default to `1.0`.
Track files are now hard-gated as declarative (`"@no-eval": true`), and each `folds[]` entry may
declare an optional static `weight` in `[0,1]` for phase-typed fold preference.
Phase-driver contracts are declared separately in `data\drivers\phases\{pop,wo,yax,sek,chen,xul}.json`
and wired from `data\schema\fold.runtime.json#phase_driver_contracts`.

Popular intent keys (mapped to existing micronauts):

```text
SUMMARIZE: / @summarize   -> reasoner
EXPLAIN:   / @explain     -> reasoner
REWRITE:   / @rewrite     -> CREATE-µ
TRANSLATE: / @translate   -> CREATE-µ
BRAINSTORM:/ @brainstorm  -> CREATE-µ
PLAN:      / @plan        -> Coordinator / ORCH-µ
CODE:      / @code        -> coder
DEBUG:     / @debug       -> coder / INFER-µ
MATH:      / @math        -> MATH-µ / INFER-µ
EMAIL:     / @email       -> CREATE-µ
MEETING:   / @meeting     -> reasoner / Coordinator
SEARCH:    / @search      -> WEB-RESEARCH-µ -> Librarian
```

The AIML bootstrap surface in `data\alice\advisor.aiml` adds request-to-expert
patterns such as `ADVISE MATH ...`, `ADVISE CODE ...`, and `ADVISE RESEARCH ...`.
Their `math:`, `code:`, and `search:` prefixes feed the existing micronaut
inference and WPF card metadata path; `KHANARY.bat` rebuilds an existing bundle
when its mirrored `advisor.aiml` is missing.

Web-research command aliases are also recognized in `key_query`:
`search_for ...`, `search_the_web ...`, `google_topic ...`,
`recent_news [topic]`, and `business_near_me [type]`.

Mx2LM n-gram responses now include deterministic cross-reference surfaces:
- `text` (composed natural response used by WPF chat rendering)
- `intent_fragments` (detected concept/intents such as salutation + wellbeing_check)
- `action_grams` (promoted action bigrams/trigrams/phrases mapped to deterministic ops)
- `learned_facts` (new subject=>predicate facts learned from non-question clauses)
- `response_phase` (phase-selected response mode such as neutral, random-chat, learned-slang, locale-language)
- `governing_track` + `activation` (track authority + on-demand activation contract for utility/research/plan paths)
- `mx2lm_gram_container` (authoritative `mx2lm.gram_container.v1` with all unigram, bigram, trigram, quadgram, coarse, function, and tool gram channels under the `Mx2LM` namespace)
- `mx2lm_pattern_container` (authoritative `mx2lm.pattern_container.v1` for cross-micronaut pattern learning: source track, phase, surface, roles, gram-container link, card/outcome)
- `tensor_projection_container` (derived `tensor.projection_container.v1` emitted by `TENSOR-µ`; projection-only tensor-bundle geometry over the gram/pattern containers, with rank, metric, Jacobian, generation, and `semantic_authority=false` metadata)
- `vector_projection_container` (legacy/rank-1 `vector.projection_container.v1` emitted by `VECTOR-µ`; still valid as the vector provider under `TENSOR-µ`)
- `key_references` (key-word/ngram/meaning/control keys + manifest expansions)
- `ngram_references` (top n-grams, related n-grams, and definition links)

The legacy `bigrams`, `trigrams`, `quadgrams`, `coarse_grams`,
`function_grams`, `tool_grams`, and `gram_channels` fields remain compatibility
views. New consumers, including WPF cards, should read
`mx2lm_gram_container` first. A greeting such as `Hello, hows it going?` may
activate `GREETING-µ` as the social/card track while the submitted prompt grams
compile through Mx2LM underneath. Promoted micronauts behave like declared
employees/agents only after they have a track, sandbox namespace, capabilities,
and card/view contract; Mx2LM supplies the shared gram evidence and records the
observed employee/agent work pattern in `mx2lm_pattern_container`.

`TENSOR-µ` is the geometric sidekick for this path. It is a tensor-bundle
representation learner, not a second semantic authority: Mx2LM/FoldGraph/
RelationISA own typed semantic structure, while `TENSOR-µ` learns derived
type-(p,q) projection geometry over that structure. `VECTOR-µ` remains the
rank-1/vector provider and compatibility container inside the broader
TENSOR-µ contract. This matters because semantic similarity is only a subset of
semantic relatedness: `@contradicts`, `@part_of`, and `@causes` can be strongly
related without meaning “nearby under plain cosine.” The tensor container records
rank, metric tensor, Jacobian, projection generation, learner version, topology
version, and alignment metadata so coordinate drift is not mistaken for semantic
change. Runtime tensors/vectors are rebuildable from the semantic journal and
should be treated as derived projection/index data.

TENSOR-µ is allowed to model more than a flat embedding. The authoritative
FoldGraph/GML topology is `B`; TENSOR-µ may learn a derived semantic-manifold
projection `M_mu` over `B`, local charts `N(k)->T_k(M_mu)`, type-(p,q) tensor
fields, local metric tensors `g_ij`, Jacobian coordinate transitions, phase/
relation conditioned metrics `g(F,theta,r)`, and typed path-distance estimates
`d_M(A,B)`. These are retrieval and relevance geometries only:
`Delta(M_mu)` or `Delta(TensorField)` does not imply `Delta(B)`, and learned
curvature, rank, or coordinate drift cannot verify relations or commit topology.

The same boundary now carries a statistical decision layer. GML nodes may carry
`prior_mass` and `risk_bound`; edges may carry `loss_cost`, `dual_twin`, and
`mutation`. `vectors.mml` declares `Risk(theta,delta)` and `DeltaF_diff`, while
`fold.cd.xml` declares the OpenMath symbols (`PriorDistribution`, `Loss`,
`Risk`, `DecisionRule`, `BayesRisk`, `MinimaxRule`, `DeltaFieldDiff`, and
related delta symbols). A minimax violation emits `INADMISSIBLE_VECTOR_RULE`,
which rejects a projection rule before it can be treated as admissible geometry.
It still does not verify a relation or mutate `B`; Ch'en/XulGate remain the
semantic/topology commit path.

`vectors.xml` is the tensor/vector/mesh provider registry for that boundary. It
declares GML as the explicit graph-topology contract between key-books and
geometry, then lists the projection providers that `TENSOR-µ` may consume:
the checked GML
corpus (`khanary_topology`, `hypercube_Q3/Q4`, `ccc_n3/n4`, and `sample`),
projection/gyro ABI books, formal grammar/law sources (`gml.khanary.ebnf`,
`gml.validator.json`, `fold.cd.xml`, `vectors.mml`, and `fold_graph.xml`),
`field.xml` node vectors, the packaged `birdsong_mesh.stb` signal mesh, local
KLSL mesh-compression surfaces, `src/gpu/dml_ops.*` execution operators, and
optional external ASX mesh/HLSL shader provenance. `fold.cd.xml` names the Fold
Graph symbols; `vectors.mml` gives the MathML FoldPair equation
`FoldPair(S)=<M[Y,X],C[Y,X]^T>` and locks the invariant that co-occurrence mass
`M` is not the WCR static weight `W_Y`. This keeps the
order strict: key-books declare compressed semantics, GML preserves typed graph
shape, and mesh/STB/shader surfaces provide derived geometry for representation
learning.

`METRIC-µ` is the parent measurement abstraction for this layer. It learns
derived relevance and traversal geometry inside admitted track compartments,
not one global metric over all Mx2LM semantics. Its contract is declared in
`metric_graph.xml`, `data\metric_graph.gml`, `data\grammar\metric_graph.mml`,
and `data\manifest\metric.manifest.json`. Each track is treated as
`Track_tau=<G_tau,N_tau,R_tau>`: track-owned grams, local neighborhoods, and
track-local relations. Cross-track comparison requires an explicit
`Bridge_alpha_beta`; otherwise the candidate is not comparable. TENSOR-µ,
VECTOR-µ, SDR, graph paths, and SCX key-books are providers under METRIC-µ. Mesh nodes
permit discovery between compartments, not execution; WCR/Yax still consumes
the resulting `R`, and Ch'en/Xul remain the verification and topology commit
path.

The METRIC-µ training objective is now explicit in `metric_graph.xml` and
`data\grammar\metric_graph.mml`. Provider scoring is local and conditioned by
track, relation, phase, fold context, and provider:

```text
score_tau(Q,K|r,F,theta,provider)
  = -ProviderDistance(Q,K|tau,r,F,theta,provider) / T_tau
```

The contrastive term admits positives from GOOD FoldDelta evidence, admitted
RelationISA edges, or declared key-book neighborhoods, while negatives are
same-track hard negatives plus explicitly admitted bridge negatives:

```text
L_contrast =
  -log(exp(score_pos) / (exp(score_pos) + sum_{K- in Neg_tau} exp(score_neg)))
```

Cross-track loss is only defined through an admitted bridge:

```text
d_alpha_beta(Q,K) = d_alpha(Q,m) + L_m + d_beta(m,K)
```

Differential comparison and decision risk are separate penalties:

```text
L_diff = ||Projection(DeltaG) - DeltaF||^2
L_risk = max(0, Risk(theta,delta) - risk_bound)^2
L_total = alpha*L_contrast + beta*L_bridge + gamma*L_diff + eta*L_risk
```

That objective trains derived metric geometry and provider indexes only. It
does not promote a relation, verify GOOD/BAD, or commit FoldGraph topology.

The same TENSOR-µ contract now admits continuous-field provider geometry over
the discrete FoldGraph `B`. `data\grammar\gml.khanary.ebnf` accepts quantum
graph and flow-field attributes such as `boundary_type`, `vertex_pot`, `length`,
`potential`, `refractive`, `wave_number`, `frequency`, `vx/vy/vz`,
`divergence`, `singularity`, `text_boundary`, `boundary_token`,
`tensor_rank_p`, `tensor_rank_q`, `is_smooth_section`, `g_00/g_01/g_10/g_11`,
`jacobian_w`, `jacobian_determinant`, and `riemann_curvature_scalar`.
`data\grammar\fold.cd.xml` declares the corresponding OpenMath symbols:
`GraphHamiltonian`, `FluxConservation`, `PlaneWaveSuperposition`,
`ScatteringMatrix`, `GradientFlow`, `LieBracket`, `EulerCharacteristic`, and
`StructuralBoundaryReset`, plus `TensorField`, `MetricTensor`,
`JacobianMatrix`, `InverseJacobian`, and `TensorTransform`.
`data\grammar\vectors.mml` binds those symbols to the Hamiltonian, flux, wave,
scattering, flow, field-index, tensor-transform, and boundary-reset equations.
The shipped `data\quantum_field_boundaries.gml` and `data\tensor_field.gml`
assets are small conformance surfaces for this layer.

This does not turn n-grams into a flat book-length stream. Structural markers
such as `◄s►`, `◄/s►`, `◄par►`, and `◄art►` are declared field boundaries and
local sinks/sources. They reset local n-gram context unless an admitted boundary
transition exists, so sentence, paragraph, article, and section structure becomes
part of the topology that TENSOR-µ/METRIC-µ may project. It remains projection
geometry: Ch'en/Xul still decide verification and topology commit.

Differentiable curves add the bridge from flat n-gram streams to continuous
provider geometry. A token, sentence, paragraph, or article stream may be
projected as `gamma:I->X`; its tangent velocity `V(t)=d gamma(t)/dt` records
local semantic drift, and `DeltaF_curve=int_gamma V(s) dot ds` records the
derived displacement. `curvature` marks structural-scale inflection: a sentence
is a low-variance arc, a paragraph break is a controlled bend, and an article
boundary is a terminal sink/reset. These are TENSOR-µ/METRIC-µ provider features
over `B`, not semantic authority.

Flat Euclidean gradient fields are the initial fast path. In that provider,
`field_kind "euclidean_gradient"` treats `vx/vy/vz` as the partial derivatives
of `grad f`, `scalar_potential` stores the projection-side potential value, and
`potential_delta` records `f(gamma(b))-f(gamma(a))`. If `curl_x/curl_y/curl_z`
are declared, the validator requires them to be zero; this certifies the
conservative projection path and lets Factory choose the cheap flat-space route
before using Riemannian, quantum, or mesh providers.

Training gradients are now explicit, but they live one level lower than
FoldDelta. `TrainingGradient:g_phi=grad_phi L_total` tunes projection-provider
parameters `phi`, where `L_total` can be the METRIC-µ objective
`alpha*L_contrast + beta*L_bridge + gamma*L_diff + eta*L_risk`. The update law
is `phi_next = phi - eta*clip(g_phi,c)`, carried by `learning_rate`,
`gradient_clip`, `grad_norm`, `loss_value`, `provider_id`, and `parameter_id`
telemetry in GML. This can update TENSOR-µ/METRIC-µ provider geometry or indexes;
it is not FoldDelta, not Ch'en proof, and not Xul commit.

Entropy gradients add uncertainty geometry. `local_entropy` stores
`H(x)=-sum_w P(w|x)logP(w|x)`, `eh_x/eh_y/eh_z` store projected components of
`grad H`, and `entropy_spike` marks sudden structural uncertainty jumps. Optional
`entropy_bound` gates high-uncertainty paths; exceeding it emits
`ENTROPY_BOUND_EXCEEDED`. The free-energy provider combines relevance and
uncertainty as `V(x)=-grad f(x)+beta^-1*grad H(x)`, with
`diffusion_beta_inv` carrying `beta^-1`. This supports exploration, repair, and
hallucination-risk gating in projection space only.

Inscribed triangular mesh geometry is now carried by the entropic-well contract
as projection metadata. `inscribed_mesh_geometry` records `circumradius_R`,
`inradius_r`, `circumcenter_coord`, and `triangular_entropy_h_delta` for local
3-node simplicial loops. The MathML contract enforces Euler's triangle bound
`R_circum >= 2*r_in`, so the holographic circumcircle cannot collapse beneath
the incircle compression core. These dual-circle measurements can mask, rank, or
flag warped projection neighborhoods; they cannot verify relations or commit
FoldGraph topology.

The packaged `models/khanary-geometry-v0.3.0` model is the local geometry
provider behind the birdsong signal mesh. Its `compiler.py` projects the canary
WAV into spectrogram ridges, a Delaunay graph, CSR adjacency, geometric MoE
expert ids, and `.brain`/`.stb` tensor artifacts. `brain1` is the earlier
10,168-node / 30,491-edge compile run; `brain2` is the packaged 30,628-node /
91,863-edge run mirrored as `data/birdsong_mesh.stb`; `brain3` is the
current-system normalized proof run packed from the same verified tensor sidecar
contract with `tools/pack_geometry_brain_stb.py`. These files are registered as
provenance sidecars under the `birdsong-stb` provider and checked by
`tools/check_geometry_model.py`, including STB0 tensor payload alignment. They
remain projection evidence, not semantic authority.

Question-like misses (for example unresolved `what is ...?`) now fall through to MISS handling, which maps to `web_research` dispatch (`WEB-RESEARCH-µ` via Librarian) instead of returning mirrored fallback text.

You can generate the same shape offline with `ngram_builder.py`:

```powershell
python ngram_builder.py --text "Mx2LM: field dag cube status" --extract-payload --pretty
python ngram_builder.py --input data\alice\mx2lm.aiml --output data\mx2lm\mx2lm_checkpoint.json --pretty
python ngram_builder.py --text "Paris is capital of France" --binary-out data\mx2lm\mx2lm.ngram.bin --pretty
```

Builder output now also includes `mx2lm_gram_container`,
`mx2lm_pattern_container`,
`vector_projection_container`,
`coarse_bigrams`/`coarse_trigrams`,
`relation_templates` (e.g., `capital of -> is_capital_of`), `compositional_signatures`,
`word_sense_candidates`, and explicit `content_words`/`function_words` with
`content_function_profile` (ratios + interleaving/binding metrics), plus
`taxonomy_relations` and `semantic_folding_projection` (hyponym/hypernym/co-hyponym/synonym/meronym
graph cues + explicit sequence/negation/control-flow limitations).
`semantic_folding_projection` also marks boolean ops as fingerprint-set overlap
operations (similarity kernel behavior), not executable control logic.
`--binary-out` writes a snapshot-only coarse/lexical n-gram prior file
(`mx2lm.ngram.bin`) for faster runtime loading; semantic authority remains
in definitions/relations.

For larger corpus ingestion, declare external and repo-local sources in
`data\ingest\khanary_dataset_sources.json`, then compile derived evidence:

```powershell
python tools\build_dataset_ingestion.py --sources data\ingest\khanary_dataset_sources.json --max-rows 200 --top-k 512 --binary-top-k 50000
python tools\check_dataset_ingestion.py
```

The generated lane writes `data\mx2lm\dataset_gram_container.json`,
`data\mx2lm\dataset_pattern_container.json`,
`data\mx2lm\dataset_vector_projection.json`,
`data\mx2lm\dataset_ngram_summary.json`,
`data\mx2lm\dataset.ngram.bin`,
`data\grammar\dataset.key-grammar.candidates.jsonl`, and
`data\ingest\dataset_ingestion_manifest.json`. External joke/chat/reasoning
rows are candidate evidence only (`semantic_authority=false`); dictionary
key-grammar sources remain authority references. The JSON artifacts are review
capped by `--top-k`; the MX2NGRAM binary has its own `--binary-top-k` cap and
reports observed-versus-emitted section counts. The manifest reports
`lexical_units` because this is key-word/gram extraction, not model-token
tokenization. Use `--max-rows 0` only for a full streaming pass once the
bootstrap artifacts are verified.

For code, use the dedicated grammar lane (`code_gram.py`) so code can be kept as
atomic `<code>...</code>` blocks while still preserving structure:

```powershell
python tools\code_gram.py --text "<code lang='cpp'>#include <vector>\nint main(){return 0;}\n</code>" --extract-code --grammar-file cpp.code-grammar.json --pretty
python tools\code_gram.py --input src\main.cpp --grammar-file cpp.code-grammar.json --output data\mx2lm\main.code-graph.json --pretty
```

`tools/code_gram.py` preserves shape via `line`/`col`/`span` metadata and emits typed
`nodes`/`edges` (plus optional `fold_fingerprints`) in a separate code-analysis lane.

To generate `.code-grammar.json` from EBNF, use `ebnf_to_code_grammar.py`:

```powershell
python tools\ebnf_to_code_grammar.py --input data\grammar\cpp.code-grammar.ebnf --output data\grammar\cpp.code-grammar.generated.json --pretty
python tools\code_gram.py --input src\main.cpp --grammar-file data\grammar\cpp.code-grammar.generated.json --pretty
```

The EBNF generator supports inline directives such as:
`@language`, `@version`, `@token`, `@edge`, and `@auto_edges`.

To seed non-junk text corpora for training/eval, download curated datasets into `data\*.jsonl`:

```powershell
python tools\download_quality_text_datasets.py --data-dir data --overwrite
```

To ingest a focused Wikipedia concept page into the same JSONL corpus format:

```powershell
python tools\download_wikipedia_article_jsonl.py --url https://en.wikipedia.org/wiki/Philosophy_of_mind --out data\wikipedia_philosophy_of_mind.jsonl --overwrite
```

Theory-focused JSONL corpora can be mapped to semantic tracks in `data\tracks\*.semantic-tracks.v1.json`.
Current theory pack:
`COMP-THEORY-OF-MIND-µ`, `PHIL-AI-µ`, `THOUGHT-OUTLINE-µ`, `THEORY-OF-MIND-µ`,
`FREE-WILL-µ`, `PHIL-SELF-µ`, `ARTIFICIAL-CONSCIOUSNESS-µ`.
Track schema now supports structured compare/contrast and QA semantics via
`contrast_sets`, `has_a_hierarchy`, `truth_semantics`, and `qa_profile`
(`data\schema\micronaut-track.schema.json` + `data\schema\micronaut-track-types.schema.json`).

Unified tensor registry RPC is available for Phase 1 KXC execution contracts:
`tensor_register`, `tensor_info`, `tensor_list`, `tensor_stats`, and `tensor_gemm`.
`UnifiedTensor` carries shape/dtype/layout/device/lifecycle/backend/CSO identity above
`TensorBuffer` and `DMLOps`; tensor lifecycle labels are execution states, not K'UHUL semantic phases.
GEMM currently uses the existing DMLOps/D3D11 path when available and falls back to CPU with
`backend: "cpu-fallback"`.

Inline GPU kernel manifests can carry raw HLSL in CDATA, but validation must run before compilation:
`kernel_source` / `shader_source` CDATA requires `cso_id` (or `@id`) equal to `cso://SHA256(unwrapped_source)`.
The reusable check lives in `src/gpu/cso_integrity.cpp`; GML validator rule `H1` fails closed with
`CSO_INTEGRITY_MISMATCH`. This is an execution-integrity gate, not semantic authority.

Lorentz frame transitions now have matching contracts: `gml.validator.json` rules `LB1`-`LB3` validate finite
rows, bounded controls, and identity-when-disabled; `fold.cd.xml` declares `LorentzBoostTensor`; and
`GramGpuScorer::LorentzBoostRows::make_pure()` builds the host-side 4x4 rows for the existing
`LorentzTransitionTensor` shader cbuffer at register `b1` (not `b0`). During `infer`, the projection/prompt and
projection/completion gyro residuals install a bounded transition on `InferenceEngine`; prompt GramStore
candidates are then scored through `GramGpuScorer::score_batch()` with deterministic vector overrides. The
`semantic_attention` response/JROM block is projection-only evidence and does not change semantic authority.

`data\shaders\svg_compute.hlsl` is the SVG-3D tensor projection kernel:
it maps typed tensor channels (activation/confidence/relevance/pattern) to geometric buffers,
but does not own semantic meaning or control authorization.
Runtime path: startup `field.svg` generation and MCP `field_dag_svg` both use this shader when available,
with deterministic CPU projection fallback if GPU compile/dispatch is unavailable.
`data\shaders\attention.hlsl` is reserved for semantic attention execution (Yax backend):
it consumes semantic records (`W,C,R`, fold/phase/relation weights, legality gates) and emits
rank/admission masks (`S=W*C*R`, thresholded admission), not transformer QKV tensors.
`data\shaders\fold_compute.hlsl` remains the fold-neighborhood compute companion kernel.
Field DAG rendering now includes an ontology overlay (`Organization`, `Perspective`, `Function`,
`Information`, `Component`, `Environment`, `Communication`, `Connector`) with typed ontology edges
and track→ontology attachment links for reasoning-graph scaffolding.

Common tool-call keys (emit `status: "tool_call_plan_ready"` with `<tools>{tool_calls}</tools>`):

```text
STATUS:    / @status      -> world_state + sandbox_list + list_keys
CUBE:      / @cube        -> world_state + field_dag
WORLD:     / @world       -> world_state
FIELD:     / @field       -> field_dag
FIELDSVG:  / @fieldsvg    -> field_dag_svg
RELGRAPH:  / @relgraph    -> relational_graph
GRAPH:     / @graph       -> relational_graph + incidence_matrix (+ fold_query if payload provided)
SEMFOLD:   / @semfold     -> semantic_fold (sparse context fingerprints + set/similarity ops)
INCIDENCE: / @incidence   -> incidence_matrix
FOLD:      / @fold        -> fold_query
KEYQUERY:  / @keyquery    -> key_query
DISPATCH:  / @dispatch    -> micronaut_dispatch
MANIFEST:  / @manifest_query -> resolve_manifest
CONFIDENCE:/ @confidence  -> confidence_get
REPLAY:    / @replay      -> jrom_replay
SANDBOX:   / @sandbox     -> sandbox_list/query/violations
INFER:     / @infer       -> infer
LISTKEYS:  / @listkeys    -> list_keys
```

When `--gguf-model` points to a Gemma `.gguf`, KHANARY preloads that GGUF into
system RAM at startup (base-model sidecar) and `infer` responses emit `ngrams`,
`replay_candidate`, and `evolve_candidate` surfaces with matching JROM events
for replay/evolution loops.

When DDS artifacts are provided (`--dds-kson/--dds-manifest/--dds-shards`),
KHANARY preloads DDS tensors into GPU residency budget first, using the selected
adapter's dedicated VRAM as the ceiling policy.

Default launcher DDS paths target:
- `E:\models\GPT-DDS\GPT-OSS\gpt_oss_20b_folds.kson`
- `E:\models\GPT-DDS\GPT-OSS\folds\model_dds_manifest.json`
- `E:\models\GPT-DDS\GPT-OSS\folds\dds_shards`

Override with environment variables to point at a Gemma-converted DDS set.

`relational_graph` now annotates each `@node:*` with graph-local metrics:
`degree`, `local_clustering` (undirected), `local_clustering_directed`,
`triangles`, and `triples`.

`incidence_matrix` exposes node↔relation algebra with separated channels:
`B` (topology), `W_E` (edge pattern strengths), `BWB^T` (node-space projection),
and `B^TB` (relation-space overlap), with optional phase-fold hyperedges.
Relation columns now include `relation_type` (canonical token) and `relation_family`.

`semantic_fold` exposes per-node sparse context fingerprints plus
`AND/OR/XOR/SUB` set operations and `hamming/jaccard/cosine/overlap` similarity
between fold profiles.
Typed projections can be constrained by relation family/type:
`edge_type=default|all|...`, `relation_type=@supports`, `relation_family=epistemic`,
so `B_tau` can be sliced without changing the fold operation.
These boolean ops are similarity-kernel bitset operations, not executable
control-flow semantics; XCFE control remains explicit (`@if`, `@op`) and
vertical phase transitions can be policy-gated via `xcfe.mode`,
`xcfe.allow_vertical`, or `xcfe.block_vertical` context flags.

## Architecture

```
main
 ├── D3D11Device + DMLOps + HotSwapManager (1792 MiB ceiling)
 ├── ManifestRegistry (css/kuhul/words/micronaut manifests)
 ├── KuhulVM (PhaseEngine + FoldEngine + XCFEExecutor + GramStore)
 ├── KLSL compiler (Lexer → Parser → IR → HLSL/WGSL)
 ├── InferenceEngine (Tokenizer + MoERouter + KVCache + Sampler)
 ├── MicronautRegistry (TrackLoader + AIMLEngine + Factory + GravityField)
 ├── Memory (PerceptionStore + ConfidenceStore + SemanticGraph + MicronautScorer)
 └── Servers (MCPServer:8764 + JsonRpcServer:8787 + JROM + RequestRouter)
```

Manifest loading is data-only: KHANARY accepts regular `*.manifest.json` files and
silently skips missing standard manifests.

## C6 Phase Cycle

```
Pop(0) → Wo(π/3) → Yax(2π/3) → Sek(π) → Ch'en(4π/3) → Xul(5π/3)
arc_weight(from, to) = cos(θ_to - θ_from)
same=+1.0  adjacent=+0.5  two-hop=−0.5  opposing=−1.0
```

## Weight Format

Place `.scxqdds` shard files in `--weights-dir`. Format:
- Magic: `SDDS`
- Chunks: `name[64] + elem_count + format + data_size + crc32 + data`
- Formats: 0=F32, 1=Q8 (INT8 packed), 2=F16

## MCP Tools

`kuhul_forge`, `manifest_resolve`, `fold_execute`, `micronaut_run`,
`grammar_check`, `phase_query`, `gram_read`, `gram_write`,
`confidence_get`, `jrom_replay`, `sandbox_query`, `sandbox_list`,
`sandbox_violations`, `world_state`, `field_dag`, `field_dag_svg`,
`relational_graph`, `incidence_matrix`, `semantic_fold`

---

## Project Datasets (Curated)

Datasets identified as direct inputs for the DDS / SCXQDDS pipeline and micronaut routing paths.
Source: [Wikipedia — List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)

| Dataset | Scale | Path | Use |
|---|---|---|---|
| **T-REx** | 11M triples | `⟁EVIDENCE` → `semantic_graph_query` | Wikipedia ↔ Wikidata aligned triples; relation extraction for SemanticGraph |
| **GLUE** | ~1M sentence pairs, 9 tasks | `⟁INFER` → `accumulate_verdict` | Multi-task NLU benchmark; each subtask maps to a C6 phase |
| **SNLI** | 570K entailment pairs | `⟁INFER` → `apply_operators` | Entailment / contradiction / neutral labels feed INFER-µ verdict ops |
| **Clinc-150** | 23,700 utterances / 150 intents | `⟁EVIDENCE` → `intent_match` / `synonym_collapse` | Intent classification corpus; direct ELIZA pattern registry input |
| **Ubuntu Dialogue Corpus** | 930K dialogues / 7.1M utterances | `⟁FIELD` → `decompose_task` | Multi-turn response selection; ELIZA + dialogue routing training |
| **Penn Treebank** | ~1M parsed words | `⟁FIELD` → `load_patterns` | Structural grammar annotations; fold operator grammar rules |
| **Deepmind Mathematics** | QA pairs (arithmetic→calculus) | `⟁INFER` → REASON-µ | Formal reasoning QA for REASON-µ semantic graph path |
| **Common Voice** | 1,118+ hours (English) | DDS tokenizer | Crowdsourced speech corpus; tokenizer/acoustic layer training |
| **Web 1T 5-gram** | ~1T tokens | Mx2LM n-gram store | N-gram scoring and `ngram_references` surface in key_query |
| **PubMed Central** | 35M+ citations | `⟁EVIDENCE` → WEB-RESEARCH-µ | Biomedical fact corpus; Librarian source quality verification |
| **Contract Understanding Atticus (CUAD)** | ~13K labeled contracts | `⟁VERIFY` → `check_provenance_weights` | License / attribution / royalty policy QA for Librarian |
| **Natural Questions** | 300K real queries | `⟁COUNTEREXAMPLE` → `knowledge_gap` | Open-domain QA; knowledge-gap recovery path ground truth |
| **SNAP Twitter / ASU Twitter** | 85M+ connections | `⟁FIELD` → MEM-µ confidence | Social graph; confidence signal for large retrieval sets |

### Download priority

1. **T-REx** — loads directly into SemanticGraph as `(subject, verb, object)` triples
2. **Clinc-150** — feeds ELIZA pattern registry and `synonym_collapse` table
3. **SNLI** — trains INFER-µ `accumulate_verdict` operator weights
4. **GLUE** — end-to-end NLU benchmark across all six C6 phases

---

## Reference Dataset Catalogue

Full catalogue from [Wikipedia — List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research).

### Reviews
| Dataset | Scale | Tasks |
|---|---|---|
| Netflix Prize | 100,480,507 ratings · 480,189 users · 17,770 movies | Rating prediction |
| Amazon Reviews | 233.1M product reviews | Classification, sentiment |
| OpinRank Review Dataset | 42,230 car + ~259,000 hotel reviews | Sentiment, clustering |
| MovieLens | ~22M ratings · 33,000 movies | Regression, clustering, classification |
| Yahoo! Music User Ratings | ~10M artist ratings | Clustering, regression |
| Car Evaluation Data Set | 1,728 instances | Classification |
| YouTube Comedy Slam Preference | 1,138,562 video pair votes | Classification |
| Skytrax User Reviews | 41,396 airline/airport reviews | Classification, regression |
| Teaching Assistant Evaluation | 151 instances | Classification |
| UIT-VSFC (Vietnamese Feedback) | 16,000 comments | Classification |
| UIT-VSMEC (Vietnamese Emotion) | 6,927 Facebook comments | Classification |
| ViOCD (Vietnamese Complaint) | 5,485 reviews | Classification |
| ViHOS: Hate Speech Spans | 26K spans · 11K comments | Span detection |

### News Articles
| Dataset | Scale | Tasks |
|---|---|---|
| NYSK Dataset | 10,421 English articles | Sentiment, topic extraction |
| Reuters Corpus Volume 1 | 810,000 news stories | Classification, clustering, summarization |
| Reuters Corpus Volume 2 | 487,000 multilingual stories | Classification, clustering, summarization |
| Thomson Reuters Text Research Collection | 1,800,370 stories | Classification, clustering, summarization |
| Saudi Newspapers Corpus | 31,030 Arabic articles | Summarization, clustering |
| RE3D | News + government entity/relation data | Classification, entity/relation recognition |
| Examiner Spam Clickbait Catalogue | 3,089,781 headlines | Clustering, events, sentiment |
| ABC Australia News Corpus | 1,186,018 headlines (2003–2019) | Clustering, events, sentiment |
| Worldwide News (20K Feeds) | 1,398,431 headlines | Clustering, events, language detection |
| Reuters News Wire Headline | 16,121,310 timestamped events (11 years) | NLP, computational linguistics |
| The Irish Times News Corpus | 1,484,340 articles (1996–2019) | NLP, computational linguistics |
| News Headlines for Sarcasm Detection | 26,709 headlines | NLP, classification |

### Messages
| Dataset | Scale | Tasks |
|---|---|---|
| Enron Corpus | ~500,000 emails | Network analysis, sentiment |
| Ling-Spam Dataset | 2,412 ham + 481 spam emails | Classification |
| SMS Spam Collection | 5,574 SMS messages | Classification |
| Twenty Newsgroups | 20,000 messages | NLP |
| Spambase Dataset | 4,601 spam emails | Spam detection, classification |

### Twitter / Social Media
| Dataset | Scale | Tasks |
|---|---|---|
| MovieTweetings | ~710,000 movie rating tweets | Classification, regression |
| Twitter100k | 100,000 image-tweet pairs | Cross-media retrieval |
| Sentiment140 | 1,578,627 tweets with emoticon labels | Sentiment analysis |
| ASU Twitter Dataset | 11,316,811 users · 85,331,846 connections | Clustering, graph analysis |
| SNAP Social Circles: Twitter | 1,768,149 Twitter network records | Clustering, graph analysis |
| Twitter Arabic Sentiment | 2,000 Arabic tweets | Classification |
| Buzz in Social Media | 140,000 records (Twitter + Tom's Hardware) | Regression, classification |
| PIT (Paraphrase/Semantic Similarity) | 18,762 tweet pairs | Regression, classification |
| Geoparse Twitter Benchmark | 6,386 location-annotated tweets | Classification, information extraction |
| SPIRS (Sarcasm Perceived/Intended) | 30,000 sarcastic/non-sarcastic tweets | Classification |
| Dutch Social Media COVID-19 | 271,342 tweets | Sentiment, multi-label classification |
| ReactionGIF Dataset | 30,000 tweets with GIF reactions | Sentiment, emotion classification |

### Dialogues
| Dataset | Scale | Tasks |
|---|---|---|
| NPS Chat Corpus | ~500,000 chat posts | NLP, linguistics |
| Twitter Triple Corpus | 4,232 A-B-A triples | NLP |
| UseNet Corpus | 7 billion forum postings | General NLP |
| NUS SMS Corpus | ~10,000 SMS messages | NLP |
| Ubuntu Dialogue Corpus | 930,000 dialogues · 7.1M utterances | Dialogue systems |
| Dialog State Tracking Challenge (DSTC2&3) | 3,200–2,300 spoken dialogues | Dialogue state tracking |
| Clinc-150 | 23,700 single-turn utterances · 150 intents | Intent classification |

### Legal
| Dataset | Scale | Tasks |
|---|---|---|
| FreeLaw | 4,940,710 court documents | NLP, linguistics |
| Pile of Law | ~50,000,000 legal/administrative records | NLP, linguistics, sentiment |
| Caselaw Access Project | ~10,000 U.S. case law volumes | NLP, linguistics |

### Text — General / NLP
| Dataset | Scale | Tasks |
|---|---|---|
| Hansard French-English | 2,869,040 sentence pairs | Translation |
| Web of Science Dataset | 46,985 hierarchical documents | Classification, categorization |
| Legal Case Reports | 4,000 Federal Court of Australia cases | Summarization, citation analysis |
| Blogger Authorship Corpus | 681,288 entries · 19,320 people | Sentiment, summarization, classification |
| Dataset for Machine Comprehension of Text | 660 stories with questions | Machine comprehension |
| Penn Treebank Project | ~1M parsed words | NLP, summarization |
| Web 1T 5-gram | ~1T tokens | Unsupervised learning |
| DEXTER Dataset | 2,600 corporate acquisition articles | Classification |
| Google Books N-grams | 2.2 TB text | Classification, clustering, regression |
| Personae Corpus | 145 Dutch essays | Classification, regression |
| PushShift | ~100,000,000 Reddit/Twitter/HN posts | NLP, sentiment, linguistics |
| SEC Filings EDGAR | Company filing documents | NLP |
| CNAE-9 | 1,080 Brazilian company descriptions | Classification |
| Sentiment Labeled Sentences | 3,000 hand-labeled sentences | Classification, sentiment |
| BlogFeedback | 60,021 blog posts | Regression |
| PubMed Central | 35M+ biomedical citations | NLP |
| USPTO Patents | U.S. patent documents | NLP |
| PhilPapers | Open philosophy publications | NLP |
| Book Corpus | Large-scale text corpus | NLP |
| Stanford NLI (SNLI) | 570,000 image caption entailment pairs | Natural language inference |
| DSL Corpus Collection | 294,000 similar-language phrases | Language discrimination |
| Urban Dictionary | 2,580,925 words + definitions | NLP, machine comprehension |
| T-REx | 11M Wikipedia–Wikidata triples | NLP, relation extraction |
| GLUE | ~1M sentences/pairs · 9 tasks | NLU benchmark |
| Contract Understanding Atticus (CUAD) | ~13,000 labeled contracts | NLP, QA |
| UIT-ViIC (Vietnamese Image Captioning) | 19,250 captions · 3,850 images | NLP, computer vision |
| UIT-ViNames (Vietnamese Names) | 26,850 annotated names | NLP |
| UIT-ViCTSD (Vietnamese Toxic Speech) | 10,000 comments · 10 domains | NLP |
| PG-19 | Project Gutenberg books | NLP |
| Deepmind Mathematics | Math QA pairs | NLP, reasoning |
| Anna's Archive | 100,356,641 published works | NLP |

### Speech
| Dataset | Scale | Tasks |
|---|---|---|
| Switchboard-1 | 260 hours · 543 speakers | Speech recognition, phonetic transcription |
| Hub5'00 | 260 hours conversational speech | Speech recognition |
| Zero Resource Speech Challenge 2015 | English 5h/12sp · Xitsonga 2.5h/24sp | Unsupervised feature discovery |
| Parkinson Speech Dataset | 1,040 recordings | Classification, regression |
| Spoken Arabic Digits | 8,800 digit recordings | Classification |
| ISOLET | 7,797 spoken letters | Classification |
| Japanese Vowels | 640 vowel sequences | Classification |
| Parkinson's Telemonitoring | 5,875 recordings | Classification |
| TIMIT | 6,300 sentences · 630 speakers | Speech recognition, classification |
| Arabic Speech Corpus | ~1,900 MSA recordings | Speech synthesis, recognition |
| Common Voice | 1,118+ hours (English) | Speech recognition |
| LJSpeech | 13,100 audiobook clips | Speech synthesis |
| Arabic Speech Commands | 12,000 WAV files · 40 keywords | Speech recognition, keyword spotting |

### Music
| Dataset | Scale | Tasks |
|---|---|---|
| Geographic Origin of Music | 1,059 audio feature samples | Geographic classification, clustering |
| Million Song Dataset | 1M songs with audio features | Classification, clustering |
| MUSDB18 | 150 multi-track recordings | Source separation |
| Free Music Archive | 106,574 CC-licensed songs | Classification, recommendation |
| Bach Choral Harmony | 5,665 Bach chorale chords | Classification |

### Other Sounds
| Dataset | Scale | Tasks |
|---|---|---|
| UrbanSound | 1,059 labeled environmental recordings | Classification |
| AudioSet | 2,084,320 × 10-second YouTube clips | Classification |
| Bird Audio Detection Challenge | 17,000+ environmental clips | Classification |
| WSJ0 Hipster Ambient Mixtures | 28,000 noise-mixed samples | Audio source separation |
| Clotho | 24,905 captions · 4,981 audio samples | Automated audio captioning |
| Pro Sound Effects | 1,272,241 effects · 672 categories | Classification, retrieval |

### Electrical Signals
| Dataset | Scale | Tasks |
|---|---|---|
| Witty Worm Dataset | 55,909 infected IP addresses | Classification |
| Cuff-Less Blood Pressure | 12,000 vital signal records | Classification, regression |
| Gas Sensor Array Drift | 13,910 sensor measurements | Classification |
| Servo Dataset | 167 servo-amplifier records | Regression |
| UJIIndoorLoc-Mag | 40,000 magnetic field location records | Classification, regression, clustering |
| Sensorless Drive Diagnosis | 58,508 motor electrical signals | Classification |

### Motion-Tracking
| Dataset | Scale | Tasks |
|---|---|---|
| PUC-Rio Body Postures | 165,632 activity records | Classification |
| Gesture Phase Segmentation | 9,900 gesture video features | Classification, clustering |
| Vicon Physical Action | 3,000 tracked physical actions | Classification |
| Daily and Sports Activities | 9,120 motor sensor records | Classification |
| Human Activity Recognition (Smartphones) | 10,299 accelerometer/gyroscope records | Classification |
| Australian Sign Language Signs | 2,565 sign language recordings | Classification |
| Weight Lifting with IMUs | 39,242 IMU measurements | Classification |
| sEMG Basic Hand Movements | 3,000 electromyographic signals | Classification |
| REALDISP Activity Recognition | 1,419 activity records | Classification |
| Heterogeneity Activity Recognition | 43,930,257 multi-device records | Classification, clustering |
| Indoor User Movement from RSS | 13,197 wireless network records | Classification |
| PAMAP2 Physical Activity Monitoring | 3,850,505 records · 9 subjects | Classification |
| OPPORTUNITY Activity Recognition | 2,551 wearable/ambient sensor records | Classification |
| Real World Activity Recognition | 3,150,000 per-sensor records | Classification |
| Toronto Rehab Stroke Pose | 3,500–6,000 frames · 19 subjects | Classification |
| Corpus of Social Touch (CoST) | 7,805 pressure-sensor gesture captures | Classification |

### Chemistry / Physics
| Dataset | Scale | Tasks |
|---|---|---|
| OpenReACT-CHON-EFH | 35,087 reaction geometries + 62,527 off-equilibrium | ML interatomic potentials |
| HIGGS Dataset | 11M Monte Carlo collision simulations | Classification |
| HEPMASS Dataset | 10,500,000 particle accelerator records | Classification |

### Engineering / Systems
| Dataset | Scale | Tasks |
|---|---|---|
| Yacht Hydrodynamics | 308 yacht performance records | Regression |
| Robot Execution Failures | 463 robotic failure records | Classification |
| Pittsburgh Bridges | 108 bridge design descriptions | Classification |
| Automobile Dataset | 205 vehicle insurance records | Regression |
| Auto MPG | 398 car fuel efficiency records | Regression |
| Energy Efficiency | 768 building efficiency records | Classification, regression |
| Airfoil Self-Noise | 1,503 aerodynamic/acoustic tests | Regression |
| Challenger O-Ring | 23 launch records | Regression |
| Statlog (Shuttle) | 58,000 NASA space shuttle records | Classification |

### Astronomy / Earth Science
| Dataset | Scale | Tasks |
|---|---|---|
| Volcanoes on Venus – JARtool | Venus images from Magellan | Classification |
| MAGIC Gamma Telescope | 19,020 simulated gamma events | Classification |
| Solar Flare Dataset | 1,389 solar event measurements | Regression, classification |
| CAMELS Multifield Dataset | 405,000 2D maps + 3D grids | Regression |
| Volcanoes of the World | 1,535 volcanic eruption records | Regression, classification |
| Seismic-Bumps | 2,584 coal mine seismic records | Classification |
| CAMELS-US / Chile / Brazil / GB / Australia | 222–897 catchment records each | Regression |
| LamaH-CE | 859 catchment records | Regression |

### Biological / Medical
| Dataset | Scale | Tasks |
|---|---|---|
| Age Dataset | 1,223,009 distinguished people records | Regression, classification |
| Synthetic Fundus Dataset | 2,500 retinal images | Classification, segmentation |
| EEG Database | 122 alcoholism study records | Classification |
| P300 Interface Dataset | 1,224 brain-computer interface records | Classification |
| Heart Disease Data Set | 303 patient records | Classification |
| Breast Cancer Wisconsin | 569 breast mass records | Classification |
| National Survey on Drug Use and Health | 55,268 health survey records | Classification, regression |

### Other Physical / Chemistry
| Dataset | Scale | Tasks |
|---|---|---|
| Wine Dataset | 178 Italian wine analyses | Classification, regression |
| Combined Cycle Power Plant | 9,568 sensor records | Regression |
| Concrete Compressive Strength | 1,030 records | Regression |
| Concrete Slump Test | 103 records | Regression |
| Musk Dataset | 6,598 molecular structures | Classification |
| Steel Plates Faults | 1,941 samples | Classification |
| Noble Metal Monometallic Nanoparticles | 425–4,000 samples | Regression |
| Noble Metal Bimetallic Nanoparticles | 138,147–162,770 samples | Regression |
| AuPdPt Trimetallic Nanoparticles | 48,136 samples | Regression |
