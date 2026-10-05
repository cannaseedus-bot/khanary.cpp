# KHANARY Apps host layer

`src/host` is reserved for the Apps host lane. It is where app-facing shells,
WebView/WebUI hosts, JSON/JROM app routes, user-session bridges, and installable
application surfaces should live.

Atomic DOM (`atomic_dom`) is the Apps widget substrate. Its purpose is to
remove redundancy. It declares ASX Blocks
through the Atomic CSS key grammar in `data/manifest/css.manifest.json`. That
manifest anchors `⟁KEYGRAMMAR`, `⟁SYNTAXUNITKINDS`, `Block`, `Component`,
`⟁WIDGET`, `⟁WIDGETCARD`, and `⟁HTMLBIND`, so Apps widgets are constrained
key-grammar projections rather than ad-hoc DOM fragments. Its job is to remove
redundancy while keeping user-specific options easy to change: branding,
runtime config, `api_token_secret_ref` slots, color switcher controls, and
palette defaults chosen from provided palettes so users do not have to hand-pick
colors. API tokens are represented as secret references, not stored values.

`AtomicDOM.js` is the Apps-side callable helper for that surface. It is callable
like a small React-style creator (`h`, `block`, `widget`, `theme`) and an
AJAX-style request-envelope helper (`request`). Those calls return Atomic DOM
records and key grammar tokens; they do not execute K'UHUL or commit state.
TypeScript-µ, K'UHUL-µ playgrounds, install buttons, status panels, and future
app widgets should render through Atomic DOM records such as `atomic_dom_widgets`
before a native/WebView shell projects them to HTML or platform UI. Atomic DOM
is a UI projection substrate; it is not semantic authority and it is not the MCP
protocol.

MCP stays separate. MCP is the model/tool protocol lane, backed by
`src/io/mcp_server.cpp`, and should not become the Apps runtime or own app user
sessions.

```text
Apps host lane                 MCP lane
──────────────────────────     ──────────────────────────
src/host                       src/io/mcp_server.cpp
human/app sessions             model/tool protocol calls
UI routes and app state         tool/resource invocation
Atomic DOM widgets             no widget/session ownership
KQL/GAS/user cloud bridges      no app identity ownership
JSON/JROM app envelopes         no topology commit authority
```

The boundary is:

```text
Apps host → JSON-RPC/completion/KQL request → Ch'en/XCFE/Xul
MCP       → tool/resource request            → Ch'en/XCFE/Xul
```

Both lanes can request runtime work. Neither lane is semantic authority by
itself.

## Apps host responsibilities

The Apps host may:

- serve app-facing HTML/WebView/native shells;
- render user-facing widgets through Atomic DOM and `AtomicDOM.js`;
- expose easy user option slots for branding, runtime config, palette choice,
  color switcher state, and `api_token_secret_ref`;
- keep UI state, app route state, and `app_user_sessions`;
- broker user-scoped cloud/application data such as GAS, Drive, Sheets, or KQL
  tables after the user signs in, represented by `user_scoped_kql_tables` and
  `gas_drive_sheet_bridges`;
- emit JSON/JROM application envelopes;
- call JSON-RPC, the completion gateway, or KQL as downstream runtime surfaces.

The Apps host must not:

- expose raw MCP tool-call semantics as its own app API;
- treat Atomic DOM widget state as semantic authority;
- claim `semantic_authority:true`;
- mutate persistent topology directly;
- commit learned state without Ch'en/XCFE/Xul authorization;
- introduce a new executable output convention outside `bin/`.

## MCP responsibilities

The MCP lane may:

- expose model-facing tools and resources;
- provide narrow, typed invocation surfaces;
- bridge to runtime allowlisted calls.

The MCP lane must not:

- own app login/session state;
- store user app data tables;
- act as the public app API;
- replace the Apps host lane.

## Build/output rule

If an Apps host target is revived from `src/host`, it must be optional and emit
to the same canonical binary directory as the rest of KHANARY:

```text
bin/
```

Examples:

```text
KHANARY_APP_HOST.exe
KHANARY_COCKPIT_HOST.exe
```

Do not create a new `_build`, `.build-*`, nested `build/*/bin`, or ad-hoc
executable root for Apps.

## Current source status

The local `src/host` source snapshot contains useful app-host pieces:

- JROM routes in `api.cpp/.hpp`;
- app-style HTTP routes in `server.cpp/.hpp`;
- JSON runtime C ABI pieces in `json_runtime_api.cpp/.hpp`;
- a WebView2 cockpit prototype in `cockpit_host.cpp`.

Those files are not part of the default build target. Before promoting any of
them, remove user-local hard-coded paths, replace stale dependencies with
repo-local contracts, and add an explicit optional CMake target that emits to
`bin/`.
