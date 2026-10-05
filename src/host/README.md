# KHANARY Apps host lane

This directory is reserved for Apps host code.

Apps host code is separate from MCP:

- Apps host: user sessions, UI routes, app state, JSON/JROM app envelopes, and
  KQL/GAS/user-cloud bridges.
- Atomic DOM (`atomic_dom`): the widget substrate for Apps surfaces. It declares
  ASX Blocks through the Atomic CSS key grammar at
  `data/manifest/css.manifest.json`, including `atomic_dom_widgets`,
  TypeScript-µ playground widgets, and K'UHUL-µ playground widgets.
- MCP: model/tool protocol calls through `src/io/mcp_server.cpp`.

Apps may call JSON-RPC, the completion gateway, or KQL as downstream runtime
surfaces. Apps must not expose raw MCP tool-call semantics as their own app API,
claim semantic authority, treat Atomic DOM widgets as semantic authority, or
commit topology without Ch'en/XCFE/Xul.

If an app host executable is added, make it an explicit optional target and emit
it to `bin/`, for example:

```text
bin/KHANARY_APP_HOST.exe
```

Do not add this directory to the default `KHANARY.exe` target without first
cleaning stale local paths and dependencies.
