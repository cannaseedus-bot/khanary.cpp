# KHANARY Public API Gateway

The public API gateway is the reachable transport surface for KHANARY runtime
requests. The production origin is:

```text
https://api.kuhul.dev:443
```

The local development gateway remains:

```text
http://127.0.0.1:17892
```

The public edge maps HTTPS port `443` into the local/user-scoped runtime path.
It is an API projection over KHANARY, KQL, MeshNet peers, and the local vector
engine. It is not semantic authority and it is not the commit layer.

## Request path

```text
api.kuhul.dev:443
  -> session/auth
  -> user_scoped_mesh_kql
  -> khanary_runtime
  -> jrom_evidence
  -> Ch'en/XCFE/Xul
```

The gateway can receive requests from a signed-in user session or a user-owned
mesh control plane. KQL tables are user scoped. Static API model manifests
contributed through the mesh are provider/routing evidence; they do not prove a
model is correct, safe, reachable, or semantically authoritative.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | service metadata |
| `GET` | `/v1` | service metadata |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | text completions through KHANARY `key_query` plus phase-gated contributors |
| `POST` | `/v1/code/modify` | code-modification proposal envelope |
| `POST` | `/v1/images/generations` | image-generation request envelope |
| `POST` | `/v1/learn/events` | learning-event intake; returns `learning_commenced` |
| `POST` | `/rpc` | public JSON-RPC bridge for allowlisted runtime methods |

## Authority boundaries

The gateway is intentionally narrow:

- A completion is a response projection, not semantic authority.
- A code-modification response is a proposal envelope; it does not apply a
  patch, commit a change, or authorize deployment.
- An image-generation response is media/request evidence, not semantic truth.
- A learning event means `learning_commenced`; it is evidence for the journal,
  not a topology mutation.
- Raw RPC is allowlisted. Unknown methods are rejected at the gateway.
- All endpoint envelopes that can affect learning or routing carry
  `semantic_authority:false`, `topology_mutation:false`,
  `commit_authorized:false`, and `next_gate:"Ch'en/XCFE/Xul"`.

Ch'en verifies, XCFE authorizes, and Xul commits. The public API can collect
evidence and route requests toward those gates; it cannot bypass them.

## Local implementation

`tools/khanary_completion_api.py` implements the local gateway. It exposes the
same path vocabulary as the public contract and advertises
`public_base_url=https://api.kuhul.dev` with `public_port=443`. Production may
run it behind a reverse proxy, service mesh, or Apps Script/user-session mesh
orchestrator, as long as the public contract remains `https://api.kuhul.dev:443`
and the authority fields remain false until the Ch'en/XCFE/Xul path authorizes a
separate commit.

`KHANARY.bat` passes the public edge metadata into the gateway with
`--public-base-url` and `--public-port`, defaulting to `https://api.kuhul.dev`
and `443`. It also writes the same `public_api` projection into
`data/ports.json` so abstract-port consumers can see that `api` is the public
edge over the local `completion` surface, not a separate semantic authority.

The machine-readable contract is
`data/grammar/api-gateway.contracts.json`, conformance fixtures are
`data/conformance/api_gateway.json`, and validation is performed by
`tools/check_api_gateway.py`.
