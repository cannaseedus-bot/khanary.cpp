# KHANARY Query Language (KQL) — User-Scoped Mesh Database Contract

KQL is the local query layer for KHANARY's user-dependent mesh database. It is intentionally SQLite-like in purpose, but it does not replace the existing vector engine. The vector engine remains the retrieval/scoring/index substrate: `vector_projection`, GramStore, factory keys, STBMap, hyperbolic event vectors, and `GramGpuScorer` provide the fast candidate set. KQL gives those substrates a user-scoped table/query surface.

A later backend can map the same table contract to SQLite, but the first backend is KHANARY-native: append-only JSONL/JROM rows plus deterministic vector/gram indexes.

## Purpose

The Google Apps Script MeshNet deployment creates a user-scoped online control plane. KQL localizes that shape inside `KHANARY.CPP`:

```text
Google user session -> user mesh id -> local user mesh tables -> KQL query surface
```

The local database is a memory/event database with vector indexes, not a standalone fact database. Rows can be observations, prompts, responses, plans, hyperbolic event projections, peer registrations, or committed facts. Retrieval uses the existing vector/gram machinery first, then KQL filters rows by user, table, phase, authority, interval class, and commit state. Only Ch'en/Xul-verified rows may enter fact/topology tables.

## KQL -> GAS deployment pattern

KQL plus GAS avoids the classic hosting-surface trap. KHANARY does not need to run a multi-tenant app server to give users an online mesh. The public surface can remain static:

```text
static landing page -> install/connect button -> Google sign-in -> GAS control plane -> KQL user scope
```

The landing page is only an entry point. It can link to the deployed Apps Script web app, show the endpoint, and provide model/peer manifests to copy or download. The signed-in user session and the script's configured permissions decide which Drive/Sheets resources are visible. KHANARY treats that Google identity as provenance and routing identity, then maps it into a local `user_mesh_id`.

This means the hosting boundary is:

```text
static page = distribution surface
GAS = user-session orchestrator
KQL = local user-scoped query/memory layer
vector engine = candidate retrieval/scoring layer
Ch'en/Xul = authority boundary
```

Browser CORS is not required for the core mesh path. Browser users can open or install the GAS web app; KHANARY peers should use server-to-server calls, local clients, or an explicit proxy when they need JSON RPC against the GAS endpoint.

## Identity boundary

A user table is keyed by a local pseudonymous user mesh id:

```text
user_mesh_id = sha256("khanary:user:" + provider + ":" + provider_user_key + ":" + local_salt)
```

For GAS sessions, `provider_user_key` is normally `Session.getTemporaryActiveUserKey()`. Email can be stored only when explicitly enabled by policy, and even then it is metadata, not authority.

```text
Google session identity = provenance/routing identity
Google session identity != verified semantic fact
```

## Initial table contract

| table | key | purpose | authority |
| --- | --- | --- | --- |
| `kql_users` | `user_mesh_id` | local user/session identity envelope | provenance only |
| `kql_user_sessions` | `session_id` | online/offline session bindings, GAS peer identity, boot id, code hash | provenance only |
| `kql_memory_events` | `memory:event:<kind>:<epoch>-<seq>` | prompt/model_response/plan rows | event evidence |
| `kql_hyperbolic_events` | `event_key` | Minkowski/future-sheet key-grammar projections | routing evidence |
| `kql_mesh_peers` | `peer_id` | active peer metadata and queues | routing/control |
| `kql_static_api_models` | `model_id` | user/peer contributed static API model manifests advertised through the mesh | provider/routing evidence |
| `kql_gram_index` | `gram_scope_key` | unigram/bigram/trigram/coarse/tool gram indexes backed by GramStore/factory keys | retrieval evidence |
| `kql_vector_index` | `vector_key` | Q/K/V, tensor slot, STBMap, and hyperbolic four-vector projections from the existing vector engine | retrieval/scoring evidence |
| `kql_fact_commits` | `fact_key` | Ch'en verified + Xul committed facts | semantic authority |
| `kql_topology_commits` | `topology_key` | Xul-authorized graph mutations | topology authority |

Every row that is not in `kql_fact_commits` or `kql_topology_commits` must carry one of:

```text
fact_authority=false
semantic_authority=false
commit_state=journaled_not_fact | candidate | Ch'en_pending
```

## Query shape

KQL should stay narrow and deterministic. Suggested first grammar:

```ebnf
query       = select_query | insert_event | link_session ;
select_query = "SELECT", field_list, "FROM", table_name,
               [ "WHERE", predicate ], [ "LIMIT", integer ] ;
insert_event = "INSERT", "EVENT", json_object, "INTO", table_name ;
link_session = "LINK", "SESSION", provider, provider_user_key, "AS", user_mesh_id ;
```

Example:

```sql
SELECT key, event_kind, phase
FROM kql_memory_events
WHERE user_mesh_id = :user AND event_kind = 'prompt'
LIMIT 50
```

Vector-backed query flow:

```text
query text -> gram channels / vector_projection / hyperbolic projection
          -> vector engine candidate keys
          -> KQL user/table/authority filters
          -> Ch'en/Xul decision boundary
```

Example hyperbolic routing read:

```sql
SELECT event_key, interval_class, rapidity_rho
FROM kql_hyperbolic_events
WHERE user_mesh_id = :user AND interval_class = 'timelike_candidate'
LIMIT 32
```

Example static API model pool read:

```sql
SELECT model_id, provider, capability, endpoint_host
FROM kql_static_api_models
WHERE user_mesh_id = :user AND capability = 'text.generate'
LIMIT 25
```

Static API model rows are provider candidates. They may describe an endpoint shape, supported operations, auth requirements, rate hints, tags, and vector keys. They do not prove that a model is safe, reachable, correct, or semantically authoritative.

Mesh contribution message:

```json
{
  "type": "static_api_model.contribute.v1",
  "to": "broadcast",
  "data": {
    "@type": "khanary.mesh.static_api_model.v1",
    "model_id": "static-api:example:echo-v1",
    "semantic_authority": false,
    "fact_authority": false,
    "commit_state": "candidate"
  }
}
```

## Runtime files

Native surfaces:

| file | role |
| --- | --- |
| `src/io/kql.h` | AST, table names, row envelope, query option/result structs |
| `src/io/kql.cpp` | first SELECT parser/evaluator over in-memory rows supplied by JSONL/JROM/index readers |
| `src/tests/test_kql.cpp` | user isolation, authority-boundary, hyperbolic-event lookup, and vector-candidate filtering tests |
| `tools/meshnet_gas_landing.html` | static install/connect page for the GAS mesh endpoint |
| `tools/meshnet_static_api_model.example.json` | example manifest for contributing a static API model through the mesh |

Planned storage surface:

| file | role |
| --- | --- |
| `src/io/kql_store.h` | append/read/index abstraction over JSONL/JROM plus vector/gram index handles |

Proposed data layout:

```text
data/users/<user_mesh_id>/
  kql_users.jsonl
  kql_user_sessions.jsonl
  kql_memory_events.jsonl
  kql_hyperbolic_events.jsonl
  kql_mesh_peers.jsonl
  kql_static_api_models.jsonl
  kql_gram_index.jsonl
  kql_fact_commits.jsonl
  kql_topology_commits.jsonl
```

## Non-negotiable invariants

1. User scoping is mandatory for all KQL reads and writes.
2. Cross-user reads require an explicit shared-mesh grant row.
3. Google session identity is provenance, not truth.
4. `MISS` is not negative evidence.
5. `UNKNOWN` can branch, defer, or retrieve, but cannot collapse to `else` or `not`.
6. Fact/topology authority requires Ch'en verification and Xul commit.
7. Hyperbolic event intervals guide routing eligibility, not semantic truth.
8. Static API model manifests are provider/routing candidates, not model-truth or semantic authority.
