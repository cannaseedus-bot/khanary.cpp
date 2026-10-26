# MML Suite — v1

Content dictionary epoch: `urn:kuhul:*:v1`

All files in this directory are MathML `<semantics>` contracts. Each carries a
`<csymbol cd="urn:kuhul:<name>:v1">` annotation and a plain-text invariant list.
The machine contract authority for each is the corresponding `.contracts.json`
sibling in `data/grammar/`.

## Foundation contracts

| File | Contract identity |
|---|---|
| `vectors.mml` | `urn:kuhul:vectors:v1` — field vector geometry |
| `metric_graph.mml` | `urn:kuhul:metric-graph:v1` — metric/geodesic graph |
| `map-filter-fold.mml` | `urn:kuhul:map-filter-fold:v1` — HOF pipeline operators |

## Operator contracts

| File | Contract identity |
|---|---|
| `SraiOperators.mml` | `urn:kuhul:srai-operators:v1` — SRAI dispatch surface (SCORE/RANK/COLLAPSE/VERIFY/TRAIN/COMPARE/RECALL/PROJECT/QUESTION) |
| `PatternKernelCalls.mml` | `urn:kuhul:pattern-kernel-calls:v1` — pattern kernel call surface |

## Daemon pattern contracts

| File | Contract identity |
|---|---|
| `DaemonPatterns.mml` | `urn:kuhul:daemon-patterns:v1` — shared daemon discussion/dialogue cycle |
| `CheeseCritic.mml` | `urn:kuhul:cheese-critic:v1` — evidence critic and challenge evaluation |
| `MicronautPatterns.mml` | `urn:kuhul:micronaut-patterns:v1` — cross-micronaut pattern recognition |

## Phase-driver contracts (one per K'UHUL phase)

| File | Phase |
|---|---|
| `phase-driver-pop.mml` | Pop |
| `phase-driver-wo.mml` | Wo |
| `phase-driver-yax.mml` | Yax |
| `phase-driver-sek.mml` | Sek |
| `phase-driver-chen.mml` | Ch'en |
| `phase-driver-xul.mml` | Xul |

## Individual daemon contracts

| File | Daemon |
|---|---|
| `daemon-eliza.mml` | ELIZA |
| `daemon-alice.mml` | ALICE |
| `daemon-think.mml` | Think |
| `daemon-reason.mml` | Reason |
| `daemon-expand.mml` | Expand |
| `daemon-coder.mml` | Coder |
| `daemon-ts_coder.mml` | TS Coder |
| `daemon-ast.mml` | AST |
| `daemon-tool.mml` | Tool |
| `daemon-memory.mml` | Memory |
| `daemon-librarian.mml` | Librarian |
| `daemon-sheogorath.mml` | Sheogorath (T=0.97, Xul) |
| `daemon-jyggalag.mml` | Jyggalag (T=0.00, Ch'en) |
| `daemon-factory.mml` | Factory |
| `daemon-evolution.mml` | Evolution |
| `daemon-atomic.mml` | Atomic |

## Micronaut CODEGEN contracts

| File | Domain |
|---|---|
| `micronaut-CODEGEN-KUHUL.mml` | K'UHUL phase/fold/XCFE op |
| `micronaut-CODEGEN-CPP.mml` | C++ source |
| `micronaut-CODEGEN-PY.mml` | Python tooling |
| `micronaut-CODEGEN-JS.mml` | JavaScript/TypeScript |
| `micronaut-CODEGEN-CSHARP.mml` | C# |
| `micronaut-CODEGEN-HTML.mml` | HTML templates |
| `micronaut-CODEGEN-CSS.mml` | CSS |
| `micronaut-CODEGEN-JSON.mml` | JSON manifests/contracts |
| `micronaut-CODEGEN-KHL.mml` | KHL host-lang |
| `micronaut-CODEGEN-KLSL.mml` | KLSL compiler/shader |

## Versioning policy

A new `v2/` sibling directory is created when the content dictionary epoch
advances (i.e., any `csymbol cd="urn:kuhul:*:v2"` is introduced). Files in
`v1/` are frozen at that point. Cross-version references are explicit:
a v2 document may import a v1 contract but must name the epoch.

Run `py -3 tools/check_srai_operators.py` and `py -3 tools/grammar/mini_transpilers.py fold-geometry-check`
after any edit to verify suite invariants.
