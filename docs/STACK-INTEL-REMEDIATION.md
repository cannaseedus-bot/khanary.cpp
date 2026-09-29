# Stack audit remediation

The original `STACK-INTEL-AUDIT.md` is a historical baseline. Use the current
conformance output and this file for implementation status; the audit generator
does not rerun verification and must not be treated as a current health check.

| Audit gap | Resolution / scope | Verification |
|---|---|---|
| TRACK_CONTRACT_DRIFT | Native admission uses the checked-in schema; no-eval is const true; ID/phase/governs, ranges and ports are checked before registration. | KHANARY_CONTRACT_TEST, KHANARY_SCHEMA_TEST |
| MANIFEST_SPECIMEN_DRIFT | Typed recursive conditional tokens preserve intended XCFE data; whole-map validation precedes mutation. Native tests explicitly select both conditional branches. | KHANARY_CONTRACT_TEST, KHANARY_SCHEMA_TEST |
| JSON_PARSE_BOUNDARY | Strict complete-input parsing, duplicate-key/escape/Unicode/number rejection and nesting bound. | Shared JSON fixtures and native contract checks |
| CANONICAL_HASH_GAP | Defined json-c14n-v1, stable serialization and SHA-256 byte goldens; four-field cycle identity helper exposed through read-only RPC. Full certificate authorization is not implied. | KHANARY_CANONICAL_TEST |
| SDDS_PARTIAL_ACCEPT | Shared CPU validator rejects malformed containers atomically before GPU allocations; every byte-prefix truncation is tested. | KHANARY_SDDS_TEST |
| CONFORMANCE_COVERAGE | Added contract, canonical, SDDS, KLSL and independent schema/asset suites. Broader GPU numerical and full compiler coverage remains explicitly deferred. | Root CTest and aggregate runner |
| MODEL_REPRO_DRIFT | Working local asset/schema check replaces unavailable generator/pytest commands; KNU version and historical hardware scope corrected. Fixed sidecar `$schema` admission. | KHANARY_GEOMETRY_ASSET_TEST |
| AUTHORITY_PROVENANCE | Router defaults to deny; host-bound authorizer and phase policy grant permission; payload flags only deny. | KHANARY_ROUTER_TEST |
| VERIFIER_OUTPUT_ABI | Stable sorted JSON verdicts, 0/1/2 exits and summary hashes for all root native/Python suites. | tools/run_conformance.py |

Run `python tools/run_conformance.py`. The report is
`build/conformance/conformance-summary.json`; build/test diagnostics are separate.
See [CONFORMANCE.md](CONFORMANCE.md) for compatibility changes, commands and limits.

Existing runtime processes are not restarted by this work. They continue running
their prior executable; the rebuilt executable is in `build/conformance/Release`.
