---
name: build-and-test
description: |
  Build, test, and run the conformance gate for the KHANARY.CPP project using the
  project's bat shim tools. Use this skill whenever the user says "build and test",
  "build then test", "check my changes", "run the tests", "does this compile",
  "run conformance", "verify my change", or wants to confirm that a fix, feature,
  or refactor actually works end-to-end. Also trigger when the user names a specific
  test target (e.g. "build KHANARY_CONTRACT_TEST and run it", "test the phase driver",
  "check if KHANARY_SEMANTIC_TEST passes"). Accepts an optional target name as $ARGUMENTS.
---

# build-and-test

Chains three stages: **build → test → conformance**, short-circuiting on the first failure.
All operations route through the project-root bat shims — never raw cmake, MSBuild, or
direct `bin\*.exe` calls, because KHANARY.CPP lives on E: drive where execute-denial and
chdir restrictions apply.

## Scope

Determine scope from `$ARGUMENTS`:

- **No argument** — build all targets, skip per-target `test.bat`, run full conformance.
- **Target name** (e.g. `KHANARY_CONTRACT_TEST`) — build that target, run `test.bat` for it,
  then run full conformance.

## Stage 1 — Build

```
build.bat [TARGET]        # omit TARGET when building all
```

`build.bat` places the build tree on C: when source is on E:, then calls MSBuild.
Binaries land in `bin/` regardless of config.

- Exit 0 → proceed to Stage 2.
- Non-zero → record `Build: FAIL`, stop here, skip remaining stages.

## Stage 2 — Per-target test (target mode only)

```
test.bat TARGET
```

`test.bat` stages the binary to C:\Temp before execution (avoids E: execute denial),
copies DLLs, and fills known default args for common targets automatically.

- Skip this stage entirely when no target was given.
- Exit 0 → proceed to Stage 3.
- Non-zero → record `Test: FAIL`, stop here, skip conformance.

## Stage 3 — Conformance

```
py -3 tools\run_conformance.py
```

This runs the full CTest suite, collects structured verdicts, and writes
`build/conformance/conformance-summary.json`. It emits a single JSON object to stdout:

```json
{"@kind": "conformance-result", "verdict": "PASS", "counts": {"suites": 32, "cases": 184}, ...}
```

Exit codes: `0` = all suites PASS, `1` = at least one FAIL, `2` = fatal (build missing, hash mismatch).

Parse `verdict`, `counts.suites`, and `counts.cases` from that JSON for the report.

## Report

After all stages complete (or short-circuit), print exactly this summary block:

```
── build-and-test ─────────────────────────────
Build:       PASS          (or FAIL: <exit code>)
Test:        PASS          (or FAIL: <exit code> / SKIPPED)
Conformance: PASS (32 suites, 184 cases)
             (or FAIL / FATAL: <message> / SKIPPED)
───────────────────────────────────────────────
```

If conformance exits 2 (FATAL), surface the error message from its JSON `message` field
or stderr so the user knows whether it was a missing build, a hash mismatch, or a Python error.

## Constraints

- Route everything through bat shims — `build.bat`, `test.bat`, `run.bat`, `read.bat`,
  `write.bat`. Reading AGENTS.md first is worth it if you're unsure which shim fits a task.
- Never write directly to E: paths from a subprocess — stage to `%TEMP%\` first.
- `run_conformance.py --skip-build` is available if you've just built and only want to
  rerun the test suite without a rebuild (useful when the user fixes a single test).
