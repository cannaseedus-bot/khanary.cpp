# Vendored driver artifacts

This directory is a repository copy of the driver bundle supplied from the
external `drivers` directory. It is an artifact/provenance layer only.

`manifest.json` is authoritative for the copied paths, SHA-256 values, and
the loading policy. The policy is deliberately `declarative-only`: KHANARY
does not discover, load, link, or invoke these DLLs as part of its normal
runtime or conformance tests. Runtime integration must be added later through
an explicit, validated adapter contract.

`llama-common.dll` is included because the llama.cpp CLI/server support DLLs
import it; the core `llama.dll` path itself uses the already bundled GGML
libraries. The factory, JSON-runtime, and evolution DLLs are retained as
optional reference artifacts only.

`examples/cssc_gate_run/` keeps the useful CSSC lowering sources and generated
HLSL/WGSL examples while excluding executables and object files. The six phase
profiles under `examples/micronauts/` are reference inputs; the runtime phase
contracts remain under `data/drivers/phases/`.

The source directories are retained so their shader and KLSL assets can be
audited alongside the existing in-repo compiler and backend projections. The
external `bin/` bundle was intentionally omitted because it is an opaque,
unneeded executable distribution rather than a llama.cpp dependency.
