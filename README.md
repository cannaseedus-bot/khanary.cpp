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

## Run

```
KHANARY.exe [options]
  --data-dir <path>      Data directory (default: ./data)
  --weights-dir <path>   .scxqdds shard directory
  --port-mcp <n>         MCP port (default: 8764)
  --port-rpc <n>         JSON-RPC port (default: 8787)
  --gpu <hint>           GPU adapter hint (default: Intel)
```

## Ports

| Port | Service |
|------|---------|
| 8764 | MCP gateway (10 tools) |
| 8787 | JSON-RPC 2.0 |

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
`confidence_get`, `jrom_replay`
