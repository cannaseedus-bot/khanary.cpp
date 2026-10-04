#!/usr/bin/env python3
"""
run_dds_probe.py — launch KHANARY.exe with DDS args and run a single forward pass.

Steps:
  1. Start KHANARY.exe --dds-kson / --dds-manifest / --dds-shards
  2. Wait for "KHANARY ready" on stdout
  3. POST {"jsonrpc":"2.0","method":"dds_probe","params":{"prompt":"Hello","max_tokens":1},"id":1}
  4. Print result and exit 0/1

Usage:
  python tools/run_dds_probe.py
  python tools/run_dds_probe.py --max-tokens 3 --prompt "What is"
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE_CANDIDATES = [
    os.path.join(ROOT, "bin", "KHANARY.exe"),
    os.path.join(ROOT, "build", "conformance", "Release", "KHANARY.exe"),
    os.path.join(ROOT, "build", "Release", "KHANARY.exe"),
    os.path.join(ROOT, "build", "RelWithDebInfo", "KHANARY.exe"),
    os.path.join(ROOT, "build", "Debug", "KHANARY.exe"),
]
EXE      = next((path for path in EXE_CANDIDATES if os.path.exists(path)), EXE_CANDIDATES[0])
SDDS_DIR = r"E:\models\GPT2\sdds"
KSON     = os.path.join(SDDS_DIR, "gpt_oss_20b_folds.kson")
MANIFEST = os.path.join(SDDS_DIR, "model_dds_manifest.json")
RPC_PORT = 8787
STARTUP_TIMEOUT_S = 900  # DDS preload takes ~3 min (385 tensors from E:\)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt",     default="Hello", help="Prompt string (default: Hello)")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="Tokens to generate (default: 1, keeps it fast)")
    ap.add_argument("--port",       type=int, default=RPC_PORT,
                    help=f"JSON-RPC port (default: {RPC_PORT})")
    ap.add_argument("--timeout",    type=int, default=STARTUP_TIMEOUT_S,
                    help=f"Seconds to wait for KHANARY ready (default: {STARTUP_TIMEOUT_S})")
    args = ap.parse_args()

    if not os.path.exists(EXE):
        print(f"ERROR: KHANARY.exe not found at {EXE}", file=sys.stderr)
        print("       Run: cmake --build build --config Release", file=sys.stderr)
        print("       Expected canonical output: bin\\KHANARY.exe", file=sys.stderr)
        return 1

    if not os.path.exists(KSON):
        print(f"ERROR: kson not found: {KSON}", file=sys.stderr)
        return 1

    cmd = [
        EXE,
        "--dds-kson",      KSON,
        "--dds-manifest",  MANIFEST,
        "--dds-shards",    SDDS_DIR,
        "--port-rpc",      str(args.port),
        "--port-mcp",      "8765",  # non-conflicting MCP port
    ]

    print(f"Starting: {EXE}")
    print(f"  kson:     {KSON}")
    print(f"  manifest: {MANIFEST}")
    print(f"  shards:   {SDDS_DIR}")
    print(f"  rpc:      :{args.port}")
    print()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    import threading
    ready = False
    deadline = time.time() + args.timeout
    print("Waiting for KHANARY ready...", flush=True)

    def _read_stdout():
        nonlocal ready
        for line in proc.stdout:
            line = line.rstrip()
            print(f"  {line}", flush=True)
            if "KHANARY ready" in line:
                ready = True
                return

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()
    reader.join(timeout=args.timeout)

    if not ready:
        proc.kill()
        print("ERROR: timed out waiting for KHANARY ready", file=sys.stderr)
        return 1

    print()
    print(f"Sending dds_probe: prompt={args.prompt!r} max_tokens={args.max_tokens}")
    url     = f"http://127.0.0.1:{args.port}"
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method":  "dds_probe",
        "params":  {"prompt": args.prompt, "max_tokens": args.max_tokens},
        "id":      1,
    }).encode()

    rpc_ok = False
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as e:
        print(f"RPC error: {e}", file=sys.stderr)
        proc.kill()
        return 1

    result = body.get("result", {})
    print()
    print("Response:")
    print(json.dumps(result, indent=2))
    print()

    ok         = result.get("ok", False)
    text       = result.get("text", "")
    error      = result.get("error", "")
    resident   = result.get("preload_resident", "?")
    total      = result.get("preload_total",    "?")
    res_mib    = result.get("resident_mib",     "?")

    if ok:
        print(f"[PASS]  forward pass ok=True  text={text!r}")
        print(f"        preload: {resident}/{total} tensors resident, {res_mib} MiB")
        rpc_ok = True
    else:
        print(f"[FAIL]  forward pass ok=False  error={error!r}")

    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()

    return 0 if rpc_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
