"""
bot.py — KHANARY.CPP assistant bot
Reads openai.yaml for endpoint config.
Sends prompts to HeavyAdvisor (oss, 17887) or Chat3 (qwen3, 17888).
Also exposes helpers for hitting KHANARY's own MCP / JSON-RPC ports.
"""

import sys
import json
import os
import yaml
import argparse
import re
import subprocess
import shlex
import shutil
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from openai import OpenAI

CONFIG_PATH = Path(__file__).parent / "openai.yaml"
DEFAULT_KHANARY_EXE = Path(__file__).parent / "build" / "Release" / "KHANARY.exe"
DEFAULT_KHANARY_CWD = Path(__file__).parent
DEFAULT_TUI_SESSION = Path(__file__).parent / ".khanary_tui.session.json"


def _enable_windows_virtual_terminal() -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle in (0, -1):
            return False
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if mode.value & enable_vt:
            return True
        return kernel32.SetConsoleMode(handle, mode.value | enable_vt) != 0
    except Exception:
        return False


def _supports_ansi_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return _enable_windows_virtual_terminal()


_ANSI_ENABLED = _supports_ansi_color()
_ANSI_RESET = "\x1b[0m"
_ANSI_WHITE = "\x1b[97m"
_ANSI_BOLD = "\x1b[1m"
_ANSI_BG_INPUT = "\x1b[48;5;238m"
_INPUT_FORM_HEIGHT = 5


def _colorize(text: str, ansi_code: str) -> str:
    if not _ANSI_ENABLED:
        return text
    return f"{ansi_code}{text}{_ANSI_RESET}"


def _user_message_style(text: str) -> str:
    if not _ANSI_ENABLED:
        return text
    return f"{_ANSI_WHITE}{_ANSI_BOLD}{text}{_ANSI_RESET}"


def _input_form_style(text: str) -> str:
    if not _ANSI_ENABLED:
        return text
    return f"{_ANSI_BG_INPUT}{_ANSI_WHITE}{_ANSI_BOLD}{text}{_ANSI_RESET}"


def _clear_terminal_screen() -> None:
    if _ANSI_ENABLED:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()
        return
    os.system("cls" if os.name == "nt" else "clear")


def _terminal_size() -> tuple[int, int]:
    fallback = shutil.get_terminal_size(fallback=(120, 40))
    cols = max(20, int(fallback.columns))
    rows = max(12, int(fallback.lines))

    if os.name != "nt" or not _stdin_tty():
        return cols, rows

    try:
        import ctypes

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class SMALL_RECT(ctypes.Structure):
            _fields_ = [
                ("Left", ctypes.c_short),
                ("Top", ctypes.c_short),
                ("Right", ctypes.c_short),
                ("Bottom", ctypes.c_short),
            ]

        class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
            _fields_ = [
                ("dwSize", COORD),
                ("dwCursorPosition", COORD),
                ("wAttributes", ctypes.c_ushort),
                ("srWindow", SMALL_RECT),
                ("dwMaximumWindowSize", COORD),
            ]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = CONSOLE_SCREEN_BUFFER_INFO()
        if handle not in (0, -1) and kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            win_cols = int(info.srWindow.Right - info.srWindow.Left + 1)
            win_rows = int(info.srWindow.Bottom - info.srWindow.Top + 1)
            if win_cols > 0 and win_rows > 0:
                return max(20, win_cols), max(12, win_rows)
    except Exception:
        pass

    return cols, rows


def _wrap_with_indent(text: str, first_prefix: str, next_prefix: str, width: int) -> list[str]:
    usable = max(20, width - len(next_prefix))
    rows: list[str] = []
    first = True
    lines = str(text).splitlines() or [""]
    for raw in lines:
        wrapped = textwrap.wrap(
            raw,
            width=usable,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        for piece in wrapped:
            prefix = first_prefix if first else next_prefix
            rows.append(f"{prefix}{piece}")
            first = False
    return rows


def _history_entry_lines(entry: dict, width: int) -> list[str]:
    role = str(entry.get("role", "assistant"))
    content = str(entry.get("content", ""))
    bubble_width = max(30, min(width - 10, int(width * 0.70)))

    if role == "user":
        label = "You> "
        pad = " " * len(label)
        lines = _wrap_with_indent(content, label, pad, bubble_width)
        align_width = max(30, width - 1)
        lines = [line.rjust(max(len(line), align_width)) for line in lines]
        return [_user_message_style(line) for line in lines]

    label = "BOT> "
    pad = " " * len(label)
    return _wrap_with_indent(content, label, pad, bubble_width)


def _stdin_tty() -> bool:
    return bool(hasattr(sys.stdin, "isatty") and sys.stdin.isatty())


def _shift_pressed() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
    except Exception:
        return False


def _compose_input_rows(buffer: str, width: int) -> list[str]:
    first_prefix = "You> "
    next_prefix = "...  "
    first_width = max(8, width - len(first_prefix))
    next_width = max(8, width - len(next_prefix))

    rows: list[str] = []
    parts = str(buffer).split("\n")
    if not parts:
        parts = [""]

    for idx, raw in enumerate(parts):
        prefix = first_prefix if idx == 0 else next_prefix
        usable = first_width if idx == 0 else next_width
        wrapped = textwrap.wrap(
            raw,
            width=usable,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        for j, piece in enumerate(wrapped):
            p = prefix if j == 0 else next_prefix
            rows.append(f"{p}{piece}")

    if not rows:
        rows = [first_prefix]
    return rows


def _draw_input_form(buffer: str, *, input_height: int) -> None:
    width, _ = _terminal_size()
    paint_width = max(1, width - 1)
    rows = _compose_input_rows(buffer, width)
    if len(rows) > input_height:
        rows = rows[-input_height:]
        if rows and not rows[0].startswith("…"):
            rows[0] = "…" + rows[0][1:] if len(rows[0]) > 1 else "…"
    if len(rows) < input_height:
        rows = ([""] * (input_height - len(rows))) + rows

    if _ANSI_ENABLED:
        if input_height > 1:
            sys.stdout.write(f"\x1b[{input_height - 1}A")
        for i in range(input_height):
            row = rows[i] if i < len(rows) else ""
            shown = row[:paint_width]
            sys.stdout.write("\r")
            sys.stdout.write(f"{_ANSI_BG_INPUT}{' ' * paint_width}{_ANSI_RESET}")
            sys.stdout.write("\r")
            pad = " " * max(0, paint_width - len(shown))
            sys.stdout.write(f"{_ANSI_BG_INPUT}{_ANSI_WHITE}{_ANSI_BOLD}{shown}{pad}{_ANSI_RESET}")
            if i < input_height - 1:
                sys.stdout.write("\n")

        cursor_col = min(paint_width, len(rows[-1]) + 1 if rows else 1)
        sys.stdout.write("\r")
        sys.stdout.write(f"\x1b[{cursor_col}G")
    else:
        row = rows[-1] if rows else "You> "
        shown = row[: max(1, width - 1)].ljust(max(1, width - 1))
        sys.stdout.write("\r" + shown)
    sys.stdout.flush()


def _read_tui_input(*, input_height: int = _INPUT_FORM_HEIGHT) -> str:
    if os.getenv("KHANARY_TUI_SIMPLE_INPUT", "") == "1" or not _ANSI_ENABLED or not _stdin_tty():
        first = input("You> ")
        if not first.endswith("\\"):
            return first
        lines = [first[:-1]]
        while True:
            nxt = input("... ")
            if nxt.endswith("\\"):
                lines.append(nxt[:-1])
                continue
            lines.append(nxt)
            return "\n".join(lines)

    if os.name != "nt":
        first = input(_input_form_style("You> "))
        if not first.endswith("\\"):
            return first
        lines = [first[:-1]]
        while True:
            nxt = input(_input_form_style("... "))
            if nxt.endswith("\\"):
                lines.append(nxt[:-1])
                continue
            lines.append(nxt)
            return "\n".join(lines)

    try:
        import msvcrt  # type: ignore
    except Exception:
        return input(_input_form_style("You> "))

    buf: list[str] = []
    _draw_input_form("", input_height=input_height)
    try:
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                if _shift_pressed():
                    buf.append("\n")
                    _draw_input_form("".join(buf), input_height=input_height)
                    continue
                if buf and buf[-1] == "\\":
                    buf.pop()
                    buf.append("\n")
                    _draw_input_form("".join(buf), input_height=input_height)
                    continue
                sys.stdout.write("\r" + _ANSI_RESET)
                sys.stdout.flush()
                return "".join(buf)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\x00", "\xe0"):
                # Swallow special key sequence second byte (arrows, etc.).
                _ = msvcrt.getwch()
                continue
            if ch in ("\b", "\x7f"):
                if buf:
                    buf.pop()
                _draw_input_form("".join(buf), input_height=input_height)
                continue
            if ch == "\t":
                buf.extend([" ", " ", " ", " "])
                _draw_input_form("".join(buf), input_height=input_height)
                continue
            if ch and ord(ch) >= 32:
                buf.append(ch)
                _draw_input_form("".join(buf), input_height=input_height)
    finally:
        sys.stdout.write(_ANSI_RESET)
        sys.stdout.flush()


def _render_tui_screen(state: dict) -> int:
    columns, lines = _terminal_size()
    width = max(40, int(columns))
    height = max(16, int(lines))
    input_height = max(3, min(_INPUT_FORM_HEIGHT, height - 8))

    header_lines = [
        "KHANARY Terminal TUI",
        f"session={state.get('session_id')} profile={state.get('profile')} mode={state.get('mode')}",
        "Type /help for commands.",
    ]

    raw_feed_lines: list[str] = []
    for row in state.get("history", []):
        raw_feed_lines.extend(_history_entry_lines(row, width))
        raw_feed_lines.append("")

    message_window = max(6, height - len(header_lines) - input_height)
    if len(raw_feed_lines) > message_window:
        feed_lines = raw_feed_lines[-message_window:]
    else:
        top_pad = [""] * (message_window - len(raw_feed_lines))
        feed_lines = top_pad + raw_feed_lines

    input_pad = [""] * input_height
    screen_lines = header_lines + feed_lines + input_pad
    _clear_terminal_screen()
    sys.stdout.write("\n".join(screen_lines))
    sys.stdout.flush()
    return input_height


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_client(cfg: dict, profile: str = "advisor") -> tuple[OpenAI, dict]:
    if profile not in cfg:
        raise KeyError(f"Unknown profile '{profile}' in {CONFIG_PATH.name}")
    p = cfg[profile]
    client = OpenAI(base_url=p["base_url"], api_key="local")
    return client, p


def chat(prompt: str, profile: str = "advisor", system: str = "") -> str:
    cfg = load_config()
    client, p = make_client(cfg, profile)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=p["model"],
        messages=messages,
        temperature=p.get("temperature", 0.2),
        max_tokens=p.get("max_tokens", 1024),
    )
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def task_outline(extra_context: str = "") -> str:
    """Ask HeavyAdvisor for next tasks on KHANARY.CPP."""
    system = (
        "You are HeavyAdvisor, a senior C++ systems engineer. "
        "Produce a concrete, prioritized task list for KHANARY.CPP. "
        "Group as BLOCKER / HIGH / MEDIUM. Name files and methods."
    )
    prompt = "What are the next tasks to complete KHANARY.CPP and reach a working KHANARY.exe?"
    if extra_context:
        prompt += f"\n\nContext:\n{extra_context}"
    return chat(prompt, profile="advisor", system=system)


def mcp_call(tool: str, params: dict, base_url: str = "http://127.0.0.1:8764") -> dict:
    """Call a KHANARY MCP tool via JSON-RPC 2.0."""
    import urllib.request
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": tool,
        "params": params,
    }).encode()
    req = urllib.request.Request(
        base_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def rpc_call(method: str, params: dict, base_url: str = "http://127.0.0.1:8787") -> dict:
    """Call KHANARY JSON-RPC 2.0 endpoint."""
    return mcp_call(method, params, base_url=base_url)


def khanary_flag_catalog() -> list[dict[str, str]]:
    """Authoritative KHANARY.exe CLI flags (mirrors src/main.cpp parse_args)."""
    return [
        {"flag": "--port-mcp <n>", "description": "MCP server port (default: 8764)"},
        {"flag": "--port-rpc <n>", "description": "JSON-RPC server port (default: 8787)"},
        {"flag": "--data-dir <path>", "description": "Data directory (default: ./data)"},
        {"flag": "--weights-dir <path>", "description": "SCXQDDS weight shards directory"},
        {"flag": "--dds-kson <path>", "description": "KAST/1 .kson driver file"},
        {"flag": "--dds-manifest <path>", "description": "DDS shard manifest JSON"},
        {"flag": "--dds-shards <path>", "description": "Directory containing fold_N.dds files"},
        {"flag": "--gpu <hint>", "description": "GPU adapter hint (default: Intel)"},
        {"flag": "--proj-book <path>", "description": "Projection ABI book JSON"},
        {"flag": "--gyro-book <path>", "description": "Gyroscope ABI book JSON"},
        {"flag": "--verbose", "description": "Verbose logging"},
        {"flag": "--help", "description": "Print KHANARY.exe help and exit"},
    ]


def _normalize_khanary_passthrough_args(raw_args: list[str]) -> list[str]:
    out = list(raw_args)
    if out and out[0] == "--":
        out = out[1:]
    return out


def run_khanary(
    exe_path: Path,
    khanary_args: list[str],
    *,
    cwd: Optional[Path] = None,
    wait: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    args = _normalize_khanary_passthrough_args(khanary_args)
    cmd = [str(exe_path), *args]
    workdir = str(cwd) if cwd else None

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "cmd": cmd,
            "cwd": workdir,
        }

    if not exe_path.exists():
        return {
            "ok": False,
            "error": f"KHANARY executable not found: {exe_path}",
            "cmd": cmd,
            "cwd": workdir,
        }

    try:
        if wait:
            proc = subprocess.run(cmd, cwd=workdir, check=False)
            return {
                "ok": proc.returncode == 0,
                "cmd": cmd,
                "cwd": workdir,
                "wait": True,
                "returncode": int(proc.returncode),
            }

        popen_kwargs: dict[str, Any] = {
            "cwd": workdir,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if os.name == "nt":
            flags = 0
            detached = getattr(subprocess, "DETACHED_PROCESS", 0)
            new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            flags = detached | new_group
            if flags:
                popen_kwargs["creationflags"] = flags

        proc = subprocess.Popen(cmd, **popen_kwargs)
        return {
            "ok": True,
            "cmd": cmd,
            "cwd": workdir,
            "wait": False,
            "pid": int(proc.pid),
        }
    except Exception as exc:
        return {
            "ok": False,
            "cmd": cmd,
            "cwd": workdir,
            "wait": wait,
            "error": str(exc),
        }


def _url_port(url: str, default: int) -> int:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.port:
            return int(parsed.port)
    except Exception:
        pass
    return int(default)


def _default_khanary_launch_args(cfg: dict) -> list[str]:
    rpc_port = _url_port(_base_url(cfg, "rpc"), 8787)
    mcp_port = _url_port(_base_url(cfg, "mcp"), 8764)
    return ["--port-rpc", str(rpc_port), "--port-mcp", str(mcp_port)]


def _start_khanary_runtime(
    state: dict,
    *,
    reason: str,
    force: bool = False,
) -> tuple[bool, str]:
    if not force and state.get("khanary_launch_attempted"):
        prior = state.get("khanary_launch_report")
        if isinstance(prior, dict) and prior.get("ok"):
            pid = prior.get("pid")
            suffix = f" (pid={pid})" if pid is not None else ""
            return True, f"[runtime] KHANARY launch already attempted{suffix}."
        return False, "[runtime] KHANARY launch already attempted and unavailable."

    cfg = state.get("cfg")
    if not isinstance(cfg, dict):
        cfg = load_config()
        state["cfg"] = cfg

    launch_args = _default_khanary_launch_args(cfg)
    report = run_khanary(
        Path(state.get("khanary_exe", DEFAULT_KHANARY_EXE)),
        launch_args,
        cwd=Path(state.get("khanary_cwd", DEFAULT_KHANARY_CWD)),
        wait=False,
        dry_run=False,
    )
    state["khanary_launch_attempted"] = True
    state["khanary_launch_report"] = report

    if report.get("ok"):
        pid = report.get("pid")
        suffix = f" pid={pid}" if pid is not None else ""
        rpc_url = _base_url(cfg, "rpc")
        if _wait_for_rpc_ready(rpc_url, wait_seconds=6.0):
            return True, f"[runtime] Launched KHANARY.exe{suffix} ({reason}) and RPC is online at {rpc_url}."
        return True, (
            f"[runtime] Launched KHANARY.exe{suffix} ({reason}) but RPC is not reachable yet at {rpc_url}."
        )

    error = str(report.get("error", "unknown error"))
    return False, f"[runtime] Failed to launch KHANARY.exe ({reason}): {error}"


def _center_cfg(cfg: dict) -> dict:
    return cfg.get("command_center", {})


def _base_url(cfg: dict, surface: str) -> str:
    if surface == "mcp":
        return cfg.get("khanary_mcp", {}).get("base_url", "http://127.0.0.1:8764")
    if surface == "rpc":
        return cfg.get("khanary_rpc", {}).get("base_url", "http://127.0.0.1:8787")
    raise ValueError(f"Unknown surface '{surface}'")


def _rpc_endpoint_ready(base_url: str, timeout: float = 1.2) -> bool:
    import urllib.request

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "list_keys",
        "params": {},
    }).encode()
    req = urllib.request.Request(
        base_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return isinstance(data, dict) and ("result" in data or "error" in data)
    except Exception:
        return False


def _wait_for_rpc_ready(base_url: str, wait_seconds: float = 6.0) -> bool:
    import time

    deadline = time.time() + max(0.0, wait_seconds)
    while time.time() < deadline:
        if _rpc_endpoint_ready(base_url, timeout=1.0):
            return True
        time.sleep(0.25)
    return _rpc_endpoint_ready(base_url, timeout=1.0)


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_tool_calls(content: str, tool_tag: str = "tools") -> tuple[list[dict], str, list[str]]:
    pattern = re.compile(
        rf"<{re.escape(tool_tag)}>\s*(.*?)\s*</{re.escape(tool_tag)}>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(content))
    if not matches:
        return [], content.strip(), []

    calls: list[dict] = []
    errors: list[str] = []
    for i, m in enumerate(matches):
        blob = _strip_code_fence(m.group(1))
        try:
            parsed = json.loads(blob)
        except Exception as exc:
            errors.append(f"tools[{i}] json parse error: {exc}")
            continue

        chunk: list[Any]
        if isinstance(parsed, dict):
            if isinstance(parsed.get("tool_calls"), list):
                chunk = parsed["tool_calls"]
            elif isinstance(parsed.get("calls"), list):
                chunk = parsed["calls"]
            else:
                chunk = [parsed]
        elif isinstance(parsed, list):
            chunk = parsed
        else:
            errors.append(f"tools[{i}] payload must be object or array")
            continue

        for j, c in enumerate(chunk):
            if isinstance(c, dict):
                calls.append(c)
            else:
                errors.append(f"tools[{i}].call[{j}] must be object")

    cleaned = pattern.sub("", content).strip()
    return calls, cleaned, errors


def _normalize_center_call(call: dict, cfg: dict) -> dict:
    cc = _center_cfg(cfg)
    verbs = cc.get("verbs", {})

    if not isinstance(call, dict):
        raise ValueError("tool call must be an object")

    resolved = dict(call)
    verb = str(resolved.get("verb", "")).strip().lower()
    verb_cfg = {}
    if verb:
        verb_cfg = verbs.get(verb, {})
        if not verb_cfg:
            raise ValueError(f"unknown verb '{verb}'")
        resolved.setdefault("surface", verb_cfg.get("surface", "rpc"))
        if verb_cfg.get("method"):
            resolved.setdefault("method", verb_cfg.get("method"))

    surface = str(resolved.get("surface", resolved.get("endpoint", "rpc"))).strip().lower()
    method = resolved.get("method", resolved.get("tool"))
    if not isinstance(method, str) or not method.strip():
        raise ValueError("missing method/tool")
    method = method.strip()

    params = resolved.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    if "query" in resolved and "query" not in params and isinstance(resolved["query"], str):
        params["query"] = resolved["query"].strip()

    query_prefix = verb_cfg.get("query_prefix")
    if isinstance(query_prefix, str) and query_prefix:
        query = params.get("query")
        if isinstance(query, str) and query.strip():
            q = query.strip()
            if not q.lower().startswith(query_prefix.lower()):
                params["query"] = f"{query_prefix}{q}"

    return {
        "surface": surface,
        "method": method,
        "params": params,
        "verb": verb if verb else None,
        "track": verb_cfg.get("track"),
    }


def _is_allowed_call(surface: str, method: str, cfg: dict) -> bool:
    cc = _center_cfg(cfg)
    if cc.get("allow_unknown_methods", False):
        return True
    allowlist = cc.get("allowlist", {})
    allowed = allowlist.get(surface, [])
    return method in allowed


def _execute_center_call(call: dict, cfg: dict, dry_run: bool = False) -> dict:
    surface = call["surface"]
    method = call["method"]
    params = call["params"]
    track = call.get("track")

    if not _is_allowed_call(surface, method, cfg):
        return {
            "ok": False,
            "blocked": True,
            "surface": surface,
            "method": method,
            "params": params,
            "track": track,
            "error": "method not in command_center allowlist",
        }

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "surface": surface,
            "method": method,
            "params": params,
            "track": track,
        }

    base_url = _base_url(cfg, surface)
    try:
        if surface == "mcp":
            response = mcp_call(method, params, base_url=base_url)
        elif surface == "rpc":
            response = rpc_call(method, params, base_url=base_url)
        else:
            raise ValueError(f"unknown surface '{surface}'")
        return {
            "ok": True,
            "surface": surface,
            "method": method,
            "params": params,
            "track": track,
            "response": response,
        }
    except Exception as exc:
        return {
            "ok": False,
            "surface": surface,
            "method": method,
            "params": params,
            "track": track,
            "error": str(exc),
        }


def _command_center_system(cfg: dict) -> str:
    cc = _center_cfg(cfg)
    allow = cc.get("allowlist", {})
    verbs = cc.get("verbs", {})
    tool_tag = cc.get("tool_tag", "tools")
    max_calls = int(cc.get("max_tool_calls", 8))

    allow_rpc = ", ".join(allow.get("rpc", []))
    allow_mcp = ", ".join(allow.get("mcp", []))
    verb_summary = []
    for verb, vcfg in verbs.items():
        surface = vcfg.get("surface", "rpc")
        method = vcfg.get("method", "")
        prefix = vcfg.get("query_prefix", "")
        track = vcfg.get("track", "")
        line = f"{verb}: {surface}.{method}"
        if prefix:
            line += f" (query_prefix='{prefix}')"
        if track:
            line += f" track={track}"
        verb_summary.append(line)

    verb_block = "\n".join(f"- {v}" for v in verb_summary) if verb_summary else "- none"
    return (
        "You are KHANARY Command Center. "
        "Plan deterministic local MCP/RPC calls for this repository.\n"
        f"When tool actions are needed, emit exactly one <{tool_tag}>...</{tool_tag}> block with strict JSON:\n"
        '{"tool_calls":[{"surface":"rpc|mcp","method":"name","params":{},"verb":"optional","query":"optional"}]}\n'
        "Do not emit methods outside the allowlist.\n"
        f"Max tool calls: {max_calls}\n"
        f"Allowed RPC: {allow_rpc}\n"
        f"Allowed MCP: {allow_mcp}\n"
        f"Verb map:\n{verb_block}\n"
        "After the tools block, no extra blocks."
    )


def command_center(prompt: str,
                   profile: Optional[str] = None,
                   dry_run: bool = False,
                   execute: bool = True) -> dict:
    cfg = load_config()
    cc = _center_cfg(cfg)
    chosen_profile = profile or cc.get("profile", "advisor")

    system = _command_center_system(cfg)
    try:
        raw = chat(prompt, profile=chosen_profile, system=system)
    except Exception as exc:
        return {
            "ok": False,
            "profile": chosen_profile,
            "prompt": prompt,
            "assistant_message": "",
            "tool_calls": [],
            "tool_results": [],
            "parse_errors": [],
            "normalize_errors": [],
            "truncated_to_max_tool_calls": False,
            "raw_response": "",
            "error": str(exc),
        }
    tool_tag = cc.get("tool_tag", "tools")
    raw_calls, assistant_text, parse_errors = extract_tool_calls(raw, tool_tag=tool_tag)

    max_calls = int(cc.get("max_tool_calls", 8))
    truncated = len(raw_calls) > max_calls
    if truncated:
        raw_calls = raw_calls[:max_calls]

    normalized: list[dict] = []
    normalize_errors: list[str] = []
    for i, call in enumerate(raw_calls):
        try:
            normalized.append(_normalize_center_call(call, cfg))
        except Exception as exc:
            normalize_errors.append(f"tool_call[{i}] normalize error: {exc}")

    results: list[dict] = []
    if execute:
        for call in normalized:
            results.append(_execute_center_call(call, cfg, dry_run=dry_run))

    ok = not parse_errors and not normalize_errors and all(r.get("ok", False) for r in results)
    return {
        "ok": ok,
        "profile": chosen_profile,
        "prompt": prompt,
        "assistant_message": assistant_text,
        "tool_calls": normalized,
        "tool_results": results,
        "parse_errors": parse_errors,
        "normalize_errors": normalize_errors,
        "truncated_to_max_tool_calls": truncated,
        "raw_response": raw,
    }


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _append_history(state: dict, role: str, content: str, **meta: Any) -> None:
    state.setdefault("history", []).append({
        "role": role,
        "content": content,
        "timestamp": _now_iso(),
        "meta": meta or {},
    })


def _build_terminal_prompt(history: list[dict], max_turns: int = 12, max_chars_each: int = 3500) -> str:
    turns = [
        h for h in history
        if h.get("role") in {"user", "assistant"} and not bool((h.get("meta") or {}).get("ui_only"))
    ]
    if len(turns) > max_turns:
        turns = turns[-max_turns:]

    lines = [
        "Terminal conversation (oldest to newest). Continue from the last user turn.",
    ]
    for row in turns:
        role = row.get("role")
        label = "User" if role == "user" else "Assistant"
        content = str(row.get("content", ""))
        lines.append(f"{label}: {_truncate_text(content, max_chars_each)}")
    return "\n\n".join(lines)


def _guess_fence_language(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".xml": "xml",
        ".xhtml": "xml",
        ".svg": "xml",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".yml": "yaml",
        ".yaml": "yaml",
    }
    return mapping.get(ext, "text")


def _load_file_as_prompt_block(path_str: str, max_chars: int = 120_000) -> tuple[bool, str]:
    if not path_str.strip():
        return False, "No file path provided."
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Not a file: {path}"

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return False, f"Failed to read file '{path}': {exc}"

    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    lang = _guess_fence_language(path)
    body = f"Loaded file: {path}\n```{lang}\n{text}\n```"
    if truncated:
        body += "\n\n[File content truncated to terminal safety limit.]"
    return True, body


def _parse_method_and_params(rest: str) -> tuple[str, dict]:
    raw = rest.strip()
    if not raw:
        raise ValueError("Expected '<method-or-tool> [json-params]'.")
    parts = raw.split(maxsplit=1)
    method = parts[0].strip()
    if not method:
        raise ValueError("Missing method/tool name.")

    if len(parts) == 1:
        return method, {}

    params_raw = parts[1].strip()
    if not params_raw:
        return method, {}
    try:
        params = json.loads(params_raw)
        if isinstance(params, dict):
            return method, params
        return method, {"value": params}
    except Exception:
        # Convenience fallback for plain query payloads.
        return method, {"query": params_raw}


def _tools_overview_text(cfg: dict) -> str:
    cc = _center_cfg(cfg)
    allow = cc.get("allowlist", {})
    verbs = cc.get("verbs", {})
    lines = ["Allowed RPC methods:"]
    for m in allow.get("rpc", []):
        lines.append(f"  - {m}")

    lines.append("")
    lines.append("Allowed MCP tools:")
    for m in allow.get("mcp", []):
        lines.append(f"  - {m}")

    lines.append("")
    lines.append("Runtime verb actions:")
    for verb, vcfg in sorted(verbs.items()):
        surface = vcfg.get("surface", "rpc")
        method = vcfg.get("method", "")
        prefix = vcfg.get("query_prefix", "")
        track = vcfg.get("track", "")
        extra = f" prefix={prefix!r}" if prefix else ""
        if track:
            extra += f" track={track}"
        lines.append(f"  /{verb:<12} -> {surface}.{method}{extra}")
    return "\n".join(lines)


def _print_tools_overview(cfg: dict) -> None:
    print(_tools_overview_text(cfg))


def _format_tool_result_summary(results: list[dict], include_raw: bool = False) -> str:
    if not results:
        return ""
    lines = ["[tool-results]"]
    for i, row in enumerate(results, start=1):
        surface = row.get("surface", "?")
        method = row.get("method", "?")
        ok = bool(row.get("ok", False))
        status = "ok" if ok else "error"
        line = f"{i}. {surface}.{method} [{status}]"
        if row.get("blocked"):
            line += " blocked"
        if row.get("error"):
            line += f" — {row['error']}"
        lines.append(line)
        if include_raw and "response" in row:
            lines.append(_truncate_text(_pretty_json(row["response"]), 2400))
    return "\n".join(lines)


def _looks_like_connection_error(message: str) -> bool:
    if not message:
        return False
    text = message.lower()
    markers = (
        "connection error",
        "api connection error",
        "connection refused",
        "actively refused",
        "winerror 10061",
        "timed out",
        "max retries exceeded",
        "failed to establish a new connection",
    )
    return any(marker in text for marker in markers)


def _unwrap_rpc_result(payload: Any) -> Any:
    if isinstance(payload, dict) and "result" in payload:
        return payload.get("result")
    return payload


def _render_key_query_payload(payload: dict, include_raw: bool = False) -> str:
    result = _unwrap_rpc_result(payload)
    if isinstance(result, dict):
        status = str(result.get("status", "")).strip()
        response = result.get("response")
        lines: list[str] = []

        if isinstance(response, str) and response.strip():
            lines.append(response.strip())
        elif isinstance(result.get("answer"), str) and result.get("answer", "").strip():
            lines.append(str(result["answer"]).strip())
        elif isinstance(result.get("message"), str) and result.get("message", "").strip():
            lines.append(str(result["message"]).strip())

        if status:
            lines.append(f"[status] {status}")

        if lines:
            if include_raw:
                lines.append(_truncate_text(_pretty_json(payload), 2400))
            return "\n".join(lines)

    if isinstance(payload, dict) and payload.get("error"):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
    if include_raw:
        return _truncate_text(_pretty_json(payload), 2400)
    return _render_structured_payload(payload, include_raw=False)


def _extract_jsonrpc_error(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    err = payload.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        code = err.get("code")
        if isinstance(msg, str) and msg.strip():
            if code is None:
                return msg.strip()
            return f"{msg.strip()} (code={code})"
    if isinstance(err, str) and err.strip():
        return err.strip()
    return ""


def _scalar_preview(obj: dict, *, skip: Optional[set[str]] = None, limit: int = 5) -> str:
    skip_keys = skip or set()
    parts: list[str] = []
    for key, value in obj.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={value}")
        elif isinstance(value, str) and value.strip():
            parts.append(f"{key}={value.strip()}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def _looks_like_markup(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    if t.startswith("```") or t.startswith("<") or "</" in t:
        return True
    return bool(re.search(r"<[a-zA-Z][^>]*>", t))


def _render_structured_payload(payload: Any, include_raw: bool = False) -> str:
    err = _extract_jsonrpc_error(payload)
    if err:
        if include_raw:
            return f"[error] {err}\n\n{_truncate_text(_pretty_json(payload), 2400)}"
        return f"[error] {err}"

    result = _unwrap_rpc_result(payload)
    lines: list[str] = []
    if isinstance(result, dict):
        response = result.get("response")
        if isinstance(response, str) and response.strip():
            lines.append(response.strip())
        elif isinstance(response, (int, float, bool)):
            lines.append(str(response))

        for key, label in (
            ("status", "status"),
            ("op", "op"),
            ("source", "source"),
            ("miss_kind", "miss"),
        ):
            value = result.get(key)
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                lines.append(f"[{label}] {value}")

        if not lines:
            for key, value in result.items():
                if isinstance(value, str) and _looks_like_markup(value):
                    lines.append(value.strip())
                    lines.append(f"[field] {key}")
                    break

        if not lines:
            preview = _scalar_preview(
                result,
                skip={
                    "route_hint",
                    "runtime_attention",
                    "research_fallback",
                    "key_references",
                    "ngram_references",
                    "semantic_folding_projection",
                },
            )
            if preview:
                lines.append(f"[structured-result] {preview}")
            else:
                lines.append("[structured-result] Response received. Use /raw on for full JSON.")
    elif isinstance(result, list):
        lines.append(f"[items] {len(result)} item(s).")
    elif isinstance(result, (str, int, float, bool)):
        lines.append(str(result))
    else:
        lines.append("[structured-result] Response received.")

    if include_raw:
        lines.append(_truncate_text(_pretty_json(payload), 2400))
    return "\n".join(lines).strip()


def _render_center_call_result(call_result: dict, include_raw: bool = False) -> str:
    surface = str(call_result.get("surface", "?"))
    method = str(call_result.get("method", "?"))
    if not bool(call_result.get("ok", False)):
        err = str(call_result.get("error", "unknown error")).strip() or "unknown error"
        out = f"[{surface}.{method}] error: {err}"
        if include_raw:
            out += "\n\n" + _truncate_text(_pretty_json(call_result), 2400)
        return out

    if call_result.get("dry_run"):
        return f"[{surface}.{method}] dry-run call prepared."

    if "response" in call_result:
        body = _render_structured_payload(call_result.get("response"), include_raw=include_raw)
        return f"[{surface}.{method}]\n{body}".strip()

    out = f"[{surface}.{method}] ok"
    if include_raw:
        out += "\n\n" + _truncate_text(_pretty_json(call_result), 2400)
    return out


def _fallback_key_query(state: dict, user_text: str) -> tuple[bool, str]:
    cfg = state.get("cfg")
    if not isinstance(cfg, dict):
        cfg = load_config()
        state["cfg"] = cfg
    rpc_url = _base_url(cfg, "rpc")
    try:
        payload = rpc_call("key_query", {"query": user_text}, base_url=rpc_url)
    except Exception as exc:
        err_text = str(exc)
        if bool(state.get("auto_launch_khanary", True)) and _looks_like_connection_error(err_text):
            started, launch_msg = _start_khanary_runtime(
                state,
                reason="RPC fallback unavailable",
                force=False,
            )
            if started:
                import time

                time.sleep(0.9)
                try:
                    payload = rpc_call("key_query", {"query": user_text}, base_url=rpc_url)
                    rendered = _render_key_query_payload(
                        payload,
                        include_raw=bool(state.get("show_raw_results", False)),
                    )
                    return True, f"{launch_msg}\n\n{rendered}"
                except Exception as retry_exc:
                    return False, (
                        f"{launch_msg}\n"
                        f"[fallback-error] rpc.key_query unavailable at {rpc_url}: {retry_exc}"
                    )
            return False, f"{launch_msg}\n[fallback-error] rpc.key_query unavailable at {rpc_url}: {err_text}"
        return False, f"[fallback-error] rpc.key_query unavailable at {rpc_url}: {err_text}"
    rendered = _render_key_query_payload(payload, include_raw=bool(state.get("show_raw_results", False)))
    return True, rendered


def _run_center_turn(state: dict, user_text: str) -> str:
    prompt = _build_terminal_prompt(state.get("history", []))
    report = command_center(
        prompt,
        profile=state["profile"],
        dry_run=bool(state.get("dry_run", False)),
        execute=bool(state.get("execute_tools", True)),
    )

    report_error = str(report.get("error", "")).strip()
    if report_error:
        if _looks_like_connection_error(report_error):
            fallback_ok, fallback_text = _fallback_key_query(state, user_text)
            if fallback_ok:
                return (
                    f"[planner-offline] {report_error}\n"
                    "[mode] using rpc.key_query fallback\n\n"
                    f"{fallback_text}"
                ).strip()
            details = [
                f"[planner-offline] {report_error}",
                fallback_text,
                "Tip: start local planner endpoint(s) from openai.yaml or use /rpc once KHANARY RPC is running.",
            ]
            if bool(state.get("show_raw_results", False)):
                details.append(_truncate_text(_pretty_json(report), 2400))
            return "\n\n".join(details).strip()

        if bool(state.get("show_raw_results", False)):
            return f"[planner-error] {report_error}\n\n{_truncate_text(_pretty_json(report), 2400)}"
        return f"[planner-error] {report_error}"

    sections: list[str] = []
    assistant_text = str(report.get("assistant_message", "")).strip()
    if assistant_text:
        sections.append(assistant_text)

    parse_errors = report.get("parse_errors", []) or []
    normalize_errors = report.get("normalize_errors", []) or []
    if parse_errors or normalize_errors:
        sections.append("[parse-errors]\n" + "\n".join([*parse_errors, *normalize_errors]))

    summary = _format_tool_result_summary(
        report.get("tool_results", []) or [],
        include_raw=bool(state.get("show_raw_results", False)),
    )
    if summary:
        sections.append(summary)

    if not sections:
        if bool(state.get("show_raw_results", False)):
            sections.append(_truncate_text(_pretty_json(report), 2400))
        else:
            sections.append("[no-output] command center returned no assistant text or tool results.")
    return "\n\n".join(sections).strip()


def _run_chat_turn(state: dict, user_text: str) -> str:
    prompt = _build_terminal_prompt(state.get("history", []))
    system = (
        "You are KHANARY terminal assistant. "
        "Continue the conversation naturally and keep responses concise and actionable."
    )
    return chat(prompt, profile=state["profile"], system=system).strip()


def _save_tui_session(state: dict, path: Path) -> None:
    snapshot = {
        "@kind": "khanary.bot.tui.session.v1",
        "saved_at": _now_iso(),
        "session_id": state.get("session_id"),
        "profile": state.get("profile"),
        "mode": state.get("mode"),
        "execute_tools": bool(state.get("execute_tools", True)),
        "dry_run": bool(state.get("dry_run", False)),
        "show_raw_results": bool(state.get("show_raw_results", False)),
        "history": state.get("history", []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pretty_json(snapshot), encoding="utf-8")


def _load_tui_session(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid session file (expected object).")
    if data.get("@kind") != "khanary.bot.tui.session.v1":
        raise ValueError("Unsupported session file kind.")
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    return {
        "session_id": data.get("session_id") or str(uuid4()),
        "profile": data.get("profile") or "advisor",
        "mode": data.get("mode") or "center",
        "execute_tools": bool(data.get("execute_tools", True)),
        "dry_run": bool(data.get("dry_run", False)),
        "show_raw_results": bool(data.get("show_raw_results", False)),
        "history": history,
    }


def run_tui(
    *,
    profile: Optional[str] = None,
    mode: str = "center",
    execute_tools: bool = True,
    dry_run: bool = False,
    show_raw_results: bool = False,
    session_file: Path = DEFAULT_TUI_SESSION,
    resume: bool = False,
    auto_launch_khanary: bool = True,
) -> None:
    cfg = load_config()

    base_profile = profile or _center_cfg(cfg).get("profile", "advisor")
    if base_profile not in cfg or not isinstance(cfg.get(base_profile), dict):
        base_profile = "advisor"

    state: dict[str, Any] = {
        "session_id": str(uuid4()),
        "profile": base_profile,
        "mode": mode if mode in {"center", "chat"} else "center",
        "execute_tools": execute_tools,
        "dry_run": dry_run,
        "show_raw_results": show_raw_results,
        "auto_launch_khanary": bool(auto_launch_khanary),
        "khanary_launch_attempted": False,
        "history": [],
        "session_file": session_file,
        "cfg": cfg,
        "khanary_exe": str(DEFAULT_KHANARY_EXE),
        "khanary_cwd": str(DEFAULT_KHANARY_CWD),
    }

    if resume and session_file.exists():
        try:
            loaded = _load_tui_session(session_file)
            state.update(loaded)
            state["session_file"] = session_file
            _append_history(
                state,
                "assistant",
                f"[session] resumed: {session_file} ({len(state.get('history', []))} entries)",
                ui_only=True,
                command="resume",
            )
        except Exception as exc:
            _append_history(state, "assistant", f"[session] resume failed: {exc}", ui_only=True, error=True)

    help_text = textwrap.dedent(
        """
        Slash commands:
          /help                         Show this help
          /model [advisor|chat]         Show or switch model profile
          /mode [center|chat]           Switch execution mode
          /tools                        Show allowlist + runtime verbs
          /mcp <tool> [json]            Call MCP tool directly
          /rpc <method> [json]          Call RPC method directly
          /memory <note|recall|list>    Append note or recall memory entries
          /load <path> [prompt]         Load TS/markup/text and send as chat input
          /save [path]                  Save session feed
          /resume [path]                Resume session feed
          /new                          Clear session feed
          /fork [path]                  Save to new session file and continue there
          /raw [on|off]                 Toggle raw tool responses in feed
          /launch [force]               Launch KHANARY.exe for local RPC/MCP surfaces
          /login [token]                Local login placeholder (no remote auth needed)
          /quit                         Exit TUI

        Runtime slash actions:
          /<verb> [query]               Execute mapped runtime verb from command_center.verbs
          /<skill-name> [query]         Fallback to key_query using '@<skill-name> ...'

        Multiline input:
          Windows interactive mode: Shift+Enter inserts a newline, Enter sends.
          Compatibility fallback: end a line with '\\' then press Enter to continue.
        """
    ).strip()

    while True:
        input_height = _render_tui_screen(state)
        try:
            line = _read_tui_input(input_height=input_height)
        except EOFError:
            print("\n[exit] EOF")
            break
        except KeyboardInterrupt:
            print("\n[exit] interrupted")
            break

        if line is None:
            continue
        text = line.strip()
        if not text:
            continue

        if text.startswith("/"):
            _append_history(state, "user", text, ui_only=True, command="slash")
            body = text[1:].strip()
            parts = body.split(maxsplit=1)
            cmd = (parts[0].strip().lower() if parts else "")
            rest = (parts[1].strip() if len(parts) > 1 else "")

            if cmd in {"quit", "exit"}:
                print("[exit] bye")
                break

            if cmd == "help":
                _append_history(state, "assistant", help_text, ui_only=True, command="help")
                continue

            if cmd == "model":
                if not rest:
                    _append_history(state, "assistant", f"[model] current={state['profile']}", ui_only=True)
                    continue
                p = rest.strip().lower()
                if p in cfg and isinstance(cfg.get(p), dict):
                    state["profile"] = p
                    _append_history(state, "assistant", f"[model] switched to {p}", ui_only=True)
                else:
                    _append_history(state, "assistant", f"[model] unknown profile: {p}", ui_only=True, error=True)
                continue

            if cmd == "mode":
                if not rest:
                    _append_history(state, "assistant", f"[mode] current={state['mode']}", ui_only=True)
                    continue
                m = rest.strip().lower()
                if m in {"center", "chat"}:
                    state["mode"] = m
                    _append_history(state, "assistant", f"[mode] switched to {m}", ui_only=True)
                else:
                    _append_history(state, "assistant", f"[mode] unsupported mode: {m}", ui_only=True, error=True)
                continue

            if cmd == "tools":
                _append_history(state, "assistant", _tools_overview_text(cfg), ui_only=True, command="tools")
                continue

            if cmd == "raw":
                if not rest:
                    state["show_raw_results"] = not bool(state.get("show_raw_results", False))
                else:
                    state["show_raw_results"] = rest.lower() in {"1", "true", "on", "yes", "y"}
                _append_history(
                    state,
                    "assistant",
                    f"[raw] show_raw_results={state['show_raw_results']}",
                    ui_only=True,
                    command="raw",
                )
                continue

            if cmd == "new":
                state["history"] = []
                state["session_id"] = str(uuid4())
                _append_history(
                    state,
                    "assistant",
                    f"[session] reset, new session_id={state['session_id']}",
                    ui_only=True,
                    command="new",
                )
                continue

            if cmd == "save":
                target = Path(rest).expanduser() if rest else Path(state["session_file"])
                if not target.is_absolute():
                    target = Path.cwd() / target
                try:
                    _save_tui_session(state, target)
                    state["session_file"] = target
                    _append_history(state, "assistant", f"[session] saved: {target}", ui_only=True, command="save")
                except Exception as exc:
                    _append_history(state, "assistant", f"[session] save failed: {exc}", ui_only=True, error=True)
                continue

            if cmd == "resume":
                target = Path(rest).expanduser() if rest else Path(state["session_file"])
                if not target.is_absolute():
                    target = Path.cwd() / target
                if not target.exists():
                    _append_history(state, "assistant", f"[session] not found: {target}", ui_only=True, error=True)
                    continue
                try:
                    loaded = _load_tui_session(target)
                    state.update(loaded)
                    state["session_file"] = target
                    _append_history(
                        state,
                        "assistant",
                        f"[session] resumed: {target} ({len(state.get('history', []))} entries)",
                        ui_only=True,
                        command="resume",
                    )
                except Exception as exc:
                    _append_history(state, "assistant", f"[session] resume failed: {exc}", ui_only=True, error=True)
                continue

            if cmd == "fork":
                if rest:
                    target = Path(rest).expanduser()
                else:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    target = Path.cwd() / f"khanary-tui-fork-{stamp}.json"
                if not target.is_absolute():
                    target = Path.cwd() / target
                try:
                    _save_tui_session(state, target)
                    state["session_file"] = target
                    state["session_id"] = str(uuid4())
                    _append_history(state, "assistant", f"[session] forked to: {target}", ui_only=True, command="fork")
                except Exception as exc:
                    _append_history(state, "assistant", f"[session] fork failed: {exc}", ui_only=True, error=True)
                continue

            if cmd == "launch":
                force = rest.strip().lower() in {"force", "restart", "again"}
                started, launch_msg = _start_khanary_runtime(
                    state,
                    reason="manual /launch",
                    force=force,
                )
                _append_history(
                    state,
                    "assistant",
                    launch_msg,
                    ui_only=True,
                    command="launch",
                    ok=bool(started),
                )
                continue

            if cmd == "login":
                if rest:
                    state["login_token"] = rest
                    _append_history(
                        state,
                        "assistant",
                        "[login] token stored in current TUI session (local use).",
                        ui_only=True,
                        command="login",
                    )
                else:
                    _append_history(
                        state,
                        "assistant",
                        "[login] local endpoints use api_key='local'; explicit login is optional.",
                        ui_only=True,
                        command="login",
                    )
                continue

            if cmd == "mcp":
                try:
                    tool, params = _parse_method_and_params(rest)
                    result = mcp_call(tool, params, base_url=_base_url(cfg, "mcp"))
                    rendered = _render_structured_payload(
                        result,
                        include_raw=bool(state.get("show_raw_results", False)),
                    )
                    _append_history(state, "assistant", rendered, command="mcp", method=tool)
                except Exception as exc:
                    _append_history(state, "assistant", f"[mcp] error: {exc}", error=True)
                continue

            if cmd == "rpc":
                try:
                    method, params = _parse_method_and_params(rest)
                    result = rpc_call(method, params, base_url=_base_url(cfg, "rpc"))
                    rendered = _render_structured_payload(
                        result,
                        include_raw=bool(state.get("show_raw_results", False)),
                    )
                    _append_history(state, "assistant", rendered, command="rpc", method=method)
                except Exception as exc:
                    _append_history(state, "assistant", f"[rpc] error: {exc}", error=True)
                continue

            if cmd == "memory":
                mem_arg = rest.strip()
                if not mem_arg:
                    _append_history(
                        state,
                        "assistant",
                        "[memory] usage: /memory <note|recall|list>",
                        ui_only=True,
                        error=True,
                    )
                    continue
                mem_l = mem_arg.lower()
                if mem_l in {"recall", "list"}:
                    query = "@memory recall"
                elif mem_l.startswith("recall "):
                    query = "@memory recall"
                else:
                    query = f"memory: {mem_arg}"
                try:
                    result = rpc_call("key_query", {"query": query}, base_url=_base_url(cfg, "rpc"))
                    rendered = _render_structured_payload(
                        result,
                        include_raw=bool(state.get("show_raw_results", False)),
                    )
                    _append_history(state, "assistant", rendered, command="memory", query=query)
                except Exception as exc:
                    _append_history(state, "assistant", f"[memory] error: {exc}", error=True)
                continue

            if cmd == "load":
                try:
                    argv = shlex.split(rest)
                except Exception as exc:
                    _append_history(state, "assistant", f"[load] parse error: {exc}", ui_only=True, error=True)
                    continue
                if not argv:
                    _append_history(
                        state,
                        "assistant",
                        "[load] usage: /load <path> [prompt-text]",
                        ui_only=True,
                        error=True,
                    )
                    continue
                file_path = argv[0]
                prompt_prefix = " ".join(argv[1:]).strip()
                ok, block = _load_file_as_prompt_block(file_path)
                if not ok:
                    _append_history(state, "assistant", f"[load] {block}", ui_only=True, error=True)
                    continue
                user_turn = block if not prompt_prefix else f"{prompt_prefix}\n\n{block}"
                _append_history(
                    state,
                    "assistant",
                    f"[load] sending file context from {file_path}",
                    ui_only=True,
                    command="load",
                )
                _append_history(state, "user", user_turn, command="load", path=file_path)
                try:
                    bot_reply = _run_center_turn(state, user_turn) if state["mode"] == "center" else _run_chat_turn(state, user_turn)
                    _append_history(state, "assistant", bot_reply)
                except Exception as exc:
                    err = f"[error] {exc}"
                    _append_history(state, "assistant", err, error=True)
                continue

            # Runtime verb actions and generic /<skill-name> fallback.
            cc_verbs = _center_cfg(cfg).get("verbs", {})
            if cmd in cc_verbs:
                try:
                    call = _normalize_center_call({"verb": cmd, "query": rest}, cfg)
                    dry = bool(state.get("dry_run", False)) or not bool(state.get("execute_tools", True))
                    result = _execute_center_call(call, cfg, dry_run=dry)
                    rendered = _render_center_call_result(
                        result,
                        include_raw=bool(state.get("show_raw_results", False)),
                    )
                    _append_history(state, "assistant", rendered, command=f"/{cmd}", verb=cmd)
                except Exception as exc:
                    _append_history(state, "assistant", f"[/{cmd}] error: {exc}", error=True)
                continue

            # Generic /<skill-name> -> key_query runtime marker.
            query = f"@{cmd}" + (f" {rest}" if rest else "")
            try:
                result = rpc_call("key_query", {"query": query}, base_url=_base_url(cfg, "rpc"))
                rendered = _render_structured_payload(
                    result,
                    include_raw=bool(state.get("show_raw_results", False)),
                )
                _append_history(state, "assistant", rendered, command="skill-fallback", skill=cmd)
            except Exception as exc:
                _append_history(state, "assistant", f"[skill:{cmd}] error: {exc}", error=True)
            continue

        # Normal chat turn (free-form, unrestricted input)
        _append_history(state, "user", text)
        try:
            bot_reply = _run_center_turn(state, text) if state["mode"] == "center" else _run_chat_turn(state, text)
            _append_history(state, "assistant", bot_reply)
        except Exception as exc:
            err = f"[error] {exc}"
            _append_history(state, "assistant", err, error=True)


def main():
    parser = argparse.ArgumentParser(description="KHANARY.CPP bot")
    sub = parser.add_subparsers(dest="cmd")

    ask = sub.add_parser("ask", help="Ask advisor a question")
    ask.add_argument("prompt", nargs="+")
    ask.add_argument("--profile", default="advisor", choices=["advisor", "chat"])

    sub.add_parser("tasks", help="Get next task list from HeavyAdvisor")

    mcp = sub.add_parser("mcp", help="Call a KHANARY MCP tool")
    mcp.add_argument("tool")
    mcp.add_argument("params", nargs="?", default="{}")
    mcp.add_argument("--base-url", default=None)

    rpc = sub.add_parser("rpc", help="Call a KHANARY JSON-RPC method")
    rpc.add_argument("method")
    rpc.add_argument("params", nargs="?", default="{}")
    rpc.add_argument("--base-url", default=None)

    center = sub.add_parser("center", help="Command center: parse and execute <tools>{tool_calls}</tools>")
    center.add_argument("prompt", nargs="+")
    center.add_argument("--profile", default=None, choices=["advisor", "chat"])
    center.add_argument("--dry-run", action="store_true", help="Plan/validate tool calls without executing")
    center.add_argument("--no-exec", action="store_true", help="Do not execute parsed tool calls")
    center.add_argument("--raw", action="store_true", help="Include raw model response in JSON output")

    tui = sub.add_parser("tui", help="Interactive terminal chat TUI with slash commands")
    tui.add_argument("--profile", default=None, choices=["advisor", "chat"])
    tui.add_argument("--mode", default="center", choices=["center", "chat"], help="center=tool-aware routing, chat=plain chat")
    tui.add_argument("--dry-run", action="store_true", help="Do not execute command-center tool calls")
    tui.add_argument("--no-exec", action="store_true", help="Disable tool execution in center mode")
    tui.add_argument("--raw-results", action="store_true", help="Show raw tool responses in terminal feed")
    tui.add_argument("--session-file", default=str(DEFAULT_TUI_SESSION), help="Session snapshot file path")
    tui.add_argument("--resume", action="store_true", help="Resume from --session-file if it exists")
    tui.add_argument(
        "--no-auto-launch-khanary",
        action="store_true",
        help="Do not auto-launch KHANARY.exe when RPC fallback is offline",
    )

    kh_flags = sub.add_parser("khanary-flags", help="List KHANARY.exe command-line flags")
    kh_flags.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    kh_run = sub.add_parser("khanary-run", help="Launch KHANARY.exe with passthrough --flags")
    kh_run.add_argument("--exe", default=str(DEFAULT_KHANARY_EXE), help="Path to KHANARY.exe")
    kh_run.add_argument("--cwd", default=str(DEFAULT_KHANARY_CWD), help="Working directory for launch")
    kh_run.add_argument("--wait", action="store_true", help="Wait for process exit and return exit code")
    kh_run.add_argument("--dry-run", action="store_true", help="Print command JSON without launching")
    kh_run.add_argument(
        "khanary_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to KHANARY.exe (use '-- --data-dir ... --verbose')",
    )

    args = parser.parse_args()

    if args.cmd == "ask":
        print(chat(" ".join(args.prompt), profile=args.profile))
    elif args.cmd == "tasks":
        print(task_outline())
    elif args.cmd == "mcp":
        cfg = load_config()
        base = args.base_url or _base_url(cfg, "mcp")
        result = mcp_call(args.tool, json.loads(args.params), base_url=base)
        print(json.dumps(result, indent=2))
    elif args.cmd == "rpc":
        cfg = load_config()
        base = args.base_url or _base_url(cfg, "rpc")
        result = rpc_call(args.method, json.loads(args.params), base_url=base)
        print(json.dumps(result, indent=2))
    elif args.cmd == "center":
        report = command_center(
            " ".join(args.prompt),
            profile=args.profile,
            dry_run=args.dry_run,
            execute=not args.no_exec,
        )
        if report.get("assistant_message"):
            print(report["assistant_message"])
            print()
        if not args.raw:
            report = dict(report)
            report.pop("raw_response", None)
        print(json.dumps(report, indent=2))
    elif args.cmd == "tui":
        run_tui(
            profile=args.profile,
            mode=args.mode,
            execute_tools=not args.no_exec,
            dry_run=bool(args.dry_run),
            show_raw_results=bool(args.raw_results),
            session_file=Path(args.session_file).expanduser(),
            resume=bool(args.resume),
            auto_launch_khanary=not bool(args.no_auto_launch_khanary),
        )
    elif args.cmd == "khanary-flags":
        flags = khanary_flag_catalog()
        if args.json:
            print(json.dumps({"ok": True, "flags": flags}, indent=2))
        else:
            print("KHANARY.exe supported flags:")
            for item in flags:
                print(f"  {item['flag']:<20} {item['description']}")
    elif args.cmd == "khanary-run":
        report = run_khanary(
            Path(args.exe),
            list(args.khanary_args),
            cwd=Path(args.cwd) if args.cwd else None,
            wait=bool(args.wait),
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(report, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
