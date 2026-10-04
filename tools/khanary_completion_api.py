"""OpenAI-compatible single completion API for KHANARY.

This gateway keeps KHANARY as the primary responder (key_query) and optionally
adds model-server contributions (chat-mu / tool-mu) when the active phase allows.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

PUBLIC_API_BASE_URL = "https://api.kuhul.dev"
PUBLIC_API_PORT = 443
LEARNING_COMMENCED_STATUS = "learning_commenced"
PUBLIC_RPC_METHODS = {
    "key_query",
    "resolve_manifest",
    "fold_query",
    "fold_graph_parse",
    "micronaut_dispatch",
    "micronaut_transpile",
    "journal_semantic_envelope",
    "journal_hyperbolic_event",
    "tensor_register",
    "tensor_info",
    "tensor_list",
    "tensor_stats",
    "tensor_gemm",
    "list_keys",
    "record_key",
    "infer",
}


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return ""


def _latest_user_prompt(messages: list[dict[str, Any]]) -> str:
    for row in reversed(messages):
        if not isinstance(row, dict):
            continue
        if row.get("role") != "user":
            continue
        prompt = _as_text(row.get("content"))
        if prompt.strip():
            return prompt.strip()
    return ""


def _extract_assistant_text(obj: dict[str, Any]) -> str:
    if not isinstance(obj, dict):
        return ""
    for key in ("response", "text", "answer", "message"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _json_request(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    req = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("utf-8") + body

    with socket.create_connection((host, port), timeout=timeout_s) as conn:
        conn.sendall(req)
        chunks: list[bytes] = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)

    raw_http = b"".join(chunks).decode("utf-8", errors="replace")
    head, sep, tail = raw_http.partition("\r\n\r\n")
    if not sep:
        raise RuntimeError("invalid HTTP response")

    status_line = head.splitlines()[0] if head else ""
    parts = status_line.split(" ", 2)
    status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
    if status >= 400:
        raise RuntimeError(f"upstream HTTP {status}: {tail[:240]}")

    data = json.loads(tail)
    if not isinstance(data, dict):
        raise RuntimeError("non-object JSON response")
    return data


@dataclass(frozen=True)
class GatewayConfig:
    host: str
    port: int
    rpc_port: int
    chat_port: int
    tool_port: int
    timeout_s: float
    model_id: str
    public_base_url: str
    public_port: int


class CompletionGateway:
    def __init__(self, cfg: GatewayConfig) -> None:
        self.cfg = cfg

    def service_info(self) -> dict[str, Any]:
        return {
            "service": "KHANARY public API gateway",
            "status": "ready",
            "public_base_url": self.cfg.public_base_url,
            "public_port": self.cfg.public_port,
            "public_endpoint": f"{self.cfg.public_base_url}:{self.cfg.public_port}",
            "local_gateway": f"http://{self.cfg.host}:{self.cfg.port}",
            "endpoints": [
                {"method": "GET", "path": "/"},
                {"method": "GET", "path": "/v1"},
                {"method": "GET", "path": "/v1/models"},
                {"method": "POST", "path": "/v1/chat/completions"},
                {"method": "POST", "path": "/v1/code/modify"},
                {"method": "POST", "path": "/v1/images/generations"},
                {"method": "POST", "path": "/v1/learn/events"},
                {"method": "POST", "path": "/rpc"},
            ],
            "authority_boundary": {
                "completion_is_authority": False,
                "code_patch_is_commit": False,
                "image_generation_is_semantic_truth": False,
                "learning_event_is_topology_mutation": False,
                "semantic_authority": False,
                "commit_path": "Ch'en/XCFE/Xul",
                "public_api_port": 443,
            },
        }

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        return _json_request(
            f"http://127.0.0.1:{self.cfg.rpc_port}/rpc",
            payload,
            self.cfg.timeout_s,
        )

    def _try_rpc(self, method: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
        try:
            return self._rpc(method, params), "recorded"
        except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
            return {"error": str(exc), "method": method}, "pending"

    def _event_id(self, prefix: str) -> str:
        return f"{prefix}-khanary-{int(time.time() * 1000)}"

    def learning_event(
        self,
        request_body: dict[str, Any],
        *,
        source: str = "api_gateway",
    ) -> dict[str, Any]:
        event_name = str(
            request_body.get("event")
            or request_body.get("@event")
            or request_body.get("type")
            or "api.learning"
        )
        payload = request_body.get("payload")
        if payload is None:
            payload = request_body.get("data")
        if payload is None:
            payload = {}

        event = {
            "@type": "khanary.api_gateway.learning_event.v1",
            "@event": event_name,
            "@source": source,
            "@status": LEARNING_COMMENCED_STATUS,
            "@semantic_authority": False,
            "@topology_mutation": False,
            "@commit_authorized": False,
            "@next_gate": "Ch'en/XCFE/Xul",
            "@payload": payload,
        }
        journal_payload = {
            "epistemic_type": "observation",
            "source": source,
            "semantic_authority": False,
            "authority_boundary": "learning event is evidence, not topology mutation",
            "event": event,
        }
        journal, journal_state = self._try_rpc("journal_semantic_envelope", journal_payload)
        return {
            "id": self._event_id("learn"),
            "object": "learning.event",
            "status": LEARNING_COMMENCED_STATUS,
            "semantic_authority": False,
            "topology_mutation": False,
            "commit_authorized": False,
            "next_gate": "Ch'en/XCFE/Xul",
            "event": event,
            "journal": {
                "attempted": True,
                "state": journal_state,
                "result": journal,
            },
        }

    def code_modify(self, request_body: dict[str, Any]) -> dict[str, Any]:
        instruction = _as_text(
            request_body.get("instructions")
            or request_body.get("input")
            or request_body.get("prompt")
            or request_body.get("query")
        ).strip()
        if not instruction:
            raise ValueError("instructions/input/prompt/query is required")

        files = request_body.get("files", [])
        if files is None:
            files = []
        if not isinstance(files, list):
            raise ValueError("files must be an array when provided")

        rpc_payload = {
            "query": f"CODE_MODIFY: {instruction}",
            "surface": "api_gateway.code_modify",
            "files": files,
            "semantic_authority": False,
        }
        route_result, route_state = self._try_rpc("key_query", rpc_payload)
        learning = self.learning_event(
            {
                "event": "code.modify.requested",
                "payload": {
                    "instruction_chars": len(instruction),
                    "file_count": len(files),
                    "route_state": route_state,
                },
            },
            source="api_gateway.code_modify",
        )
        return {
            "id": self._event_id("codemod"),
            "object": "code.modification.proposal",
            "status": "proposal_ready" if route_state == "recorded" else "proposal_pending",
            "applied": False,
            "commit_authorized": False,
            "semantic_authority": False,
            "authority_boundary": "code modification endpoint returns a proposal envelope; it does not apply or commit patches",
            "next_gate": "Ch'en/XCFE/Xul",
            "route": {
                "method": "key_query",
                "state": route_state,
                "result": route_result,
            },
            "learning": {
                "status": learning["status"],
                "event_id": learning["id"],
                "journal_state": learning["journal"]["state"],
            },
        }

    def image_generation(self, request_body: dict[str, Any]) -> dict[str, Any]:
        prompt = _as_text(request_body.get("prompt") or request_body.get("input") or request_body.get("query")).strip()
        if not prompt:
            raise ValueError("prompt/input/query is required")
        n_value = request_body.get("n", 1)
        try:
            n = max(1, min(16, int(n_value)))
        except (TypeError, ValueError):
            raise ValueError("n must be an integer") from None
        size = str(request_body.get("size") or "1024x1024")

        route_result, route_state = self._try_rpc(
            "key_query",
            {
                "query": f"IMAGE_GENERATION: {prompt}",
                "surface": "api_gateway.images",
                "n": n,
                "size": size,
                "semantic_authority": False,
            },
        )
        learning = self.learning_event(
            {
                "event": "image.generation.requested",
                "payload": {
                    "prompt_chars": len(prompt),
                    "n": n,
                    "size": size,
                    "route_state": route_state,
                },
            },
            source="api_gateway.images",
        )
        return {
            "id": self._event_id("image"),
            "object": "image.generation.request",
            "created": int(time.time()),
            "status": "request_routed" if route_state == "recorded" else "backend_required",
            "data": [],
            "semantic_authority": False,
            "authority_boundary": "image generation output is media, not semantic truth",
            "next_gate": "Ch'en/XCFE/Xul",
            "request": {
                "prompt": prompt,
                "n": n,
                "size": size,
            },
            "route": {
                "method": "key_query",
                "state": route_state,
                "result": route_result,
            },
            "learning": {
                "status": learning["status"],
                "event_id": learning["id"],
                "journal_state": learning["journal"]["state"],
            },
        }

    def rpc_bridge(self, request_body: dict[str, Any]) -> dict[str, Any]:
        if request_body.get("jsonrpc") != "2.0":
            raise ValueError("jsonrpc must be '2.0'")
        method = request_body.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("method is required")
        if method not in PUBLIC_RPC_METHODS:
            raise ValueError(f"rpc method not public allowlist: {method}")
        params = request_body.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object when provided")
        return self._rpc(method, params)

    def _query_model_server(
        self,
        port: int,
        messages: list[dict[str, Any]],
        khanary_hint: str,
        preferred_model: str,
    ) -> tuple[str, str]:
        if port <= 0:
            return "", "disabled"
        model_messages = list(messages[-20:])
        if khanary_hint:
            model_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a phase-constrained KHANARY contributor. "
                        "Add only concise missing detail and do not override authoritative KHANARY output."
                    ),
                },
                {"role": "system", "content": f"KHANARY draft: {khanary_hint}"},
            ] + model_messages

        payload = {
            "model": preferred_model or "local",
            "messages": model_messages,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 256,
        }
        try:
            row = _json_request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                payload,
                self.cfg.timeout_s,
            )
        except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
            return "", f"error:{exc}"

        choices = row.get("choices")
        if not isinstance(choices, list) or not choices:
            return "", "no_choices"
        first = choices[0]
        if not isinstance(first, dict):
            return "", "invalid_choice"
        msg = first.get("message")
        if not isinstance(msg, dict):
            return "", "missing_message"
        text = _as_text(msg.get("content")).strip()
        return text, "ok" if text else "empty"

    def _phase_allows_contribution(self, phase: str) -> bool:
        phase_name = phase.split(".", 1)[0]
        return phase_name in {"Pop", "Wo", "Yax", "Sek"}

    def _prefer_tool_server(self, phase: str, governing_track: str) -> bool:
        phase_name = phase.split(".", 1)[0]
        track = governing_track.lower()
        if phase_name == "Sek":
            return True
        return any(token in track for token in ("tool", "infer", "math", "code", "research"))

    def complete(self, request_body: dict[str, Any]) -> dict[str, Any]:
        messages = request_body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be an array")
        prompt = _latest_user_prompt(messages)
        if not prompt:
            raise ValueError("no user message content in request")

        rpc_env = self._rpc("key_query", {"query": prompt})
        rpc_result = rpc_env.get("result")
        if not isinstance(rpc_result, dict):
            raise RuntimeError("KHANARY key_query returned no result object")

        khanary_text = _extract_assistant_text(rpc_result)
        status = str(rpc_result.get("status", ""))
        phase = str(rpc_result.get("response_phase", "Wo.key_query"))
        track = str(rpc_result.get("governing_track", "Mx2LM"))
        activation = rpc_result.get("activation")
        activation_required = isinstance(activation, dict) and bool(activation.get("required"))

        contribution_text = ""
        contribution_source = "none"
        contribution_state = "skipped"
        contribution_needed = status in {"no_fact_found", "web_research_dispatched", "research_plan_ready"}
        if self._phase_allows_contribution(phase) and (contribution_needed or activation_required or not khanary_text):
            use_tool = self._prefer_tool_server(phase, track)
            if use_tool:
                contribution_text, contribution_state = self._query_model_server(
                    self.cfg.tool_port,
                    messages,
                    khanary_text,
                    "tool-mu",
                )
                contribution_source = "tool_mu"
                if not contribution_text:
                    contribution_text, contribution_state = self._query_model_server(
                        self.cfg.chat_port,
                        messages,
                        khanary_text,
                        "chat-mu",
                    )
                    contribution_source = "chat_mu" if contribution_text else contribution_source
            else:
                contribution_text, contribution_state = self._query_model_server(
                    self.cfg.chat_port,
                    messages,
                    khanary_text,
                    "chat-mu",
                )
                contribution_source = "chat_mu"
                if not contribution_text:
                    contribution_text, contribution_state = self._query_model_server(
                        self.cfg.tool_port,
                        messages,
                        khanary_text,
                        "tool-mu",
                    )
                    contribution_source = "tool_mu" if contribution_text else contribution_source

        final_text = khanary_text.strip()
        if not final_text and contribution_text:
            final_text = contribution_text.strip()
        elif final_text and contribution_text and contribution_needed:
            final_text = f"{final_text}\n\n{contribution_text.strip()}"
        elif not final_text:
            final_text = "No response available."

        model = str(request_body.get("model") or self.cfg.model_id)
        created = int(time.time())
        completion_id = f"chatcmpl-khanary-{created}"
        response: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "khanary": {
                "status": status,
                "governing_track": track,
                "response_phase": phase,
                "activation_required": activation_required,
                "model_contribution": {
                    "used": bool(contribution_text),
                    "source": contribution_source,
                    "state": contribution_state,
                },
            },
        }
        learning = self.learning_event(
            {
                "event": "text.completion.requested",
                "payload": {
                    "completion_id": completion_id,
                    "model": model,
                    "status": status,
                    "governing_track": track,
                    "response_phase": phase,
                    "prompt_chars": len(prompt),
                    "model_contribution_used": bool(contribution_text),
                },
            },
            source="api_gateway.chat_completions",
        )
        response["khanary"]["learning"] = {
            "status": learning["status"],
            "event_id": learning["id"],
            "journal_state": learning["journal"]["state"],
        }
        return response


class CompletionHandler(BaseHTTPRequestHandler):
    gateway: CompletionGateway
    cfg: GatewayConfig

    def _send_json(self, status: int, payload: dict[str, Any], *, content_type: str = "application/json") -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def _send_stream(self, payload: dict[str, Any], text: str) -> None:
        chunk_id = payload["id"]
        model = payload["model"]
        created = payload["created"]

        first = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        }
        end = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        data = (
            f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/v1"}:
            self._send_json(200, self.gateway.service_info())
            return
        if path == "/v1/models":
            payload = {
                "object": "list",
                "data": [{"id": self.cfg.model_id, "object": "model", "owned_by": "khanary-local"}],
            }
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_length).decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            if path == "/v1/chat/completions":
                completion = self.gateway.complete(body)
                if bool(body.get("stream")):
                    msg = completion.get("choices", [{}])[0].get("message", {})
                    text = _as_text(msg.get("content")).strip()
                    self._send_stream(completion, text)
                else:
                    self._send_json(200, completion)
                return
            if path == "/v1/code/modify":
                self._send_json(200, self.gateway.code_modify(body))
                return
            if path == "/v1/images/generations":
                self._send_json(200, self.gateway.image_generation(body))
                return
            if path == "/v1/learn/events":
                self._send_json(200, self.gateway.learning_event(body))
                return
            if path == "/rpc":
                self._send_json(200, self.gateway.rpc_bridge(body))
                return
            self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
        except (OSError, RuntimeError, TimeoutError) as exc:
            self._send_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
        except Exception as exc:  # defensive hard-fail surface
            self._send_json(500, {"error": {"message": str(exc), "type": "server_error"}})

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Keep this service quiet; KHANARY launcher handles user-visible logs.
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17892)
    parser.add_argument("--rpc-port", type=int, default=8787)
    parser.add_argument("--chat-port", type=int, default=17889)
    parser.add_argument("--tool-port", type=int, default=17890)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--model-id", default="khanary-single-completion")
    parser.add_argument("--public-base-url", default=PUBLIC_API_BASE_URL)
    parser.add_argument("--public-port", type=int, default=PUBLIC_API_PORT)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = GatewayConfig(
        host=args.host,
        port=int(args.port),
        rpc_port=int(args.rpc_port),
        chat_port=int(args.chat_port),
        tool_port=int(args.tool_port),
        timeout_s=float(args.timeout),
        model_id=str(args.model_id),
        public_base_url=str(args.public_base_url),
        public_port=int(args.public_port),
    )
    gateway = CompletionGateway(cfg)

    handler_cls = CompletionHandler
    handler_cls.gateway = gateway
    handler_cls.cfg = cfg

    server = ThreadingHTTPServer((cfg.host, cfg.port), handler_cls)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": cfg.host,
                "port": cfg.port,
                "rpc_port": cfg.rpc_port,
                "chat_port": cfg.chat_port,
                "tool_port": cfg.tool_port,
                "model_id": cfg.model_id,
                "public_base_url": cfg.public_base_url,
                "public_port": cfg.public_port,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
