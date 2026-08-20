"""A tiny OpenAI-compatible (vLLM-like) HTTP server for tests."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockVLLM/0.1"

    def log_message(self, *args):  # silence request logging
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._check_required_header()
        if getattr(self, "_header_error", False):
            return
        if self.path.rstrip("/").endswith("/models"):
            self._send_json({"object": "list", "data": [{"id": "mock-model"}]})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        self._check_required_header()
        if getattr(self, "_header_error", False):
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.request_bodies.append((self.path, body))
        if self.path.rstrip("/").endswith("/tokenize"):
            prompt = body.get("prompt", "")
            self._send_json({"tokens": list(range(len(prompt.split())))})
            return
        if self.path.rstrip("/").endswith("/chat/completions"):
            max_tokens = int(body.get("max_completion_tokens", body.get("max_tokens", 8)))
            reasoning = int(getattr(self.server, "reasoning_tokens", 0))
            delay = float(getattr(self.server, "token_delay", 0.002))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(max_tokens):
                if i < reasoning:
                    delta = {"reasoning_content": f"thk{i} "}
                else:
                    delta = {"content": f"tok{i} "}
                choice = {"index": 0, "delta": delta}
                if getattr(self.server, "token_ids", True):
                    # vLLM's return_token_ids extension: real token IDs per chunk
                    choice["token_ids"] = [100 + i]
                chunk = {
                    "id": "cmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "mock-model",
                    "choices": [choice],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                if delay:
                    time.sleep(delay)
            usage_chunk = {
                "id": "cmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "mock-model",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": max_tokens},
            }
            self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._send_json({"error": "not found"}, 404)

    def _check_required_header(self):
        required_header = getattr(self.server, "required_header", None)
        self._header_error = bool(
            required_header
            and self.headers.get(required_header[0]) != required_header[1]
        )
        if self._header_error:
            self._send_json({"error": "missing required header"}, 401)


class MockVLLMServer:
    """ThreadingHTTPServer speaking just enough of the vLLM/OpenAI API."""

    def __init__(
        self,
        token_delay: float = 0.001,
        reasoning_tokens: int = 0,
        token_ids: bool = True,
        required_header: tuple[str, str] | None = None,
    ):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.token_delay = token_delay
        self.server.reasoning_tokens = reasoning_tokens
        self.server.token_ids = token_ids
        self.server.required_header = required_header
        self.server.request_bodies = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def start(self) -> "MockVLLMServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
