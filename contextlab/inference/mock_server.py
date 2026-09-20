"""Small OpenAI-compatible HTTP server for CPU-only integration tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockHandler(BaseHTTPRequestHandler):
    server_version = "ContextLabMock/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._json(
                200, {"object": "list", "data": [{"id": "contextlab-mock", "object": "model"}]}
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "invalid JSON"}})
            return
        content = json.dumps({"action": "finish", "summary": "mock endpoint response"})
        prompt_tokens = sum(
            max(1, len(str(message.get("content", "")).encode()) // 4)
            for message in request.get("messages", [])
        )
        completion_tokens = max(1, len(content.encode()) // 4)
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            midpoint = len(content) // 2
            chunks: list[dict[str, Any]] = [
                {
                    "id": "mock",
                    "model": "contextlab-mock",
                    "choices": [{"delta": {"content": piece}, "index": 0}],
                }
                for piece in (content[:midpoint], content[midpoint:])
            ] + [
                {
                    "id": "mock",
                    "model": "contextlab-mock",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(
            200,
            {
                "id": "mock",
                "model": "contextlab-mock",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve(host: str = "127.0.0.1", port: int = 8010) -> None:
    ThreadingHTTPServer((host, port), MockHandler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
