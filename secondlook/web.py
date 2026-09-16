"""Read-only operator HTTP interface. There are deliberately no motion routes."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .runtime import InspectionRuntime


def make_server(runtime: InspectionRuntime, host: str = "127.0.0.1",
                port: int = 8088) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Operator app must bind to loopback; use an SSH tunnel")
    static = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, code: int, body: bytes, mime: str, observation_id: str | None = None):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if observation_id:
                self.send_header("X-Observation-Id", observation_id)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            request = urlparse(self.path)
            path = request.path
            if path in ("/api/status", "/api/health"):
                status = runtime.status()
                if path == "/api/health":
                    status = {"service": "ok", "camera": status["camera"]["status"],
                              "detector": status["detector"],
                              "controller": "DISARMED", "task_ready": False}
                self.send(200, json.dumps(status, allow_nan=False).encode(), "application/json")
            elif path in ("/api/frame/latest.jpg", "/api/detector.jpg"):
                expected_id = parse_qs(request.query).get("observation_id", [None])[0]
                result = runtime.frame_jpeg() if path == "/api/frame/latest.jpg" else runtime.detector_jpeg(expected_id)
                if result:
                    self.send(200, result[0], "image/jpeg", result[1])
                else:
                    self.send(503, b'{"error":"No fresh observation available"}', "application/json")
            elif path in ("/", "/app.js", "/style.css"):
                name, mime = {"/": ("index.html", "text/html; charset=utf-8"),
                              "/app.js": ("app.js", "text/javascript"),
                              "/style.css": ("style.css", "text/css")}[path]
                self.send(200, (static / name).read_bytes(), mime)
            else:
                self.send(404, b'{"error":"Not found"}', "application/json")

    return ThreadingHTTPServer((host, port), Handler)
