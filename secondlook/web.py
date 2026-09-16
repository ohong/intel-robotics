"""Operator HTTP interface. There are deliberately no motion routes.

The only POST routes are voice: speech sets an allowlisted task instruction,
and narration text becomes audio. Neither arms or moves the robot.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .runtime import InspectionRuntime
from .voice import AUDIO_TYPES, MAX_AUDIO_BYTES, MAX_SPEECH_CHARS, VoiceError


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

        def json_error(self, code: int, message: str):
            self.send(code, json.dumps({"error": message}).encode(), "application/json")

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/api/voice/command", "/api/voice/speak"):
                return self.json_error(404, "Not found")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self.json_error(411, "Content-Length required")
            limit = MAX_AUDIO_BYTES if path == "/api/voice/command" else 4096
            if not 0 < length <= limit:
                return self.json_error(413, "Request body empty or too large")
            body = self.rfile.read(length)
            mime = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            try:
                if path == "/api/voice/command":
                    if mime not in AUDIO_TYPES:
                        return self.json_error(415, "Unsupported audio type")
                    result = runtime.voice_command(body, mime)
                    return self.send(200, json.dumps(result).encode(), "application/json")
                try:
                    text = json.loads(body).get("text")
                except (ValueError, AttributeError):
                    text = None
                if not isinstance(text, str) or not text.strip() or len(text) > MAX_SPEECH_CHARS:
                    return self.json_error(400, "text must be 1-500 characters")
                audio, audio_mime = runtime.speak(text)
                self.send(200, audio, audio_mime)
            except VoiceError as error:
                self.json_error(502 if runtime.voice else 503, str(error))

    return ThreadingHTTPServer((host, port), Handler)
