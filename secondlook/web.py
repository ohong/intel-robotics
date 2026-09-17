"""Operator HTTP interface. There are deliberately no motion routes.

The only POST routes are voice: speech or a GPT-Live phrase sets an allowlisted
task instruction, narration text becomes audio, and a GPT-Live SDP offer is
exchanged for an answer. None of them arms or moves the robot.
"""
from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .live import MAX_SDP_BYTES, LiveClient, LiveError
from .replay import ReplayDataset
from .runtime import InspectionRuntime
from .studio_robot import StudioRobotSource
from .voice import AUDIO_TYPES, MAX_AUDIO_BYTES, MAX_SPEECH_CHARS, VoiceError


RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
VIDEO_ROUTE = re.compile(r"/api/replay/video/(\d+)/(\d+)/(\d+)$")
EPISODE_ROUTE = re.compile(r"/api/replay/episodes/(\d+)$")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                ".urdf": "application/xml", ".stl": "model/stl",
                ".json": "application/json"}


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range into inclusive bounds; None means the whole file."""
    match = RANGE.match(header or "")
    if not match or match.groups() == ("", ""):
        return None
    first, last = match.groups()
    if first == "":
        start, end = max(0, size - int(last)), size - 1
    else:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start > end or start >= size:
        raise ValueError("Unsatisfiable range")
    return start, end


def make_server(runtime: InspectionRuntime, host: str = "127.0.0.1",
                port: int = 8088, replay: ReplayDataset | None = None,
                robot: StudioRobotSource | None = None, live: LiveClient | None = None) -> ThreadingHTTPServer:
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
            elif path == "/api/evidence/recent":
                try:
                    limit = int(parse_qs(request.query).get("n", ["200"])[0])
                except ValueError:
                    return self.json_error(400, "n must be an integer")
                records = runtime.evidence.tail(min(max(limit, 1), 1000))
                self.send(200, json.dumps({"records": records}, allow_nan=False).encode(), "application/json")
            elif path.startswith("/api/replay/"):
                self.replay_get(path)
            elif path in ("/api/robot/latest", "/api/robot/stream"):
                if robot is None:
                    return self.json_error(404, "No Studio robot session configured")
                if path == "/api/robot/latest":
                    return self.send(200, json.dumps(robot.latest(), allow_nan=False).encode(), "application/json")
                self.robot_stream()
            elif path == "/api/live":
                body = live.status() if live else {"status": "UNAVAILABLE", "error": "OPENAI_API_KEY is not configured"}
                self.send(200, json.dumps(body).encode(), "application/json")
            elif path == "/mission":
                self.send(200, (static / "mission.html").read_bytes(), STATIC_TYPES[".html"])
            elif path.startswith("/static/"):
                self.static_get(path.removeprefix("/static/"))
            elif path in ("/", "/app.js", "/style.css"):
                name, mime = {"/": ("index.html", "text/html; charset=utf-8"),
                              "/app.js": ("app.js", "text/javascript"),
                              "/style.css": ("style.css", "text/css")}[path]
                self.send(200, (static / name).read_bytes(), mime)
            else:
                self.send(404, b'{"error":"Not found"}', "application/json")

        def static_get(self, relative: str):
            file = (static / relative).resolve()
            mime = STATIC_TYPES.get(file.suffix)
            if not mime or not file.is_relative_to(static.resolve()) or not file.is_file():
                return self.json_error(404, "Not found")
            if file.suffix == ".stl":
                return self.send_file(file, mime)  # large and pinned, so let the browser cache it
            self.send(200, file.read_bytes(), mime)

        def replay_get(self, path: str):
            if replay is None:
                return self.json_error(404, "No replay dataset configured")
            episode, video = EPISODE_ROUTE.match(path), VIDEO_ROUTE.match(path)
            try:
                if path == "/api/replay/episodes":
                    return self.send(200, json.dumps(replay.summary()).encode(), "application/json")
                if episode:
                    body = json.dumps(replay.episode(int(episode.group(1)))).encode()
                    return self.send(200, body, "application/json")
                if video:
                    return self.send_file(replay.video_file(*map(int, video.groups())), "video/mp4")
            except KeyError:
                pass
            self.json_error(404, "Not found")

        def robot_stream(self):
            # Server-sent events: each packet at most 30 Hz, and a status event at least once a second
            # so the page can mark the pose stale when telemetry stops.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            last_sequence, last_sent = None, 0.
            try:
                while True:
                    with robot.changed:
                        robot.changed.wait(1.)
                    time.sleep(max(0., 1 / 30 - (time.monotonic() - last_sent)))
                    state = robot.latest()
                    if state.get("sequence") == last_sequence and time.monotonic() - last_sent < 1:
                        continue
                    last_sequence, last_sent = state.get("sequence"), time.monotonic()
                    self.wfile.write(f"data: {json.dumps(state, allow_nan=False)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def send_file(self, file: Path, mime: str):
            size = file.stat().st_size
            try:
                bounds = byte_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                return self.end_headers()
            start, end = bounds or (0, size - 1)
            self.send_response(206 if bounds else 200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=3600")
            if bounds:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            remaining = end - start + 1
            try:
                with file.open("rb") as stream:
                    stream.seek(start)
                    while remaining:
                        chunk = stream.read(min(remaining, 1 << 20))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json_error(self, code: int, message: str):
            self.send(code, json.dumps({"error": message}).encode(), "application/json")

        def same_origin(self) -> bool:
            # Browsers send Origin on every POST; a page on another site cannot forge ours.
            return self.headers.get("Origin") == f"http://{self.headers.get('Host', '')}"

        def do_POST(self):
            path = urlparse(self.path).path
            limits = {"/api/voice/command": MAX_AUDIO_BYTES, "/api/voice/speak": 4096,
                      "/api/task/instruction": 1024, "/api/live/session": MAX_SDP_BYTES}
            if path not in limits:
                return self.json_error(404, "Not found")
            if path in ("/api/task/instruction", "/api/live/session") and not self.same_origin():
                return self.json_error(403, "Cross-origin request refused")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self.json_error(411, "Content-Length required")
            if not 0 < length <= limits[path]:
                return self.json_error(413, "Request body empty or too large")
            body = self.rfile.read(length)
            mime = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if path == "/api/live/session":
                return self.live_session(body, mime)
            try:
                if path == "/api/voice/command":
                    if mime not in AUDIO_TYPES:
                        return self.json_error(415, "Unsupported audio type")
                    result = runtime.voice_command(body, mime)
                    return self.send(200, json.dumps(result).encode(), "application/json")
                key = "phrase" if path == "/api/task/instruction" else "text"
                try:
                    text = json.loads(body).get(key)
                except (ValueError, AttributeError):
                    text = None
                if path == "/api/task/instruction":
                    if not isinstance(text, str) or not text.strip() or len(text) > 200:
                        return self.json_error(400, "phrase must be 1-200 characters")
                    result = runtime.instruction_command(text.strip(), {"source": "gpt-live"})
                    return self.send(200, json.dumps(result).encode(), "application/json")
                if not isinstance(text, str) or not text.strip() or len(text) > MAX_SPEECH_CHARS:
                    return self.json_error(400, "text must be 1-500 characters")
                audio, audio_mime = runtime.speak(text)
                self.send(200, audio, audio_mime)
            except VoiceError as error:
                self.json_error(502 if runtime.voice else 503, str(error))

        def live_session(self, body: bytes, mime: str):
            if live is None:
                return self.json_error(503, "GPT-Live unavailable: OPENAI_API_KEY is not configured")
            if mime != "application/sdp" or not body.startswith(b"v=0"):
                return self.json_error(400, "SDP offer required")
            try:
                answer = live.create_session(body.decode())
            except (LiveError, UnicodeError) as error:
                return self.json_error(502, str(error))
            self.send(200, answer.encode(), "application/sdp")

    return ThreadingHTTPServer((host, port), Handler)
