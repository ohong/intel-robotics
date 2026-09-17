import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from secondlook.live import TOOLS, LiveClient
from secondlook.runtime import CameraSource, InspectionRuntime
from secondlook.voice import load_commands
from secondlook.web import make_server

COMMANDS = load_commands(Path(__file__).resolve().parents[1] / "config/voice-commands.json")
OFFER = "v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\ns=-\r\n"
KEY = "sk-test-not-a-real-key"


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


class StandInOpenAI(BaseHTTPRequestHandler):
    """Local HTTP stand-in for the Live sessions endpoint; records what the app sent."""
    requests: list = []
    status = 200

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StandInOpenAI.requests.append((self.headers["Authorization"], body))
        reply = (json.dumps({"session": {"id": "sess_1"}, "transport": {"type": "webrtc", "sdp": "v=0\r\nanswer\r\n"}})
                 if self.status == 200 else json.dumps({"error": {"message": "bad model"}}))
        data = reply.encode()
        self.send_response(self.status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LiveRouteTests(unittest.TestCase):
    def start(self, live):
        self.tmp = tempfile.TemporaryDirectory()
        self.evidence = Path(self.tmp.name) / "evidence.jsonl"
        self.runtime = InspectionRuntime(CameraSource(max_age=1), evidence_kind="mock",
                                         evidence_path=self.evidence, voice_commands=COMMANDS)
        self.server = make_server(self.runtime, port=0, live=live)
        self.base = serve(self.server)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def post(self, path, body, mime, origin="same"):
        headers = {"Content-Type": mime}
        if origin:
            headers["Origin"] = self.base if origin == "same" else origin
        try:
            with urlopen(Request(self.base + path, data=body, method="POST", headers=headers)) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def get(self, path):
        with urlopen(self.base + path) as response:
            return json.loads(response.read())

    def test_without_key_live_is_unavailable(self):
        self.start(None)
        self.assertEqual(self.get("/api/live")["status"], "UNAVAILABLE")
        self.assertEqual(self.post("/api/live/session", OFFER.encode(), "application/sdp")[0], 503)

    def test_cross_origin_and_originless_posts_are_refused(self):
        self.start(None)
        for origin in ("http://evil.example", None):
            self.assertEqual(self.post("/api/task/instruction", b'{"phrase":"sort the lego blocks"}',
                                       "application/json", origin)[0], 403)
            self.assertEqual(self.post("/api/live/session", OFFER.encode(), "application/sdp", origin)[0], 403)
        self.assertIsNone(self.get("/api/status")["task_instruction"])

    def test_instruction_route_sets_allowlisted_task_only(self):
        self.start(None)
        code, body = self.post("/api/task/instruction", b'{"phrase":"move the arm left"}', "application/json")
        self.assertEqual((code, json.loads(body)["matched"]), (200, False))
        self.assertIsNone(self.get("/api/status")["task_instruction"])
        code, body = self.post("/api/task/instruction", b'{"phrase":"sort the lego blocks"}', "application/json")
        self.assertEqual((code, json.loads(body)["matched"]), (200, True))
        status = self.get("/api/status")
        self.assertTrue(status["task_instruction"].startswith("Sort LEGO"))
        self.assertEqual(status["controller"]["status"], "DISARMED")
        records = [json.loads(line) for line in self.evidence.read_text().splitlines()]
        self.assertEqual([r["outcome"] for r in records], ["REJECTED_UNKNOWN_TASK", "INSTRUCTION_SET"])
        self.assertEqual(records[-1]["provenance"]["source"], "gpt-live")
        self.assertEqual(self.post("/api/task/instruction", b'{"phrase":""}', "application/json")[0], 400)

    def test_session_exchanges_offer_for_answer_with_read_only_tools(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), StandInOpenAI)
        endpoint = serve(upstream) + "/v1/live/sessions"
        StandInOpenAI.requests, StandInOpenAI.status = [], 200
        try:
            self.start(LiveClient(KEY, endpoint=endpoint))
            self.assertEqual(self.get("/api/live")["status"], "READY")
            code, body = self.post("/api/live/session", OFFER.encode(), "application/sdp")
            self.assertEqual((code, body), (200, b"v=0\r\nanswer\r\n"))
            auth, sent = StandInOpenAI.requests[-1]
            self.assertEqual(auth, f"Bearer {KEY}")
            self.assertEqual(sent["transport"], {"type": "webrtc", "sdp": OFFER})
            self.assertEqual(sent["session"]["delegation"]["responses"]["tools"], TOOLS)
            self.assertEqual(self.post("/api/live/session", b"hello", "application/sdp")[0], 400)

            StandInOpenAI.status = 400
            code, body = self.post("/api/live/session", OFFER.encode(), "application/sdp")
            self.assertEqual(code, 502)
            self.assertIn("bad model", json.loads(body)["error"])
            self.assertNotIn(KEY, body.decode())
        finally:
            upstream.shutdown()
            upstream.server_close()



class ToolScopeTests(unittest.TestCase):
    def test_tools_are_the_read_only_allowlist(self):
        # A new tool must be added here deliberately; none may arm, move, or stop the robot.
        self.assertEqual([tool["name"] for tool in TOOLS],
                         ["get_status", "get_recent_evidence", "get_arm_pose", "set_task_instruction"])


if __name__ == "__main__":
    unittest.main()
