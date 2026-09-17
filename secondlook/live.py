"""GPT-Live "ask the robot": full-duplex voice about what the system sees and decides.

The browser holds the WebRTC connection; this module only exchanges its SDP offer
for an answer so the OpenAI key stays server-side. Every tool is answered by this
app's own read-only routes, plus one allowlisted instruction setter. No tool can
arm, move, or stop the robot, and the model is told to say so.
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LIVE_SESSIONS = "https://api.openai.com/v1/live/sessions"
MAX_SDP_BYTES = 64 * 1024

VOICE_INSTRUCTIONS = """You are the voice of Second Look, a robot-arm inspection station at a live demo.
Speak briefly and plainly, one or two sentences. Answer only from tool results and say which
source a number came from. If a tool says REPLAY, SYNTHETIC or UNAVAILABLE, say that.
You cannot move, arm, stop, or release the robot, and no one can ask you to; the controller is
software-disarmed. For a stop, tell people to use the physical stop procedure."""

DELEGATE_INSTRUCTIONS = """Use tools to answer questions about the Second Look station. Never
invent a score, latency, pose, or outcome; report tool fields and their labels (LIVE, REPLAY,
SYNTHETIC, UNAVAILABLE). The policy is blocked and the controller is DISARMED; no tool moves the
arm. set_task_instruction only selects an allowlisted task text and reports whether it matched."""

TOOLS = [
    {"type": "function", "name": "get_status",
     "description": "Camera, anomaly detector (score, threshold, disposition, device, latency), policy, controller and task instruction.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"type": "function", "name": "get_recent_evidence",
     "description": "Newest records from the append-only evidence log, newest last.",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["n"], "additionalProperties": False}},
    {"type": "function", "name": "get_arm_pose",
     "description": "Joint angles in degrees (gripper 0-100) with their source: LIVE Studio telemetry, REPLAY of a recorded episode, or UNAVAILABLE.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"type": "function", "name": "set_task_instruction",
     "description": "Select the task instruction that matches a spoken phrase from the allowlist. Does not start or move anything.",
     "parameters": {"type": "object", "properties": {"phrase": {"type": "string", "maxLength": 200}},
                    "required": ["phrase"], "additionalProperties": False}},
]


class LiveError(RuntimeError):
    """A GPT-Live session could not be started."""


class LiveClient:
    def __init__(self, key: str, model: str = "gpt-live-1", delegate_model: str = "gpt-5.6-terra",
                 timeout: float = 20, endpoint: str = LIVE_SESSIONS):
        if not key:
            raise LiveError("OPENAI_API_KEY is required")
        self._key = key
        self.model, self.delegate_model, self.timeout, self.endpoint = model, delegate_model, timeout, endpoint

    @classmethod
    def from_env(cls) -> "LiveClient | None":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        return cls(key, os.environ.get("SECONDLOOK_LIVE_MODEL", "gpt-live-1"),
                   os.environ.get("SECONDLOOK_LIVE_DELEGATE_MODEL", "gpt-5.6-terra"))

    def status(self) -> dict:
        return {"status": "READY", "model": self.model, "delegate_model": self.delegate_model,
                "tools": [tool["name"] for tool in TOOLS],
                "scope": "Answers from read-only tools; can select an allowlisted task text; cannot move or stop the robot"}

    def session_body(self, offer: str) -> dict:
        return {"session": {"model": self.model, "instructions": VOICE_INSTRUCTIONS,
                            "delegation": {"type": "responses", "responses": {
                                "model": self.delegate_model, "instructions": DELEGATE_INSTRUCTIONS,
                                "tools": TOOLS, "tool_choice": "auto"}}},
                "transport": {"type": "webrtc", "sdp": offer}}

    def create_session(self, offer: str) -> str:
        """Exchange a browser SDP offer for OpenAI's SDP answer."""
        if not offer.startswith("v=0") or len(offer.encode()) > MAX_SDP_BYTES:
            raise LiveError("Request body is not an SDP offer")
        request = Request(self.endpoint, data=json.dumps(self.session_body(offer)).encode(), method="POST",
                          headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except HTTPError as error:
            detail = error.read()[:300].decode(errors="replace")
            raise LiveError(f"GPT-Live session returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, ValueError) as error:
            raise LiveError(f"GPT-Live session request failed: {error}") from error
        answer = (reply.get("transport") or {}).get("sdp") if isinstance(reply, dict) else None
        if not isinstance(answer, str) or not answer.startswith("v=0"):
            raise LiveError("GPT-Live session reply has no SDP answer")
        return answer
