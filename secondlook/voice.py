"""Speech input and spoken narration. Voice selects a task instruction; it never
moves, arms, or stops the robot.

fal.ai calls use the stdlib only. Model ids come from the environment so the
provider can change without code edits. Narration sentences are templated from
runtime status fields; SmolVLA emits actions, not reasoning text.
"""
from __future__ import annotations

from collections import OrderedDict
import difflib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FAL_RUN = "https://fal.run/"
FAL_UPLOAD = "https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
AUDIO_TYPES = {"audio/webm", "audio/wav", "audio/x-wav", "audio/mp4", "audio/mpeg", "audio/ogg"}
MAX_AUDIO_BYTES = 5 * 1024 * 1024
MAX_SPEECH_CHARS = 500


class VoiceError(RuntimeError):
    """A speech request produced no usable result."""


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def load_commands(path: Path) -> list[dict]:
    commands = json.loads(path.read_text())
    if not isinstance(commands, list) or not commands:
        raise ValueError("Voice commands must be a non-empty list")
    for command in commands:
        phrases = command.get("phrases") if isinstance(command, dict) else None
        if (not isinstance(command.get("instruction"), str) or not command["instruction"].strip() or
                not isinstance(phrases, list) or not phrases or
                not all(isinstance(p, str) and normalize(p) for p in phrases)):
            raise ValueError("Each voice command needs an instruction and non-empty phrases")
    return commands


def match_command(transcript: str, commands: list[dict], threshold: float = .75,
                  margin: float = .08) -> dict | None:
    """Return the one clearly best command, or None when unmatched or ambiguous."""
    heard = normalize(transcript)
    if not heard:
        return None
    scored = sorted(
        ((max(difflib.SequenceMatcher(None, heard, normalize(p)).ratio()
              for p in [c["instruction"], *c["phrases"]]), i) for i, c in enumerate(commands)),
        reverse=True)
    best, index = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best < threshold or best - runner_up < margin:
        return None
    return {"instruction": commands[index]["instruction"], "score": round(best, 3)}


class Narrator:
    """Turns status transitions into short sentences, rate-limited per topic."""

    def __init__(self, min_interval: float = 4.0, clock: Callable[[], float] = time.monotonic):
        self.min_interval = min_interval
        self.clock = clock
        self._spoken: dict[str, tuple[str, float]] = {}

    def _say(self, topic: str, key: str, sentence: str, out: list[str], throttle: bool = True) -> None:
        previous = self._spoken.get(topic)
        if previous and previous[0] == key:
            return
        now = self.clock()
        if throttle and previous and now - previous[1] < self.min_interval:
            return
        self._spoken[topic] = (key, now)
        out.append(sentence)

    def update(self, status: dict) -> list[str]:
        out: list[str] = []
        instruction = status.get("task_instruction")
        if instruction:
            self._say("instruction", instruction, f"Task set: {instruction}", out, throttle=False)
        camera = status.get("camera", {}).get("status")
        if camera:
            self._say("camera", camera, "Camera is live." if camera == "LIVE"
                      else "I have lost the camera feed.", out)
        detector = status.get("detector", {})
        disposition = detector.get("disposition") if detector.get("status") == "FRESH" else detector.get("status")
        if disposition:
            score, threshold = detector.get("score"), detector.get("threshold")
            if disposition in ("NORMAL", "ANOMALOUS", "UNCERTAIN", "UNKNOWN") and _finite(score):
                limit = f" against threshold {threshold:.3g}" if _finite(threshold) else ", with no validated threshold"
                sentence = f"The block looks {disposition.lower()}: score {score:.3g}{limit}."
            elif disposition == "INVALID":
                sentence = "I can't judge this view; the observation is invalid."
            else:
                sentence = f"Defect detector is {disposition.lower()}."
            self._say("detector", disposition, sentence, out)
        policy = status.get("policy", {})
        if policy.get("status"):
            sentence = (f"Policy is blocked, so I will not move. {policy.get('reason', '')}".strip()
                        if policy["status"] == "BLOCKED" else f"Policy status: {policy['status'].lower()}.")
            self._say("policy", policy["status"], sentence, out, throttle=False)
        return out


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


class FalClient:
    """Synchronous fal.ai calls. The key stays server-side and out of logs."""

    def __init__(self, key: str, stt_model: str = "fal-ai/wizper",
                 tts_model: str = "fal-ai/kokoro/american-english", voice: str = "am_michael",
                 timeout: float = 30, cache_size: int = 64):
        if not key:
            raise VoiceError("FAL_KEY is required")
        self._key = key
        self.stt_model, self.tts_model, self.voice, self.timeout = stt_model, tts_model, voice, timeout
        self._cache: OrderedDict[tuple[str, str], tuple[bytes, str]] = OrderedDict()
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "FalClient | None":
        key = os.environ.get("FAL_KEY", "").strip()
        if not key:
            return None
        return cls(key, os.environ.get("SECONDLOOK_STT_MODEL", "fal-ai/wizper"),
                   os.environ.get("SECONDLOOK_TTS_MODEL", "fal-ai/kokoro/american-english"),
                   os.environ.get("SECONDLOOK_TTS_VOICE", "am_michael"))

    def _request(self, url: str, label: str, body: bytes, content_type: str,
                 method: str = "POST", auth: bool = True) -> bytes:
        headers = {"Content-Type": content_type}
        if auth:
            headers["Authorization"] = f"Key {self._key}"
        try:
            with urlopen(Request(url, data=body, method=method, headers=headers), timeout=self.timeout) as response:
                return response.read()
        except HTTPError as error:
            detail = error.read()[:300].decode(errors="replace")
            raise VoiceError(f"{label} returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise VoiceError(f"{label} request failed: {error}") from error

    def _call(self, model: str, payload: dict) -> dict:
        body = self._request(FAL_RUN + model, model, json.dumps(payload).encode(), "application/json")
        try:
            return json.loads(body)
        except ValueError as error:
            raise VoiceError(f"{model} returned invalid JSON") from error

    def _upload(self, data: bytes, mime: str) -> str:
        """fal models reject many inline data URLs, so audio goes to fal storage first."""
        extension = mime.split("/")[1].removeprefix("x-")
        initiate = json.dumps({"content_type": mime, "file_name": f"command.{extension}"}).encode()
        try:
            target = json.loads(self._request(FAL_UPLOAD, "fal upload", initiate, "application/json"))
            upload_url, file_url = target["upload_url"], target["file_url"]
        except (ValueError, KeyError, TypeError) as error:
            raise VoiceError("fal upload returned no upload URL") from error
        self._request(upload_url, "fal upload", data, mime, method="PUT", auth=False)
        return file_url

    def transcribe(self, audio: bytes, mime: str) -> str:
        if mime not in AUDIO_TYPES or not audio or len(audio) > MAX_AUDIO_BYTES:
            raise VoiceError("Unsupported or oversized audio")
        text = self._call(self.stt_model, {"audio_url": self._upload(audio, mime), "task": "transcribe", "language": "en"}).get("text")
        if not isinstance(text, str) or not text.strip():
            raise VoiceError("Transcription was empty")
        return text.strip()

    def speak(self, text: str) -> tuple[bytes, str]:
        text = text.strip()
        if not text or len(text) > MAX_SPEECH_CHARS:
            raise VoiceError("Speech text must be 1-500 characters")
        key = (self.voice, text)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        url = self._call(self.tts_model, {"prompt": text, "voice": self.voice}).get("audio", {}).get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise VoiceError("Speech response had no audio URL")
        try:
            with urlopen(url, timeout=self.timeout) as response:
                result = (response.read(), response.headers.get_content_type())
        except (URLError, TimeoutError) as error:
            raise VoiceError(f"Speech audio download failed: {error}") from error
        with self._cache_lock:
            self._cache[key] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result
