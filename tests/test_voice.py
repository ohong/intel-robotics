import json
import os
from pathlib import Path
import tempfile
import unittest

from secondlook.voice import FalClient, Narrator, VoiceError, load_commands, match_command

COMMANDS = load_commands(Path(__file__).resolve().parents[1] / "config/voice-commands.json")


class MatchTests(unittest.TestCase):
    def test_example_phrases_match_their_instruction(self):
        sort = match_command("Sort this block into the right bin.", COMMANDS)
        move = match_command("pick up a block and move it to the inspection area", COMMANDS)
        self.assertTrue(sort["instruction"].startswith("Sort LEGO"))
        self.assertTrue(move["instruction"].startswith("Pick up a block"))

    def test_unrelated_or_empty_speech_does_not_match(self):
        self.assertIsNone(match_command("what's the weather today", COMMANDS))
        self.assertIsNone(match_command("   ", COMMANDS))

    def test_ambiguous_speech_is_rejected(self):
        commands = [{"instruction": "A", "phrases": ["move the block left"]},
                    {"instruction": "B", "phrases": ["move the block right"]}]
        self.assertIsNone(match_command("move the block", commands))

    def test_invalid_command_file_fails(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
            json.dump([{"instruction": "x", "phrases": []}], stream)
            stream.flush()
            with self.assertRaises(ValueError):
                load_commands(Path(stream.name))


class NarratorTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.narrator = Narrator(min_interval=4, clock=lambda: self.now)

    def status(self, disposition="ANOMALOUS", score=0.83):
        return {"task_instruction": "Sort blocks.", "camera": {"status": "LIVE"},
                "detector": {"status": "FRESH", "disposition": disposition, "score": score, "threshold": 0.61},
                "policy": {"status": "BLOCKED", "reason": "No checkpoint."}}

    def test_first_status_narrates_each_topic_once(self):
        lines = self.narrator.update(self.status())
        self.assertEqual(lines, ["Task set: Sort blocks.", "Camera is live.",
                                 "The block looks anomalous: score 0.83 against threshold 0.61.",
                                 "Policy is blocked, so I will not move. No checkpoint."])
        self.assertEqual(self.narrator.update(self.status()), [])

    def test_detector_flicker_is_rate_limited(self):
        self.narrator.update(self.status())
        self.now = 1
        self.assertEqual(self.narrator.update(self.status("NORMAL", .2)), [])
        self.now = 6
        self.assertEqual(self.narrator.update(self.status("NORMAL", .2)),
                         ["The block looks normal: score 0.2 against threshold 0.61."])

    def test_missing_detector_result_does_not_invent_a_score(self):
        status = self.status()
        status["detector"] = {"status": "BLOCKED", "reason": "No artifact"}
        self.assertIn("Defect detector is blocked.", self.narrator.update(status))


class FalClientTests(unittest.TestCase):
    def test_missing_key_means_voice_unavailable(self):
        saved = os.environ.pop("FAL_KEY", None)
        try:
            self.assertIsNone(FalClient.from_env())
        finally:
            if saved is not None:
                os.environ["FAL_KEY"] = saved

    def test_rejects_bad_input_before_network(self):
        client = FalClient("unused")
        with self.assertRaises(VoiceError):
            client.transcribe(b"x", "text/plain")
        with self.assertRaises(VoiceError):
            client.speak("x" * 501)


@unittest.skipUnless(os.environ.get("FAL_KEY"), "FAL_KEY not set; live fal.ai round trip skipped")
class LiveFalRoundTrip(unittest.TestCase):
    def test_spoken_command_transcribes_back_to_same_instruction(self):
        client = FalClient.from_env()
        audio, mime = client.speak("Sort this block into the right bin.")
        mime = "audio/wav" if mime in ("audio/x-wav", "audio/wave") else mime
        transcript = client.transcribe(audio, mime)
        self.assertTrue(match_command(transcript, COMMANDS)["instruction"].startswith("Sort LEGO"), transcript)


class StandInVoice:
    """Local stand-in for FalClient; exercises the real runtime and HTTP server."""
    stt_model, tts_model = "stand-in-stt", "stand-in-tts"

    def __init__(self, transcript):
        self.transcript = transcript

    def transcribe(self, audio, mime):
        return self.transcript

    def speak(self, text):
        return b"RIFF-audio", "audio/wav"


class VoiceServerTests(unittest.TestCase):
    def start(self, voice):
        import threading
        from secondlook.runtime import CameraSource, InspectionRuntime
        from secondlook.web import make_server
        self.tmp = tempfile.TemporaryDirectory()
        self.runtime = InspectionRuntime(CameraSource(max_age=1), evidence_kind="mock",
                                         evidence_path=Path(self.tmp.name) / "evidence.jsonl",
                                         voice=voice, voice_commands=COMMANDS)
        self.server = make_server(self.runtime, port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def post(self, path, body, mime):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        try:
            with urlopen(Request(self.base + path, data=body, method="POST",
                                 headers={"Content-Type": mime})) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def status(self):
        from urllib.request import urlopen
        with urlopen(self.base + "/api/status") as response:
            return json.loads(response.read())

    def test_matched_speech_sets_instruction_and_stays_disarmed(self):
        self.start(StandInVoice("sort this block into the right bin"))
        code, body = self.post("/api/voice/command", b"audio", "audio/webm;codecs=opus")
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)["matched"])
        status = self.status()
        self.assertTrue(status["task_instruction"].startswith("Sort LEGO"))
        self.assertEqual(status["controller"]["status"], "DISARMED")
        self.assertIn("Task set: Sort LEGO", status["narration"][0]["text"])
        records = [json.loads(line) for line in (Path(self.tmp.name) / "evidence.jsonl").read_text().splitlines()]
        self.assertEqual(records[-1]["outcome"], "INSTRUCTION_SET")

    def test_unknown_speech_leaves_instruction_and_is_narrated(self):
        self.start(StandInVoice("open the pod bay doors"))
        code, body = self.post("/api/voice/command", b"audio", "audio/webm")
        self.assertEqual((code, json.loads(body)["matched"]), (200, False))
        status = self.status()
        self.assertIsNone(status["task_instruction"])
        self.assertTrue(any("not a known task" in line["text"] for line in status["narration"]))

    def test_speak_returns_audio_and_rejects_bad_requests(self):
        self.start(StandInVoice("x"))
        self.assertEqual(self.post("/api/voice/speak", b'{"text":"hello"}', "application/json"),
                         (200, b"RIFF-audio"))
        self.assertEqual(self.post("/api/voice/speak", b'{"text":""}', "application/json")[0], 400)
        self.assertEqual(self.post("/api/voice/command", b"audio", "text/plain")[0], 415)
        self.assertEqual(self.post("/api/arm", b"{}", "application/json")[0], 404)

    def test_no_key_reports_unavailable(self):
        self.start(None)
        self.assertEqual(self.status()["voice"]["status"], "UNAVAILABLE")
        self.assertEqual(self.post("/api/voice/command", b"audio", "audio/webm")[0], 503)
