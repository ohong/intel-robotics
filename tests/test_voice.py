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
