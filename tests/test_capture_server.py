import json
from pathlib import Path
import tempfile
import time
import unittest

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:
    if exc.name not in {'numpy', 'PIL'}:
        raise
    np = None

from scripts.cv_capture_server import Collection


@unittest.skipIf(np is None, 'Synthetic image checks need numpy and Pillow')
class CaptureServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.collection = Collection(Path(self.temp.name))
        self.specimen = self.collection.new_specimen()['specimen_id']

    def tearDown(self):
        self.collection.close()
        self.temp.cleanup()

    def publish(self, sequence=1, age=0, value=43):
        now = time.monotonic() - age
        rgb = np.full((12, 16, 3), value, dtype=np.uint8)
        self.collection.publish(rgb, {'sequence': sequence, 'capture_timestamp': now,
                                     'capture_timestamp_ns': int(now * 1e9), 'stored_color': 'RGB'})
        return next(reversed(self.collection.frames)), rgb

    def capture(self, key, label='normal'):
        return self.collection.capture({'frame_id': key, 'specimen_id': self.specimen, 'label': label})

    def test_exact_displayed_frame_png_and_resume(self):
        displayed, expected = self.publish()
        self.publish(sequence=2, value=99)
        row = self.capture(displayed)
        np.testing.assert_array_equal(np.asarray(Image.open(Path(self.temp.name) / row['path'])), expected)
        self.assertEqual(row['role'], 'unassigned')
        self.assertTrue(row['ground_truth']['authoritative'])
        self.collection.close()
        self.collection = Collection(Path(self.temp.name))
        self.assertEqual(self.collection.state()['count'], 1)
        self.assertIn(self.specimen, self.collection.specimens)
        with self.assertRaisesRegex(ValueError, 'already saved'):
            self.capture(displayed)

    def test_freshness_label_conflict_and_payload(self):
        key, _ = self.publish(age=4)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.capture(key)
        key, _ = self.publish(sequence=2)
        self.capture(key)
        key, _ = self.publish(sequence=3)
        with self.assertRaisesRegex(ValueError, 'different label'):
            self.capture(key, 'anomalous')
        for body in ([], {}, {'frame_id': []}, {'frame_id':key,'specimen_id':self.specimen,'label':''}):
            with self.assertRaises(ValueError):
                self.collection.capture(body)

    def test_unknown_not_authoritative_and_partial_journal_rejected(self):
        key, _ = self.publish()
        row = self.capture(key, 'unknown')
        self.assertFalse(row['ground_truth']['authoritative'])
        self.collection.close()
        with (Path(self.temp.name) / 'captures.jsonl').open('a') as stream:
            stream.write('{')
        with self.assertRaisesRegex(ValueError, 'Incomplete journal'):
            Collection(Path(self.temp.name))


if __name__ == '__main__':
    unittest.main()
