"""Synthetic byte fixtures for lossless transfer; no model dependencies."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).parent))
import smolvla_weight_delta as delta


def safefile(path, tensors):
    offset, payload, header = 0, b'', {}
    for key, data in tensors.items():
        header[key] = {'dtype': 'U8', 'shape': [len(data)], 'data_offsets': [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


class DeltaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base, self.target, self.bundle, self.output = [self.root / name for name in ('base', 'target', 'bundle.zip', 'output')]
        safefile(self.base, {'model.a': b'unchanged', 'model.b': b'old', 'model.removed': b'x'})
        safefile(self.target, {'_model.new': b'inactive', '_model.a': b'unchanged', '_model.b': b'NEW'})
        self.pin = delta.sha(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_reconstruction_with_rename_change_addition_and_different_order(self):
        result = delta.create(self.base, self.target, self.bundle, self.pin)
        self.assertEqual(result['changed_tensors'], 2)
        self.assertEqual(result['reused_tensors'], 1)
        delta.reconstruct(self.base, self.bundle, self.output, self.pin)
        self.assertEqual(self.output.read_bytes(), self.target.read_bytes())

    def test_base_pin_and_overwrite_protection(self):
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            delta.create(self.base, self.target, self.bundle, '0' * 64)
        delta.create(self.base, self.target, self.bundle, self.pin)
        with self.assertRaisesRegex(ValueError, 'Preserve existing'):
            delta.create(self.base, self.target, self.bundle, self.pin)
        self.base.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            delta.reconstruct(self.base, self.bundle, self.output, self.pin)
        self.assertFalse(self.output.exists())

    def test_corrupt_changed_payload_fails_without_partial_output(self):
        delta.create(self.base, self.target, self.bundle, self.pin)
        bad = self.root / 'bad.zip'
        with zipfile.ZipFile(self.bundle) as source, zipfile.ZipFile(bad, 'w') as dest:
            for name in source.namelist():
                data = source.read(name)
                dest.writestr(name, b'wrong' if name.startswith('tensors/') else data)
        with self.assertRaisesRegex(ValueError, 'Corrupted tensor'):
            delta.reconstruct(self.base, bad, self.output, self.pin)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.weight-reconstruct-*')))


if __name__ == '__main__':
    unittest.main()
