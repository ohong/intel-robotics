"""Manifest merging uses synthetic byte fixtures, never physical defect evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('cv_manifest', Path(__file__).parents[1] / 'scripts/cv_manifest.py')
manifest_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest_tool)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, name, role='train', session=None, specimen=None, content=None):
        root = self.root / name
        root.mkdir()
        (root / 'source.png').write_bytes(content or name.encode())
        row = {'path': 'source.png', 'role': role, 'session_id': session or name,
               'specimen_id': specimen or name, 'label': 'unknown',
               'ground_truth': {'source': 'synthetic byte fixture', 'authoritative': False}}
        path = root / 'manifest.json'
        path.write_text(json.dumps({'schema_version': 1, 'capture_status': 'COMPLETE', 'images': [row]}))
        return str(path)

    def test_merge_relative_paths_and_preserve_unknown_ground_truth(self):
        inputs = [self.source('a'), self.source('b', role='test')]
        output = self.root / 'combined' / 'manifest.json'
        result = manifest_tool.merge(inputs, str(output), engineering_smoke=True)
        self.assertEqual(result['count'], 2)
        rows = json.loads(output.read_text())['images']
        self.assertEqual((output.parent / rows[0]['path']).read_bytes(), b'a')
        self.assertEqual(rows[0]['label'], 'unknown')
        self.assertFalse(rows[0]['ground_truth']['authoritative'])
        with self.assertRaises(FileExistsError):
            manifest_tool.merge(inputs, str(output), engineering_smoke=True)

    def test_requires_explicit_smoke_for_unknown(self):
        with self.assertRaises(ValueError):
            manifest_tool.merge([self.source('a')], str(self.root / 'out.json'))
        self.assertFalse((self.root / 'out.json').exists())

    def test_rejects_cross_capture_session_and_specimen_and_hash_leaks(self):
        for key in ('session', 'specimen', 'content'):
            with self.subTest(key=key):
                kwargs = {key: b'same' if key == 'content' else 'same'}
                inputs = [self.source(key + '-a', **kwargs), self.source(key + '-b', role='test', **kwargs)]
                with self.assertRaises(ValueError):
                    manifest_tool.merge(inputs, str(self.root / (key + '.json')), engineering_smoke=True)

    def test_partial_capture_is_rejected(self):
        source = Path(self.source('a'))
        value = json.loads(source.read_text())
        value['capture_status'] = 'PARTIAL_OR_FAILED'
        source.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            manifest_tool.merge([str(source)], str(self.root / 'out.json'), engineering_smoke=True)


if __name__ == '__main__':
    unittest.main()
