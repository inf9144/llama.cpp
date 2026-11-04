import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('build_pair', Path(__file__).with_name('build_pair.py'))
assert spec is not None and spec.loader is not None
builds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builds)


class BuildPairTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.reference = self.make_server('reference')
        self.measure = self.make_server('measure')
        self.destination = self.root / 'saved baseline'

    def make_server(self, name):
        binary = self.root / name / 'llama-server'
        binary.parent.mkdir()
        # Read the version from the selected library directory, not from argv[0].
        binary.write_text('#!/usr/bin/env python3\nimport os, pathlib\n'
                          "print((pathlib.Path(os.environ['LD_LIBRARY_PATH'].split(os.pathsep)[0]) / 'library-version').read_text())\n")
        binary.chmod(0o755)
        self.version(binary, '96aa0367b')
        return binary

    def version(self, binary, commit):
        binary.with_name('library-version').write_text(f'version: 0.7.0-dev (build 12067, commit {commit})\n')

    def test_snapshot_checks_runtime_libraries_and_preserves_original_build(self):
        ids = builds.snapshot(self.reference, self.measure, self.destination)
        self.assertEqual(ids, dict(reference='b12067-96aa0367b', measure='b12067-96aa0367b'))
        self.assertEqual(json.loads((self.destination / 'build-pair.json').read_text()), ids)
        self.version(self.reference, '123456789')
        self.version(self.measure, '123456789')
        self.assertEqual(builds.snapshot(self.reference, self.measure, self.destination), ids)

    def test_mismatched_measurement_rejected_before_copy(self):
        self.version(self.measure, '8bafd739a')
        with self.assertRaisesRegex(ValueError, 'Different runtime builds'):
            builds.snapshot(self.reference, self.measure, self.destination)
        self.assertFalse(self.destination.exists())

    def test_previously_saved_mismatch_is_rejected_without_overwrite(self):
        builds.snapshot(self.reference, self.measure, self.destination)
        path = self.destination / 'measure/llama-server'
        self.version(path, '8bafd739a')
        with self.assertRaisesRegex(ValueError, 'Different runtime builds'):
            builds.snapshot(self.reference, self.measure, self.destination)
        self.assertIn('8bafd739a', path.with_name('library-version').read_text())

    def test_incomplete_saved_directory_is_not_replaced(self):
        self.destination.mkdir()
        marker = self.destination / 'keep'
        marker.write_text('existing data')
        with self.assertRaises(OSError):
            builds.snapshot(self.reference, self.measure, self.destination)
        self.assertEqual(marker.read_text(), 'existing data')

    def test_unrecognized_version_is_rejected(self):
        self.reference.with_name('library-version').write_text('not a version')
        with self.assertRaisesRegex(ValueError, 'No runtime build ID'):
            builds.build_id(self.reference)


if __name__ == '__main__':
    unittest.main()
