import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('decode_ab', Path(__file__).with_name('decode-ab.py'))
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DecodeSuiteTests(unittest.TestCase):
    def test_width_validation(self):
        self.assertEqual(runner.widths('3,5,7'), [3, 5, 7])
        for value in ('', '0', '8', '3,3', '3,x'):
            with self.assertRaises(argparse.ArgumentTypeError):
                runner.widths(value)

    def run_suite(self, root: Path, codes: list[int]) -> tuple[int, list[list[str]]]:
        commands = []

        def run(command, *, cwd, check):
            self.assertEqual(cwd, root)
            self.assertFalse(check)
            commands.append(command)
            target = Path(command[command.index('--output') + 1])
            target.mkdir()
            (target / 'suite.json').write_text('{}')
            return subprocess.CompletedProcess(command, codes[len(commands) - 1])

        argv = ['decode-ab.py', '--repo', str(root), '--output', 'suite', '--profile']
        with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', run), \
                contextlib.redirect_stdout(io.StringIO()):
            status = runner.main()
        return status, commands

    def test_all_widths_forward_profile_and_use_one_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            status, commands = self.run_suite(root, [0, 0, 0])
            self.assertEqual(status, 0)
            self.assertEqual([c[c.index('--draft-max') + 1] for c in commands], ['3', '5', '7'])
            for command in commands:
                self.assertIn('--profile', command)
                self.assertIn('--no-archive', command)
            report = json.loads((root / 'suite/decode-suite.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['runs']), 3)
            with tarfile.open(root / 'suite-handoff.tar.gz') as archive:
                self.assertIn('suite/draft-7/suite.json', archive.getnames())

    def test_first_failure_stops_and_archives_partial_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            status, commands = self.run_suite(root, [0, 2])
            self.assertEqual(status, 1)
            self.assertEqual(len(commands), 2)
            report = json.loads((root / 'suite/decode-suite.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['runs']['5']['exit_code'], 2)
            with tarfile.open(root / 'suite-handoff.tar.gz') as archive:
                self.assertIn('suite/decode-suite.json', archive.getnames())


if __name__ == '__main__':
    unittest.main()
