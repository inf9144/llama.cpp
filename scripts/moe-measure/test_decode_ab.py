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
    def test_sampling_comparison_changes_only_sampling_and_skips_extra_profiling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = dict(messages=[dict(role='user', content='same prompt')], temperature=0.7, seed=42,
                          max_tokens=128, stream=False, cache_prompt=False,
                          dynatemp_range=0.4, top_n_sigma=1.5, dry_multiplier=0.2)
            request = root / 'custom.json'
            request.write_text(json.dumps(source))
            native_log = root / 'native.log'
            native_log.write_text('native numerical checks passed\n')
            commands = []

            def run(command, *, cwd, check):
                self.assertEqual(cwd, root)
                self.assertFalse(check)
                commands.append(command)
                target = Path(command[command.index('--output') + 1])
                target.mkdir()
                requests = {phase: dict(timings=dict(predicted_per_second=40.0 if len(commands) == 1 else 50.0,
                                                     draft_n=100, draft_n_accepted=50 if len(commands) == 1 else 70),
                                        request_wall_s=12, usage=dict(prompt_tokens=5121, completion_tokens=128),
                                        content_sha256='different-' + str(len(commands))) for phase in ('cold', 'warm')}
                (target / 'comparison.json').write_text(json.dumps({'candidate': {'01-reference': {'requests': requests}}}))
                return subprocess.CompletedProcess(command, 0)

            argv = ['decode-ab.py', '--repo', str(root), '--output', 'suite', '--draft-widths', '3', '--profile',
                    '--request', str(request), '--compare-sampling', '--baseline-reference', 'saved/server', '--attach', str(native_log)]
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', run), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(runner.main(), 0)
            self.assertEqual(len(commands), 2)
            self.assertIn('--profile', commands[0])
            self.assertNotIn('--profile', commands[1])
            self.assertIn('--reference-only', commands[1])
            self.assertEqual(commands[1][commands[1].index('--baseline-reference') + 1], 'saved/server')
            generated = json.loads((root / 'suite/manufacturer-request.json').read_text())
            for key in ('messages', 'seed', 'max_tokens', 'stream', 'cache_prompt'):
                self.assertEqual(generated[key], source[key])
            self.assertEqual(json.loads(request.read_text()), source)
            self.assertEqual(generated['samplers'], ['top_k', 'top_p', 'temperature'])
            self.assertEqual((generated['dynatemp_range'], generated['dry_multiplier'], generated['top_n_sigma']), (0, 0, -1))
            self.assertEqual((generated['temperature'], generated['top_k'], generated['top_p'], generated['min_p']), (1, 20, 0.95, 0))
            report = json.loads((root / 'suite/decode-suite.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual((root / 'suite/native-tests/native.log').read_bytes(), native_log.read_bytes())
            self.assertEqual(report['attachments'][0]['sha256'], runner.runner.sha(native_log))
            comparison = report['sampling']['runs']['3']['comparison']['warm']
            self.assertEqual(comparison['decode_change_percent'], 25)
            self.assertEqual(comparison['manufacturer']['acceptance'], 0.7)
            self.assertIn('keine Antwortqualitaet', output.getvalue())

    def test_sampling_profile_failure_archives_without_claiming_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'moe-first-request.json').write_text('{}')
            argv = ['decode-ab.py', '--repo', str(root), '--output', 'suite', '--draft-widths', '3', '--compare-sampling']
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 2)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 1)
            report = json.loads((root / 'suite/decode-suite.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['sampling']['runs']['3']['exit_code'], 2)
            self.assertTrue((root / 'suite-handoff.tar.gz').is_file())

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
