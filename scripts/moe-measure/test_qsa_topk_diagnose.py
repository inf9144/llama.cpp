import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import qsa_topk_diagnose as diagnose


def event(pass_id):
    data = dict(pass_=pass_id, finite_cutoff_ambiguous_rows=2, device='ROCm0')
    data['pass'] = data.pop('pass_')
    if pass_id == 1:
        data.update(changed_score_rows=1, changed_visible_set_rows=2, matching_score_hash_changed_visible_set_rows=1)
    return data


def frozen():
    return dict(device='ROCm0', input_unchanged=True,
                repetitions=[dict(changed_visible_sets_from_previous=i % 2, changed_visible_sets_from_original=2) for i in range(8)])


class QsaTopkDiagnoseTests(unittest.TestCase):
    def test_summary_separates_input_changes_from_selection_changes(self):
        first, second = event(0), event(1)
        first['frozen_replay'] = frozen()
        result = diagnose.summarize([first, second], dict(complete=True, prompt_tokens=5121, calls_per_pass=1))
        self.assertEqual(result['changed_score_rows'], 1)
        self.assertEqual(result['matching_score_hash_changed_visible_set_rows'], 1)
        self.assertEqual(result['frozen_max_changed_rows'], 1)
        self.assertEqual(result['frozen_max_changed_from_original'], 2)
        self.assertEqual(result['frozen_replay_devices'], ['ROCm0'])

    def test_incomplete_coverage_or_unfrozen_input_is_rejected(self):
        done = dict(complete=True, prompt_tokens=5121, calls_per_pass=1)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            diagnose.summarize([event(0)], done)
        for field, value in (('input_unchanged', False), ('repetitions', [])):
            first = event(0)
            first['frozen_replay'] = frozen()
            first['frozen_replay'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'frozen-input'):
                diagnose.summarize([first, event(1)], done)
        cpu = [event(0), event(1)]
        for row in cpu: row['device'] = 'CPU'
        with self.assertRaisesRegex(ValueError, 'No ROCm'):
            diagnose.summarize(cpu, done)

    def run_suite(self, root, status=0, timeout=False, bad_prompt_hash=False):
        source = root / 'previous'
        source.mkdir()
        tokens = [1, 2, 3]
        (source / 'prompt-tokens.json').write_text(json.dumps(tokens))
        (source / 'report.json').write_text(json.dumps(dict(complete=True, draft_max=0,
                                                          prompt_sha256='wrong' if bad_prompt_hash else diagnose.diagnose.digest(tokens))))
        binary = root / 'build-rocm-moe-stream-reference/bin/test-qsa-topk-trace'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'test binary')
        original = ('./build-rocm-moe-reference/bin/llama-server --cache-ram 0 --moe-cache-mib 15200,9600 '
                    '--port 8080 --spec-draft-n-max 3 --ctx-size 262144\n')
        (root / 'moe-server-reference.fish').write_text(original)
        requests = []

        class FakeProcess:
            pid = 123456

            def __init__(self, cmd, cwd, stdout, stderr, start_new_session):
                requests.append((cmd, cwd, start_new_session))
                self.waits = 0
                for data in (event(0), event(1)):
                    stdout.write(('QSA_TOPK ' + json.dumps(data) + '\n').encode())
                stdout.write(b'QSA_TOPK_DONE {"complete":true,"prompt_tokens":3,"calls_per_pass":1}\n')
                if status != 0: stdout.write(b'QSA TopK trace failed: startup fixture failure\n')

            def wait(self, timeout):
                self.waits += 1
                if timeout == diagnose.TIMEOUT_SECONDS and timeout_enabled:
                    raise subprocess.TimeoutExpired('trace', timeout)
                return status

            def poll(self):
                return None if timeout_enabled and self.waits == 1 else status

        timeout_enabled = timeout
        with mock.patch.object(sys, 'argv', ['qsa_topk_diagnose.py', '--repo', str(root), '--source-run', 'previous', '--output', 'out']), \
                mock.patch.object(diagnose.subprocess, 'Popen', FakeProcess), \
                mock.patch.object(diagnose.diagnose.runner, 'port_open', return_value=False), \
                mock.patch.object(diagnose.os, 'killpg') as kill, contextlib.redirect_stdout(io.StringIO()):
            code = diagnose.main()
        self.assertEqual((root / 'moe-server-reference.fish').read_text(), original)
        return code, requests, json.loads((root / 'out/report.json').read_text()), kill.call_args_list

    def test_suite_reuses_exact_tokens_and_cache_budgets_and_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, requests, report, kills = self.run_suite(root)
            self.assertEqual(code, 0)
            self.assertTrue(report['complete'])
            self.assertEqual(len(requests), 1)
            self.assertTrue(requests[0][2])
            self.assertEqual(kills, [])
            launcher = (root / 'out/trace.fish').read_text()
            self.assertIn('--moe-cache-mib 15200,9600', launcher)
            self.assertIn('--spec-draft-n-max 0', launcher)
            self.assertIn('test-qsa-topk-trace', launcher)
            self.assertTrue(launcher.endswith(' \\\n  --tokens ' + shlex.quote(str(root / 'out/prompt-tokens.json')) + '\n'))
            self.assertEqual(json.loads((root / 'out/prompt-tokens.json').read_text()), [1, 2, 3])
            self.assertTrue((root / 'out-handoff.tar.gz').is_file())

    def test_failed_native_trace_preserves_partial_events_and_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, _, report, _ = self.run_suite(root, status=1)
            self.assertEqual(code, 1)
            self.assertFalse(report['complete'])
            self.assertIn('status 1', report['failure'])
            self.assertIn('startup fixture failure', report['failure'])
            self.assertEqual(len(json.loads((root / 'out/events.json').read_text())), 2)
            self.assertTrue((root / 'out-handoff.tar.gz').is_file())

    def test_timeout_signals_only_the_owned_process_group_and_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            code, _, report, kills = self.run_suite(root, timeout=True)
            self.assertEqual(code, 1)
            self.assertIn('TimeoutExpired', report['failure'])
            self.assertEqual(kills, [mock.call(123456, diagnose.signal.SIGTERM)])
            self.assertEqual(report['captured_topk_calls'], 2)
            self.assertEqual(len(json.loads((root / 'out/events.json').read_text())), 2)
            self.assertTrue((root / 'out-handoff.tar.gz').is_file())

    def test_partial_trace_keeps_complete_events_and_skips_a_truncated_last_line(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'trace.log'
            log.write_text('QSA_TOPK ' + json.dumps(event(0)) + '\nQSA_TOPK {"pass":')
            events, done, error = diagnose.read_trace(log, partial=True)
            self.assertEqual(events, [event(0)])
            self.assertEqual(done, {})
            self.assertEqual(error, '')
            with self.assertRaises(json.JSONDecodeError):
                diagnose.read_trace(log)

    def test_prompt_digest_mismatch_stops_before_model_start(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            root = Path(folder)
            with self.assertRaises(SystemExit):
                self.run_suite(root, bad_prompt_hash=True)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
