import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('completion_diagnose', Path(__file__).with_name('completion_diagnose.py'))
assert spec is not None and spec.loader is not None
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def response(tokens, tops=None):
    if tops is None:
        tops = [[dict(id=t, logprob=-0.1), dict(id=t + 100, logprob=-3.0)] for t in tokens]
    return dict(tokens=tokens, timings=dict(cache_n=0, prompt_n=3),
                completion_probabilities=[dict(id=t, top_logprobs=top) for t, top in zip(tokens, tops)])


class CompletionDiagnoseTests(unittest.TestCase):
    def run_mocked_suite(self, root, fail=False, different_prompt=False, draft_max=3, unexpected_drafts=False, baseline_reference=None):
        for case in diagnose.CASES:
            binary = root / diagnose.runner.BINARIES[case][0]
            binary.parent.mkdir(parents=True)
            binary.write_bytes(case.encode())
        extra = []
        if baseline_reference is not None:
            binary = baseline_reference if baseline_reference.is_absolute() else root / baseline_reference
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b'corrected-reference')
            extra = ['--baseline-reference', str(baseline_reference)]
        (root / 'moe-server-reference.fish').write_text(
            './build-rocm-moe-reference/bin/llama-server --cache-ram 0 '
            '--spec-draft-n-max 3 --moe-cache-mib 15200,9600 --port 8080\n')
        (root / 'moe-first-request.json').write_text(json.dumps(dict(
            messages=[dict(role='user', content='test')], max_tokens=3, seed=42)))
        sessions, requests = [], []

        @contextlib.contextmanager
        def fake_server(_root, folder, launcher, candidate):
            folder.mkdir(parents=True)
            sessions.append((folder.parent.name, folder.name, candidate))
            self.assertIn('--moe-cache-mib 15200,9600', launcher.read_text())
            self.assertIn(f'--spec-draft-n-max {draft_max}', launcher.read_text())
            yield

        def fake_http(folder, name, endpoint, payload=None):
            if endpoint == 'apply-template':
                return {'prompt': 'changed prompt' if different_prompt and folder.parent.name == 'candidate' else 'fixed prompt'}
            if endpoint == 'tokenize': return {'tokens': [1, 2, 3]}
            self.assertEqual(endpoint, 'completion')
            assert payload is not None
            requests.append((folder.parent.name, folder.name, payload))
            if fail: raise RuntimeError('injected request failure')
            tokens = [40] if folder.name == 'replay' else [10, 20 if folder.parent.name == 'baseline' else 21, 30]
            data = response(tokens)
            data['timings']['prompt_n'] = len(payload['prompt'])
            data['timings']['draft_n'] = 1 if unexpected_drafts else (0 if folder.name == 'replay' else draft_max)
            return data

        with mock.patch.object(sys, 'argv', ['completion_diagnose.py', '--repo', str(root), '--output', 'out', '--draft-max', str(draft_max), *extra]), \
                mock.patch.object(diagnose, 'server', fake_server), mock.patch.object(diagnose, 'http', fake_http), \
                contextlib.redirect_stdout(io.StringIO()):
            status = diagnose.main()
        return status, sessions, requests, json.loads((root / 'out/report.json').read_text())

    def test_corrected_baseline_override_selects_and_records_the_requested_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            relative = Path('build-rocm-moe-corrected-reference/bin/llama-server')
            status, _, _, report = self.run_mocked_suite(root, baseline_reference=relative)
            self.assertEqual(status, 0)
            self.assertEqual(report['binaries']['baseline']['binary'], str((root / relative).resolve()))
            self.assertEqual(report['binaries']['baseline']['binary_sha256'], diagnose.runner.sha(root / relative))
            self.assertIn(str(root / relative), (root / 'out/baseline.fish').read_text())
            old = root / diagnose.runner.BINARIES['baseline'][0]
            self.assertEqual(old.read_bytes(), b'baseline')
            self.assertEqual(report['binaries']['candidate']['binary'], str(root / diagnose.runner.BINARIES['candidate'][0]))

    def test_absolute_baseline_override_with_spaces_is_quoted_in_the_launcher(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'corrected reference/server'
            status, _, _, report = self.run_mocked_suite(root, baseline_reference=binary)
            self.assertEqual(status, 0)
            self.assertEqual(report['binaries']['baseline']['binary'], str(binary.resolve()))
            self.assertIn(diagnose.shlex.quote(str(binary)), (root / 'out/baseline.fish').read_text())

    def test_suite_replays_the_same_baseline_prefix_on_both_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            status, sessions, requests, report = self.run_mocked_suite(Path(folder))
            self.assertEqual(status, 0)
            self.assertTrue(report['complete'])
            self.assertEqual(report['replay_positions'], [0, 1])
            self.assertEqual(len(sessions), 4)
            replay = [payload for _, phase, payload in requests if phase == 'replay']
            self.assertEqual([p['prompt'] for p in replay], [[1, 2, 3], [1, 2, 3, 10]]*2)
            self.assertTrue(all(p['n_predict'] == 1 for p in replay))
            self.assertFalse(report['comparisons']['baseline_candidate_first']['tokens_equal'])
            self.assertTrue(report['replay_comparisons']['1']['tokens_equal'])
            self.assertTrue((Path(folder) / 'out-handoff.tar.gz').is_file())

    def test_suite_archives_partial_results_after_request_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            status, _, _, report = self.run_mocked_suite(Path(folder), fail=True)
            self.assertEqual(status, 1)
            self.assertFalse(report['complete'])
            self.assertIn('injected request failure', report['failure'])
            self.assertTrue((Path(folder) / 'out-handoff.tar.gz').is_file())

    def test_zero_draft_control_preserves_repeats_and_fixed_prefix_probes(self):
        with tempfile.TemporaryDirectory() as folder:
            status, sessions, requests, report = self.run_mocked_suite(Path(folder), draft_max=0)
            self.assertEqual(status, 0)
            self.assertTrue(report['complete'])
            self.assertEqual(report['draft_max'], 0)
            self.assertEqual(len(sessions), 4)
            self.assertEqual(len([p for _, phase, p in requests if phase == 'greedy']), 4)
            self.assertTrue(all(run['timings']['draft_n'] == 0 for runs in report['greedy'].values() for run in runs))
            replay = [p['prompt'] for _, phase, p in requests if phase == 'replay']
            self.assertEqual(replay, [[1, 2, 3], [1, 2, 3, 10]] * 2)

    def test_zero_draft_control_rejects_actual_drafts_and_preserves_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            status, _, _, report = self.run_mocked_suite(Path(folder), draft_max=0, unexpected_drafts=True)
            self.assertEqual(status, 1)
            self.assertFalse(report['complete'])
            self.assertIn('Expected no speculative draft tokens', report['failure'])
            self.assertTrue((Path(folder) / 'out-handoff.tar.gz').is_file())

    def test_different_rendered_prompts_stop_before_candidate_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            status, _, requests, report = self.run_mocked_suite(Path(folder), different_prompt=True)
            self.assertEqual(status, 1)
            self.assertIn('Rendered prompt differs', report['failure'])
            self.assertTrue(all(case == 'baseline' for case, _, _ in requests))

    def test_launcher_preserves_cache_budget_and_other_settings(self):
        source = ('#!/bin/fish\n./build-rocm-moe-reference/bin/llama-server '
                  '--cache-ram 0 --spec-draft-n-max 5 --moe-cache-mib 15200,9600 '
                  '--port 8080 --ctx-size 262144 --temp 0.7\n')
        path = Path('/tmp/my repo/server')
        cached = diagnose.launch_text(source, path)
        other = diagnose.launch_text(source, Path('/tmp/other/server'))
        self.assertIn("'/tmp/my repo/server'", cached)
        self.assertEqual(cached.replace("'/tmp/my repo/server'", '/tmp/other/server').replace("'/tmp/my repo'", '/tmp/other'), other)
        self.assertIn('--moe-cache-mib 15200,9600', cached)
        self.assertIn('--spec-draft-n-max 3', cached)
        self.assertIn('--ctx-size 262144 --temp 0.7', cached)
        self.assertEqual(diagnose.launch_text(source, path, 0), cached.replace('--spec-draft-n-max 3', '--spec-draft-n-max 0'))
        with self.assertRaisesRegex(ValueError, '0 or 3'):
            diagnose.launch_text(source, path, 5)
        for bad in (source.replace('--port 8080', '--port 8081'),
                    source.replace('--moe-cache-mib 15200,9600', '--moe-cache-mib 0,0'),
                    source + '--cache-ssd-dir /tmp/cache',
                    source.replace('--spec-draft-n-max 5', '')):
            with self.assertRaises(ValueError):
                diagnose.launch_text(bad, path)

    def test_launcher_uses_saved_libraries_before_inherited_current_build_paths(self):
        source = ('set -lx LD_LIBRARY_PATH /tmp/current/bin $LD_LIBRARY_PATH\n'
                  './build-rocm-moe-reference/bin/llama-server --cache-ram 0 '
                  '--spec-draft-n-max 3 --moe-cache-mib 15200,9600 --port 8080\n')
        path = Path('/tmp/saved reference/bin/llama-server')
        text = diagnose.launch_text(source, path)
        prefix = "set -lx LD_LIBRARY_PATH '/tmp/saved reference/bin' $LD_LIBRARY_PATH\n'/tmp/saved reference/bin/llama-server'"
        self.assertIn(prefix, text)
        self.assertLess(text.index('/tmp/current/bin'), text.index(prefix))

    def test_greedy_controls_and_prompt_replay(self):
        request = diagnose.greedy_request([1, 2, 3], 1, 42)
        self.assertEqual(request['prompt'], [1, 2, 3])
        self.assertEqual(request['samplers'], ['temperature'])
        self.assertEqual(request['temperature'], 0)
        self.assertEqual(request['dynatemp_range'], 0)
        self.assertEqual(request['mirostat'], 0)
        self.assertFalse(request['post_sampling_probs'])
        self.assertFalse(request['cache_prompt'])
        data = response([4])
        diagnose.validate_completion(data, request['prompt'], single=True)
        data['timings']['draft_n'] = 1
        with self.assertRaisesRegex(ValueError, 'speculative'):
            diagnose.validate_completion(data, request['prompt'], single=True)

    def test_missing_misaligned_cached_and_nongreedy_results_are_rejected(self):
        for change in ('cache', 'prompt', 'row', 'argmax', 'finite'):
            data = response([4])
            if change == 'cache': data['timings']['cache_n'] = 1
            if change == 'prompt': data['timings']['prompt_n'] = 2
            if change == 'row': data['completion_probabilities'][0]['id'] = 5
            if change == 'argmax': data['completion_probabilities'][0]['top_logprobs'][1]['logprob'] = 0.0
            if change == 'finite': data['completion_probabilities'][0]['top_logprobs'][0]['logprob'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnose.validate_completion(data, [1, 2, 3])

    def test_probabilities_after_first_changed_token_are_not_compared(self):
        left = response([10, 20, 30])
        right = response([10, 21, 31])
        right['completion_probabilities'][1]['top_logprobs'] = [dict(id=20, logprob=-0.2), dict(id=21, logprob=-0.1)]
        right['completion_probabilities'][2]['top_logprobs'] = [dict(id=30, logprob=-100), dict(id=31, logprob=-0.1)]
        result = diagnose.compare(left, right)
        self.assertEqual(result['first_different_token'], 1)
        self.assertEqual(result['compared_prefix_rows'], 2)
        self.assertAlmostEqual(result['max_shared_logprob_delta'], 0.1)
        self.assertEqual(result['at_first_difference']['left']['id'], 20)
        self.assertEqual(result['at_first_difference']['right']['id'], 21)

    def test_equal_tokens_do_not_hide_changes_in_the_top_probabilities(self):
        left = response([10])
        right = response([10], [[dict(id=10, logprob=-0.15), dict(id=222, logprob=-4)]])
        result = diagnose.compare(left, right)
        self.assertTrue(result['tokens_equal'])
        self.assertEqual(result['minimum_shared_top_ids'], 1)
        self.assertEqual(result['rows_with_top_id_changes'], 1)
        self.assertAlmostEqual(result['max_shared_logprob_delta'], 0.05)

    def test_shorter_completion_has_a_length_difference_without_a_missing_row_access(self):
        result = diagnose.compare(response([10]), response([10, 20]))
        self.assertFalse(result['tokens_equal'])
        self.assertEqual(result['first_different_token'], 1)
        self.assertNotIn('at_first_difference', result)


if __name__ == '__main__':
    unittest.main()
