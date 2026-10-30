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
    def run_mocked_suite(self, root, fail=False, different_prompt=False):
        for case in diagnose.CASES:
            binary = root / diagnose.runner.BINARIES[case][0]
            binary.parent.mkdir(parents=True)
            binary.write_bytes(case.encode())
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
            data['timings']['draft_n'] = 0 if folder.name == 'replay' else 3
            return data

        with mock.patch.object(sys, 'argv', ['completion_diagnose.py', '--repo', str(root), '--output', 'out']), \
                mock.patch.object(diagnose, 'server', fake_server), mock.patch.object(diagnose, 'http', fake_http), \
                contextlib.redirect_stdout(io.StringIO()):
            status = diagnose.main()
        return status, sessions, requests, json.loads((root / 'out/report.json').read_text())

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
        self.assertEqual(cached.replace("'/tmp/my repo/server'", '/tmp/other/server'), other)
        self.assertIn('--moe-cache-mib 15200,9600', cached)
        self.assertIn('--spec-draft-n-max 3', cached)
        self.assertIn('--ctx-size 262144 --temp 0.7', cached)
        for bad in (source.replace('--port 8080', '--port 8081'),
                    source.replace('--moe-cache-mib 15200,9600', '--moe-cache-mib 0,0'),
                    source + '--cache-ssd-dir /tmp/cache',
                    source.replace('--spec-draft-n-max 5', '')):
            with self.assertRaises(ValueError):
                diagnose.launch_text(bad, path)

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
