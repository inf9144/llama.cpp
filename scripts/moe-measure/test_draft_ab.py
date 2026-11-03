import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('draft_ab', Path(__file__).with_name('draft-ab.py'))
assert spec is not None and spec.loader is not None
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def trace(devices):
    return ('load_tensors: offloaded 49/49 layers to GPU\n'
            'load_tensors: ROCm0 model buffer size = 2586.50 MiB\n'
            'load_tensors: ROCm1 model buffer size = 1838.81 MiB\n'
            'load_tensors: offloaded 6/6 layers to GPU\n' +
            ''.join(f'load_tensors: {device} model buffer size = 300.00 MiB\n' for device in devices) +
            'common_speculative_impl_draft_dflash: - n_max=3\n')


class DraftSuiteTests(unittest.TestCase):
    def test_launcher_replaces_draft_aliases_and_pins_shared_head_before_overrides(self):
        binary = Path('/build with spaces/bin/server')
        text = ('set -lx LD_LIBRARY_PATH /saved/lib $LD_LIBRARY_PATH\n' + shlex.quote(str(binary)) + ' \\\n'
                '  --device ROCm0,ROCm1 --tensor-split 60,40 --moe-cache-mib 15200,9600 \\\n'
                "  --override-tensor '^output.*=ROCm1' \\\n"
                '  --spec-draft-n-max 3 --device-draft="ROCm0,ROCm1" --spec-draft-sampling greedy\n')
        actual = suite.draft_launcher(text, binary, 'probabilistic', 'ROCm0')
        self.assertIn('--device ROCm0,ROCm1 --tensor-split 60,40 --moe-cache-mib 15200,9600', actual)
        self.assertIn('--spec-draft-sampling probabilistic', actual)
        self.assertIn('--spec-draft-device ROCm0', actual)
        self.assertNotIn('--device-draft', actual)
        pin = '--override-tensor ' + shlex.quote(suite.SHARED_HEAD_OVERRIDE)
        self.assertLess(actual.index(pin), actual.index("--override-tensor '^output.*=ROCm1'"))
        self.assertTrue(actual.startswith('set -lx LD_LIBRARY_PATH /saved/lib $LD_LIBRARY_PATH\n'))
        for bad in ('--spec-draft-device ROCm0 --device-draft ROCm1', '--spec-draft-device \\\n'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                suite.draft_launcher(shlex.quote(str(binary)) + ' ' + bad, binary, 'greedy', 'ROCm0')

    def test_placement_excludes_target_buffers_and_rejects_silent_split(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'trace.log'
            log.write_text(trace(['ROCm0']))
            result = suite.draft_placement(log, 'ROCm0')
            self.assertEqual(result['model_buffers_mib'], {'ROCm0': 300.0})
            self.assertIn('not a process peak', result['limitation'])
            log.write_text(trace(['ROCm0', 'ROCm1']))
            with self.assertRaisesRegex(ValueError, 'placement mismatch'):
                suite.draft_placement(log, 'ROCm0')

    @unittest.skipUnless(shutil.which('fish'), 'Fish unavailable')
    def test_factorial_rounds_reverse_order_and_keep_request_and_templates(self):
        self.run_fixture()

    @unittest.skipUnless(shutil.which('fish'), 'Fish unavailable')
    def test_failures_keep_partial_archive_and_stop(self):
        for failure in ('placement', 'build', 'tokens'):
            with self.subTest(failure=failure):
                self.run_fixture(failure)

    def run_fixture(self, failure=None):
        with tempfile.TemporaryDirectory(prefix='draft suite ') as folder:
            root = Path(folder)
            request = dict(messages=[dict(role='user', content='fixed prompt')], seed=42,
                           max_tokens=128, stream=False, cache_prompt=False, temperature=0.7,
                           dynatemp_range=0.4, top_n_sigma=1.5, dry_multiplier=0.2)
            original = json.dumps(request)
            (root / 'moe-first-request.json').write_text(original)
            template = ('#!/usr/bin/env fish\n'
                        './build-rocm-moe-reference/bin/llama-server \\\n'
                        '  --cache-ram 0 --port 8080 --device ROCm0,ROCm1 \\\n'
                        '  --tensor-split 60,40 --moe-cache-mib 15200,9600 --spec-draft-n-max 7\n')
            (root / 'moe-server-reference.fish').write_text(template)
            binary = root / 'build with spaces/bin/llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('unused binary')
            binary.with_name('libllama.so').write_text('same library')
            (binary.parent.parent / 'CMakeCache.txt').write_text('LLAMA_MOE_MEASURE:BOOL=OFF\n')
            calls = []

            def run(repo, target, phase, script, profiler, payload, env, args, *, candidate, fish_no_config):
                self.assertEqual(repo, root)
                self.assertEqual(phase, '01-reference')
                self.assertFalse(profiler)
                self.assertTrue(candidate and fish_no_config)
                self.assertNotIn('LLAMA_ARG_SPEC_DRAFT_SAMPLING', env)
                self.assertNotIn('LLAMA_ARG_SPEC_DRAFT_DEVICE', env)
                self.assertEqual(env['LLAMA_ARG_OVERRIDE_TENSOR'], suite.SHARED_HEAD_OVERRIDE + ',^blk.*=CPU')
                variant = target.parent.name
                calls.append(variant)
                generated = json.loads(payload)
                for key in ('messages', 'seed', 'max_tokens', 'stream', 'cache_prompt'):
                    self.assertEqual(generated[key], request[key])
                self.assertEqual(generated['samplers'], ['top_k', 'top_p', 'temperature'])
                self.assertEqual((generated['temperature'], generated['top_k'], generated['top_p']), (1, 20, 0.95))
                self.assertEqual((generated['dry_multiplier'], generated['dynatemp_range'], generated['top_n_sigma']), (0, 0, -1))
                launcher = script.read_text()
                self.assertIn('--spec-draft-n-max 3', launcher)
                self.assertIn('--override-tensor ' + shlex.quote(suite.SHARED_HEAD_OVERRIDE), launcher)
                self.assertIn('--device ROCm0,ROCm1', launcher)
                sampling = 'probabilistic' if variant.startswith('prob-') else 'greedy'
                devices = ['ROCm0'] if variant.endswith('rocm0') else ['ROCm0', 'ROCm1']
                self.assertIn('--spec-draft-sampling ' + sampling, launcher)
                self.assertIn('--spec-draft-device ' + ','.join(devices), launcher)
                self.assertIn(shlex.quote(str(binary.parent)) + ' $LD_LIBRARY_PATH', launcher)
                target.mkdir(parents=True)
                (target / 'instrumented').mkdir()
                if failure == 'placement' and len(calls) == 3:
                    devices = ['ROCm0', 'ROCm1']
                (target / 'instrumented/trace.log').write_text(trace(devices))
                (target / 'props.json').write_text(json.dumps(dict(build_info='other' if failure == 'build' and len(calls) == 3 else 'same')))
                row = dict(usage=dict(prompt_tokens=5121, completion_tokens=127 if failure == 'tokens' and len(calls) == 3 else 128),
                           timings=dict(predicted_per_second=70 if sampling == 'probabilistic' else 60,
                                        draft_n=150, draft_n_accepted=110 if sampling == 'probabilistic' else 90),
                           request_wall_s=10, content_sha256=variant)
                return dict(requests=dict(cold=row, warm=row))

            argv = ['draft-ab.py', '--repo', str(root), '--binary', str(binary), '--output', 'suite']
            with patch.object(sys, 'argv', argv), patch.object(suite.runner, 'port_open', return_value=False), \
                    patch.object(suite.runner, 'run_phase', side_effect=run), \
                    patch.object(suite.subprocess, 'check_output', return_value='fixture-head'), \
                    patch.dict(os.environ, {'LLAMA_ARG_SPEC_DRAFT_SAMPLING': 'unexpected', 'LLAMA_ARG_SPEC_DRAFT_DEVICE': 'ROCm2', 'LLAMA_ARG_OVERRIDE_TENSOR': '^blk.*=CPU'}), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(suite.main(), 1 if failure else 0)
            report = json.loads((root / 'suite/draft-suite.json').read_text())
            self.assertEqual(report['complete'], failure is None)
            self.assertTrue(report['shared_head']['target_placement_changes'])
            self.assertFalse(report['shared_head']['extra_weight_copy'])
            self.assertEqual((root / 'moe-first-request.json').read_text(), original)
            self.assertEqual((root / 'moe-server-reference.fish').read_text(), template)
            with tarfile.open(root / 'suite-handoff.tar.gz') as archive:
                self.assertIn('suite/draft-suite.json', archive.getnames())
                self.assertIn('suite/source-request.json', archive.getnames())
            if failure:
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(report['runs']), 3)
                self.assertNotIn('summary', report)
                self.assertTrue(report['errors'])
            else:
                expected = ['greedy-split', 'prob-split', 'greedy-rocm0', 'prob-rocm0']
                self.assertEqual(calls, expected + list(reversed(expected)))
                self.assertEqual(len(report['runs']), 8)
                self.assertEqual(report['summary']['prob-split']['warm']['decode_tps_median'], 70)
                self.assertAlmostEqual(report['summary']['prob-split']['warm']['decode_change_percent'], 100/6)
                self.assertEqual(report['summary']['prob-split']['warm']['content_hashes'], ['prob-split']*2)
                self.assertIn('keine Verteilungsaequivalenz', output.getvalue())


if __name__ == '__main__':
    unittest.main()
