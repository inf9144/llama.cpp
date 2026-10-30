import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('stream_ab', Path(__file__).with_name('stream-ab.py'))
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

LINKS = ('ROCm0 host link = 16.0 GT/s x16, limiting device 0000:03:00.0 (GPU 0000:03:00.0)\n'
         'ROCm1 host link = 16.0 GT/s x4, limiting device 0000:00:02.0 (GPU 0000:14:00.0)\n')


class StreamActivationTests(unittest.TestCase):
    def test_scratch_required_on_each_cache_gpu(self):
        caches = ('ROCm0 MoE cache size = 15200.0 MiB\nROCm1 MoE cache size = 9600.0 MiB\n' + LINKS +
                  'ROCm0 prefill execution = ROCm0\nROCm1 prefill execution = ROCm1\n')
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'trace.log'
            for scratch in ('', 'ROCm0 bounded miss scratch = 32 MiB\n'):
                trace.write_text(caches + scratch)
                with self.assertRaises(RuntimeError):
                    runner.require_candidate_scratch(trace)
            trace.write_text(caches + 'ROCm0 bounded miss scratch = 32 MiB\nROCm1 bounded miss scratch = 32 MiB\n')
            self.assertEqual(runner.require_candidate_scratch(trace), 'ROCm0 -> ROCm0, ROCm1 -> ROCm1')

    def test_fallback_and_missing_stage_are_rejected(self):
        active = {'kinds': {'cold': {'hybrid_plan': 235, 'stage_h2d': 1}, 'warm': {'hybrid_plan': 235}}}
        runner.require_candidate_stream(active)
        active['kinds']['warm']['work_copy'] = 705
        with self.assertRaises(RuntimeError):
            runner.require_candidate_stream(active)
        active['kinds']['warm'].pop('work_copy')
        active['kinds']['cold'].pop('stage_h2d')
        with self.assertRaises(RuntimeError):
            runner.require_candidate_stream(active)

    def test_redirect_requires_all_announced_execution_devices(self):
        text = (LINKS + 'ROCm0 MoE cache size = 15200.0 MiB\nROCm1 MoE cache size = 9600.0 MiB\n'
                'ROCm0 bounded miss scratch = 29 MiB\n'
                'ROCm0 prefill execution = ROCm0\nROCm1 prefill execution = ROCm0\n')
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'trace.log'
            trace.write_text(text)
            self.assertEqual(runner.require_candidate_scratch(trace), 'ROCm0 -> ROCm0, ROCm1 -> ROCm0')
            trace.write_text(text + 'ROCm1 prefill execution = ROCm1\n')
            with self.assertRaises(RuntimeError):
                runner.require_candidate_scratch(trace)

    def test_multi_gpu_topology_must_be_visible(self):
        text = ('ROCm0 MoE cache size = 15200.0 MiB\nROCm1 MoE cache size = 9600.0 MiB\n'
                'ROCm0 bounded miss scratch = 29 MiB\n'
                'ROCm0 prefill execution = ROCm0\nROCm1 prefill execution = ROCm0\n')
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'trace.log'
            for links in ('', LINKS.splitlines(keepends=True)[0] + 'ROCm1 host link = unknown\n'):
                trace.write_text(text + links)
                with self.assertRaisesRegex(RuntimeError, 'Topologie unvollständig'):
                    runner.require_candidate_scratch(trace)
            trace.write_text(text + LINKS)
            self.assertEqual(runner.require_candidate_scratch(trace), 'ROCm0 -> ROCm0, ROCm1 -> ROCm0')

    def test_zero_acceptance_after_many_drafts_is_rejected(self):
        with self.assertRaises(RuntimeError):
            runner.require_candidate_acceptance({'timings': {'draft_n': 375, 'draft_n_accepted': 0}})
        runner.require_candidate_acceptance({'timings': {'draft_n': 144, 'draft_n_accepted': 78}})
        runner.require_candidate_acceptance({'timings': {'draft_n': 3, 'draft_n_accepted': 0}})

    def test_early_candidate_failure_can_still_print_and_archive(self):
        # main() prints before creating the handoff. Missing candidate results
        # must not raise another exception that skips that archive operation.
        result = {'request_wall_s': 1, 'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                  'content_sha256': 'a'*64, 'timings': {}}
        report = {'baseline': {'01-reference': {'requests': {'cold': result, 'warm': result}}},
                  'candidate': {}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            runner.print_comparison(report)
        self.assertIn('baseline', output.getvalue())
        self.assertNotIn('Delta Candidate', output.getvalue())


if __name__ == '__main__':
    unittest.main()
