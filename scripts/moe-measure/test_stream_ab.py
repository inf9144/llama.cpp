import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('stream_ab', Path(__file__).with_name('stream-ab.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StreamActivationTests(unittest.TestCase):
    def test_scratch_required_on_each_cache_gpu(self):
        caches = 'ROCm0 MoE cache size = 15200.0 MiB\nROCm1 MoE cache size = 9600.0 MiB\n'
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'trace.log'
            for scratch in ('', 'ROCm0 bounded miss scratch = 32 MiB\n'):
                trace.write_text(caches + scratch)
                with self.assertRaises(RuntimeError):
                    runner.require_candidate_scratch(trace)
            trace.write_text(caches + 'ROCm0 bounded miss scratch = 32 MiB\nROCm1 bounded miss scratch = 32 MiB\n')
            runner.require_candidate_scratch(trace)

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
