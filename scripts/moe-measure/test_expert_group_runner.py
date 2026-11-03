import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('fish'), 'Fish unavailable')
class ExpertGroupRunnerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.repo = self.root / 'checkout'
        self.home_dir = self.root / 'user home'
        self.sdk = self.home_dir / 'opt/rocm-10.1'
        self.tools = self.root / 'tools'
        self.calls = self.root / 'calls.jsonl'
        self.startup = self.root / 'startup-sdk.txt'
        self.tools.mkdir()
        (self.sdk / 'include/hip').mkdir(parents=True)
        scripts = self.repo / 'scripts/moe-measure'
        scripts.mkdir(parents=True)
        shutil.copy2(Path(__file__).with_name('run_expert_group.fish'), scripts)
        self.script = scripts / 'run_expert_group.fish'
        self.preamble = '''#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ['MOE_RUNNER_CALLS'], 'a') as log:
    log.write(json.dumps({'program': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:],
                          'sdk': os.environ.get('ROCM_PATH'), 'hip': os.environ.get('HIP_PATH')}) + '\\n')
'''
        self.executable(self.tools / 'git', self.preamble + "print('feature/moe-cache-prefill')\n")
        self.executable(self.tools / 'date', self.preamble + "print('fixture-' + str(len(pathlib.Path(os.environ['MOE_RUNNER_CALLS']).read_text().splitlines())))\n")
        self.executable(self.sdk / 'bin/amdclang++', self.preamble + 'sys.stdin.read()\n')
        self.executable(self.tools / 'cmake', self.preamble + '''
a = sys.argv[1:]
if '--build' in a:
    build = pathlib.Path(a[a.index('--build') + 1])
    targets = a[a.index('--target') + 1:]
    for name in targets:
        if name.startswith('-'): break
        out = build / 'bin' / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('#!/bin/sh\\necho passed\\n')
        out.chmod(0o755)
else:
    build = pathlib.Path(a[a.index('-B') + 1])
    compiler = next(x.split('=', 1)[1] for x in a if x.startswith('-DCMAKE_HIP_COMPILER='))
    mode = next(x.split('=', 1)[1] for x in a if x.startswith('-DLLAMA_MOE_MEASURE='))
    if os.environ.get('MOE_RUNNER_WRONG_COMPILER'): compiler = '/opt/rocm/bin/amdclang++'
    hip = 'OFF' if os.environ.get('MOE_RUNNER_DROP_HIP') else 'ON'
    if os.environ.get('MOE_RUNNER_DROP_MEASURE') and mode == 'ON': mode = 'OFF'
    build.mkdir(parents=True, exist_ok=True)
    (build / 'CMakeCache.txt').write_text('CMAKE_HIP_COMPILER:FILEPATH=' + compiler + '\\nGGML_HIP:BOOL=' + hip + '\\nLLAMA_MOE_MEASURE:BOOL=' + mode + '\\n')
''')
        (scripts / 'completion_diagnose.py').write_text(self.preamble + '''
out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
out.mkdir(parents=True)
comparison = dict(tokens_equal=True, rows_with_top_id_changes=0, max_shared_logprob_delta=0.0)
(out / 'report.json').write_text(json.dumps(dict(complete=True, comparisons={'repeat': comparison}, replay_comparisons={'0': comparison})))
''')
        (scripts / 'decode-ab.py').write_text(self.preamble)
        config = self.root / 'config/fish'
        config.mkdir(parents=True)
        (config / 'config.fish').write_text('set -gx ROCM_PATH /opt/rocm\nset -gx HIP_PATH /opt/rocm\n'
                                          "printf '%s' $ROCM_PATH > " + shlex.quote(str(self.startup)) + '\n')
        self.env = {**os.environ, 'HOME': str(self.home_dir), 'XDG_CONFIG_HOME': str(config.parent),
                    'ROCM_PATH': str(self.sdk), 'HIP_PATH': str(self.sdk),
                    'PATH': str(self.tools) + os.pathsep + os.environ['PATH'], 'MOE_RUNNER_CALLS': str(self.calls)}
        for phase in ('reference', 'measure'):
            build = self.repo / f'build-rocm-moe-stream-{phase}'
            self.executable(build / 'bin/llama-server', '#!/bin/sh\necho original-' + phase + '\n')
            compiler = '/opt/rocm/bin/amdclang++' if phase == 'reference' else str(self.sdk / 'bin/amdclang++')
            (build / 'CMakeCache.txt').write_text('CMAKE_HIP_COMPILER:FILEPATH=' + compiler + '\nGGML_HIP:BOOL=OFF\n')

    def executable(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def run_script(self, no_config=False, **environment):
        command = ['fish', *(['--no-config'] if no_config else []), str(self.script)]
        return subprocess.run(command, cwd=self.repo, env={**self.env, **environment},
                              text=True, capture_output=True, timeout=30)

    def recorded(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_startup_override_and_changed_compiler_recover_then_reuse(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.startup.read_text(), '/opt/rocm')
        calls = self.recorded()
        compile_checks = [c for c in calls if c['program'] == 'amdclang++']
        self.assertEqual(len(compile_checks), 1)
        self.assertEqual(compile_checks[0]['sdk'], str(self.sdk))
        self.assertEqual(compile_checks[0]['hip'], str(self.sdk))
        configurations = [c for c in calls if c['program'] == 'cmake' and '-B' in c['args']]
        self.assertEqual(len(configurations), 2)
        self.assertIn('--fresh', configurations[0]['args'])
        self.assertNotIn('--fresh', configurations[1]['args'])
        for call in configurations:
            self.assertIn(f'-DCMAKE_HIP_COMPILER={self.sdk}/bin/amdclang++', call['args'])
            self.assertIn(f'-DCMAKE_PREFIX_PATH={self.sdk}', call['args'])
            self.assertIn('-DGGML_HIP=ON', call['args'])
        builds = [c for c in calls if c['program'] == 'cmake' and '--build' in c['args']]
        self.assertEqual(len(builds), 2)
        self.assertTrue(all('-j28' in c['args'] for c in builds))
        self.assertEqual(calls[-1]['program'], 'decode-ab.py')
        self.assertIn('--compare-sampling', calls[-1]['args'])
        self.assertIn('--profile', calls[-1]['args'])
        before = self.repo / 'build-rocm-moe-before-expert-group/reference/llama-server'
        original = before.read_bytes()
        self.assertIn(b'original-reference', original)
        self.startup.unlink()
        result = self.run_script(no_config=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.startup.exists())
        self.assertEqual(before.read_bytes(), original)
        configurations = [c for c in self.recorded() if c['program'] == 'cmake' and '-B' in c['args']]
        self.assertTrue(all('--fresh' not in c['args'] for c in configurations[2:]))

    def test_missing_requested_sdk_does_not_fall_back(self):
        (self.sdk / 'bin/amdclang++').unlink()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Expected ROCm 10.1 SDK', result.stderr)
        self.assertFalse(any(c['program'] in ('cmake', 'amdclang++') for c in self.recorded()))

    def test_draft_runner_builds_only_reference_with_rocm10_and_j28(self):
        shutil.copy2(Path(__file__).with_name('run_draft_ab.fish'), self.script.parent)
        self.script = self.script.with_name('run_draft_ab.fish')
        self.script.with_name('draft-ab.py').write_text(self.preamble)
        result = self.run_script(no_config=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.recorded()
        configurations = [c for c in calls if c['program'] == 'cmake' and '-B' in c['args']]
        self.assertEqual(len(configurations), 1)
        self.assertIn('--fresh', configurations[0]['args'])
        self.assertIn(f'-DCMAKE_HIP_COMPILER={self.sdk}/bin/amdclang++', configurations[0]['args'])
        self.assertIn(f'-DCMAKE_PREFIX_PATH={self.sdk}', configurations[0]['args'])
        self.assertIn('-DLLAMA_MOE_MEASURE=OFF', configurations[0]['args'])
        builds = [c['args'] for c in calls if c['program'] == 'cmake' and '--build' in c['args']]
        self.assertEqual(builds, [['--build', 'build-rocm-moe-stream-reference', '--target', 'llama-server', '-j28']])
        self.assertEqual(calls[-1]['program'], 'draft-ab.py')
        self.assertFalse((self.repo / 'build-rocm-moe-before-expert-group').exists())

    def test_lost_hip_stops_before_build(self):
        result = self.run_script(MOE_RUNNER_DROP_HIP='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('HIP is disabled', result.stderr)
        self.assertFalse(any('--build' in c['args'] for c in self.recorded()))

    def test_wrong_compiler_after_configuration_stops_before_build(self):
        result = self.run_script(MOE_RUNNER_WRONG_COMPILER='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Wrong HIP compiler', result.stderr)
        self.assertFalse(any('--build' in c['args'] for c in self.recorded()))

    def test_lost_measurement_mode_stops_before_measurement_build(self):
        result = self.run_script(MOE_RUNNER_DROP_MEASURE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Wrong instrumentation mode', result.stderr)
        builds = [c for c in self.recorded() if c['program'] == 'cmake' and '--build' in c['args']]
        self.assertEqual(len(builds), 1)
        self.assertNotIn('decode-ab.py', [c['program'] for c in self.recorded()])


if __name__ == '__main__':
    unittest.main()
