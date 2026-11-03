import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('cmake'), 'CMake unavailable')
class RocmSdkOptionsTests(unittest.TestCase):
    def test_existing_flags_and_package_cache_use_selected_sdk(self):
        options = Path(__file__).with_name('rocm-sdk-options.cmake').resolve()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sdk = root / 'selected SDK'
            (sdk / 'include/hip').mkdir(parents=True)
            for lib, package in (('lib', 'hip'), ('lib64', 'hip-lang'), ('lib', 'hipblas')):
                path = sdk / lib / 'cmake' / package
                path.mkdir(parents=True)
                (path / f'{package}-config.cmake').touch()
            source = root / 'source'
            source.mkdir()
            (source / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.21)\nproject(sdk_probe NONE)\n')
            build = root / 'build'
            env = {**os.environ, 'ROCM_PATH': str(sdk)}
            command = ['cmake', '-S', str(source), '-B', str(build), '-C', str(options)]
            old = ['-DCMAKE_HIP_FLAGS:STRING=-O3 -g', '-DCMAKE_PREFIX_PATH:STRING=/custom/prefix',
                   '-Dhip_DIR:PATH=/opt/rocm/lib/cmake/hip', '-Drocblas_DIR:PATH=/opt/rocm/lib/cmake/rocblas']
            subprocess.run(['cmake', '-S', str(source), '-B', str(build), *old], env=env, check=True, capture_output=True)

            def cache():
                result = {}
                for line in (build / 'CMakeCache.txt').read_text().splitlines():
                    if '=' in line and not line.startswith(('#', '//')):
                        key, value = line.split('=', 1)
                        result[key.split(':', 1)[0]] = value
                return result

            subprocess.run(command, env=env, check=True, capture_output=True)
            first = cache()
            flags = shlex.split(first['CMAKE_HIP_FLAGS'])
            self.assertEqual(flags[:2], ['-O3', '-g'])
            self.assertIn(f'--rocm-path={sdk}', flags)
            self.assertIn(f'--hip-path={sdk}', flags)
            self.assertIn(f'-I{sdk}/include', flags)
            self.assertEqual(first['hip_DIR'], str(sdk / 'lib/cmake/hip'))
            self.assertEqual(first['hip-lang_DIR'], str(sdk / 'lib64/cmake/hip-lang'))
            self.assertEqual(first['CMAKE_HIP_COMPILER_ROCM_ROOT'], str(sdk))
            self.assertNotIn('rocblas_DIR', first)
            self.assertEqual(first['CMAKE_PREFIX_PATH'].split(';')[0], str(sdk))
            self.assertIn('/custom/prefix', first['CMAKE_PREFIX_PATH'].split(';'))
            subprocess.run(command, env=env, check=True, capture_output=True)
            second = cache()
            self.assertEqual(first['CMAKE_HIP_FLAGS'], second['CMAKE_HIP_FLAGS'])
            self.assertEqual(first['CMAKE_PREFIX_PATH'], second['CMAKE_PREFIX_PATH'])

    def test_missing_selected_headers_stop_configuration(self):
        options = Path(__file__).with_name('rocm-sdk-options.cmake').resolve()
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(['cmake', '-P', str(options)],
                                    env={**os.environ, 'ROCM_PATH': folder}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must contain include/hip', result.stderr)


if __name__ == '__main__':
    unittest.main()
