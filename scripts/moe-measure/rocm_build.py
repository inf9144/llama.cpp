"""Shared ROCm 10.1 configuration for the local benchmark runners."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SDK = Path.home() / "opt/rocm-10.1"
TARGETS = "gfx1100;gfx1101"


@contextmanager
def build_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".moe-benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"In {directory} läuft bereits ein Benchmark-Build.") from error
        yield


def rocm_env(sdk):
    sdk = sdk.expanduser().resolve()
    if not os.access(sdk / "bin/amdclang++", os.X_OK) or not (sdk / "include/hip").is_dir():
        raise ValueError(f"ROCm SDK mit amdclang++ und HIP-Headern fehlt unter {sdk}")
    return dict(os.environ, ROCM_PATH=str(sdk), HIP_PATH=str(sdk),
                PATH=os.pathsep.join(filter(None, (str(sdk / "bin"), os.environ.get("PATH")))),
                LD_LIBRARY_PATH=os.pathsep.join(filter(None, (
                    str(sdk / "lib"), str(sdk / "lib64"), os.environ.get("LD_LIBRARY_PATH")))))


def build_targets(source, build, env, jobs, measure, targets, *, extra=()):
    if jobs < 1:
        raise ValueError("--jobs muss positiv sein")
    sdk = Path(env["ROCM_PATH"])
    compiler = sdk / "bin/amdclang++"
    with build_lock(build):
        cache = build / "CMakeCache.txt"
        if cache.is_file():
            entries = dict(line.split("=", 1) for line in cache.read_text().splitlines()
                           if line and line[0] not in "#/" and "=" in line)
            for key, expected in (("CMAKE_HOME_DIRECTORY", source), ("CMAKE_HIP_COMPILER", compiler)):
                existing = next((value for entry, value in entries.items() if entry.split(":", 1)[0] == key), None)
                if existing and Path(existing).resolve() != expected.resolve():
                    raise ValueError(f"{build} verwendet {existing} für {key}. Bitte ein freies --build-dir wählen.")

        command = [
            "cmake", "-S", str(source), "-B", str(build),
            f"-DCMAKE_HIP_COMPILER={compiler}", f"-DCMAKE_PREFIX_PATH={sdk}",
            "-C", str(ROOT / "scripts/moe-measure/rocm-sdk-options.cmake"),
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=ON", "-DGGML_HIP_GRAPHS=ON",
            f"-DGPU_TARGETS={TARGETS}", f"-DCMAKE_HIP_ARCHITECTURES={TARGETS}",
            f"-DLLAMA_MOE_MEASURE={'ON' if measure else 'OFF'}", *extra,
        ]
        if not cache.is_file():
            command.extend(["-G", "Ninja"])
        print(f"Baue {' '.join(targets)} in {build} mit {sdk} und -j{jobs}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", *targets, f"-j{jobs}"],
                       cwd=ROOT, env=env, check=True)
