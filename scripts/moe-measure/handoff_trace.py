#!/usr/bin/env python3
"""Measure a reproducible split MoE cache-miss handoff on both RDNA3 devices."""
import argparse
import bisect
import collections
import csv
import datetime
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tarfile

from profile import union_ns


ROOT = Path(__file__).resolve().parents[2]
REPEATS = 64
MISS_COUNTS = (1, 3, 8)
TIMEOUT = 120
HOST_TIMES = (
    "observed_us", "plan_begin_us", "plan_end_us", "upload_begin_us", "upload_end_us",
    "ready_begin_us", "ready_end_us",
)


def span(row):
    start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
    if end < start or start <= 0:
        raise ValueError("Ungültiger Profiler-Zeitstempel")
    return start, end


def read_trace(raw, name):
    paths = list(raw.rglob(f"*{name}_trace.csv"))
    if len(paths) != 1:
        raise ValueError(f"Erwartet eine {name}_trace.csv, gefunden {len(paths)}")
    with paths[0].open(newline="") as source:
        return sorted(csv.DictReader(source), key=span)


def clipped_ns(spans, start, end):
    return union_ns((max(start, a), min(end, b)) for a, b in spans if a < end and b > start)


def summarize(raw, log):
    with log.open() as source:
        host = [json.loads(line.removeprefix("handoff ")) for line in source if line.startswith("handoff ")]
    expected = [28] + [misses for misses in MISS_COUNTS for _ in range(REPEATS)]
    if [sample["misses"] for sample in host] != expected:
        raise ValueError(f"Miss-Sequenz unvollständig oder unerwartet ({len(host)} von {len(expected)} Einträgen)")

    kernels = read_trace(raw, "kernel")
    publish = [row for row in kernels if "publish_lookup" in row["Kernel_Name"]]
    markers = [row for row in kernels if "scale_f32" in row["Kernel_Name"]]
    finish = [row for row in kernels if "finish_lookup" in row["Kernel_Name"]]
    waits = [row for row in kernels if any(name in row["Kernel_Name"].lower()
             for name in ("__amd_rocclr_batchmemop", "__amd_rocclr_streamopswait"))]
    apis = read_trace(raw, "hip_api")
    ready = [row for row in apis if row["Function"] == "hipStreamWriteValue64"]
    if any(len(rows) != len(host) for rows in (publish, markers, finish, ready)):
        raise ValueError(f"Trace unvollständig (Host={len(host)}, Publish={len(publish)}, "
                         f"Marker={len(markers)}, Finish={len(finish)}, Ready-API={len(ready)})")
    threads = {row["Thread_Id"] for row in ready}
    agents = {row["Agent_Id"] for row in publish}
    if len(threads) != 1 or len(agents) != 1:
        raise ValueError("Erwartet genau einen Service-Thread und einen GPU-Agent pro Prozess")
    thread, = threads
    agent, = agents
    submits = [row for row in apis if row["Thread_Id"] == thread and row["Function"].startswith("hipMemcpyAsync")]
    submit_times = [span(row)[0] for row in submits]
    copies = collections.defaultdict(list)
    for row in read_trace(raw, "memory_copy"):
        if row["Direction"] == "MEMORY_COPY_HOST_TO_DEVICE" and row["Destination_Agent_Id"] == agent:
            copies[row["Correlation_Id"]].append(span(row))
    wait_spans = [span(row) for row in waits if row["Agent_Id"] == agent]

    samples = []
    next_submit, last_finish = 0, 0
    for version, (h, p, m, f, r) in enumerate(zip(host, publish, markers, finish, ready), 1):
        pub_start, pub_end = span(p)
        marker_start, marker_end = span(m)
        finish_start, finish_end = span(f)
        ready_start, ready_end = span(r)
        times = [h[k] for k in HOST_TIMES]
        if h["version"] != version or times != sorted(times) or times[0] <= 0:
            raise ValueError(f"Ungültige Host-Reihenfolge bei Version {version}")
        if not (last_finish <= pub_start <= pub_end <= marker_start <= marker_end <= finish_start):
            raise ValueError(f"Publish, Marker und Finish liegen nicht in derselben Runde ({version})")
        if m["Agent_Id"] != agent or f["Agent_Id"] != agent:
            raise ValueError(f"Abweichender GPU-Agent in Runde {version}")
        last_finish = finish_end
        if h["bytes"] <= 0 or h["pinned_bytes"] != h["bytes"] or not h["staged_copies"] == h["ranges"] == 3*h["misses"] + 1:
            raise ValueError(f"Unvollständiges Pinned-Staging in Runde {version}")
        if not 0 <= h["pool_wait_us"] <= h["pool_acquire_us"] or h["staging_us"] < 0:
            raise ValueError(f"Ungültige Pool-Zeit in Runde {version}")

        # The traced API is nested inside the host stopwatch. This brackets
        # the clock offset even when rocprof and ggml use different epochs.
        # ggml_time_us rounds down to a whole microsecond.
        offset_lo = ready_end - (h["ready_end_us"] + 1)*1000
        offset_hi = ready_start - h["ready_begin_us"]*1000
        if offset_lo > offset_hi:
            raise ValueError(f"Host- und Profiler-Zeit lassen sich in Runde {version} nicht zuordnen")
        offset = (offset_lo + offset_hi)/2
        uncertainty = (offset_hi - offset_lo + 1000)/2

        end_submit = bisect.bisect_left(submit_times, ready_start, lo=next_submit)
        batch = submits[next_submit:end_submit]
        next_submit = end_submit
        if len(batch) != h["staged_copies"]:
            raise ValueError(f"HIP-Uploadaufrufe fehlen in Runde {version} ({len(batch)}/{h['staged_copies']})")
        # The same clock bracket must also contain the complete upload batch.
        upload_start = h["upload_begin_us"]*1000 + offset_lo
        upload_end = (h["upload_end_us"] + 1)*1000 + offset_hi
        copy_spans = []
        hip_copy_ns = 0
        patch_copy_traced = False
        for index, api in enumerate(batch):
            a, b = span(api)
            transfer = copies.pop(api["Correlation_Id"], [])
            is_patch = index == len(batch) - 1
            # The final range is the small map patch. It can use a path that
            # does not appear as an async-copy record. Require every weight.
            if not upload_start <= a <= b <= upload_end or (not transfer and not is_patch):
                raise ValueError(f"HIP-Aufruf oder H2D-Transfer fehlt im Zeitfenster von Runde {version}")
            if any(not pub_start <= start <= end <= finish_start for start, end in transfer):
                raise ValueError(f"H2D-Transfer liegt außerhalb von Publish/Finish in Runde {version}")
            copy_spans.extend(transfer)
            hip_copy_ns += b - a
            if is_patch:
                patch_copy_traced = bool(transfer)

        gap = finish_start - marker_end
        copy_in_gap = clipped_ns(copy_spans, marker_end, finish_start)
        wait = clipped_ns(wait_spans, marker_end, finish_start)
        samples.append({
            **h,
            "publish_end_to_observed_us": (h["observed_us"]*1000 + 500 + offset - pub_end)/1000,
            "clock_uncertainty_us": uncertainty/1000,
            "host_service_us": h["ready_end_us"] - h["observed_us"],
            "plan_us": h["plan_end_us"] - h["plan_begin_us"],
            "pack_us": h["upload_begin_us"] - h["plan_end_us"],
            "upload_us": h["upload_end_us"] - h["upload_begin_us"],
            "hip_copy_api_us": hip_copy_ns/1000,
            "hip_ready_api_us": (ready_end - ready_start)/1000,
            "shared_window_us": (marker_end - pub_end)/1000,
            "finish_gap_us": gap/1000,
            "traced_wait_us": wait/1000 if wait else None,
            "h2d_us": union_ns(copy_spans)/1000,
            "patch_copy_traced": patch_copy_traced,
            "h2d_in_finish_gap_us": copy_in_gap/1000,
            "finish_gap_without_h2d_us": (gap - copy_in_gap)/1000,
        })
    if next_submit != len(submits):
        raise ValueError("Unzugeordnete HIP-Uploads nach der letzten Runde")

    phases = {}
    for misses in MISS_COUNTS:
        group = [row for row in samples[1:] if row["misses"] == misses]
        metrics = {}
        for key in group[0]:
            if not key.endswith("_us") or key in HOST_TIMES:
                continue
            values = sorted(row[key] for row in group if row[key] is not None)
            if values:
                metrics[key] = dict(n=len(values), median=statistics.median(values),
                                    p95=values[math.ceil(0.95*len(values)) - 1])
        phases[misses] = dict(rounds=len(group), bytes=group[0]["bytes"], ranges=group[0]["ranges"], us=metrics)
    return dict(agent=agent, rounds=len(samples) - 1, phases=phases, samples=samples,
                graph_launches=sum(row["Function"] == "hipGraphLaunch" and span(row)[0] >= span(publish[0])[0]
                                   for row in apis))


def run_profile(command, log, env):
    with log.open("w") as output:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                rc = process.wait(timeout=TIMEOUT)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # rocprof starts the executable as a child. Stop both so a
                # stalled wait cannot keep using the GPU after this run.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
    if rc != 0:
        raise RuntimeError(f"rocprofv3/Test endete mit Status {rc}; Details in {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=Path.home() / "opt/rocm-10.1")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-rocm-moe-stream-measure")
    parser.add_argument("--jobs", type=int, default=28)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs muss positiv sein")
    rocm = args.rocm.expanduser().resolve()
    build = args.build_dir.expanduser().resolve()
    compiler = rocm / "bin/amdclang++"
    profiler = rocm / "bin/rocprofv3"
    for path in (compiler, profiler):
        if not os.access(path, os.X_OK):
            parser.error(f"Fehlendes ausführbares Programm {path}")
    if not (rocm / "include/hip").is_dir():
        parser.error(f"HIP-Header fehlen unter {rocm}/include/hip")
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_HIP_COMPILER:"):
                existing = Path(line.split("=", 1)[1])
                if existing.resolve() != compiler.resolve():
                    parser.error(f"{build} verwendet {existing}. Bitte ein freies --build-dir wählen.")

    env = dict(os.environ, ROCM_PATH=str(rocm), HIP_PATH=str(rocm),
               PATH=os.pathsep.join([str(rocm / "bin"), os.environ.get("PATH", "")]),
               LD_LIBRARY_PATH=os.pathsep.join([str(rocm / "lib"), str(rocm / "lib64")] +
                                               os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)).rstrip(os.pathsep),
               GGML_ROCM_MALLOC_STAGE_POOL="1", GGML_ROCM_MALLOC_STAGE_POOL_MIB="512",
               GGML_ROCM_MALLOC_STAGE_POOL_TENSOR="")
    configure = [
        "cmake", "-S", str(ROOT), "-B", str(build),
        f"-DCMAKE_HIP_COMPILER={compiler}", f"-DCMAKE_PREFIX_PATH={rocm}",
        "-C", str(ROOT / "scripts/moe-measure/rocm-sdk-options.cmake"),
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=ON", "-DGGML_HIP_GRAPHS=ON",
        "-DGPU_TARGETS=gfx1100;gfx1101", "-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101",
        "-DLLAMA_MOE_MEASURE=ON", "-DLLAMA_BUILD_TESTS=ON",
    ]
    if not cache.is_file():
        configure.extend(["-G", "Ninja"])
    print(f"Baue test-moe-stream mit {rocm} und -j{args.jobs}", flush=True)
    subprocess.run(configure, cwd=ROOT, env=env, check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "test-moe-stream", f"-j{args.jobs}"],
                   cwd=ROOT, env=env, check=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = ROOT / "results" / f"moe-handoff-trace-{stamp}-{os.getpid()}"
    output.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = dict(commit=commit, sdk=str(rocm), verify_width=4, devices={}, errors={})
    for backend in ("ROCm0", "ROCm1"):
        raw = output / backend
        raw.mkdir()
        log = raw / "fixture.log"
        command = [
            str(profiler), "--hip-runtime-trace", "--kernel-trace", "--memory-copy-trace",
            "--output-format", "csv", "--output-directory", str(raw), "--",
            str(build / "bin/test-moe-stream"), "--backend", backend, "--handoff-trace",
        ]
        print(f"{backend} misst je {REPEATS} Runden mit 1, 3 und 8 Cache-Misses", flush=True)
        try:
            run_profile(command, log, env)
            result = summarize(raw, log)
            report["devices"][backend] = result
            for misses, phase in result["phases"].items():
                med = {key: value["median"] for key, value in phase["us"].items()}
                print(f"  {misses} Misses, Median in µs   Planung {med['plan_us']:.1f}   "
                      f"Staging {med['staging_us']:.1f}   HIP-Copy-API {med['hip_copy_api_us']:.1f}   "
                      f"H2D {med['h2d_us']:.1f}   GPU-Lücke {med['finish_gap_us']:.1f}")
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
            report["errors"][backend] = str(error)
            print(f"{backend} unvollständig ({error}). Log unter {log}", flush=True)
        except KeyboardInterrupt:
            report["errors"][backend] = "Abgebrochen"
            break
    report["complete"] = len(report["devices"]) == 2
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    archive = output.with_name(f"{output.name}-handoff.tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(output, arcname=output.name)
    print(f"Handoff-Archiv {archive}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
