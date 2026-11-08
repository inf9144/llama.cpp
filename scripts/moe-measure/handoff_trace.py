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
from rocm_build import ROOT, SDK, build_targets, rocm_env


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


def summarize_compact_ab(log):
    with log.open() as source:
        samples = [json.loads(line.removeprefix("upload_ab ")) for line in source if line.startswith("upload_ab ")]
    if len(samples) != 2*REPEATS*len(MISS_COUNTS):
        raise ValueError(f"Unvollständiger Transfervergleich ({len(samples)} Runden)")
    phases = {}
    for misses in MISS_COUNTS:
        phase = {}
        for mode in ("separate", "compact"):
            rows = [s for s in samples if s["misses"] == misses and s["mode"] == mode]
            values = sorted(r["graph_us"] for r in rows)
            expected_copies = 3*misses + 1 if mode == "separate" else 1
            if len(rows) != REPEATS or values[0] <= 0 or any(r["copies"] != expected_copies for r in rows):
                raise ValueError(f"Ungültige Samples für {misses} Misses / {mode}")
            byte_sizes = {(r["bytes"], r["h2d_bytes"]) for r in rows}
            if len(byte_sizes) != 1:
                raise ValueError(f"Wechselnde Nutzlast für {misses} Misses / {mode}")
            (payload, transferred), = byte_sizes
            if payload <= 0 or transferred < payload or (mode == "separate" and transferred != payload):
                raise ValueError(f"Ungültige H2D-Nutzlast für {misses} Misses / {mode}")
            phase[mode] = dict(rounds=len(rows), copies=expected_copies, payload_bytes=payload, h2d_bytes=transferred,
                               median_us=statistics.median(values), p95_us=values[math.ceil(0.95*len(values)) - 1])
        a, b = phase["separate"], phase["compact"]
        if a["payload_bytes"] != b["payload_bytes"]:
            raise ValueError(f"Unterschiedliche Nutzlast für {misses} Misses")
        phase["median_delta_percent"] = 100*(b["median_us"]/a["median_us"] - 1)
        phases[misses] = phase
    return dict(phases=phases, samples=samples)


def run_capture(command, log, env):
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
        raise RuntimeError(f"Test endete mit Status {rc}; Details in {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=SDK)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-rocm-moe-stream-measure")
    parser.add_argument("--jobs", type=int, default=28)
    parser.add_argument("--compact-ab", action="store_true", help="Sammeltransfer gegen einzelne Kopien ohne Profiler messen")
    args = parser.parse_args()
    build = args.build_dir.expanduser().resolve()
    try:
        env = rocm_env(args.rocm)
        profiler = Path(env["ROCM_PATH"]) / "bin/rocprofv3"
        if not args.compact_ab and not os.access(profiler, os.X_OK):
            raise ValueError(f"Fehlendes ausführbares Programm {profiler}")
        env.update(GGML_ROCM_MALLOC_STAGE_POOL="1", GGML_ROCM_MALLOC_STAGE_POOL_MIB="512",
                   GGML_ROCM_MALLOC_STAGE_POOL_TENSOR="")
        build_targets(ROOT, build, env, args.jobs, True, ("test-moe-stream",), extra=("-DLLAMA_BUILD_TESTS=ON",))
        env["LD_LIBRARY_PATH"] = str(build / "bin") + os.pathsep + env["LD_LIBRARY_PATH"]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "moe-compact-ab" if args.compact_ab else "moe-handoff-trace"
    output = ROOT / "results" / f"{name}-{stamp}-{os.getpid()}"
    output.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = dict(commit=commit, sdk=env["ROCM_PATH"], verify_width=4, compact_ab=args.compact_ab, devices={}, errors={},
                  runtime_env={key: env.get(key) for key in ("HSA_ENABLE_SDMA", "HSA_ENABLE_PEER_SDMA", "GGML_CUDA_DISABLE_GRAPHS")})
    for backend in ("ROCm0", "ROCm1"):
        raw = output / backend
        raw.mkdir()
        log = raw / "fixture.log"
        command = [str(build / "bin/test-moe-stream"), "--backend", backend,
                   "--compact-ab" if args.compact_ab else "--handoff-trace"]
        if not args.compact_ab:
            command = [str(profiler), "--hip-runtime-trace", "--kernel-trace", "--memory-copy-trace",
                       "--output-format", "csv", "--output-directory", str(raw), "--", *command]
        print(f"{backend} misst je {REPEATS} Runden mit 1, 3 und 8 Cache-Misses"
              + (" pro Variante, ohne Profiler" if args.compact_ab else ""), flush=True)
        try:
            run_capture(command, log, env)
            result = summarize_compact_ab(log) if args.compact_ab else summarize(raw, log)
            report["devices"][backend] = result
            for misses, phase in result["phases"].items():
                if args.compact_ab:
                    a, b = phase["separate"], phase["compact"]
                    print(f"  {misses} Misses, Verify-Graph p50/p95 in µs   "
                          f"{a['median_us']:.1f}/{a['p95_us']:.1f} -> {b['median_us']:.1f}/{b['p95_us']:.1f}   "
                          f"({phase['median_delta_percent']:+.1f}% p50)")
                    continue
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
