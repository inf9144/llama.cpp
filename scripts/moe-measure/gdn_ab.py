#!/usr/bin/env python3
"""Measure just the GDN fusion with matched builds and alternating order."""
import argparse
import datetime
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

from build_pair import build_id
from rocm_build import ROOT, SDK, build_lock, build_targets, rocm_env

runner = import_module("stream-ab")
REVISIONS = {"baseline": "84f8054b405cd17c43bb13eceb36edd68ce0eb56",
             "candidate": "271f2968fe1eb6e5e98cfe032e491f2d8f8ba39a"}


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def build_pair(directory, env, jobs, *, revisions=REVISIONS, extra=(), reuse_baseline=False):
    source = directory / "source"
    build = directory / "build"
    snapshots = directory / "runtimes"
    manifest = {"sdk": env["ROCM_PATH"], "revisions": revisions}
    if extra:
        manifest["extra"] = list(extra)

    def check_snapshot(folder, variants=revisions):
        binaries = {variant: folder / variant / "llama-server" for variant in variants}
        ids = {variant: build_id(binary, env) for variant, binary in binaries.items()}
        for variant, identity in ids.items():
            if not revisions[variant].startswith(identity.split("-", 1)[1]):
                raise ValueError(f"{variant} meldet {identity}, erwartet {revisions[variant][:9]}")
        return binaries, ids

    previous = None
    if reuse_baseline and snapshots.exists():
        saved = json.loads((snapshots / "pair.json").read_text())
        old_candidate = saved.get("revisions", {}).get("candidate")
        if old_candidate and saved == dict(manifest, revisions=dict(revisions, candidate=old_candidate)) and saved != manifest:
            previous = snapshots
            snapshots = directory / f"runtimes-{revisions['candidate']}"

    if snapshots.exists():
        if json.loads((snapshots / "pair.json").read_text()) != manifest:
            raise ValueError(f"SDK, Revision oder Build-Optionen in {snapshots} weichen ab. Bitte ein freies --build-dir wählen.")
        return check_snapshot(snapshots)
    if previous is not None:
        check_snapshot(previous, ("baseline",))
        runner.say(f"Verwende vorhandene Baseline aus {previous}")

    if not source.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(source), revisions["baseline"]],
                       check=True)
    if (Path(git(source, "rev-parse", "--show-toplevel")).resolve() != source or
            git(source, "rev-parse", "--path-format=absolute", "--git-common-dir") !=
            git(ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir") or
            git(source, "status", "--porcelain") or git(source, "branch", "--show-current")):
        raise ValueError(f"{source} muss ein sauberer, separater Worktree ohne aktiven Branch sein.")

    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        staging = Path(tmp) / "runtimes"
        for variant, rev in revisions.items():
            if variant == "baseline" and previous is not None:
                shutil.copytree(previous / variant, staging / variant, symlinks=True)
                continue
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", rev], check=True)
            build_targets(source, build, env, jobs, False, ("llama-server",), extra=(
                "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF", *extra,
            ))
            shutil.copytree(build / "bin", staging / variant, symlinks=True)
        check_snapshot(staging)
        runner.json_write(staging / "pair.json", manifest)
        staging.rename(snapshots)
    return check_snapshot(snapshots)


def workload(result):
    t = result["timings"]
    if not 0 < t["draft_n_accepted"] <= t["draft_n"]:
        raise ValueError("DFlash-Verify war nicht aktiv.")
    return (result["content_sha256"], result["usage"]["prompt_tokens"], result["usage"]["completion_tokens"],
            t["draft_n"], t["draft_n_accepted"])


def summarize(pairs):
    summary = {}
    for phase in ("cold", "warm"):
        rows = [[pair[variant]["requests"][phase] for variant in REVISIONS] for pair in pairs]
        if len({workload(r) for row in rows for r in row}) != 1:
            raise ValueError(f"{phase} hat unterschiedliche Antworten oder DFlash-Zähler. Details in den Einzelruns.")
        summary[phase] = {}
        for key in ("prompt_ms", "predicted_ms", "request_wall_s"):
            raw = [[r[key] if key in r else r["timings"][key] for r in row] for row in rows]
            if any(value <= 0 for pair in raw for value in pair):
                raise ValueError(f"Fehlende Messzeit für {phase}/{key}")
            delta = [100*(b/a-1) for a, b in raw]
            summary[phase][key] = dict(baseline=statistics.median(a for a, _ in raw),
                                       candidate=statistics.median(b for _, b in raw),
                                       paired_delta_percent=statistics.median(delta),
                                       min_delta_percent=min(delta), max_delta_percent=max(delta))
        summary[phase]["decode_tps"] = {variant: statistics.median(
            1000*r["usage"]["completion_tokens"]/r["timings"]["predicted_ms"] for r in (row[i] for row in rows))
            for i, variant in enumerate(REVISIONS)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=SDK)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-rocm-gdn-ab")
    parser.add_argument("--jobs", type=int, default=28)
    parser.add_argument("--pairs", type=int, default=4, help="Gerade Zahl gewerteter A/B-Paare (Standard 4)")
    parser.add_argument("--request", type=Path, default=ROOT / "moe-first-request.json")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--shutdown-timeout", type=int, default=240)
    args = parser.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        parser.error("--pairs muss gerade und mindestens 2 sein")
    if not (ROOT / "moe-server-reference.fish").is_file() or runner.port_open():
        parser.error("moe-server-reference.fish und ein freier Port 8080 werden benötigt")
    try:
        request = runner.prepare_request(args.request, production_controls=True)
        env = rocm_env(args.rocm)
        env["GGML_CUDA_DISABLE_FUSION"] = "0"
        directory = args.build_dir.resolve()
        with build_lock(directory):
            binaries, ids = build_pair(directory, env, args.jobs)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = ROOT / "results" / f"gdn-ab-{stamp}-{os.getpid()}"
    output.mkdir(parents=True)
    report: dict = dict(complete=False, builds=ids, revisions=REVISIONS, sdk=env["ROCM_PATH"], pairs=[])
    args.draft_max = 3
    args.production_controls = True
    args.baseline_reference = binaries["baseline"]
    args.candidate_reference = binaries["candidate"]
    (output / "request.json").write_bytes(request)
    try:
        launchers = runner.prepare_launchers(ROOT, output, args, kinds=("reference",))
        report["binary_sha256"] = {k: v["reference"]["binary_sha256"] for k, v in launchers.items()}
        for i in range(args.pairs + 1):
            order = list(REVISIONS)
            if i % 2 == 0:
                order.reverse()
            label = f"pair-{i:02d}" if i else "prime"
            runner.say(f"{label} {' -> '.join(order)}" + (" (Dateicache vorwärmen)" if i == 0 else ""))
            pair = dict(order=order)
            for variant in order:
                phase = output / label / variant
                phase.parent.mkdir(parents=True, exist_ok=True)
                pair[variant] = runner.run_phase(ROOT, phase, "01-reference", launchers[variant]["reference"]["script"],
                                                 False, request, env, args, candidate=True, expected_build=ids[variant])
            if i:
                report["pairs"].append(pair)
            else:
                report["priming"] = pair
        report["summary"] = summarize(report["pairs"])
        for phase, metrics in report["summary"].items():
            rate = metrics["decode_tps"]
            print(f"{phase} Decode-Median {rate['baseline']:.2f} -> {rate['candidate']:.2f} t/s")
            for key, row in metrics.items():
                if key == "decode_tps":
                    continue
                print(f"  {key} {row['paired_delta_percent']:+.2f}% "
                      f"[{row['min_delta_percent']:+.2f}%, {row['max_delta_percent']:+.2f}%]")
        report["complete"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__} {error}"
        print(report["error"])
    runner.json_write(output / "report.json", report)
    print(f"Handoff-Archiv {runner.archive_suite(output)}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
