#!/usr/bin/env python3
"""Compare codex-compat and the upstream merge on the same ROCm workload."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

import completion_diagnose as diag
from gdn_ab import build_pair, git, workload
from rocm_build import ROOT, SDK, build_lock, rocm_env
from rollback_ab import SLOT, encode, launcher, prepare_request

runner = diag.runner
BASELINE = "8e4537c9713a05d2a131300fa8485964e7013f38"
UPSTREAM = "0a94a3f289687883617b9a2abbc7e0ade2d6b6c4"
VARIANTS = ("baseline", "candidate")


def summarize(pairs, phase):
    rows = [[pair[variant][phase] for variant in VARIANTS] for pair in pairs]

    def signature(result):
        return (*workload(result), result["timings"]["cache_n"], result["timings"]["prompt_n"])

    if len({signature(r) for row in rows for r in row}) != 1:
        raise ValueError(f"{phase}: Ausgabe, Prompt-Reuse oder DFlash-Zähler unterscheiden sich")

    summary = {}
    for key in ("prompt_ms", "predicted_ms", "request_wall_s"):
        times = [[r["request_wall_s"] if key == "request_wall_s" else r["timings"][key] for r in row] for row in rows]
        if any(t <= 0 for row in times for t in row):
            raise ValueError(f"{phase}: fehlende Messzeit für {key}")
        summary[key] = {
            "median": {variant: statistics.median(row[i] for row in times) for i, variant in enumerate(VARIANTS)},
            "paired_delta_pct": [100*(b/a - 1) for a, b in times],
        }
    for kind in ("prefill", "decode"):
        summary[f"{kind}_tps"] = {
            variant: statistics.median(runner.rate(row[i], kind) for row in rows)
            for i, variant in enumerate(VARIANTS)
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=SDK)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-rocm-upstream-ab")
    parser.add_argument("--jobs", type=int, default=28)
    parser.add_argument("--pairs", type=int, default=2, help="Gerade Anzahl A/B-Paare pro Fall")
    parser.add_argument("--case", choices=("both", "short", "long"), default="both")
    parser.add_argument("--launcher", type=Path, default=ROOT / "moe-server-reference.fish")
    parser.add_argument("--request", type=Path, default=ROOT / "moe-first-request.json")
    parser.add_argument("--long-request", type=Path, help="Optionaler langer Chat-Request statt synthetischer Wiederholung")
    parser.add_argument("--repeat-prefix", type=int, help="Wiederholungen für langen Kontext (Standard 36, mit --long-request 1)")
    parser.add_argument("--min-context", type=int, default=98304)
    parser.add_argument("--cache-mib", default="15600,11300")
    parser.add_argument("--request-timeout", type=int, default=1800)
    args = parser.parse_args()
    if (args.pairs < 2 or args.pairs % 2 or args.jobs < 1 or args.min_context < 1 or args.request_timeout < 1 or
            (args.repeat_prefix is not None and args.repeat_prefix < 1) or
            not re.fullmatch(r"[1-9]\d*,[1-9]\d*", args.cache_mib)):
        parser.error("Gerade Paarzahl ab 2 und positive Budgets, Kontextgrenzen und Build-Parameter erforderlich")
    if runner.port_open():
        parser.error("Port 8080 muss für den Benchmark frei sein")

    cases = ("short", "long") if args.case == "both" else (args.case,)
    budget = tuple(map(int, args.cache_mib.split(",")))
    try:
        source = args.launcher.read_text()
        requests = {}
        if "short" in cases:
            requests["short"] = dict(prepare_request(args.request, 1, 128), cache_prompt=False)
        if "long" in cases:
            repeat = args.repeat_prefix or (1 if args.long_request else 36)
            requests["long"] = prepare_request(args.long_request or args.request, repeat, 256)
        revisions = dict(baseline=BASELINE, candidate=git(ROOT, "rev-parse", "HEAD"))
        for ancestor in (BASELINE, UPSTREAM):
            subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", ancestor, revisions["candidate"]],
                           check=True)
        env = rocm_env(args.rocm)
        directory = args.build_dir.resolve()
        with build_lock(directory):
            binaries, builds = build_pair(directory, env, args.jobs, revisions=revisions)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = ROOT / "results" / f"upstream-ab-{stamp}"
    out.mkdir(parents=True)
    payloads = {name: encode(req) for name, req in requests.items()}
    report: dict = dict(complete=False, revisions=revisions, upstream=UPSTREAM, builds=builds, sdk=env["ROCM_PATH"],
                        cache_mib=budget, request_sha256={k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()},
                        pairs={}, summary={}, workload_differences={})

    try:
        with tempfile.TemporaryDirectory(prefix="upstream-ab-slot-", dir=out.parent) as tmp:
            snapshot = Path(tmp)
            launches = {v: out / f"launch-{v}.fish" for v in VARIANTS}
            for v in VARIANTS:
                launches[v].write_text(launcher(source, binaries[v], budget, snapshot))

            def run(case, variant, folder, *, seed=False):
                with diag.server(ROOT, folder, launches[variant], candidate=True, env=env, note="Upstream merge A/B"):
                    runner.require_runtime_build(json.loads((folder / "props.json").read_text()), builds[variant])
                    if seed:
                        result = runner.send_request(encode(dict(requests["long"], max_tokens=1)), folder / "prefill",
                                                     args.request_timeout, save_response=False)
                        saved = diag.http(folder, "slot-save", "slots/0?action=save", SLOT)
                        if saved["n_saved"] < args.min_context:
                            raise ValueError(f"Nur {saved['n_saved']} Tokens gespeichert. --repeat-prefix erhöhen.")
                        return dict(prefill=result, snapshot=saved)

                    phases = ("cold", "warm") if case == "short" else ("restored",)
                    min_cached = 0
                    if case == "long":
                        saved = report["long_seed"]["snapshot"]["n_saved"]
                        restored = diag.http(folder, "slot-restore", "slots/0?action=restore", SLOT)
                        if restored["n_restored"] != saved:
                            raise ValueError(f"{variant}: restaurierter Slot passt nicht zum Seed-Slot")
                        min_cached = max(args.min_context, (95*saved + 99)//100)

                    results = {}
                    for phase in phases:
                        r = results[phase] = runner.send_request(payloads[case], folder / phase, args.request_timeout,
                                                                 min_cached_tokens=min_cached, save_response=False)
                        t = r["timings"]
                        runner.say(f"{folder.name}/{phase}: Prefill {runner.rate(r, 'prefill')} t/s, "
                                   f"Decode {runner.rate(r, 'decode')} t/s, "
                                   f"Cache {t.get('cache_n', 0)}, Draft {t.get('draft_n_accepted')}/{t.get('draft_n')}")
                    return results

            for case in cases:
                if case == "short":
                    runner.say("Dateicache mit einem Referenzlauf vorwärmen")
                    run(case, "baseline", out / "prime")
                else:
                    runner.say("Langen Kontext einmal auf der Baseline aufbauen und Slot 0 speichern")
                    report["long_seed"] = run(case, "baseline", out / "long-seed", seed=True)

                pairs = report["pairs"][case] = []
                for i in range(args.pairs):
                    order = VARIANTS if i % 2 == 0 else VARIANTS[::-1]
                    runner.say(f"{case} Paar {i+1}: {' -> '.join(order)}")
                    pair: dict = {"order": order}
                    pairs.append(pair)
                    for variant in order:
                        pair[variant] = run(case, variant, out / f"{case}-{i+1:02d}-{variant}")

                for phase in pairs[0]["baseline"]:
                    try:
                        metrics = report["summary"][phase] = summarize(pairs, phase)
                    except (KeyError, TypeError, ValueError) as error:
                        report["workload_differences"][phase] = str(error)
                        runner.say(f"{phase}: A/B nicht direkt vergleichbar ({error})")
                        continue
                    for kind in ("prefill", "decode"):
                        rate = metrics[f"{kind}_tps"]
                        print(f"{phase} {kind} Median {rate['baseline']:.2f} -> {rate['candidate']:.2f} t/s")
                    for key in ("prompt_ms", "predicted_ms", "request_wall_s"):
                        delta = metrics[key]["paired_delta_pct"]
                        print(f"  {key} {statistics.median(delta):+.2f}% [{min(delta):+.2f}%, {max(delta):+.2f}%]")

        report["complete"] = not report["workload_differences"]
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__} {error}"
        runner.say(report["error"])
    runner.json_write(out / "report.json", report)
    print(f"Handoff-Archiv {runner.archive_suite(out)}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
