#!/usr/bin/env python3
"""Capture one real Verify workload and replay the existing cache policy on the CPU."""
import argparse
import collections
import datetime
from importlib import import_module
import json
import math
import os
from pathlib import Path
import subprocess

from build_pair import build_id
from rocm_build import ROOT, SDK, build_targets, rocm_env

runner = import_module("stream-ab")
MIB = 1024*1024


def allocation(group, slots):
    align = group["alignment"]
    return sum(((size*(slots + 1) + align - 1)//align)*align for size in group["bank_bytes"])


def slots_at_budget(group, budget):
    # Keep the recorded prefill placement when testing alternative capacities.
    capacity = group["experts"]*len(group["layers"])
    lo, hi = group["min_slots"], capacity
    remote = group["backend"] != group["prefill_backend"]
    best = None
    while lo <= hi:
        slots = (lo + hi)//2
        keep_remote = not remote or group["prefill_link"]*(capacity - slots) > group["owner_link"]*capacity
        if keep_remote and allocation(group, slots) <= budget:
            best, lo = slots, slots + 1
        else:
            hi = slots - 1
    return best


def load_replay(trace, result, backend, shift_mib):
    """Keep startup/prefill history and score small target graphs after the last large prefill."""
    groups = {}
    steps = collections.defaultdict(list)
    with trace.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            if not line.startswith("MOE_MEASURE\t"):
                continue
            event = json.loads(line.split("\t", 1)[1])
            if event["kind"] not in ("policy_group", "hybrid_plan") or "policy_cache" not in event:
                continue
            key = (event["policy_cache"], event["cache_group"])
            if event["kind"] == "policy_group":
                if key in groups:
                    raise ValueError(f"Doppelte Cache-Gruppenbeschreibung {key}")
                groups[key] = event
            else:
                steps[key].append(event)

    def in_request(step):
        return result["monotonic_start_us"] <= step["us"] <= result["monotonic_end_us"]

    caches = {key[0] for key, group in groups.items() if group["backend"] == backend and
              any(step.get("device_lookup") and in_request(step) for step in steps[key])}
    if len(caches) != 1:
        raise ValueError(f"Erwartet genau einen aktiven Device-Lookup-Cache auf {backend}, gefunden {len(caches)}")
    cache, = caches
    selected = {key: group for key, group in groups.items() if key[0] == cache and group["backend"] == backend}
    last_prefill = max((step["graph"] for key in selected for step in steps[key]
                        if in_request(step) and not step["force_resident"]), default=0)
    graph_ids = set()
    rows = []
    for key, group in sorted(selected.items()):
        size = group["allocated_bytes"]
        original = group["slots"]
        if size != allocation(group, original) or slots_at_budget(group, size) != original:
            raise ValueError(f"Cache-Gruppenbudget passt nicht zu den gemessenen Slots {key}")
        if not steps[key]:
            raise ValueError(f"Cache-Gruppe ohne Routing-Aufzeichnung {key}")
        options = [original]
        for delta in (-shift_mib, shift_mib):
            alternative = slots_at_budget(group, size + delta*MIB)
            if alternative is not None and alternative != original:
                options.append(alternative)
        for step in steps[key]:
            step["scored"] = (in_request(step) and step["graph"] > last_prefill and
                              step.get("device_lookup", False) and 1 <= step["tokens"] <= 8)
            if step["scored"]:
                graph_ids.add(step["graph"])
        rows.append(dict(group, tries=sorted(set(options)), steps=steps[key]))
    if len(graph_ids) < 32:
        raise ValueError(f"Nur {len(graph_ids)} auswertbare Verify-Graphen. Längeren Request oder --max-tokens wählen.")
    return dict(backend=backend, graphs=sorted(graph_ids), groups=rows, shift_mib=shift_mib)


def recommend(replay_input, curves):
    groups = {g["cache_group"]: g for g in replay_input["groups"]}
    base = {key: g["slots"] for key, g in groups.items()}
    graphs = replay_input["graphs"]
    cut = len(graphs)//2
    by_group = {}
    verified = 0
    for item in curves:
        key = item["group"]
        group = groups[key]
        by_group[key] = {}
        for curve in item["curves"]:
            slots = curve["slots"]
            misses = dict(curve["misses_by_graph"])
            bundle = sum(group["bank_bytes"])
            by_group[key][slots] = [bundle*misses.get(graph, 0) for graph in graphs]
            if slots == base[key]:
                if curve["verified_steps"] != len(group["steps"]):
                    raise ValueError(f"Baseline-Replay unvollständig für Gruppe {key}")
                verified += curve["verified_steps"]
    if set(by_group) != set(groups) or any(set(by_group[key]) != set(g["tries"]) for key, g in groups.items()):
        raise ValueError("Slot-Kurven unvollständig")

    def totals(layout):
        return [sum(row) for row in zip(*(by_group[key][slots] for key, slots in layout.items()))]

    def metrics(values):
        return dict(train_bytes=sum(values[:cut]), holdout_bytes=sum(values[cut:]),
                    p95_holdout_bytes=sorted(values[cut:])[math.ceil(0.95*(len(values) - cut)) - 1])

    baseline = metrics(totals(base))
    budget = sum(g["allocated_bytes"] for g in groups.values())
    candidate = None
    best_cost = baseline["train_bytes"]
    for donor, donor_group in groups.items():
        lower = min(donor_group["tries"])
        if lower >= base[donor]:
            continue
        for receiver, receiver_group in groups.items():
            upper = max(receiver_group["tries"])
            if donor == receiver or upper <= base[receiver]:
                continue
            delta_bytes = (allocation(donor_group, lower) - donor_group["allocated_bytes"] +
                           allocation(receiver_group, upper) - receiver_group["allocated_bytes"])
            if delta_bytes > 0:
                continue
            proposed = dict(base)
            proposed.update({donor: lower, receiver: upper})
            result = metrics(totals(proposed))
            if result["train_bytes"] < best_cost:
                best_cost = result["train_bytes"]
                candidate = dict(result, allocated_bytes=budget + delta_bytes,
                                 slots=proposed, donor=donor, receiver=receiver)
    if candidate:
        candidate["holdout_delta_percent"] = (100*(candidate["holdout_bytes"]/baseline["holdout_bytes"] - 1)
                                              if baseline["holdout_bytes"] else None)
        candidate["holdout_improves"] = candidate["holdout_bytes"] < baseline["holdout_bytes"]
    return dict(verified_steps=verified, train_graphs=cut, holdout_graphs=len(graphs) - cut,
                allocated_bytes=budget, slots=base, baseline=baseline, candidate=candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=SDK)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-rocm-moe-stream-measure")
    parser.add_argument("--jobs", type=int, default=28)
    parser.add_argument("--backend", default="ROCm1")
    parser.add_argument("--request", type=Path, default=ROOT / "moe-first-request.json")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--shift-mib", type=int, default=256)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--shutdown-timeout", type=int, default=240)
    args = parser.parse_args()
    if args.shift_mib < 1 or args.max_tokens < 128:
        parser.error("--shift-mib muss positiv sein und --max-tokens mindestens 128")
    if not (ROOT / "moe-server-measure.fish").is_file() or runner.port_open():
        parser.error("moe-server-measure.fish und ein freier Port 8080 werden benötigt")
    build = args.build_dir.resolve()
    try:
        request = json.loads(runner.prepare_request(args.request, production_controls=True))
        request.pop("max_completion_tokens", None)
        request["max_tokens"] = args.max_tokens
        env = rocm_env(args.rocm)
        build_targets(ROOT, build, env, args.jobs, True, ("llama-server", "test-moe-cache-replay"), extra=(
            "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TESTS=ON",
        ))
        args.candidate_measure = build / "bin/llama-server"
        identity = build_id(args.candidate_measure, env)
        env["LD_LIBRARY_PATH"] = str(build / "bin") + os.pathsep + env["LD_LIBRARY_PATH"]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = ROOT / "results" / f"moe-cache-replay-{stamp}-{os.getpid()}"
    output.mkdir(parents=True)
    args.draft_max = 3
    args.production_controls = True
    report = dict(complete=False, build=identity, sdk=env["ROCM_PATH"], backend=args.backend)
    runner.json_write(output / "request.json", request)
    payload = (output / "request.json").read_bytes()
    try:
        launchers = runner.prepare_launchers(ROOT, output, args, variants=("candidate",), kinds=("measure",))
        phase = output / "measure"
        runner.say(f"Starte einen Request mit bis zu {args.max_tokens} Completion-Tokens")
        result = runner.run_phase(ROOT, phase, "02-measure", launchers["candidate"]["measure"]["script"],
                                  False, payload, env, args, candidate=True, expected_build=identity,
                                  request_phases=("cold",))
        report["capture"] = result
        trace = phase / "instrumented/trace.log"
        replay_input = load_replay(trace, result["requests"]["cold"], args.backend, args.shift_mib)
        input_file = output / "replay-input.json"
        input_file.write_text(json.dumps(replay_input, separators=(",", ":")) + "\n")
        command = [str(build / "bin/test-moe-cache-replay"), str(input_file)]
        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
        (output / "replay.log").write_text(proc.stderr)
        if proc.returncode:
            raise ValueError(f"Offline-Replay fehlgeschlagen (Status {proc.returncode}). Details in replay.log")
        curves = json.loads(proc.stdout)
        runner.json_write(output / "curves.json", curves)
        report["replay"] = recommend(replay_input, curves)
        replay = report["replay"]
        runner.say(f"{replay['verified_steps']} Baseline-Planungsaufrufe reproduziert, "
                   f"{replay['allocated_bytes']/MIB:.2f} MiB Bank-Budget auf {args.backend}")
        candidate = replay["candidate"]
        if candidate:
            base, slots = replay["slots"], candidate["slots"]
            for key in (candidate["donor"], candidate["receiver"]):
                runner.say(f"Gruppe {key} Slots {base[key]} -> {slots[key]}")
            delta = candidate["holdout_delta_percent"]
            runner.say(f"Holdout {replay['holdout_graphs']} Verify-Graphen, "
                       f"Miss-MiB {replay['baseline']['holdout_bytes']/MIB:.2f} -> {candidate['holdout_bytes']/MIB:.2f}"
                       + (f" ({delta:+.2f}%)" if delta is not None else ""))
        else:
            runner.say("Die getestete Umverteilung senkt die Miss-Bytes der ersten Hälfte nicht")
        report["complete"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__} {error}"
        print(report["error"])
    runner.json_write(output / "report.json", report)
    print(f"Handoff-Archiv {runner.archive_suite(output)}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
