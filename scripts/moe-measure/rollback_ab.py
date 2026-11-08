#!/usr/bin/env python3
"""Compare tail rollback and decode with the same restored long-context slot."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tempfile

import completion_diagnose as diag
from gdn_ab import build_pair, workload
from rocm_build import ROOT, SDK, build_lock, rocm_env

runner = diag.runner
REVISIONS = {
    "baseline": "45d2115f9aa83592be0e842f80579df317d095e0",
    "candidate": "dfe904ca3e5fa62864984cbd1d1969363d30a2af",
}
SLOT = {"filename": "prefix.bin"}


def prepare_request(path, repeat, max_tokens):
    request = json.loads(path.read_text())
    seed = request.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32 - 1:
        raise ValueError("Request braucht einen festen Seed")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages or request.get("n", 1) != 1:
        raise ValueError("Ein Chat-Request mit messages und genau einer Antwort wird benötigt")
    if repeat > 1:
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") != "user" or not isinstance(last.get("content"), str):
            raise ValueError("Für --repeat-prefix > 1 muss die letzte User-Nachricht Text sein")
        text = last["content"]
        sections = [f"Reference excerpt {i}:\n{text}" for i in range(1, repeat)]
        last["content"] = "\n\n".join([*sections, "Current task:\n" + text])
    request.update(runner.MANUFACTURER_SAMPLING)
    request.pop("max_completion_tokens", None)
    request.pop("n_predict", None)
    request.update(seed=seed, cache_prompt=True, stream=False, max_tokens=max_tokens)
    return request


def launcher(source, binary, budget, snapshot):
    text = diag.launch_text(source, binary, production_controls=True)
    if re.search(r"(?<!\S)--slot-save-path(?=[=\s]|$)", text):
        raise ValueError("Benchmark-Vorlage muss einen freien Slot-Speicherpfad haben")
    text, n = re.subn(r"(--moe-cache-mib\s+)\d+,\d+(?=\s|$)",
                      lambda m: m[1] + ",".join(map(str, budget)), text)
    if n != 1:
        raise ValueError("Erwartet ein MiB-Budget für genau zwei GPUs in moe-server-reference.fish")
    invocation = shlex.quote(str(binary))
    flags = f"--slot-save-path {shlex.quote(str(snapshot))} --no-context-shift"
    return text.replace(invocation + " ", invocation + " " + flags + " ", 1)


def encode(request):
    return json.dumps(request, ensure_ascii=False).encode()


def verify_work(result):
    t = result["timings"]
    rounds = t["draft_verify_rounds"]
    if rounds < 8 or t["draft_tail_trim_us"] < 0 or t["predicted_ms"] <= 0:
        raise ValueError("Zu wenig Verify-Runden oder ungültige Messdaten")
    return (*workload(result), t["cache_n"], t["prompt_n"], rounds)


def summarize(pairs):
    rows = [[pair[variant] for variant in REVISIONS] for pair in pairs]
    if len({verify_work(r) for row in rows for r in row}) != 1:
        raise ValueError("Prompt-Reuse, Ausgabe oder DFlash-Zähler unterscheiden sich")
    result = {}
    for i, variant in enumerate(REVISIONS):
        runs = [row[i] for row in rows]
        result[variant] = dict(
            decode_tps=statistics.median(1000*r["usage"]["completion_tokens"]/r["timings"]["predicted_ms"] for r in runs),
            trim_us_per_verify=statistics.median(r["timings"]["draft_tail_trim_us"]/r["timings"]["draft_verify_rounds"] for r in runs),
        )
    result["paired_decode_ms_delta_percent"] = [
        100*(b["timings"]["predicted_ms"]/a["timings"]["predicted_ms"] - 1) for a, b in rows
    ]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rocm", type=Path, default=SDK)
    p.add_argument("--build-dir", type=Path, default=ROOT / "build-rocm-rollback-ab")
    p.add_argument("--jobs", type=int, default=28)
    p.add_argument("--request", type=Path, default=ROOT / "moe-first-request.json")
    p.add_argument("--repeat-prefix", type=int, default=36, help="1 für einen bereits langen Chat-Request")
    p.add_argument("--min-context", type=int, default=98304, help="Mindestzahl wiederverwendeter Prompt-Tokens")
    p.add_argument("--completion", type=int, default=256)
    p.add_argument("--cache-mib", default="15600,11300", help="Normales ROCm0,ROCm1-Budget; zweiter Lauf halbiert es")
    p.add_argument("--pairs", type=int, default=1, help="A/B-Paare je Cachegröße")
    p.add_argument("--request-timeout", type=int, default=1800)
    args = p.parse_args()
    if (args.repeat_prefix < 1 or args.min_context < 1 or args.pairs < 1 or args.completion < 32 or args.request_timeout < 1
            or not re.fullmatch(r"[1-9]\d*,[1-9]\d*", args.cache_mib)):
        p.error("Positive Anzahl/Budgets sowie mindestens 32 Completion-Tokens erforderlich")
    normal = tuple(map(int, args.cache_mib.split(",")))
    half = tuple(mib//2 for mib in normal)
    if min(half) < 1 or runner.port_open():
        p.error("Positives halbes Budget und freier Port 8080 erforderlich")

    try:
        request = prepare_request(args.request, args.repeat_prefix, args.completion)
        source = (ROOT / "moe-server-reference.fish").read_text()
        env = rocm_env(args.rocm)
        env["GGML_CUDA_DISABLE_FUSION"] = "0"
        directory = args.build_dir.resolve()
        with build_lock(directory):
            binaries, builds = build_pair(directory, env, args.jobs, revisions=REVISIONS,
                                          extra=("-DCMAKE_CXX_FLAGS=-DLLAMA_ROLLBACK_BENCH",))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        p.exit(1, f"{error}\n")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = ROOT / "results" / f"rollback-ab-{stamp}"
    out.mkdir(parents=True)
    payload = encode(request)
    budgets = {"normal": normal, "half": half}
    report: dict = dict(complete=False, revisions=REVISIONS, builds=builds, sdk=env["ROCM_PATH"],
                        request_sha256=hashlib.sha256(payload).hexdigest(), budgets=budgets,
                        pairs={}, summary={})
    try:
        with tempfile.TemporaryDirectory(prefix="rollback-slot-", dir=out.parent) as directory:
            snapshot = Path(directory)

            def run(variant, budget, folder, *, seed=False):
                launch = out / ("launch-" + folder.name + ".fish")
                launch.write_text(launcher(source, binaries[variant], budget, snapshot))
                with diag.server(ROOT, folder, launch, candidate=False, env=env, note="Long-context rollback A/B"):
                    props = json.loads((folder / "props.json").read_text())
                    runner.require_runtime_build(props, builds[variant])
                    if seed:
                        runner.say("Baue den langen Prompt einmal auf und speichere Slot 0")
                        result = runner.send_request(encode(dict(request, max_tokens=1)), folder / "prefill",
                                                     args.request_timeout, save_response=False)
                        saved = diag.http(folder, "slot-save", "slots/0?action=save", SLOT)
                        if saved["n_saved"] <= args.min_context:
                            raise ValueError(f"Nur {saved['n_saved']} Tokens im Slot. --repeat-prefix erhöhen.")
                        return dict(prefill=result, snapshot=saved)
                    restored = diag.http(folder, "slot-restore", "slots/0?action=restore", SLOT)
                    n_saved = report["seed"]["snapshot"]["n_saved"]
                    if restored["n_restored"] != n_saved:
                        raise ValueError("Restaurierter Slot passt nicht zum Seed-Slot")
                    result = runner.send_request(payload, folder / "decode", args.request_timeout, save_response=False,
                                                 min_cached_tokens=max(args.min_context, (95*n_saved + 99)//100))
                    verify_work(result)
                    runner.say(f"{folder.name}: {result['timings']['cache_n']} Prompt-Tokens wiederverwendet, "
                               f"{runner.rate(result, 'decode')} t/s")
                    return result

            report["seed"] = run("baseline", normal, out / "seed", seed=True)
            for i, (name, budget) in enumerate(budgets.items()):
                pairs = report["pairs"][name] = []
                for j in range(args.pairs):
                    order = list(REVISIONS)
                    if (i + j) % 2:
                        order.reverse()
                    runner.say(f"{name} {budget[0]},{budget[1]} MiB, Paar {j + 1}: {' -> '.join(order)}")
                    pair = {"order": order}
                    pairs.append(pair)
                    for variant in order:
                        pair[variant] = run(variant, budget, out / f"{name}-{j + 1:02d}-{variant}")
                s = report["summary"][name] = summarize(pairs)
                a, b = s["baseline"], s["candidate"]
                print(f"{name}: Decode {a['decode_tps']:.2f} -> {b['decode_tps']:.2f} t/s, "
                      f"Trim {a['trim_us_per_verify']:.1f} -> {b['trim_us_per_verify']:.1f} µs/Verify")
        report["complete"] = True
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__} {error}"
        runner.say(report["error"])
    runner.json_write(out / "report.json", report)
    print(f"Handoff-Archiv: {runner.archive_suite(out)}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
