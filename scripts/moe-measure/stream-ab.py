#!/usr/bin/env python3
"""MoE stream A/B suite: original 82478ef vs. bounded-stream candidate.

Run from ~/src/llama.cpp. Existing Fish templates, model settings, binaries and
repo's scripts/moe-measure/{run,analyze,profile}.py are used. Prompt cache OFF.
Each phase uses a fresh server process and executes cold + warm requests in it.
Monotonic MOE_MEASURE timestamps allow host trace splitting per request on Linux.
Profiler CSVs are summarized for the whole phase (not split per request).
Candidate runs must prove bounded streaming activation before timing requests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
PORT = 8080
BINARIES = {
    "baseline": ("build-rocm-moe-reference/bin/llama-server", "build-rocm-moe-measure/bin/llama-server"),
    "candidate": ("build-rocm-moe-stream-reference/bin/llama-server", "build-rocm-moe-stream-measure/bin/llama-server"),
}
PHASES = [("01-reference", "reference", False), ("02-measure", "measure", False), ("03-profile", "measure", True)]


def say(text: str) -> None:
    print(f"[{dt.datetime.now().astimezone().strftime('%H:%M:%S')}] {text}", flush=True)


def json_write(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for data in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def children_by_comm(pid: int, name: str) -> list[int]:
    tree: dict[int, list[int]] = {}
    names: dict[int, str] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            child_pid = int(path.name)
            status = (path / "status").read_text(errors="replace")
            match = re.search(r"(?m)^PPid:\s*(\d+)", status)
            if match:
                tree.setdefault(int(match[1]), []).append(child_pid)
                names[child_pid] = (path / "comm").read_text().strip()
        except (OSError, ValueError):
            continue
    result, pending, visited = [], [pid], set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        for child in tree.get(node, []):
            if names.get(child) == name:
                result.append(child)
            pending.append(child)
    return result


def port_open() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1):
            return True
    except OSError:
        return False


def wait_health(process: subprocess.Popen, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Serverstart abgebrochen, Exit={process.returncode}")
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as response:
                if response.status == 200:
                    if len(children_by_comm(process.pid, "llama-server")) != 1:
                        raise RuntimeError("Health-Response stammt nicht eindeutig vom gestarteten llama-server")
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError("Server wurde nicht bereit. Prüfe controller.log und trace.log")


def stop_server(process: subprocess.Popen, seconds: int) -> int:
    targets = children_by_comm(process.pid, "llama-server")
    if len(targets) != 1:
        raise RuntimeError(f"Genau einen eigenen llama-server erwartet: {targets}")
    os.kill(targets[0], signal.SIGTERM)
    process.wait(timeout=seconds)
    return process.returncode


def send_request(request: bytes, folder: Path, seconds: int) -> dict:
    folder.mkdir()
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    http_request = urllib.request.Request(url, data=request, headers={"Content-Type": "application/json"}, method="POST")
    t0_us = time.monotonic_ns() // 1000
    try:
        with urllib.request.urlopen(http_request, timeout=seconds) as response:
            payload = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        (folder / "response-error.txt").write_bytes(error.read())
        raise RuntimeError(f"HTTP {error.code}: siehe response-error.txt") from error
    t1_us = time.monotonic_ns() // 1000
    (folder / "response.json").write_bytes(payload)
    data = json.loads(payload)
    if status != 200 or data.get("error"):
        raise RuntimeError(f"Unerwartete Antwort: {status}, {str(data)[:500]}")
    usage = data.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    if cached != 0 or not usage.get("prompt_tokens"):
        raise RuntimeError(f"Prompt-Cache-Check fehlgeschlagen: cached={cached}, usage={usage}")
    choices = [{"message": c.get("message"), "text": c.get("text"), "finish_reason": c.get("finish_reason")} for c in data.get("choices", [])]
    choice_hash = hashlib.sha256(json.dumps(choices, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    result = {"http_status": status, "request_wall_s": round((t1_us-t0_us)/1e6, 4),
              "monotonic_start_us": t0_us, "monotonic_end_us": t1_us,
              "response_sha256": hashlib.sha256(payload).hexdigest(), "content_sha256": choice_hash,
              "usage": usage, "timings": data.get("timings", {}),
              "finish_reasons": [x.get("finish_reason") for x in choices]}
    json_write(folder / "request-result.json", result)
    return result


def split_host_trace(trace: Path, folder: Path, intervals: dict[str, tuple[int, int]]) -> dict:
    """ggml_time_us() and Python time.monotonic_ns() both use CLOCK_MONOTONIC on Linux."""
    paths = {key: folder / f"trace-{key}.log" for key in intervals}
    files = {k: p.open("w", encoding="utf-8") for k, p in paths.items()}
    counts = Counter()
    kinds = {k: Counter() for k in intervals}
    try:
        with trace.open("r", encoding="utf-8", errors="replace") as inp:
            for line in inp:
                if not line.startswith("MOE_MEASURE\t"):
                    continue
                try:
                    event = json.loads(line.split("\t", 1)[1])
                    us = int(event["us"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise RuntimeError("Unlesbarer MOE_MEASURE-Zeitstempel") from exc
                matches = [phase for phase, (a, b) in intervals.items() if a <= us <= b]
                if len(matches) == 1:
                    phase = matches[0]
                    files[phase].write(line)
                    counts[phase] += 1
                    kinds[phase][event.get("kind", "?")] += 1
                else:
                    counts["unassigned"] += 1
    finally:
        for out in files.values():
            out.close()
    if not counts.get("cold") or not counts.get("warm"):
        raise RuntimeError(f"Keine Events für Cold oder Warm: {dict(counts)}; nicht blind zusammenfassen")
    return {"records": dict(counts), "kinds": {k: dict(v) for k, v in kinds.items()},
            "limitation": "MOE_MEASURE Host-Events per monotonic request bounds; pending asynchronous events at boundaries may be unassigned."}


def invoke(root: Path, args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=root, check=True)


def host_summary(host: dict) -> dict:
    layers = host.get("layers", [])
    return {"records": host.get("records", 0), "layer_batches": len(layers),
            "incomplete_groups": len(host.get("incomplete_groups", [])),
            "h2d_gib": round(sum(x.get("h2d_bytes", 0) for x in layers)/2**30, 4),
            "d2d_gib": round(sum(x.get("d2d_bytes", 0) for x in layers)/2**30, 4),
            "routing_wait_ms": round(sum(x.get("routing_wait_us", 0) for x in layers)/1000, 3),
            "upload_enqueue_ms": round(sum(x.get("upload_enqueue_us", 0) for x in layers)/1000, 3),
            "note": "Expert-callback transfers only; not all host-to-device transfers."}


def rate(result: dict, kind: str) -> float | None:
    timings = result.get("timings") or {}
    for key in (("prompt_per_second", "prompt_ms", "prompt_n") if kind == "prefill" else ("predicted_per_second", "predicted_ms", "predicted_n"))[:1]:
        val = timings.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return round(float(val), 2)
    ms, n = (("prompt_ms", "prompt_n") if kind == "prefill" else ("predicted_ms", "predicted_n"))
    if isinstance(timings.get(ms), (int, float)) and timings[ms] > 0 and isinstance(timings.get(n), (int, float)):
        return round(1000 * timings[n] / timings[ms], 2)
    return None


def require_candidate_scratch(trace: Path) -> str:
    """Reject a silent whole-group fallback before spending time on requests."""
    text = trace.read_text(encoding="utf-8", errors="replace")
    caches = {m[0] for m in re.findall(r"\b(ROCm\d+)\s+MoE cache size\s*=\s*([0-9.]+)", text) if float(m[1]) > 0}
    scratch = set(re.findall(r"\b(ROCm\d+)\s+bounded miss scratch\s*=", text))
    execution: dict[str, set[str]] = {}
    for owner, device in re.findall(r"\b(ROCm\d+)\s+prefill execution\s*=\s*(ROCm\d+)", text):
        execution.setdefault(owner, set()).add(device)
    if len(caches) > 1:
        links = {device for device, speed, width in
                 re.findall(r"\b(ROCm\d+)\s+host link\s*=\s*([0-9.]+) GT/s x(\d+)", text)
                 if float(speed) > 0 and int(width) > 0}
        if caches - links or caches - execution.keys():
            raise RuntimeError(f"Candidate-Topologie unvollständig: Cache-GPUs={sorted(caches)}, "
                               f"PCIe-Pfade={sorted(links)}, Platzierung={sorted(execution)}. "
                               "Host-Link-Zeilen in trace.log prüfen; kein stiller Multi-GPU-Fallback im A/B-Test.")
    required = set().union(*(execution.get(owner, {owner}) for owner in caches))
    if not caches or required - scratch:
        raise RuntimeError(f"Candidate-Streaming inaktiv: Cache-GPUs={sorted(caches)}, Staging-GPUs={sorted(scratch)}. "
                           f"Geplante Prefill-GPUs={sorted(required)}. "
                           "Keine Performance-Auswertung eines stillen Fallbacks; Candidate-Build/Dispatch prüfen.")
    return ", ".join(f"{owner} -> {'/'.join(sorted(execution.get(owner, {owner})))}" for owner in sorted(caches))


def require_candidate_acceptance(result: dict) -> None:
    timing = result.get("timings") or {}
    drafted, accepted = timing.get("draft_n"), timing.get("draft_n_accepted")
    if isinstance(drafted, (int, float)) and drafted >= 32 and accepted == 0:
        raise RuntimeError(f"Candidate: {drafted} Draft-Tokens, kein einziger akzeptiert. "
                           "DFlash-Korrektheit prüfen; diese Suite wertet das nicht als Performance-Erfolg.")


def require_candidate_stream(split: dict) -> None:
    for phase in ("cold", "warm"):
        kinds = split["kinds"].get(phase, {})
        if not kinds.get("hybrid_plan") or kinds.get("work_copy", 0):
            raise RuntimeError(f"Candidate/{phase}: Streaming nicht vollständig aktiv: {kinds}. "
                               "Experten-Arbeitskopien sind weiterhin im Pfad; diesen Lauf nicht als Streaming-Benchmark werten.")
    if not split["kinds"]["cold"].get("stage_h2d"):
        raise RuntimeError("Candidate/cold: keine stage_h2d-Transfers; Miss-Staging wurde nicht nachgewiesen.")


def run_phase(root: Path, out: Path, name: str, launcher_script: Path, profiler: bool,
              request: bytes, env: dict, cli: argparse.Namespace, candidate: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    raw = out / "gpu-raw"
    executable = ["fish", str(launcher_script)]
    if profiler:
        executable = ["rocprofv3", "--runtime-trace", "--output-format", "csv", "--output-directory", str(raw), "--", *executable]
    cmd = [sys.executable, "scripts/moe-measure/run.py", "--output", str(out / "instrumented"),
           "--phase", "mixed-dflash", "--note", "Cold/Warm; SSD/RAM prompt cache disabled; fixed seed", "--", *executable]
    json_write(out / "launch.json", cmd)
    if port_open():
        raise RuntimeError(f"Port {PORT} belegt. Fremden Server vor Suite beenden.")
    results: dict[str, dict] = {}
    with (out / "controller.log").open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        stopped = False
        try:
            wait_health(proc, cli.startup_timeout)
            placement = ""
            if candidate:
                placement = require_candidate_scratch(out / "instrumented" / "trace.log")
            say(f"{name}: server bereit" + (f"; Candidate-Prefill: {placement}" if candidate else ""))
            for phase in ("cold", "warm"):
                result = send_request(request, out / phase, cli.request_timeout)
                results[phase] = result
                if candidate:
                    require_candidate_acceptance(result)
                say(f"{name}/{phase}: {result['request_wall_s']} s; Prompt {result['usage'].get('prompt_tokens')}; "
                    f"Prefill {rate(result, 'prefill')} t/s; Decode {rate(result, 'decode')} t/s")
            stop_server(proc, cli.shutdown_timeout)
            stopped = True
            meta_path = out / "instrumented" / "metadata.json"
            if not meta_path.is_file():
                raise RuntimeError("run.py hat metadata.json nicht geschrieben")
            meta = json.loads(meta_path.read_text())
            rc = meta.get("exit_code")
            # Profilers may propagate SIGTERM from their controlled child; only accept after BOTH HTTP 200s.
            if rc != 0 or proc.returncode != 0:
                if not (profiler and rc == 143 and proc.returncode in (0, 143)):
                    raise RuntimeError(f"Unerwarteter Stop: run.py={rc}, launcher={proc.returncode}")
            if name == "01-reference":
                return {"requests": results}
            trace = out / "instrumented" / "trace.log"
            intervals = {key: (value["monotonic_start_us"], value["monotonic_end_us"]) for key, value in results.items()}
            split = split_host_trace(trace, out, intervals)
            analysis = {}
            for phase in ("cold", "warm"):
                target = out / f"host-{phase}.json"
                invoke(root, ["scripts/moe-measure/analyze.py", str(out / f"trace-{phase}.log"),
                              "--phase", "mixed-dflash", "--output", str(target)])
                report = json.loads(target.read_text())
                summary = host_summary(report)
                if summary["incomplete_groups"]:
                    raise RuntimeError(f"Unvollständige MoE-Gruppen für {phase}: {summary['incomplete_groups']}")
                analysis[phase] = summary
            json_write(out / "trace-split.json", split)
            if candidate:
                require_candidate_stream(split)
            if profiler:
                if not raw.is_dir() or not any(raw.rglob("*.csv")):
                    raise RuntimeError("Keine ROCprofiler-CSV-Dateien; Exit 143 allein reicht nicht")
                invoke(root, ["scripts/moe-measure/profile.py", str(raw), "--output", str(out / "gpu.json")])
                gpu = json.loads((out / "gpu.json").read_text())
                if not gpu.get("kernels"):
                    raise RuntimeError("Profiler enthielt keine GPU-Kernel")
                return {"requests": results, "host": analysis, "trace_split": split,
                        "gpu_profile_whole_session": {"kernel_groups": len(gpu["kernels"]),
                                                      "warnings": gpu.get("coverage_warnings", [])}}
            return {"requests": results, "host": analysis, "trace_split": split}
        finally:
            if not stopped and proc.poll() is None:
                try:
                    targets = children_by_comm(proc.pid, "llama-server")
                    if len(targets) == 1:
                        os.kill(targets[0], signal.SIGTERM)
                        proc.wait(timeout=cli.shutdown_timeout)
                    else:
                        say(f"ACHTUNG: Server-Prozesse nicht eindeutig: {targets}; kein Blind-Kill")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    say(f"ACHTUNG: Shutdown unvollständig: {exc}")


def print_comparison(variants: dict) -> None:
    print("\nA/B-VERGLEICH (uninstrumentierte 01-reference; Host-Bytes aus 02-measure)\n")
    print(f"{'Variante':12} {'Cache':6} {'Wall(s)':>9} {'Prefill t/s':>12} {'Decode t/s':>11} {'H2D GiB':>9} {'D2D GiB':>9} {'Prompt':>8} {'Comp':>6} {'Hash':>8}")
    for variant in ("baseline", "candidate"):
        if "01-reference" not in variants.get(variant, {}):
            continue
        r = variants[variant]
        for phase in ("cold", "warm"):
            p = r["01-reference"]["requests"][phase]
            q = r.get("02-measure", {}).get("host", {}).get(phase, {})
            val = lambda x: "n/a" if x is None else str(x)
            print(f"{variant:12} {phase:6} {p['request_wall_s']:9.3f} {val(rate(p, 'prefill')):>12} "
                  f"{val(rate(p, 'decode')):>11} {val(q.get('h2d_gib')):>9} {val(q.get('d2d_gib')):>9} "
                  f"{p['usage'].get('prompt_tokens', 0):8} {p['usage'].get('completion_tokens',0):6} {p['content_sha256'][:8]:>8}")
    if all("01-reference" in variants.get(variant, {}) for variant in ("baseline", "candidate")):
        print("\nDelta Candidate vs. Baseline (uninstrumentiert):")
        for phase in ("cold", "warm"):
            base = variants["baseline"]["01-reference"]["requests"][phase]
            test = variants["candidate"]["01-reference"]["requests"][phase]
            a, b = rate(base, "prefill"), rate(test, "prefill")
            if a and b:
                print(f"  {phase}: Prefill {b/a*100-100:+.2f}% ({a} -> {b} t/s)")
            wa, wb = base["request_wall_s"], test["request_wall_s"]
            if wa:
                print(f"  {phase}: Request-Wall {wb/wa*100-100:+.2f}% ({wa:.3f} -> {wb:.3f} s; kleiner ist besser)")
            if base["content_sha256"] != test["content_sha256"]:
                print(f"  WARNUNG: {phase} unterschiedliche Completion-Hashes; Generierungsarbeit/DFlash prüfen")
            for variant in ("baseline", "candidate"):
                p = variants[variant]["01-reference"]["requests"][phase]
                timing = p.get("timings") or {}
                d, acc = timing.get("draft_n"), timing.get("draft_n_accepted")
                if isinstance(d, (float, int)) and d > 0 and isinstance(acc, (float, int)):
                    print(f"  {variant}/{phase}: DFlash draft_n={d}, accepted={acc} ({100*acc/d:.1f}%)")
    print("\nHinweis: N/A bedeutet, dass die Serverantwort keine dedizierte Timing-Metrik lieferte.")
    print("Bytezahlen sind HOST-instrumentiert, keine PCIe-Laufzeiten. Gleiche Hashes/Completion/DFlash prüfen.")


def prepare_launchers(root: Path, suite: Path, arguments: argparse.Namespace) -> dict:
    generated = {}
    pattern = r"(?m)^\s*\./build-rocm-moe-(?:reference|measure)/bin/llama-server(?=\s|$)"
    for variant in ("baseline", "candidate"):
        generated[variant] = {}
        for key in ("reference", "measure"):
            source = root / f"moe-server-{key}.fish"
            text = source.read_text()
            if re.search(r"--cache-ssd(?:-\S+)?", text):
                raise ValueError(f"{source}: SSD-Promptcache muss deaktiviert sein")
            if "--cache-ram 0" not in text:
                raise ValueError(f"{source}: '--cache-ram 0' fehlt")
            port = re.search(r"--port\s+(\d+)", text)
            if port and int(port[1]) != PORT:
                raise ValueError(f"{source} hat Port {port[1]} != {PORT}")
            supplied = getattr(arguments, f"{variant}_{key}")
            binary = (supplied if supplied.is_absolute() else (root / supplied)).resolve()
            if not binary.is_file():
                raise ValueError(f"Binary fehlt: {binary}. Zuerst mit cmake bauen.")
            if len(re.findall(pattern, text)) != 1:
                raise ValueError(f"{source}: alter llama-server-Pfad nicht eindeutig gefunden")
            result = re.sub(pattern, str(binary), text)
            target = suite / f"launch-{variant}-{key}.fish"
            target.write_text(result)
            generated[variant][key] = {"script": target, "binary": str(binary), "binary_sha256": sha(binary)}
    return generated


def archive_suite(suite: Path) -> Path:
    target = suite.with_name(suite.name + "-handoff.tar.gz")
    with tarfile.open(target, "w:gz", compresslevel=1) as archive:
        archive.add(suite, arcname=suite.name)
    return target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.home() / "src" / "llama.cpp")
    p.add_argument("--request", type=Path, default=Path("moe-first-request.json"))
    p.add_argument("--output", type=Path)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--profile", action="store_true", help="Auch ROCprofv3 auf beiden Varianten (Gesamt-Session) ausführen")
    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--request-timeout", type=int, default=900)
    p.add_argument("--shutdown-timeout", type=int, default=240)
    for key, paths in BINARIES.items():
        p.add_argument(f"--{key}-reference", type=Path, default=Path(paths[0]))
        p.add_argument(f"--{key}-measure", type=Path, default=Path(paths[1]))
    a = p.parse_args()
    global PORT
    PORT = a.port
    root = a.repo.expanduser().resolve()
    if not root.is_dir():
        p.error(f"Repository nicht gefunden: {root}")
    needed = ["scripts/moe-measure/run.py", "scripts/moe-measure/analyze.py", "scripts/moe-measure/profile.py",
              "moe-server-reference.fish", "moe-server-measure.fish"]
    for path in needed:
        if not (root / path).is_file():
            p.error(f"Fehlende Datei: {root / path}")
    request_path = (a.request if a.request.is_absolute() else root / a.request).resolve()
    if not request_path.is_file():
        p.error(f"Testrequest fehlt: {request_path}")
    payload = request_path.read_bytes()
    json_request = json.loads(payload)
    if json_request.get("cache_prompt") is not False or json_request.get("stream") is not False or json_request.get("seed") is None:
        p.error("Request muss cache_prompt=false, stream=false, festen seed enthalten")
    if not shutil.which("fish"):
        p.error("Fish nicht gefunden")
    if PORT <= 0 or PORT > 65535:
        p.error("Port ungültig")
    if port_open():
        p.error(f"Port {PORT} belegt; laufenden llama-server zuerst beenden")
    env = os.environ.copy()
    rocm = Path(env.get("ROCM_PATH", str(Path.home() / "opt/rocm-10.1"))).expanduser()
    if rocm.is_dir():
        env.setdefault("ROCM_PATH", str(rocm))
        env.setdefault("HIP_PATH", str(rocm))
        env["PATH"] = str(rocm / "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(rocm / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    if a.profile and not shutil.which("rocprofv3", path=env.get("PATH")):
        p.error("ROCprofv3 nicht gefunden; ohne --profile starten")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    suite = (a.output or Path("results") / f"moe-stream-ab-{stamp}")
    suite = (suite if suite.is_absolute() else root / suite).resolve()
    suite.mkdir(parents=True, exist_ok=False)
    (suite / "request.json").write_bytes(payload)
    for name in ("reference", "measure"):
        shutil.copy2(root / f"moe-server-{name}.fish", suite / f"template-{name}.fish")
    try:
        launchers = prepare_launchers(root, suite, a)
    except ValueError as exc:
        p.error(str(exc))
    try:
        repo_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        repo_head = "unknown"
    json_write(suite / "suite.json", {
        "repo_HEAD_during_run": repo_head,
        "baseline_source_expected_not_verified": "82478efb935b4e6a99946132553435d6f55fe8cb",
        "candidate_source_expected_not_verified": repo_head,
        "binary_provenance_note": "Binary SHA256s recorded. Build-source attribution is NOT verified by this runner; confirm existing baseline builds are from 82478ef.",
        "binaries": {variant: {key: {k: v for k, v in entry.items() if k != "script"} for key, entry in xs.items()} for variant, xs in launchers.items()},
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "cache_prompt": False, "profile_enabled": a.profile, "candidate_activation_required": True, "started": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configuration": "Existing fish templates, SSD/RAM prompt cache off, same settings across variants."
    })
    report, errors = {}, []
    phases = PHASES if a.profile else PHASES[:2]
    try:
        for variant in ("baseline", "candidate"):
            report[variant] = {}
            for name, kind, profiler in phases:
                say(f"=== {variant} / {name} ===")
                target = suite / variant / name
                target.parent.mkdir(exist_ok=True)
                try:
                    result = run_phase(root, target, name, launchers[variant][kind]["script"], profiler, payload, env, a, candidate=variant == "candidate")
                    report[variant][name] = result
                    json_write(suite / "comparison.json", report)
                    say(f"{variant}/{name}: OK")
                except Exception as exc:
                    errors.append(f"{variant}/{name}: {type(exc).__name__}: {exc}")
                    say(f"FEHLER: {errors[-1]}")
                    raise
    except KeyboardInterrupt:
        errors.append("Controller per Tastatur unterbrochen")
    except Exception:
        pass
    finally:
        json_write(suite / "suite-result.json", {"complete": not errors, "failures": errors})
        json_write(suite / "comparison.json", report)
        print_comparison(report)
        if not a.no_archive:
            try:
                target = archive_suite(suite)
                say(f"Handoff-Archiv: {target}")
            except Exception as exc:
                errors.append(f"Archivierung fehlgeschlagen: {exc}")
                say(errors[-1])
        say("FERTIG" if not errors else "UNVOLLSTÄNDIG")
        say(f"Ergebnisse: {suite}")
        for error in errors:
            say(error)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
