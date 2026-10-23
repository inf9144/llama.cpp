#!/usr/bin/env python3
"""Run the same server workload against baseline, disabled, set-only, cpy-only and enabled ReBAR paths."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, payload=None, timeout=3600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin", type=Path, required=True)
    parser.add_argument("--experiment-bin", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="baseline,off,set,cpy,on")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--predict", type=int, default=512)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--stats", action="store_true", help="Enable aggregate transfer instrumentation; run separately from throughput measurements")
    parser.add_argument("server_args", nargs=argparse.REMAINDER, help="Identical llama-server flags after --; do not include --host or --port")
    args = parser.parse_args()
    cases = args.cases.split(",")
    if not cases or any(case not in {"baseline", "off", "set", "cpy", "on"} for case in cases):
        parser.error("--cases must contain baseline, off, set, cpy and/or on")
    if args.repetitions < 1 or args.warmup < 0 or args.predict < 1:
        parser.error("repetitions and predict must be positive; warmup must be nonnegative")
    server_args = args.server_args[1:] if args.server_args[:1] == ["--"] else args.server_args
    if any(arg.split("=", 1)[0] in {"--host", "--port"} for arg in server_args):
        parser.error("use --port before --; the server is bound to 127.0.0.1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prompts = [(path.name, path.read_text(encoding="utf-8")) for path in args.prompt_file]
    metadata = {
        "platform": platform.platform(),
        "baseline_bin": str(args.baseline_bin.resolve()),
        "experiment_bin": str(args.experiment_bin.resolve()),
        "server_args": server_args,
        "cases": cases,
        "stats": args.stats,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "predict": args.predict,
        "prompts": [{"name": name, "sha256": hashlib.sha256(prompt.encode()).hexdigest()} for name, prompt in prompts],
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("GGML_VK_", "RADV_", "VK_"))},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    url = f"http://127.0.0.1:{args.port}"
    all_rows: list[dict[str, Any]] = []
    for case in cases:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        env = os.environ.copy()
        env.update({
            "GGML_VK_IDLE_REBAR_FASTPATH": "0" if case in {"baseline", "off"} else "1",
            "GGML_VK_IDLE_REBAR_SET": "1" if case in {"set", "on"} else "0",
            "GGML_VK_IDLE_REBAR_CPY": "1" if case in {"cpy", "on"} else "0",
            "GGML_VK_IDLE_REBAR_STATS": "1" if args.stats and case != "baseline" else "0",
        })
        binary = args.baseline_bin if case == "baseline" else args.experiment_bin
        command = [str(binary.resolve()), *server_args, "--host", "127.0.0.1", "--port", str(args.port)]
        print(f"Starting {case}; log: {args.output_dir / (case + '.log')}", flush=True)
        with (args.output_dir / f"{case}.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{case}: server exited with {process.returncode}")
                    try:
                        request(url + "/health", timeout=2)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"{case}: server did not become ready")
                        time.sleep(0.25)
                for name, prompt in prompts:
                    for repeat in range(-args.warmup, args.repetitions):
                        start = time.monotonic()
                        result = request(url + "/completion", {
                            "prompt": prompt, "n_predict": args.predict, "temperature": 0,
                            "seed": 1234, "ignore_eos": True, "cache_prompt": False,
                            "stream": False, "id_slot": 0,
                        }, timeout=args.request_timeout)
                        if result.get("truncated") or result["timings"]["predicted_n"] != args.predict:
                            raise RuntimeError(f"{case}: truncated prompt or incomplete generation; results are not comparable")
                        row = {
                            "case": case, "prompt": name, "repeat": repeat, "warmup": repeat < 0,
                            "wall_s": time.monotonic() - start, "timings": result["timings"],
                            "content_sha256": hashlib.sha256(result.get("content", "").encode()).hexdigest(),
                        }
                        with (args.output_dir / "results.jsonl").open("a") as output:
                            output.write(json.dumps(row) + "\n")
                        if repeat >= 0:
                            all_rows.append(row)
                            print(f"{case} {name} #{repeat}: pp={row['timings']['prompt_per_second']:.2f} "
                                  f"tg={row['timings']['predicted_per_second']:.2f} tok/s", flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    summary = []
    for case in cases:
        for name, _ in prompts:
            rows = [row for row in all_rows if row["case"] == case and row["prompt"] == name]
            summary.append({
                "case": case, "prompt": name,
                "prefill_median_tps": statistics.median(row["timings"]["prompt_per_second"] for row in rows),
                "decode_median_tps": statistics.median(row["timings"]["predicted_per_second"] for row in rows),
                "prompt_tokens": sorted({row["timings"]["prompt_n"] for row in rows}),
                "decode_tokens": sorted({row["timings"]["predicted_n"] for row in rows}),
                "content_hashes": sorted({row["content_sha256"] for row in rows}),
            })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
