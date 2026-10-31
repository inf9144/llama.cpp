#!/usr/bin/env python3
"""Trace real QSA top-k scores and replay frozen inputs without changing kernels."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

import completion_diagnose as diagnose


def summarize(events: list[dict], done: dict) -> dict:
    first = [e for e in events if e['pass'] == 0]
    second = [e for e in events if e['pass'] == 1]
    if not done.get('complete') or not first or len(first) != len(second) or len(first) != done.get('calls_per_pass'):
        raise ValueError('Missing or inconsistent trace coverage')
    devices = sorted({e['device'] for e in events})
    if not any(device.startswith('ROCm') for device in devices):
        raise ValueError('No ROCm TOP_K coverage; inspect backend placement')
    frozen = [e['frozen_replay'] for e in events if 'frozen_replay' in e]
    if any(not probe.get('input_unchanged') or len(probe['repetitions']) != 8 for probe in frozen):
        raise ValueError('Incomplete frozen-input replay')
    return dict(complete=True, prompt_tokens=done['prompt_tokens'], calls_per_pass=len(first), topk_devices=devices,
                finite_cutoff_ambiguous_rows=sum(e['finite_cutoff_ambiguous_rows'] for e in events),
                changed_score_rows=sum(e['changed_score_rows'] for e in second),
                changed_visible_set_rows=sum(e['changed_visible_set_rows'] for e in second),
                matching_score_hash_changed_visible_set_rows=sum(e['matching_score_hash_changed_visible_set_rows'] for e in second),
                frozen_replay_devices=[probe['device'] for probe in frozen],
                frozen_max_changed_rows=max((r['changed_visible_sets_from_previous'] for probe in frozen for r in probe['repetitions'][1:]), default=0),
                frozen_max_changed_from_original=max((r['changed_visible_sets_from_original'] for probe in frozen for r in probe['repetitions']), default=0),
                note='Instrumented prefill only. Finite cutoff ties and changing sets on frozen scores can identify selection ambiguity. No timing, decode or verify conclusions.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.home() / 'src/llama.cpp')
    parser.add_argument('--source-run', type=Path, required=True, help='Directory of the previous completion diagnostic')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    source = (args.source_run if args.source_run.is_absolute() else root / args.source_run).resolve()
    original = json.loads((source / 'report.json').read_text())
    tokens = json.loads((source / 'prompt-tokens.json').read_text())
    if not original.get('complete') or original.get('draft_max') != 0 or not isinstance(tokens, list) or not 1 <= len(tokens) <= 8192 or any(type(t) is not int or t < 0 for t in tokens):
        parser.error('Expected a complete zero-draft diagnostic with 1..8192 prompt token IDs')
    if diagnose.digest(tokens) != original.get('prompt_sha256'):
        parser.error('Source prompt token digest does not match the previous report')
    binary = root / 'build-rocm-moe-stream-reference/bin/test-qsa-topk-trace'
    if not binary.is_file(): parser.error('Build the test-qsa-topk-trace target in build-rocm-moe-stream-reference first')
    if diagnose.runner.port_open(): parser.error('Port 8080 is occupied; stop the running server before loading the diagnostic model')
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out = args.output or Path('results') / f'qsa-topk-diagnose-{stamp}'
    out = (out if out.is_absolute() else root / out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runner = diagnose.runner
    runner.json_write(out / 'prompt-tokens.json', tokens)
    launcher = out / 'trace.fish'
    launcher.write_text(diagnose.launch_text((root / 'moe-server-reference.fish').read_text(), binary, 0).rstrip()
                        + ' \\\n  --tokens ' + shlex.quote(str(out / 'prompt-tokens.json')) + '\n')
    libraries = {p.resolve() for pattern in ('libllama*.so*', 'libggml*.so*') for p in binary.parent.glob(pattern)}
    report: dict = dict(complete=False, binary=str(binary), binary_sha256=runner.sha(binary),
                        library_sha256={str(p): runner.sha(p) for p in sorted(libraries)},
                        source_run=str(source), prompt_sha256=diagnose.digest(tokens))
    process = None
    failure = None
    try:
        runner.say('QSA TopK trace: two prefills, at most eight frozen replays per device; no generation or profiler')
        with (out / 'trace.log').open('wb') as log:
            process = subprocess.Popen(['fish', str(launcher)], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status = process.wait(timeout=900)
        events, done, native_error = [], {}, ''
        for line in (out / 'trace.log').read_text(errors='replace').splitlines():
            if line.startswith('QSA_TOPK '): events.append(json.loads(line[len('QSA_TOPK '):]))
            if line.startswith('QSA_TOPK_DONE '): done = json.loads(line[len('QSA_TOPK_DONE '):])
            if line.startswith('QSA TopK trace failed:'): native_error = line
        runner.json_write(out / 'events.json', events)
        if status != 0: raise RuntimeError(f'Trace exited with status {status}; {native_error or "inspect trace.log"}')
        report.update(summarize(events, done))
        for key in ('topk_devices', 'finite_cutoff_ambiguous_rows', 'changed_score_rows', 'changed_visible_set_rows',
                    'matching_score_hash_changed_visible_set_rows', 'frozen_replay_devices',
                    'frozen_max_changed_rows', 'frozen_max_changed_from_original'):
            runner.say(f'{key}: {report[key]}')
    except (Exception, KeyboardInterrupt) as exc:
        failure = f'{type(exc).__name__}: {exc}'
        report['failure'] = failure
        runner.say('Diagnostic incomplete: ' + failure)
    finally:
        if process is not None and process.poll() is None:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=10)
        runner.json_write(out / 'report.json', report)
        runner.say(f'Handoff: {runner.archive_suite(out)}')
    return int(failure is not None)


if __name__ == '__main__':
    raise SystemExit(main())
