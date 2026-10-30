#!/usr/bin/env python3
"""Compare repeated greedy tokens and fixed-prefix probabilities, baseline/candidate.

Uses the existing reference binaries, cache budgets and draft maximum 3. This is
a numerical diagnostic, not a speed benchmark. No backend or sampler code changes.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request

spec = importlib.util.spec_from_file_location('stream_ab', Path(__file__).with_name('stream-ab.py'))
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

CASES = ('baseline', 'candidate')


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def first_difference(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def greedy_request(prompt: list[int], count: int, seed: int) -> dict:
    return dict(prompt=prompt, n_predict=count, seed=seed, temperature=0.0,
                dynatemp_range=0.0, mirostat=0, samplers=['temperature'], cache_prompt=False,
                stream=False, return_tokens=True, n_probs=32, post_sampling_probs=False)


def validate_completion(data: dict, prompt: list[int], single: bool = False) -> None:
    tokens = data.get('tokens')
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('Missing generated token IDs')
    timing = data.get('timings') or {}
    if timing.get('cache_n') != 0 or timing.get('prompt_n') != len(prompt):
        raise ValueError('Prompt token count or prompt-cache check failed')
    if single and (len(tokens) != 1 or timing.get('draft_n', 0) != 0):
        raise ValueError('Single-token replay must run without speculative drafts')
    rows = data.get('completion_probabilities') or []
    if len(rows) < len(tokens) or [r.get('id') for r in rows[:len(tokens)]] != tokens:
        raise ValueError('Generated tokens and probability rows do not align')
    for token, row in zip(tokens, rows):
        top = row.get('top_logprobs') or []
        if not top or any(not isinstance(t.get('logprob'), (int, float)) or not math.isfinite(t['logprob']) for t in top):
            raise ValueError('Missing or non-finite pre-sampling probabilities')
        best = max(t['logprob'] for t in top)
        if token not in {t['id'] for t in top if t['logprob'] == best}:
            raise ValueError('Generated token is not a raw-probability argmax; inspect sampler settings')


def probability_delta(a: dict, b: dict) -> dict:
    left = {x['id']: x['logprob'] for x in a['top_logprobs']}
    right = {x['id']: x['logprob'] for x in b['top_logprobs']}
    shared = left.keys() & right.keys()
    def margin(values):
        ordered = sorted(values.values(), reverse=True)
        return ordered[0] - ordered[1] if len(ordered) > 1 else None
    return dict(shared_top_ids=len(shared), only_left=sorted(left.keys() - right.keys()),
                only_right=sorted(right.keys() - left.keys()),
                max_shared_logprob_delta=max((abs(left[t] - right[t]) for t in shared), default=None),
                left_top1_margin=margin(left), right_top1_margin=margin(right))


def compare(a: dict, b: dict) -> dict:
    index = first_difference(a['tokens'], b['tokens'])
    limit = min(len(a['tokens']), len(b['tokens']), index + 1 if index is not None else len(a['tokens']))
    deltas = [probability_delta(a['completion_probabilities'][i], b['completion_probabilities'][i]) for i in range(limit)]
    result = dict(tokens_equal=index is None, first_different_token=index,
                  left_tokens=len(a['tokens']), right_tokens=len(b['tokens']), compared_prefix_rows=limit,
                  rows_with_top_id_changes=sum(bool(d['only_left'] or d['only_right']) for d in deltas),
                  minimum_shared_top_ids=min((d['shared_top_ids'] for d in deltas), default=0),
                  max_shared_logprob_delta=max((d['max_shared_logprob_delta'] for d in deltas
                                               if d['max_shared_logprob_delta'] is not None), default=None))
    if index is not None and index < limit:
        result['at_first_difference'] = dict(left=a['completion_probabilities'][index],
                                              right=b['completion_probabilities'][index], delta=deltas[index])
    return result


def launch_text(source: str, binary: Path) -> str:
    if re.search(r'--cache-ssd(?:-\S+)?', source) or '--cache-ram 0' not in source:
        raise ValueError('Existing template must disable SSD/RAM prompt caching')
    ports = re.findall(r'--port\s+(\d+)', source)
    if len(ports) != 1 or int(ports[0]) != runner.PORT:
        raise ValueError('Expected port 8080 in the existing template')
    text, n = re.subn(r'(?m)^\s*\./build-rocm-moe-reference/bin/llama-server(?=\s|$)', lambda _: shlex.quote(str(binary)), source)
    if n != 1:
        raise ValueError('Reference server path is not unique in moe-server-reference.fish')
    text, n = re.subn(r'(--spec-draft-n-max\s+)\d+', lambda m: m[1] + '3', text)
    if n != 1:
        raise ValueError('Draft maximum is not unique in the existing template')
    values = re.findall(r'--moe-cache-mib\s+([0-9]+(?:,[0-9]+)*)(?=\s|$)', text)
    if len(values) != 1 or not any(int(v) > 0 for v in values[0].split(',')):
        raise ValueError('Expected one positive MoE cache budget in the existing template')
    return text


def http(folder: Path, name: str, endpoint: str, payload: dict | None = None) -> dict:
    if payload is not None:
        runner.json_write(folder / (name + '-request.json'), payload)
    request = urllib.request.Request(f'http://{runner.HOST}:{runner.PORT}/{endpoint}',
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        (folder / (name + '-error.txt')).write_bytes(exc.read())
        raise RuntimeError(f'{endpoint}: HTTP {exc.code}; inspect {folder}') from exc
    (folder / (name + '.json')).write_bytes(raw)
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get('error'):
        raise RuntimeError(f'{endpoint}: invalid response; inspect {folder}')
    return data


@contextlib.contextmanager
def server(root: Path, folder: Path, launcher: Path, candidate: bool):
    folder.mkdir(parents=True)
    if runner.port_open():
        raise RuntimeError('Port 8080 is occupied; no unrelated process will be stopped')
    cmd = [sys.executable, 'scripts/moe-measure/run.py', '--output', str(folder / 'instrumented'),
           '--phase', 'mixed-dflash', '--note', 'Greedy correctness diagnostic, baseline/candidate, unchanged cache budgets',
           '--', 'fish', str(launcher)]
    runner.json_write(folder / 'launch.json', cmd)
    with (folder / 'controller.log').open('wb') as log:
        process = subprocess.Popen(cmd, cwd=root, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            runner.wait_health(process, 900)
            trace = folder / 'instrumented/trace.log'
            if candidate:
                runner.require_candidate_scratch(trace)
            http(folder, 'props', 'props')
            yield
        finally:
            if process.poll() is None:
                runner.stop_server(process, 240)
        metadata = json.loads((folder / 'instrumented/metadata.json').read_text())
        if process.returncode != 0 or metadata.get('exit_code') != 0:
            raise RuntimeError(f'Server exited unexpectedly; inspect {folder}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.home() / 'src/llama.cpp')
    parser.add_argument('--request', type=Path, default=Path('moe-first-request.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    binaries = {case: root / runner.BINARIES[case][0] for case in CASES}
    for binary in binaries.values():
        if not binary.is_file():
            parser.error(f'Missing reference binary: {binary}')
    source = (root / 'moe-server-reference.fish').read_text()
    request = json.loads((args.request if args.request.is_absolute() else root / args.request).read_text())
    messages = request.get('messages')
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get('content'), str) for m in messages):
        parser.error('This diagnostic requires a text-only chat request')
    count, seed = request.get('max_tokens', 128), request.get('seed')
    if type(count) is not int or not 1 <= count <= 512 or type(seed) is not int or not 0 <= seed < 2**32 - 1:
        parser.error('Expected 1..512 max_tokens and a fixed uint32 seed other than the random-seed sentinel')
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out = args.output or Path('results') / f'moe-completion-diagnose-{stamp}'
    out = (out if out.is_absolute() else root / out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runner.json_write(out / 'original-request.json', request)
    provenance = {}
    for case, binary in binaries.items():
        libs = {p.resolve() for pattern in ('libllama*.so*', 'libggml*.so*') for p in binary.parent.glob(pattern)}
        provenance[case] = dict(binary=str(binary), binary_sha256=runner.sha(binary),
                                library_sha256={str(p): runner.sha(p) for p in sorted(libs)})
    report: dict = dict(complete=False, binaries=provenance, draft_max=3,
                        note='Existing baseline/candidate and cache budgets. Plain greedy; no timing conclusions or full-vocabulary logit equivalence.',
                        greedy={}, comparisons={}, replay={})
    prompt = None
    rendered = None
    launchers = {}
    failure = None
    try:
        for case in CASES:
            launcher = out / (case + '.fish')
            launcher.write_text(launch_text(source, binaries[case]))
            launchers[case] = launcher
            folder = out / case / 'greedy'
            runner.say(f'{case}: repeated greedy generation, draft maximum 3')
            with server(root, folder, launcher, case == 'candidate'):
                text = http(folder, 'template', 'apply-template', request).get('prompt')
                if not isinstance(text, str):
                    raise ValueError('Expected a text prompt from apply-template')
                ids = http(folder, 'tokenize', 'tokenize', dict(content=text, add_special=True, parse_special=True)).get('tokens')
                if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
                    raise ValueError('Invalid prompt token IDs')
                if prompt is None:
                    prompt, rendered = ids, text
                    runner.json_write(out / 'prompt-tokens.json', prompt)
                    report['prompt_sha256'] = digest(prompt)
                elif ids != prompt or text != rendered:
                    raise ValueError('Rendered prompt differs between runs; see template/tokenize responses')
                runs = []
                for repeat in range(2):
                    data = http(folder, f'repeat-{repeat}', 'completion', greedy_request(prompt, count, seed))
                    validate_completion(data, prompt)
                    runs.append(data)
                    runner.say(f'{case}/repeat-{repeat}: {len(data["tokens"])} tokens, token hash {digest(data["tokens"])[:12]}')
                report['greedy'][case] = runs
            runner.json_write(out / 'report.json', report)
        left, right = report['greedy']['baseline'], report['greedy']['candidate']
        report['comparisons'] = dict(baseline_repeat=compare(left[0], left[1]), candidate_repeat=compare(right[0], right[1]),
                                     baseline_candidate_first=compare(left[0], right[0]), baseline_candidate_second=compare(left[1], right[1]))
        differences = [x['first_different_token'] for x in report['comparisons'].values() if x['first_different_token'] is not None]
        positions = sorted({0, min(differences) if differences else min(32, len(left[0]['tokens']) - 1)})
        report['replay_positions'] = positions
        assert prompt is not None
        for case in CASES:
            folder = out / case / 'replay'
            report['replay'][case] = {}
            runner.say(f'{case}: fixed-prefix next-token replay at {positions}; no speculative drafts')
            with server(root, folder, launchers[case], case == 'candidate'):
                for position in positions:
                    prefix = prompt + left[0]['tokens'][:position]
                    data = http(folder, f'prefix-{position}', 'completion', greedy_request(prefix, 1, seed))
                    validate_completion(data, prefix, single=True)
                    report['replay'][case][str(position)] = data
            runner.json_write(out / 'report.json', report)
        report['replay_comparisons'] = {str(p): compare(report['replay']['baseline'][str(p)], report['replay']['candidate'][str(p)]) for p in positions}
        report['complete'] = True
        for name, comparison in {**report['comparisons'], **{'replay_' + k: v for k, v in report['replay_comparisons'].items()}}.items():
            runner.say(f'{name}: equal tokens={comparison["tokens_equal"]}, first difference={comparison["first_different_token"]}, max shared logprob delta={comparison["max_shared_logprob_delta"]}')
    except (Exception, KeyboardInterrupt) as exc:
        failure = f'{type(exc).__name__}: {exc}'
        report['failure'] = failure
        runner.say('Diagnostic incomplete: ' + failure)
    finally:
        runner.json_write(out / 'report.json', report)
        runner.say(f'Handoff: {runner.archive_suite(out)}')
    return int(failure is not None)


if __name__ == '__main__':
    raise SystemExit(main())
