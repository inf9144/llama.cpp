#!/usr/bin/env python3
"""Compare DFlash2 draft sampling and device placement with Swift sampling defaults.

Use one uninstrumented candidate binary for all four variants, with three draft
tokens. Each fresh server gets cold/warm requests. Repeat in reverse order to
reduce order bias. The existing Fish/JSON templates are copied, never edited.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess

spec = importlib.util.spec_from_file_location('decode_ab', Path(__file__).with_name('decode-ab.py'))
assert spec is not None and spec.loader is not None
decode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode)
runner = decode.runner

VARIANTS = {
    'greedy-split': ('greedy', 'ROCm0,ROCm1'),
    'prob-split': ('probabilistic', 'ROCm0,ROCm1'),
    'greedy-rocm0': ('greedy', 'ROCm0'),
    'prob-rocm0': ('probabilistic', 'ROCm0'),
}


def draft_launcher(text: str, binary: Path, sampling: str, devices: str) -> str:
    invocation = shlex.quote(str(binary))
    start = re.search(r'(?m)^' + re.escape(invocation) + r'(?=\s|$)', text)
    if start is None:
        raise ValueError('Generated server invocation not found')
    prefix, command = text[:start.end()], text[start.end():]
    added = []
    for flags, value in ((('--spec-draft-sampling',), sampling),
                         (('--spec-draft-device', '--device-draft', '-devd'), devices)):
        option = '(?:' + '|'.join(map(re.escape, flags)) + ')'
        pattern = r'(?<!\S)' + option + r'''(?:=|\s+)(?:"[^"\n]*"|'[^'\n]*'|[^\s\\]+)'''
        command, count = re.subn(pattern, lambda _: flags[0] + ' ' + shlex.quote(value), command)
        present = len(re.findall(r'(?<!\S)' + option + r'(?=[=\s]|$)', command))
        if count > 1 or present > 1 or (count == 0 and present):
            raise ValueError(f'Ambiguous draft option: {flags[0]}')
        if count == 0:
            added.append(flags[0] + ' ' + shlex.quote(value))
    return prefix + (' ' + ' '.join(added) if added else '') + command


def draft_placement(trace: Path, devices: str) -> dict:
    text = trace.read_text(encoding='utf-8', errors='replace')
    text = text.split('common_speculative_impl_draft_dflash:', 1)[0]
    groups = re.split(r'load_tensors: offloaded \d+/\d+ layers to GPU', text)
    if len(groups) != 3:
        raise ValueError('Expected target and draft model allocation logs; inspect trace.log')
    buffers = {}
    for device, size in re.findall(r'load_tensors:\s+(ROCm\d+)\s+model buffer size\s*=\s*([0-9.]+) MiB', groups[-1]):
        buffers[device] = buffers.get(device, 0.0) + float(size)
    active = {device for device, size in buffers.items() if size > 0}
    if active != set(devices.split(',')):
        raise ValueError(f'Draft placement mismatch: requested {devices}, observed {buffers}')
    return dict(model_buffers_mib=buffers,
                limitation='Model weight buffers only; not a process peak or compute/KV reservation measurement.')


def summarize(runs: list[dict]) -> dict:
    summary = {}
    print('\nDFLASH-VERGLEICH (Herstellersettings; gleicher Build; uninstrumentiert)\n', flush=True)
    print(f"{'Variante':14} {'Cache':6} {'Decode Median [min,max]':>28} {'Akzeptiert/Draft':>23} {'Wall Median(s)':>15}", flush=True)
    for variant in VARIANTS:
        rows = [r for r in runs if r['variant'] == variant]
        if not rows:
            continue
        summary[variant] = {}
        for phase in ('cold', 'warm'):
            requests = [r['result']['requests'][phase] for r in rows]
            rates = [runner.rate(r, 'decode') for r in requests]
            if any(rate is None or rate <= 0 for rate in rates):
                raise ValueError(f'Missing decode timing: {variant}/{phase}')
            rates = [float(rate) for rate in rates if rate is not None]
            drafted = sum(r['timings']['draft_n'] for r in requests)
            accepted = sum(r['timings']['draft_n_accepted'] for r in requests)
            row: dict = dict(runs=len(rows), decode_tps_median=statistics.median(rates),
                       decode_tps_min=min(rates), decode_tps_max=max(rates),
                       request_wall_s_median=statistics.median(r['request_wall_s'] for r in requests),
                       draft_n=drafted, draft_n_accepted=accepted, acceptance=accepted/drafted,
                       completion_tokens=[r['usage']['completion_tokens'] for r in requests],
                       content_hashes=[r['content_sha256'] for r in requests])
            row['repeat_hashes_equal'] = len(set(row['content_hashes'])) == 1
            summary[variant][phase] = row
            print(f"{variant:14} {phase:6} {row['decode_tps_median']:7.2f} [{min(rates):6.2f},{max(rates):6.2f}] "
                  f"{accepted:6}/{drafted:<6} ({100*accepted/drafted:4.1f}%) {row['request_wall_s_median']:15.3f}", flush=True)
    if 'greedy-split' in summary:
        print('\nDelta gegen greedy-split (Median):', flush=True)
        for variant, phases in summary.items():
            if variant == 'greedy-split':
                continue
            for phase, row in phases.items():
                base = summary['greedy-split'][phase]
                row['decode_change_tps'] = row['decode_tps_median']-base['decode_tps_median']
                row['decode_change_percent'] = 100*(row['decode_tps_median']/base['decode_tps_median']-1)
                row['wall_change_percent'] = 100*(row['request_wall_s_median']/base['request_wall_s_median']-1)
                row['acceptance_change_percentage_points'] = 100*(row['acceptance']-base['acceptance'])
                print(f"  {variant}/{phase}: Decode {row['decode_change_tps']:+.2f} t/s ({row['decode_change_percent']:+.2f}%); "
                      f"Request-Wall {row['wall_change_percent']:+.2f}%", flush=True)
    print('Texte duerfen zwischen Draft-Modi abweichen. Dies misst Durchsatz und Akzeptanz, keine Verteilungsaequivalenz oder Antwortqualitaet.', flush=True)
    return summary


def validate_result(result: dict, prompt_tokens: int | None, expected_tokens: int) -> int:
    for phase in ('cold', 'warm'):
        row = result['requests'][phase]
        usage, timings = row['usage'], row.get('timings') or {}
        tokens = usage.get('prompt_tokens')
        if not isinstance(tokens, int) or tokens <= 0 or (prompt_tokens is not None and tokens != prompt_tokens):
            raise ValueError(f'Prompt token count changed: {tokens} != {prompt_tokens}')
        prompt_tokens = tokens
        if usage.get('completion_tokens') != expected_tokens:
            raise ValueError(f'Incomplete comparison: expected {expected_tokens} output tokens, got {usage.get("completion_tokens")}')
        drafted, accepted = timings.get('draft_n'), timings.get('draft_n_accepted')
        if not isinstance(drafted, int) or drafted <= 0 or not isinstance(accepted, int) or not 0 < accepted <= drafted:
            raise ValueError(f'Missing or invalid DFlash timings: drafted={drafted}, accepted={accepted}')
        if runner.rate(row, 'decode') is None:
            raise ValueError('Missing decode timing')
    assert prompt_tokens is not None
    return prompt_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.home() / 'src/llama.cpp')
    parser.add_argument('--request', type=Path, default=Path('moe-first-request.json'))
    parser.add_argument('--binary', type=Path, default=Path('build-rocm-moe-stream-reference/bin/llama-server'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rounds', type=int, choices=range(1, 6), default=2)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--startup-timeout', type=int, default=900)
    parser.add_argument('--request-timeout', type=int, default=900)
    parser.add_argument('--shutdown-timeout', type=int, default=240)
    parser.add_argument('--no-archive', action='store_true')
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    request_path = args.request if args.request.is_absolute() else root / args.request
    source = json.loads(request_path.read_text())
    if source.get('cache_prompt') is not False or source.get('stream') is not False or type(source.get('seed')) is not int or not 0 <= source['seed'] < 4294967295:
        parser.error('Request must use cache_prompt=false, stream=false and a fixed integer seed')
    expected_tokens = source.get('max_tokens', source.get('n_predict'))
    if not isinstance(expected_tokens, int) or expected_tokens < 1:
        parser.error('Request needs a positive max_tokens or n_predict')
    runner.PORT = args.port
    if not 0 < args.port <= 65535 or runner.port_open():
        parser.error('Invalid or occupied port; stop the existing server before running')
    if not shutil.which('fish'):
        parser.error('Fish not found')
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = args.output or Path('results') / f'moe-draft-ab-{stamp}'
    output = (output if output.is_absolute() else root / output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict = dict(complete=False, rounds=args.rounds, draft_max=3, variants=VARIANTS,
                        sampling_source=decode.MANUFACTURER_SOURCE, sampling=decode.MANUFACTURER_SAMPLING,
                        runs=[], errors=[],
                        limitations=['Different draft modes can produce different texts at the same seed.',
                                     'Two repetitions of one prompt are not a broad workload or quality evaluation.',
                                     'Cold means a fresh server/expert cache; the OS page cache is not cleared.',
                                     'Explicit CLI requests the sampling mode; server properties do not expose its execution.',
                                     'No GPU profiler or allocation-peak measurement.'])
    try:
        shutil.copy2(request_path, output / 'source-request.json')
        shutil.copy2(root / 'moe-server-reference.fish', output / 'template-reference.fish')
        source.update(decode.MANUFACTURER_SAMPLING)
        payload = (json.dumps(source, ensure_ascii=False) + '\n').encode()
        (output / 'request.json').write_bytes(payload)
        options = argparse.Namespace(candidate_reference=args.binary, draft_max=3)
        entry = runner.prepare_launchers(root, output, options, variants=('candidate',), kinds=('reference',))['candidate']['reference']
        binary = Path(entry['binary'])
        report['binary'] = dict(path=str(binary), sha256=entry['binary_sha256'])
        report['shared_libraries'] = {p.name: runner.sha(p) for p in sorted(binary.parent.glob('*.so*')) if p.is_file()}
        report['repo_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        cache = binary.parent.parent / 'CMakeCache.txt'
        if cache.is_file():
            if not re.search(r'(?m)^LLAMA_MOE_MEASURE:[^=]+=OFF$', cache.read_text()):
                raise ValueError('Expected LLAMA_MOE_MEASURE=OFF; use run_draft_ab.fish to build')
            shutil.copy2(cache, output / 'CMakeCache.txt')
        launchers = {}
        template = entry['script'].read_text()
        for name, (sampling, devices) in VARIANTS.items():
            script = output / f'launch-{name}.fish'
            script.write_text(draft_launcher(template, binary, sampling, devices))
            subprocess.run(['fish', '--no-config', '-n', str(script)], check=True, capture_output=True, text=True)
            launchers[name] = script
        env = os.environ.copy()
        for key in ('LLAMA_ARG_SPEC_DRAFT_SAMPLING', 'LLAMA_ARG_SPEC_DRAFT_DEVICE'):
            env.pop(key, None)
        prompt_tokens = None
        build_id = None
        for repeat in range(args.rounds):
            order = list(VARIANTS)
            if repeat % 2:
                order.reverse()
            for name in order:
                target = output / f'round-{repeat + 1}' / name / '01-reference'
                runner.say(f'=== Round {repeat + 1}/{args.rounds}: {name} ===')
                result = runner.run_phase(root, target, '01-reference', launchers[name], False,
                                          payload, env, args, candidate=True, fish_no_config=True)
                row = dict(repeat=repeat + 1, variant=name, directory=str(target.relative_to(output)), result=result)
                report['runs'].append(row)
                runner.json_write(output / 'draft-suite.json', report)
                prompt_tokens = validate_result(result, prompt_tokens, expected_tokens)
                row['draft_placement'] = draft_placement(target / 'instrumented/trace.log', VARIANTS[name][1])
                props = json.loads((target / 'props.json').read_text())
                current_build = props.get('build_info')
                if current_build is None or (build_id is not None and current_build != build_id):
                    raise ValueError(f'Runtime build changed: {current_build} != {build_id}')
                build_id = current_build
                row['runtime_build'] = current_build
        report['complete'] = True
    except KeyboardInterrupt:
        report['errors'].append('Interrupted by user')
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        runner.say(report['errors'][-1])
    finally:
        if report['complete']:
            report['summary'] = summarize(report['runs'])
        runner.json_write(output / 'draft-suite.json', report)
        if not args.no_archive:
            runner.say(f'Handoff-Archiv: {runner.archive_suite(output)}')
        runner.say(f'Ergebnisse: {output}; ' + ('FERTIG' if report['complete'] else 'UNVOLLSTAENDIG'))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
