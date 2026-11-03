#!/usr/bin/env python3
"""Run the existing cold/warm A/B fixture at draft widths 3, 5 and 7.

All other model and request settings stay in the existing Fish/JSON templates.
Pass --profile for ROCprofv3, --compare-sampling for an extra uninstrumented
Swift manufacturer-sampling comparison. One handoff includes all widths and failures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile

spec = importlib.util.spec_from_file_location('stream_ab', Path(__file__).with_name('stream-ab.py'))
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

MANUFACTURER_SOURCE = 'https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF#usage'
MANUFACTURER_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                             presence_penalty=0.0, frequency_penalty=0.0, repeat_penalty=1.0,
                             dynatemp_range=0.0, dynatemp_exponent=1.0, top_n_sigma=-1.0,
                             dry_multiplier=0.0, xtc_probability=0.0, typical_p=1.0,
                             mirostat=0, adaptive_target=-1.0, min_keep=0,
                             samplers=['top_k', 'top_p', 'temperature'])


def sampling_comparison(custom: Path, manufacturer: Path) -> dict:
    before = json.loads((custom / 'comparison.json').read_text())['candidate']['01-reference']['requests']
    after = json.loads((manufacturer / 'comparison.json').read_text())['candidate']['01-reference']['requests']
    result = {}
    print('\nSAMPLING-VERGLEICH (gleicher Kandidat; uninstrumentiert)\n', flush=True)
    print(f"{'Sampling':14} {'Cache':6} {'Decode t/s':>11} {'Akzeptiert/Draft':>24} {'Tokens':>6} {'Wall(s)':>9}", flush=True)
    for phase in ('cold', 'warm'):
        rows = {}
        for name, requests in (('custom', before), ('manufacturer', after)):
            r = requests[phase]
            t = r.get('timings') or {}
            drafted, accepted = t.get('draft_n', 0), t.get('draft_n_accepted', 0)
            rows[name] = dict(decode_tps=runner.rate(r, 'decode'), request_wall_s=r['request_wall_s'],
                              draft_n=drafted, draft_n_accepted=accepted,
                              acceptance=accepted/drafted if drafted else None, usage=r['usage'],
                              content_sha256=r['content_sha256'])
            acceptance = f'{accepted}/{drafted} ({100*accepted/drafted:.1f}%)' if drafted else 'n/a'
            print(f"{name:14} {phase:6} {str(rows[name]['decode_tps']):>11} {acceptance:>24} {r['usage'].get('completion_tokens', 0):6} {r['request_wall_s']:9.3f}", flush=True)
        a, b = rows['custom']['decode_tps'], rows['manufacturer']['decode_tps']
        rows['decode_change_percent'] = 100*(b/a - 1) if a and b else None
        if rows['decode_change_percent'] is not None:
            print(f"  {phase}: Hersteller-Decode {rows['decode_change_percent']:+.2f}% gegenueber Custom", flush=True)
        result[phase] = rows
    print('Unterschiedliche Texte zwischen Sampling-Profilen sind erwartet. Dies misst Geschwindigkeit und Akzeptanz, keine Antwortqualitaet.', flush=True)
    return result


def widths(value: str) -> list[int]:
    try:
        result = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Draft-Breiten müssen Ganzzahlen sein') from exc
    if not result or len(result) != len(set(result)) or any(x < 1 or x > 7 for x in result):
        raise argparse.ArgumentTypeError('Eindeutige Draft-Breiten zwischen 1 und 7 erwartet')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.home() / 'src/llama.cpp')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--draft-widths', type=widths, default=widths('3,5,7'))
    parser.add_argument('--no-archive', action='store_true')
    parser.add_argument('--compare-sampling', action='store_true', help='Zusatzlauf mit Swift-Hersteller-Sampling, nur uninstrumentiertes A/B')
    parser.add_argument('--attach', type=Path, action='append', default=[], help='Zusaetzliche Testlogs ins Handoff aufnehmen')
    args, forwarded = parser.parse_known_args()
    if any(x == '--draft-max' or x.startswith('--draft-max=') for x in forwarded):
        parser.error('--draft-max wird von dieser Suite gesetzt; --draft-widths verwenden')
    if args.compare_sampling and '--candidate-profile-only' in forwarded:
        parser.error('--compare-sampling benoetigt uninstrumentierte Referenzlaeufe')
    root = args.repo.expanduser().resolve()
    attachments = [path if path.is_absolute() else root / path for path in args.attach]
    if any(not path.is_file() for path in attachments) or len({path.name for path in attachments}) != len(attachments):
        parser.error('--attach benoetigt vorhandene Dateien mit eindeutigen Dateinamen')
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = args.output or Path('results') / f'moe-decode-ab-{stamp}'
    output = (output if output.is_absolute() else root / output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict = {'widths': args.draft_widths, 'complete': False, 'runs': {}}
    failed = False
    try:
        if attachments:
            logs = output / 'native-tests'
            logs.mkdir()
            report['attachments'] = []
            for path in attachments:
                shutil.copy2(path, logs / path.name)
                report['attachments'].append(dict(name=path.name, sha256=runner.sha(path)))
        if args.compare_sampling:
            request_parser = argparse.ArgumentParser(add_help=False)
            request_parser.add_argument('--request', type=Path, default=Path('moe-first-request.json'))
            request_args, sampling_forwarded = request_parser.parse_known_args(forwarded)
            source = request_args.request
            source = source if source.is_absolute() else root / source
            request = json.loads(source.read_text())
            request.update(MANUFACTURER_SAMPLING)
            sampling_request = output / 'manufacturer-request.json'
            runner.json_write(sampling_request, request)
            sampling_forwarded = [x for x in sampling_forwarded if x not in ('--profile', '--reference-only')]
            sampling_report: dict = dict(source=MANUFACTURER_SOURCE, checked='2026-11-03', overrides=MANUFACTURER_SAMPLING, runs={})
            report['sampling'] = sampling_report
        for width in args.draft_widths:
            target = output / f'draft-{width}'
            command = [sys.executable, str(Path(__file__).with_name('stream-ab.py')),
                       '--repo', str(root), '--output', str(target), '--draft-max', str(width), '--no-archive', *forwarded]
            print(f'\n=== Draft {width}; Target-Verify bis {width + 1} Tokens ===', flush=True)
            result = subprocess.run(command, cwd=root, check=False)
            report['runs'][str(width)] = {'exit_code': result.returncode, 'directory': target.name}
            if result.returncode:
                failed = True
                break
            if args.compare_sampling:
                sampling_target = output / f'draft-{width}-manufacturer'
                command = [sys.executable, str(Path(__file__).with_name('stream-ab.py')),
                           '--repo', str(root), '--output', str(sampling_target), '--draft-max', str(width),
                           '--no-archive', '--reference-only', '--request', str(sampling_request), *sampling_forwarded]
                print(f'\n=== Draft {width}; Swift Hersteller-Sampling ===', flush=True)
                result = subprocess.run(command, cwd=root, check=False)
                sampling_report['runs'][str(width)] = dict(exit_code=result.returncode, directory=sampling_target.name)
                if result.returncode:
                    failed = True
                    break
                comparison = sampling_comparison(target, sampling_target)
                sampling_report['runs'][str(width)]['comparison'] = comparison
                runner.json_write(output / f'sampling-draft-{width}.json', comparison)
        report['complete'] = not failed and len(report['runs']) == len(args.draft_widths)
    finally:
        (output / 'decode-suite.json').write_text(json.dumps(report, indent=2) + '\n')
        if not args.no_archive:
            archive = output.with_name(output.name + '-handoff.tar.gz')
            with tarfile.open(archive, 'w:gz') as stream:
                stream.add(output, arcname=output.name)
            print(f'Handoff-Archiv: {archive}', flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
