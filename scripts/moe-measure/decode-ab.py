#!/usr/bin/env python3
"""Run the existing cold/warm A/B fixture at draft widths 3, 5 and 7.

All other model and request settings stay in the existing Fish/JSON templates.
Pass --profile for ROCprofv3. One handoff contains all widths, including failures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import tarfile


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
    args, forwarded = parser.parse_known_args()
    if any(x == '--draft-max' or x.startswith('--draft-max=') for x in forwarded):
        parser.error('--draft-max wird von dieser Suite gesetzt; --draft-widths verwenden')
    root = args.repo.expanduser().resolve()
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = args.output or Path('results') / f'moe-decode-ab-{stamp}'
    output = (output if output.is_absolute() else root / output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict = {'widths': args.draft_widths, 'complete': False, 'runs': {}}
    failed = False
    try:
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
