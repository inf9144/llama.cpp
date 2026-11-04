#!/usr/bin/env python3
"""Check runtime build IDs before preserving an A/B reference/measurement pair."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def build_id(binary: Path) -> str:
    binary = binary.resolve()
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(binary.parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    result = subprocess.run([str(binary), '--version'], env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True, timeout=30)
    match = re.search(r'\(build (\d+), commit ([0-9a-f]{7,40}(?:-dirty)?)\)', result.stdout)
    if match is None:
        raise ValueError(f'No runtime build ID from {binary}: {result.stdout[:500]}')
    return f'b{match[1]}-{match[2]}'


def check_pair(reference: Path, measure: Path) -> dict:
    ids = dict(reference=build_id(reference), measure=build_id(measure))
    if ids['reference'] != ids['measure']:
        raise ValueError(f"Different runtime builds: reference={ids['reference']}, measure={ids['measure']}. "
                         'Rebuild both from the same source before saving a profiling baseline.')
    return ids


def snapshot(reference: Path, measure: Path, destination: Path) -> dict:
    if destination.exists():
        return check_pair(destination / 'reference/llama-server', destination / 'measure/llama-server')
    ids = check_pair(reference, measure)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + '-', dir=destination.parent) as folder:
        staged = Path(folder) / 'snapshot'
        for name, binary in (('reference', reference), ('measure', measure)):
            shutil.copytree(binary.resolve().parent, staged / name, symlinks=True)
        if check_pair(staged / 'reference/llama-server', staged / 'measure/llama-server') != ids:
            raise ValueError('Build changed while saving the baseline')
        (staged / 'build-pair.json').write_text(json.dumps(ids, indent=2) + '\n')
        staged.rename(destination)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('measure', type=Path)
    parser.add_argument('destination', type=Path, nargs='?')
    args = parser.parse_args()
    try:
        ids = (snapshot(args.reference, args.measure, args.destination) if args.destination is not None
               else check_pair(args.reference, args.measure))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'{exc}\n')
    print('Matched runtime builds: ' + ids['reference'])


if __name__ == '__main__':
    main()
