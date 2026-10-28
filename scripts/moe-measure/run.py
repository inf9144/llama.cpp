#!/usr/bin/env python3
"""Run an existing command unchanged, with phase metadata and separate stderr trace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--phase', required=True, choices=['prefill-only', 'decode-only', 'verify-only', 'mixed-dflash', 'mixed'])
parser.add_argument('--note', default='', help='Cache budgets, warm/cold status, prompt/seed and GPU split')
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command:
    parser.error('Missing command after --')
args.output.mkdir(parents=True, exist_ok=False)
metadata = dict(schema=1, phase=args.phase, note=args.note, command=command, cwd=os.getcwd(),
                visibility={k:os.environ.get(k) for k in ['ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']},
                start_unix_ns=time.time_ns())
metadata['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
(args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
with (args.output/'stdout.log').open('wb') as out, (args.output/'trace.log').open('wb') as err:
    status = subprocess.run(command, stdout=out, stderr=err).returncode
metadata.update(exit_code=status, end_unix_ns=time.time_ns())
(args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
raise SystemExit(status if status >= 0 else 128-status)
