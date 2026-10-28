#!/usr/bin/env python3
"""Summarize rocprofv3 CSV exports; retain agent IDs, never assume GPU ordinals."""
import argparse
import collections
import csv
import json
from pathlib import Path


def union_ns(spans):
    total, end = 0, -1
    for a, b in sorted(spans):
        if b < a:
            raise ValueError('Reversed profiler timestamps')
        total += max(0, b-max(a, end))
        end = max(end, b)
    return total


def summarize(directory):
    kernels = collections.defaultdict(list)
    copies = collections.defaultdict(list)
    active = collections.defaultdict(list)
    allocations = []
    found = []
    tables = collections.Counter()
    for path in sorted(directory.rglob('*.csv')):
        if not any(path.name.endswith(s) for s in ('kernel_trace.csv', 'memory_copy_trace.csv', 'memory_allocation_trace.csv')):
            continue
        found.append(str(path))
        tables[next(s for s in ('kernel_trace.csv', 'memory_copy_trace.csv', 'memory_allocation_trace.csv') if path.name.endswith(s))] += 1
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                start, end = int(row['Start_Timestamp']), int(row['End_Timestamp'])
                if path.name.endswith('kernel_trace.csv'):
                    key = (row['Agent_Id'], row['Kernel_Name'])
                    kernels[key].append((start, end))
                    active[row['Agent_Id']].append((start, end))
                elif path.name.endswith('memory_copy_trace.csv'):
                    copies[(row['Direction'], row['Source_Agent_Id'], row['Destination_Agent_Id'])].append((start, end))
                    for agent in {row['Source_Agent_Id'], row['Destination_Agent_Id']}:
                        active[agent].append((start, end))
                else:
                    # Partition by file/process: virtual addresses can repeat across processes.
                    allocations.append((end, str(path), row))
    if not found:
        raise ValueError('No supported rocprofv3 CSV exports found')
    live = {}
    sizes = collections.defaultdict(int)
    peaks = collections.defaultdict(int)
    unknown_frees = 0
    unsupported_allocations = 0
    for _, process, row in sorted(allocations, key=lambda x: x[0]):
        addr = int(row['Address'], 0)
        if not addr:
            continue
        key = (process, addr)
        if 'FREE' in row['Operation']:
            item = live.pop(key, None)
            if item is None:
                unknown_frees += 1
            else:
                agent, size = item
                sizes[(process, agent)] -= size
        elif 'ALLOCATE' in row['Operation']:
            if key in live:
                raise ValueError(f'Duplicate live allocation address {key}')
            agent, size = row['Agent_Id'], int(row['Allocation_Size'])
            live[key] = (agent, size)
            sizes[(process, agent)] += size
            peaks[(process, agent)] = max(peaks[(process, agent)], sizes[(process, agent)])
        else:
            unsupported_allocations += 1
    return dict(files=found, tables=dict(tables),
                coverage_warnings=(['Missing allocation table: no allocation peak can be established.'] if not tables['memory_allocation_trace.csv'] else []) +
                    (['Allocation accounting is incomplete: inspect unknown frees/operations.'] if unknown_frees or unsupported_allocations else []),
                kernels=[dict(agent=k[0], name=k[1], calls=len(v), summed_ns=sum(b-a for a,b in v), active_union_ns=union_ns(v)) for k,v in kernels.items()],
                copies=[dict(direction=k[0], source=k[1], destination=k[2], calls=len(v), summed_ns=sum(b-a for a,b in v), active_union_ns=union_ns(v)) for k,v in copies.items()],
                combined_active_union_ns={k:union_ns(v) for k,v in active.items()},
                allocation_peaks=[dict(process=p, agent=a, bytes=n) for (p,a),n in peaks.items()],
                unknown_frees=unknown_frees, unsupported_allocation_events=unsupported_allocations,
                limitations=['Kernel names are retained; dense and MoE GEMMs must not be blindly conflated.',
                             'Summed durations overlap; use union durations, not their sum, for active time.',
                             'Allocation peaks are process/agent HSA allocations, not expert-only live bytes or driver VRAM.',
                             'Keep each collection in its own directory; duplicate exports would double-count.',
                             'Copy bytes come from the host expert trace when profiler CSV has no size field.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory), indent=2)+'\n')
