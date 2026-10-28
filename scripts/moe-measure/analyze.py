#!/usr/bin/env python3
"""Analyze measurement-build stderr. All timing fields here are HOST times."""
import argparse
import collections
import json
import math
import re
from pathlib import Path


def analyze(lines, phase):
    groups = {}
    reservations = collections.defaultdict(int)
    copies = collections.defaultdict(list)
    count = 0
    for line in lines:
        if not line.startswith('MOE_MEASURE\t'):
            continue
        event = json.loads(line.split('\t', 1)[1])
        if event.get('schema') != 1:
            raise ValueError('Unsupported measurement schema')
        count += 1
        device = (event['context'], event.get('backend', ''))
        if event['kind'] == 'reservation':
            reservations[device] = max(reservations[device], event['bytes'])
            continue
        if 'tensor' not in event:
            continue
        match = re.search(r'(?:^|\.)blk\.(\d+)\.', event['tensor'])
        layer = 'blk.' + match[1] if match else event['tensor']
        key = (event['context'], event['graph'], event.get('backend', ''), layer)
        group = groups.setdefault(key, dict(phase=phase, context=key[0], graph=key[1], backend=key[2], layer=layer,
                                           tokens=event['tokens'], graph_type=event['graph_type'], banks={},
                                           h2d_bytes=0, d2d_bytes=0, routing_wait_us=0, upload_enqueue_us=0,
                                           counts=None, hits=set()))
        kind = event['kind']
        if kind == 'routing':
            histogram = event['counts']
            if group['counts'] is not None and group['counts'] != histogram:
                raise ValueError(f'Conflicting routing records for {key}; do not silently merge')
            group['counts'] = histogram
            group['routing_wait_us'] += event['wait_us']
            if 'resident' in event:
                group['hits'].update(i for i, n in enumerate(histogram) if n and event['resident'][i])
        if 'expert_bytes' in event:
            group['banks'][event['tensor']] = event['expert_bytes']
        if kind in ('h2d', 'admission_h2d'):
            group['h2d_bytes'] += event['bytes']
            group['upload_enqueue_us'] += event['enqueue_us']
        elif kind == 'd2d':
            group['d2d_bytes'] += event['bytes']
            group['hits'].update(range(event['first'], event['first'] + event['count']))
        elif kind == 'work_copy':
            copies[(event['context'], event['graph'], event['backend'], event.get('split', 0))].append((event['address'], event['address'] + event['alloc_bytes']))
    if not count:
        raise ValueError('No MOE_MEASURE records; enable LLAMA_MOE_MEASURE at build time')
    results = []
    incomplete = []
    for key, group in groups.items():
        if group['counts'] is None:
            incomplete.append(list(key))
            continue
        used = sum(n > 0 for n in group['counts'])
        misses = sum(n > 0 and i not in group['hits'] for i, n in enumerate(group['counts']))
        bundle = sum(group['banks'].values())
        group['unique_experts'] = used
        group['resident_experts'] = used - misses
        group['miss_experts'] = misses
        group['compact_payload_bytes'] = used * bundle
        group['expert_bundle_bytes'] = bundle
        group['staging'] = {str(slots): dict(capacity_bytes=slots*bundle, double_buffer_capacity_bytes=2*slots*bundle,
                                           miss_packets=math.ceil(misses/slots)) for slots in (8, 16, 32)}
        del group['hits']
        results.append(group)
    address_unions = []
    for key, spans in copies.items():
        total = 0
        end = -1
        for a, b in sorted(spans):
            total += max(0, b - max(a, end))
            end = max(end, b)
        address_unions.append(dict(context=key[0], graph=key[1], backend=key[2], split=key[3], observed_weight_address_union_bytes=total))
    return dict(schema=1, phase=phase, records=count, layers=results,
                reservation_highwater=[dict(context=k[0], backend=k[1], bytes=v) for k, v in reservations.items()],
                weight_address_unions=address_unions, incomplete_groups=incomplete,
                limitations=['Host enqueue/wait times are not GPU durations.',
                             'Per-split input-copy address union is NOT whole-process or cross-split simultaneous live peak.',
                             'Kernel durations and allocation lifetimes require the GPU profiler.',
                             'Generic scheduler transfers outside expert callbacks are not counted here.',
                             'Staging estimates exclude activations, metadata and alignment; banks assume an expert bundle.',
                             'Phase is an external workload label, never inferred from token width.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--phase', required=True, help='Known workload label, e.g. prefill-only or mixed-dflash')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.trace.open() as stream:
        report = analyze(stream, args.phase)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['records']} events, {len(report['layers'])} layer batches, {len(report['incomplete_groups'])} incomplete groups")


if __name__ == '__main__':
    main()
