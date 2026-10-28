import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('analyze', Path(__file__).with_name('analyze.py'))
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


class AnalyzeTests(unittest.TestCase):
    def test_aliases_transfers_and_slots(self):
        base = dict(schema=1, context=1, graph=2, backend='ROCm1', tokens=512, graph_type=0, us=0)
        events = [dict(kind='routing', tensor='blk.3.ffn_up_exps.weight', counts=[3, 0, 2, 1], wait_us=7),
                  dict(kind='work_copy', tensor='blk.3.ffn_up_exps.weight', expert_bytes=100, address=1000, alloc_bytes=400),
                  dict(kind='work_copy', tensor='blk.3.ffn_down_exps.weight', expert_bytes=200, address=1200, alloc_bytes=800),
                  dict(kind='d2d', tensor='blk.3.ffn_up_exps.weight', expert_bytes=100, first=2, count=1, bytes=100, enqueue_us=1),
                  dict(kind='h2d', tensor='blk.3.ffn_down_exps.weight', expert_bytes=200, first=0, count=1, bytes=200, enqueue_us=2),
                  dict(kind='reservation', bytes=2048)]
        result = analyzer.analyze(['MOE_MEASURE\t'+json.dumps(base | e) for e in events], 'known-prefill')
        layer = result['layers'][0]
        self.assertEqual((layer['unique_experts'], layer['miss_experts']), (3, 2))
        self.assertEqual(layer['compact_payload_bytes'], 900)
        self.assertEqual(layer['staging']['8']['double_buffer_capacity_bytes'], 4800)
        self.assertEqual(result['weight_address_unions'][0]['observed_weight_address_union_bytes'], 1000)
        self.assertEqual(result['reservation_highwater'][0]['bytes'], 2048)

    def test_missing_trace(self):
        with self.assertRaises(ValueError):
            analyzer.analyze(['ordinary log'], 'decode')

    def test_incomplete_is_visible(self):
        row = dict(schema=1, context=1, graph=1, backend='CPU', kind='bank', tensor='unknown', tokens=1, graph_type=0, expert_bytes=128)
        report = analyzer.analyze(['MOE_MEASURE\t'+json.dumps(row)], 'mixed')
        self.assertEqual(len(report['incomplete_groups']), 1)
        self.assertFalse(report['layers'])


if __name__ == '__main__':
    unittest.main()
