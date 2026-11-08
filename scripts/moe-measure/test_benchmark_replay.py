"""Check A/B workload equality and held-out cache replay accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cache_replay as cache
import gdn_ab as gdn
import handoff_trace as handoff


class BenchmarkTests(unittest.TestCase):
    def test_gdn_pairs_use_matched_work(self):
        def result(time, digest="same"):
            return dict(content_sha256=digest, usage=dict(prompt_tokens=5121, completion_tokens=128),
                        timings=dict(draft_n=125, draft_n_accepted=85, prompt_ms=8500, predicted_ms=time),
                        request_wall_s=8.5+time/1000)

        def pair(before, after):
            return {name: dict(requests=dict(cold=result(time), warm=result(time)))
                    for name, time in (("baseline", before), ("candidate", after))}

        pairs = [pair(1000, 950), pair(1500, 1485)]
        self.assertAlmostEqual(gdn.summarize(pairs)["warm"]["predicted_ms"]["paired_delta_percent"], -3)
        pairs[1]["candidate"]["requests"]["warm"] = result(1485, "different")
        with self.assertRaisesRegex(ValueError, "unterschiedliche Antworten"):
            gdn.summarize(pairs)

    @patch.object(handoff, "REPEATS", 2)
    def test_compact_ab_requires_same_payload(self):
        rows = [dict(mode=mode, misses=misses, graph_us=100 if mode == "separate" else 75, bytes=4096,
                     h2d_bytes=4096 if mode == "separate" else 4160,
                     copies=3*misses + 1 if mode == "separate" else 1)
                for misses in handoff.MISS_COUNTS for mode in ("separate", "compact") for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "fixture.log"

            def run():
                log.write_text("".join("upload_ab " + json.dumps(r) + "\n" for r in rows))
                return handoff.summarize_compact_ab(log)

            self.assertEqual(run()["phases"][3]["median_delta_percent"], -25)
            rows[0]["h2d_bytes"] -= 1
            with self.assertRaises(ValueError):
                run()

    def test_slot_ceiling_keeps_recorded_prefill_placement(self):
        group = dict(backend="ROCm1", prefill_backend="ROCm0", owner_link=64, prefill_link=256,
                     experts=512, layers=[5], min_slots=80, bank_bytes=[1024], alignment=128)
        self.assertEqual(cache.slots_at_budget(group, 999*1024), 383)

    def test_replay_selects_on_training_and_reports_holdout(self):
        groups = [dict(cache_group=idx, slots=10, allocated_bytes=11*1024, bank_bytes=[1024],
                       alignment=128, tries=choices, steps=[None])
                  for idx, choices in ((1, [8, 10]), (2, [10, 12]))]
        replay_input = dict(groups=groups, graphs=[1, 2])
        curves = [
            dict(group=1, curves=[
                dict(slots=8, verified_steps=0, misses_by_graph=[[1, 3], [2, 7]]),
                dict(slots=10, verified_steps=1, misses_by_graph=[[1, 2], [2, 2]])]),
            dict(group=2, curves=[
                dict(slots=10, verified_steps=1, misses_by_graph=[[1, 4], [2, 4]]),
                dict(slots=12, verified_steps=0, misses_by_graph=[[1, 1], [2, 2]])]),
        ]
        report = cache.recommend(replay_input, curves)
        candidate = report["candidate"]
        self.assertEqual(candidate["slots"], {1: 8, 2: 12})
        self.assertEqual(candidate["allocated_bytes"], report["allocated_bytes"])
        self.assertFalse(candidate["holdout_improves"])
        self.assertEqual(candidate["holdout_delta_percent"], 50)


if __name__ == "__main__":
    unittest.main()
