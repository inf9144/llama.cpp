"""Check A/B workload equality and held-out cache replay accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cache_replay as cache
import gdn_ab as gdn
import handoff_trace as handoff
import rollback_ab as rollback


class BenchmarkTests(unittest.TestCase):
    def test_rollback_launcher_and_prompt_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "request.json"
            source.write_text(json.dumps(dict(seed=42, messages=[dict(role="user", content="sample")],
                                              max_completion_tokens=128, cache_prompt=False)))
            request = rollback.prepare_request(source, 2, 256)
        self.assertTrue(request["cache_prompt"])
        self.assertEqual(request["max_tokens"], 256)
        self.assertNotIn("max_completion_tokens", request)
        self.assertEqual(request["messages"][0]["content"].count("sample"), 2)
        text = rollback.launcher(
            "./build-rocm-moe-reference/bin/llama-server --cache-ram 0 "
            "--port 8080 --spec-draft-n-max 3 --moe-cache-mib 15200,9600\n",
            Path("/tmp/rollback binary/llama-server"), (7800, 5650), Path("/tmp/rollback slot"))
        self.assertIn("--moe-cache-mib 7800,5650", text)
        self.assertIn("--slot-save-path '/tmp/rollback slot' --no-context-shift", text)
        self.assertIn("--spec-draft-sampling probabilistic", text)

    def test_rollback_summary_requires_same_output_reuse_and_verify_work(self):
        def result(trim, t_ms):
            return dict(content_sha256="same", usage=dict(prompt_tokens=150000, completion_tokens=256),
                        timings=dict(draft_n=252, draft_n_accepted=171, draft_verify_rounds=85,
                                     draft_tail_trim_us=trim, cache_n=149996, prompt_n=4, predicted_ms=t_ms))

        a, b = result(8500, 6000), result(1700, 5700)
        summary = rollback.summarize([dict(baseline=a, candidate=b)])
        self.assertEqual(summary["baseline"]["trim_us_per_verify"], 100)
        self.assertEqual(summary["candidate"]["trim_us_per_verify"], 20)
        self.assertAlmostEqual(summary["paired_decode_ms_delta_percent"][0], -5)
        b["timings"]["cache_n"] -= 4
        with self.assertRaisesRegex(ValueError, "Prompt-Reuse"):
            rollback.summarize([dict(baseline=a, candidate=b)])

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
        rows: list[dict] = [dict(mode=mode, misses=misses, graph_us=100 if mode == "separate" else 75, bytes=4096,
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
