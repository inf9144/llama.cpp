"""Check the MoE handoff join without a GPU or rocprofv3."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import handoff_trace as handoff


class HandoffTraceTests(unittest.TestCase):
    def fixture(self, directory, *, omit_copy=False, omit_patch=False, omit_finish=False):
        host, kernels, api, copies = [], [], [], []

        def event(rows, start, end, **fields):
            rows.append(dict(Start_Timestamp=start, End_Timestamp=end, **fields))

        for i, misses in enumerate((28, 1, 3, 8)):
            # Host and profiler clocks have different origins. Large copy
            # ranges overlap the last shared kernel and each other.
            start = 100_000 + i*1_000
            origin = start*1000 + 701_000
            ranges = 3*misses + 1
            host.append(dict(
                version=i+1, misses=misses, bytes=2_329_600*misses+4, pinned_bytes=2_329_600*misses+4,
                ranges=ranges, staged_copies=ranges, observed_us=start+6,
                plan_begin_us=start+7, plan_end_us=start+9, upload_begin_us=start+10,
                upload_end_us=start+30, pool_acquire_us=1, pool_wait_us=0,
                staging_us=5, ready_begin_us=start+32, ready_end_us=start+35,
            ))
            event(kernels, origin, origin+5000, Agent_Id="Agent 2", Kernel_Name="publish_lookup")
            event(kernels, origin+38000, origin+40000, Agent_Id="Agent 2", Kernel_Name="scale_f32<float>")
            if not (omit_finish and i == 1):
                event(kernels, origin+90000, origin+92000, Agent_Id="Agent 2", Kernel_Name="finish_lookup<true>")
            event(api, origin+32300, origin+34800, Function="hipStreamWriteValue64", Thread_Id=321,
                  Correlation_Id=i*1000)
            for j in range(ranges):
                correlation = i*1000+j+1
                event(api, origin+11000+100*j, origin+11080+100*j,
                      Function="hipMemcpyAsync", Thread_Id=321, Correlation_Id=correlation)
                if (omit_copy and i == 1 and j == 0) or (omit_patch and j == ranges-1):
                    continue
                event(copies, origin+20000+400*j, origin+43000+400*j, Destination_Agent_Id="Agent 2",
                      Direction="MEMORY_COPY_HOST_TO_DEVICE", Correlation_Id=correlation)

        # Startup wait on the same device is outside every measured interval.
        event(kernels, 1000, 2000, Agent_Id="Agent 2", Kernel_Name="__amd_rocclr_batchMemOp")
        event(kernels, 101_741_000, 101_790_000, Agent_Id="Agent 2", Kernel_Name="__amd_rocclr_streamOpsWait")
        for name, rows in (("kernel", kernels), ("hip_api", api), ("memory_copy", copies)):
            with (directory / f"1_{name}_trace.csv").open("w", newline="") as out:
                writer = csv.DictWriter(out, rows[0].keys())
                writer.writeheader()
                writer.writerows(reversed(rows))
        log = directory / "fixture.log"
        log.write_text("".join(f"handoff {json.dumps(row)}\n" for row in host))
        return log

    @patch.object(handoff, "REPEATS", 1)
    def test_correlated_metrics_and_missing_wait_node(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            report = handoff.summarize(raw, self.fixture(raw, omit_patch=True))
        self.assertEqual(report["rounds"], 3)
        self.assertEqual(report["agent"], "Agent 2")
        sample = report["samples"][1]
        self.assertEqual(sample["plan_us"], 2)
        self.assertEqual(sample["finish_gap_us"], 50)
        self.assertAlmostEqual(sample["h2d_us"], 23.8)
        self.assertAlmostEqual(sample["finish_gap_without_h2d_us"], 46.2)
        self.assertAlmostEqual(sample["publish_end_to_observed_us"], 1.05)
        self.assertEqual(sample["traced_wait_us"], 49)
        self.assertFalse(sample["patch_copy_traced"])
        self.assertIsNone(report["samples"][2]["traced_wait_us"])

    @patch.object(handoff, "REPEATS", 1)
    def test_incomplete_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for options in (dict(omit_copy=True), dict(omit_finish=True)):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    handoff.summarize(raw, self.fixture(raw, **options))


if __name__ == "__main__":
    unittest.main()
