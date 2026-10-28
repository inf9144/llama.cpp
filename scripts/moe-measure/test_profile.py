import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('profile_moe', Path(__file__).with_name('profile.py'))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(profile.union_ns([(0, 10), (5, 20), (30, 40)]), 30)

    def test_free_recovers_original_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'1_memory_allocation_trace.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['Start_Timestamp', 'End_Timestamp', 'Address', 'Agent_Id', 'Allocation_Size', 'Operation'])
                writer.writeheader()
                for t, addr, agent, size, op in [(0,'0x1000','Agent 2',100,'ALLOCATE'), (1,'0x2000','Agent 2',50,'ALLOCATE'), (2,'0x1000','',0,'FREE'), (3,'0x2000','',0,'FREE')]:
                    writer.writerow(dict(Start_Timestamp=t, End_Timestamp=t, Address=addr, Agent_Id=agent, Allocation_Size=size, Operation=op))
            report = profile.summarize(Path(directory))
            self.assertEqual(report['allocation_peaks'][0]['bytes'], 150)
            self.assertEqual(report['unknown_frees'], 0)


if __name__ == '__main__':
    unittest.main()
