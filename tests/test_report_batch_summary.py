from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark'))
from report_batches import _summary

class BatchSummary(unittest.TestCase):
    def test_last_gpu_call_is_not_middle_small_batch(self):
        records={i:(1,128 if i<256 else 10) for i in range(266)}
        self.assertEqual(_summary(records,266,True,False),dict(regular=128,tail=10,irregular=False,observed=[10,128]))
    def test_true_middle_small_call(self):
        records={i:(1,128 if i<128 else (64 if i<192 else 128)) for i in range(320)}
        self.assertTrue(_summary(records,320,True,False)['irregular'])
    def test_joined_frames_and_cpu(self):
        self.assertEqual(_summary({0:(128,256),128:(128,256),256:(10,10)},266,True,False)['tail'],10)
        self.assertEqual(_summary({0:(1,128),1:(1,128)},2,True,True)['regular'],1)
        self.assertIsNone(_summary({0:(1,128)},1,True,False))
