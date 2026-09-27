import unittest
from methods.openpi_rlt.plug_v2.control_clock import next_deadline
class ControlClockTests(unittest.TestCase):
 def test_late_initial_inference(self):
  self.assertGreaterEqual(next_deadline(0.,.1035,.103)-.103,1/30)
 def test_short_remaining_slack(self):
  self.assertGreaterEqual(next_deadline(0.,.0331,.033)-.033,1/30-1e-12)
 def test_jitter(self):
  import random
  rng=random.Random(42);deadline=0.;sent=[]
  for _ in range(10000):
   publication=deadline+rng.choice([0.,.0002,.003,.04,.1]);sent.append(publication)
   deadline=next_deadline(deadline,publication+.0001,publication)
  self.assertTrue(all(b-a>=1/30-1e-10 for a,b in zip(sent,sent[1:])))
 def test_nominal_frequency(self):
  tick=0.
  for _ in range(300):tick=next_deadline(tick,tick,tick)
  self.assertAlmostEqual(tick,10.,places=9)
 def test_invalid(self):
  for period in (0,-1):
   with self.assertRaises(ValueError):next_deadline(0,0,period=period)
if __name__=='__main__':unittest.main()
