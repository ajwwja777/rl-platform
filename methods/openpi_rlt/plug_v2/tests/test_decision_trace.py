import unittest
import numpy as np
from methods.openpi_rlt.plug_v2.decision_trace import decision_evidence, delayed_target
from methods.openpi_rlt.plug_v2.rtc_queue import RTCQueue

class DecisionTraceTest(unittest.TestCase):
    def plan(self, req):
        p=np.tile(np.arange(50,dtype=np.float32)[:,None],(1,14))
        p[:req.prefix_length]=req.prefix[:req.prefix_length]
        return p
    def test_delayed_action_is_saved_before_hil_truncates_it(self):
        q=RTCQueue();q.resume();first=q.request(np.zeros(14));p=self.plan(first)
        self.assertTrue(q.complete(first,p))
        for _ in range(4):q.pop()
        req=q.request(np.zeros(14));self.assertEqual(req.prefix_length,6)
        p=self.plan(req);e=decision_evidence(req,p,p,p,actor_key='immutable-checkpoint')
        self.assertTrue(q.complete(req,p))
        sent=np.array([q.pop() for _ in range(8)])
        np.testing.assert_array_equal(sent,p[:8])
        q.pause('HIL')
        self.assertEqual(e['action_start_tick'],10)
        self.assertEqual(e['action_stop_tick'],20)
        self.assertEqual(e['nominal_next_decision_tick'],14)
        self.assertEqual(e['conditioned_plan'][6:16].shape,(10,14))
        p[:]=999
        self.assertFalse(np.any(e['conditioned_plan']==999))
    def test_late_plan_cannot_be_accepted_after_pause(self):
        q=RTCQueue();q.resume();req=q.request(np.zeros(14));p=self.plan(req)
        decision_evidence(req,p,p,p);q.pause('operator');self.assertFalse(q.complete(req,p))
    def test_changed_prefix_is_rejected(self):
        q=RTCQueue();q.resume();req=q.request(np.zeros(14));p=self.plan(req);q.complete(req,p)
        for _ in range(4):q.pop()
        req=q.request(np.zeros(14));p=self.plan(req);bad=p.copy();bad[0,7]+=1
        with self.assertRaisesRegex(ValueError,'committed prefix'):decision_evidence(req,p,bad,p)
    def test_regular_c10_target_equals_original_chunk_discount(self):
        rewards=np.arange(10)/10;g=.99
        self.assertAlmostEqual(delayed_target(rewards,10,.7,gamma=g,terminal=False),
                               sum(g**i*r for i,r in enumerate(rewards))+g**10*.7)
    def test_terminal_in_extra_prefix_ticks_keeps_true_reward_time(self):
        rewards=np.zeros(14);rewards[-1]=1
        self.assertAlmostEqual(delayed_target(rewards,14,999,gamma=.99,terminal=True),.99**13)
    def test_runtime_request_schedule_has_one_decision_per_tick(self):
        q=RTCQueue();q.resume();requests=[]
        for tick in range(100):
            if q.request_due:
                req=q.request(np.zeros(14));requests.append((req.start_tick,req.prefix_length))
                q.complete(req,self.plan(req))
                # Runtime checks request_due again after the future returns.
                self.assertFalse(q.request_due)
            q.pop()
        self.assertEqual(requests,[(0,0)]+[(i,6) for i in range(4,100,10)])
        q.pause();q.resume();self.assertTrue(q.request_due)
        self.assertEqual(q.request(np.zeros(14)).prefix_length,0)

    def test_uncorrected_reference_tail_is_never_queued(self):
        q=RTCQueue();q.resume();req=q.request(np.zeros(14));p=self.plan(req)
        p[10:]=999  # Deliberately unsafe unused reference tail.
        q.complete(req,p)
        self.assertEqual(sorted(q.commands),list(range(10)))
        for _ in range(4):q.pop()
        self.assertTrue(q.request_due)
        req=q.request(np.zeros(14));p=self.plan(req);p[16:]=999;q.complete(req,p)
        self.assertEqual(sorted(q.commands),list(range(4,20)))
        self.assertFalse(any(np.any(x==999) for x in q.commands.values()))

    def test_wrong_reward_span_rejected(self):
        with self.assertRaises(ValueError):delayed_target([0]*10,14,0,gamma=.99,terminal=True)

if __name__=='__main__':unittest.main()
