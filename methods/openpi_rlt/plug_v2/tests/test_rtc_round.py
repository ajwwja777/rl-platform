import unittest,math
import numpy as np
from methods.openpi_rlt.plug_v2 import rtc_upstream_core  # adds the pinned upstream src path
from methods.openpi_rlt.plug_v2.rtc_round import wilson,fisher_two_sided,auc,bootstrap_auc,deployment_reasons,choose_checkpoint,value_report,assign_releases,read_switch_log,arm_stats
from methods.openpi_rlt.plug_v2.awbc_train import remaining_frames,discounted_success

def summary(mse=3e-5,accel=2e-4,ref=4e-5):
    return {d:{'deployment_matched_mse':mse,'deployment_accel_rms':accel,'ref_matched_conditioner_mse':ref} for d in ('0','6')}

class RoundTest(unittest.TestCase):
    def test_wilson_and_fisher(self):
        lo,hi=wilson(4,10);self.assertAlmostEqual(lo,.168,places=2);self.assertAlmostEqual(hi,.687,places=2)
        self.assertTrue(math.isnan(wilson(0,0)[0]))
        self.assertAlmostEqual(fisher_two_sided(34,49,8,73),fisher_two_sided(8,73,34,49))
        self.assertLess(fisher_two_sided(34,49,8,73),1e-4)
        self.assertAlmostEqual(fisher_two_sided(3,2,3,2),1.)

    def test_auc_and_bootstrap(self):
        self.assertEqual(auc([1,2],[0]),1.);self.assertEqual(auc([0],[1,2]),0.);self.assertEqual(auc([1],[1]),.5)
        v=[.9,.8,.7,.2,.1,.3,.4];y=[1,1,1,0,0,0,0];p,lo,hi=bootstrap_auc(v,y,iters=300)
        self.assertEqual(p,1.);self.assertLessEqual(lo,hi);self.assertEqual(bootstrap_auc(v,y,iters=300),(p,lo,hi))

    def test_deployment_gate_only_checks_safety_metrics(self):
        base={'summary':summary()}
        self.assertEqual(deployment_reasons(base,{'summary':summary(mse=3.1e-5)}),[])
        self.assertEqual(deployment_reasons(base,{'summary':summary(mse=3.3e-5)}),['0:deployment_matched_mse','6:deployment_matched_mse'])
        self.assertIn('0:reference_regression',deployment_reasons(base,{'summary':summary(mse=3e-5,ref=2e-5)}))

    def test_choose_latest_passing_checkpoint_that_moved_the_actor(self):
        ev=[{'step':100,'actor_version':50,'summary':summary()},{'step':200,'actor_version':50,'summary':summary()},
            {'step':300,'actor_version':80,'summary':summary(mse=3.1e-5)},{'step':400,'actor_version':110,'summary':summary(mse=9e-5)}]
        chosen,log=choose_checkpoint(ev,50)
        self.assertEqual(chosen['step'],300);self.assertEqual([g['step'] for g in log],[300,400]);self.assertTrue(log[1]['reasons'])
        self.assertIsNone(choose_checkpoint(ev[:2],50)[0])

    def test_value_report_uses_autonomous_group(self):
        rows=[{'groups':['autonomous'],'success':s,'q_data':q,'q_data_early':q,'q_actor':q+.1} for s,q in ((1,.9),(1,.8),(0,.2),(0,.3))]
        rows.append({'groups':['hil'],'success':True,'q_data':0.,'q_data_early':0.,'q_actor':0.})
        r=value_report({'value_episodes':rows})
        self.assertEqual((r['n_success'],r['n_failure']),(2,2));self.assertEqual(r['q_data']['auc'],1.);self.assertAlmostEqual(r['q_actor_minus_q_data'],.1)

    def test_discounted_success_target(self):
        frame=np.array([0,10,20,30]);duration=np.array([10,10,10,16])
        np.testing.assert_array_equal(remaining_frames(frame,duration),[46,36,26,16])
        out=discounted_success([1,1,0,1],frame,duration,.99)
        np.testing.assert_allclose(out,[.99**46,.99**36,0.,.99**16],rtol=1e-6)
        self.assertTrue(np.all(np.diff(out[[0,1,3]])>0))  # success returns rise toward the insertion
        with self.assertRaises(ValueError):discounted_success([1],[0],[1],1.5)

    def test_episodes_are_assigned_to_the_release_active_at_their_start(self):
        import tempfile,json,os
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,'log.jsonl')
            with open(path,'w') as f:
                for t,r in ((200,'b.json'),(100,'a.json')):f.write(json.dumps({'time':t,'release':r,'actor_updates':1})+'\n')
                f.write('{broken\n')
            log=read_switch_log(path)
        self.assertEqual([e['release'] for e in log],['a.json','b.json'])
        rows=[{'start':50},{'start':100},{'start':150},{'start':250},{}]
        self.assertEqual([r['release'] for r in assign_releases(rows,log)],[None,'a.json','a.json','b.json',None])
        rows=[dict(r,outcome=o,hil=False) for r,o in zip(rows,('success','success','failure','success','failure'))]
        s=arm_stats(rows,'a.json','release');self.assertEqual((s['n_auto'],s['auto_success']),(2,1))

if __name__=='__main__':unittest.main()
