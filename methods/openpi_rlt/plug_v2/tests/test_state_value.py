import unittest
import numpy as np
from methods.openpi_rlt.plug_v2 import rtc_upstream_core  # adds the pinned upstream src path
from methods.openpi_rlt.plug_v2.state_value import inner_split,class_weights,fit_value,predict,advantages
from methods.openpi_rlt.plug_v2.awbc_train import attach_value_advantages

def synthetic(n_episodes=40,rows=12,dim=6,seed=0):
    """Episodes whose success is readable from feature 0; features drift toward the outcome over time."""
    rng=np.random.default_rng(seed);x=[];y=[];g=[]
    for e in range(n_episodes):
        s=e%2==0
        for t in range(rows):
            v=rng.normal(size=dim);v[0]+=(1. if s else -1.)*(.5+t/rows);x.append(v);y.append(float(s));g.append(f'ep{e}')
    return np.array(x,np.float32),np.array(y,np.float32),np.array(g)

class StateValueTest(unittest.TestCase):
    def test_inner_split_is_episode_level_deterministic_and_nonempty(self):
        g=np.repeat([f'u{i}' for i in range(30)],4);a=inner_split(g);self.assertTrue(np.array_equal(a,inner_split(g)))
        for u in set(g):self.assertEqual(len(set(a[g==u])),1)
        self.assertTrue(a.any() and not a.all())
        one=inner_split(np.array(['x','x','y']));self.assertTrue(one.any() and not one.all())

    def test_class_weights_balance_outcomes(self):
        y=np.array([1,1,1,0]);w=class_weights(y,y.mean())
        self.assertAlmostEqual(float(w[y==1].sum()),float(w[y==0].sum()))

    def test_fit_predicts_outcome_on_unseen_episodes(self):
        x,y,g=synthetic();model,report=fit_value(x,y,g,epochs=30)
        xt,yt,_=synthetic(n_episodes=20,seed=1);p=predict(model,xt)
        self.assertEqual(p.shape,(len(xt),));self.assertTrue(np.all((p>0)&(p<1)))
        self.assertGreater(p[yt==1].mean()-p[yt==0].mean(),.3)
        self.assertLessEqual(report['best_epoch'],report['epochs_run']-1);self.assertGreater(report['inner_val_episodes'],0)

    def test_advantage_is_value_change_or_terminal_outcome(self):
        a=advantages([.2,.5,.6],[.4,.5,.9],[False,False,True],[1.,1.,0.])
        np.testing.assert_allclose(a,[.2,0.,-.6],atol=1e-6)

    def test_attach_writes_one_advantage_per_replay_row_and_scores_validation(self):
        rng=np.random.default_rng(3);ordered=[];episodes=[];rows=[];value_rows=[]
        for e in range(24):
            s=e%2==0;n=6;z=rng.normal(size=(n,8)).astype(np.float32);z[:,0]+=1. if s else -1.
            d=dict(z_rl=z,proprio=rng.normal(size=(n,7)).astype(np.float32),next_z_rl=z,next_proprio=rng.normal(size=(n,7)).astype(np.float32),
                   source_chunk=np.zeros((n,10),np.uint8),frame=np.arange(n)*10,done=np.arange(n)==n-1)
            m=dict(uuid=f'e{e}',split='val' if e>=18 else 'train',expert=False,phase='online',success=s);episodes.append((m,d))
            if m['split']=='train':
                eid=len(ordered);ordered.append((m,d))
                for i in range(n):rows.append({});value_rows.append((i,eid,bool(d['done'][i]),float(s),m['uuid']))
        report=attach_value_advantages(rows,value_rows,ordered,episodes)
        self.assertTrue(all('awbc_adv' in r and np.isfinite(r['awbc_adv']) for r in rows))
        self.assertEqual(report['heldout_autonomous']['episodes'],6);self.assertEqual(report['heldout_autonomous']['success'],3)
        self.assertGreater(report['heldout_autonomous']['auc_mean'],.8)
        with self.assertRaises(RuntimeError):attach_value_advantages(rows[:-1],value_rows,ordered,episodes)

    def test_effort_variant_excludes_demonstrations_and_uses_effort_features(self):
        rng=np.random.default_rng(5);ordered=[];episodes=[];rows=[];value_rows=[];calls=[]
        for e in range(26):
            expert=e<2;s=expert or e%2==0;n=6
            d=dict(z_rl=rng.normal(size=(n,8)).astype(np.float32),proprio=rng.normal(size=(n,7)).astype(np.float32),next_proprio=rng.normal(size=(n,7)).astype(np.float32),
                   source_chunk=np.full((n,10),2 if expert else 0,np.uint8),frame=np.arange(n)*10,next_frame=np.arange(n)*10+10,done=np.arange(n)==n-1)
            d['next_z_rl']=d['z_rl']
            m=dict(uuid=f'e{e}',split='val' if e>=20 else 'train',expert=expert,phase='demonstrations' if expert else 'online',success=s);episodes.append((m,d))
            if m['split']=='train':
                eid=len(ordered);ordered.append((m,d))
                for i in range(n):rows.append({});value_rows.append((i,eid,bool(d['done'][i]),float(s),m['uuid']))
        def fake_effort(m,d):
            # Outcome is readable only from the effort channel: failures saturate.
            calls.append(m['uuid']);assert not m['expert'];v=np.full((len(d['frame']),2),.2 if m['success'] else 1.,np.float32);return v,v
        report=attach_value_advantages(rows,value_rows,ordered,episodes,effort=True,effort_fn=fake_effort)
        self.assertTrue(report['fit_excludes_demonstrations']);self.assertEqual(report['inputs'],'z_rl+proprio+gripper_effort')
        self.assertEqual(report['fit_rows_total'],len(rows)-12)
        self.assertTrue(all(rows[k]['awbc_adv']==0 for k,(_,e,_,_,_) in enumerate(value_rows) if ordered[e][0]['expert']))
        self.assertGreater(report['heldout_autonomous']['auc_mean'],.9)
        self.assertNotIn('e0',calls)

if __name__=='__main__':unittest.main()
