import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
import torch
from torch import nn
from methods.openpi_rlt.plug_v2 import replay,learning
from methods.openpi_rlt.plug_v2.replay_contract import teacher_rows,policy_rows,pack_rows,assign_splits,validate_arrays
from methods.openpi_rlt.plug_v2.rtc_queue import CommandFilter
AUDIT=learning.ROOT/'runs/plug_v2/audits/warmup-repair-20260918'
torch.set_num_threads(2)

def fill(rows):
    for r in rows:
        r.setdefault('z',np.zeros(2048,np.float32));r.setdefault('context',np.zeros(99,np.float32))
        r.setdefault('ref',np.zeros((10,14),np.float32))
    return pack_rows(rows)

def factual(n=30):
    commands=np.zeros((n,14),np.float32);commands[:,7]=np.arange(n)*.001
    frames=[dict(valid_for_training=True,phase='rollout',mode='policy',command=c.tolist(),
                 tick=i+1,generation=1,ros_timestamp=i/30,state=np.zeros(14).tolist()) for i,c in enumerate(commands)]
    plans=[]
    for sequence,t in enumerate(range(0,n,10),1):
        prefix=np.zeros((6,14),np.float32);d=6
        if t+6>n:continue
        prefix[:]=commands[t:t+6]
        plans.append(dict(generation=1,tick=t,sequence=sequence,prefix_length=d,state=np.zeros(14).tolist(),
            prefix=prefix.tolist(),context=np.r_[np.zeros(14),prefix.reshape(-1),1].tolist(),
            ref=np.zeros((10,14)).tolist(),z=np.full(2048,sequence).tolist()))
    return frames,plans,commands

class TemporalTests(unittest.TestCase):
    def test_pause_and_same_generation_gap_do_not_bootstrap_or_reward(self):
        valid=np.ones(32,bool);valid[12:17]=False
        rows=teacher_rows(np.zeros((32,14)),valid,np.zeros(32),terminal_frame=31,terminal_success=True,bc_allowed=True)
        a=fill(rows);validate_arrays(a)
        self.assertFalse(a['bootstrap'][a['frame']<12][-5:].any())
        self.assertFalse(a['reward'][a['truncated']].any())
        self.assertTrue(a['reward'][a['done']].sum()>0)

    def test_generation_boundary_is_not_fake_failure_terminal(self):
        g=np.r_[np.zeros(12),np.ones(18)]
        a=fill(teacher_rows(np.zeros((30,14)),np.ones(30,bool),g,terminal_frame=29,terminal_success=True,bc_allowed=True))
        self.assertFalse(a['done'][a['generation']==0].any())
        self.assertFalse(a['bootstrap'][(a['generation']==0)&(a['frame']+a['duration']>=12)].any())

    def test_successor_is_ten_steps_not_next_stride_two_row(self):
        a=fill(teacher_rows(np.zeros((30,14)),np.ones(30,bool),np.zeros(30),terminal_frame=29,terminal_success=False,bc_allowed=True))
        self.assertEqual(a['frame'][a['next_row'][0]],10)

    def test_factual_delay_masks_only_executed_tail_and_reward_uses_full_time(self):
        f,p,c=factual(19)
        a=fill(policy_rows(f,p,terminal_success=True,allow_bc=False))
        self.assertEqual(a['future_mask'].sum(axis=1).tolist(),[10,3])
        np.testing.assert_array_equal(a['action'][0],c[6:16])
        self.assertEqual(a['duration'].tolist(),[10,9])
        self.assertEqual(np.where(a['reward'][1])[0].tolist(),[8])
        self.assertEqual(a['next_row'][0],1)
        self.assertFalse(a['bc_mask'].any())

    def test_committed_tail_is_present_in_next_decision_prefix(self):
        f,p,c=factual(30)
        a=fill(policy_rows(f,p,terminal_success=False,allow_bc=False))
        np.testing.assert_array_equal(a['action'][0,4:10],a['context'][1,14:98].reshape(6,14))
        self.assertEqual(a['duration'][0],10)
        self.assertEqual(a['future_mask'][0].sum(),10)

    def test_zero_delay_has_ten_executed_actions(self):
        f,p,c=factual(20);p[0].update(prefix_length=0,context=np.zeros(99).tolist())
        a=fill(policy_rows(f,p,terminal_success=False,allow_bc=True))
        self.assertEqual(a['future_mask'][0].sum(),10)
        np.testing.assert_array_equal(a['action'][0],c[:10])

    def test_duplicate_request_tick_uses_last_accepted_factual_plan(self):
        f,p,c=factual(20);old=copy.deepcopy(p[0]);old['sequence']=0;old['z']=np.full(2048,-99).tolist()
        a=fill(policy_rows(f,[old]+p,terminal_success=False,allow_bc=False))
        self.assertEqual(a['z'][0,0],1)
        self.assertEqual(len(a['z']),2)

    def test_locked_prefix_mismatch_is_rejected(self):
        f,p,c=factual();p[0]['prefix'][0][7]+=1.
        # Keep context internally consistent so this specifically tests actually executed commands.
        p[0]['context']=np.r_[np.zeros(14),np.asarray(p[0]['prefix']).reshape(-1),1].tolist()
        with self.assertRaises(AssertionError):policy_rows(f,p,terminal_success=False,allow_bc=False)

    def test_missed_control_tick_is_truncation(self):
        f,p,c=factual();f[15]['valid_for_training']=False
        a=fill(policy_rows(f,p,terminal_success=False,allow_bc=False))
        i=np.where(a['decision_tick']==10)[0][0]
        self.assertFalse(a['bootstrap'][i]);self.assertFalse(a['done'][i]);self.assertTrue(a['truncated'][i])

    def test_bad_arrays_cannot_silently_cross_generations(self):
        a=fill(teacher_rows(np.zeros((30,14)),np.ones(30,bool),np.zeros(30),terminal_frame=29,terminal_success=False,bc_allowed=True))
        a['generation'][a['next_row'][0]]=42
        with self.assertRaises(ValueError):validate_arrays(a)

    def test_full_uuid_split_is_fixed_and_twenty_percent_initially(self):
        items=[('s'+str(i),'success') for i in range(55)]+[('f'+str(i),'failure') for i in range(20)]
        split=assign_splits(items,{})
        self.assertEqual(sum(split[u]=='val' for u,g in items if g=='success'),11)
        self.assertEqual(sum(split[u]=='val' for u,g in items if g=='failure'),4)
        newer=assign_splits(items+[('new-s','success'),('new-f','failure')],split)
        self.assertTrue(all(newer[u]==v for u,v in split.items()))

    def test_pre_bad_frame_expert_fragment_has_no_success_reward(self):
        a=fill(teacher_rows(np.zeros((8,14)),np.ones(8,bool),np.zeros(8),terminal_frame=-1,terminal_success=True,bc_allowed=True))
        self.assertFalse(a['done'].any());self.assertFalse(a['reward'].any())

class LearnerTests(unittest.TestCase):
    def batch(self):
        z=torch.zeros(4,2048);context=torch.zeros(4,99);ref=torch.zeros(4,10,14)
        return dict(z=z,context=context,ref=ref,action=ref.clone(),next_z=z.clone(),next_context=context.clone(),next_ref=ref.clone(),
            future_mask=torch.ones(4,10,dtype=torch.bool),next_future_mask=torch.ones(4,10,dtype=torch.bool),
            bc_mask=torch.zeros(4,10,dtype=torch.bool),reward=torch.zeros(4,10),duration=torch.full((4,),10.),
            done=torch.zeros(4,dtype=torch.bool),bootstrap=torch.zeros(4,dtype=torch.bool),successful=torch.ones(4,dtype=torch.bool))

    def test_hil_success_policy_actions_are_not_positive_bc(self):
        b=self.batch();actor=learning.Actor();critic=learning.Twin()
        b['action'][:]=100.
        self.assertEqual(learning.evaluate(actor,critic,b)['heldout_bc'],0.)

    def test_bootstrap_false_ignores_arbitrarily_changed_next_state(self):
        b=self.batch();actor=learning.Actor();critic=learning.Twin()
        x=learning.evaluate(actor,critic,b)['critic_td']
        b['next_z'][:]=10000.;b['next_context'][:]=1000.
        y=learning.evaluate(actor,critic,b)['critic_td']
        self.assertAlmostEqual(x,y,places=7)

    def test_conditioner_matches_runtime_both_delays(self):
        for d in (0,6):
            state=np.zeros(14,np.float32);prefix=np.zeros((50,14),np.float32)
            prefix[:d,7:13]=np.arange(d)[:,None]*np.asarray([1,1,-1,1,1,1])*.001
            context=np.r_[state,(prefix[:6]-state).reshape(-1) if d else np.zeros(84),d/6].astype(np.float32)
            raw=np.zeros((50,14),np.float32);raw[:,7:13]=[.03,.03,-.03,.03,.03,.03]
            actual=CommandFilter().plan(raw,state,prefix,d)[d:d+10]
            other=learning.conditioned(torch.tensor(raw[d:d+10])[None],torch.tensor(context)[None])[0].numpy()
            np.testing.assert_allclose(actual,other,atol=1e-7,rtol=1e-6)

    def test_critic_and_evaluation_ignore_padding(self):
        b=self.batch();b['future_mask'][:,4:]=False;b['bc_mask'][:,:4]=True
        actor=learning.Actor();critic=learning.Twin();before=learning.evaluate(actor,critic,b)
        b['action'][:,4:]=1000.;b['ref'][:,4:]=1000.
        after=learning.evaluate(actor,critic,b)
        self.assertAlmostEqual(before['heldout_bc'],after['heldout_bc'],places=7)
        self.assertAlmostEqual(before['critic_td'],after['critic_td'],places=7)
        self.assertAlmostEqual(before['smooth'],after['smooth'],places=7)

    def test_old_temporal_cache_refused(self):
        with tempfile.TemporaryDirectory(dir=AUDIT) as tmp:
            np.savez_compressed(Path(tmp)/'old.npz',metadata=np.array(json.dumps(dict(cohort='plug_v2',replay_version=1))))
            with self.assertRaisesRegex(ValueError,'obsolete'):learning.Sampler(tmp)

    def test_two_synthetic_cpu_optimizer_steps_obey_actor_ratio_and_masks(self):
        b=self.batch();b['action'][:]=100.;actor=learning.Actor();critic=learning.Twin()
        ta=copy.deepcopy(actor);tc=copy.deepcopy(critic)
        ao=torch.optim.Adam(actor.parameters(),lr=1e-4);co=torch.optim.Adam(critic.parameters(),lr=1e-4)
        before=[p.clone().detach() for p in actor.parameters()]
        first=learning.update(actor,critic,ta,tc,b,ao,co,1)
        self.assertTrue(all(torch.equal(x,p) for x,p in zip(before,actor.parameters())))
        second=learning.update(actor,critic,ta,tc,b,ao,co,2)
        self.assertEqual(second['imitation'],0.)
        self.assertTrue(all(np.isfinite(v) for v in second.values()))

class PreparationTests(unittest.TestCase):
    def test_teacher_features_have_no_future_prefix_and_cache_is_immutable(self):
        import pyarrow as pa,pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(dir=AUDIT) as tmp:
            tmp=Path(tmp);root=tmp/'data';folder=root/'data/chunk-000';folder.mkdir(parents=True)
            state=np.zeros((12,14),np.float32);action=state.copy();action[:,7]=np.arange(12)*.001
            parquet=folder/'episode_000000.parquet'
            pq.write_table(pa.table({'observation.state':state.tolist(),'action':action.tolist()}),parquet)
            calls=[]
            class RPC:
                def get_server_metadata(self):return {'checkpoint':'fixture-3999'}
                def infer(self,obs):
                    self_outer.assertEqual(obs['prefix_length'],0)
                    self_outer.assertFalse(obs['action_prefix'].any())
                    calls.append(obs)
                    return dict(prefix_length=0,z_rl=np.zeros(2048,np.float32),ref_chunk=np.zeros((50,14),np.float32))
            self_outer=self
            class Video:
                streams=NS(video=[None])
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def decode(self,*args):return [NS(to_ndarray=lambda **kwargs:np.zeros((2,2,3),np.uint8)) for _ in range(12)]
            with patch.object(replay,'RUN',tmp/'run'),patch('av.open',lambda *args:Video()):
                kw=dict(uuid='fixture',group='expert',split='train',model_rpc=RPC())
                target=replay.prepare_episode(root,**kw);self.assertEqual(len(calls),6)
                with np.load(target) as a:self.assertEqual(json.loads(str(a['metadata']))['replay_version'],2)
                replay.prepare_episode(root,**kw);self.assertEqual(len(calls),6)
                old=target.read_bytes();action[:,7]+=1.
                pq.write_table(pa.table({'observation.state':state.tolist(),'action':action.tolist()}),parquet)
                with self.assertRaisesRegex(ValueError,'source changed'):replay.prepare_episode(root,**kw)
                self.assertEqual(old,target.read_bytes())

if __name__=='__main__':unittest.main()
