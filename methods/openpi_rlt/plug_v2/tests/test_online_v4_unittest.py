import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from methods.openpi_rlt.plug_v2.online_learner import bellman_target
from methods.openpi_rlt.plug_v2 import online_cycle_v4 as cycle
from methods.openpi_rlt.plug_v2.conditioning_v2 import conditioned
from methods.openpi_rlt.plug_v2.rtc_queue import CommandFilter
R=Path('/media/agilex/Getea1/jiaan/projects/rlt');RUN=R/'runs/plug_v2'
class A:
 def __call__(self,z,c,ref):return ref
class Q:
 def __call__(self,z,c,a,mask):return a[:,0,7],a[:,0,7]
class Tests(unittest.TestCase):
 def batch(self):
  b={'z':torch.zeros(3,2048),'next_z':torch.zeros(3,2048),'next_context':torch.zeros(3,99),
     'next_ref':torch.zeros(3,10,14),'next_action':torch.ones(3,10,14)*.2,
     'next_hil':torch.tensor([True,False,True]),'next_future_mask':torch.ones(3,10,dtype=torch.bool),
     'reward':torch.zeros(3,10),'bootstrap':torch.tensor([True,True,False]),'duration':torch.ones(3)*10}
  b['reward'][2,9]=1
  return b
 def test_hil_uses_factual_action_and_terminal_no_bootstrap(self):
  t=bellman_target(A(),Q(),self.batch())
  self.assertAlmostEqual(float(t[0]),.2*.99**10,places=6)
  self.assertEqual(float(t[1]),0)
  self.assertAlmostEqual(float(t[2]),.99**9,places=6)
 def test_disconnected_pause_cannot_propagate(self):
  b=self.batch();b['bootstrap'][:]=False;b['reward'][:]=0
  self.assertTrue(torch.equal(bellman_target(A(),Q(),b),torch.zeros(3)))
 def test_conditioner_exact_including_d6(self):
  rng=np.random.default_rng(7)
  for d in (0,6):
   for _ in range(30):
    state=np.zeros(14,np.float32);state[7:13]=[.5,1.4,-1.2,0,1.,-.8]
    filt=CommandFilter(velocity=.1,acceleration=.9)
    first=np.broadcast_to(state,(50,14)).copy()+rng.normal(0,.05,(50,14)).astype(np.float32)
    pre=filt.plan(first,state,np.zeros((50,14),np.float32),0)
    raw=np.broadcast_to(state,(50,14)).copy()+rng.normal(0,.1,(50,14)).astype(np.float32)
    c=np.r_[state,(pre[:6]-state).reshape(-1) if d else np.zeros(84),d/6].astype(np.float32)
    got=conditioned(torch.tensor(raw[d:d+10])[None],torch.tensor(c)[None]).numpy()[0]
    expect=filt.plan(raw,state,pre,d)[d:d+10]
    np.testing.assert_allclose(got,expect,atol=2e-7)
 def run_cycle_case(self,outcome,initial_report=None):
  (RUN/'tests').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=RUN/'tests') as root:
   root=Path(root);receipt=root/'report.json';receipt.write_text(json.dumps(initial_report or {}))
   def work(ids):
    if outcome=='rejected':raise RuntimeError('gate rejection')
    return {'phase':'accepted','actor_version':600}
   with patch.object(cycle,'RUN',root),patch.object(cycle,'selected',return_value=(None,{'report':str(receipt)})),patch.object(cycle,'session_phase',return_value='waiting_scene'),patch.object(cycle,'available',return_value={str(i):None for i in range(5)}),patch.object(cycle,'work',side_effect=work) as mocked,patch.object(cycle.time,'sleep',side_effect=StopIteration):
    with self.assertRaises(StopIteration):cycle.cycle(5)
    d=json.loads((root/'learning/v4/cycle.json').read_text())
    return d,mocked.call_count
 def test_accepted_batch_consumed_once(self):
  d,n=self.run_cycle_case('accepted');self.assertEqual(n,1);self.assertEqual(len(d['consumed_uuids']),5);self.assertFalse(d['rejected_uuids'])
 def test_rejection_retained_without_busy_retry(self):
  d,n=self.run_cycle_case('rejected');self.assertEqual(n,1);self.assertFalse(d['consumed_uuids']);self.assertEqual(len(d['rejected_uuids']),5)
 def test_publish_crash_receipt_deduplicates(self):
  d,n=self.run_cycle_case('accepted',{'new_uuids':[str(i) for i in range(5)]});self.assertEqual(n,0)
 def test_supported_target_uses_only_state_value(self):
  from methods.openpi_rlt.plug_v2.supported_learner import data_target
  b=self.batch()
  calls=[]
  def value(z,c):
   calls.append((tuple(z.shape),tuple(c.shape)))
   return torch.full((len(z),),.4)
  target=data_target(value,b)
  self.assertEqual(len(calls),1)
  self.assertAlmostEqual(float(target[0]),.4*.99**10,places=6)
  self.assertAlmostEqual(float(target[2]),.99**9,places=6)
  self.assertFalse(target.requires_grad)
 def test_supported_update_never_queries_unobserved_Q_actions(self):
  from methods.openpi_rlt.plug_v2.supported_learner import SupportedCritic,update
  from methods.openpi_rlt.plug_v2.learning import Actor
  import copy
  torch.set_num_threads(2)
  b=self.batch();b.update(context=b['next_context'].clone(),ref=b['next_ref'].clone(),
   action=b['next_action'].clone(),future_mask=b['next_future_mask'].clone(),
   hil=torch.tensor([True,True,False]),expert=torch.tensor([True,False,False]),
   bc_mask=torch.ones(3,10,dtype=torch.bool))
  actor=Actor();critic=SupportedCritic();ta=copy.deepcopy(actor);tq=copy.deepcopy(critic)
  calls=[]
  def check(module,args):calls.append(torch.equal(args[2],b['action']))
  for q in (critic.q1,critic.q2,tq.q1,tq.q2):q.register_forward_pre_hook(check)
  class Aug:
   def sample(self,n,device):
    return {'z':b['z'],'c':b['context'],'ref':b['ref'],'target':b['action'],'mask':b['bc_mask']}
  metrics=update(actor,critic,ta,tq,b,torch.optim.Adam(actor.parameters()),torch.optim.Adam(critic.parameters()),2,Aug())
  self.assertTrue(calls);self.assertTrue(all(calls))
  self.assertLessEqual(metrics['advantage_weight_max'],5.00001)
 def test_real_release_publish_select_and_rejection_preserves_pointer(self):
  import shutil
  from methods.openpi_rlt.plug_v2 import online_release
  source=RUN/'learning/supported-initial-20260919'
  (RUN/'tests').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=RUN/'tests') as root:
   root=Path(root);dest=root/'learning/candidate';dest.mkdir(parents=True)
   shutil.copy2(source/'actor.pt',dest/'actor.pt');shutil.copy2(source/'report.json',dest/'report.json')
   report=json.loads((dest/'report.json').read_text())
   with patch.object(cycle,'RUN',root),patch.object(online_release,'RUN',root):
    cycle.publish(dest/'actor.pt',report)
    path,manifest=online_release.selected()
    self.assertEqual(path,dest/'actor.pt');self.assertEqual(manifest['actor_version'],500)
    before=(root/'learning/v4/current.json').read_bytes()
    bad=dest/'bad.pt';torch.save({'status':'rejected'},bad)
    with self.assertRaises(ValueError):cycle.publish(bad,report)
    self.assertEqual(before,(root/'learning/v4/current.json').read_bytes())
 def test_rejected_checkpoint_never_published(self):
  (RUN/'tests').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=RUN/'tests') as root:
   root=Path(root);p=root/'rejected.pt';torch.save({'status':'rejected'},p)
   with patch.object(cycle,'RUN',root):
    with self.assertRaises(ValueError):cycle.publish(p,{'accepted':True,'selected_step':123})
   self.assertFalse((root/'learning/v4/current.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
