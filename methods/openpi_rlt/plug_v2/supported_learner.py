"""Conservative IQL-style residual learner; not an exact RLT reproduction. No robot imports.
Only factual actions enter Q training; state-value bootstrap and advantage-weighted BC.
Verified temporal/HIL links only; no actor maximization over unseen Q actions.
"""
import argparse,copy,json,random,time,hashlib,os
from pathlib import Path
import numpy as np,torch
from .learning import Actor,Twin,Q,JOINTS,EXPECTED_STATS,atomic_torch
from .conditioning_v2 import conditioned
from .online_contract import ContractData
from .training_contract import Augmentation,masked_mse,smooth_acceleration
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'
CONTRACT='plug-v2-online-v4-v010-a090'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class SupportedCritic(Twin):
    def __init__(self):
        super().__init__();self.value_net=Q()
    def value(self,z,c):
        return self.value_net(z,c,c[:,None,:14].expand(-1,10,-1))
def data_target(value,b):
    with torch.no_grad():
        return (b['reward']*(.99**torch.arange(10,device=b['z'].device))).sum(-1)+b['bootstrap']*(.99**b['duration'])*value(b['next_z'],b['next_context'])
def weighted_mse(pred,target,mask,weight):
    error=(pred[...,JOINTS]-target[...,JOINTS]).square().mean(-1)/(.05**2)
    w=mask*weight[:,None]
    return (error*w).sum()/w.sum().clamp(min=1)
def update(actor,critic,ta,tq,b,ao,qo,step,augmentation):
    with torch.no_grad():
        factual_q=torch.minimum(*tq(b['z'],b['context'],b['action'],b['future_mask']))
    value=critic.value(b['z'],b['context']);diff=factual_q-value
    value_loss=(torch.where(diff>0,.7,.3)*diff.square()).mean()
    target=data_target(critic.value,b)
    q1,q2=critic(b['z'],b['context'],b['action'],b['future_mask'])
    qloss=((q1-target).square()+(q2-target).square()).mean()
    qo.zero_grad();(qloss+value_loss).backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);qo.step()
    metrics={'critic_loss':float(qloss.detach()),'value_loss':float(value_loss.detach()),
             'human_mask_ratio':float((b['hil']&~b['expert']).float().mean())}
    if step%2==0:
        with torch.no_grad():
            factual_q=torch.minimum(*critic(b['z'],b['context'],b['action'],b['future_mask']))
            advantage=factual_q-critic.value(b['z'],b['context'])
            weight=torch.exp((advantage/.2).clamp(-5,1.609438)).detach()
        raw=actor(b['z'],b['context'],b['ref'])
        pred=conditioned(raw,b['context']);reference=conditioned(b['ref'],b['context'])
        human=b['hil'][:,None]&b['bc_mask'];policy=(~b['hil'])[:,None]&b['bc_mask']
        other=(~b['hil'])[:,None]&~b['bc_mask']&b['future_mask']
        hb=weighted_mse(raw,b['action'],human,weight)
        pb=weighted_mse(pred,b['action'],policy,weight)
        rb=masked_mse(pred,reference,other)
        extra=augmentation.sample(64,'cpu')
        eb=masked_mse(actor(extra['z'],extra['c'],extra['ref']),extra['target'],extra['mask'])
        smooth=smooth_acceleration(pred,b['context'],b['future_mask'])
        loss=10*(hb+pb+eb)+rb+smooth
        ao.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);ao.step()
        metrics.update(actor_loss=float(loss.detach()),bc_human_penalty=float(hb.detach()),
                       bc_reference_penalty=float(rb.detach()),bc_policy_penalty=float(pb.detach()),
                       causal_bc=float(eb.detach()),smooth=float(smooth.detach()),
                       advantage_weight_mean=float(weight.mean()),advantage_weight_max=float(weight.max()))
    with torch.no_grad():
        for target,source in ((ta,actor),(tq,critic)):
            for p,q in zip(target.parameters(),source.parameters()):p.lerp_(q,.005)
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('nonfinite update')
    return metrics

def evaluation(actor,critic,b):
    with torch.no_grad():
        raw=actor(b['z'],b['context'],b['ref']);pred=conditioned(raw,b['context']);ref=conditioned(b['ref'],b['context'])
        mask=b['bc_mask'];human=b['hil'][:,None]&mask
        factual_bc=masked_mse(pred,b['action'],mask)
        raw_bc=masked_mse(raw,b['action'],human)
        smooth=smooth_acceleration(pred,b['context'],b['future_mask'])
        target=data_target(critic.value,b)
        q1,q2=critic(b['z'],b['context'],b['action'],b['future_mask'])
        td=((q1-target).square()+(q2-target).square()).mean()
        return {'factual_bc':float(factual_bc),'human_raw_bc':float(raw_bc),'smooth':float(smooth),
                'critic_td':float(td),'max_raw_residual':float((raw-b['ref']).abs().max()),
                'max_command_change_from_reference':float((pred-ref).abs().max())}
def train(command,steps,out,previous=None):
    out=Path(out).resolve();out.relative_to((RUN/'learning').resolve());out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    data=ContractData(out);augmentation=Augmentation()
    actor=Actor();critic=SupportedCritic();ao=torch.optim.Adam(actor.parameters(),lr=3e-5);qo=torch.optim.Adam(critic.parameters(),lr=1e-4)
    start=0
    if previous:
        saved=torch.load(previous,map_location='cpu',weights_only=False)
        if saved.get('execution_contract')!=CONTRACT or saved.get('learning_algorithm')!='supported_iql_residual_v1':raise ValueError('online base contract mismatch')
        actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic'])
        ao.load_state_dict(saved['actor_optimizer']);qo.load_state_dict(saved['critic_optimizer']);start=saved['global_step']
    ta=copy.deepcopy(actor);tq=copy.deepcopy(critic)
    if previous:
        ta.load_state_dict(saved['target_actor']);tq.load_state_dict(saved['target_critic'])
    validation=data.sample(1024,'cpu',split='val')
    baseline=evaluation(actor,critic,validation);best=None;best_score=float('inf');records=[]
    t=time.perf_counter()
    for local in range(1,steps+1):
        metrics=update(actor,critic,ta,tq,data.sample(128,'cpu'),ao,qo,start+local,augmentation)
        if local%100==0:
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':start+local,**metrics})+'\n')
        if local in (100,500,1000,2000,5000,steps):
            ev=evaluation(actor,critic,validation)
            # Gate numerical correctness, bounded residual, BC, smoothness and TD.
            passed=(all(np.isfinite(v) for v in ev.values()) and ev['max_raw_residual']<=.050001
                    and ev['factual_bc']<=baseline['factual_bc']*1.03+1e-6
                    and ev['human_raw_bc']<=baseline['human_raw_bc']*1.03+1e-6
                    and ev['smooth']<=baseline['smooth']*1.10+1e-6
                    and ev['critic_td']<=baseline['critic_td']*1.10+1e-6)
            score=ev['human_raw_bc']+ev['factual_bc']+ev['smooth']
            if previous:
                baseline_score=baseline['human_raw_bc']+baseline['factual_bc']+baseline['smooth']
                passed=passed and score<baseline_score*.999

            rec={'step':start+local,'passed':passed,'metrics':ev,'seconds':time.perf_counter()-t}
            records.append(rec);print('V4_VALIDATION',json.dumps(rec),flush=True)
            state={'cohort':'plug_v2','status':'accepted' if passed else 'rejected',
                   'global_step':start+local,'rtc_context_dim':99,'stats_sha256':EXPECTED_STATS,
                   'execution_contract':CONTRACT,'learning_algorithm':'supported_iql_residual_v1','replay_version':4,'stage1_checkpoint':data.model_checkpoint,
                   'actor':copy.deepcopy(actor.state_dict()),'critic':copy.deepcopy(critic.state_dict()),
                   'target_actor':copy.deepcopy(ta.state_dict()),'target_critic':copy.deepcopy(tq.state_dict()),
                   'actor_optimizer':copy.deepcopy(ao.state_dict()),'critic_optimizer':copy.deepcopy(qo.state_dict()),
                   'baseline':baseline,'evaluation':ev,'autonomous_improvement_verified':False}
            atomic_torch(out/'checkpoints'/('step_%d.pt'%(start+local)),state)
            if passed and score<best_score:best_score=score;best=state
    report={'baseline':baseline,'checkpoints':records,'accepted':best is not None,
            'selected_step':None if best is None else best['global_step'],'seconds':time.perf_counter()-t,
            'source_uuids':sorted({d['metadata']['source_uuid'] for d in data.items}),
            'train_uuid':sorted({d['metadata']['source_uuid'] for d in data.items if d['metadata']['split']=='train'}),
            'val_uuid':sorted({d['metadata']['source_uuid'] for d in data.items if d['metadata']['split']=='val'}),
            'sampling':'expert30/HIL20/assisted-or-autonomous-success-policy25/failure-policy25',
            'published':False}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if best is None:raise RuntimeError('No candidate passed; previous model retained')
    atomic_torch(out/'actor.pt',best)
    return out/'actor.pt',report
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=5000);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path);a=p.parse_args()
    if not 1<=a.steps<=20000:raise ValueError('invalid steps')
    import fcntl
    from .training_flow import session_phase
    with (RUN/'learning/operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if session_phase() not in ('stopped','offline','disarmed','ready','waiting_scene'):
            raise RuntimeError('end/pause completed Session before offline training')
        train('online' if a.previous else 'warmup',a.steps,a.output,a.previous)
if __name__=='__main__':main()
