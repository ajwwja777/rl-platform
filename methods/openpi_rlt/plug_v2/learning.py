# Frozen Stage-1 features; tiny actor/twin-critic only. No ROS/CAN imports.
import argparse,copy,hashlib,json,os,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/plug_v2'
JOINTS=list(range(7,13))
EXPECTED_STATS='373d9a01bbcc0907dbbc27b091768774f98933cb28cdf3404a14f378ee02b49d'
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.z=nn.Sequential(nn.Linear(2048,256),nn.LayerNorm(256))
        self.context=nn.Sequential(nn.Linear(99,64),nn.LayerNorm(64),nn.Tanh())
        self.ref=nn.Sequential(nn.Linear(60,256),nn.LayerNorm(256),nn.Tanh())
        self.trunk=nn.Sequential(nn.Linear(576,256),nn.LayerNorm(256),nn.GELU(),
                                nn.Linear(256,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,60))
        nn.init.zeros_(self.trunk[-1].weight);nn.init.zeros_(self.trunk[-1].bias)
    def forward(self,z,context,ref,dropout=0.):
        r=ref[...,JOINTS]-context[:,:14,None].transpose(1,2)[...,JOINTS]
        encoded_ref=r.reshape(len(z),60)
        if dropout:
            keep=(torch.rand(len(z),1,device=z.device)>=dropout).to(z.dtype)
            encoded_ref=encoded_ref*keep
        x=torch.cat([self.z(z),self.context(context),self.ref(encoded_ref)],dim=-1)
        # Bounded residual initialization preserves the valid reference at step zero.
        correction=.05*torch.tanh(self.trunk(x).reshape(-1,10,6))
        result=ref.clone();result[...,JOINTS]=ref[...,JOINTS]+correction
        return result
    def act(self,z,context,ref):
        device=next(self.parameters()).device
        tensors=[torch.as_tensor(np.asarray(v),dtype=torch.float32,device=device)[None] for v in (z,context,ref)]
        return self(*tensors)[0].detach().cpu().numpy()
class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.z=nn.Sequential(nn.Linear(2048,256),nn.LayerNorm(256))
        self.context=nn.Sequential(nn.Linear(99,64),nn.LayerNorm(64),nn.Tanh())
        self.action=nn.Sequential(nn.Linear(60,256),nn.LayerNorm(256),nn.Tanh())
        self.trunk=nn.Sequential(nn.Linear(576,256),nn.LayerNorm(256),nn.GELU(),
                                nn.Linear(256,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,1))
    def forward(self,z,context,action,mask=None):
        delta=(action[...,JOINTS]-context[:,None,JOINTS])/.05
        if mask is not None:delta=delta*mask[...,None].to(delta.dtype)
        return self.trunk(torch.cat([self.z(z),self.context(context),
                                    self.action(delta.reshape(len(z),60))],dim=-1)).squeeze(-1)
class Twin(nn.Module):
    def __init__(self):super().__init__();self.q1=Q();self.q2=Q()
    def forward(self,*args):return self.q1(*args),self.q2(*args)
def conditioned(raw,context):
    # The differentiable counterpart of the actual 30Hz CommandFilter.
    alpha=1-np.exp(-1/(30*(-.05/np.log(.65))))
    prefix=context[:,14:98].reshape(-1,6,14)+context[:,None,:14]
    d=context[:,-1]>0
    anchor=torch.where(d[:,None],prefix[:,-1],context[:,:14])
    output=[]
    for t in range(10):
        current=context[:,:14].clone()
        current[:,JOINTS]=anchor[:,JOINTS]+torch.clamp(alpha*(raw[:,t,JOINTS]-anchor[:,JOINTS]),
                                                   -.2/30,.2/30)
        output.append(current);anchor=current
    return torch.stack(output,dim=1)
def load_actor(path):
    path=Path(path)
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('cohort')!='plug_v2' or data.get('status')!='accepted':
        raise ValueError('actor is not an accepted plug_v2 artifact')
    if data.get('rtc_context_dim')!=99 or data.get('stats_sha256')!=EXPECTED_STATS:
        raise ValueError('actor RTC/stats contract mismatch')
    manifest=ROOT/'deployments/plug_v2/manifest.json'
    if manifest.exists() and data.get('stage1_checkpoint')!=json.loads(manifest.read_text())['checkpoint']:
        raise ValueError('actor trained with a different frozen Stage-1 checkpoint')
    actor=Actor();actor.load_state_dict(data['actor'],strict=True);actor.eval()
    return actor,int(data['global_step'])
class Sampler:
    def __init__(self,folder):
        self.train={g:[] for g in ('expert','success','failure')};self.val=[];self.uuids=set()
        self.model_checkpoint=None
        for path in sorted(Path(folder).glob('*.npz')):
            if path.name.endswith('.source.npz'):continue
            with np.load(path,allow_pickle=False) as f:
                data={k:f[k].copy() for k in f.files}
            meta=json.loads(str(data.pop('metadata')))
            if meta.get('cohort')!='plug_v2':raise ValueError('mixed camera cohort')
            if meta.get('replay_version')!=2:raise ValueError('obsolete replay temporal contract; prepare v2 first')
            from methods.openpi_rlt.plug_v2.replay_contract import validate_arrays
            validate_arrays(data)
            if self.model_checkpoint is None:self.model_checkpoint=meta['model_checkpoint']
            if meta['model_checkpoint']!=self.model_checkpoint:raise ValueError('mixed Stage-1 checkpoints')
            if not len(data['z']):continue
            self.uuids.add(meta['source_uuid'])
            data['metadata']=meta
            if meta['split']=='train':self.train[meta['group']].append(data)
            else:self.val.append(data)
        train_uuid={d['metadata']['source_uuid'] for group in self.train.values() for d in group}
        val_uuid={d['metadata']['source_uuid'] for d in self.val}
        if train_uuid&val_uuid:raise ValueError('heldout UUID leakage')
        if any(not self.train[g] for g in self.train):raise ValueError('manual warmup requires fresh expert, success and failure groups')
        if not self.val:raise ValueError('heldout episodes required')
        if not {'success','failure'}.issubset({d['metadata']['group'] for d in self.val}):
            raise ValueError('fresh heldout success and failure episodes required')
    def sample(self,size,device,validation=False):
        chosen=[]
        if validation:
            names=random.choices(['expert','success','failure'],weights=[.3,.4,.3],k=size)
            groups=[[d for d in self.val if d['metadata']['group']==g] for g in names]
        else:
            names=random.choices(['expert','success','failure'],weights=[.3,.4,.3],k=size)
            groups=[self.train[g] for g in names]
        visits={}
        for name,group in zip(names,groups):
            visit=visits.get(name,0);visits[name]=visit+1
            episode=group[visit%len(group)] if validation else random.choice(group)
            i=random.randrange(len(episode['z']))
            row={k:v[i] for k,v in episode.items() if k!='metadata'}
            row['successful']=episode['metadata']['group']!='failure'
            chosen.append(row)
        keys=[k for k in chosen[0] if k!='frame']
        return {k:torch.as_tensor(np.asarray([r[k] for r in chosen]),device=device,
                    dtype=torch.bool if k in ('done','successful','hil','future_mask','next_future_mask','bc_mask','bootstrap','truncated') else torch.float32) for k in keys}
def update(actor,critic,target_actor,target_critic,batch,actor_opt,critic_opt,step):
    z,c,a,r=batch['z'],batch['context'],batch['action'],batch['reward']
    with torch.no_grad():
        na=conditioned(target_actor(batch['next_z'],batch['next_context'],batch['next_ref']),batch['next_context'])
        q1,q2=target_critic(batch['next_z'],batch['next_context'],na,batch.get('next_future_mask'))
        discounted=(r*(.99**torch.arange(10,device=r.device))).sum(dim=-1)
        bootstrap=batch.get('bootstrap',~batch['done'])
        target=discounted+bootstrap.float()*(.99**batch['duration'])*torch.minimum(q1,q2)
    q1,q2=critic(z,c,a,batch['future_mask']);critic_loss=((q1-target)**2+(q2-target)**2).mean()
    critic_opt.zero_grad();critic_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.)
    critic_opt.step()
    metrics={'critic_loss':float(critic_loss.detach()),'q':float(q1.detach().mean())}
    if step%2==0:
        for p in critic.parameters():p.requires_grad_(False)
        predicted=conditioned(actor(z,c,batch['ref'],dropout=.5),c)
        value=critic.q1(z,c,predicted,batch['future_mask'])
        executed=batch['future_mask']
        supervised=batch.get('bc_mask',executed & batch['successful'][:,None])
        mask=executed[...,None].float()
        supervised_mask=supervised[...,None].float()
        reference_conditioned=conditioned(batch['ref'],c)
        difference=(predicted[...,JOINTS]-a[...,JOINTS])/.05
        reference_difference=(predicted[...,JOINTS]-reference_conditioned[...,JOINTS])/.05
        denominator=torch.clamp(mask.sum()*6,min=1)
        imitation=(difference.square()*supervised_mask).sum()/denominator
        reference_anchor=(reference_difference.square()*(mask-supervised_mask)).sum()/denominator
        bc=imitation+reference_anchor
        residual=reference_difference
        pair=(executed[:,1:] & executed[:,:-1])[...,None].float()
        triple=(executed[:,2:] & executed[:,1:-1] & executed[:,:-2])[...,None].float()
        velocity=(torch.diff(residual,dim=1).square()*pair).sum()/torch.clamp(pair.sum()*6,min=1)
        acceleration=(torch.diff(residual,n=2,dim=1).square()*triple).sum()/torch.clamp(triple.sum()*6,min=1)
        first=executed[:,0,None].float()
        boundary=(residual[:,0].square()*first).sum()/torch.clamp(first.sum()*6,min=1)
        smooth=velocity+acceleration+boundary
        loss=10*bc-.1*value.mean()+10*smooth
        actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.)
        actor_opt.step()
        for p in critic.parameters():p.requires_grad_(True)
        metrics.update(actor_loss=float(loss.detach()),bc=float(bc.detach()),
                       imitation=float(imitation.detach()),reference_anchor=float(reference_anchor.detach()),
                       smooth=float(smooth.detach()))
    with torch.no_grad():
        for target_module,source in ((target_actor,actor),(target_critic,critic)):
            for tp,sp in zip(target_module.parameters(),source.parameters()):tp.lerp_(sp,.005)
    if not all(np.isfinite(x) for x in metrics.values()):raise RuntimeError('nonfinite RL update')
    return metrics
def evaluate(actor,critic,b):
    with torch.no_grad():
        pred=conditioned(actor(b['z'],b['context'],b['ref']),b['context'])
        reference_conditioned=conditioned(b['ref'],b['context'])
        executed=b['future_mask']
        supervised=b.get('bc_mask',executed & b['successful'][:,None])
        mask=supervised[...,None].float()
        bc=(((pred[...,JOINTS]-b['action'][...,JOINTS])**2)*mask).sum()/torch.clamp(mask.sum()*6,min=1)
        def smooth(x):
            x=x[...,JOINTS]
            pair=(executed[:,1:] & executed[:,:-1])[...,None].float()
            triple=(executed[:,2:] & executed[:,1:-1] & executed[:,:-2])[...,None].float()
            return float((torch.diff(x,dim=1).square()*pair).sum()/torch.clamp(pair.sum()*6,min=1)
                         +(torch.diff(x,n=2,dim=1).square()*triple).sum()/torch.clamp(triple.sum()*6,min=1))
        q1,q2=critic(b['z'],b['context'],b['action'],b['future_mask'])
        next_action=conditioned(actor(b['next_z'],b['next_context'],b['next_ref']),b['next_context'])
        nq1,nq2=critic(b['next_z'],b['next_context'],next_action,b.get('next_future_mask'))
        rewards=(b['reward']*(.99**torch.arange(10,device=pred.device))).sum(dim=-1)
        bootstrap=b.get('bootstrap',~b['done'])
        target=rewards+bootstrap.float()*(.99**b['duration'])*torch.minimum(nq1,nq2)
        td=((q1-target).square()+(q2-target).square()).mean()
        pair=executed[:,1:] & executed[:,:-1]
        velocity=(torch.diff(pred[...,JOINTS],dim=1).abs()*pair[...,None]).max()
        prefix=b['context'][:,14:98].reshape(-1,6,14)+b['context'][:,None,:14]
        anchor=torch.where((b['context'][:,-1]>0)[:,None],prefix[:,-1],b['context'][:,:14])
        first=((pred[:,0,JOINTS]-anchor[:,JOINTS]).abs()*executed[:,0,None]).max()
        return {'heldout_bc':float(bc),'smooth':smooth(pred),'reference_smooth':smooth(reference_conditioned),
                'critic_td':float(td),'max_step':float(torch.maximum(velocity,first)),
                'q':float(critic.q1(b['z'],b['context'],pred,b['future_mask']).mean()),
                'supervised_actions':int(supervised.sum()),'bootstrap_rows':int(bootstrap.sum())}
def atomic_torch(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('wb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def train(args):
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    data=Sampler(RUN/'replay/v2');device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.set_float32_matmul_precision('high')
    actor=Actor().to(device);critic=Twin().to(device)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=1e-4,fused=(device.type=='cuda'))
    critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-4,fused=(device.type=='cuda'))
    folder=Path(args.output_dir).resolve() if getattr(args,'output_dir',None) else RUN/'learning'/args.command
    if getattr(args,'output_dir',None):
        folder.relative_to((RUN/'learning').resolve())
        if args.command!='warmup':raise ValueError('isolated output only supported for fresh warmup')
        if (folder/'actor.pt').exists() or (folder/'validation.jsonl').exists():raise ValueError('comparison output already trained; preserve it')
    start_step=0
    if args.command=='online':
        previous=folder/'actor.pt'
        if not previous.exists():previous=RUN/'learning/warmup/actor.pt'
        saved=torch.load(previous,map_location=device,weights_only=False)
        assert saved['cohort']=='plug_v2' and saved['status']=='accepted'
        actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic'])
        actor_opt.load_state_dict(saved['actor_optimizer']);critic_opt.load_state_dict(saved['critic_optimizer'])
        start_step=saved['global_step']
    target_actor=copy.deepcopy(actor);target_critic=copy.deepcopy(critic)
    if args.command=='online':
        target_actor.load_state_dict(saved['target_actor']);target_critic.load_state_dict(saved['target_critic'])
    heldout=data.sample(512,device,validation=True)
    baseline=evaluate(actor,critic,heldout);best_score=float('inf');best=None
    duration=time.perf_counter()
    checkpoints={100,500,2000,5000,10000,20000,args.steps}
    for local in range(1,args.steps+1):
        step=start_step+local
        metrics=update(actor,critic,target_actor,target_critic,data.sample(128,device),actor_opt,critic_opt,step)
        if local%100==0:print('RL_UPDATE',step,json.dumps(metrics),flush=True)
        if local in checkpoints:
            result=evaluate(actor,critic,heldout)
            accepted=(all(np.isfinite(v) for v in result.values()) and result['max_step']<=.2/30+1e-6
                      and result['smooth']<=max(baseline['smooth']*1.3,1e-5)
                      and result['heldout_bc']<=baseline['heldout_bc']*1.1+1e-6
                      and result['critic_td']<=baseline['critic_td']*1.1+1e-6)
            score=result['heldout_bc']+10*result['smooth']
            if accepted and score<best_score:
                best_score=score
                best={'cohort':'plug_v2','status':'accepted','global_step':step,
                    'rtc_context_dim':99,'stats_sha256':EXPECTED_STATS,'replay_version':2,
                    'stage1_checkpoint':data.model_checkpoint,'actor':copy.deepcopy(actor.state_dict()),
                    'critic':copy.deepcopy(critic.state_dict()),
                    'target_actor':copy.deepcopy(target_actor.state_dict()),
                    'target_critic':copy.deepcopy(target_critic.state_dict()),
                    'actor_optimizer':copy.deepcopy(actor_opt.state_dict()),
                    'critic_optimizer':copy.deepcopy(critic_opt.state_dict()),
                    'evaluation':result,'baseline':baseline}
            if getattr(args,'save_checkpoints',False):
                candidate={'cohort':'plug_v2','status':'accepted' if accepted else 'rejected','global_step':step,
                    'rtc_context_dim':99,'stats_sha256':EXPECTED_STATS,'replay_version':2,
                    'stage1_checkpoint':data.model_checkpoint,'actor':actor.state_dict(),
                    'critic':critic.state_dict(),'target_actor':target_actor.state_dict(),
                    'target_critic':target_critic.state_dict(),'actor_optimizer':actor_opt.state_dict(),
                    'critic_optimizer':critic_opt.state_dict(),'evaluation':result,'baseline':baseline}
                path=folder/'checkpoints'/('step_%d.pt'%step)
                if path.exists():raise ValueError('checkpoint exists; preserve it')
                atomic_torch(path,candidate)
            receipt={'step':step,'accepted':accepted,'evaluation':result,
                     'seconds':time.perf_counter()-duration,'best_step':None if best is None else best['global_step']}
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'status.json').write_text(json.dumps(receipt,indent=2))
            with (folder/'validation.jsonl').open('a') as f:
                f.write(json.dumps(receipt)+'\n');f.flush();os.fsync(f.fileno())
            print('RL_VALIDATION',json.dumps(receipt),flush=True)
    if best is None:raise RuntimeError('No RL checkpoint passed heldout/continuity validation; prior actor retained')
    atomic_torch(folder/'actor.pt',best)
    (folder/'ready.json').write_text(json.dumps(
       {'cohort':'plug_v2','status':'accepted','actor_version':best['global_step'],
        'learner_final_step':start_step+args.steps,'critic_updates':args.steps,
        'actor_updates':(start_step+args.steps)//2-start_step//2,
        'elapsed_seconds':time.perf_counter()-duration,'baseline':baseline,
        'evaluation':best['evaluation'],'sample_ratio':[.3,.4,.3],'replay_version':2,
        'train_episodes':{g:len(v) for g,v in data.train.items()},
        'heldout_episodes':len(data.val)},indent=2))
    print('RL_ACTOR_READY',best['global_step'],flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['warmup','online'])
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--output-dir',default=None)
    p.add_argument('--save-checkpoints',action='store_true')
    args=p.parse_args()
    if not 1<=args.steps<=20000:raise ValueError('invalid update count')
    # No automatic warmup trigger or robot Session is created here.
    train(args)
if __name__=='__main__':main()
