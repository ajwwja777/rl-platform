"""Explicit candidate-only learner bootstrap; native Actor wire format unchanged.

Run in a fresh learner process. Existing runtime/default entry never installs it.
Completed journal Episodes provide observed behavior returns, including HIL;
this does not convert assisted successes into autonomous labels.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(path):
    path=Path(path).resolve();profile=json.loads(path.read_text())
    if profile.get('schema')!='held-gripper-hil8-clip-v2':
        raise ValueError('Unsupported candidate training contract')
    root=Path(__file__).resolve().parents[3]
    for file,digest in profile.get('training_source_sha256',{}).items():
        target=(root/file).resolve()
        # The pinned upstream may be a readonly symlink in isolated worktrees.
        if sha256(target)!=digest:raise ValueError('Candidate training code identity mismatch: '+file)
    for name in ['teacher','normalization']:
        item=profile[name];target=(path.parent/item['file']).resolve()
        if path.parent not in target.parents or sha256(target)!=item['sha256']:
            raise ValueError('Candidate asset identity mismatch: '+name)
        profile[name]['path']=str(target)
    if not 0<=profile['mc_weight']<=1 or profile['retention_weight']<0 or profile['actor_q_weight']<0:
        raise ValueError('Invalid candidate loss contract')
    if profile.get('publication_policy')!='staged':
        raise ValueError('Candidate updates must be staged')
    return profile


class SupportedIndex:
    """Stable snapshot of a trusted journal; reject duplicate/source-conflicting credit.

    The Replay commits at Episode end. During an append the last incomplete
    pickle is ignored, and rows without a complete return are never accepted.
    """
    def __init__(self,journal,gamma,retention_episodes):
        self.path=Path(journal);self.gamma=gamma
        self.retained={tuple(k)for k in retention_episodes};self.signature=None;self.lookup={};self.records={}

    def refresh(self):
        from .target_attribution import reconstruct_observed_returns
        st=self.path.stat();signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if signature==self.signature:return
        rows=[]
        with self.path.open('rb')as f:
            while f.tell()<st.st_size:
                try:row=pickle.load(f)
                except (EOFError,pickle.UnpicklingError):break
                if f.tell()>st.st_size:break
                rows.append(row)
        if not rows:self.lookup={};self.signature=signature;return
        fields=['collection_phase_id','episode_id','step_id','source_chunk','rewards','done','success']
        data={k:np.stack([r[k]for r in rows])for k in fields}
        observed,report=reconstruct_observed_returns(data,self.gamma)
        # Require full prefix as well as continuity; do not bless evicted partial Episodes.
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[(int(r['collection_phase_id']),int(r['episode_id']))].append(i)
        for ids in groups.values():
            if not np.isin(data['source_chunk'][ids],[0,1,2,3]).all():observed[ids]=np.nan
            if min(int(rows[i]['step_id'])for i in ids)!=0:observed[ids]=np.nan
        keys=[(int(r['collection_phase_id']),int(r['episode_id']),int(r['step_id']))for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate journal identity')
        self.records={k:{name:np.asarray(rows[i][name]).copy()for name in ['source_chunk','rewards','done','success']}for i,k in enumerate(keys)}
        self.lookup={k:(float(observed[i]),np.isfinite(observed[i]),k[:2]in self.retained)for i,k in enumerate(keys)}
        self.signature=signature

    def attach(self,batch):
        self.refresh();count=len(batch['episode_id'])
        pairs=[self.lookup.get(tuple(int(batch[k][i])for k in ['collection_phase_id','episode_id','step_id']),(0.,False,False))for i in range(count)]
        if not all(p[1]for p in pairs):raise ValueError('Candidate batch contains incomplete or conflicting Episode evidence')
        for i in range(count):
            key=tuple(int(batch[k][i])for k in ['collection_phase_id','episode_id','step_id'])
            for name,expected in self.records[key].items():
                if name in batch and not np.array_equal(np.asarray(batch[name][i]),expected):
                    raise ValueError('Candidate batch and journal evidence disagree: '+name)
        return dict(batch,mc_return=np.asarray([p[0]for p in pairs],np.float32),mc_valid=np.ones(count,bool),retention_mask=np.asarray([p[2]for p in pairs],bool))


def build_components(cfg,profile):
    from rlt_online_rl import trainer
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from .held_gripper import HeldGripperCritic
    from .guarded_retained_actor import make_retained_train_step
    from .separate_actor_batch import with_separate_actor_batch
    if cfg.action_dim!=7 or cfg.proprio_dim!=7 or cfg.chunk_len!=10:
        raise ValueError('Candidate requires right-arm7D C10')
    budget=profile['runtime_budget']
    if cfg.warmup_post_collect_updates!=budget['base_step'] or cfg.grad_updates_per_cycle!=budget['updates_per_new_transition'] or cfg.freeze_after_warmup:
        raise ValueError('Candidate new-arrival budget contract differs')
    if cfg.warmup_q_weight!=profile['actor_q_weight'] or cfg.warmup_bc_weight!=cfg.online_bc_weight:
        raise ValueError('Candidate warmup resolver would change the trained loss')
    expected=profile['training_config']
    for key,value in expected.items():
        if key=='action_norm_stats_path':continue
        if getattr(cfg,key)!=value:raise ValueError('Candidate config differs from trained contract: '+key)
    if sha256(cfg.action_norm_stats_path)!=profile['normalization']['sha256']:
        raise ValueError('Runtime normalization differs from trained contract')
    adapter=ActionRepresentationAdapter.from_config(cfg)
    actor,critic=trainer._make_networks(cfg)
    if profile['held_gripper']:
        critic=HeldGripperCritic(critic,float(adapter.stats.q01[6]),float(adapter.stats.q99[6]))
    teacher=pickle.loads(Path(profile['teacher']['path']).read_bytes())
    if profile.get('target_policy')!='clip' or profile.get('actor_hil_slots')!=8:
        raise ValueError('Unvalidated candidate training policy')
    run=with_separate_actor_batch(make_retained_train_step(profile['mc_weight'],trainer._tree_to_jax(teacher['actor_params']),profile['retention_weight'],target_policy='clip'))
    return actor,critic,run


def install_supported_learner(profile_path,journal):
    """Install only inside a fresh selected learner, before constructor/spawn work."""
    from rlt_online_rl import trainer,replay
    if getattr(trainer,'_supported_candidate_installed',False):raise RuntimeError('Duplicate candidate bootstrap')
    profile=load_profile(profile_path)
    native_make=trainer._make_networks
    native_step=trainer.train_step
    native_class=trainer.LearnerService
    native_client=replay.ReplayClient
    bound={}
    def make(cfg):
        # build_components uses the unpatched constructor to avoid recursion.
        trainer._make_networks=native_make
        trainer.train_step=native_step
        try:a,c,run=build_components(cfg,profile)
        finally:
            trainer._make_networks=make
            trainer.train_step=train
        bound['run']=run
        return a,c
    def train(*args,**kwargs):
        if not bound:raise RuntimeError('Candidate training components not constructed')
        if float(kwargs['q_weight'])!=profile['actor_q_weight']:
            raise ValueError('Resolved Actor Q loss differs from candidate')
        from .supported_sampler import unpack_actor_batch
        state,batch,*rest=args
        return bound['run'](state,unpack_actor_batch(batch),*rest,**kwargs)
    index=SupportedIndex(journal,profile['training_config']['gamma'],profile['retention_episodes'])
    from .supported_sampler import SupportedSampler
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    checkpoint=Path(profile_path).parent/'checkpoints/latest.pkl'
    restore=pickle.loads(checkpoint.read_bytes())
    if restore['sampling_state']['global_step']!=restore['state']['global_step']:
        raise ValueError('Sampler/checkpoint update mismatch')
    from rlt_online_rl.config import RLTOnlineRLConfig
    cfg=dict(restore['rl_config']);cfg['action_norm_stats_path']=profile['normalization']['path']
    sampler=SupportedSampler(index,ActionRepresentationAdapter.from_config(RLTOnlineRLConfig(**cfg)),state=restore['sampling_state'],actor_hil_slots=8,floor=profile['episode_id_floor'])
    class Client(native_client):
        def sample_batch(self,*a,**kw):return sampler.sample_batch(*a,**kw)
    profile_sha=sha256(profile_path)
    class Learner(native_class):
        def __init__(self,cfg,service,*args,**kwargs):
            checkpoint=Path(service.checkpoint_dir)/'latest.pkl'
            served=Path(service.actor_snapshot_path)
            if 'candidates'not in checkpoint.parts or 'pending_actor'not in served.parts:
                raise ValueError('Candidate checkpoint / publication paths are not isolated and staged')
            payload=pickle.loads(checkpoint.read_bytes())
            if payload.get('candidate_profile_sha256')!=profile_sha:
                raise ValueError('Candidate checkpoint resume contract mismatch')
            super().__init__(cfg,service,*args,**kwargs)
        def save_checkpoint(self):
            if int(self.state.global_step)!=sampler.step:
                raise RuntimeError('Cannot checkpoint unsynchronized sampler')
            path=super().save_checkpoint()
            # Native saves are complete before adding the mandatory resume identity.
            for p in [Path(path),Path(path).with_name('step_%d.pkl'%int(self.state.global_step))]:
                payload=pickle.loads(p.read_bytes());payload['candidate_profile_sha256']=profile_sha
                payload['sampling_state']=sampler.state()
                temporary=p.with_suffix('.contract.tmp')
                with temporary.open('wb')as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
                temporary.replace(p)
            return path
    original_train_once=Learner.train_once
    def train_once(self,*args,**kwargs):
        result=original_train_once(self,*args,**kwargs)
        if result is not None:
            record=sampler.last_receipt
            if record['global_step']!=int(self.state.global_step):raise RuntimeError('Batch audit step mismatch')
            path=Path(profile_path).parent/'online/metrics/actor_critic_batches.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a')as f:f.write(json.dumps(record)+'\n')
        return result
    Learner.train_once=train_once
    replay.ReplayClient=Client;trainer._make_networks=make;trainer.train_step=train;trainer.LearnerService=Learner
    trainer._supported_candidate_installed=True
    return profile


def run_learner(system,profile_path,config_path):
    install_supported_learner(profile_path,system.replay.journal_path)
    from integrations.cobot_runtime.replay_audit import install_batch_audit
    install_batch_audit(config_path)
    from run_online_rl import _run_learner_service
    return _run_learner_service(system)



def validate_runtime_environment(profile,environ):
    """Validate settings, accepting the web alias only for identical execution."""
    from dataclasses import asdict
    from methods.openpi_rlt.cobot_adapter.execution_profiles import selected_profile
    expected=profile['runtime_environment']
    for key,value in expected.items():
        if key!='COBOT_RLT_EXECUTION_PROFILE'and environ.get(key)!=str(value):
            raise ValueError('Candidate environment differs from offline delivery contract: '+key)
    required=selected_profile(environ={'COBOT_RLT_EXECUTION_PROFILE':expected['COBOT_RLT_EXECUTION_PROFILE']})
    actual=selected_profile(environ=environ)
    if required is None or actual is None or required[1]!=actual[1]:
        raise ValueError('Candidate execution settings differ from fixed50 RTC/filter contract')
    return dict(name=actual[0],settings=asdict(actual[1]))


def run_registered(upstream_root,argv,profile_path):
    """Optional supervisor entry. It is not activated by model selection alone."""
    import functools,multiprocessing as mp,sys,os
    from .runtime import guarded_worker
    import run_online_rl as native
    profile=load_profile(profile_path)
    config_path=argv[argv.index('--config')+1]
    from rlt_online_rl.config import load_system_config_yaml
    system=load_system_config_yaml(config_path)
    paths=[system.replay.journal_path,system.learner_service.checkpoint_dir,
           system.learner_service.actor_snapshot_path,system.actor_service.snapshot_path]
    if any('candidates'not in Path(p).parts for p in paths):
        raise ValueError('Every candidate asset and Replay must be isolated')
    if system.actor_service.snapshot_path==system.learner_service.actor_snapshot_path:
        raise ValueError('Candidate learner cannot publish directly to served Actor')
    actual_execution=validate_runtime_environment(profile,os.environ)
    receipt=Path(config_path).parent/'metrics/runtime-contract.json'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(profile_sha256=sha256(profile_path),actual_execution=actual_execution,publication_policy='staged'),indent=2))
    # The new journal excludes old DEV. Reserve legacy DEV IDs rather than reuse
    # them for fresh Episodes; the Task5 UUID remains the raw collection identity.
    install_environment_floor(profile_path)
    original=native._spawn_process
    def spawn(name,target,*args,**kwargs):
        if name=='learner_service':target=functools.partial(run_learner,profile_path=profile_path,config_path=config_path)
        return original(name,guarded_worker,target,*args,**kwargs)
    native._spawn_process=spawn
    def stop(processes,*,logger,grace_sec=5.):
        rank={'env_driver':0,'learner_service':1,'actor_service':2,'replay_manager':3}
        for p in sorted(processes,key=lambda p:rank.get(p.name,0)):
            native._terminate_process(p,logger=logger,grace_sec=15. if p.name=='learner_service'else grace_sec)
    native._terminate_processes=stop
    sys.argv=[str(Path(upstream_root)/'rlt_online_rl/scripts/run_online_rl.py'),*argv]
    mp.set_start_method('spawn',force=True)
    args=native._parse_args()
    if args.system.role=='learner_service':return run_learner(args.system,profile_path,config_path)
    if args.system.role!='all':raise ValueError('Candidate entry supports all or learner_service only')
    return native.main(args)


def install_environment_floor(profile_path):
    """Spawn-safe ID reservation; no invented Replay rows or budget."""
    from rlt_online_rl import inference
    floor=int(load_profile(profile_path)['episode_id_floor'])
    original=inference.EnvDriver._next_episode_id
    installed=getattr(original,'_candidate_episode_floor',None)
    if installed is not None:
        if installed!=floor:raise ValueError('Candidate Episode floor changed')
        return
    def next_id(self):return max(floor,original(self))
    next_id._candidate_episode_floor=floor
    inference.EnvDriver._next_episode_id=next_id
