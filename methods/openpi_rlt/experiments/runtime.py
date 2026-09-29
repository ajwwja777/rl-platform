"""Opt-in learner adapter. Installed only inside an experimental learner process."""
import pickle
from pathlib import Path
import numpy as np
from integrations.cobot_runtime.replay_audit import identity
from .credit import episode_credit, make_train_step, Profile

class CreditIndex:
    def __init__(self,path,gamma):
        self.path=Path(path);self.gamma=gamma
        self.offset=0;self.signature=None;self.rows=[];self.lookup={}
    def refresh(self):
        st=self.path.stat();signature=(st.st_dev,st.st_ino)
        if signature!=self.signature or st.st_size<self.offset:
            self.offset=0;self.rows=[];self.signature=signature
        if st.st_size==self.offset:return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            while f.tell()<st.st_size:
                start=f.tell()
                try:r=pickle.load(f)
                except (EOFError,pickle.UnpicklingError):break
                if f.tell()>st.st_size:break
                self.rows.append({k:r[k] for k in ("collection_phase","collection_phase_id","episode_id","step_id","done","success","rewards") if k in r})
                self.offset=f.tell()
        values,valid=episode_credit(self.rows,self.gamma)
        keys=[identity(r) for r in self.rows]
        if len(keys)!=len(set(keys)):raise ValueError("Ambiguous Replay identities; candidate credit is unsafe")
        self.lookup={key:(values[i],valid[i]) for i,key in enumerate(keys)}
    def attach(self,batch):
        self.refresh()
        count=len(batch["episode_id"])
        pairs=[self.lookup.get(identity({k:v[i] for k,v in batch.items()
            if k in ("collection_phase_id","episode_id","step_id")}), (0.,False)) for i in range(count)]
        return dict(batch,mc_return=np.array([p[0] for p in pairs],np.float32),
                    mc_valid=np.array([p[1] for p in pairs],bool))

def install_learner(profile,journal,gamma):
    """Process-local opt-in. Baseline code paths never call this function."""
    from rlt_online_rl import replay,trainer
    options=Profile(**profile)
    if options.sampling!="uniform":
        raise ValueError("Online adapter currently supports uniform sampling only; tail experiments remain offline")
    if options.mc_weight==0:return
    if getattr(trainer,"_credit_experiment_installed",False):
        raise RuntimeError("Only one credit profile may be installed per learner process")
    base=replay.ReplayClient
    index=CreditIndex(journal,gamma)
    class CreditClient(base):
        def sample_batch(self,*args,**kwargs):
            return index.attach(super().sample_batch(*args,**kwargs))
    replay.ReplayClient=CreditClient
    trainer.train_step=make_train_step(options.mc_weight)
    trainer._credit_experiment_installed=True

def run_learner(system,profile,config_path):
    # mp.spawn imports this function, then installs the adapter in the child.
    # Installing only in the supervisor would silently leave spawned training unchanged.
    install_learner(profile,system.replay.journal_path,system.rl.gamma)
    from integrations.cobot_runtime.replay_audit import install_batch_audit
    install_batch_audit(config_path)
    from run_online_rl import _run_learner_service
    _run_learner_service(system)

def run_registered(upstream_root,argv,profile_name):
    import functools,json,multiprocessing as mp,sys
    project=Path(__file__).resolve().parents[3]
    registry=json.loads((project/"configs/experiments/credit_ablation.json").read_text())
    profile=registry["profiles"][profile_name]
    options=Profile(**profile)
    if options.sampling!="uniform" or not options.mc_weight:
        raise ValueError("Choose a registered uniform MC candidate")
    sys.path.insert(0,str(Path(upstream_root)/"rlt_online_rl/scripts"))
    import run_online_rl as native
    config_path=argv[argv.index("--config")+1]
    from rlt_online_rl.config import load_system_config_yaml
    system=load_system_config_yaml(config_path)
    # Experimental outputs must be separately named; never overwrite online/latest.
    for path in (system.learner_service.checkpoint_dir,system.learner_service.actor_snapshot_path):
        if "candidates" not in Path(path).parts:
            raise ValueError("Experimental checkpoints must live under a candidates directory")
    original=native._spawn_process
    def spawn(name,target,*args,**kwargs):
        if name=="learner_service":
            target=functools.partial(run_learner,profile=profile,config_path=config_path)
        return original(name,target,*args,**kwargs)
    native._spawn_process=spawn
    def stop_in_dependency_order(processes, *, logger, grace_sec=5.0):
        # Stop commands first, then flush Learner while Replay is reachable.
        rank={"env_driver":0,"learner_service":1,"actor_service":2,"replay_manager":3}
        for process in sorted(processes,key=lambda p:rank.get(p.name,0)):
            native._terminate_process(process,logger=logger,
                grace_sec=15.0 if process.name=="learner_service" else grace_sec)
    native._terminate_processes=stop_in_dependency_order
    sys.argv=[str(Path(upstream_root)/"rlt_online_rl/scripts/run_online_rl.py"),*argv]
    mp.set_start_method("spawn",force=True)
    args=native._parse_args()
    if args.system.role=="learner_service":return run_learner(args.system,profile,config_path)
    if args.system.role!="all":raise ValueError("Experimental entry supports all or learner_service")
    return native.main(args)
