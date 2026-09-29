#!/usr/bin/env python3
"""Exercise real candidate restore/update/publish/restart with synthetic arrival counts.

Reads an existing Replay batch, but never adds data or writes production artifacts.
"""
import argparse,json,os,pickle,sys,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"third_party/openpi-rlt/rlt_online_rl/src")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    import numpy as np,yaml,jax
    from rlt_online_rl import replay,trainer
    from rlt_online_rl.config import RLTOnlineRLConfig,LearnerServiceConfig
    from methods.openpi_rlt.experiments.runtime import install_learner
    cfg=yaml.safe_load(a.config.read_text());rl=RLTOnlineRLConfig(**cfg["experiment"]["rl"])
    journal=Path(cfg["runtime"]["replay"]["journal_path"]);rows=[]
    with journal.open("rb") as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    batch={k:np.stack([r[k] for r in rows[:128]]) for k in rows[0] if k!="collection_phase"}
    class Replay:
        def __init__(self,*args,**kwargs):self.adds=len(rows)
        def stats(self):return {"size":self.adds,"adds_total":self.adds}
        def sample_batch(self,*args,**kwargs):return {k:v.copy() for k,v in batch.items()}
    replay.ReplayClient=Replay
    install_learner({"mc_weight":.3},str(journal),rl.gamma)
    source=replay.ReplayClient()
    parent=ROOT/"runtime/experiments";parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="candidate-validation-",dir=parent) as td:
        root=Path(td);ckpt=root/"checkpoints";ckpt.mkdir()
        shutil.copyfile(Path(cfg["runtime"]["learner_service"]["checkpoint_dir"])/"latest.pkl",ckpt/"latest.pkl")
        service_cfg=LearnerServiceConfig(sample_batch_size=128,checkpoint_dir=str(ckpt),
            actor_snapshot_path=str(root/"actor_snapshot.pkl"),push_actor_interval_steps=1,checkpoint_interval_steps=1000)
        service=trainer.LearnerService(rl,service_cfg,source,metrics_path=str(root/"metrics.jsonl"),rng=jax.random.PRNGKey(42))
        initial=int(service.state.global_step)
        assert service.train_once() is None,"Unexpected catch-up updates on start"
        source.adds+=1
        metrics=[service.train_once() for _ in range(rl.grad_updates_per_cycle)]
        assert all(m is not None for m in metrics)
        assert service.train_once() is None,"UTD budget did not stop"
        assert int(service.state.global_step)==initial+5
        assert all(abs(m["mc_effective_weight"]-.3)<1e-5 for m in metrics)
        service.save_checkpoint()
        restored=trainer.LearnerService(rl,service_cfg,source,metrics_path=str(root/"restart.jsonl"),rng=jax.random.PRNGKey(0))
        for x,y in zip(jax.tree.leaves(service.state),jax.tree.leaves(restored.state)):
            np.testing.assert_array_equal(x,y)
        assert restored.train_once() is None
        snapshot=pickle.load((root/"actor_snapshot.pkl").open("rb"))
        receipt={"status":"passed","initial_step":initial,"final_step":int(restored.state.global_step),
            "published_step":snapshot["global_step"],"published_actor":snapshot["version"],
            "mc_weight":metrics[-1]["mc_effective_weight"],"replay_writes":0,"robot_publishers":0,
            "startup_catchup_updates":0,"updates_per_synthetic_arrival":5,"restart_state_exact":True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
if __name__=="__main__":main()
