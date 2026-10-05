#!/usr/bin/env python3
"""Reproduce and verify seed budget behavior with real networks, synthetic counts."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]
os.environ.setdefault('JAX_PLATFORMS','cpu')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True,help='Read-only archived Warmup for cached batch')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new receipt path')
    import numpy as np
    import yaml
    import jax
    from integrations.cobot_runtime.online_seed import prepare
    from rlt_online_rl.config import system_config_from_mapping
    from rlt_online_rl.trainer import LearnerService
    sources={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.seed.rglob('*') if p.is_file()}
    source_replay_hash=hashlib.sha256(args.replay.read_bytes()).hexdigest()
    rows=[]
    with args.replay.open('rb') as stream:
        for _ in range(128):rows.append(pickle.load(stream))
    batch={k:np.stack([r[k] for r in rows]) for k in rows[0] if k!='collection_phase'}
    class Source:
        adds=4013
        def stats(self):return dict(size=self.adds,adds_total=self.adds)
        def sample_batch(self,*a,**kw):return {k:v.copy() for k,v in batch.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='seed-budget-',dir=str(args.output.parent)) as temporary:
        root=Path(temporary)
        journal=root/'synthetic_metadata_journal.pkl'
        with journal.open('wb') as stream:
            for i in range(4013):pickle.dump(dict(episode_id=i),stream)
        config=yaml.safe_load((ROOT/'configs/rlt/plug_v3_yyshadow/online_rl.yaml').read_text())
        config['runtime']['replay']['journal_path']=str(journal)
        source_config=root/'source.yaml';source_config.write_text(yaml.safe_dump(config))
        historical=root/'inherited';new=root/'new_arrivals'
        results={}
        for policy,destination in [('inherit',historical),('new_arrivals',new)]:
            target=root/(policy+'.yaml')
            prepare(args.seed,destination,source_config,target,root/policy,replay_budget_policy=policy)
            cfg=system_config_from_mapping(yaml.safe_load(target.read_text()))
            source=Source()
            learner=LearnerService(cfg.rl,cfg.learner_service,source,metrics_path=str(root/policy/'metrics/learner.jsonl'))
            original=pickle.loads((args.seed/'checkpoints/latest.pkl').read_bytes())['state']
            for key in original:
                for a,b in zip(jax.tree_util.tree_leaves(getattr(learner.state,key)),jax.tree_util.tree_leaves(original[key])):
                    np.testing.assert_array_equal(a,b)
            results[policy]=dict(startup_pending_updates=learner._pending_update_budget,
                replay_anchor=learner._warmup_ready_adds_total,full_state_equal_to_seed=True)
            if policy=='new_arrivals':
                assert learner.train_once() is None
                assert learner._pending_update_budget==0
                source.adds+=1
                metrics=[learner.train_once() for _ in range(5)]
                assert all(m is not None and all(np.isfinite(v) for v in m.values()) for m in metrics)
                assert int(learner.state.global_step)==5005 and int(learner.state.actor_version)==2502
                assert learner.train_once() is None
                learner.flush_artifacts()
                restored=LearnerService(cfg.rl,cfg.learner_service,source,metrics_path=str(root/policy/'metrics/restart.jsonl'))
                for a,b in zip(jax.tree_util.tree_leaves(restored.state),jax.tree_util.tree_leaves(learner.state)):
                    np.testing.assert_array_equal(a,b)
                assert restored.train_once() is None
                results[policy].update(new_arrival_updates=5,step=5005,actor=2502,restart_exact=True)
        assert results['inherit']['startup_pending_updates']==7230
    assert sources=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert source_replay_hash==hashlib.sha256(args.replay.read_bytes()).hexdigest()
    result=dict(status='passed',results=results,source_sha256={str(p):h for p,h in sources.items()},
        archived_replay_sha256=source_replay_hash,production_writes=0,robot_publishers=0,
        boundary='4013 is a synthetic arrival-count fixture matching the read-only audit count. Uses actual Warmup5k state and cached training batch; no whole production Replay copied or changed.')
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
