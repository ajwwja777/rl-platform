"""Paired no-ROS EXPO action ablations on read-only recorded observations.

No training, HTTP service, Replay insert or robot publisher. Every arm receives
identical cached VLA candidates, observations, physical prefix and selection RNG.
This measures open-loop action roughness, not closed-loop success or stability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def roughness(actions):
    """Within-window six-joint metrics; never join unrelated windows."""
    x = np.asarray(actions)[..., :6]
    delta = np.diff(x, axis=-2)
    accel = np.diff(delta, axis=-2)
    return dict(delta_rms_rad=float(np.sqrt(np.mean(delta**2))),
                second_difference_rms_rad=float(np.sqrt(np.mean(accel**2))),
                max_delta_rad=float(np.max(np.abs(delta))),
                reversals=int(np.sum((delta[..., 1:, :]*delta[..., :-1, :] < 0) &
                    (np.minimum(abs(delta[..., 1:, :]), abs(delta[..., :-1, :])) > .005))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('adapter','project','bundle','snapshot','episodes','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--windows',type=int,default=8)
    args=p.parse_args()
    if args.windows<1 or args.windows>32:p.error('windows must be in [1,32] per episode')
    if args.output.exists():p.error('output already exists')
    if args.episodes.resolve() in args.output.resolve().parents:p.error('output must be outside recorded episodes')
    sys.path.insert(0,str(args.adapter))
    from native_online import source_path,load_native
    source_path(args.project)
    from actor_snapshot import strip_training,restore,contract
    from core import limited_plan
    import jax
    import expo_ft.agents.alg.realtime_expo_ft as native
    agent,config,_=load_native(args.bundle,5,aligned=True)
    agent=strip_training(agent)
    agent,manifest=restore(args.snapshot,agent,contract(args.bundle,args.project,5))
    if manifest['actor_version'] != 0:raise ValueError('This initial-policy diagnostic requires version0')
    captured={}
    original=native._jitted_fast_select
    def capture(*a,**kw):
        result=original(*a,**kw)
        captured.clear();captured.update(result)
        return result
    native._jitted_fast_select=capture
    rows=[];arrays={};identities=[]
    arms=[('base',True,0,.1),('q_only',False,0,.1),
          ('edit_010',False,32,.1),('edit_003',False,32,.03),('edit_001',False,32,.01)]
    def observation(z):return {k:z[k].copy() for k in ('state','left_wrist_image','right_wrist_image')}
    for directory in sorted(args.episodes.iterdir()):
        if not (directory/'finished.json').is_file():continue
        files=sorted(directory.glob('[0-9]*.npz'))
        if len(files)<16:continue
        for start in np.unique(np.linspace(0,len(files)-16,args.windows,dtype=int)):
            with np.load(files[start],allow_pickle=False) as z:obs=observation(z)
            seed=jax.random.PRNGKey(1000+len(identities))
            for delay in (0,2,5):
                prefix=None
                with np.load(files[start+delay],allow_pickle=False) as z:current=observation(z)
                if delay:
                    values=[]
                    for file in files[start:start+delay]:
                        with np.load(file,allow_pickle=False) as z:values.append(z['actions'].copy())
                    physical=np.stack(values)
                    transformed=agent.actor.input_transforms({**obs,'actions':physical.copy()})['actions']
                    prefix=np.pad(np.asarray(transformed),((0,0),(0,25)))
                    anchor=physical[-1]
                else:anchor=obs['state']
                fixed=agent.replace(rng=jax.device_put(seed,agent.rng.sharding))
                cached,selected_agent=fixed.sample_pre_cache(dict(obs),prefix)
                np.asarray(cached)
                for name,base,edits,scale in arms:
                    variant=selected_agent.replace(n_edit_samples=edits,edit_scale=scale)
                    started=time.monotonic()
                    actions,_,_=variant.sample_actions(dict(current),precached=cached,delay=delay,only_base_actions=base)
                    actions=np.asarray(actions)
                    if actions.shape!=(8,7) or not np.isfinite(actions).all():raise ValueError('Invalid output')
                    limited=limited_plan(actions,anchor,True)
                    idx=None if captured['idx'] is None else int(captured['idx'])
                    row=dict(episode=directory.name,frame=int(start),delay=delay,arm=name,
                        selected_index=idx,edit_selected=bool(idx is not None and idx>=32),
                        raw=roughness(actions),limited=roughness(limited),seconds=time.monotonic()-started)
                    rows.append(row);key=f'case_{len(rows):04d}'
                    arrays[key+'_raw']=actions;arrays[key+'_limited']=limited
                    row['array_key']=key
                print(json.dumps({'completed_conditions':len(rows)//5,'episode':directory.name,'frame':int(start),'delay':delay}),flush=True)
            identities.append(dict(episode=directory.name,frame=int(start),file_sha256=hashlib.sha256(files[start].read_bytes()).hexdigest()))
    if not rows:raise ValueError('No eligible recorded windows')
    summary={}
    for delay in (0,2,5):
        summary[str(delay)]={}
        for name,*_ in arms:
            group=[r for r in rows if r['delay']==delay and r['arm']==name]
            summary[str(delay)][name]=dict(count=len(group),edited_fraction=float(np.mean([r['edit_selected'] for r in group])),
                **{stage:{metric:float(np.mean([r[stage][metric] for r in group])) for metric in group[0][stage]} for stage in ('raw','limited')})
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'actions.npz',**arrays)
    report=dict(actor_version=0,manifest_sha256=hashlib.sha256((args.snapshot/'manifest.json').read_bytes()).hexdigest(),
        robot_commands=0,training_updates=0,protocol='paired fixed recorded observations/prefix and cached32 candidates; within-window metrics only; not counterfactual closed-loop rollout',
        episodes=str(args.episodes),identities=identities,summary=summary,rows=rows)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
