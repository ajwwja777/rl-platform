"""Isolated RTC replay: full actions, factual decision clock, immutable UUID split.
Human d6 prefixes are explicit expert commanded continuations, not future images.
Policy raw actions are admitted only after reference-only command reconstruction.
"""
import json,sys,os,argparse,hashlib
from pathlib import Path
import numpy as np
from .upstream_replay import ROOT,RUN,nearest
from .rpc import ModelClient
from .replay import atomic_npz
from .rtc_queue import CommandFilter
from .training_flow import session_phase,rtc_training_idle

def context(state,prefix,d):
    c=np.r_[state,(prefix[:6]-state).reshape(-1) if d else np.zeros(84),d/6].astype(np.float32)
    return np.r_[c[7:14],c[:7],c[14:]].astype(np.float32)

PASSIVE=np.r_[0:7,13]

def corrected_prefix(prefix,passive_commands):
    """Use factual right-joint commands and the episode's latched action-space passive commands."""
    value=np.asarray(prefix,np.float32).copy();passive=np.asarray(passive_commands,np.float32)
    if value.ndim!=2 or value.shape[1]!=14 or passive.shape!=(14,) or not np.isfinite(value).all() or not np.isfinite(passive).all():
        raise ValueError('invalid RTC prefix/passive command shape')
    value[:,PASSIVE]=passive[PASSIVE]
    return value

def rtc_schedule(length,terminal):
    """One causal C10/d6 chain: d0@start, then d6@start+4+10k.

    A terminal chain is shifted by at most nine frames so its final full action
    chunk ends exactly at the factual terminal; no terminal action is padded.
    """
    n=int(length)
    if n<10:return []
    start=(n-10 if n<20 else n%10) if terminal else 0
    rows=[dict(offset=start,delay=0,action_slice=(start,start+10))]
    tick=start+4
    while tick+16<=n:
        rows.append(dict(offset=tick,delay=6,action_slice=(tick+6,tick+16)));tick+=10
    for i,row in enumerate(rows):
        if i+1<len(rows):
            row.update(next_position=i+1,duration=rows[i+1]['offset']-row['offset'],done=False,td_valid=True,reward_index=None)
        else:
            done=bool(terminal and row['action_slice'][1]==n);duration=n-row['offset'] if done else min(16,n-row['offset'])
            row.update(next_position=None,duration=duration,done=done,td_valid=done,reward_index=duration-1 if done else None)
    return rows

def link_censored_rows(rows,times,max_gap_sec=1.0,fps=30.0):
    """Connect a censored chain tail to the next observed decision across a short handover gap."""
    ordered=sorted(rows,key=lambda row:row['frame']);linked=0
    for pos,row in enumerate(ordered):
        if row['done'] or row['td_valid']:continue
        successor=next((other for other in ordered[pos+1:] if other['frame']>row['frame']),None)
        if successor is None:continue
        gap=float(times[successor['frame']]-times[row['frame']])
        if not 0<gap<=max_gap_sec:continue
        row['next_key']=successor['key'];row['next_frame']=successor['frame']
        row['duration']=max(1,int(round(gap*fps)));row['td_valid']=True;linked+=1
    return linked

def source(item):
    import pyarrow.parquet as pq
    root=Path(item['root']);i=item['index'];tab=pq.read_table(root/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet')
    states=np.stack(tab['observation.state'].to_pylist()).astype(np.float32);actions=np.stack(tab['action'].to_pylist()).astype(np.float32)
    if item['expert']:
        n=len(states);return states,actions,np.arange(n)/30,np.ones(n,bool),np.ones(n,bool),np.arange(n),n-1 if item['terminal'] else -1,None,None
    with np.load(item['trace']) as f:frames=json.loads(str(f['frames_json']));plans=json.loads(str(f['plans_json']))
    with np.load(root/'source_facts.npz') as f:
        stamps=np.maximum(f['rollout/topic_timestamp/front_left'],f['rollout/topic_timestamp/front_right'])
        cmd=f['rollout/coordinator_command'];cmd_t=f['rollout/topic_timestamp/coordinator_right'];cmd_ok=f['rollout/valid_mask/coordinator_right']
    with np.load(root/'training_mask.npz') as f:mask=f['valid']
    times=np.array([f['ros_timestamp'] for f in frames]);vi=nearest(stamps,times);states=np.array([f['state'] for f in frames],np.float32)
    actions=np.zeros_like(states);valid=np.zeros(len(frames),bool);human=np.zeros(len(frames),bool)
    terminal=max((j for j,f in enumerate(frames) if f['valid_for_training']),default=-1)
    for j,f in enumerate(frames):
        k=vi[j]
        if not f['valid_for_training'] or not mask[k] or abs(times[j]-stamps[k])>.05:continue
        if f['phase']=='rollout' and f['command'] is not None:actions[j]=f['command'];valid[j]=True
        elif f['phase']=='hil' and 'right' in f['mode'] and cmd_ok[k] and abs(times[j]-cmd_t[k])<=.05:
            actions[j]=cmd[k];valid[j]=True;human[j]=True
    valid &= np.isfinite(states).all(1)&np.isfinite(actions).all(1)
    return states,actions,times,valid,human,vi,terminal,frames,plans

def segments(mask,times):
    parts=[];part=[]
    for i,ok in enumerate(mask):
        if part and (not ok or not .015<times[i]-times[part[-1]]<=.05):parts.append(part);part=[]
        if ok:part.append(i)
    if part:parts.append(part)
    return parts

_legacy_actor=None
def legacy_raw(plan,meta):
    global _legacy_actor
    if meta.get('actor_candidate')!='rtc-corrective-r1' or plan.get('actor_version')!=2000:return None
    import torch
    from .corrective_policy import CorrectiveActor
    from .fixed_candidate import manifest
    if _legacy_actor is None:
        m=manifest(require_ready=False)
        if m['sha256']!='a197ab49db46bf7089740eb868400df57861051c34eeeaee625394d713a44918':raise ValueError('historical actor lineage changed')
        saved=torch.load(m['path'],map_location='cpu',weights_only=False)
        _legacy_actor=CorrectiveActor(m['inference_config']);_legacy_actor.load_state_dict(saved['actor']);_legacy_actor.eval();torch.set_num_threads(2)
    with torch.no_grad():return _legacy_actor(*[torch.tensor(np.asarray(plan[k],np.float32)[None]) for k in ('z','context','ref')]).numpy()[0]

def one(item,out,rpc,profile):
    import av
    target=out/'episodes'/f"{item['uuid']}_{item['index']:06d}.npz"
    if target.exists():
        with np.load(target) as f:return json.loads(str(f['metadata']))
    states,actions,times,valid,human,vi,terminal,frames,plans=source(item);rows=[];feature_jobs={};chain_number=0
    trace_meta={}
    if plans:
        with np.load(item['trace']) as trace:trace_meta=json.loads(str(trace['metadata']))
    sources=[('human',valid&human,2)]
    if not item['expert']:
        policy_source=0 if trace_meta.get('actor_version',-1)==-1 else 1
        sources.insert(0,('policy',valid&~human,policy_source))
    for kind,mask,source_id in sources:
        for seg in segments(mask,times):
            schedule=rtc_schedule(len(seg),seg[-1]==terminal)
            chain=('chain',chain_number);chain_number+=1
            keys=[chain+(pos,) for pos in range(len(schedule))]
            for pos,spec in enumerate(schedule):
                offset=spec['offset'];d=spec['delay'];i=seg[offset]
                prefix=np.zeros((50,14),np.float32)
                if d:prefix[:6]=actions[seg[offset:offset+6]]
                key=keys[pos];next_key=keys[spec['next_position']] if spec['next_position'] is not None else None
                rewards=np.zeros(16,np.float32)
                if spec['done'] and item['success']:rewards[spec['reward_index']]=1
                row=dict(key=key,next_key=next_key,proprio=None,
                    action_chunk=actions[[seg[x] for x in range(*spec['action_slice'])],7:14],rewards=rewards,
                    done=spec['done'],duration=spec['duration'],td_valid=spec['td_valid'],
                    source_chunk=np.full(10,source_id,np.uint8),frame=i,next_frame=i,delay=d,kind=kind,
                    position=i/max(1,len(times)-1))
                rows.append(row);feature_jobs[key]=dict(frame=i,delay=d,prefix=prefix,chain=chain,row=row)

    needed={int(vi[j['frame']]) for j in feature_jobs.values()}
    images={i:{} for i in needed};root=Path(item['root']);index=item['index']
    if needed:
        for cam,dest in [('cam_high','base_0_rgb'),('cam_left_wrist','left_wrist_0_rgb'),('cam_right_wrist','right_wrist_0_rgb')]:
            video=root/f'videos/chunk-{index//1000:03d}/observation.images.{cam}/episode_{index:06d}.mp4'
            with av.open(str(video)) as c:
                for j,f in enumerate(c.decode(c.streams.video[0])):
                    if j in needed:images[j][dest]=f.to_ndarray(format='rgb24')

    features={};passive_latches={}
    for key,job in feature_jobs.items():
        i=job['frame'];d=job['delay'];prefix=job['prefix']
        if d:
            if job['chain'] not in passive_latches:raise ValueError('d6 feature precedes its d0 passive latch')
            prefix=corrected_prefix(prefix,passive_latches[job['chain']]);job['prefix']=prefix
        if len(images[int(vi[i])])!=3:raise ValueError('missing aligned RGB')
        response=rpc.infer({'state':states[i],'images':images[int(vi[i])],'action_prefix':prefix,'prefix_length':d})
        ref=np.asarray(response['ref_chunk'],np.float32)
        if not np.array_equal(ref[:d],prefix[:d]):raise ValueError('teacher prefix changed')
        if d==0:passive_latches[job['chain']]=ref[0].copy()
        features[key]=(np.asarray(response['z_rl'],np.float32),ref[d:d+10,7:14])
        job['row']['proprio']=context(states[i],prefix,d)
    handover_links=link_censored_rows(rows,times)
    policy_audit=[{'frame':r['frame'],'delay':r['delay'],'duration':r['duration'],'done':r['done']} for r in rows if r['kind']=='policy']
    lookup={r['key']:r for r in rows};packed=[]
    for row in rows:
        nextrow=lookup.get(row['next_key'])
        if nextrow is None:
            if row.get('_terminal_tail',0):
                row['done']=True;row['duration']=row['_terminal_tail'];row['rewards'][:]=0
                if item['success']:row['rewards'][row['duration']-1]=1
            row['td_valid']=bool(row['done']);nextrow=row
        row['next_frame']=nextrow['frame']
        z,ref=features[row['key']];nz,nref=features[nextrow['key']]
        packed.append({k:v for k,v in row.items() if k not in ('key','next_key','kind','_terminal_tail')} | dict(z_rl=z,ref_chunk=ref,next_z_rl=nz,next_ref_chunk=nref,next_proprio=nextrow['proprio']))
    meta={**item,'format':'upstream_RTC99_causal_clock_passive_latch_SMDP_v2','rows':len(rows),'human_rows':sum(r['kind']=='human' for r in rows),'policy_rows':len(policy_audit),
          'terminal_rows':sum(r['done'] for r in rows),'td_valid_rows':sum(r['td_valid'] for r in rows),'policy_audit':policy_audit,'production_publish':False,
          'model_metadata':rpc.get_server_metadata(),'sources_unchanged':True,'feature_contract':'runtime passive commands latched from d0 Stage1; right prefix factual',
          'decision_clock':'d0@start,d6@start+4+10k','short_handover_smdp_links':handover_links,'short_handover_max_gap_sec':1.0}
    arrays={k:np.asarray([r[k] for r in packed]) for k in packed[0]} if packed else {}
    atomic_npz(target,**arrays,metadata=np.array(json.dumps(meta)));print('RTC_DATA',item['uuid'],len(rows),meta['terminal_rows'],flush=True);return meta

def collect_items():
    from .storage import releases_for_phase
    from .replay_contract import assign_splits
    from .cli import atomic
    base=ROOT.parent.parent/'data/rlt/plug_v2';expert=base/'demonstrations/lerobot'
    selection=json.loads((expert/'selection_manifest.json').read_text())['episodes'];last={}
    for e in selection:last[e['source_uuid']]=max(last.get(e['source_uuid'],-1),e['episode_index'])
    items=[dict(root=str(expert),index=e['episode_index'],uuid=e['source_uuid'],split=e['split'],expert=True,success=True,phase='demonstrations',terminal=e['episode_index']==last[e['source_uuid']]) for e in selection]
    releases=[]
    for phase in ('warmup','online'):
        for path in releases_for_phase(phase):
            d=json.loads(path.read_text());m=d['metadata']
            if d.get('cohort')=='plug_v2' and d.get('status')=='validated' and m['outcome'] in ('success','failure'):
                releases.append((phase,path,d))
    registry_path=RUN/'replay/split_registry.json';prior=json.loads(registry_path.read_text())
    registry=assign_splits([(d['source_uuid'],d['metadata']['outcome']) for _,_,d in releases],prior)
    if any(registry[k]!=v for k,v in prior.items()):raise ValueError('existing UUID split changed')
    if registry!=prior:atomic(registry_path,registry)
    for phase,path,d in releases:
        m=d['metadata'];items.append(dict(root=str(path.parent),index=0,uuid=d['source_uuid'],split=registry[d['source_uuid']],expert=False,success=m['outcome']=='success',phase=phase,trace=m['trace']))
    return items

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.relative_to(RUN.resolve());out.mkdir(exist_ok=True)
    old=collect_items()
    reconstruction=json.loads((RUN/'diagnostics/upstream-audit-20260919/rtc-reference-reconstruction.json').read_text())
    profiles={r['uuid']:r['matching_profiles'] for r in reconstruction['rows']}
    result=[];rpc=ModelClient()
    try:
        for item in old:
            if not rtc_training_idle():raise RuntimeError('Session must remain stopped')
            result.append(one(item,out,rpc,profiles.get(item['uuid'],[])))
            (out/'progress.json').write_text(json.dumps({'phase':'preparing','pid':os.getpid(),'episodes':len(result),'total':len(old),'production_publish':False}))
    finally:rpc.close()
    (out/'manifest.json').write_text(json.dumps(result,indent=2));(out/'progress.json').write_text(json.dumps({'phase':'completed','episodes':len(result),'rows':sum(m['rows'] for m in result),'production_publish':False}));print('RTC_DATA_COMPLETE',flush=True)
if __name__=='__main__':main()
