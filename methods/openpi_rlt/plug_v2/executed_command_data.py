"""EXPERIMENT: executed-command critic dataset. Isolated RTC replay: full actions, factual decision clock, immutable UUID split.
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
    states,actions,times,valid,human,vi,terminal,frames,plans=source(item);rows=[];feature_jobs={};teacher_keys={}
    for seg in segments(valid&human,times):
        for d in (0,6):
            last=len(seg)-10-d
            if last<0:continue
            anchors=sorted(set(list(range(0,last+1,10))+[last]+([j+4 for j in range(0,len(seg)-9,10) if j+4<=last] if d==6 else [])))
            for pos,offset in enumerate(anchors):
                i=seg[offset];done=seg[-1]==terminal and offset==last
                nxt=seg[anchors[pos+1]] if pos+1<len(anchors) else i
                duration=(anchors[pos+1]-offset) if pos+1<len(anchors) else len(seg)-offset
                prefix=np.zeros((50,14),np.float32)
                if d:prefix[:6]=actions[seg[offset:offset+6]]
                key=(i,d);feature_jobs[key]=prefix
                rewards=np.zeros(16,np.float32)
                if done and item['success']:rewards[duration-1]=1
                row=dict(key=key,next_key=(nxt,d),proprio=context(states[i],prefix,d),
                    action_chunk=actions[seg[offset+d:offset+d+10],7:14],rewards=rewards,
                    done=done,duration=duration,td_valid=done or nxt!=i,source_chunk=np.full(10,2,np.uint8),frame=i,
                    next_frame=nxt,delay=d,kind='human',position=i/max(1,len(times)-1))
                rows.append(row);teacher_keys[key]=row
    for row in rows:
        if row['delay']==0 and not row['done']:
            successor=(row['frame']+4,6)
            row['next_key']=successor if successor in teacher_keys else None
            row['next_frame']=row['frame']+4 if successor in teacher_keys else row['frame']
            row['duration']=4;row['td_valid']=successor in teacher_keys
    features={}
    previous=RUN/'diagnostics/rtc-clean-20260920/episodes'/target.name
    if previous.exists() and previous!=target:
        with np.load(previous) as saved:
            if 'source_chunk' in saved:
                for j in np.where(saved['source_chunk'][:,0]==2)[0]:
                    features[int(saved['frame'][j]),int(saved['delay'][j])]=(saved['z_rl'][j].copy(),saved['ref_chunk'][j].copy())
    needed={int(vi[i]) for i,d in feature_jobs if (i,d) not in features};images={i:{} for i in needed}
    root=Path(item['root']);index=item['index']
    if needed:
        for cam,dest in [('cam_high','base_0_rgb'),('cam_left_wrist','left_wrist_0_rgb'),('cam_right_wrist','right_wrist_0_rgb')]:
            video=root/f'videos/chunk-{index//1000:03d}/observation.images.{cam}/episode_{index:06d}.mp4'
            with av.open(str(video)) as c:
                for j,f in enumerate(c.decode(c.streams.video[0])):
                    if j in needed:images[j][dest]=f.to_ndarray(format='rgb24')
    for (i,d),prefix in feature_jobs.items():
        if (i,d) in features:continue
        if len(images[int(vi[i])])!=3:raise ValueError('missing aligned RGB')
        response=rpc.infer({'state':states[i],'images':images[int(vi[i])],'action_prefix':prefix,'prefix_length':d})
        if not np.array_equal(np.asarray(response['ref_chunk'])[:d],prefix[:d]):raise ValueError('teacher prefix changed')
        features[i,d]=(np.asarray(response['z_rl'],np.float32),np.asarray(response['ref_chunk'][d:d+10,7:14],np.float32))
    policy_audit=[];trace_meta={}
    if plans:
        with np.load(item['trace']) as trace:trace_meta=json.loads(str(trace['metadata']))
        if trace_meta.get('actor_candidate')=='rtc-corrective-r1':profile=['bounded']
        if any(p.get('trace_schema')==2 for p in plans):profile=['recorded_schema2']
    if plans and profile:
        latest={}
        for p in plans:
            k=(int(p['generation']),int(p['tick']))
            if k not in latest or int(p['sequence'])>int(latest[k]['sequence']):latest[k]=p
        sent={(int(f['generation']),int(f['tick'])-1):i for i,f in enumerate(frames) if valid[i] and not human[i] and f['command'] is not None}
        for (g,t),p in sorted(latest.items()):
            if (g,t) not in sent:continue
            d=int(p['prefix_length']);i=sent[g,t];observed=[j for j in range(10) if (g,t+d+j) in sent]
            if not observed:continue
            prefix=np.zeros((50,14),np.float32);prefix[:6]=p['prefix'];raw=np.tile(np.asarray(p['state'],np.float32),(50,1));raw[:d]=prefix[:d];raw[d:d+10]=p['ref']
            proposal=np.asarray(p['ref'],np.float32).copy()
            if p.get('trace_schema')==2:
                proposal=np.asarray(p['raw_action_plan'],np.float32)[d:d+10].copy()
            elif p.get('actor_version',-1)!=-1:
                restored=legacy_raw(p,trace_meta)
                if restored is None:continue
                proposal[:,7:13]=restored[:,7:13]
            raw[d:d+10]=proposal
            possible=[]
            for name in profile:
                if name=='recorded_schema2':
                    prediction=np.asarray(p['conditioned_plan'],np.float32)[d:d+10]
                    if all(np.max(abs(prediction[j]-actions[sent[g,t+d+j]]))<1e-6 for j in observed):possible.append(prediction)
                    continue
                filt=CommandFilter(velocity=.1,acceleration=.9) if name=='bounded' else CommandFilter()
                if name=='historical_legacy_without_clamp':filt.lower[:]=-np.inf;filt.upper[:]=np.inf
                try:prediction=filt.plan(raw,p['state'],prefix,d)[d:d+10]
                except ValueError:continue
                if all(np.max(abs(prediction[j]-actions[sent[g,t+d+j]]))<1e-6 for j in observed):possible.append(prediction)
            if not possible or any(np.max(abs(x-possible[0]))>1e-6 for x in possible):continue
            # The whole raw chosen action is known; execution coverage is audited,
            # never encoded as critic input or padded using terminal duration.
            sequence=[]
            for k in range(t,t+16):
                if (g,k) not in sent:break
                j=sent[g,k]
                if sequence and not .015<times[j]-times[sequence[-1]]<=.05:break
                sequence.append(j)
            if not sequence:continue
            following=sorted(k for pg,k in latest if pg==g and k>t)
            next_tick=following[0] if following else t+10
            next_plan=latest.get((g,next_tick));next_i=sent.get((g,next_tick))
            done=False;duration=min(10,len(sequence));next_key=None;td_valid=False
            if next_plan is not None and next_i is not None and 0<next_tick-t<len(sequence):
                next_key=('policy',g,next_tick);duration=next_tick-t;td_valid=True
            elif sequence[-1]==terminal:
                done=True;duration=len(sequence);td_valid=True
            elif sequence[-1]+1<len(frames):
                j=sequence[-1]+1
                if (j,0) in teacher_keys and .015<times[j]-times[sequence[-1]]<=.05:
                    next_key=(j,0);duration=len(sequence);td_valid=True
            rewards=np.zeros(16,np.float32)
            if done and item['success']:rewards[duration-1]=1
            key=('policy',g,t);features[key]=(np.asarray(p['z'],np.float32),np.asarray(p['ref'],np.float32)[:,7:14])
            row=dict(key=key,next_key=next_key,proprio=context(np.asarray(p['state'],np.float32),prefix,d),action_chunk=possible[0][:,7:14],
                rewards=rewards,done=done,duration=duration,td_valid=td_valid,source_chunk=np.full(10,0 if p.get('actor_version',-1)==-1 else 1,np.uint8),frame=i,next_frame=next_i if next_i is not None else i,
                delay=d,kind='policy_reference',position=i/max(1,len(times)-1),_terminal_tail=(len(sequence) if sequence[-1]==terminal else 0));rows.append(row)
            policy_audit.append({'tick':t,'executed_tail':len(observed),'duration':duration,'done':done})
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
    meta={**item,'format':'upstream_RTC99_executed_commands_SMDP_v2','rows':len(rows),'human_rows':sum(r['kind']=='human' for r in rows),'policy_rows':len(policy_audit),
          'terminal_rows':sum(r['done'] for r in rows),'td_valid_rows':sum(r['td_valid'] for r in rows),'policy_audit':policy_audit,'production_publish':False,
          'model_metadata':rpc.get_server_metadata(),'sources_unchanged':True}
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
