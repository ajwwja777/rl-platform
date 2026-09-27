"""Pure temporal replay contract. No model, optimizer or robot I/O."""
import hashlib
import math
import numpy as np
VERSION=2
CHUNK=10

def segments(valid,generation):
    valid=np.asarray(valid,bool);generation=np.asarray(generation,np.int64)
    if valid.shape!=generation.shape or valid.ndim!=1:raise ValueError('mask shape mismatch')
    result=[];i=0
    while i<len(valid):
        if not valid[i]:i+=1;continue
        start=i;g=generation[i]
        while i<len(valid) and valid[i] and generation[i]==g:i+=1
        result.append((start,i,int(g)))
    return result

def teacher_rows(actions,valid,generation,*,terminal_frame,terminal_success,bc_allowed):
    actions=np.asarray(actions,np.float32);rows=[]
    for segment,(start,end,g) in enumerate(segments(valid,generation)):
        for t in range(start,end,2):
            duration=min(CHUNK,end-t);mask=np.arange(CHUNK)<duration
            terminal=t+duration>terminal_frame>=t
            reward=np.zeros(CHUNK,np.float32)
            if terminal and terminal_success:reward[terminal_frame-t]=1.
            rows.append(dict(frame=t,decision_tick=t,stream=1,segment=segment,generation=g,
                duration=duration,done=terminal,reward=reward,
                action=actions[np.minimum(t+np.arange(CHUNK),end-1)].copy(),
                future_mask=mask,bc_mask=mask & bool(bc_allowed),hil=True,
                next_anchor=t+duration,segment_end=end))
    return rows

def _array(value,shape):
    array=np.asarray(value,np.float32)
    if array.shape!=shape or not np.isfinite(array).all():raise ValueError('invalid factual plan '+str(shape))
    return array

def policy_rows(frames,plans,*,terminal_success,allow_bc,frame_allowed=None):
    # Runtime frame.tick is the NEXT queue tick, after the logged command was sent.
    # Only the latest accepted plan at a given (generation, request tick) governs its tail.
    latest={}
    for p in plans:
        key=(int(p['generation']),int(p['tick']))
        if key not in latest or int(p['sequence'])>int(latest[key]['sequence']):latest[key]=p
    valid=[];generation=[]
    for i,f in enumerate(frames):
        good=bool(f['valid_for_training'] and f['phase']=='rollout' and f['command'] is not None)
        if frame_allowed is not None:good &= bool(frame_allowed[i])
        valid.append(good);generation.append(f['generation'])
    terminal_trace=max((i for i,f in enumerate(frames) if f['valid_for_training']),default=-1)
    rows=[];skipped=[]
    for segment,(start,end,g) in enumerate(segments(valid,generation)):
        # A missed control tick is a truncation, never permission to stitch across it.
        by_tick={};parts=[];part={};previous=None
        for i in range(start,end):
            tick=int(frames[i]['tick'])-1
            if previous is not None and tick!=previous+1:
                if part:parts.append(part)
                part={}
            part[tick]=i;previous=tick
        if part:parts.append(part)
        for part_number,by_tick in enumerate(parts):
            for (pg,t),p in sorted(latest.items()):
                if pg!=g or t not in by_tick:continue
                indices=[]
                for offset in range(CHUNK):
                    if t+offset not in by_tick:break
                    indices.append(by_tick[t+offset])
                if not indices:continue
                d=int(p['prefix_length'])
                if d not in (0,6):raise ValueError('untrained factual RTC delay')
                state=_array(p['state'],(14,));prefix=_array(p['prefix'],(6,14))
                context=_array(p['context'],(99,))
                expected=np.r_[state,(prefix-state).reshape(-1) if d else np.zeros(84),d/6].astype(np.float32)
                np.testing.assert_allclose(context,expected,atol=1e-7,rtol=1e-6)
                commands=np.asarray([frames[i]['command'] for i in indices],np.float32)
                if not np.isfinite(commands).all():raise ValueError('nonfinite executed commands')
                # The stored locked prefix must match commands actually sent, not teacher labels.
                if d:
                    np.testing.assert_array_equal(commands[:min(d,len(commands))],prefix[:min(d,len(commands))])
                duration=len(indices);action=np.broadcast_to(state,(CHUNK,14)).copy()
                # This decision commits TEN post-prefix actions. Its last six become
                # the next decision's fixed queue prefix; they still belong to THIS action.
                # Rewards/discount use the ten-tick decision interval, not a shifted window.
                tail=[by_tick[t+d+k] for k in range(CHUNK) if t+d+k in by_tick]
                count=len(tail)
                if count:action[:count]=np.asarray([frames[i]['command'] for i in tail],np.float32)
                mask=np.arange(CHUNK)<count
                terminal=indices[-1]==terminal_trace
                reward=np.zeros(CHUNK,np.float32)
                if terminal and terminal_success:reward[duration-1]=1.
                rows.append(dict(frame=indices[0],decision_tick=t,stream=0,
                    segment=segment*10000+part_number,generation=g,duration=duration,
                    done=terminal,reward=reward,action=action,future_mask=mask,
                    bc_mask=mask & bool(allow_bc),hil=False,next_anchor=t+duration,
                    segment_end=max(by_tick)+1,z=_array(p['z'],(2048,)),
                    context=context,ref=_array(p['ref'],(CHUNK,14))))
    return rows

def pack_rows(rows):
    if not rows:return None
    # Exact same-stream/same-segment successor. No searchsorted across a pause/HIL gap.
    lookup={(r['stream'],r['segment'],r['generation'],r['decision_tick']):i for i,r in enumerate(rows)}
    successors=[];bootstrap=[]
    for i,r in enumerate(rows):
        key=(r['stream'],r['segment'],r['generation'],r['next_anchor'])
        other=lookup.get(key)
        enabled=not r['done'] and other is not None
        bootstrap.append(enabled);successors.append(other if enabled else i)
    keys=('z','context','ref','action','reward','done','duration','future_mask','bc_mask',
          'frame','hil','stream','segment','generation','decision_tick')
    arrays={k:np.asarray([r[k] for r in rows]) for k in keys}
    arrays['bootstrap']=np.asarray(bootstrap,bool)
    arrays['truncated']=~arrays['done'] & ~arrays['bootstrap']
    arrays['next_row']=np.asarray(successors,np.int64)
    for key in ('z','context','ref','future_mask'):
        arrays['next_'+key]=arrays[key][successors]
    validate_arrays(arrays)
    return arrays

def validate_arrays(a):
    n=len(a['z']);i=np.arange(n);j=a['next_row'];boot=a['bootstrap']
    if np.any(j<0) or np.any(j>=n):raise ValueError('invalid successor index')
    if np.any(a['done'] & boot):raise ValueError('terminal bootstrap')
    for key in ('generation','segment','stream'):
        if np.any(a[key][j[boot]]!=a[key][i[boot]]):raise ValueError('cross-boundary bootstrap')
    if np.any(a['decision_tick'][j[boot]]!=a['decision_tick'][i[boot]]+a['duration'][i[boot]]):
        raise ValueError('incorrect Bellman time span')
    if np.any(a['bc_mask'] & ~a['future_mask']):raise ValueError('padding used by imitation')
    if np.any(a['reward'][~a['done']]):raise ValueError('reward on nonterminal/truncation')
    if np.any(a['reward']*(np.arange(CHUNK)[None,:]>=a['duration'][:,None])):
        raise ValueError('reward on unexecuted time')
    for key in ('z','context','ref','action','reward','next_z','next_context','next_ref'):
        if not np.isfinite(a[key]).all():raise ValueError('nonfinite replay '+key)

def assign_splits(items,previous):
    # Initial 20% per outcome group, exact count, by original UUID. Existing splits never move.
    registry=dict(previous)
    for group in ('success','failure'):
        uuids=sorted({u for u,g in items if g==group},key=lambda u:hashlib.sha256(('42:'+u).encode()).digest())
        if len(uuids)<3:raise ValueError('at least three fresh '+group+' episodes required')
        unseen=[u for u in uuids if u not in registry]
        if not any(u in registry for u in uuids):
            heldout=set(uuids[-max(1,math.ceil(len(uuids)*.2)):])
            for u in uuids:registry[u]='val' if u in heldout else 'train'
        else:
            for u in unseen:registry[u]='val' if int(hashlib.sha256(('42:'+u).encode()).hexdigest(),16)%5==0 else 'train'
            if not any(registry[u]=='val' for u in uuids):
                if not unseen:raise ValueError('no heldout UUID; do not change previous training splits')
                registry[unseen[-1]]='val'
        if not any(registry[u]=='train' for u in uuids):raise ValueError('no train UUID for '+group)
    return registry
