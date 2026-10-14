"""Versioned inference-only state: no optimizer, Replay or target VLA transfer."""
import dataclasses
import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
from online_protocol import atomic_json


def contract(bundle,project,delay):
    digest=hashlib.sha256()
    for directory in ('expo_ft','configs'):
        for source in sorted((Path(project)/directory).rglob('*.py')):
            digest.update(str(source.relative_to(project)).encode());digest.update(source.read_bytes())
    for name in ('train_pi_robo.py','scripts/dynamic_pick/run_server.sh'):
        digest.update((Path(project)/name).read_bytes())
    for name in ('deployment.json','norm_stats.json'):digest.update((Path(bundle)/name).read_bytes())
    return dict(source_bundle_sha256=digest.hexdigest(),rtc_delay=int(delay),recipe='native_sync',N=32,format=1)


def trees(agent):
    return dict(actor=agent.actor.get_params(agent.actor_train_state).filter(agent.actor.train_config.trainable_filter),
                encoder=agent.batch_encoder.params,edit=agent.edit_actor.params,critic=agent.target_critic.params)


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def export(root,agent,version,identity):
    import jax
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    destination=root/str(version)
    if destination.exists():
        existing=json.loads((destination/'manifest.json').read_text())
        if existing['identity']!=identity:raise ValueError('Actor snapshot identity changed')
        for section,tree in trees(agent).items():
            for index,(_,value) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
                record=next(x for x in existing['records'] if x['section']==section and x['index']==index)
                digest=hashlib.sha256();flat=value.reshape(-1)
                for start in range(0,flat.size,1024*1024):digest.update(np.asarray(flat[start:start+1024*1024]).tobytes())
                if digest.hexdigest()!=record['raw_sha256']:raise ValueError('Existing Actor version differs from restored learner')
        return existing
    temporary=root/('.pending-'+str(uuid.uuid4()));temporary.mkdir()
    records=[]
    for section,tree in trees(agent).items():
        for index,(key,value) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
            record=dict(section=section,index=index,path=jax.tree_util.keystr(key),shape=list(value.shape),dtype=str(value.dtype))
            seed=version==0 and section=='actor'
            record['seed']=seed
            name='%s-%d.gz'%(section,index)
            digest=hashlib.sha256();flat=value.reshape(-1)
            stream=None if seed else gzip.open(temporary/name,'wb',compresslevel=1)
            try:
                for start in range(0,flat.size,1024*1024):
                    chunk=np.asarray(flat[start:start+1024*1024]).tobytes();digest.update(chunk)
                    if stream is not None:stream.write(chunk)
            finally:
                if stream is not None:stream.close()
            record['raw_sha256']=digest.hexdigest()
            if not seed:
                with (temporary/name).open('rb') as handle:os.fsync(handle.fileno())
                record.update(file=name,bytes=(temporary/name).stat().st_size,sha256=digest_file(temporary/name))
            records.append(record)
    manifest=dict(format=1,actor_version=int(version),actor_step=int(agent.actor_train_state.step),identity=identity,records=records)
    atomic_json(temporary/'manifest.json',manifest)
    temporary.rename(destination)
    fd=os.open(str(root),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return manifest


def strip_training(agent):
    state=dataclasses.replace(agent.actor_train_state,params=agent.actor.get_params(agent.actor_train_state),ema_params=None,opt_state=())
    return agent.replace(actor_train_state=state,target_actor_params=None,_infer_cache=None,
        critic=agent.critic.replace(opt_state=()),batch_encoder=agent.batch_encoder.replace(opt_state=()),
        filter_critic=agent.filter_critic.replace(opt_state=()),edit_actor=agent.edit_actor.replace(opt_state=()),temp=agent.temp.replace(opt_state=()))


def restore(directory,agent,identity):
    import jax
    import flax.nnx as nnx
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['format']!=1 or manifest['identity']!=identity:raise ValueError('Actor snapshot contract mismatch')
    sections={}
    for section,tree in trees(agent).items():
        pairs,structure=jax.tree_util.tree_flatten_with_path(tree)
        records=[record for record in manifest['records'] if record['section']==section]
        if len(pairs)!=len(records):raise ValueError('Actor snapshot tree changed')
        values=[]
        for index,((key,template),record) in enumerate(zip(pairs,records)):
            if (record['index'],record['path'],record['shape'],record['dtype'])!=(index,jax.tree_util.keystr(key),list(template.shape),str(template.dtype)):
                raise ValueError('Actor snapshot leaf contract changed')
            if record['seed']:
                value=np.asarray(template)
            else:
                name='%s-%d.gz'%(section,index)
                if record['file']!=name or digest_file(directory/name)!=record['sha256']:raise ValueError('Actor payload checksum mismatch')
                expected=int(np.prod(template.shape))*np.dtype(template.dtype).itemsize
                with gzip.open(directory/name,'rb') as stream:raw=stream.read(expected+1)
                if len(raw)!=expected:raise ValueError('Actor payload size mismatch')
                value=np.frombuffer(raw,dtype=template.dtype).reshape(template.shape)
            if hashlib.sha256(value.tobytes()).hexdigest()!=record['raw_sha256']:raise ValueError('Actor raw checksum mismatch')
            values.append(template if record['seed'] else jax.device_put(value,agent.rng.sharding))
        sections[section]=jax.tree_util.tree_unflatten(structure,values)
    params=nnx.State.merge(agent.actor.get_params(agent.actor_train_state).filter(agent.actor.train_config.freeze_filter),sections['actor'])
    state=dataclasses.replace(agent.actor_train_state,params=params,ema_params=None,opt_state=(),step=manifest['actor_step'])
    result=agent.replace(actor_train_state=state,batch_encoder=agent.batch_encoder.replace(params=sections['encoder']),
        edit_actor=agent.edit_actor.replace(params=sections['edit']),target_critic=agent.target_critic.replace(params=sections['critic']),_infer_cache=None)
    return result.cache_infer_params(),manifest


def prune(root,current):
    """Keep current and previous complete snapshots; only this run's numeric dirs."""
    import shutil
    root=Path(root).resolve()
    complete=sorted((int(p.name),p) for p in root.iterdir() if p.is_dir() and p.name.isdigit() and (p/'manifest.json').is_file())
    previous=max((version for version,_ in complete if version<current),default=None)
    for version,path in complete:
        if version not in (current,previous) and version<current and path.resolve().parent==root:
            shutil.rmtree(path)
