"""Bounded-memory complete online state; immutable frozen weights stay in seed."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np


def save(root,agent,updater,step):
    import jax
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    destination=root/str(step)
    temporary=root/('.pending-'+str(uuid.uuid4()));temporary.mkdir()
    frozen={x.unsafe_buffer_pointer() for x in jax.tree.leaves(agent.actor.get_params(agent.actor_train_state).filter(updater.frozen))}
    records=[]
    for section,tree in [('agent',agent.replace(_infer_cache=None)),('target',updater.target)]:
        for index,(path,value) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
            if section=='agent' and not hasattr(value,'dtype'):value=jax.numpy.asarray(value)
            record=dict(section=section,index=index,path=jax.tree_util.keystr(path),shape=list(np.shape(value)),dtype=str(value.dtype) if hasattr(value,'dtype') else str(np.asarray(value).dtype))
            if isinstance(value,jax.Array) and value.unsafe_buffer_pointer() in frozen:
                record['seed']=True
            else:
                name=section+'-'+str(index)+'.bin';record['file']=name
                digest=hashlib.sha256();flat=value.reshape(-1) if hasattr(value,'reshape') else np.asarray(value).reshape(-1)
                with (temporary/name).open('wb') as stream:
                    for start in range(0,flat.size,1024*1024):
                        chunk=np.asarray(flat[start:start+1024*1024]).tobytes()
                        stream.write(chunk);digest.update(chunk)
                    stream.flush();os.fsync(stream.fileno())
                record['sha256']=digest.hexdigest()
            records.append(record)
    with (temporary/'manifest.json').open('w') as stream:
        json.dump(dict(format=1,step=step,records=records),stream);stream.flush();os.fsync(stream.fileno())
    directory_fd=os.open(str(temporary),os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    temporary.rename(destination)
    directory_fd=os.open(str(root),os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    return step


def restore(root,agent,updater,step):
    import jax
    directory=Path(root)/str(step)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['format']!=1 or manifest['step']!=step:raise ValueError('Unsupported online checkpoint')
    restored={}
    for section,tree in [('agent',agent.replace(_infer_cache=None)),('target',updater.target)]:
        pairs,structure=jax.tree_util.tree_flatten_with_path(tree)
        records=[r for r in manifest['records'] if r['section']==section]
        if len(records)!=len(pairs):raise ValueError('Checkpoint tree changed')
        values=[]
        for index,((path,template),record) in enumerate(zip(pairs,records)):
            dtype=np.asarray(0,dtype=template.dtype).dtype if hasattr(template,'dtype') else np.dtype(jax.numpy.asarray(template).dtype)
            if (record['index'],record['path'],record['shape'],record['dtype'])!=(index,jax.tree_util.keystr(path),list(np.shape(template)),str(dtype)):
                raise ValueError('Checkpoint leaf contract changed: '+str(path))
            if record.get('seed'):values.append(template);continue
            filename=directory/record['file'];digest=hashlib.sha256()
            with filename.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=record['sha256']:raise ValueError('Checkpoint checksum mismatch')
            value=np.fromfile(filename,dtype=dtype).reshape(np.shape(template))
            if section=='agent':value=jax.device_put(value,agent.rng.sharding);jax.block_until_ready(value)
            values.append(value)
        restored[section]=jax.tree_util.tree_unflatten(structure,values)
    updater.target=restored['target']
    return restored['agent']
