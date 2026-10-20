"""Owned host snapshots remain compatible after optimizer buffers are donated."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import flax.struct as struct
import flax.nnx as nnx
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))

@struct.dataclass
class State:
    params: object
    step: object
    opt_state: object
    ema_params: object = None

@struct.dataclass
class Net:
    params: object

@struct.dataclass
class Agent:
    actor: object = struct.field(pytree_node=False)
    actor_train_state: object
    rng: object
    batch_encoder: object
    edit_actor: object
    target_critic: object
    _infer_cache: object = None
    def cache_infer_params(self):
        return self

def example():
    params = nnx.State({'train': nnx.VariableState(nnx.Param, jnp.arange(12,dtype=jnp.float32).reshape(3,4)),
                        'frozen': nnx.VariableState(nnx.Param, jnp.ones(4))})
    config = SimpleNamespace(trainable_filter=nnx.PathContains('train'),freeze_filter=nnx.PathContains('frozen'))
    actor = SimpleNamespace(get_params=lambda state: state.params,train_config=config)
    shared = Net({'v': jnp.arange(4,dtype=jnp.bfloat16)})
    agent = Agent(actor,State(params,jnp.asarray(21,jnp.int32),{'m':jnp.full((3,4),2.,jnp.float32)}),
                  jax.random.PRNGKey(42),shared,Net({'v':jnp.arange(3,dtype=jnp.float32)}),shared)
    updater = SimpleNamespace(frozen=config.freeze_filter,target={'ema':np.arange(12,dtype=np.float32).reshape(3,4)})
    return agent,updater

def test_checkpoint_and_inference_are_readable_by_existing_consumers(tmp_path):
    from continuous_checkpoint import capture,write
    from stream_checkpoint import restore as restore_checkpoint
    from actor_snapshot import restore as restore_actor
    agent,updater=example()
    snapshot=capture(agent,updater)
    result=write(tmp_path,7,21,{'recipe':'test'},snapshot,True)
    template,restorer=example()
    restored=restore_checkpoint(tmp_path/'checkpoints',template,restorer,7)
    for expected,actual in zip(jax.tree.leaves(agent),jax.tree.leaves(restored)):
        np.testing.assert_array_equal(expected,actual)
        assert expected.dtype==actual.dtype
    np.testing.assert_array_equal(restorer.target['ema'],np.arange(12,dtype=np.float32).reshape(3,4))
    actor,manifest=restore_actor(tmp_path/'actors'/'21',template,{'recipe':'test'})
    np.testing.assert_array_equal(actor.actor_train_state.params['train'].value,np.arange(12).reshape(3,4))
    np.testing.assert_array_equal(actor.batch_encoder.params['v'],np.arange(4))
    assert actor.batch_encoder.params['v'].dtype==jnp.bfloat16
    assert actor.actor_train_state.step==21 and actor.actor_train_state.opt_state==()
    assert result['checkpoint']==7 and result['actor_manifest']==manifest
    assert snapshot['host_bytes']==sum(v.nbytes for v in snapshot['buffers'].values())
    assert result['host_bytes']==snapshot['host_bytes']
    assert all(isinstance(v,np.ndarray) and v.flags.owndata and not v.flags.writeable for v in snapshot['buffers'].values())

def test_capture_owns_mutable_arrays_before_background_write(tmp_path):
    from continuous_checkpoint import capture,write
    from stream_checkpoint import restore
    agent,updater=example()
    snap=capture(agent,updater)
    updater.target['ema'][:]=-50
    weights=agent.actor_train_state.params['train'].value
    jax.jit(lambda x:x+100,donate_argnums=(0,))(weights).block_until_ready()
    assert weights.is_deleted()
    write(tmp_path,1,21,{'recipe':'test'},snap,False)
    template,other=example()
    rebuilt=restore(tmp_path/'checkpoints',template,other,1)
    np.testing.assert_array_equal(rebuilt.actor_train_state.params['train'].value,np.arange(12).reshape(3,4))
    np.testing.assert_array_equal(other.target['ema'],np.arange(12).reshape(3,4))
    assert not (tmp_path/'actors').exists()

def test_seed_weights_are_skipped_and_exact_aliases_share_owned_storage():
    from continuous_checkpoint import capture
    agent,updater=example()
    snap=capture(agent,updater)
    frozen=[r for r in snap['checkpoint_records'] if r.get('seed')]
    assert len(frozen)==1 and 'frozen' in frozen[0]['path'] and 'buffer' not in frozen[0]
    assert snap['host_bytes']==176  # 48 train +4 step +48 optimizer +8 RNG +8 encoder +12 edit +48 CPU EMA
    # encoder and target Critic alias exactly: inference maps to one owned copy.
    encoder=next(r for r in snap['actor_records'] if r['section']=='encoder')
    critic=next(r for r in snap['actor_records'] if r['section']=='critic')
    assert encoder['buffer']==critic['buffer']
    def check(value):
        assert not isinstance(value,jax.Array)
        if isinstance(value,dict):
            for child in value.values():check(child)
        elif isinstance(value,(list,tuple)):
            for child in value:check(child)
    check(snap)

def test_capture_rejects_inference_parameters_absent_from_checkpoint():
    from continuous_checkpoint import capture
    agent,updater=example()
    # A getter allocating different parameters would defeat buffer-identity mapping.
    agent.actor.get_params=lambda state:nnx.State({'train':nnx.VariableState(nnx.Param,state.params['train'].value+1),
                                                 'frozen':state.params['frozen']})
    with pytest.raises(ValueError,match='checkpoint'):
        capture(agent,updater)

def test_publication_cannot_overwrite_an_existing_identity(tmp_path):
    from continuous_checkpoint import capture,write
    agent,updater=example();snap=capture(agent,updater)
    write(tmp_path,1,21,{'recipe':'first'},snap,True)
    manifest=(tmp_path/'actors'/'21'/'manifest.json').read_bytes()
    with pytest.raises(ValueError,match='identity|differs'):
        write(tmp_path,2,21,{'recipe':'other'},snap,True)
    assert (tmp_path/'actors'/'21'/'manifest.json').read_bytes()==manifest

def test_invalid_capture_shape_is_rejected_before_publishing(tmp_path):
    from continuous_checkpoint import capture,write
    agent,updater=example();snap=capture(agent,updater)
    record=next(r for r in snap['checkpoint_records'] if not r.get('seed'))
    record['shape']=[999]
    with pytest.raises(ValueError,match='shape|capture'):
        write(tmp_path,1,21,{},snap,True)
    assert not (tmp_path/'actors'/'21'/'manifest.json').exists()


def test_existing_native_export_is_reused_without_replacing_payloads(tmp_path):
    from continuous_checkpoint import capture,write
    from actor_snapshot import export
    agent,updater=example();identity={'recipe':'test'}
    native=export(tmp_path/'actors',agent,21,identity)
    before={p.name:(p.stat().st_ino,p.read_bytes()) for p in (tmp_path/'actors'/'21').iterdir()}
    result=write(tmp_path,1,21,identity,capture(agent,updater),True)
    assert result['actor_manifest']==native
    after={p.name:(p.stat().st_ino,p.read_bytes()) for p in (tmp_path/'actors'/'21').iterdir()}
    assert before==after


@pytest.mark.parametrize('devices',[2,4])
def test_sharded_capture_restores_with_unchanged_values_and_dtypes(tmp_path,devices):
    from continuous_checkpoint import capture,write
    from stream_checkpoint import restore
    if len(jax.devices())<devices:pytest.skip('Run with four virtual CPU devices')
    agent,updater=example()
    mesh=jax.sharding.Mesh(np.array(jax.devices()[:devices]),('data',))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    split=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('data'))
    agent=jax.tree.map(lambda x:jax.device_put(x,split if x.ndim and x.shape[0]%devices==0 else replicated),agent)
    write(tmp_path,2,21,{'recipe':'test'},capture(agent,updater),True)
    template,restorer=example()
    restored=restore(tmp_path/'checkpoints',template,restorer,2)
    for expected,actual in zip(jax.tree.leaves(agent),jax.tree.leaves(restored)):
        np.testing.assert_array_equal(expected,actual)
        assert expected.dtype==actual.dtype


def test_background_writer_never_accesses_jax_or_the_live_updater(tmp_path,monkeypatch):
    from continuous_checkpoint import capture,write
    import concurrent.futures
    import device_layout
    agent,updater=example();owned=capture(agent,updater)
    def forbidden(*args,**kwargs):raise AssertionError('Writer accessed a device')
    monkeypatch.setattr(jax,'device_get',forbidden)
    monkeypatch.setattr(device_layout,'buffer_identity',forbidden)
    updater.target['ema'][:]=999
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result=pool.submit(write,tmp_path,2,21,{},owned,True).result(timeout=3)
    assert result['checkpoint']==2 and result['actor_manifest']['actor_version']==21
    target=next(r for r in json.loads((tmp_path/'checkpoints'/'2'/'manifest.json').read_text())['records'] if r['section']=='target')
    np.testing.assert_array_equal(np.fromfile(tmp_path/'checkpoints'/'2'/target['file'],dtype=np.float32),np.arange(12))


def test_python_scalars_do_not_alias_when_temporary_device_addresses_are_reused(monkeypatch):
    from continuous_checkpoint import capture
    import device_layout
    original=device_layout.buffer_identity
    def reused_scalar_address(value):
        # Model allocator reuse after each temporary conversion is released.
        if value.shape==() and value.dtype==jnp.float32:
            return ('reused-temporary-scalar-address',)
        return original(value)
    monkeypatch.setattr(device_layout,'buffer_identity',reused_scalar_address)
    agent,updater=example()
    agent=agent.replace(actor_train_state=agent.actor_train_state.replace(
        opt_state={'first':0.125,'second':0.25,'third':0.75}))
    captured=capture(agent,updater)
    actual={r['path']:captured['buffers'][r['buffer']].item()
            for r in captured['checkpoint_records'] if '.opt_state[' in r['path']}
    assert actual=={".actor_train_state.opt_state['first']":0.125,
                    ".actor_train_state.opt_state['second']":0.25,
                    ".actor_train_state.opt_state['third']":0.75}
