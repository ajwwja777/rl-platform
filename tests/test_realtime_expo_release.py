import hashlib,json,pickle
from pathlib import Path
import pytest,yaml
from methods.openpi_rlt.realtime_expo.release import prepare_runtime,CONTRACT

def fixture(tmp_path):
    stage=tmp_path/'stage1';(stage/'params').mkdir(parents=True)
    cfg=dict(action_dim=7,proprio_dim=7,z_dim=2048,chunk_len=10,action_representation='delta_chunk')
    for mode in ('frozen','online'):
        root=tmp_path/mode;(root/'actor_snapshot').mkdir(parents=True);(root/'checkpoints').mkdir()
        (root/'action_norm_stats.json').write_text('{}')
        (root/'actor_snapshot/actor_snapshot.pkl').write_bytes(pickle.dumps(dict(global_step=7000,rl_config=cfg)))
        (root/'checkpoints/latest.pkl').write_bytes(pickle.dumps(dict(progress=dict(warmup_ready_adds_total=551))))
        config=dict(experiment=dict(rl=dict(warmup_post_collect_updates=7000)),runtime=dict(actor_service=dict(snapshot_path=str(root/'actor_snapshot/actor_snapshot.pkl')),env_driver=dict(control_frequency_hz=10,replay_feature_batch_size=1)))
        (tmp_path/f'{mode}.yaml').write_text(yaml.safe_dump(config))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    m=dict(contract=CONTRACT,cameras=['left_wrist_0_rgb','right_wrist_0_rgb'],stage1=str(stage),source_step=7000,replay_seed=dict(adds_total=551),norm_sha256=digest(tmp_path/'frozen/action_norm_stats.json'),frozen_actor_sha256=digest(tmp_path/'frozen/actor_snapshot/actor_snapshot.pkl'))
    (tmp_path/'release.json').write_text(json.dumps(m));return tmp_path

def test_online_refuses_frozen_actor_or_changed_budget(tmp_path):
    root=fixture(tmp_path)
    with pytest.raises(ValueError,match='selected Actor'):prepare_runtime(root,'online',root/'frozen/actor_snapshot/actor_snapshot.pkl',root/'out.yaml')
    path=root/'online/checkpoints/latest.pkl';path.write_bytes(pickle.dumps(dict(progress=dict(warmup_ready_adds_total=552))))
    with pytest.raises(ValueError,match='budget anchor'):prepare_runtime(root,'online',root/'online/actor_snapshot/actor_snapshot.pkl',root/'out.yaml')

def test_frozen_evaluation_can_read_online_actor_without_learning(tmp_path):
    root=fixture(tmp_path);selected=root/'online/actor_snapshot/actor_snapshot.pkl'
    prepare_runtime(root,'frozen',selected,root/'out.yaml')
    assert yaml.safe_load((root/'out.yaml').read_text())['runtime']['actor_service']['snapshot_path']==str(selected)

def test_normalization_and_camera_contract_fail_closed(tmp_path):
    root=fixture(tmp_path);path=root/'release.json';m=json.loads(path.read_text());m['cameras'].append('base_0_rgb');path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='cameras'):prepare_runtime(root,'frozen',root/'frozen/actor_snapshot/actor_snapshot.pkl',root/'out.yaml')
    m['cameras'].pop();path.write_text(json.dumps(m));(root/'frozen/action_norm_stats.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='normalization'):prepare_runtime(root,'frozen',root/'frozen/actor_snapshot/actor_snapshot.pkl',root/'out.yaml')


def test_release_pickle_uses_public_numpy_constructors():
    import numpy as np
    from methods.openpi_rlt.realtime_expo.release import portable_pickle
    data={'a':np.arange(12,dtype=np.float32).reshape(3,4),'b':np.array(7,dtype=np.int32),'c':np.float32(.5)}
    encoded=portable_pickle(data)
    assert b'numpy._core' not in encoded
    restored=pickle.loads(encoded)
    for key in data:np.testing.assert_array_equal(restored[key],data[key])
