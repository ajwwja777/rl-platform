import hashlib
import json
import pickle
from pathlib import Path

import pytest
import yaml

from integrations.cobot_runtime.online_seed import prepare, latest_branch


def fixture(tmp_path, step=5000):
    seed=tmp_path/'seed'
    (seed/'checkpoints').mkdir(parents=True)
    (seed/'actor_snapshot').mkdir()
    state={key: {'weight': [1,2,3]} for key in ('actor_params','target_actor_params',
        'critic_params','target_critic_params','actor_opt_state','critic_opt_state')}
    state.update(rng=[1,2],global_step=step,actor_version=2500)
    (seed/'checkpoints/latest.pkl').write_bytes(pickle.dumps(dict(state=state,progress=dict(warmup_ready_adds_total=600))))
    (seed/'actor_snapshot/actor_snapshot.pkl').write_bytes(pickle.dumps(dict(version=2500,global_step=step,actor_params=state['actor_params'])))
    (seed/'action_norm_stats.json').write_text('{}')
    replay=tmp_path/'replay.pkl';replay.write_bytes(b'existing replay')
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(experiment=dict(rl=dict(freeze_after_warmup=True)),runtime=dict(
        actor_service={},learner_service={},monitoring={},env_driver={},replay=dict(journal_path=str(replay))))))
    return seed,config,replay


def test_fork_retains_full_resume_state_seed_and_replay_and_registers_new_steps(tmp_path):
    seed,config,replay=fixture(tmp_path)
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in seed.rglob('*') if p.is_file()}
    before=replay.read_bytes();branch=tmp_path/'models/online_from_5000/run1';target=tmp_path/'run/online.yaml'
    prepare(seed,branch,config,target,tmp_path/'run')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert (branch/'checkpoints/latest.pkl').read_bytes()==(seed/'checkpoints/latest.pkl').read_bytes()
    saved=yaml.safe_load(target.read_text());runtime=saved['runtime']
    assert runtime['learner_service']['checkpoint_dir']==str(branch/'checkpoints')
    assert runtime['actor_service']['snapshot_path']==str(branch/'actor_snapshot/actor_snapshot.pkl')
    assert saved['experiment']['rl']['freeze_after_warmup'] is False
    assert replay.read_bytes()==before and runtime['replay']['journal_path']==str(replay)
    assert latest_branch(tmp_path/'models')['source_step']==5000
    with pytest.raises(ValueError,match='already_exists'):prepare(seed,branch,config,target,tmp_path/'run')


def test_incompatible_or_missing_seed_does_not_create_a_branch(tmp_path):
    seed,config,_=fixture(tmp_path,step=6000);branch=tmp_path/'new'
    with pytest.raises(ValueError,match='identity_mismatch'):prepare(seed,branch,config,tmp_path/'cfg',tmp_path/'run')
    assert not branch.exists()
    (seed/'checkpoints/latest.pkl').unlink()
    with pytest.raises(ValueError,match='full_training'):prepare(seed,branch,config,tmp_path/'cfg',tmp_path/'run')
    assert not branch.exists()


def test_stage1_registration_keeps_nvme_separate_from_online_usb_storage():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'configs/rlt/plug_v3_yyshadow/manifest.json').read_text())
    assert manifest['checkpoint']==manifest['stage1_root']=='/home/agilex/jiaan/data/rlt/plug_insertion/reference_4999'
    assert manifest['model_root'].startswith('/media/agilex/Getea1/')
