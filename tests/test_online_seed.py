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
    replay=tmp_path/'replay.pkl'
    with replay.open('wb') as stream:
        for i in range(700): pickle.dump(dict(episode_id=i), stream)
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(experiment=dict(rl=dict(freeze_after_warmup=True,warmup_post_collect_updates=5000)),runtime=dict(
        actor_service={},learner_service={},monitoring={},env_driver={},replay=dict(journal_path=str(replay))))))
    return seed,config,replay


def test_fork_retains_full_resume_state_seed_and_replay_and_registers_new_steps(tmp_path):
    seed,config,replay=fixture(tmp_path)
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in seed.rglob('*') if p.is_file()}
    before=replay.read_bytes();branch=tmp_path/'models/online_from_5000/run1';target=tmp_path/'run/online.yaml'
    prepare(seed,branch,config,target,tmp_path/'run')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    original=pickle.loads((seed/'checkpoints/latest.pkl').read_bytes())
    created=pickle.loads((branch/'checkpoints/latest.pkl').read_bytes())
    assert created['state']==original['state']
    assert created['progress']['warmup_ready_adds_total']==700
    assert original['progress']['warmup_ready_adds_total']==600
    metadata=json.loads((branch/'seed.json').read_text())
    assert metadata['replay_budget_policy']=='new_arrivals'
    assert metadata['replay_boundary']['adds_total']==700
    saved=yaml.safe_load(target.read_text());runtime=saved['runtime']
    assert runtime['learner_service']['checkpoint_dir']==str(branch/'checkpoints')
    assert runtime['actor_service']['snapshot_path']==str(branch/'actor_snapshot/actor_snapshot.pkl')
    assert runtime['learner_service']['actor_snapshot_path']==runtime['actor_service']['snapshot_path']
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


def test_same_version_different_actor_parameters_refused(tmp_path):
    seed,config,_=fixture(tmp_path)
    path=seed/'actor_snapshot/actor_snapshot.pkl'
    value=pickle.loads(path.read_bytes());value['actor_params']['weight'][0]=99
    path.write_bytes(pickle.dumps(value))
    with pytest.raises(ValueError,match='identity_mismatch'):
        prepare(seed,tmp_path/'new',config,tmp_path/'cfg',tmp_path/'run')
    assert not (tmp_path/'new').exists()


def test_partial_replay_tail_refused_without_creating_assets(tmp_path):
    seed,config,replay=fixture(tmp_path)
    with replay.open('ab') as stream:stream.write(pickle.dumps(dict(episode_id=999))[:-2])
    with pytest.raises(ValueError,match='replay_incomplete'):
        prepare(seed,tmp_path/'new',config,tmp_path/'cfg',tmp_path/'run')
    assert not (tmp_path/'new').exists()


def test_historical_catchup_requires_explicit_inherit_policy(tmp_path):
    seed,config,_=fixture(tmp_path)
    branch=tmp_path/'new'
    prepare(seed,branch,config,tmp_path/'cfg',tmp_path/'run',replay_budget_policy='inherit')
    created=pickle.loads((branch/'checkpoints/latest.pkl').read_bytes())
    assert created['progress']['warmup_ready_adds_total']==600
    assert json.loads((branch/'seed.json').read_text())['replay_budget_policy']=='inherit'


def test_legacy_snapshot_step_requires_exact_full_checkpoint_parameter_match(tmp_path):
    seed,config,_=fixture(tmp_path)
    path=seed/'actor_snapshot/actor_snapshot.pkl'
    actor=pickle.loads(path.read_bytes());actor.pop('global_step')
    path.write_bytes(pickle.dumps(actor))
    branch=tmp_path/'new'
    prepare(seed,branch,config,tmp_path/'cfg',tmp_path/'run')
    assert json.loads((branch/'seed.json').read_text())['actor_step_source']=='parameter_matched_full_checkpoint'
    actor['actor_params']['weight'][0]=99;path.write_bytes(pickle.dumps(actor))
    with pytest.raises(ValueError,match='identity_mismatch'):
        prepare(seed,tmp_path/'bad',config,tmp_path/'cfg2',tmp_path/'run2')


def test_staged_learning_separates_served_actor_from_candidate_exports(tmp_path):
    seed,config,replay=fixture(tmp_path)
    branch=tmp_path/'models/online_from_5000/staged';target=tmp_path/'online.yaml'
    prepare(seed,branch,config,target,tmp_path/'run',publication_policy='staged')
    saved=yaml.safe_load(target.read_text());runtime=saved['runtime']
    served=Path(runtime['actor_service']['snapshot_path'])
    pending=Path(runtime['learner_service']['actor_snapshot_path'])
    assert served!=pending and served.is_file() and not pending.exists()
    before=served.read_bytes();pending.parent.mkdir(parents=True)
    pending.write_bytes(pickle.dumps({'version':9999}))
    assert served.read_bytes()==before
    metadata=json.loads((branch/'seed.json').read_text())
    assert metadata['publication_policy']=='staged'
    assert latest_branch(tmp_path/'models')['checkpoint']==str(served)
    assert runtime['replay']['journal_path']==str(replay)


def test_invalid_publication_policy_creates_no_assets(tmp_path):
    seed,config,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='publication_policy_invalid'):
        prepare(seed,tmp_path/'new',config,tmp_path/'cfg',tmp_path/'run',publication_policy='unverified')
    assert not (tmp_path/'new').exists()


"""Actual launcher seed block -> native CLI, no service/robot startup."""
import os
import subprocess
import sys


def launch_seed_block(seed, source_config, branch, target, run, **choices):
    root=Path(__file__).resolve().parents[1]
    text=(root/'scripts/rlt_up.sh').read_text()
    marker='if [[ "$MODE" == online && -n "${COBOT_RLT_BRANCH_CONFIG:-}" ]]; then'
    block=text[text.index(marker):text.index('# Separate physical publication')]
    env=os.environ.copy()
    for key in ('COBOT_RLT_SEED_PUBLICATION_POLICY','COBOT_RLT_SEED_REPLAY_BUDGET_POLICY'):
        env.pop(key,None)
    env.update(MODE='online',ONLINE_PY=sys.executable,RLT_CONFIG=str(source_config),
        COBOT_RLT_ONLINE_SEED=str(seed),COBOT_RLT_SEED_DESTINATION=str(branch),
        COBOT_RLT_BRANCH_CONFIG=str(target),COBOT_RLT_BRANCH_RUN=str(run),
        CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONPATH=str(root))
    env.update(choices)
    return subprocess.run(['bash','-eu','-c',block],env=env,cwd=root,
                          capture_output=True,text=True,timeout=30)


@pytest.mark.parametrize('publication,budget,anchor',[
    (None,None,700),('staged','new_arrivals',700),('staged','inherit',600)])
def test_actual_launch_seed_block_forwards_choices_and_preserves_seed(tmp_path,publication,budget,anchor):
    seed,config,replay=fixture(tmp_path)
    before={p:p.read_bytes() for p in seed.rglob('*') if p.is_file()}
    replay_before=replay.read_bytes()
    branch=tmp_path/'branch';target=tmp_path/'selected.yaml';run=tmp_path/'run'
    choices={}
    if publication is not None:choices['COBOT_RLT_SEED_PUBLICATION_POLICY']=publication
    if budget is not None:choices['COBOT_RLT_SEED_REPLAY_BUDGET_POLICY']=budget
    result=launch_seed_block(seed,config,branch,target,run,**choices)
    assert result.returncode==0,result.stderr
    metadata=json.loads((branch/'seed.json').read_text())
    assert metadata['publication_policy']==(publication or 'automatic')
    assert metadata['branch_warmup_ready_adds_total']==anchor
    saved=yaml.safe_load(target.read_text())['runtime']
    assert (saved['actor_service']['snapshot_path']!=saved['learner_service']['actor_snapshot_path'])==(publication=='staged')
    source=pickle.loads((seed/'checkpoints/latest.pkl').read_bytes())
    created=pickle.loads((branch/'checkpoints/latest.pkl').read_bytes())
    assert created['state']==source['state']
    assert all(p.read_bytes()==v for p,v in before.items()) and replay.read_bytes()==replay_before
    after={p:p.read_bytes() for p in branch.rglob('*') if p.is_file()}
    config_after=target.read_bytes()
    resumed=launch_seed_block(seed,config,branch,target,run,**choices)
    assert resumed.returncode==0,resumed.stderr
    assert all(p.read_bytes()==v for p,v in after.items()) and target.read_bytes()==config_after


@pytest.mark.parametrize('field,bad',[
    ('COBOT_RLT_SEED_PUBLICATION_POLICY','unverified'),
    ('COBOT_RLT_SEED_REPLAY_BUDGET_POLICY','unknown')])
def test_actual_launch_rejects_invalid_explicit_policy_before_branch_creation(tmp_path,field,bad):
    seed,config,_=fixture(tmp_path)
    branch=tmp_path/'branch';target=tmp_path/'selected.yaml'
    result=launch_seed_block(seed,config,branch,target,tmp_path/'run',**{field:bad})
    assert result.returncode!=0 and not branch.exists() and not target.exists()


@pytest.mark.parametrize('field,initial,requested',[
    ('publication','automatic','staged'),('publication','staged','automatic'),
    ('budget','new_arrivals','inherit'),('budget','inherit','new_arrivals')])
def test_actual_launch_refuses_reused_branch_policy_mismatch_without_mutating_it(tmp_path,field,initial,requested):
    seed,config,_=fixture(tmp_path)
    branch=tmp_path/'branch';target=tmp_path/'selected.yaml';run=tmp_path/'run'
    kwargs={'publication_policy':initial} if field=='publication' else {'replay_budget_policy':initial}
    prepare(seed,branch,config,target,run,**kwargs)
    before={p:p.read_bytes() for p in branch.rglob('*') if p.is_file()};config_before=target.read_bytes()
    key='COBOT_RLT_SEED_PUBLICATION_POLICY' if field=='publication' else 'COBOT_RLT_SEED_REPLAY_BUDGET_POLICY'
    result=launch_seed_block(seed,config,branch,target,run,**{key:requested})
    assert result.returncode!=0 and 'does not match explicit request' in result.stderr
    assert all(p.read_bytes()==v for p,v in before.items()) and target.read_bytes()==config_before


def test_actual_launch_refuses_tampered_staged_paths(tmp_path):
    seed,config,_=fixture(tmp_path)
    branch=tmp_path/'branch';target=tmp_path/'selected.yaml';run=tmp_path/'run'
    prepare(seed,branch,config,target,run,publication_policy='staged')
    data=yaml.safe_load(target.read_text());runtime=data['runtime']
    runtime['learner_service']['actor_snapshot_path']=runtime['actor_service']['snapshot_path']
    target.write_text(yaml.safe_dump(data));before=target.read_bytes()
    result=launch_seed_block(seed,config,branch,target,run,COBOT_RLT_SEED_PUBLICATION_POLICY='staged')
    assert result.returncode!=0 and 'Actor paths do not match seed metadata' in result.stderr
    assert target.read_bytes()==before
