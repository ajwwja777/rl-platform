import importlib.util
import json
from pathlib import Path
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('runtime_fork',ROOT/'scripts/fork_supported_runtime.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def source(tmp_path):
    root=tmp_path/'candidates/source';root.mkdir(parents=True)
    for name in module.FILES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
    pending=root/'pending_actor/actor_snapshot.pkl';pending.parent.mkdir();pending.write_bytes(b'pending actor distinct from served')
    profile={'schema':'held-gripper-hil8-clip-v2','publication_policy':'staged'}
    for k,n in [('teacher','teacher.pkl'),('normalization','action_norm_stats.json')]:profile[k]={'file':n,'sha256':module.sha(root/n)}
    (root/'profile.json').write_text(json.dumps(profile))
    cfg={'experiment':{'rl':{'action_norm_stats_path':str(root/'action_norm_stats.json')}},'runtime':{'actor_service':{'snapshot_path':str(root/'actor_snapshot/actor_snapshot.pkl')},'learner_service':{'actor_snapshot_path':str(pending),'checkpoint_dir':str(root/'checkpoints')},'replay':{'journal_path':str(root/'replay/replay_journal.pkl')}}}
    (root/'online.yaml').write_text(yaml.safe_dump(cfg));return root

def test_fork_preserves_pending_identity_and_relocates_paths(tmp_path):
    src=source(tmp_path);dst=tmp_path/'candidates/run'
    before={str(p.relative_to(src)):module.sha(p) for p in src.rglob('*') if p.is_file()}
    receipt=module.fork(src,dst)
    assert (dst/'pending_actor/actor_snapshot.pkl').read_bytes()==b'pending actor distinct from served'
    assert all(module.sha(dst/n)==h for n,h in receipt['copied_sha256'].items())
    assert str(src) not in (dst/'online.yaml').read_text()
    assert yaml.safe_load((dst/'online.yaml').read_text())['runtime']['learner_service']['checkpoint_dir']==str(dst/'checkpoints')
    assert before=={str(p.relative_to(src)):module.sha(p) for p in src.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):module.fork(src,dst)

def test_missing_pending_rejected_before_any_destination_write(tmp_path):
    src=source(tmp_path);(src/'pending_actor/actor_snapshot.pkl').unlink();dst=tmp_path/'candidates/run'
    with pytest.raises(FileNotFoundError):module.fork(src,dst)
    assert not dst.exists()

def test_teacher5_registry_distinguishes_modes_and_keeps_staged_paths():
    rows=json.loads((ROOT/'configs/deployment_models.json').read_text())['models'];by={r['id']:r for r in rows}
    frozen=by['plug-v3-teacher5-8127'];online=by['plug-v3-teacher5-online']
    assert frozen['mode']=='frozen' and not frozen['capabilities']['train']
    assert online['mode']=='online' and online['capabilities']['train'] and online['evaluation_allowed'] is False
    assert frozen['checkpoint']!=online['checkpoint']
    assert all(r['step']==8127 and r['actor_version']==4063 and r['execution_profile']=='async_rtc50' for r in [frozen,online])
    assert online['collection_directory_policy']=='model' and 'teacher5-8127-online-v1' in online['data_directories']['collection']
    assert by['plug-v3-hil8-online']['step']==7281
    assert online['training_method']!=by['plug-v3-hil8-online']['training_method']
