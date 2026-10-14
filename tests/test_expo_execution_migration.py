import importlib.util,hashlib,json
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('migration',Path(__file__).parents[1]/'scripts/migrate_expo_device_layout.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixture(tmp_path,monkeypatch):
 root=tmp_path/'run';root.mkdir();prev=tmp_path/'previous';prev.mkdir()
 current=Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'
 for name in m.FILES:(prev/name).write_text('previous '+name)
 contract={'recipe':'native_sync','update_layout':'native_full_batch','learner_only':True,'edit_scale':.01,'adapter_source_sha256':m.digest(prev)}
 (root/'contract.json').write_text(json.dumps(contract))
 (root/'status.json').write_text(json.dumps({'phase':'ready','pid':1234,'actor_version':5}))
 meta={'checkpoint':1,'updates':5,'accepted':['ep'],'steps':162,'credit':12}
 (root/'committed.json').write_text(json.dumps(meta))
 d=root/'episodes/ep';d.mkdir(parents=True);(d/'finished.json').write_text('{}')
 d=root/'checkpoints/1';d.mkdir(parents=True);(d/'state.bin').write_bytes(b'optimizer-rng-target')
 record={'file':'state.bin','sha256':hashlib.sha256((d/'state.bin').read_bytes()).hexdigest()}
 (d/'manifest.json').write_text(json.dumps({'records':[record]}))
 def absent(*args):raise ProcessLookupError()
 monkeypatch.setattr(m.os,'kill',absent)
 return root,prev,current,contract

def test_migration_changes_only_adapter_identity_and_keeps_rollback(tmp_path,monkeypatch):
 root,prev,current,old=fixture(tmp_path,monkeypatch)
 state=(root/'committed.json').read_bytes();result=m.migrate(root,prev,current)
 assert json.loads(Path(result['backup']).read_text())==old
 new=json.loads((root/'contract.json').read_text())
 assert new=={**old,'adapter_source_sha256':m.digest(current,True)}
 assert (root/'committed.json').read_bytes()==state

@pytest.mark.parametrize('failure',['live','pending','corrupt','identity','updating'])
def test_migration_refuses_unsafe_or_unverified_state(tmp_path,monkeypatch,failure):
 root,prev,current,old=fixture(tmp_path,monkeypatch)
 if failure=='live':monkeypatch.setattr(m.os,'kill',lambda *args:None)
 if failure=='pending':
  p=root/'episodes/pending';p.mkdir();(p/'finished.json').write_text('{}')
 if failure=='corrupt':(root/'checkpoints/1/state.bin').write_bytes(b'bad')
 if failure=='identity':(prev/m.FILES[0]).write_text('changed')
 if failure=='updating':(root/'status.json').write_text(json.dumps({'phase':'updating'}))
 with pytest.raises(ValueError):m.migrate(root,prev,current)
 assert json.loads((root/'contract.json').read_text())==old
