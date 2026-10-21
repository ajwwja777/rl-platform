import importlib.util,json,sys
from pathlib import Path
S=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(S))
import expo_metrics_observer as observer

def test_version_stages_and_history_survive_same_training_step(tmp_path):
 store=observer.Store(tmp_path/'metrics.sqlite')
 status={'model':{'id':'expo','runtime_adapter':'expo_online','online_run_dir':'/test/run'},'online':{'deployment_id':'expo','pid':4,'learner_step':102,'critic_updates':2040,'actor_updates':102,'committed_learner_step':102,'published_actor_version':98,'served_actor_version':89,'downloaded_actor_version':89,'download':{'state':'retrying','actor_version':98},'updated_at':10}}
 store.ingest(status)
 assert store.payload()['runs'][0]['progress']['committed_learner_step']==102
 status['online'].update(served_actor_version=98,downloaded_actor_version=98)
 store.ingest(status);store.ingest(status)
 events=store.payload()['runs'][0]['version_events']
 assert len([e for e in events if e['stage']=='served' and e['version']==98])==1
 assert any(e['version']==89 and e['stage']=='served' for e in events)

def test_transfer_partial_is_not_downloaded_and_wrong_manifest_rejected(tmp_path):
 from expo_progress import transfer_bytes
 d=tmp_path/'98';d.mkdir();(d/'actor-0.gz').write_bytes(b'a'*4);(d/'edit-0.download').write_bytes(b'b'*3)
 manifest={'actor_version':98,'records':[{'file':'actor-0.gz','bytes':4},{'file':'edit-0.gz','bytes':6}]}
 p=transfer_bytes(d,98,manifest);assert p['bytes_received']==7 and p['bytes_total']==10 and p['percent']==70
 assert not p['verified_complete']
 manifest['actor_version']=99;p=transfer_bytes(d,98,manifest);assert p['bytes_total'] is None and p['percent'] is None

def test_sampling_reports_recipe_not_invented_draws():
 from expo_progress import sampling_summary
 p=sampling_summary({'async_mode':'native_async_v1','batch_size':64,'utd_ratio':20,'seed_steps':3301,'online_steps':5659,'replay_steps':8960})
 assert p['critic_draws_per_group']==1280 and p['actor_batch_size']==64
 assert p['actual_sample_composition'] is None
 assert p['seed_fraction']==3301/8960
 assert sampling_summary({'recipe':'other'})['critic_draws_per_group'] is None

def test_probe_uses_selected_option_directory(tmp_path):
 from expo_progress import ProgressProbe
 default=tmp_path/'online';selected=tmp_path/'online--d3-edit005'
 for root,version in [(default,1),(selected,9)]:
  root.mkdir();(root/'installed.json').write_text(json.dumps({'actor_version':version,'installed_at':20}))
 status={'model':{'id':'expo','runtime_adapter':'expo_online','online_run_dir':str(default),'expo_online_options':{'defaults':{'rtc_delay':5,'edit_scale':.01}}},'online':{'deployment_id':'expo','rtc_delay':3,'edit_scale':.05}}
 probe=ProgressProbe(tmp_path/'cache',None)
 assert probe.enrich(status)['online']['installed_receipt']['actor_version']==9
