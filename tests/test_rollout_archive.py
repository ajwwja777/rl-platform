import runpy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from integrations.cobot_runtime.rollout_archive import save_snapshot, history, capture_current


def snapshot(uid, times):
    return dict(episode_uuid=uid,actor_version=66,points=[dict(seconds=t,timestamp=100+t,q=t/10) for t in times])


def test_history_retains_rounds_and_merges_rolling_windows(tmp_path):
    save_snapshot(tmp_path,snapshot('one',[1,2]))
    save_snapshot(tmp_path,snapshot('one',[2,3]))
    save_snapshot(tmp_path,snapshot('two',[4]))
    rows,old=history(tmp_path,'one')
    assert len(rows)==2 and [p['seconds'] for p in old['points']]==[1,2,3]
    assert history(tmp_path/'other','one')==([],None)
    assert history(tmp_path,"../../etc/passwd")[1] is None


def test_concurrent_archive_writers_do_not_lose_points(tmp_path):
    save_snapshot(tmp_path,snapshot('one',[0]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i:save_snapshot(tmp_path,snapshot('one',[i])),range(1,20)))
    assert len(history(tmp_path,'one')[1]['points'])==20


def test_expo_boundary_flush_archives_prior_round_even_before_periodic_flush(tmp_path):
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter/monitoring.py'))
    h=api['RolloutHistory'](tmp_path)
    h.selected(dict(episode_uuid='one',started_at=1),dict(actor_version=1,selected_q=.1))
    h.selected(dict(episode_uuid='two',started_at=2),dict(actor_version=2,selected_q=.2))
    h.flush()
    rows,old=history(tmp_path,'one')
    assert len(rows)==2 and old['points'][0]['q']==.1
    capture_current(tmp_path)
    assert len(history(tmp_path)[0])==2
