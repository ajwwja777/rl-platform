import json
import queue
import time
from pathlib import Path
import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter import monitoring as m
from methods.openpi_rlt.cobot_adapter.q_monitor import AsyncQMonitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_native_metrics_online_only_deduplicated_restoration_and_partial_line(tmp_path):
    write(tmp_path/'metrics/learner_status.json', dict(global_step=7002, warmup_required_updates=7000, timestamp=time.time()))
    rows = [dict(global_step=s, q1_mean=.1, q2_mean=.9, actor_q=0, target_q_mean=.2, critic_loss=.03, timestamp=s) for s in (6999,7000,7001,7001,7002,7003)]
    (tmp_path/'metrics/learner_metrics.jsonl').write_text('\n'.join(map(json.dumps, rows))+'\n{"partial":')
    history=m.read_history(tmp_path)
    assert [p['critic_step'] for p in history['training']['points']]==[7001,7002]
    assert history['training']['points'][0]['q']==.1
    assert history['training']['points'][0]['critic_loss']==.03
    assert history['learner']['running']
    write(tmp_path/'metrics/learner_status.json',dict(global_step=7002,timestamp=1))
    assert not m.read_history(tmp_path)['learner']['running']
    assert m.read_history(tmp_path)['training']['points']


def test_success_only_online_collection_and_actual_hil():
    base=dict(outcome='success',count=10,hil_steps=0,collection_phase='online',session_use='collection')
    rows=[dict(base,episode_uuid=str(i),finished_at=i) for i in range(25)]
    rows[-1]['hil_steps']=1
    rows += [dict(base,episode_uuid='unlabeled',outcome='saved'),dict(base,episode_uuid='unknown',hil_steps=None),dict(base,episode_uuid='warm',collection_phase='warmup'),dict(base,episode_uuid='eval',session_use='evaluation')]
    result=m.success_series(rows)
    assert result['points'][-1]['rate']==95
    assert result['points'][-1]['total']==20
    assert result['points'][-1]['assisted']==1
    assert result['excluded']==2 and result['completed']==27


def test_monitor_queue_overflow_never_waits_and_does_not_mutate_plan():
    from types import SimpleNamespace as S
    monitor=AsyncQMonitor.__new__(AsyncQMonitor)
    monitor.events=queue.Queue(maxsize=1);monitor.enabled=True;monitor.identity='one';monitor.dropped=0
    actions=np.ones((10,7),np.float32)
    plan=S(start_features=S(z_rl=np.ones(4),proprio=np.ones(7)),action_chunk=actions,source=1,actor_param_version=8)
    monitor.adopted(plan);monitor.adopted(plan)
    assert monitor.dropped==1
    event=monitor.events.get_nowait();actions[:]=2
    assert np.all(event['actions']==1)
    assert event['actor_version']==8 and event['episode_uuid']=='one'


def test_native_cpu_scoring_pins_checkpoint_and_normalization(tmp_path,monkeypatch):
    import sys,pickle,yaml
    upstream=Path(__file__).resolve().parents[1]/'third_party/openpi-rlt/rlt_online_rl/src'
    monkeypatch.syspath_prepend(str(upstream))
    import jax
    import jax.numpy as jnp
    from rlt_online_rl.networks import TwinCritic
    from methods.openpi_rlt.cobot_adapter.q_monitor import CheckpointScorer
    critic=TwinCritic(4,7,10,7,16,1)
    params=jax.tree.map(np.asarray,critic.init_params(jax.random.PRNGKey(42)))
    checkpoint=tmp_path/'checkpoints';checkpoint.mkdir()
    norm=tmp_path/'norm.json'
    norm.write_text(json.dumps({'norm_stats':{'actions':{'q01':[-1]*7,'q99':[1]*7}}}))
    cfg={'experiment':{'rl':dict(z_dim=4,critic_hidden_dim=16,critic_num_layers=1,action_representation='delta_chunk',action_norm_stats_path=str(norm))},'runtime':{'learner_service':{'checkpoint_dir':str(checkpoint)}}}
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    def save(step):
        with (checkpoint/'latest.pkl').open('wb') as f:pickle.dump(dict(state=dict(global_step=step,critic_params=params)),f)
    save(7);scorer=CheckpointScorer(str(config));save(9)
    assert scorer.version==7
    event=dict(z=np.ones(4,np.float32),proprio=np.ones(7,np.float32)*.2,actions=np.ones((10,7),np.float32)*.4)
    normalized=scorer.adapter.normalize_chunk(event['actions'],event['proprio'])
    expected=critic.q_values(params,jnp.asarray(event['z'][None]),jnp.asarray(event['proprio'][None]),jnp.asarray(normalized[None]))[0]
    assert scorer.score(event)==pytest.approx(float(expected[0]),abs=1e-6)
    assert CheckpointScorer(str(config)).version==9
    # Exercise the actual spawned CPU process, including episode boundaries.
    from types import SimpleNamespace as S
    monitor=AsyncQMonitor(tmp_path/'run',config)
    try:
        monitor.begin(dict(session_use='collection',task5_episode_uuid='one'),'online')
        path=tmp_path/'run/telemetry/rollout.json'
        deadline=time.monotonic()+20
        while time.monotonic()<deadline and not m.read_json(path).get('critic_version'):
            time.sleep(.05)
        assert m.read_json(path)['critic_version']==9
        save(11)
        plan=S(start_features=S(z_rl=event['z'],proprio=event['proprio']),action_chunk=event['actions'],source=1,actor_param_version=8)
        monitor.adopted(plan)
        monitor.send('gap',reason='hil')
        monitor.adopted(plan)
        monitor.end(dict(outcome='success'),dict(rl_steps=10,human_steps=1,mixed_steps=0,actor_version_end=8))
        while time.monotonic()<deadline and m.read_json(path).get('active',True):
            time.sleep(.05)
        result=m.read_json(path)
        assert result['critic_version']==9 and not result['active']
        assert [p['q'] is None for p in result['points']]==[False,True,False]
        assert result['points'][0]['q']==pytest.approx(float(expected[0]),abs=1e-6)
        success=m.read_history(tmp_path/'run')['success']
        assert success['points'][0]['rate']==0 and success['points'][0]['assisted']==1
        monitor.begin(dict(session_use='evaluation',task5_episode_uuid='eval'),'online')
        assert not monitor.enabled
    finally:
        monitor.close()

    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps(dict(schema='held-gripper-hil8-clip-v2',held_gripper=True)))
    monkeypatch.setenv('COBOT_RLT_SUPPORTED_PROFILE',str(profile))
    held=CheckpointScorer(str(config))
    projected=held.critic.project(jnp.asarray(normalized[None]),jnp.asarray(event['proprio'][None]))
    expected_held=critic.q_values(params,jnp.asarray(event['z'][None]),jnp.asarray(event['proprio'][None]),projected)[0]
    assert held.score(event)==pytest.approx(float(expected_held[0]),abs=1e-6)
