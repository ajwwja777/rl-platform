"""Real compact trace writer with a gated storage sink; no robot or sleeps."""
import json
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import AtomicEpisodeTraceWriter, RosTask2IO
from methods.openpi_rlt.cobot_adapter.queued_trace import QueuedEpisodeTraceWriter
from methods.openpi_rlt.cobot_adapter.trace import EpisodeOutcome


def record(value=1):
    return dict(action=np.full(7, value, np.float32), reward=0., done=False,
                observation={'state':np.zeros(7), 'images':{'camera':np.zeros((4,4,3))}},
                next_observation={'state':np.ones(7), 'images':{}}, actor_param_version=3500)


def gated(tmp_path, **kwargs):
    writer=AtomicEpisodeTraceWriter(tmp_path)
    entered, release=threading.Event(), threading.Event()
    original=writer.append
    def append(row):
        entered.set()
        assert release.wait(3), 'test did not release storage'
        original(row)
    writer.append=append
    queued=QueuedEpisodeTraceWriter(writer, **kwargs)
    return queued, entered, release


def test_blocked_storage_does_not_block_append_and_keeps_order_identity(tmp_path):
    queued, entered, release=gated(tmp_path)
    first=record(1)
    try:
        queued.append(first)
        assert entered.wait(1)
        first['action'][:]=99
        queued.append(record(2))
        assert not release.is_set() and queued.diagnostics()['pending']==2
        release.set()
        queued.finalize('success',identity={'episode_uuid':'one'})
        rows=[json.loads(x)for x in next(tmp_path.glob('*_success.jsonl')).read_text().splitlines()]
        assert [r['action'][0]for r in rows]==[1,2]
        assert 'images'not in rows[0]['observation']
        assert rows[-1]['done'] and rows[-1]['episode_uuid']=='one'
        assert rows[0]['actor_param_version']==3500
    finally:
        release.set();queued.close()


def test_overflow_is_bounded_latched_and_not_silently_dropped(tmp_path):
    queued, entered, release=gated(tmp_path,capacity=2)
    queued.append(record());assert entered.wait(1);queued.append(record(2))
    try:
        with pytest.raises(RuntimeError,match='queue_full'):queued.append(record(3))
        assert queued.diagnostics()['accepted']==2
        with pytest.raises(RuntimeError,match='storage_failed'):queued.finalize('success')
    finally:
        release.set()
        with pytest.raises(RuntimeError):queued.close()


def test_storage_failure_blocks_real_replay_gate(tmp_path):
    writer=AtomicEpisodeTraceWriter(tmp_path)
    failed=threading.Event()
    def append(row):
        failed.set();raise OSError('disk unavailable')
    writer.append=append
    queued=QueuedEpisodeTraceWriter(writer)
    io=object.__new__(RosTask2IO);io._trace_writer=queued
    from methods.openpi_rlt.cobot_adapter.cobot_online_env import CobotOnlineEnv
    env=object.__new__(CobotOnlineEnv);env._io=io;env._shadow_mode=False;env._last_outcome=EpisodeOutcome.SUCCESS
    queued.append(record());assert failed.wait(1)
    try:
        with pytest.raises(RuntimeError,match='disk unavailable'):env.replay_commit_allowed()
        assert not list(tmp_path.glob('*_success.jsonl'))
    finally:
        with pytest.raises(RuntimeError):queued.close()


def test_pending_timeout_cannot_be_marked_complete(tmp_path):
    queued, entered, release=gated(tmp_path)
    queued.append(record());assert entered.wait(1)
    try:
        with pytest.raises(RuntimeError,match='flush_timeout'):queued.flush(timeout=.01)
        assert queued.diagnostics()['written']==0
        assert not list(tmp_path.glob('*_success.jsonl'))
    finally:
        release.set();queued.close()


def test_episode_boundaries_and_collection_wrapper_keep_identity(tmp_path):
    from integrations.cobot_runtime.shared_model_env import CollectionTrace
    writer=AtomicEpisodeTraceWriter(tmp_path)
    queued=QueuedEpisodeTraceWriter(writer)
    lifecycle=SimpleNamespace(collecting=True)
    wrapper=CollectionTrace(queued,lifecycle)
    io=object.__new__(RosTask2IO);io._trace_writer=wrapper
    try:
        wrapper.start_episode();wrapper.append(record(1));io.flush_raw_trace()
        wrapper.finalize('failure',identity={'episode_uuid':'first'})
        wrapper.start_episode();wrapper.append(record(2));wrapper.finalize('success',identity={'episode_uuid':'second'})
        assert json.loads(next(tmp_path.glob('*_failure.jsonl')).read_text())['episode_uuid']=='first'
        assert json.loads(next(tmp_path.glob('*_success.jsonl')).read_text())['action'][0]==2
        assert wrapper._root==tmp_path
        lifecycle.collecting=False
        wrapper.append(record(3));io.flush_raw_trace()
        assert io.trace_diagnostics()=={}
        assert queued.diagnostics()['accepted']==2
    finally:queued.close()


def test_age_guard_does_not_wait_for_stalled_storage(tmp_path):
    now=[0.]
    queued, entered, release=gated(tmp_path,clock=lambda:now[0],max_pending_age=.5)
    queued.append(record());assert entered.wait(1);now[0]=.6
    try:
        with pytest.raises(RuntimeError,match='storage_backlog'):queued.check_health()
        with pytest.raises(RuntimeError,match='storage_failed'):queued.flush()
    finally:
        release.set()
        with pytest.raises(RuntimeError):queued.close()


def test_real_executor_can_publish_while_storage_is_blocked(monkeypatch,tmp_path):
    from test_async_execution import setup
    engine, env, fake, clock=setup(monkeypatch,50)
    queued, entered, release=gated(tmp_path)
    fake.record_raw_step=queued.append
    fake.check_trace_health=queued.check_health
    try:
        _,rewards,done,info=env.execute_chunk(fake.sample().observation)
        assert not done and len(rewards)==10 and len(fake.published)==25
        assert entered.wait(1) and not release.is_set()
        assert queued.diagnostics()['accepted']==10
        times=np.array([t for t,_ in fake.published]);assert np.diff(times).min()>=.02-1e-9
    finally:
        release.set();queued.close();engine.close()


def test_actual_io_is_switched_when_async_executor_is_constructed(monkeypatch,tmp_path):
    from methods.openpi_rlt.cobot_adapter.async_execution import AsyncExecution
    from methods.openpi_rlt.cobot_adapter.execution_profiles import selected_profile
    io=object.__new__(RosTask2IO);io._trace_writer=AtomicEpisodeTraceWriter(tmp_path)
    monkeypatch.setenv('COBOT_RLT_EXECUTION_PROFILE','async_rtc50')
    env=SimpleNamespace(_io=io)
    engine=AsyncExecution(env,*selected_profile())
    try:
        assert isinstance(io._trace_writer,QueuedEpisodeTraceWriter)
        io.enable_async_trace()
        assert not isinstance(io._trace_writer.writer,QueuedEpisodeTraceWriter)
    finally:
        io._trace_writer.close();engine.close()


def test_fault_reports_last_loop_costs_and_flushes_only_after_pause(monkeypatch,capsys):
    from test_async_execution import setup
    engine,env,io,clock=setup(monkeypatch,50)
    def slow_record(row):clock.value+=.154
    io.record_raw_step=slow_record
    def flush(timeout):
        assert io.pauses[-1] is True
        raise RuntimeError('diagnostic flush blocked')
    io.flush_raw_trace=flush
    io.trace_diagnostics=lambda:{'pending':1}
    try:
        with pytest.raises(RuntimeError,match='deadline'):env.execute_chunk(io.sample().observation)
        text=next(x for x in capsys.readouterr().out.splitlines()if x.startswith('[rlt-execution-fault] '))
        fault=json.loads(text.split(' ',1)[1])
        assert fault['execution_stats']['last_record_step_ms']==pytest.approx(154)
        assert fault['trace_storage']=={'pending':1}
        assert fault['trace_flush_error']=='diagnostic flush blocked'
        assert fault['attempt']['published'] is False
    finally:engine.close()
