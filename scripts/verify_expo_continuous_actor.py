"""Isolated, no-ROS GPU probe of native continuous inference and background staging.

Snapshot inputs are read-only. Receipts and the JSON report live beside --output.
Timing includes local Actor calls and synchronization, but excludes ROS, HTTP,
robot scheduling and learner/network load. Deadline comparisons are descriptive.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

import numpy as np


def summary(rows, deadline):
    values = np.asarray([row['total_seconds'] for row in rows], dtype=float)
    return dict(count=len(rows), maximum_seconds=float(values.max()) if len(values) else None,
                p50_seconds=float(np.percentile(values, 50)) if len(values) else None,
                p95_seconds=float(np.percentile(values, 95)) if len(values) else None,
                above_d5_reference=int(sum(value > deadline for value in values)))


def qualify(args, result):
    from continuous_actor import ContinuousActor
    actor_args = SimpleNamespace(project=args.project, bundle=args.bundle, snapshot_root=args.snapshots,
        snapshot=args.snapshots/str(args.initial), model_id='expo-continuous-isolated-cobot-probe',
        rtc_delay=5, edit_scale=.01)
    started = time.monotonic()
    actor = ContinuousActor(actor_args)
    result['load_and_warmup_seconds'] = time.monotonic() - started
    result['initial_status'] = actor.status()
    assert actor.meta['updates'] == args.initial
    # This probe never writes next to the supplied snapshots or a live run journal.
    receipt_root = args.output.parent / ('actor-probe-receipts-' + result['probe_uuid'])
    actor._receipt = lambda identity: receipt_root/(identity + '.json')
    result['receipt_directory'] = str(receipt_root)
    warm_calls = []
    def forbid_warm():
        warm_calls.append(time.monotonic())
        raise AssertionError('Staging attempted inference warmup')
    actor.warm_inference = forbid_warm
    with np.load(args.bundle.parent/'common/dev_observation.npz', allow_pickle=False) as archive:
        observation = {key: archive[key].copy() for key in ('state', 'left_wrist_image', 'right_wrist_image')}
    identity = str(uuid.uuid4())
    envelope = {**observation, 'episode_uuid': np.asarray(identity)}
    with actor.lock:
        actor.begin({'episode_uuid': identity})
    prefix = np.zeros((0, 7), np.float32)
    origin = time.monotonic()

    def invoke(route, payload):
        # Same inference mutex as HTTP. Record transient busy periods separately.
        before = time.monotonic()
        busy = 0
        while not actor.lock.acquire(blocking=False):
            busy += 1
            if time.monotonic() - before > 5:
                raise TimeoutError('Actor inference mutex unavailable for five seconds')
            time.sleep(.001)
        acquired = time.monotonic()
        try:
            value = actor.infer(route, payload)
        finally:
            actor.lock.release()
        return value, time.monotonic() - acquired, acquired - before, busy

    def precache():
        return invoke('/precache', {**envelope, 'prefix': prefix})

    def select(cached):
        return invoke('/select', {**envelope, 'precached': np.asarray(cached['precached'], np.float32),
            'delay': np.asarray(cached['delay']), 'actor_version': np.asarray(cached['actor_version']),
            'inference_token': np.asarray(cached['inference_token'])})

    def sample(window):
        nonlocal prefix
        before = time.monotonic()
        phase_before = actor.stage_phase
        rng_before = np.asarray(actor.agent.rng).copy()
        cached, pre_seconds, pre_lock, pre_busy = precache()
        np.testing.assert_array_equal(actor.agent.rng, rng_before)
        selected, select_seconds, select_lock, select_busy = select(cached)
        actions = np.asarray(selected['actions'], np.float32)
        assert actions.shape == (8, 7) and np.isfinite(actions).all()
        assert not np.array_equal(np.asarray(actor.agent.rng), rng_before), 'Selection did not advance RNG'
        assert os.getpid() == result['pid'], 'Actor process changed'
        prefix = actions[-5:].copy()
        row = dict(window=window, seconds_since_probe_start=before-origin,
            stage_phase_before=phase_before, stage_phase_after=actor.stage_phase,
            precache_actor_version=cached['actor_version'], actor_version=selected['actor_version'],
            delay=cached['delay'], precache_seconds=pre_seconds, select_seconds=select_seconds,
            lock_wait_seconds=pre_lock+select_lock, busy_retries=pre_busy+select_busy,
            total_seconds=time.monotonic()-before)
        result['samples'].append(row)
        return row

    for _ in range(args.samples):
        sample('before')
    rng_before = np.asarray(actor.agent.rng).copy()
    canceled, *_ = precache()
    np.testing.assert_array_equal(actor.agent.rng, rng_before)
    sample('before')  # A newly selected token retires the unselected transaction.
    committed_rng = np.asarray(actor.agent.rng).copy()
    try:
        select(canceled)
    except ValueError as exc:
        assert 'token' in str(exc)
    else:
        raise AssertionError('Canceled/retired precache token was accepted')
    np.testing.assert_array_equal(actor.agent.rng, committed_rng)
    result['canceled_precache_preserves_rng'] = True
    result['retired_token_rejected'] = True

    stage_started = time.monotonic()
    with actor.lock:
        actor.stage(args.snapshots/str(args.updated))
    stage_count = 0
    while actor.meta['updates'] != args.updated or stage_count < args.samples:
        if time.monotonic() - stage_started > args.stage_timeout:
            raise TimeoutError('Background staging/adoption exceeded probe timeout')
        if actor.stage_phase == 'stage_failed':
            raise RuntimeError('Background stage failed: ' + str(actor.stage_error))
        sample('stage_window')
        stage_count += 1
        if actor.meta['updates'] == args.updated and 'stage_to_adoption_seconds' not in result:
            result['stage_to_adoption_seconds'] = time.monotonic() - stage_started
        # Allow the staging worker its short publication mutex between requests.
        time.sleep(.001)
    actor.stage_thread.join(timeout=5)
    assert not actor.stage_thread.is_alive(), 'Staging worker did not finish'
    assert actor.active == identity and actor.phase == 'rollout'
    assert not warm_calls, 'Unexpected staging warmup'
    for _ in range(args.samples):
        sample('after')
    with actor.lock:
        receipt = actor.finish({'episode_uuid': identity})
        assert actor.finish({'episode_uuid': identity}) == receipt
    assert receipt['actor_version'] == args.initial
    assert receipt['last_actor_version'] == args.updated
    assert args.initial in receipt['policy_versions'] and args.updated in receipt['policy_versions']
    result['receipt'] = receipt

    # A second episode must retain the current live RNG, ignoring a caller's seed.
    second = str(uuid.uuid4())
    before = np.asarray(actor.agent.rng).copy()
    with actor.lock:
        actor.begin(dict(episode_uuid=second, rollout_rng=[123, 456]))
    np.testing.assert_array_equal(actor.agent.rng, before)
    envelope['episode_uuid'] = np.asarray(second)
    precache()  # Deliberately never select this candidate set.
    with actor.lock:
        second_receipt = actor.finish({'episode_uuid': second})
    np.testing.assert_array_equal(second_receipt['rollout_rng'], before)
    result.update(persistent_episode_rng=True, canceled_episode_precache_preserves_rng=True,
                  second_receipt=second_receipt, process_restart=False, stage_warmup_calls=len(warm_calls),
                  final_status=actor.status(), passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'bundle', 'snapshots', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--initial', type=int, required=True)
    parser.add_argument('--updated', type=int, required=True)
    parser.add_argument('--samples', type=int, default=24, help='Minimum pairs per before/stage-window/after group')
    parser.add_argument('--stage-timeout', type=float, default=180.)
    args = parser.parse_args()
    if not 0 <= args.initial < args.updated or not 20 <= args.samples <= 200 or args.stage_timeout <= 0:
        parser.error('Require increasing nonnegative versions, 20..200 samples and positive timeout')
    for name in ('project', 'bundle', 'snapshots', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'))
    from native_online import source_path
    from online_protocol import atomic_json
    source_path(args.project)
    result = dict(passed=False, pid=os.getpid(), probe_uuid=str(uuid.uuid4()),
        initial_actor=args.initial, updated_actor=args.updated, samples=[],
        robot_execution=False, ros_publishers=False, production_data=False,
        d5_reference_seconds=.5,
        timing_scope='In-process real GPU Actor; includes synchronization; excludes HTTP, ROS, robot scheduling and remote Learner load')
    exit_code = 0
    try:
        qualify(args, result)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        exit_code = 1
    finally:
        rows = result['samples']
        groups = {name: [row for row in rows if row['window'] == name] for name in ('before', 'stage_window', 'after')}
        groups['actually_loading'] = [row for row in rows if row['stage_phase_before'] == 'loading']
        result['latency_summary'] = {name: summary(group, .5) for name, group in groups.items()}
        result['at_least_20_pairs_while_loading'] = len(groups['actually_loading']) >= 20
        result['deadline_comparison_is_acceptance'] = False
        result['cross_version_selection_count'] = sum(row['precache_actor_version'] != row['actor_version'] for row in rows)
        atomic_json(args.output, result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
