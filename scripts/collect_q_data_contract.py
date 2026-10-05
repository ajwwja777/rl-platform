"""Read-only CPU evidence collector; no model/framework/service imports."""
import collections
import hashlib
import io
import json
from pathlib import Path
import pickle
import time

import h5py
import numpy as np

ROOT = Path('/media/agilex/Getea1/jiaan/data/datasets/plug_insertion')
EPISODES = [184, 193, 212, 215, 217, 233]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def array_digest(a):
    a = np.asarray(a)
    return digest(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes())


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    return dict(n=int(a.size), mean=float(a.mean()),
                quantiles=dict(zip(['p05', 'p50', 'p95', 'p99', 'max'],
                                  np.quantile(a, [.05, .5, .95, .99, 1]).tolist())))


def errors(a, b, mask):
    selected = np.asarray(mask, bool)
    finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    valid = selected & finite
    delta = np.abs(a[valid] - b[valid])
    return dict(selected=int(selected.sum()), finite_pairs=int(valid.sum()),
                invalid_pairs=int((selected & ~finite).sum()),
                per_dimension_mae=delta.mean(0).tolist() if len(delta) else None,
                per_dimension_p95=np.quantile(delta, .95, axis=0).tolist() if len(delta) else None,
                exact_equal_rows=int(np.all(delta == 0, axis=1).sum()))


def main():
    journal = ROOT / 'derived/rl-platform/rlt/replay_clean_v1/replay_journal.pkl'
    before = journal.stat()
    payload = journal.read_bytes()
    stream = io.BytesIO(payload)
    records = []
    historical_boundary = None
    while stream.tell() < len(payload):
        records.append(pickle.load(stream))
        if len(records) == 3917:
            historical_boundary = stream.tell()
    result = dict(observed_at=time.time(), journal=str(journal), journal_sha256=digest(payload),
                  journal_rows=len(records), historical_3917_prefix_sha256=digest(payload[:historical_boundary]),
                  clock_boundary='trace timestamp is max ROS topic stamp, not monotonic publication time',
                  episodes=[], traces=[], recordings=[], code_hashes={}, matched_hold_examples=[])
    norm_path = Path('/media/agilex/Getea1/jiaan/model/rl-platform/rlt/plug_insertion/online/action_norm_stats.json')
    norm_bytes = norm_path.read_bytes()
    norm = json.loads(norm_bytes)['norm_stats']['actions']
    scale = np.asarray(norm['q99'], np.float32)-np.asarray(norm['q01'], np.float32)+1e-6
    result['normalization'] = dict(path=str(norm_path), sha256=digest(norm_bytes), stats=norm, scale=scale.tolist())
    result['replay_dtypes'] = {k:str(np.asarray(records[0][k]).dtype) for k in ['action_chunk','ref_chunk','proprio','next_proprio']}
    for ep in EPISODES:
        rs = [r for r in records if int(r['episode_id']) == ep and r.get('collection_phase') == 'online']
        compact = []
        for r in rs:
            compact.append({k: np.asarray(r[k]).tolist() for k in
                            ['proprio', 'next_proprio', 'action_chunk', 'ref_chunk', 'source_chunk', 'rewards']}
                           | {k: int(r[k]) for k in ['step_id', 'episode_id', 'success']}
                           | {'done': bool(r['done'])})
        terminal = [r for r in rs if bool(r['done'])]
        result['episodes'].append(dict(episode_id=ep, rows=compact,
            observed_terminal_targets=[float(np.dot(np.asarray(r['rewards']), .99 ** np.arange(len(r['rewards']))))
                                       for r in terminal],
            normalization_roundtrip=None,
            per_window_td_target=None, per_window_q=None))
    replay_action_index = collections.defaultdict(list)
    for record_index, record in enumerate(records):
        replay_action_index[np.asarray(record['action_chunk'], dtype=np.float16).tobytes()].append(record_index)
    # Inspect all Online traces and retained Reference traces with interventions.
    for f in sorted((ROOT / 'derived/rl-platform/rlt/traces').rglob('*.jsonl')):
        data = f.read_bytes()
        rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
        if not rows:
            continue
        is_online = f.parent.name == 'online'
        if not is_online and not any(r.get('human_controlled') for r in rows):
            continue
        summary = dict(path=str(f), sha256=digest(data), rows=len(rows),
                       terminal_rows=sum(bool(r.get('done')) for r in rows),
                       outcomes=sorted({str(r.get('outcome')) for r in rows}),
                       first_timestamp=rows[0]['timestamp'], last_timestamp=rows[-1]['timestamp'],
                       source_groups={}, human_runs=[], replay_exact_window_matches=[])
        actions = np.stack([np.asarray(r['action'], np.float32)[-7:] for r in rows])
        for start in range(max(0, len(rows)-9)):
            indices = replay_action_index.get(actions[start:start+10].astype(np.float16).tobytes(), [])
            for record_index in indices:
                record = records[record_index]
                if not np.array_equal(record['source_chunk'], [r['source'] for r in rows[start:start+10]]):
                    continue
                summary['replay_exact_window_matches'].append(dict(
                    replay_index=record_index, episode_id=int(record['episode_id']), step_id=int(record['step_id']),
                    phase=record.get('collection_phase'), trace_row=start,
                    matching_transform='float32 raw action -> float16 Replay serialization',
                    trace_first_timestamp=rows[start]['timestamp'],
                    trace_last_timestamp=rows[start+9]['timestamp'],
                    start_state_exact=bool(np.array_equal(np.asarray(record['proprio'],np.float32),
                                             np.asarray(rows[start]['observation']['state'][-7:],np.float32))),
                    next_state_exact=bool(np.array_equal(np.asarray(record['next_proprio'],np.float32),
                                             np.asarray(rows[start+9]['next_observation']['state'][-7:],np.float32))),
                    source_chunk=np.asarray(record['source_chunk']).tolist()))
                if (int(record['source_chunk'][0]) in [2,3]
                        and np.array_equal(np.asarray(record['proprio'],np.float32),
                                           np.asarray(rows[start]['observation']['state'][-7:],np.float32))
                        and np.array_equal(actions[start],np.asarray(record['proprio'],np.float32))
                        and len(result['matched_hold_examples'])<12):
                    original=actions[start]
                    rounded=np.asarray(record['action_chunk'][0],np.float32)
                    result['matched_hold_examples'].append(dict(
                        episode_id=int(record['episode_id']), phase=record.get('collection_phase'),
                        step_id=int(record['step_id']), trace=str(f), trace_row=start,
                        proprio=original.tolist(), original_action=original.tolist(),
                        stored_action=rounded.tolist(), original_joint_delta=[0.]*6,
                        stored_joint_delta=(rounded-original)[:6].tolist(),
                        normalized_joint_delta_error=((rounded-original)*2/scale)[:6].tolist()))
        for human in [False, True]:
            selected = [r for r in rows if bool(r.get('human_controlled')) == human]
            adjacent = [(a, b) for a, b in zip(rows, rows[1:])
                        if bool(a.get('human_controlled')) == human and bool(b.get('human_controlled')) == human]
            intervals = [b['timestamp'] - a['timestamp'] for a, b in adjacent]
            positive = [x for x in intervals if 0 < x < .5]
            summary['source_groups']['human' if human else 'policy'] = dict(
                rows=len(selected), same_source_adjacent_pairs=len(adjacent),
                retained_interval_lt500ms=stats(np.asarray(positive) * 1000),
                excluded_nonpositive=sum(x <= 0 for x in intervals),
                excluded_ge500ms=sum(x >= .5 for x in intervals),
                action_exactly_next_feedback=sum(np.array_equal(r['action'], r['next_observation']['state']) for r in selected),
                action_exactly_start_feedback=sum(np.array_equal(r['action'], r['observation']['state']) for r in selected))
            if selected:
                original = np.asarray([r['action'][-7:] for r in selected], np.float32)
                restored = original.astype(np.float16).astype(np.float32)
                error = np.abs(restored-original)
                summary['source_groups']['human' if human else 'policy']['fp16_absolute_action_error'] = dict(
                    per_dimension_mae=error.mean(0).tolist(),
                    per_dimension_p95=np.quantile(error,.95,axis=0).tolist(),
                    per_dimension_max=error.max(0).tolist(),
                    changed_dimensions=int(np.count_nonzero(error)), dimensions=int(error.size))
                summary['source_groups']['human' if human else 'policy']['normalized_quantization_error'] = dict(
                    per_dimension_mae=(error*2/scale).mean(0).tolist(),
                    per_dimension_p95=np.quantile(error*2/scale,.95,axis=0).tolist())
                pairs = [(a,b) for a,b in adjacent if 0<b['timestamp']-a['timestamp']<.5]
                if pairs:
                    v0 = np.asarray([a['action'][-7:] for a,b in pairs],np.float32)
                    v1 = np.asarray([b['action'][-7:] for a,b in pairs],np.float32)
                    delta = v1-v0
                    restored_delta = v1.astype(np.float16).astype(np.float32)-v0.astype(np.float16).astype(np.float32)
                    moving = np.abs(delta)>1e-7
                    erased = moving & (restored_delta==0)
                    summary['source_groups']['human' if human else 'policy']['fp16_motion_erasure'] = dict(
                        pairs=len(pairs), moving_per_dimension=moving.sum(0).tolist(),
                        erased_per_dimension=erased.sum(0).tolist(), threshold=1e-7)
        run = []
        for r in rows + [dict(human_controlled=False)]:
            if r.get('human_controlled'):
                run.append(r)
            elif run:
                summary['human_runs'].append(dict(rows=len(run), first_timestamp=run[0]['timestamp'],
                    last_timestamp=run[-1]['timestamp'], source_timestamp_span=run[-1]['timestamp']-run[0]['timestamp'],
                    assumed_20hz_span=(len(run)-1)*.05,
                    actor_versions=sorted({r['actor_param_version'] for r in run})))
                run = []
        result['traces'].append(summary)
    for ep, recording_index in [(212, 27), (215, 30), (217, 32)]:
        f = ROOT / f'recordings/rl-platform/rlt/online/episode_{recording_index:06d}.hdf5'
        label = f.with_suffix('.labels.json')
        with h5py.File(f, 'r') as h:
            arrays = {k: np.asarray(h[k]) for k in ['action', 'observations/qpos',
                'rollout/coordinator_command', 'rollout/is_intervention_right',
                'rollout/sample_timestamp', 'rollout/topic_timestamp/rear_right',
                'rollout/valid_mask/coordinator_command', 'rollout/valid_mask/qpos']}
            feedback = arrays['observations/qpos'][:, 7:]
            command = arrays['rollout/coordinator_command'][:, 7:]
            human = arrays['rollout/is_intervention_right']
            result['recordings'].append(dict(path=str(f), bytes=f.stat().st_size,
                candidate_replay_episode=ep,
                join_status='time-compatible candidate; Replay lacks UUID; not a proven join',
                uuid=str(h.attrs['episode_uuid']), start_timestamp=float(h.attrs['start_timestamp']),
                end_timestamp=float(h.attrs['end_timestamp']), fps=float(h.attrs['fps']),
                label=json.loads(label.read_text()), label_sha256=digest(label.read_bytes()),
                dataset_sha256={k: array_digest(v) for k, v in arrays.items()},
                human_command_feedback=errors(command, feedback, human),
                policy_command_feedback=errors(command, feedback, ~human),
                human_action_command=errors(arrays['action'][:, 7:], command, human),
                human_joint_delta=stats(np.abs(command[human, :6]-feedback[human, :6]).reshape(-1)),
                observation_clock='sample_timestamp monotonic; rear_right topic_timestamp ROS',
                numeric={k: v.tolist() for k, v in arrays.items() if k != 'action'}))
    p = Path('/home/agilex/jiaan/project/rl-platform')
    for n in ['methods/openpi_rlt/cobot_adapter/cobot_online_env.py',
              'methods/openpi_rlt/cobot_adapter/online_runtime.py',
              'methods/openpi_rlt/plug_v3_yyshadow/right_arm_env.py',
              'methods/openpi_rlt/cobot_adapter/cobot_ros1.py']:
        result['code_hashes'][n] = digest((p/n).read_bytes())
    after = journal.stat()
    result['journal_stable'] = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    if not result['journal_stable']:
        raise RuntimeError('Replay changed while reading; refuse output')
    # Missing source fields may contain NaN. Encode them explicitly as null.
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value): return None
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        return value
    print(json.dumps(clean(result), allow_nan=False))


if __name__ == '__main__':
    main()
