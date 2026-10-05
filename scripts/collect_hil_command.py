"""Read-only HDF5/trace comparison; compatible snapshots are not exact actions."""
import collections
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    request = json.load(sys.stdin)
    grouped = collections.defaultdict(list)
    for match in request['matches']:
        grouped[(match['split'], match['episode'], match['step'])].append(match)
    unique = [v[0] for v in grouped.values() if len(v) == 1]
    files = {}
    steps = {}
    for match in unique:
        if 2 not in match['trace_sources']:
            continue
        path = match['trace']
        if path not in files:
            content = Path(path).read_bytes()
            files[path] = {'sha256': hashlib.sha256(content).hexdigest(), 'rows': [
                r for r in (json.loads(line) for line in content.splitlines() if line.strip())
                if np.asarray(r.get('action', [])).shape == (7,) and
                'observation' in r and 'next_observation' in r]}
        rows = files[path]['rows']
        for offset, row in enumerate(rows[match['trace_start_row']:match['trace_start_row']+10]):
            if row['source'] != 2:
                continue
            identity = (match['split'], match['episode'], match['step']+offset, row['timestamp'])
            steps[identity] = row
    relevant = []
    times = [identity[-1] for identity in steps]
    for asset in request['census']['rows']:
        begin = float(asset['attrs']['start_timestamp'])
        end = float(asset['attrs']['end_timestamp'])
        if any(begin-0.15 <= t <= end+0.15 for t in times):
            relevant.append(asset)
    raw = []
    receipts = []
    for asset in relevant:
        arrays = {}
        with h5py.File(asset['path'], 'r') as handle:
            for name in ['observations/qpos', 'rollout/front_observation',
                         'rollout/coordinator_command', 'rollout/is_intervention_right',
                         'rollout/valid_mask/coordinator_right', 'rollout/valid_mask/front_right',
                         'rollout/topic_timestamp/front_right', 'rollout/topic_timestamp/coordinator_right',
                         'rollout/sample_timestamp', 'rollout/control_source_right']:
                arrays[name] = handle[name][:]
        digest = hashlib.sha256()
        for name, value in sorted(arrays.items()):
            digest.update(name.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        receipts.append(dict(path=asset['path'], uuid=asset['attrs'].get('episode_uuid'),
                             frames=asset['frames'], low_dimensional_arrays_sha256=digest.hexdigest()))
        raw.append((asset, arrays))
    output = {'steps': [], 'assets': receipts,
              'trace_assets': [{'path': p, 'sha256': d['sha256']} for p, d in files.items()],
              'unique_hil_windows': sum(2 in m['trace_sources'] for m in unique),
              'boundary': 'Exact Replay/trace window identity inherited. HDF5 comparison uses a '
              'past-only 150ms FRONT ROS stamp compatibility interval. No command publication '
              'receipt joins these trace states; even singleton snapshots are NOT recovered '
              'training action identity. No interpolation or nearest-frame selection.'}
    for identity, row in sorted(steps.items()):
        split, episode, step, event = identity
        state = np.asarray(row['observation']['state'], np.float32)
        action = np.asarray(row['action'], np.float32)
        next_state = np.asarray(row['next_observation']['state'], np.float32)
        compatible = []
        exact = []
        for asset, arrays in raw:
            begin = float(asset['attrs']['start_timestamp'])
            end = float(asset['attrs']['end_timestamp'])
            if not begin-0.15 <= event <= end+0.15:
                continue
            feedback = arrays['rollout/front_observation'][:, 7:14].astype(np.float32)
            commands = arrays['rollout/coordinator_command'][:, 7:14].astype(np.float32)
            stamp = arrays['rollout/topic_timestamp/front_right']
            command_stamp = arrays['rollout/topic_timestamp/coordinator_right']
            # Coordinator can be newer than sampled feedback, but must itself precede trace event.
            mask = ((stamp <= event) & (event-stamp <= .15) &
                    (command_stamp <= event) & (event-command_stamp <= .25) &
                    arrays['rollout/is_intervention_right'].astype(bool) &
                    arrays['rollout/valid_mask/front_right'].astype(bool) &
                    arrays['rollout/valid_mask/coordinator_right'].astype(bool) &
                    np.isfinite(feedback).all(axis=1) & np.isfinite(commands).all(axis=1))
            for index in np.flatnonzero(mask):
                item = {'path': asset['path'], 'frame': int(index),
                        'front_stamp': float(stamp[index]), 'command_stamp': float(command_stamp[index]),
                        'feedback_matches_state': bool(np.array_equal(feedback[index], state)),
                        'feedback_matches_next_state': bool(np.array_equal(feedback[index], next_state)),
                        'feedback_matches_recorded_action': bool(np.array_equal(feedback[index], action)),
                        'command_minus_feedback': (commands[index]-feedback[index]).tolist(),
                        'command_minus_trace_action': (commands[index]-action).tolist()}
                compatible.append(item)
                # Historical HIL action is the measured end-of-step feedback,
                # not necessarily the transition's start state.
                if item['feedback_matches_recorded_action'] and item['feedback_matches_next_state']:
                    exact.append(item)
        output['steps'].append({'split': split, 'episode': episode, 'step': step, 'timestamp': event,
            'action_equals_state': bool(np.array_equal(action, state)),
            'action_equals_next_state': bool(np.array_equal(action, next_state)),
            'state_equals_next_state': bool(np.array_equal(state, next_state)),
            'compatible_snapshots': compatible, 'exact_action_feedback_compatible_snapshots': len(exact),
            'all_exact_candidate_commands_equal': bool(exact) and all(
                x['command_minus_trace_action'] == exact[0]['command_minus_trace_action'] for x in exact),
            'command_identity_status': 'insufficient_evidence'})
    print(json.dumps(output, allow_nan=False))


if __name__ == '__main__':
    main()
