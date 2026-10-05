"""Explicit opt-in HIL command targets; unavailable evidence never becomes feedback."""
from __future__ import annotations

import os

import numpy as np


def selected_target() -> str:
    target = os.environ.get('COBOT_RLT_HIL_TARGET', 'feedback')
    if target not in {'feedback', 'coordinator_command'}:
        raise ValueError('COBOT_RLT_HIL_TARGET must be feedback or coordinator_command')
    if target == 'coordinator_command' and os.environ.get('COBOT_RLT_HIL_SAMPLING', 'legacy') != 'logical20':
        raise ValueError('Coordinator HIL targets require explicit logical20 sampling')
    if target == 'coordinator_command' and os.environ.get('COBOT_RLT_RAW_OBSERVATION_CONTRACT', 'legacy') != 'trace':
        raise ValueError('Coordinator HIL targets require actual trace observation identities')
    return target


def right_command_at_step_start(sample, expert_mask):
    """Read the pre-wait command message, not end-state or rear-arm pose.

    Matches the upstream decision order (latest command, period, next state).
    This is received-command evidence, not a hardware execution acknowledgement.
    """
    state = np.asarray(sample.observation['state'], np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('Coordinator HIL targets currently require the finite right-arm 7D contract')
    mode = sample.mode.strip().lower()
    sides = set(mode.split(':', 1)[1].split('+')) if mode.startswith('manual:') else set()
    if len(expert_mask) != 2 or not expert_mask[1] or 'right' not in sides:
        raise ValueError('Coordinator HIL target requires right-arm takeover at step start')
    evidence = getattr(sample, 'io_evidence', None) or {}
    command = evidence.get('coordinator_commands', {}).get('right', {})
    if evidence.get('mode') != sample.mode or not command.get('valid') or command.get('topic') != '/master/joint_right':
        raise ValueError('Fresh right coordinator command evidence is unavailable')
    try:
        target = np.asarray(command['target'], np.float32)
        stamp, captured = float(command['ros_stamp']), float(evidence['captured_ros_time'])
        arrival, capture_monotonic = float(command['arrival_monotonic']), float(evidence['captured_monotonic'])
        takeover_start = float(evidence['right_takeover_started_monotonic'])
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError('Coordinator HIL target has incomplete command clocks') from error
    ages = np.asarray([captured-stamp, capture_monotonic-arrival], np.float64)
    if target.shape != (7,) or not np.isfinite(target).all() or not np.isfinite(ages).all() or np.any(ages < 0) or np.any(ages > .25):
        raise ValueError('Coordinator HIL target is malformed, stale or from a future clock')
    if not np.isfinite(takeover_start) or not takeover_start <= arrival <= capture_monotonic:
        raise ValueError('Coordinator command was not received in the current right-arm takeover epoch')
    return target.copy(), {
        'semantics': 'received_coordinator_command_at_step_start; no execution acknowledgement',
        'topic': command['topic'], 'ros_stamp': stamp, 'captured_ros_time': captured,
        'arrival_monotonic': arrival, 'captured_monotonic': capture_monotonic,
        'right_takeover_started_monotonic': takeover_start,
        'ros_age_seconds': float(ages[0]), 'arrival_age_seconds': float(ages[1]),
    }
