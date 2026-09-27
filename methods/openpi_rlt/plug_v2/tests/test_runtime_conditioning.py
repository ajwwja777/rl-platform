import importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("rtc_queue_under_test", Path(__file__).parents[1] / "rtc_queue.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RTCQueue = MODULE.RTCQueue
QueueFault = MODULE.QueueFault
CommandFilter = MODULE.CommandFilter
PIPER_LOWER = MODULE.PIPER_LOWER
PIPER_UPPER = MODULE.PIPER_UPPER
PiperWorkspaceGuard = MODULE.PiperWorkspaceGuard
piper_link6_xyz = MODULE.piper_link6_xyz


def primed_queue(delta):
    queue = RTCQueue(tracking_joints=np.arange(7, 13))
    queue.resume()
    request = queue.request(np.zeros(14, np.float32))
    plan = np.zeros((50, 14), np.float32)
    plan[:, 11] = delta
    queue.complete(request, plan)
    return queue


def test_tracking_guard_remains_fail_closed_at_004_rad():
    assert primed_queue(0.039).pop(np.zeros(14, np.float32))[11] == pytest.approx(0.039)
    queue = primed_queue(0.041)
    with pytest.raises(QueueFault, match=r"joint=11.*bound=0.040000"):
        queue.pop(np.zeros(14, np.float32))
    assert not queue.active


def test_corrective_conditioning_respects_velocity_acceleration_and_passive_joints():
    filt = CommandFilter(velocity=0.10, acceleration=0.9)
    raw = np.ones((50, 14), np.float32)
    state = np.zeros(14, np.float32)
    prefix = np.zeros((50, 14), np.float32)
    output = filt.plan(raw, state, prefix, 0)
    velocity_steps = np.diff(np.vstack([state, output]), axis=0)[:, 7:13]
    acceleration_steps = np.diff(np.vstack([np.zeros((1, 6), np.float32), velocity_steps]), axis=0)
    assert np.max(np.abs(velocity_steps)) <= 0.10 / 30 + 1e-7
    assert np.max(np.abs(acceleration_steps)) <= 0.9 / 30 / 30 + 1e-7
    assert np.array_equal(output[:, :7], np.zeros((50, 7), np.float32))
    assert np.array_equal(output[:, 13], np.zeros(50, np.float32))


def test_unreachable_piper_j5_request_is_clamped_without_command_windup():
    filt = CommandFilter(velocity=0.10, acceleration=0.9)
    state = np.zeros(14, np.float32)
    state[11] = np.float32(1.220)
    raw = np.broadcast_to(state, (50, 14)).copy()
    raw[:, 11] = np.float32(1.272)
    output = filt.plan(raw, state, np.zeros((50, 14), np.float32), 0)
    assert PIPER_UPPER[4] == pytest.approx(1.22108, abs=1e-6)
    assert output[:, 11].max() <= PIPER_UPPER[4]
    assert np.max(np.abs(output[:, 11] - state[11])) < 0.002
    assert np.all(output[:, 7:13] >= PIPER_LOWER - 1e-7)
    assert np.all(output[:, 7:13] <= PIPER_UPPER + 1e-7)


def test_out_of_limit_committed_prefix_is_rejected():
    filt = CommandFilter(velocity=0.10, acceleration=0.9)
    state = np.zeros(14, np.float32)
    prefix = np.zeros((50, 14), np.float32)
    prefix[:6, 11] = PIPER_UPPER[4] + 0.01
    with pytest.raises(ValueError, match="committed prefix outside Piper joint limits"):
        filt.plan(np.zeros((50, 14), np.float32), state, prefix, 6)


def test_limit_clamped_committed_prefix_does_not_wind_up_next_plan():
    filt = CommandFilter(velocity=0.10, acceleration=0.9)
    state = np.zeros(14, np.float32)
    state[11] = PIPER_UPPER[4]
    prefix = np.broadcast_to(state, (50, 14)).copy()
    raw = prefix.copy()
    raw[:, 11] = PIPER_UPPER[4] + 0.2
    output = filt.plan(raw, state, prefix, 6)
    assert np.all(output[:, 11] <= PIPER_UPPER[4])
    assert np.allclose(output[:, 11], PIPER_UPPER[4])


def test_late_rtc_plan_is_rejected_and_queue_can_reanchor_at_zero_prefix():
    queue = primed_queue(0.0)
    request = queue.request(np.zeros(14, np.float32))
    assert request.prefix_length == 6
    assert queue.is_current(request)
    for _ in range(6):
        queue.pop(np.zeros(14, np.float32))
    with pytest.raises(QueueFault, match='rtc_deadline_missed'):
        queue.pop(np.zeros(14, np.float32))
    assert not queue.is_current(request)
    assert not queue.complete(request, np.zeros((50, 14), np.float32))
    queue.resume()
    fresh = queue.request(np.zeros(14, np.float32))
    assert fresh.prefix_length == 0


def _state(right):
    state = np.zeros(14, np.float32)
    state[7:13] = np.asarray(right, np.float32)
    return state


def test_piper_numpy_fk_matches_deployed_urdf_reference():
    joints = [0.5573532581, 1.4695698023, -1.2868438959,
              -0.0454939529, 1.1997460127, -0.8545466661]
    expected = [0.2251633229479561, 0.14668475407140377, 0.31677141042950185]
    assert np.allclose(piper_link6_xyz(joints), expected, atol=1e-7)


@pytest.mark.parametrize("start, accepted", [
    (
        [0.5573881269, 1.4700931311, -1.2865996361, -0.0451276265, 1.1995890141, -0.8542675972],
        [
            [0.5597953796, 1.4722038507, -1.3230924606, -0.0401386432, 1.2189168930, -0.8580877781],
            [0.5417582989, 1.8262821436, -1.4028464556, -0.0517040156, 1.0716197491, -0.7868813872],
        ],
    ),
    (
        [0.5577893257, 1.4703372717, -1.2866345644, -0.0454590656, 1.1998680830, -0.8541978002],
        [
            [0.5577893257, 1.4766520262, -1.3013398647, -0.0519133434, 1.1916345358, -0.8701416254],
            [0.5379380584, 1.9848306179, -1.6129071712, -0.0634787157, 1.2226325274, -0.8730198741],
        ],
    ),
])
def test_workspace_guard_accepts_recorded_success_extrema(start, accepted):
    guard = PiperWorkspaceGuard(_state(start))
    for joints in accepted:
        guard.validate_measured(_state(joints))


def test_workspace_guard_rejects_recorded_upward_wander():
    start = [0.5570043921, 1.4700232744, -1.2863554955,
             -0.0451799594, 1.2000250816, -0.8541803360]
    upward = [0.5448808074, 1.5808799267, -1.4998177290,
              -0.0671419576, 1.1385174990, -0.8452664614]
    guard = PiperWorkspaceGuard(_state(start))
    with pytest.raises(ValueError, match=r"workspace_envelope_exceeded.*axis=z"):
        guard.validate_measured(_state(upward))


def test_workspace_guard_rejects_unsafe_future_before_queueing():
    start = [0.5570043921, 1.4700232744, -1.2863554955,
             -0.0451799594, 1.2000250816, -0.8541803360]
    upward = [0.5448808074, 1.5808799267, -1.4998177290,
              -0.0671419576, 1.1385174990, -0.8452664614]
    plan = np.broadcast_to(_state(start), (50, 14)).copy()
    plan[15, 7:13] = upward
    guard = PiperWorkspaceGuard(_state(start))
    with pytest.raises(ValueError, match=r"source=plan index=15.*axis=z"):
        guard.validate_plan(plan, 0, 16)


def test_explicit_passive_commands_survive_measured_feedback_drift():
    filt = CommandFilter(velocity=0.10, acceleration=0.9)
    raw = np.ones((50, 14), np.float32)
    measured = np.zeros(14, np.float32)
    measured[:7] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.0003]
    measured[13] = 0.0320
    passive = measured.copy()
    passive[:7] = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.0002]
    passive[13] = 0.0318
    output = filt.plan(
        raw,
        measured,
        np.zeros((50, 14), np.float32),
        0,
        passive_commands=passive,
    )
    passive_indices = np.r_[0:7, 13]
    assert np.array_equal(
        output[:, passive_indices],
        np.broadcast_to(passive[passive_indices], (50, 8)),
    )


def test_passive_command_latch_reuses_d0_commands_until_reanchored():
    latch = MODULE.PassiveCommandLatch()
    first = np.zeros((50, 14), np.float32)
    first[0] = np.arange(14, dtype=np.float32)
    later = np.full((50, 14), 99.0, np.float32)
    assert np.array_equal(latch.update(first, 0), first[0])
    assert np.array_equal(latch.update(later, 6), first[0])
    latch.reset()
    assert np.array_equal(latch.update(later, 0), later[0])


def test_passive_command_latch_keeps_episode_start_across_d0_reanchor():
    latch = MODULE.PassiveCommandLatch()
    episode_start = np.zeros((50, 14), np.float32)
    episode_start[0] = np.arange(14, dtype=np.float32)
    reanchor = np.full((50, 14), 77.0, np.float32)
    assert np.array_equal(latch.update(episode_start, 0), episode_start[0])
    assert np.array_equal(latch.update(reanchor, 0), episode_start[0])


def test_deadline_recovery_budget_allows_three_reanchors_then_fails_closed():
    budget = MODULE.DeadlineRecoveryBudget(max_reanchors=3)
    assert [budget.record_miss() for _ in range(4)] == [True, True, True, False]
    assert budget.count == 4
    budget.reset_episode()
    assert budget.count == 0
    assert budget.record_miss() is True
