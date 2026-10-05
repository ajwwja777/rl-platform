"""Optional logical20 RLT executor: asynchronous RTC, regular publication, HIL epochs.

Hardware authority remains in the existing Task2 runtime and I/O. Shared queue
and physical-time filtering come from VLA. Native rewards and Replay schema stay
unchanged; raw traces retain the complete emitted sub-tick sequence.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import copy
from types import SimpleNamespace
import time
import threading
import numpy as np
from methods.openpi_rlt.cobot_adapter.episode_control import EpisodePhase
from methods.openpi_rlt.cobot_adapter.trace import ControlSource, EpisodeOutcome
from .execution_profiles import load_shared


class AsyncExecution:
    def __init__(self, env, name, profile, clock=time.monotonic):
        self.env, self.name, self.config, self.clock = env, name, profile, clock
        Queue, self.events, Filter = load_shared()
        self.queue = Queue()
        self.filter = Filter(profile.smoothing_tau_sec, profile.joint_velocity_limit)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rlt-rtc-plan')
        self.health_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rlt-recorder-check')
        self.health_future = None
        self.health_lock = threading.RLock()
        self.health_started = None
        self.health_identity = None
        self.backend = None
        self.terminal_outcome = None
        self.last_policy_source = ControlSource.BASE
        self.epoch, self.request_id = 0, 0
        self.future, self.pending, self.plan = None, None, None
        self.anchors = {}
        self.stats = dict(profile=name, logical_hz=20, publish_hz=profile.publish_hz,
                          inference_requests=0, stale_results=0, emitted_commands=0,
                          rtc_training_data=profile.rtc, smoothing=profile.smoothing_tau_sec > 0)
        self.timeline_origin = None
        self.publication_index = 0
        self.segment_start = self.last_command = self.last_ref = None
        self.segment_ref_start = None

    def invalidate(self):
        self.epoch += 1
        self.queue.reset()
        self.filter.reset()
        if self.future is not None:
            self.future.cancel()
        self.pending = None
        self.plan = None
        self.timeline_origin = None
        self.publication_index = 0
        self.segment_ref_start = None

    def close(self):
        self.invalidate()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.health_pool.shutdown(wait=True, cancel_futures=True)

    def set_episode(self, episode_id):
        self.invalidate()
        self.anchors.clear()
        for key in ("inference_requests","stale_results","emitted_commands"):
            self.stats[key] = 0
        self.last_policy_source = ControlSource.BASE
        self.backend.episode_id = episode_id
        self.terminal_outcome = None

    def check_report(self):
        with self.health_lock:
            if self.health_future is None:
                return
            reason = None
            if self.health_future.done():
                try:
                    self.health_future.result()
                except Exception as error:
                    reason = 'task5_recorder_unavailable: ' + str(error)
                self.health_future = None
            elif time.monotonic() - self.health_started > 1.0:
                reason = 'task5_recorder_health_timeout'
            application = getattr(self.env._io, '_session_application', None)
            if application and self.health_identity is not None:
                current = application.snapshot()
                if (current.episode_id, current.generation) != self.health_identity:
                    return
            if reason:
                io = self.env._io
                io.set_policy_paused(True)
                io.set_chunk_ready(False)
                application = getattr(io, '_session_application', None)
                if application and application.snapshot().phase.value in {'rollout', 'hil'}:
                    application.mark_terminal_pending(reason, pause_capture=False)
                self.stats['recorder_warning'] = reason

    def report_chunk(self, latency_sec, actor_version):
        # Recorder HTTP is not part of the Stage1/Actor inference budget.
        # One independent worker; do not enqueue unbounded stale checks.
        with self.health_lock:
            self.check_report()
            if self.health_future is not None:
                return
            self.health_started = time.monotonic()
            application = getattr(self.env._io, '_session_application', None)
            snapshot = application.snapshot() if application else None
            self.health_identity = (snapshot.episode_id, snapshot.generation) if snapshot else None
            def report():
                started = time.monotonic()
                try:
                    self.env._io.report_chunk(latency_sec=latency_sec, actor_version=actor_version)
                finally:
                    self.stats["last_recorder_check_ms"] = (time.monotonic()-started)*1000
            self.health_future = self.health_pool.submit(report)

    def sample(self):
        io = self.env._io
        sample = io.sample_control(right_arm_only=True) if hasattr(io, "sample_control") else io.sample()
        # ROS outcome is consumed on read. Internal publication probes must
        # retain it until the logical trace/finalization boundary sees it.
        if sample.outcome is not None:
            self.terminal_outcome = sample.outcome
        return SimpleNamespace(observation=sample.observation,mode=sample.mode,
            outcome=self.terminal_outcome,paused=sample.paused,timestamp=sample.timestamp)

    def observation(self, observation):
        result = dict(observation)
        result.pop('rtc', None)
        if self.config.rtc and self.plan is not None:
            pending = self.queue.remaining_actions()[:, :7]
            if len(pending):
                result['rtc'] = dict(previous_actions=pending.copy(), delay_steps=0,
                                     execution_horizon=min(self.config.replan_after_steps, len(pending)))
        return result

    def fresh_plan(self, observation):
        if self.backend is None:
            raise RuntimeError('Async execution backend was not installed in EnvDriver')
        # One serialized worker owns Stage1/Actor RPC. Never overlap an old HIL
        # request with fresh inference or Replay finalization on the same socket.
        if self.future is not None:
            try:
                self.future.result(timeout=10.)
            except TimeoutError as exc:
                raise RuntimeError('Old inference did not finish after pause/HIL') from exc
            except Exception:
                pass
            self.future = None
        request = dict(observation)
        request.pop('rtc', None)
        epoch = self.epoch
        self.future = self.pool.submit(self.backend.plan, copy.deepcopy(request), self.env._episode_steps)
        plan = self.future.result(timeout=10.)
        self.future = None
        sample = self.sample()
        state = self.env._runtime.observe_mode(sample.mode)
        if epoch != self.epoch or sample.paused or sample.outcome is not None or state.phase is not EpisodePhase.ROLLOUT:
            self.invalidate()
            return False
        self.queue.initialize(np.concatenate([plan.action_chunk, plan.ref_chunk,
            np.full((len(plan.action_chunk),1),plan.actor_param_version,np.float32),
            np.full((len(plan.action_chunk),1),plan.source,np.float32)], -1))
        self.plan = plan
        self.anchors[self.env._episode_steps] = plan.start_features
        self.segment_start = self.env._state(observation).copy()
        self.last_command = self.segment_start.copy()
        self.last_ref = self.segment_start.copy()
        self.segment_ref_start = self.segment_start.copy()
        self.filter.reset(self.segment_start)
        self.timeline_origin = self.clock()
        self.publication_index = 0
        self.env._runtime.install_fresh_plan(self.env._runtime.snapshot().generation)
        self.env._io.set_chunk_ready(True)
        self.stats['inference_requests'] += 1
        return True

    def request_next(self, observation):
        if self.future is not None or self.queue.cursor < self.config.replan_after_steps:
            return
        snapshot = self.queue.begin_inference(str(self.epoch), self.request_id)
        self.request_id += 1
        request = copy.deepcopy(observation)
        if self.config.rtc:
            request['rtc'] = dict(previous_actions=snapshot.previous_actions[:, :7].copy(),
                                  delay_steps=self.config.max_delay_steps,
                                  execution_horizon=self.config.max_delay_steps+1)
        else:
            request.pop('rtc', None)
        self.pending = (self.epoch, snapshot, self.env._episode_steps)
        self.future = self.pool.submit(self.backend.plan, request, self.env._episode_steps)
        self.stats['inference_requests'] += 1

    def accept_result(self):
        if self.future is None or not self.future.done():
            return
        future, pending = self.future, self.pending
        self.future, self.pending = None, None
        if pending is None or pending[0] != self.epoch:
            self.stats['stale_results'] += 1
            return
        plan = future.result()
        _, snapshot, step = pending
        actions = np.asarray(plan.action_chunk, np.float32).copy()
        # Actor is downstream of RTC Reference. Preserve the committed prefix
        # after Actor refinement so it cannot undo inpainting's fixed actions.
        n = min(self.config.max_delay_steps, len(snapshot.previous_actions))
        actions[:n] = snapshot.previous_actions[:n, :7]
        bundle = np.concatenate([actions, plan.ref_chunk,
            np.full((len(actions),1),plan.actor_param_version,np.float32),
            np.full((len(actions),1),plan.source,np.float32)], -1)
        bundle[:n,14:] = snapshot.previous_actions[:n,14:]
        actual_delay = self.queue.cursor - snapshot.execution_horizon
        self.stats['last_actual_delay_steps'] = actual_delay
        self.stats['allowed_delay_steps'] = self.config.max_delay_steps
        if actual_delay > self.config.max_delay_steps:
            raise RuntimeError(
                f"RTC delay exceeded budget: actual={actual_delay} logical steps "
                f"({actual_delay/self.config.logical_hz:.3f}s), "
                f"allowed={self.config.max_delay_steps} "
                f"({self.config.max_delay_steps/self.config.logical_hz:.3f}s); "
                f"model_ms={self.stats.get('last_model_inference_ms', 'unknown')}, "
                f"recorder_check_ms={self.stats.get('last_recorder_check_ms', 'unknown')}; "
                "policy paused; keep Stage1 and recover runtime")
        delay = self.queue.complete_inference(snapshot.session_id, snapshot.request_id,
                                               self.config.max_delay_steps, bundle)
        self.plan = plan
        self.anchors[step] = plan.start_features
        self.stats['last_actual_delay_steps'] = delay

    def wait_until(self, timestamp):
        remaining = timestamp-self.clock()
        if remaining > 0:
            self.env._sleep(remaining)
        elif remaining < -.5/self.config.publish_hz:
            self.stats['last_deadline_lateness_ms'] = -remaining * 1000
            raise RuntimeError(
                f'Execution clock missed its deadline; no catch-up command burst; '
                f'late_ms={-remaining*1000:.1f}, publish_hz={self.config.publish_hz}, '
                f'logical_step={self.env._episode_steps}')

    def wait_active(self, timestamp, epoch):
        # Pause/HIL invalidates the old timeline. Check authority BEFORE its
        # deadline: an operator interruption is not a publisher overrun.
        def active():
            check = self.sample()
            state = self.env._runtime.observe_mode(check.mode)
            if state.phase is EpisodePhase.FAULT:
                raise RuntimeError(state.fault_reason or 'Control coordinator fault')
            if (epoch != self.epoch or check.paused or check.outcome is not None
                    or state.phase is not EpisodePhase.ROLLOUT):
                self.env._io.set_chunk_ready(False)
                if epoch == self.epoch:
                    self.invalidate()
                return False
            return True
        if not active():
            return False
        try:
            self.wait_until(timestamp)
        except RuntimeError:
            # Pause can arrive after the precheck while this thread is
            # descheduled. Recheck authority before propagating a late clock.
            if not active():
                return False
            raise
        return active()

    def execute_chunk(self, observation=None, policy_planner=None, control_hz=None):
        env, io, runtime = self.env, self.env._io, self.env._runtime
        if control_hz is not None and float(control_hz) != self.config.logical_hz:
            raise ValueError('Publication Hz must not replace logical Replay Hz')
        current = self.observation(observation or self.sample().observation)
        trace, rewards, outcome = [], [], None
        paused_last = False
        if env._shadow_mode:
            raise ValueError('Use the offline execution audit for this profile; live shadow mode is unsupported')
        try:
            while len(trace) < env._chunk_exec_horizon:
                sample = self.sample()
                before = runtime.snapshot()
                state = runtime.observe_mode(sample.mode)
                if state.phase is EpisodePhase.FAULT:
                    raise RuntimeError(state.fault_reason or 'Control coordinator fault')
                if sample.outcome is not None:
                    outcome = EpisodeOutcome(sample.outcome)
                    if trace:
                        trace[-1].update(reward=float(outcome is EpisodeOutcome.SUCCESS), done=True, outcome=outcome.value)
                        rewards[-1] = trace[-1]['reward']
                        current = trace[-1]['next_observation']
                    break
                if sample.paused and state.phase is EpisodePhase.ROLLOUT:
                    if not paused_last:
                        io.set_chunk_ready(False)
                        self.invalidate()
                    paused_last = True
                    current = sample.observation
                    env._sleep(1/self.config.publish_hz)
                    continue
                paused_last = False
                if state.phase is EpisodePhase.HIL:
                    if self.plan is not None:
                        io.set_chunk_ready(False)
                        self.invalidate()
                    current = dict(current)
                    current.pop('rtc', None)
                    env._sleep(1/self.config.logical_hz)
                    after = self.sample()
                    executed = env._state(after.observation)
                    ref_action = executed.copy()
                    source, version, publications = runtime.control_source(self.last_policy_source), -1, []
                    next_observation = dict(after.observation)
                    next_observation.pop('rtc', None)
                else:
                    if self.plan is None or state.fresh_plan_required:
                        self.invalidate()
                        current = sample.observation
                        if not self.fresh_plan(current):
                            continue
                    self.check_report()
                    self.accept_result()
                    # Start of one logical20 segment. Queue consumption, model
                    # delay and Replay step IDs all use this clock, never pub Hz.
                    current = self.observation(current)
                    target_bundle = self.queue.pop()
                    target, ref_target = target_bundle[:7], target_bundle[7:14]
                    version, source = int(target_bundle[14]), ControlSource(int(target_bundle[15]))
                    self.last_policy_source = source
                    # This request starts at the NEXT logical boundary, below.
                    publications = []
                    epoch = self.epoch
                    interval_origin = self.timeline_origin
                    interval_index = self.publication_index
                    for timestamp, alpha in self.events(self.publication_index, 20, self.config.publish_hz):
                        if not self.wait_active(interval_origin+timestamp, epoch):
                            break
                        check = self.sample()
                        checked = runtime.observe_mode(check.mode)
                        if check.outcome is not None or check.paused or checked.phase is not EpisodePhase.ROLLOUT:
                            io.set_chunk_ready(False)
                            self.invalidate()
                            break
                        if epoch != self.epoch:
                            break
                        requested = self.segment_start + alpha*(target-self.segment_start)
                        filtered = self.filter.apply(requested, env._state(check.observation), 1/self.config.publish_hz)
                        # Gripper is not EMA filtered. Its physical velocity
                        # cap is separate and scales with publication period.
                        filtered[6] = self.last_command[6] + np.clip(filtered[6]-self.last_command[6],
                            -self.config.gripper_velocity_limit/self.config.publish_hz,
                            self.config.gripper_velocity_limit/self.config.publish_hz)
                        command = runtime.safe_policy_target(filtered, env._state(check.observation))
                        if io.publish_policy_action(command) is False:
                            io.set_chunk_ready(False)
                            self.invalidate()
                            break
                        limits = np.full(7,self.config.joint_velocity_limit/self.config.publish_hz)
                        limits[6] = self.config.gripper_velocity_limit/self.config.publish_hz
                        if np.any(np.abs(command-self.last_command)>limits+1e-6):
                            raise RuntimeError("Feedback safety clamp would break physical rate limit")
                        self.last_command = command.copy()
                        self.filter.previous = command.copy()
                        self.last_ref = self.segment_ref_start + alpha*(ref_target-self.segment_ref_start) if hasattr(self,'segment_ref_start') else ref_target.copy()
                        publications.append(dict(timestamp=float(io.ros.Time.now().to_sec()) if hasattr(getattr(io,'ros',None),'Time') else float(check.timestamp),
                                                 monotonic_timestamp=float(self.clock()), action=command.copy(),
                                                 ref_action=self.last_ref.copy(), actor_param_version=int(version)))
                        self.stats['emitted_commands'] += 1
                    # A partial segment can end on pause/HIL. Never wait on
                    # the invalidated interval's deadline after breaking out.
                    if epoch == self.epoch:
                        self.wait_active(interval_origin+(interval_index+1)/20, epoch)
                    after = self.sample()
                    after_state = runtime.observe_mode(after.mode)
                    if after_state.phase is EpisodePhase.FAULT:
                        raise RuntimeError(after_state.fault_reason or 'Control coordinator fault')
                    if not publications and after.outcome is not None and after_state.phase is EpisodePhase.ROLLOUT:
                        outcome = EpisodeOutcome(after.outcome)
                        if trace:
                            trace[-1].update(reward=float(outcome is EpisodeOutcome.SUCCESS),done=True,outcome=outcome.value)
                            rewards[-1] = trace[-1]["reward"]
                            current = trace[-1]["next_observation"]
                        break
                    if not publications and (after.paused or after_state.phase is EpisodePhase.ROLLOUT) and after.outcome is None:
                        if not after.paused:
                            raise RuntimeError('Hardware rejected policy publication; no executed transition recorded')
                        current = after.observation
                        continue
                    if after_state.phase is EpisodePhase.HIL:
                        source = runtime.control_source(source)
                        executed = env._state(after.observation)
                    else:
                        executed = self.last_command.copy() if publications else env._state(after.observation)
                    ref_action = self.last_ref.copy() if self.last_ref is not None else executed.copy()
                    if epoch == self.epoch:
                        self.segment_start = target.copy()
                        self.segment_ref_start = ref_target.copy()
                        self.publication_index += 1
                    next_observation = self.observation(after.observation)
                env._episode_steps += 1
                outcome = EpisodeOutcome(after.outcome) if after.outcome is not None else None
                if outcome is None and env._max_episode_steps is not None and env._episode_steps >= env._max_episode_steps:
                    io.set_chunk_ready(False)
                    self.invalidate()
                    io.mark_terminal_pending('max_episode_steps')
                    outcome = EpisodeOutcome(io.wait_terminal_outcome())
                reward = float(outcome is EpisodeOutcome.SUCCESS)
                record = dict(observation=current, action=np.asarray(executed,np.float32),
                    ref_action=np.asarray(ref_action,np.float32), reward=reward,
                    next_observation=next_observation, human_controlled=source in (ControlSource.HUMAN,ControlSource.MIXED),
                    source=int(source), actor_param_version=int(version), done=outcome is not None,
                    outcome=None if outcome is None else outcome.value, expert_mask=list(runtime.snapshot().expert_mask),
                    timestamp=float(after.timestamp), shadow=env._shadow_mode,
                    execution_profile=self.name, logical_hz=20, publish_hz=self.config.publish_hz,
                    io_evidence_before_step=getattr(sample, 'io_evidence', None),
                    io_evidence_after_step=getattr(after, 'io_evidence', None),
                    publications=publications)
                trace.append(record); rewards.append(reward)
                io.record_raw_step(record)
                current = next_observation
                if outcome is not None:
                    break
                if runtime.snapshot().phase is EpisodePhase.ROLLOUT and not after.paused and self.plan is not None:
                    self.request_next(current)
            if outcome is not None:
                env._last_outcome = outcome
                io.set_chunk_ready(False)
                self.invalidate()
                runtime.finish_episode(outcome)
                if env._phase_controller is not None:
                    env._phase_controller.finish_episode()
                # Serialized feature client must be quiet before Replay build.
                if self.future is not None:
                    try: self.future.result(timeout=10.)
                    except TimeoutError: raise RuntimeError('Inference still busy at episode finalization')
                    except Exception: pass
                    self.future = None
            sources = {int(r['source']) for r in trace}
            return current, rewards, outcome is not None, dict(step_trace=trace,
                success=int(outcome is EpisodeOutcome.SUCCESS),
                source=next(iter(sources)) if len(sources)==1 else int(ControlSource.MIXED),
                intervention_flag=any(r['human_controlled'] for r in trace), chunk_start_features=None,
                policy_anchor_offsets=[], policy_anchor_features=[],
                outcome=None if outcome is None else outcome.value, replay_eligible=env.replay_commit_allowed(),
                drop_transition=env._shadow_mode or outcome is EpisodeOutcome.ABORTED,
                execution=dict(self.stats))
        except Exception as exc:
            self.stats["last_error"] = str(exc)
            io.set_chunk_ready(False)
            io.set_policy_paused(True)
            application = getattr(io, "_session_application", None)
            if application is not None:
                application._controller.fail("execution_timing_failed: " + str(exc))
            self.invalidate()
            raise
