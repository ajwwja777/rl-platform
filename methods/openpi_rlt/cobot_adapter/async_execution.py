"""Optional logical20 RLT executor: asynchronous RTC, regular publication, HIL epochs.

Hardware authority remains in the existing Task2 runtime and I/O. Shared queue
and physical-time filtering come from VLA. Native rewards and Replay schema stay
unchanged; raw traces retain the complete emitted sub-tick sequence.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import copy
import json
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
        enable_trace = getattr(env._io, "enable_async_trace", None)
        if enable_trace is not None:
            enable_trace()
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
        self.anchor_inputs = {}
        self.inference_events = []
        self.evidence_lock = threading.Lock()
        self.plan_event_id = 0
        self.stats = dict(profile=name, logical_hz=20, publish_hz=profile.publish_hz,
                          inference_requests=0, stale_results=0, emitted_commands=0,
                          rtc_training_data=profile.rtc, smoothing=profile.smoothing_tau_sec > 0)
        self.timeline_origin = None
        self.publication_index = 0
        self.clock_shift_sec = 0.
        self.clock_window = None
        self.clock_window_shift_sec = 0.
        self.last_publish_started = None
        self.minimum_publication_time = None
        self.last_wait_deadline = None
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
        self.clock_shift_sec = 0.
        self.clock_window = None
        self.clock_window_shift_sec = 0.
        self.last_publish_started = None
        self.minimum_publication_time = None
        self.last_wait_deadline = None
        self.segment_ref_start = None

    def close(self):
        self.invalidate()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.health_pool.shutdown(wait=True, cancel_futures=True)

    def set_episode(self, episode_id):
        self.invalidate()
        self.anchors.clear()
        self.anchor_inputs.clear()
        with self.evidence_lock:
            self.inference_events.clear()
        for key in ("inference_requests","stale_results","emitted_commands",
                    "deadline_misses","clock_rebases","clock_shift_ms",
                    "max_deadline_lateness_ms"):
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
            outcome=self.terminal_outcome,paused=sample.paused,timestamp=sample.timestamp,
            io_evidence=getattr(sample, 'io_evidence', None),
            received_monotonic=float(self.clock()))

    def observation(self, observation):
        result = dict(observation)
        result.pop('rtc', None)
        if self.config.rtc and self.plan is not None:
            pending = self.queue.remaining_actions()[:, :7]
            if len(pending):
                result['rtc'] = dict(previous_actions=pending.copy(), delay_steps=0,
                                     execution_horizon=min(self.config.replan_after_steps, len(pending)))
        return result

    def diagnostic_plan(self, observation, logical_step):
        # Numeric request evidence only. Never infer a plan's anchor from the
        # state present when a queued command later executes.
        event_id = self.plan_event_id
        self.plan_event_id += 1
        started = float(self.clock())
        epoch = self.epoch
        episode_id = getattr(self.backend, "episode_id", None)
        plan = self.backend.plan(observation, logical_step)
        event = dict(request_id=event_id, logical_step=int(logical_step),
                     execution_epoch=int(epoch), episode_id=episode_id,
                     request_started_monotonic=started,
                     request_finished_monotonic=float(self.clock()),
                     anchor_state=self.env._state(observation).copy(),
                     actor_param_version=int(plan.actor_param_version),
                     rtc_requested="rtc" in observation,
                     semantics="request and completed plan; not proof of queue acceptance")
        with self.evidence_lock:
            self.inference_events.append(event)
        return plan

    def take_inference_events(self):
        with self.evidence_lock:
            events, self.inference_events = self.inference_events, []
        return events

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
        self.future = self.pool.submit(self.diagnostic_plan, copy.deepcopy(request), self.env._episode_steps)
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
        from .input_audit import observation_receipt
        self.anchor_inputs[self.env._episode_steps] = observation_receipt(request)["sha256"]
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
        from .input_audit import observation_receipt
        self.pending = (self.epoch, snapshot, self.env._episode_steps, observation_receipt(request)["sha256"])
        self.future = self.pool.submit(self.diagnostic_plan, request, self.env._episode_steps)
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
        _, snapshot, step, input_identity = pending
        actions = np.asarray(plan.action_chunk, np.float32).copy()
        # Actor is downstream of RTC Reference. Preserve the committed prefix
        # after Actor refinement so it cannot undo inpainting's fixed actions.
        n = min(self.config.max_delay_steps, len(snapshot.previous_actions))
        actions[:n] = snapshot.previous_actions[:n, :7]
        bundle = np.concatenate([actions, plan.ref_chunk,
            np.full((len(actions),1),plan.actor_param_version,np.float32),
            np.full((len(actions),1),plan.source,np.float32)], -1)
        # Keep the reference belonging to each committed old action too.
        # Mixing a new reference with an old action changes Replay/BC provenance.
        bundle[:n] = snapshot.previous_actions[:n]
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
        self.anchor_inputs[step] = input_identity
        self.stats['last_actual_delay_steps'] = delay

    def _shift_clock(self, seconds):
        if seconds <= 1e-9:
            return
        # Bound slowdown within each logical C10 chunk, rather than accepting
        # indefinitely slow hardware. No training-time or logical-step change.
        window = self.env._episode_steps // self.env._chunk_exec_horizon
        if window != self.clock_window:
            self.clock_window = window
            self.clock_window_shift_sec = 0.
        self.clock_window_shift_sec += seconds
        if self.clock_window_shift_sec >= 1/self.config.logical_hz - 1e-9:
            raise RuntimeError(
                'Execution clock missed its deadline; no catch-up command burst; '
                f'cumulative_shift_ms={self.clock_window_shift_sec*1000:.1f}, '
                f'window_logical_steps={self.env._chunk_exec_horizon}, '
                f'publish_hz={self.config.publish_hz}, logical_step={self.env._episode_steps}')
        self.clock_shift_sec += seconds
        self.stats['clock_rebases'] = self.stats.get('clock_rebases', 0)+1
        self.stats['clock_shift_ms'] = self.clock_shift_sec*1000
        self.stats['clock_window_shift_ms'] = self.clock_window_shift_sec*1000

    def wait_until(self, timestamp):
        deadline = timestamp+self.clock_shift_sec
        # A late command must never compress the next physical interval. The
        # logical boundary itself is not a publication and has no such floor.
        if self.minimum_publication_time is not None:
            spacing_shift = max(0., self.minimum_publication_time-deadline)
            self._shift_clock(spacing_shift)
            deadline += spacing_shift
        remaining = deadline-self.clock()
        if remaining > 0:
            self.env._sleep(remaining)
        # Recheck AFTER sleep too; OS descheduling used to bypass the guard.
        lateness = max(0., self.clock()-deadline)
        self.stats['last_deadline_lateness_ms'] = lateness*1000
        self.stats['max_deadline_lateness_ms'] = max(
            self.stats.get('max_deadline_lateness_ms', 0.), lateness*1000)
        if lateness > .5/self.config.publish_hz:
            self.stats['deadline_misses'] = self.stats.get('deadline_misses', 0)+1
        if lateness >= 1/self.config.logical_hz - 1e-9:
            raise RuntimeError(
                'Execution clock missed its deadline; no catch-up command burst; '
                f'late_ms={lateness*1000:.1f}, publish_hz={self.config.publish_hz}, '
                f'logical_step={self.env._episode_steps}')
        # Bounded jitter shifts future deadlines. Do not burst delayed targets,
        # skip logical transitions, consume extra RTC rows, or invent commands.
        self._shift_clock(lateness)
        self.last_wait_deadline = deadline+lateness

    def wait_active(self, timestamp, epoch, *, publication=False):
        # Pause/HIL invalidates the old timeline. Check authority BEFORE its
        # deadline: an operator interruption is not a publisher overrun.
        def active():
            sample_started = float(self.clock())
            check = self.sample()
            self.stats['last_control_sample_ms'] = (self.clock()-sample_started)*1000
            state = self.env._runtime.observe_mode(check.mode)
            if state.phase is EpisodePhase.FAULT:
                raise RuntimeError(state.fault_reason or 'Control coordinator fault')
            if (epoch != self.epoch or check.paused or check.outcome is not None
                    or state.phase is not EpisodePhase.ROLLOUT):
                self.env._io.set_chunk_ready(False)
                if epoch == self.epoch:
                    self.invalidate()
                return False
            # Return the fresh post-wait feedback to the caller. Sampling it
            # again after this check used to add work to EVERY publication.
            return check
        if not active():
            return False
        self.minimum_publication_time = (
            self.last_publish_started+1/self.config.publish_hz
            if publication and self.last_publish_started is not None else None)
        try:
            self.wait_until(timestamp)
        except RuntimeError:
            # Pause can arrive after the precheck while this thread is
            # descheduled. Recheck authority before propagating a late clock.
            if not active():
                return False
            raise
        finally:
            self.minimum_publication_time = None
        return active()

    def execute_chunk(self, observation=None, policy_planner=None, control_hz=None):
        env, io, runtime = self.env, self.env._io, self.env._runtime
        if control_hz is not None and float(control_hz) != self.config.logical_hz:
            raise ValueError('Publication Hz must not replace logical Replay Hz')
        current = self.observation(observation or self.sample().observation)
        trace, rewards, outcome = [], [], None
        publications, publication_attempt = [], None
        paused_last = False
        if env._shadow_mode:
            raise ValueError('Use the offline execution audit for this profile; live shadow mode is unsupported')
        try:
            while len(trace) < env._chunk_exec_horizon:
                publications, publication_attempt = [], None
                loop_started = float(self.clock())
                self.stats['loop_started_monotonic'] = loop_started
                health = getattr(io, 'check_trace_health', None)
                if health is not None:
                    health()
                sample = self.sample()
                self.stats['last_loop_sample_ms'] = (self.clock()-loop_started)*1000
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
                hil_command_receipt = None
                logical_proposal = logical_reference = None
                if state.phase is EpisodePhase.HIL:
                    if self.plan is not None:
                        io.set_chunk_ready(False)
                        self.invalidate()
                    current = dict(current)
                    current.pop('rtc', None)
                    if env._hil_target == 'coordinator_command':
                        from .hil_targets import right_command_at_step_start
                        hil_command, hil_command_receipt = right_command_at_step_start(sample, runtime.snapshot().expert_mask)
                    env._sleep(1/self.config.logical_hz)
                    after = self.sample()
                    executed = hil_command if hil_command_receipt is not None else env._state(after.observation)
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
                    acceptance_started = float(self.clock())
                    self.check_report()
                    self.accept_result()
                    self.stats['last_plan_accept_ms'] = (self.clock()-acceptance_started)*1000
                    # Start of one logical20 segment. Queue consumption, model
                    # delay and Replay step IDs all use this clock, never pub Hz.
                    current = self.observation(current)
                    target_bundle = self.queue.pop()
                    target, ref_target = target_bundle[:7], target_bundle[7:14]
                    logical_proposal, logical_reference = target.copy(), ref_target.copy()
                    version, source = int(target_bundle[14]), ControlSource(int(target_bundle[15]))
                    self.last_policy_source = source
                    # This request starts at the NEXT logical boundary, below.
                    publications = []
                    epoch = self.epoch
                    interval_origin = self.timeline_origin
                    interval_index = self.publication_index
                    for timestamp, alpha in self.events(self.publication_index, 20, self.config.publish_hz):
                        # Interpolation and the causal filter depend on the last
                        # emitted target, not on a future camera frame. Prepare
                        # them in idle time; retain fresh feedback/authority and
                        # the safety clamp immediately before publication.
                        prepare_started = float(self.clock())
                        publication_attempt = dict(
                            nominal_scheduled_monotonic=float(interval_origin+timestamp),
                            prepare_started_monotonic=prepare_started, published=False,
                            actor_param_version=int(version), execution_epoch=int(epoch))
                        requested = self.segment_start + alpha*(target-self.segment_start)
                        filtered = self.filter.apply(requested, self.last_command, 1/self.config.publish_hz)
                        filtered[6] = self.last_command[6] + np.clip(filtered[6]-self.last_command[6],
                            -self.config.gripper_velocity_limit/self.config.publish_hz,
                            self.config.gripper_velocity_limit/self.config.publish_hz)
                        prepare_finished = float(self.clock())
                        publication_attempt['target_prepare_ms'] = (prepare_finished-prepare_started)*1000
                        check = self.wait_active(interval_origin+timestamp, epoch, publication=True)
                        if not check:
                            break
                        effective_deadline = self.last_wait_deadline
                        publication_attempt['scheduled_monotonic'] = float(effective_deadline)
                        # wait_active already sampled and validated this state.
                        # Clamp against that current feedback, not the pre-wait
                        # filter input. A pause racing after it is rejected by IO.
                        safety_started = float(self.clock())
                        command = runtime.safe_policy_target(filtered, env._state(check.observation))
                        limits = np.full(7,self.config.joint_velocity_limit/self.config.publish_hz)
                        limits[6] = self.config.gripper_velocity_limit/self.config.publish_hz
                        if np.any(np.abs(command-self.last_command)>limits+1e-6):
                            raise RuntimeError("Feedback safety clamp would break physical rate limit")
                        publish_started = float(self.clock())
                        preparation_lateness = max(0., publish_started-effective_deadline)
                        publication_attempt.update(
                            publish_started_monotonic=publish_started,
                            control_sample_ms=self.stats['last_control_sample_ms'],
                            safety_check_ms=(publish_started-safety_started)*1000,
                            post_wait_lateness_ms=preparation_lateness*1000)
                        # Keep both single-stall and C10 cumulative bounds. Work
                        # moved before the wait is absorbed, not hidden/forgiven.
                        if preparation_lateness >= 1/self.config.logical_hz - 1e-9:
                            raise RuntimeError(
                                'Execution clock missed its deadline; no catch-up command burst; '
                                f'late_ms={preparation_lateness*1000:.1f}, '
                                f'publish_hz={self.config.publish_hz}, logical_step={env._episode_steps}; '
                                'control preparation exceeded budget')
                        self._shift_clock(preparation_lateness)
                        self.stats['last_target_prepare_ms'] = publication_attempt['target_prepare_ms']
                        self.stats['last_safety_check_ms'] = publication_attempt['safety_check_ms']
                        # If IO raises mid-call, publication status is unknown;
                        # do not report an unobserved rejection as "not sent".
                        publication_attempt['published'] = None
                        accepted = io.publish_policy_action(command)
                        publication_attempt['published'] = accepted is not False
                        publish_finished = float(self.clock())
                        self.last_publish_started = publish_started
                        self.stats["last_publish_duration_ms"] = (publish_finished-publish_started)*1000
                        if accepted is False:
                            io.set_chunk_ready(False)
                            self.invalidate()
                            break
                        publication_attempt['published'] = True
                        publication_attempt['publish_finished_monotonic'] = publish_finished
                        self.last_command = command.copy()
                        self.filter.previous = command.copy()
                        self.last_ref = self.segment_ref_start + alpha*(ref_target-self.segment_ref_start) if hasattr(self,'segment_ref_start') else ref_target.copy()
                        publications.append(dict(timestamp=float(io.ros.Time.now().to_sec()) if hasattr(getattr(io,'ros',None),'Time') else float(check.timestamp),
                                                 monotonic_timestamp=publish_finished, action=command.copy(),
                                                 scheduled_monotonic=float(effective_deadline),
                                                 nominal_scheduled_monotonic=float(interval_origin+timestamp),
                                                 clock_shift_sec=float(self.clock_shift_sec),
                                                 publish_started_monotonic=publish_started,
                                                 publish_finished_monotonic=publish_finished,
                                                 control_sample_ms=publication_attempt['control_sample_ms'],
                                                 target_prepare_ms=publication_attempt['target_prepare_ms'],
                                                 prepare_started_monotonic=prepare_started,
                                                 prepare_finished_monotonic=prepare_finished,
                                                 safety_check_ms=publication_attempt['safety_check_ms'],
                                                 post_wait_lateness_ms=preparation_lateness*1000,
                                                 publish_duration_ms=(publish_finished-publish_started)*1000,
                                                 feedback_received_monotonic=check.received_monotonic,
                                                 feedback_state=env._state(check.observation).copy(),
                                                 execution_epoch=int(epoch),
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
                        if env._hil_target == 'coordinator_command':
                            raise ValueError('Policy-to-HIL boundary has no single start-of-step human command; Episode not submitted')
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
                    execution_settings=dict(logical_hz=self.config.logical_hz,
                        publish_hz=self.config.publish_hz, rtc=self.config.rtc,
                        smoothing_tau_sec=self.config.smoothing_tau_sec,
                        joint_velocity_limit=self.config.joint_velocity_limit,
                        gripper_velocity_limit=self.config.gripper_velocity_limit,
                        replan_after_steps=self.config.replan_after_steps,
                        max_delay_steps=self.config.max_delay_steps),
                    io_evidence_before_step=getattr(sample, 'io_evidence', None),
                    io_evidence_after_step=getattr(after, 'io_evidence', None),
                    publications=publications,
                    interval_start_sample_received_monotonic=sample.received_monotonic,
                    sample_received_monotonic=after.received_monotonic,
                    planned_action=logical_proposal, logical_ref_action=logical_reference,
                    proposal_semantics="queued logical Actor target before interpolation/filter/clamp; no fresh HIL counterfactual" if logical_proposal is not None else "not_recorded_during_HIL",
                    inference_events=self.take_inference_events())
                record['hil_target_mode'] = env._hil_target
                record['hil_command_receipt'] = hil_command_receipt
                if hil_command_receipt is not None:
                    record['action_semantics'] = 'coordinator_command_at_step_start'
                trace.append(record); rewards.append(reward)
                record_started = float(self.clock())
                io.record_raw_step(record)
                self.stats["last_record_step_ms"] = (self.clock()-record_started)*1000
                current = next_observation
                if outcome is not None:
                    break
                if runtime.snapshot().phase is EpisodePhase.ROLLOUT and not after.paused and self.plan is not None:
                    request_started = float(self.clock())
                    self.request_next(current)
                    self.stats['last_request_submit_ms'] = (self.clock()-request_started)*1000
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
            # Do not fabricate a logical transition for a partial interval.
            # The owned process log retains physical receipts even when normal
            # record_raw_step was never reached. Pause takes priority over I/O.
            fault = dict(event="execution_fault", error=str(exc),
                episode_id=getattr(self.backend, 'episode_id', None),
                logical_step=int(env._episode_steps), execution_epoch=int(self.epoch),
                monotonic_timestamp=float(self.clock()), replay_eligible=False,
                partial_publications=publications, attempt=publication_attempt,
                clock_window_shift_ms=self.clock_window_shift_sec*1000,
                clock_shift_ms=self.clock_shift_sec*1000,
                completed_logical_rows_in_call=len(trace), execution_stats=dict(self.stats),
                trace_storage=getattr(io, 'trace_diagnostics', lambda: {})())
            # Flush only AFTER revoking publication. An unwritable trace stays
            # incomplete and must never be admitted to Replay.
            flush = getattr(io, 'flush_raw_trace', None)
            if flush is not None:
                try:
                    flush(timeout=2.0)
                except Exception as flush_error:
                    fault['trace_flush_error'] = str(flush_error)
            try:
                print('[rlt-execution-fault] '+json.dumps(fault,
                    default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()),
                    flush=True)
            except Exception as log_error:
                self.stats['fault_log_error'] = str(log_error)
            application = getattr(io, "_session_application", None)
            if application is not None:
                application._controller.fail("execution_timing_failed: " + str(exc))
            self.invalidate()
            raise
