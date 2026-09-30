"""Project-owned protocol seams for optional execution; fixed upstream is untouched."""
import time
import numpy as np
from .execution_profiles import selected_profile


class PlannerBackend:
    def __init__(self, driver):
        self.driver = driver
        self.episode_id = -1
        self.request_id = 0

    def plan(self, observation, logical_step):
        from rlt_online_rl.inference import normalize_feature_payload, _chunk_features_from_payload, maybe_refine_chunk, PolicyPlan
        started = time.monotonic()
        d = self.driver
        raw = d._feature_provider.get_features(observation)
        if 'rtc' in observation and raw.get('rtc_used') is not True:
            raise RuntimeError('Stage1 did not confirm RTC; refusing an unconditioned replacement')
        payload = normalize_feature_payload(raw, d._rl_config, observation=observation)
        features = _chunk_features_from_payload(payload, d._rl_config)
        result = maybe_refine_chunk(d._actor_client, z_rl=features.z_rl, proprio=features.proprio,
            ref_chunk=features.ref_chunk, request_id=f'rtc:{self.episode_id}:{logical_step}:{self.request_id}',
            episode_id=self.episode_id, step_id=logical_step,
            deterministic=d._env_config.actor_deterministic, on_error_fallback=d._env_config.safe_fallback_to_ref)
        self.request_id += 1
        actions = np.asarray(result.refined_chunk, np.float32)
        if d._safe_action_filter is not None:
            actions = d._safe_action_filter(actions)
        inferred = time.monotonic()
        d._env._execution.stats["last_model_inference_ms"] = (inferred-started)*1000
        d._env._io.report_chunk(latency_sec=inferred-started, actor_version=int(result.actor_param_version))
        d._env._execution.stats["last_recorder_check_ms"] = (time.monotonic()-inferred)*1000
        return PolicyPlan(action_chunk=actions, ref_chunk=features.ref_chunk, source=int(result.source),
                          start_features=features, actor_param_version=int(result.actor_param_version))


def install():
    """Only a selected profile can activate hooks; original functions are otherwise exact."""
    if selected_profile() is None:
        return
    from rlt_online_rl import inference
    cls = inference.EnvDriver
    if getattr(cls, '_cobot_execution_installed', False):
        return
    original_init, original_episode = cls.__init__, cls.run_episode
    original_append, original_close = cls._append_raw_chunk, cls.close

    def init(driver, *args, **kwargs):
        original_init(driver, *args, **kwargs)
        engine = getattr(driver._env, '_execution', None)
        if engine is None:
            raise RuntimeError('Selected execution profile needs the Cobot execution environment')
        if driver._env_config.enable_human_override:
            raise ValueError('Use authoritative Task2 HIL; optional upstream human override is unsupported here')
        engine.backend = PlannerBackend(driver)
        application = getattr(driver._env._io, "_session_application", None)
        if application is not None:
            status = application.status
            application.status = lambda: {**status(), "execution": dict(engine.stats)}


    def episode(driver, episode_id):
        engine = driver._env._execution
        engine.set_episode(episode_id)
        result = original_episode(driver, episode_id)
        result['execution'] = dict(engine.stats)
        return result

    def append(driver, raw_episode, **kwargs):
        result = original_append(driver, raw_episode, **kwargs)
        engine = driver._env._execution
        for step, features in list(engine.anchors.items()):
            if step < len(raw_episode.steps):
                driver._record_feature_anchor(raw_episode, raw_episode.steps[step].observation_idx, features)
                raw_episode.policy_start_steps.append(step)
                del engine.anchors[step]
        return result

    def close(driver):
        engine = getattr(driver._env, '_execution', None)
        try:
            if engine is not None:
                engine.close()
        finally:
            original_close(driver)

    cls.__init__, cls.run_episode, cls._append_raw_chunk, cls.close = init, episode, append, close
    cls._cobot_execution_installed = True
