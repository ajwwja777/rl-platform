"""Opt-in RLT execution configuration, independent of model/ROS environments."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ExecutionProfile:
    logical_hz: int = 20
    publish_hz: int = 40
    replan_after_steps: int = 5
    max_delay_steps: int = 4
    smoothing_tau_sec: float = .08
    joint_velocity_limit: float = .6
    gripper_velocity_limit: float = .08
    rtc: bool = True

    def __post_init__(self):
        if self.logical_hz != 20 or self.publish_hz not in (20, 30, 40, 50):
            raise ValueError('Current RLT Actor/Replay require logical20; choose publication20/30/40/50')
        if type(self.replan_after_steps) is not int or type(self.max_delay_steps) is not int:
            raise ValueError('RTC horizons must be integers')
        if not 0 < self.max_delay_steps < 10-self.replan_after_steps or self.replan_after_steps < 1:
            raise ValueError('RTC delay must leave a usable pending prefix')
        if not 0 <= self.smoothing_tau_sec <= .5 or not 0 < self.joint_velocity_limit <= .6 or not 0 < self.gripper_velocity_limit <= .08:
            raise ValueError('Invalid physical-time conditioning')
        if type(self.rtc) is not bool:
            raise ValueError('rtc must be boolean')


def normalize_options(options):
    if not isinstance(options, dict) or set(options) - {'enabled', 'publish_hz', 'rtc', 'smoothing'}:
        raise ValueError('invalid_execution_options')
    if type(options.get('enabled')) is not bool:
        raise ValueError('execution_options_enabled_must_be_boolean')
    if not options['enabled']:
        if set(options) != {'enabled'}:
            raise ValueError('disabled_execution_options_use_model_defaults')
        return {'enabled': False}
    if set(options) != {'enabled', 'publish_hz', 'rtc', 'smoothing'}:
        raise ValueError('execution_options_require_hz_rtc_smoothing')
    if type(options['publish_hz']) is not int or type(options['rtc']) is not bool or type(options['smoothing']) is not bool:
        raise ValueError('invalid_execution_option_types')
    ExecutionProfile(publish_hz=options['publish_hz'], rtc=options['rtc'],
                     smoothing_tau_sec=.08 if options['smoothing'] else 0.)
    return dict(options)


def describe_execution(model, root=ROOT):
    options = model.get('execution_options')
    if options is not None:
        options = normalize_options(options)
    if options and options['enabled']:
        return {**options, 'logical_hz': 20, 'profile': 'user_options', 'asynchronous': True}
    name = model.get('execution_profile', 'faithful20')
    if name == 'faithful20':
        return dict(enabled=False, publish_hz=model.get('control_hz', 20), logical_hz=20,
                    rtc=False, smoothing=False, profile=name, asynchronous=False)
    registry = json.loads((Path(root)/'configs/execution_profiles.json').read_text())
    profile = ExecutionProfile(**registry['profiles'][name])
    return dict(enabled=False, publish_hz=profile.publish_hz, logical_hz=20,
                rtc=profile.rtc, smoothing=profile.smoothing_tau_sec > 0,
                profile=name, asynchronous=True)


def selected_profile(root=ROOT, environ=None):
    environ = os.environ if environ is None else environ
    raw = environ.get('COBOT_RLT_EXECUTION_OPTIONS')
    if raw:
        options = normalize_options(json.loads(raw))
        if options['enabled']:
            return 'user_options', ExecutionProfile(publish_hz=options['publish_hz'], rtc=options['rtc'],
                        smoothing_tau_sec=.08 if options['smoothing'] else 0.)
    name = environ.get('COBOT_RLT_EXECUTION_PROFILE', '')
    if not name:
        models = json.loads((Path(root)/'configs/deployment_models.json').read_text())['models']
        row = next((x for x in models if x['id'] == environ.get('COBOT_DEPLOYMENT_MODEL_ID')), {})
        name = row.get('execution_profile', '')
    if not name or name == 'faithful20':
        return None
    registry = json.loads((Path(root)/'configs/execution_profiles.json').read_text())
    if name not in registry['profiles']:
        raise ValueError('Unknown execution profile: '+name)
    return name, ExecutionProfile(**registry['profiles'][name])


def load_shared(root=ROOT):
    runtime = Path(root).parent/'vla-platform/integrations/cobot/pi05/dagger/common/runtime_lib'
    if not (runtime/'execution_methods/execution_timing.py').is_file():
        raise RuntimeError('Deploy the registered sibling vla-platform execution components')
    sys.path.insert(0, str(runtime))
    from execution_methods.rtc.action_queue import ThreadSafeActionChunk
    from execution_methods.execution_timing import publication_events, CausalJointFilter
    return ThreadSafeActionChunk, publication_events, CausalJointFilter


if __name__ == '__main__':
    selected = selected_profile()
    if selected:
        print(selected[0])
        print(int(selected[1].rtc))
