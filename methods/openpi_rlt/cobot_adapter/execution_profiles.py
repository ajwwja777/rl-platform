"""Opt-in RLT execution configuration, independent of model/ROS environments."""
from dataclasses import dataclass, replace
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
    smoothing_order: int = 1

    def __post_init__(self):
        if self.logical_hz != 20 or self.publish_hz not in (20, 30, 40, 50):
            raise ValueError('Current RLT Actor/Replay require logical20; choose publication20/30/40/50')
        if type(self.replan_after_steps) is not int or type(self.max_delay_steps) is not int:
            raise ValueError('RTC horizons must be integers')
        if not 0 < self.max_delay_steps < 10-self.replan_after_steps or self.replan_after_steps < 1:
            raise ValueError('RTC delay must leave a usable pending prefix')
        if not 0 <= self.smoothing_tau_sec <= .5 or not 0 < self.joint_velocity_limit <= .6 or not 0 < self.gripper_velocity_limit <= .08:
            raise ValueError('Invalid physical-time conditioning')
        if type(self.smoothing_order) is not int or self.smoothing_order not in (1,2):
            raise ValueError('Smoothing order must be 1 or 2')
        if self.smoothing_order == 2 and self.smoothing_tau_sec <= 0:
            raise ValueError('Two-pole filtering requires positive tau')
        if type(self.rtc) is not bool:
            raise ValueError('rtc must be boolean')


SMOOTH_MODEL_ID = 'plug-v3-hil8-explore-smooth'


def candidate_profile(profile, model_id):
    if model_id != SMOOTH_MODEL_ID:
        return profile
    if profile.publish_hz != 50 or not profile.rtc or profile.smoothing_tau_sec <= 0:
        raise ValueError('RLT smooth candidate requires 50 Hz, RTC and smoothing enabled')
    return replace(profile, smoothing_tau_sec=.12, smoothing_order=2)


def normalize_options(options):
    import runpy
    shared = ROOT.parent/'vla-platform/integrations/cobot/execution_options.py'
    return runpy.run_path(str(shared))['normalize_options'](options)


def describe_execution(model, root=ROOT):
    options = model.get('execution_options')
    if options is not None:
        options = normalize_options(options)
    if options and options['enabled']:
        result = {**options, 'logical_hz': 20, 'profile': 'user_options', 'asynchronous': True}
        if model.get('id') == SMOOTH_MODEL_ID:
            candidate_profile(ExecutionProfile(publish_hz=options['publish_hz'],rtc=options['rtc'],smoothing_tau_sec=.08 if options['smoothing'] else 0.), SMOOTH_MODEL_ID)
            result.update(smoothing_order=2,smoothing_tau_sec=.12)
        return result
    name = model.get('execution_profile', 'faithful20')
    if name == 'faithful20':
        return dict(enabled=False, publish_hz=model.get('control_hz', 20), logical_hz=20,
                    rtc=False, smoothing=False, profile=name, asynchronous=False)
    registry = json.loads((Path(root)/'configs/execution_profiles.json').read_text())
    profile = candidate_profile(ExecutionProfile(**registry['profiles'][name]),model.get('id'))
    return dict(enabled=False, publish_hz=profile.publish_hz, logical_hz=20,
                rtc=profile.rtc, smoothing=profile.smoothing_tau_sec > 0,
                profile=name, asynchronous=True, smoothing_order=profile.smoothing_order,
                smoothing_tau_sec=profile.smoothing_tau_sec)


def selected_profile(root=ROOT, environ=None):
    environ = os.environ if environ is None else environ
    raw = environ.get('COBOT_EXECUTION_OPTIONS') or environ.get('COBOT_RLT_EXECUTION_OPTIONS')
    if raw:
        options = normalize_options(json.loads(raw))
        if options['enabled']:
            return 'user_options', candidate_profile(ExecutionProfile(publish_hz=options['publish_hz'], rtc=options['rtc'],
                        smoothing_tau_sec=.08 if options['smoothing'] else 0.),environ.get('COBOT_DEPLOYMENT_MODEL_ID'))
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
    return name, candidate_profile(ExecutionProfile(**registry['profiles'][name]),environ.get('COBOT_DEPLOYMENT_MODEL_ID'))


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
