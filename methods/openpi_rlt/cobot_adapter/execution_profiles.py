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


def selected_profile(root=ROOT, environ=None):
    environ = os.environ if environ is None else environ
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
