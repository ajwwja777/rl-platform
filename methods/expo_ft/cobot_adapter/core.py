"""Pure control ownership and physical-action contract for the native EXPO actor."""
import math
import numpy as np

COORDINATORS = {"task2_teach_handover", "task2_teach_button_handover", "task2_teach_handover_node"}


class Gate:
    def __init__(self):
        self.manual_pause = True
        self.hil_active = False
        self.mode = "unknown"
        self.generation = 0
        self.intervention_count = 0
        self.fault = None

    @property
    def paused(self):
        return self.manual_pause or self.hil_active or self.mode != "policy" or bool(self.fault)

    def request(self, paused, caller):
        node = caller.rsplit("/", 1)[-1]
        operator = node == "cobot_deployment_command" or node.startswith("cobot_deployment_command_")
        if operator:
            if not paused and (self.hil_active or self.mode != "policy"):
                return False, "Teaching/fault is active; keep paused"
            self.manual_pause = paused
            if not paused:
                self.fault = None
        elif node in COORDINATORS:
            if paused and not self.hil_active:
                self.intervention_count += 1
            self.hil_active = paused
        elif paused:
            self.manual_pause = True
        else:
            return False, "Protective pause requires operator Continue"
        self.generation += 1
        return True, "paused" if self.paused else "fresh resume requested"

    def update_mode(self, mode):
        previous = self.mode
        self.mode = mode
        active = mode.startswith("manual:")
        if active and not self.hil_active:
            self.intervention_count += 1
        if active != self.hil_active or previous != mode:
            self.generation += 1
        self.hil_active = active
        if mode == "fault":
            self.fail("hardware handover fault")

    def fail(self, reason):
        self.manual_pause = True
        self.fault = str(reason)
        self.generation += 1


def validate_observation(observation):
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("Expected finite left-arm 7D feedback")
    for key in ("left_wrist_image", "right_wrist_image"):
        image = np.asarray(observation[key])
        if image.shape != (180, 320, 3) or image.dtype != np.uint8 or int(image.max()) <= 1:
            raise ValueError("Invalid required RGB image: " + key)
    return observation


def limited_plan(actions, anchor, smoothing=False):
    actions = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(anchor, dtype=np.float64).copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Invalid EXPO action chunk")
    if previous.shape != (7,) or not np.isfinite(previous).all():
        raise ValueError("Invalid command anchor")
    limits = np.array([.03] * 6 + [.004])  # per 0.1s logical step, rad/meters
    result = []
    for value in actions:
        target = value.copy()
        target[6] = np.clip(target[6], 0., .08)
        if smoothing:
            target = previous + (1. - math.exp(-.1 / .08)) * (target - previous)
        previous += np.clip(target - previous, -limits, limits)
        result.append(previous.copy())
    return np.asarray(result, dtype=np.float32)


def execution_options(raw):
    if isinstance(raw, dict) and 'expo' in raw:
        from online_options import normalize
        normalize(raw['expo'])
        raw = {key: value for key, value in raw.items() if key != 'expo'}
    if not raw or raw == {"enabled": False}:
        return {"publish_hz": 10, "rtc": True, "smoothing": False}
    if set(raw) != {"enabled", "publish_hz", "rtc", "smoothing"} or raw["enabled"] is not True:
        raise ValueError("Invalid EXPO execution options")
    if type(raw["publish_hz"]) is not int or raw["publish_hz"] not in (10, 20, 30, 40, 50):
        raise ValueError("EXPO publisher supports 10/20/30/40/50Hz overrides, logical rate remains 10Hz")
    if type(raw["rtc"]) is not bool or type(raw["smoothing"]) is not bool:
        raise ValueError("RTC/smoothing must be booleans")
    return {key: raw[key] for key in ("publish_hz", "rtc", "smoothing")}
