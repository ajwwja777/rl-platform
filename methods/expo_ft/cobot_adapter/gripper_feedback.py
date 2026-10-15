"""Opening-only feedback anchor. RTC committed prefixes remain immutable."""
import math
import time
import numpy as np
from core import limited_plan

STEP = .004  # metres per 100ms, unchanged
DEADBAND = .001

class FeedbackPlan:
    def __init__(self, actions, anchor, smoothing, rtc, delay):
        self.raw = np.asarray(actions, dtype=np.float64).copy()
        self.plan = limited_plan(actions, anchor, smoothing)
        self.smoothing, self.rtc, self.delay = smoothing, rtc, delay
        self.starts = np.r_[float(anchor[6]), self.plan[:-1,6]].copy()
        self.committed = False

    def prepare(self, index, last, actual):
        if not np.isfinite(actual) or not 0 <= actual <= .08:
            raise ValueError('Invalid gripper position feedback')
        if self.committed:
            return self.plan
        previous = float(last[6])
        for j in range(index, len(self.plan)):
            desired = float(np.clip(self.raw[j,6], 0., .08))
            # Contact can leave command=0 while feedback=27mm. Re-anchor only
            # when policy explicitly requests opening beyond measured position.
            base = actual if desired > actual + DEADBAND and previous < actual - DEADBAND else previous
            self.starts[j] = base
            target = base + (1.-math.exp(-.1/.08))*(desired-base) if self.smoothing else desired
            previous = base + float(np.clip(target-base,-STEP,STEP))
            self.plan[j,6] = previous
        if self.rtc and index == 8-self.delay:
            self.committed = True
        return self.plan

    def interpolate(self, last, target, fraction, index):
        value = last + (target-last)*fraction
        value[6] = self.starts[index] + (target[6]-self.starts[index])*fraction
        return value


class FeedbackExecution:
    """Shared by field-accepted frozen preview and opt-in Online runtime."""
    def feedback_enabled(self):
        return bool(getattr(self.args, 'gripper_feedback', False))

    def make_plan(self, actions, last):
        if not self.feedback_enabled():
            return super().make_plan(actions, last)
        self.feedback_plan = FeedbackPlan(actions,last,self.options['smoothing'],self.options['rtc'],self.rtc_delay)
        return self.feedback_plan.plan

    def prepare_plan(self, plan, index, last):
        if not self.feedback_enabled():
            return plan
        with self.lock:
            feedback = self.samples.get('state')
            if feedback is None or time.monotonic()-feedback[1] > .25:
                raise ValueError('Left feedback expired before gripper planning')
            stamp = feedback[0].header.stamp.to_sec()
            if stamp and self.ros.Time.now().to_sec()-stamp > .25:
                raise ValueError('Left feedback timestamp expired before gripper planning')
            actual = float(feedback[0].position[6])
        return self.feedback_plan.prepare(index,last,actual)

    def interpolate_command(self, last, target, fraction, index):
        if not self.feedback_enabled():
            return super().interpolate_command(last,target,fraction,index)
        return self.feedback_plan.interpolate(last,target,fraction,index)
