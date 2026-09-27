# Generation-safe RTC command queue. No ROS/CAN imports or robot publisher.
from dataclasses import dataclass
from threading import RLock
import numpy as np

# Match the Piper firmware/ROS driver limits exactly (0.017444 rad per degree).
PIPER_RAD_PER_DEG = np.float32(0.017444)
PIPER_LOWER = np.asarray([-150., 0., -170., -100., -70., -120.], np.float32) * PIPER_RAD_PER_DEG
PIPER_UPPER = np.asarray([150., 180., 0., 100., 70., 120.], np.float32) * PIPER_RAD_PER_DEG

PIPER_WORKSPACE_MIN = np.asarray([-.015, -.015, -.110], np.float32)
PIPER_WORKSPACE_MAX = np.asarray([.125, .080, .012], np.float32)
PIPER_WORKSPACE_RADIUS = np.float32(.160)

# Exact joint1..joint6 origins and signed Z axes from the deployed Piper URDF.
# This tiny NumPy FK keeps the 30 Hz safety process independent of Pinocchio.
_PIPER_CHAIN = (
    ((0., 0., .123), (0., 0., -1.5708), 1.),
    ((0., 0., 0.), (1.5708, -.10095, -1.5708), 1.),
    ((.28503, 0., 0.), (0., 0., 1.3826), 1.),
    ((.021984, .25075, 0.), (-1.5708, 0., 0.), -1.),
    ((0., 0., 0.), (1.5708, -.087266, 0.), 1.),
    ((0., .091, .0014165), (-1.5708, -1.5708, 0.), -1.),
)

def _rpy_rotation(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = np.asarray([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = np.asarray([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    return rz @ ry @ rx

def piper_link6_xyz(joints):
    q = np.asarray(joints, np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('invalid Piper joint vector for workspace guard')
    transform = np.eye(4)
    for angle, (xyz, rpy, axis_sign) in zip(q, _PIPER_CHAIN):
        origin = np.eye(4)
        origin[:3, :3] = _rpy_rotation(*rpy)
        origin[:3, 3] = xyz
        theta = axis_sign * angle
        c, s = np.cos(theta), np.sin(theta)
        joint = np.eye(4)
        joint[:3, :3] = ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))
        transform = transform @ origin @ joint
    return transform[:3, 3].astype(np.float32)

class PiperWorkspaceGuard:
    """Fail closed when the right link6 leaves the successful plug envelope."""

    def __init__(self, start_state, lower=PIPER_WORKSPACE_MIN,
                 upper=PIPER_WORKSPACE_MAX, max_radius=PIPER_WORKSPACE_RADIUS):
        state = np.asarray(start_state, np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('invalid episode start state for workspace guard')
        self.start_xyz = piper_link6_xyz(state[7:13])
        self.lower = np.asarray(lower, np.float32).copy()
        self.upper = np.asarray(upper, np.float32).copy()
        self.max_radius = float(max_radius)
        if self.lower.shape != (3,) or self.upper.shape != (3,) or np.any(self.lower >= self.upper):
            raise ValueError('invalid workspace envelope')
        if self.max_radius <= 0:
            raise ValueError('invalid workspace radius')

    def _validate_joints(self, joints, source, index=None):
        delta = piper_link6_xyz(joints) - self.start_xyz
        axes = ('x', 'y', 'z')
        for axis, value, low, high in zip(axes, delta, self.lower, self.upper):
            if value < low or value > high:
                where = '' if index is None else f' index={index}'
                raise ValueError(
                    f'workspace_envelope_exceeded: source={source}{where} axis={axis} '
                    f'delta={value:.6f} allowed=[{low:.6f},{high:.6f}]'
                )
        radius = float(np.linalg.norm(delta))
        if radius > self.max_radius:
            where = '' if index is None else f' index={index}'
            raise ValueError(
                f'workspace_envelope_exceeded: source={source}{where} radius={radius:.6f} '
                f'allowed<={self.max_radius:.6f}'
            )
        return delta

    def validate_measured(self, state):
        state = np.asarray(state, np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('invalid measured state for workspace guard')
        return self._validate_joints(state[7:13], 'measured')

    def validate_plan(self, plan, start, stop):
        commands = np.asarray(plan, np.float32)
        if commands.shape != (50, 14) or not np.isfinite(commands).all():
            raise ValueError('invalid conditioned plan for workspace guard')
        if not 0 <= start < stop <= len(commands):
            raise ValueError('invalid workspace plan interval')
        for index in range(start, stop):
            self._validate_joints(commands[index, 7:13], 'plan', index)

class QueueFault(RuntimeError):
    pass

def vector(value, shape):
    array = np.asarray(value, np.float32).copy()
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('nonfinite or invalid command shape')
    array.setflags(write=False)
    return array

@dataclass(frozen=True)
class Request:
    generation: int
    sequence: int
    start_tick: int
    state: np.ndarray
    prefix: np.ndarray
    prefix_length: int

class RTCQueue:
    # Commands are final conditioned absolute commands, rather than raw VLA output.
    # One request each ten ticks; six already-committed ticks cover inference.
    def __init__(self, horizon=50, chunk=10, delay=6, action_dim=14, tracking_joints=None):
        if not 0 <= delay <= chunk <= horizon-delay:
            raise ValueError('invalid trained RTC/execution horizons')
        self.horizon,self.chunk,self.delay,self.action_dim=horizon,chunk,delay,action_dim
        joints=np.r_[0:6,7:13] if tracking_joints is None else np.asarray(tracking_joints,dtype=np.int64)
        if joints.ndim!=1 or not len(joints) or len(set(joints.tolist()))!=len(joints) or np.any((joints<0)|(joints>=action_dim)):
            raise ValueError('invalid tracking joints')
        self.tracking_joints=joints
        self._lock=RLock(); self.generation=0; self.sequence=0
        self.last_request_tick=None;self.next_request_tick=0
        self.next_tick=0; self.active=False; self.fault=None; self.pending=None; self.commands={}
    def pause(self, reason=None):
        with self._lock:
            self.generation+=1; self.active=False; self.fault=reason
            self.pending=None; self.commands={}
    def resume(self):
        with self._lock:
            self.generation+=1; self.active=True; self.fault=None
            self.next_tick=0; self.last_request_tick=None; self.next_request_tick=0; self.pending=None; self.commands={}
    def request(self, measured):
        with self._lock:
            if not self.active: raise QueueFault('policy_paused')
            if self.pending is not None: raise QueueFault('request_already_pending')
            d=self.delay if self.commands else 0
            prefix=np.zeros((self.horizon,self.action_dim),np.float32)
            for i in range(d):
                if self.next_tick+i not in self.commands:
                    self.pause('insufficient_committed_prefix'); raise QueueFault(self.fault)
                prefix[i]=self.commands[self.next_tick+i]
            self.sequence+=1
            req=Request(self.generation,self.sequence,self.next_tick,
                        vector(measured,(self.action_dim,)),vector(prefix,prefix.shape),d)
            self.pending=req
            self.last_request_tick=self.next_tick
            self.next_request_tick=self.next_tick+(self.chunk-self.delay if d==0 else self.chunk)
            return req
    def complete(self, request, conditioned_plan):
        with self._lock:
            if not self.active or self.pending is not request or request.generation!=self.generation:
                return False
            plan=vector(conditioned_plan,(self.horizon,self.action_dim))
            d=request.prefix_length
            if not np.array_equal(plan[:d],request.prefix[:d]):
                self.pause('committed_prefix_modified'); raise QueueFault(self.fault)
            first=request.start_tick+d
            if self.next_tick>first:
                self.pause('rtc_deadline_missed'); raise QueueFault(self.fault)
            # Only actor-owned C10 commands are committed. Request the next plan
            # six ticks before this chunk ends; no uncorrected VLA tail is queued.
            for i in range(d,self.chunk+d):
                self.commands[request.start_tick+i]=plan[i]
            self.pending=None
            return True
    def pop(self, measured=None, tracking_bound=.04):
        with self._lock:
            if not self.active: raise QueueFault('policy_paused')
            req=self.pending
            if req is not None and (req.prefix_length==0 or self.next_tick>=req.start_tick+req.prefix_length):
                if req.prefix_length:
                    self.pause('rtc_deadline_missed')
                raise QueueFault(self.fault or 'waiting_initial_inference')
            command=self.commands.get(self.next_tick)
            if command is None:
                self.pause('rtc_queue_empty'); raise QueueFault(self.fault)
            if measured is not None:
                state=vector(measured,(self.action_dim,))
                joints=self.tracking_joints
                delta=np.abs(command[joints]-state[joints])
                if np.max(delta)>tracking_bound:
                    local=int(np.argmax(delta));joint=int(joints[local]);detail=(f'tracking_bound_exceeded: joint={joint} delta={delta[local]:.6f} '+
                      f'bound={tracking_bound:.6f} command={command[joint]:.6f} measured={state[joint]:.6f}')
                    self.pause(detail); raise QueueFault(self.fault)
            result=command.copy(); del self.commands[self.next_tick]; self.next_tick+=1
            return result
    def is_current(self, request):
        with self._lock:
            return (self.active and self.pending is request and
                    request.generation == self.generation)

    @property
    def request_due(self):
        with self._lock:
            return (self.active and self.pending is None and self.next_tick>=self.next_request_tick
                    and self.last_request_tick!=self.next_tick)

class DeadlineRecoveryBudget:
    """Bound automatic deadline reanchors within one operator episode."""

    def __init__(self, max_reanchors=3):
        if isinstance(max_reanchors, bool) or not isinstance(max_reanchors, int) or max_reanchors < 0:
            raise ValueError('max_reanchors must be a non-negative integer')
        self.max_reanchors = max_reanchors
        self.count = 0

    def reset_episode(self):
        self.count = 0

    def record_miss(self):
        self.count += 1
        return self.count <= self.max_reanchors


class PassiveCommandLatch:
    """Keep action-space commands for dimensions that policy control does not move."""

    def __init__(self):
        self._commands = None

    def reset(self):
        self._commands = None

    def update(self, proposed_plan, prefix_length):
        plan = vector(proposed_plan, (50, 14))
        delay = int(prefix_length)
        if not 0 <= delay <= 6:
            raise ValueError('untrained prefix length')
        if delay == 0 and self._commands is None:
            self._commands = plan[0].copy()
        elif self._commands is None:
            raise ValueError('passive commands unavailable before d0 plan')
        return self._commands.copy()


class CommandFilter:
    # Express smoothing/rate limits in seconds, preserving the old 20Hz settings.
    def __init__(self, fps=30., tau=-.05/np.log(.65), velocity=.2, acceleration=None, active='right'):
        if fps<=0 or tau<=0 or velocity<=0 or (acceleration is not None and acceleration<=0) or active not in ('left','right','both'):
            raise ValueError('invalid conditioner settings')
        self.alpha=1-np.exp(-1/(fps*tau)); self.step=velocity/fps
        self.accel_step=None if acceleration is None else acceleration/(fps*fps)
        self.joints=np.r_[0:6] if active=='left' else np.r_[7:13] if active=='right' else np.r_[0:6,7:13]
        repeats=2 if active=='both' else 1
        self.lower=np.tile(PIPER_LOWER,repeats);self.upper=np.tile(PIPER_UPPER,repeats)
    def plan(self, raw, measured, prefix, prefix_length, *, passive_commands=None):
        requested=vector(raw,(50,14)); state=vector(measured,(14,))
        passive=state if passive_commands is None else vector(passive_commands,(14,))
        fixed=vector(prefix,(50,14)); d=int(prefix_length)
        if not 0<=d<=6: raise ValueError('untrained prefix length')
        output=np.broadcast_to(passive,requested.shape).copy()
        output[:d]=fixed[:d]
        if d and (np.any(fixed[:d,self.joints]<self.lower) or np.any(fixed[:d,self.joints]>self.upper)):
            raise ValueError('committed prefix outside Piper joint limits')
        anchor=state.copy() if not d else fixed[d-1].copy()
        previous_delta=np.zeros(len(self.joints),np.float32) if d<2 else fixed[d-1,self.joints]-fixed[d-2,self.joints]
        for t in range(d,50):
            command=passive.copy()
            target=np.clip(requested[t,self.joints],self.lower,self.upper)
            delta=np.clip(self.alpha*(target-anchor[self.joints]),-self.step,self.step)
            if self.accel_step is not None:
                delta=np.clip(delta,previous_delta-self.accel_step,previous_delta+self.accel_step)
            active_command=np.clip(anchor[self.joints]+delta,self.lower,self.upper)
            command[self.joints]=active_command
            output[t]=command;previous_delta=active_command-anchor[self.joints];anchor=command
        return output
