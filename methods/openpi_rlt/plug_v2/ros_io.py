# ROS subscriber/publisher adapter for the new RTC runtime.
# Old CAN/arm/camera launch nodes remain untouched.
import time
import numpy as np
from methods.openpi_rlt.cobot_adapter.cobot_ros1 import (
    RosTask2IO,CobotIOSample,build_machine_a_observation)
from methods.openpi_rlt.plug_v2.rtc_queue import QueueFault
class RGBBridge:
    def __init__(self,bridge):self.bridge=bridge
    def imgmsg_to_cv2(self,message,encoding=None):
        return self.bridge.imgmsg_to_cv2(message,'rgb8')
class RTCIO(RosTask2IO):
    def __init__(self,queue,**kwargs):
        self.queue=queue
        super().__init__(**kwargs)
        self.bridge=RGBBridge(self.bridge)
    def ready(self):
        with self._condition:
            if len(self._images)!=3 or len(self._joints)!=2:return False
            messages=[*self._images.values(),*self._joints.values()]
            stamps=[self._stamp(m) for m in messages]
            now=self.ros.Time.now().to_sec()
            return max(now-s for s in stamps)<.3 and max(stamps)-min(stamps)<=.1
    def arm_policy(self):
        if not self.ready():raise RuntimeError('fresh cameras and front state required before onsite arm')
        super().arm_policy()
    def set_policy_paused(self,paused):
        with self._condition:
            if paused:
                if self.queue.active or self.queue.pending is not None or self.queue.commands:
                    self.queue.pause()
            else:
                if self._mode!='policy' or not self.ready():raise RuntimeError('robot feedback not ready')
                if not self.queue.active:self.queue.resume()
            super().set_policy_paused(paused)
    def measured(self):
        with self._condition:
            if len(self._joints)!=2:raise QueueFault('front_state_unavailable')
            now=self.ros.Time.now().to_sec()
            if max(now-self._stamp(v) for v in self._joints.values())>.2:
                raise QueueFault('front_state_stale')
            state=np.concatenate([np.asarray(self._joints[s].position[:7],np.float32)
                                 for s in ('left','right')])
            if state.shape!=(14,) or not np.isfinite(state).all():raise QueueFault('invalid_front_state')
            return state
    def observation(self):
        with self._condition:
            if not self.ready():raise QueueFault('camera_or_state_stale')
            images={k:self.bridge.imgmsg_to_cv2(v) for k,v in self._images.items()}
            return build_machine_a_observation(left_state=self._joints['left'].position[:7],
                       right_state=self._joints['right'].position[:7],images=images,prompt=self._prompt)
    def publish_next(self,tracking_bound=.04):
        with self._condition:
            if self._paused or self._mode!='policy':return None
            command=self.queue.pop(self.measured(),tracking_bound=tracking_bound)
            published=self.publish_policy_action(command)
            if not published and not self.shadow_mode:raise QueueFault('command_not_published')
            self._chunk_ready=True
            return command
    def _mode_callback(self,message):
        value=str(getattr(message,'data','fault')).strip().lower()
        with self._condition:
            self._mode=value
            if value!='policy':self.set_policy_paused(True)
            self._condition.notify_all()
        # No repeated policy status callback can resume/reanchor a running queue.
        app=self._session_application
        if app is not None:
            if value=='policy':app.update_takeover(left=False,right=False)
            elif value.startswith('manual:'):
                sides=value.split(':',1)[1].split('+')
                app.update_takeover(left='left' in sides,right='right' in sides)
            else:app.mark_terminal_pending('handover_'+value)
    def _pause_service(self,request):
        requested=bool(getattr(request,'data',True))
        if requested:
            self.set_policy_paused(True)
            return self._response(True,'paused',set_bool=True)
        if self._session_application is None or self._session_application.snapshot().phase.value!='rollout':
            return self._response(False,'Resume from the console first',set_bool=True)
        with self._condition:
            if not self.queue.active:return self._response(False,'fresh Session resume required',set_bool=True)
            self._paused=False
        return self._response(True,'policy ready',set_bool=True)
