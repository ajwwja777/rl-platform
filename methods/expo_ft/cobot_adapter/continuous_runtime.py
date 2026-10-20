"""Native continuous Actor transport with the existing RTC, HIL and gripper path."""
import time
import numpy as np
from monitoring import selection_timing
import online_runtime
from continuous_remote import ContinuousRemote
from continuous_monitoring import ContinuousRolloutHistory


class ContinuousRuntime(ContinuousRemote, online_runtime.OnlineRuntime):
    def __init__(self, args, ros):
        super().__init__(args, ros)
        self.monitor_history = ContinuousRolloutHistory(args.run_dir)

    def infer(self, observation, prefix):
        value = super().infer(observation, prefix)
        self.local_actor_version = max(self.local_actor_version, value['actor_version'])
        return value

    def resolve_inference(self, value):
        if not self.episode:
            raise ValueError('Online episode has ended')
        observation = self.observation()
        selected_at = time.monotonic()
        payload = self.array_rpc('/select', **observation, episode_uuid=self.episode,
            precached=np.asarray(value['precached'], np.float32), delay=value['delay'],
            actor_version=value['actor_version'], inference_token=value['inference_token'])
        self.local_actor_version = max(self.local_actor_version, payload['actor_version'])
        payload.update(selection_timing(value, selected_at))
        self.monitor_history.selected(self.journal.active, payload)
        return np.asarray(payload['actions'], np.float32)


if __name__ == '__main__':
    online_runtime.OnlineRuntime = ContinuousRuntime
    online_runtime.main()
