"""7D Stage1 RTC bridge; reuse the VLA project's flow sampler, not its 14D bridge."""
from pathlib import Path
import importlib
import sys
import numpy as np

def load_sampler(overlay):
    path=Path(overlay).resolve()
    if not (path/"rtc_openpi/sampler.py").is_file():
        raise ValueError("Registered VLA rtc_overlay directory is required")
    sys.path.insert(0,str(path))
    module=importlib.import_module("rtc_openpi.sampler")
    if Path(module.__file__).resolve().parent.parent != path:
        raise RuntimeError("A different RTC overlay is already imported")
    return module.rtc_sample_actions

def encode_prefix(observation, request, transform, horizon=50, action_dim=32):
    if set(request)!={"previous_actions","delay_steps","execution_horizon"}:
        raise ValueError("RTC request requires previous_actions, delay_steps, execution_horizon")
    previous=np.asarray(request["previous_actions"],np.float32)
    delay=request["delay_steps"];execution=request["execution_horizon"]
    if previous.ndim!=2 or previous.shape[1]!=7 or not np.isfinite(previous).all():
        raise ValueError("RTC previous_actions must be finite (N,7) physical actions")
    if type(delay) is not int or type(execution) is not int or not 0<=delay<execution<=len(previous)<=horizon:
        raise ValueError("RTC requires 0 <= delay < execution <= prefix length <= model horizon")
    values=dict(observation,actions=previous.copy())
    values["state"]=np.asarray(observation["state"]).copy()
    transformed=transform(values)
    encoded=np.asarray(transformed.get("actions"),np.float32)
    if encoded.shape!=(len(previous),action_dim) or not np.isfinite(encoded).all():
        raise ValueError("RTC action normalization/padding contract mismatch")
    padded=np.zeros((horizon,action_dim),np.float32);padded[:len(previous)]=encoded
    return padded,delay,execution
