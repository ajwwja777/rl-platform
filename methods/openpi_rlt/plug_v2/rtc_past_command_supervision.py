"""Causal teacher-observation RTC BC augmentation; not factual TD.
Six hypothetical committed commands are planned only from measured current
state and the previous two sent SDK commands. Future actions are labels only.
"""
import numpy as np
from .rtc_queue import CommandFilter

def planned_prefix(current,past_two,mode):
 state=np.asarray(current,np.float32);past=np.asarray(past_two,np.float32)
 if state.shape!=(14,) or past.shape!=(2,14) or not np.isfinite(past).all() or not np.isfinite(state).all():raise ValueError('invalid observed history')
 if mode not in ('hold','linear'):raise ValueError('unknown prefix mode')
 step=.2/30;joints=list(range(7,13));velocity=np.zeros(6,np.float32) if mode=='hold' else np.clip(past[1,joints]-past[0,joints],-step,step)
 output=np.broadcast_to(state,(50,14)).copy();anchor=state[joints].copy()
 for i in range(6):
  desired=past[1,joints]+(i+1)*velocity;anchor+=np.clip(desired-anchor,-step,step);output[i,joints]=anchor
 return output

def build_segment(state,actions,images,start,end,rpc):
 rows=[]
 for number,t in enumerate(range(start+2,end-6,10)):
  mode='hold' if number%2==0 else 'linear';prefix=planned_prefix(state[t],actions[t-2:t],mode)
  result=rpc.infer({'state':state[t].copy(),'images':images[t],'action_prefix':prefix,'prefix_length':6})
  if int(result['prefix_length'])!=6:raise ValueError('wrong frozen model RTC delay')
  raw=np.asarray(result['ref_chunk'],np.float32)
  if raw.shape!=(50,14) or not np.isfinite(raw).all():raise ValueError('invalid reference')
  c=np.r_[state[t],(prefix[:6]-state[t]).reshape(-1),1].astype(np.float32);count=min(10,end-t-6);mask=np.arange(10)<count
  if c.shape!=(99,):raise ValueError('context must contain exactly six immutable commands')
  rows.append({'z':np.asarray(result['z_rl'],np.float32),'c':c,'ref':raw[6:16].copy(),'target':actions[np.minimum(t+6+np.arange(10),end-1)].copy(),'mask':mask,'frame':t,'prefix_origin_tick':t-1,'prefix_max_state_delta':float(np.max(abs(prefix[:6,7:13]-state[t,7:13]))),'prefix_mode':mode,'prefix_source_command_frames':[t-2,t-1],'factual_RL_transition':False})
 return rows
