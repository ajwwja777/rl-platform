"""Explicit, conservative policy-to-HIL observed-return links.
This is value under the recorded assisted controller, NOT autonomous-policy Q.
No actor optimization or production replay mutation uses these links.
"""
import numpy as np

def link_handover(a,frames,teacher_stamps):
    next_row=a['next_row'].copy();bootstrap=a['bootstrap'].copy();edges=[];rejected=[]
    spans=a['duration'].astype(np.float64).copy()
    control_stamps=np.asarray([f['ros_timestamp'] for f in frames],np.float64)
    continuous=np.ones(len(next_row),bool)
    def fresh(stamps,start,end):
        if start<0 or end>=len(stamps):return False
        dt=np.diff(stamps[start:end+1])
        return bool(np.all((dt>0)&(dt<=.05)))
    for i in range(len(next_row)):
        stamps=teacher_stamps if a['stream'][i] else control_stamps
        start=int(a['frame'][i]);end=start+int(a['duration'][i])-1
        continuous[i]=fresh(stamps,start,end)
    for i in np.where(bootstrap)[0]:
        j=int(next_row[i]);stamps=teacher_stamps if a['stream'][i] else control_stamps
        if not(continuous[i] and continuous[j] and fresh(stamps,int(a['frame'][i]),int(a['frame'][j]))):
            bootstrap[i]=False;next_row[i]=i
            rejected.append({'row':int(i),'reason':'normal_edge_or_window_clock_gap'})
    for i in np.where((a['stream']==0)&a['truncated'])[0]:
        last=int(a['frame'][i])+int(a['duration'][i])-1
        if last+1>=len(frames):continue
        before,after=frames[last:last+2]
        if not continuous[i]:rejected.append({'row':int(i),'reason':'policy_window_clock_gap'});continue
        if after['phase']!='hil':continue
        reason=None
        if not(before['valid_for_training'] and after['valid_for_training']):reason='invalid_control'
        elif before['phase']!='rollout' or before['mode']!='policy':reason='not_policy_boundary'
        elif not(after['mode'].startswith('manual:') and 'right' in after['mode'][7:].split('+')):reason='not_right_HIL'
        elif int(after['generation'])!=int(before['generation'])+1:reason='unmodelled_generation_change'
        elif int(after['tick'])!=int(before['tick']):reason='tick_discontinuity'
        dt=float(after['ros_timestamp'])-float(before['ros_timestamp'])
        if not(0<dt<=.05):reason='control_gap'
        candidates=np.where((a['stream']==1)&(a['generation']==after['generation']))[0]
        if not len(candidates):reason=reason or 'missing_teacher_observation'
        if reason:rejected.append({'row':int(i),'reason':reason});continue
        stamps=teacher_stamps[a['frame'][candidates]]
        j=int(candidates[np.argmin(abs(stamps-after['ros_timestamp']))]);stamp=teacher_stamps[int(a['frame'][j])]
        if not(0<stamp-before['ros_timestamp']<=.05) or abs(stamp-after['ros_timestamp'])>.034:
            rejected.append({'row':int(i),'reason':'teacher_observation_gap'});continue
        if not continuous[j]:
            rejected.append({'row':int(i),'reason':'teacher_window_clock_gap'});continue
        next_row[i]=j;bootstrap[i]=True
        spans[i]=(stamp-float(frames[int(a['frame'][i])]['ros_timestamp']))*30
        if not(0<spans[i]<=a['duration'][i]+1.1):raise ValueError('unbounded handover duration')
        edges.append({'row':int(i),'next_row':j,'last_policy_trace_frame':last,
                      'first_HIL_trace_frame':last+1,'control_dt_sec':dt,
                      'next_state_dt_sec':float(stamp-before['ros_timestamp']),'discount_span_ticks':float(spans[i]),
                      'generation_before':int(before['generation']),
                      'generation_after':int(after['generation']),
                      'kind':'continuous_observed_right_HIL_takeover',
                      'value_semantics':'recorded_assisted_controller_not_autonomous_Q'})
    # Observed discounted rewards along exact successors. Terminal rewards
    # stay at their factual rows; nothing is backfilled across truncations.
    returns=np.zeros(len(next_row),np.float32);reachable=np.zeros(len(next_row),bool)
    discounts=.99**np.arange(10)
    def visit(i,seen):
        if i in seen:raise ValueError('cyclic observed trajectory')
        if not continuous[i]:return 0.,False
        own=float(np.sum(a['reward'][i]*discounts));has=bool(np.any(a['reward'][i]>0))
        if bootstrap[i]:
            value,connected=visit(int(next_row[i]),seen|{i})
            own+=float(.99**spans[i])*value;has|=connected
        return own,has
    for i in range(len(next_row)):returns[i],reachable[i]=visit(i,set())
    return {'next_row':next_row,'bootstrap':bootstrap,'observed_return':returns,
            'success_reachable':reachable,'discount_span_ticks':spans,'continuous_window':continuous},edges,rejected
