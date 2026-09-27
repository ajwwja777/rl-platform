"""Foreground operator handle; Session stop is separate from model ownership."""
import json,signal,time,fcntl
from . import cli

def registered_session():
    try:return json.loads((cli.BACKEND/'processes.json').read_text()).get('children',{}).get('session')
    except (OSError,ValueError):return None

def stop_owned(owner,previous):
    with (cli.BACKEND/'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=registered_session()
        if owner is not None and current!=owner:
            print('Session已由另一终端替换；未停止新Session。',flush=True);return
        if owner is None and current not in (None,previous):
            # The requested new Session may have just appeared before readiness.
            owner=current
        cli.end_session()
        if not cli.occupied(8026):cli.atomic(cli.BACKEND/'request.json',{'command':'session-stop'})
    print('Session已结束，模型保留。再次rlt_up即可；rlt_down才释放模型。',flush=True)

def watch(previous):
    owner=None;last=None;last_profile=None
    def interrupted(*_):raise KeyboardInterrupt
    old={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    print('此终端保持运行；Ctrl+C仅结束Session，不归位、不释放模型。',flush=True)
    try:
        while True:
            current=registered_session()
            if owner is None and current is not None and current!=previous:owner=current
            if owner is not None and current!=owner:
                print('Session已被替换或停止，本终端退出。',flush=True);return
            try:
                state=json.loads((cli.BACKEND/'state.json').read_text())
                if state['phase'] in ('offline','fault'):
                    raise RuntimeError('RLT后端 '+state['phase']+': '+str(state.get('error_code')))
                snapshot=cli.http('/api/session') if owner is not None and cli.occupied(8026) else {}
                phase=snapshot.get('phase',state['phase'])
                profile=snapshot.get('runtime_profile') or {}
                profile_key=(snapshot.get('actor_mode'),snapshot.get('actor_candidate'),snapshot.get('actor_version'),
                             profile.get('revision'),profile.get('control_hz'),profile.get('velocity_rad_per_sec'),
                             profile.get('tracking_bound_rad'),profile.get('joint_limits'),
                             snapshot.get('rtc_deadline_recoveries'),snapshot.get('model_rpc_reconnects'))
                if phase!=last or (profile and profile_key!=last_profile):
                    detail=''
                    if snapshot.get('actor_mode'):
                        detail+=' · actor='+str(snapshot['actor_mode'])
                        if snapshot.get('actor_candidate'):detail+=':'+str(snapshot['actor_candidate'])
                        if snapshot.get('actor_version') not in (None,-1):detail+='@'+str(snapshot['actor_version'])
                    if profile:
                        if profile.get('revision'):detail+=' · rev='+str(profile['revision'])
                        detail+=f" · RTC={profile.get('control_hz')}Hz/{profile.get('inference')}"
                        if profile.get('velocity_rad_per_sec') is not None:
                            detail+=f" · v={profile['velocity_rad_per_sec']:.2f}rad/s"
                        if profile.get('acceleration_rad_per_sec2') is not None:
                            detail+=f" a={profile['acceleration_rad_per_sec2']:.1f}rad/s²"
                        detail+=f" · tracking={profile.get('tracking_bound_rad'):.2f}rad"
                        if profile.get('joint_limits'):detail+=' limits='+str(profile['joint_limits'])
                    recoveries=snapshot.get('rtc_deadline_recoveries',0)
                    reconnects=snapshot.get('model_rpc_reconnects',0)
                    if recoveries or reconnects:
                        detail+=f' · recoveries={recoveries} rpc_reconnects={reconnects}'
                    if phase in ('fault','terminal_pending') and snapshot.get('fault_reason'):
                        detail+=' · reason='+str(snapshot['fault_reason'])
                    print('RLT: '+phase+detail,flush=True);last=phase;last_profile=profile_key
                if snapshot.get('phase')=='stopped':
                    print('网页已结束Session，终端退出；模型保留。',flush=True);return
            except (OSError,ValueError):pass
            time.sleep(1)
    except KeyboardInterrupt:
        print('正在暂停并结束Session，等待写盘；模型保留…',flush=True)
        # Repeated Ctrl+C must not interrupt the first stop/flush request.
        for s in old:signal.signal(s,signal.SIG_IGN)
        stop_owned(owner,previous)
    finally:
        for s,handler in old.items():signal.signal(s,handler)
