"""Execute deployed infer method with frozen images and command-following fixture, no ROS."""
import ast,time,json,threading,argparse,types
from pathlib import Path
import numpy as np
from .rtc_actor_client import select
from .rtc_queue import RTCQueue,CommandFilter,PassiveCommandLatch,PiperWorkspaceGuard
from .rpc import ModelClient
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'
def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18023);a=p.parse_args()
    tree=ast.parse((ROOT/'methods/openpi_rlt/plug_v2/runtime.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runtime');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='infer')
    ns={'np':np,'time':time,'json':json,'__package__':'methods.openpi_rlt.plug_v2'};exec(compile(ast.Module(body=[method],type_ignores=[]),'deployed-runtime-infer','exec'),ns)
    actor,version=select(a.release,a.port);rpc=ModelClient();results=[]
    scenes=sorted((RUN/'scene-snapshots').glob('*/*.npz'))[-5:]
    try:
        for scene in scenes:
            with np.load(scene) as f:
                state=f['state'].copy();images={k:f[v].copy() for k,v in [('base_0_rgb','camera_high'),('left_wrist_0_rgb','camera_left'),('right_wrist_0_rgb','camera_right')]}
            q=RTCQueue(tracking_joints=np.arange(7,13));q.resume();obj=types.SimpleNamespace(queue=q,filter=CommandFilter(velocity=.1,acceleration=.9),passive_commands=PassiveCommandLatch(),rpc=rpc,actor=actor,actor_version=version,args=types.SimpleNamespace(explore=False),v5=True,workspace_guard=PiperWorkspaceGuard(state),frame_lock=threading.RLock(),plans=[])
            lat=[];ticks=[];commands=[];start=state.copy();error=None
            try:
                for tick in range(104):
                    if q.request_due:
                        req=q.request(state);accepted,elapsed=ns['infer'](obj,req,dict(state=state,images=images,prompt='Insert the held plug into the socket.'));assert accepted
                        ticks.append(tick);lat.append(elapsed)
                    state=q.pop(state);commands.append(state.copy())
                assert ticks==[0]+list(range(4,104,10)),ticks
            except Exception as e:error=str(e)
            commands=np.asarray(commands);passive=np.r_[0:7,13]
            assert len(commands)>0
            assert np.array_equal(commands[:,passive],np.broadcast_to(commands[0,passive],commands[:,passive].shape))
            assert np.allclose(commands[:,6],.0002,atol=1e-5)
            delta=np.diff(np.vstack([start,commands])[:,7:13],axis=0)
            assert abs(delta).max()<=.1/30+1e-6
            assert abs(np.diff(np.vstack([np.zeros((1,6)),delta]),axis=0)).max()<=.9/900+1e-6
            results.append(dict(scene=str(scene),commands=len(commands),request_ticks=ticks,latency=lat,error=error,max_velocity=float(abs(delta).max()*30)))
        report=dict(scope='actual runtime infer/Stage1/HTTP actor; saved images and ideal command-following fixture, no robot physics or motion',actor_version=version,results=results,passed=all(x['error'] is None and max(x['latency'][1:])<.2 for x in results))
        a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:rpc.close()
if __name__=='__main__':main()
