"""Two-wrist Stage1 websocket service. Never creates robot publishers."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--port',type=int,default=8030)
    ap.add_argument('--rtc-overlay',type=Path)
    ap.add_argument('--warmup-rtc',action='store_true')
    ap.add_argument('--validate-only',action='store_true')
    a=ap.parse_args();root=a.project_root.resolve();up=root/'third_party/openpi-rlt'
    sys.path[:0]=[str(root),str(up/'src'),str(up/'scripts'),str(up/'packages/openpi-client/src')]
    import pyarrow,torch
    import jax,numpy as np
    cache=Path(os.environ.get('JAX_COMPILATION_CACHE_DIR',str(root/'runtime/cache/jax/realtime-expo')))
    cache.mkdir(parents=True,exist_ok=True);jax.config.update('jax_compilation_cache_dir',str(cache))
    from .policy import load,CAMERAS
    from .dataset import PROMPT
    started=time.perf_counter();policy=load(a.checkpoint,root/'outputs/realtime-expo-runtime',rtc_overlay=a.rtc_overlay)
    load_seconds=time.perf_counter()-started
    dummy=dict(images={k:np.zeros((224,224,3),np.uint8) for k in CAMERAS},state=np.zeros(7,np.float32),prompt=PROMPT)
    timings=[policy.infer(dummy)['policy_timing']['infer_ms'] for _ in range(3)]
    rtc=[]
    if a.warmup_rtc:
        if a.rtc_overlay is None:raise ValueError('RTC overlay required')
        request=dict(dummy,rtc=dict(previous_actions=np.zeros((5,7),np.float32),delay_steps=4,execution_horizon=5))
        for _ in range(3):
            result=policy.infer(request)
            if not result['rtc_used']:raise ValueError('RTC not accepted')
            rtc.append(result['policy_timing']['infer_ms'])
    receipt=dict(status='passed',pid=os.getpid(),metadata=policy.metadata,inference_ms=timings,rtc_inference_ms=rtc,load_seconds=load_seconds,robot_publishers=0)
    output=root/'outputs/realtime-expo-runtime/model-server/validation.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+'\n');print('MODEL_VALIDATED',json.dumps(receipt),flush=True)
    if a.validate_only:return
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    print('MODEL_READY',flush=True)
    WebsocketPolicyServer(policy=policy,host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()

if __name__=='__main__':main()
