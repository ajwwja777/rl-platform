"""Send cached observations to a read-only CPU worker; only reports return.

No source installation, production data writes, Stage1 or live service requests.
The remote worker is passed through stdin as Python source, never shell-expanded.
"""
import argparse,base64,hashlib,io,json,pickle,subprocess
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--dev',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--host',default='agilex@10.7.165.64');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with a.dev.open('rb')as f:
        while True:
            try:rows.append(pickle.load(f))
            except EOFError:break
    arrays={k:np.stack([r[k]for r in rows])for k in ['z_rl','proprio','ref_chunk','action_chunk']}
    b=io.BytesIO();np.savez_compressed(b,**arrays);encoded=base64.b64encode(b.getvalue()).decode()
    worker=Path(__file__).with_name('evaluate_cached_actor_inputs.py').read_text().replace('sys.stdin.read()',repr(encoded))
    argv=['ssh',a.host,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','taskset','-c','20,21','/home/agilex/jiaan/project/rl-platform/envs/online/bin/python','-']
    result=subprocess.run(argv,input=worker,text=True,capture_output=True,timeout=60)
    (a.output/'stderr.log').write_text(result.stderr);result.check_returncode()
    report=json.loads(result.stdout);(a.output/'predictions.npz').write_bytes(base64.b64decode(report.pop('result_npz_base64')))
    report.update(request_fields=list(arrays),request_bytes=len(b.getvalue()),request_sha256=hashlib.sha256(b.getvalue()).hexdigest(),command=argv)
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='protected_sha256'}))
if __name__=='__main__':main()
