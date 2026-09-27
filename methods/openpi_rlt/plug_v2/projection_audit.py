import json,argparse
from pathlib import Path
import numpy as np,torch
from .chunk_projection import project_correction
from .rtc_upstream_core import original_context
from .conditioning_v2 import conditioned
ROOT=Path(__file__).resolve().parents[3];RUN=ROOT/'runs/plug_v2'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=RUN/'diagnostics/rtc-upstream-20260920-v2');parser.add_argument('--runs',nargs='+',default=['rtc-upstream-delta10-20260920','rtc-upstream-delta300-20260920']);args=parser.parse_args()
    torch.set_num_threads(2);data=args.data/'episodes';rows=[]
    for folder in args.runs:
        run=RUN/'learning'/folder
        for step in (500,1000,2000,5000):
            if not (run/f'evaluation_{step}.json').exists():continue
            for path in data.glob('*.npz'):
                with np.load(path) as f:
                    m=json.loads(str(f['metadata']))
                    if m['split']!='val' or not m['rows']:continue
                    d={k:f[k].copy() for k in f.files if k!='metadata'}
                saved=run/f'eval_{step}_{m["uuid"]}_{m["index"]}.npz'
                if not saved.exists():continue
                with np.load(saved) as f:pred=f['pred'].copy()
                c=original_context(d['proprio']);full=np.tile(c[:,None,:14],(1,10,1));ref=full.copy();ref[:,:,7:14]=d['ref_chunk'];target=full.copy();target[:,:,7:14]=d['action_chunk']
                with torch.no_grad():target_cmd=conditioned(torch.tensor(target),torch.tensor(c)).numpy()[:,:,7:13]
                for name,value in [('reference',d['ref_chunk']),('raw',pred),('constant',project_correction(pred,d['ref_chunk'],0)),('linear',project_correction(pred,d['ref_chunk'],1))]:
                    a=full.copy();a[:,:,7:14]=value
                    with torch.no_grad():cmd=conditioned(torch.tensor(a),torch.tensor(c)).numpy()[:,:,7:13]
                    for delay in (0,6):
                        at=(d['source_chunk'][:,0]==2)&(d['delay']==delay)
                        if not at.any():continue
                        delta=np.diff(cmd[at],axis=1);acc=np.diff(delta,axis=1)
                        rows.append(dict(model=folder,step=step,projection=name,delay=delay,uuid=m['uuid'],
                            raw_mse=float(np.mean((value[at,:,:6]-d['action_chunk'][at,:,:6])**2)),
                            matched_mse=float(np.mean((cmd[at]-target_cmd[at])**2)),
                            actual_human_mse=float(np.mean((cmd[at]-d['action_chunk'][at,:,:6])**2)),
                            accel_rms=float(np.sqrt(np.mean(acc**2))),raw_step_p95=float(np.quantile(abs(np.diff(value[at,:,:6],axis=1)),.95))))
    summary=[]
    for model,step,projection,delay in sorted({(r['model'],r['step'],r['projection'],r['delay']) for r in rows}):
        selected=[r for r in rows if (r['model'],r['step'],r['projection'],r['delay'])==(model,step,projection,delay)]
        summary.append(dict(model=model,step=step,projection=projection,delay=delay,**{k:float(np.mean([r[k] for r in selected])) for k in ('raw_mse','matched_mse','actual_human_mse','accel_rms','raw_step_p95')}))
    out=args.data/'projection-audit.json';out.write_text(json.dumps({'scope':'counterfactual same-input regression, not closed-loop success','summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
