#!/usr/bin/env python3
"""Trace every archived expert C10 to its original 30Hz frames and reference."""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import interpolate_chunk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    a.output.mkdir(parents=True)
    hist=a.root/'outputs/rlt/plug_v3_yyshadow/history/warmup_20260925_trials'
    journal=hist/'experts120_5000/replay/replay_journal.pkl'
    dataset=Path('/data/LFT-W02_data/jiaan/scratch/cobot-realworld-rl/data/rlt/plug_v3_yyshadow/plug_v3_yyshadow_demonstrations')
    cache_root=a.root/'outputs/offline-model-selection-20261005/stage1_full'
    receipt=json.loads((cache_root/'report.json').read_text())
    if not receipt.get('complete'):raise ValueError('full forward cache incomplete')
    byep={}
    with journal.open('rb') as f:
        while True:
            try:r=pickle.load(f)
            except EOFError:break
            if int(r['episode_id'])>=100000:byep.setdefault(int(r['episode_id'])-100000,[]).append(r)
    report={'training_journal':str(journal),'training_journal_sha256':hashlib.sha256(journal.read_bytes()).hexdigest(),
        'materializer_sha256':hashlib.sha256((hist/'materialize_experts120_v3.py').read_bytes()).hexdigest(),
        'stage1_forward_identity':receipt['identity'],'episodes':[],
        'boundary':['All 120 Episodes used in both Stage1 and Warmup; training-input audit, not an independent test.',
            'Corrected-reference comparison uses cached batch8 Stage1 with training prompt; old materializer used integer PyAV decoded frames and batch1. Numerical/image-path differences are not isolated here.',
            'Original action/proprio/next-state reconstruction is exact to 2e-6 native units.',
            'Old rollout records store only C10 legacy references; retiming to 20Hz needs up to Stage1 index13.5, which cannot be recovered from ten stored points without a new real-model forward.',
            'No Replay rewritten; heterogeneous old-rollout/new-expert reference training is not automatically promoted.']}
    for ep,rows in sorted(byep.items()):
        path=dataset/('data/chunk-000/episode_%06d.parquet'%ep);table=pq.read_table(path)
        state=np.asarray(table['observation.state'].to_pylist(),np.float32)
        action=np.asarray(table['action'].to_pylist(),np.float32);ts=np.asarray(table['timestamp'].to_pylist(),np.float64)
        grid=np.arange(ts[0],ts[-1],.05);nearest=np.clip(np.searchsorted(ts,grid),1,len(ts)-1)
        nearest=np.where(np.abs(ts[nearest]-grid)<np.abs(ts[nearest-1]-grid),nearest,nearest-1)
        positions=np.unique(np.r_[nearest,len(ts)-1]).astype(int)
        expected_starts=sorted(set(range(0,len(positions)-9,10))|{len(positions)-10})
        if sorted(int(r['step_id']) for r in rows)!=expected_starts:raise ValueError('expert anchor identity mismatch')
        cache=dict(np.load(cache_root/('episode_%03d.npz'%ep)))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=next(e['parquet_sha256'] for e in receipt['episodes'] if e['episode']==ep):raise ValueError('source changed')
        errors=[];old=[];new=[];durations=[];reference_cache_difference=[]
        for r in rows:
            s=int(r['step_id']);indices=positions[s:s+10];anchor=indices[0]
            errors.extend([float(np.max(np.abs(r['proprio']-state[anchor]))),
                float(np.max(np.abs(r['action_chunk']-action[indices]))),
                float(np.max(np.abs(r['next_proprio']-state[positions[min(s+10,len(positions)-1)]])))])
            native=cache['predicted_native'][anchor]
            corrected=interpolate_chunk(native,30,20,10)
            old.append(np.abs(r['ref_chunk']-action[indices]).mean(0));new.append(np.abs(corrected-action[indices]).mean(0))
            reference_cache_difference.append(np.abs(r['ref_chunk']-native[:10]).mean(0))
            durations.append(float(ts[indices[-1]]-ts[indices[0]]))
        if max(errors)>2e-6:raise ValueError('actual training actions do not match reconstructed source')
        report['episodes'].append({'episode':ep,'transitions':len(rows),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'reconstruction_max_error':max(errors),'actual_c10_first_to_last_seconds_mean':float(np.mean(durations)),
            'legacy_ref_native_span_seconds':9/30,'required_ref_native_span_seconds':9/20,
            'old_reference_target_mae_per_dim':np.mean(old,axis=0).tolist(),
            'corrected_reference_target_mae_per_dim':np.mean(new,axis=0).tolist(),
            'old_ref_vs_cached_batch8_first10_mae_per_dim':np.mean(reference_cache_difference,axis=0).tolist()})
    report['expert_episodes']=len(byep);report['expert_transitions']=sum(e['transitions'] for e in report['episodes'])
    report['status']='已验证' if report['expert_episodes']==120 and report['expert_transitions']==1186 else '失败'
    if report['status']!='已验证':raise ValueError('expected complete expert corpus')
    (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({k:report[k] for k in ['status','expert_episodes','expert_transitions']},ensure_ascii=False))


if __name__=='__main__':main()
