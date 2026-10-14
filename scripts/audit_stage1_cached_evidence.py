#!/usr/bin/env python3
"""Revalidate archived Stage1 arrays without model loading or GPU use."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import time_targets, interpolate_chunk, action_summary

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def audit(root, output):
    if output.exists(): raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    result={'boundary':'Historical TRAIN-seen caches. No new forward, no independent TEST, no robot success inference.',
            'script_sha256':sha(__file__),'groups':[], 'failures':[]}
    for group in ['stage1_full','stage1_service_numeric_training','stage1_service_numeric']:
        folder=root/group; report=json.loads((folder/'report.json').read_text())
        ds=Path(report['dataset']); rows=[]
        assert report['complete']
        for row in report['episodes']:
            ep=row['episode']; cache=folder/f'episode_{ep:03d}.npz'
            parquet=ds/'data'/f'chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet'
            checks={'cache_sha':sha(cache)==row['cache_sha256'],
                    'parquet_sha':sha(parquet)==row['parquet_sha256'],
                    'receipt_exact':json.loads((folder/f'episode_{ep:03d}.json').read_text())==row}
            with np.load(cache,allow_pickle=False) as arr:
                pred=arr['predicted_native']; raw=arr['recorded_action']; state=arr['proprio']
                checks['full_frame_coverage']=len(pred)==len(raw)==row['frames']==row['samples']
                checks['finite']=all(np.isfinite(arr[k]).all() for k in arr.files)
                t30,m30=time_targets(raw,30,30,50); t20,m20=time_targets(raw,30,20,10)
                variants={'sample30_first10':(pred[:,:10],t30[:,:10],m30[:,:10]),
                          'sample30_full50':(pred,t30,m30),
                          'legacy_first10_at20':(pred[:,:10],t20,m20),
                          'retimed30to20_first10':(interpolate_chunk(pred,30,20,10),t20,m20),
                          'hold_current_state_at20':(np.broadcast_to(state[:,None,:],t20.shape),t20,m20)}
                metrics={k:action_summary(*v) for k,v in variants.items()}
                err=max(float(np.max(np.abs(np.asarray(metrics[k][f])-row[k][f])))
                        for k in metrics for f in ['mae_per_dim','rmse_per_dim','p95_abs_per_dim','bias_per_dim'])
                checks['metrics_recomputed']=err<1e-10
                rows.append({'episode':ep,'frames':len(pred),'checks':checks,'max_metric_error':err,
                             'mae':{k:v['mae_per_dim'] for k,v in metrics.items()}})
                for k,ok in checks.items():
                    if not ok: result['failures'].append([group,ep,k])
        ids=[x['episode'] for x in rows]
        if sorted(ids)!=sorted(report['requested_episodes']) or len(ids)!=len(set(ids)):
            result['failures'].append([group,'coverage_or_duplicate'])
        manifest=ds/'conversion_manifest.json'
        if sha(manifest)!=report['identity']['manifest']: result['failures'].append([group,'manifest'])
        result['groups'].append({'name':group,'report_sha256':sha(folder/'report.json'),
            'identity':report['identity'],'source_code':report['code_head'],'episodes':len(rows),
            'frames':sum(x['frames'] for x in rows),'inference_path':report['identity'].get('inference_path','analysis'),
            'rows':rows})
    values=np.array([[r['mae'][k] for k in ['legacy_first10_at20','retimed30to20_first10']] for r in result['groups'][0]['rows']])
    delta=values[:,1,:6].mean(1)-values[:,0,:6].mean(1)
    rng=np.random.default_rng(42); samples=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    result['full_analysis_time_comparison']={'unit':'rad','episode_equal_joint_mae':values[:,:,:6].mean((0,2)).tolist(),
        'paired_delta_ci95':np.quantile(samples,[.025,.975]).tolist(),'uncertainty_unit':'whole training-seen Episode; fitting variability only'}
    result['status']='已验证' if not result['failures'] else '失败'
    (output/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for i,label in enumerate(['Legacy30 values at 20Hz','Retimed 30 -> 20Hz']):
        axs[0].plot(np.arange(1,7),values[:,i,:6].mean(0)*1000,'o-',label=label)
    axs[0].set(xlabel='Right arm joint',ylabel='Episode-equal MAE (mrad)',title='134 TRAIN-seen Episodes / analysis graph');axs[0].legend(fontsize=8)
    axs[1].hist(delta*1000,bins=25);axs[1].axvline(0,color='black',lw=1)
    axs[1].set(xlabel='Retimed minus legacy joint MAE (mrad)',ylabel='Complete Episodes',title='Reference only; no Actor / no robot result')
    fig.tight_layout();fig.savefig(output/'reference_time_contract.png',dpi=160);plt.close(fig)
    print(json.dumps({'status':result['status'],'groups':[(g['name'],g['episodes'],g['frames']) for g in result['groups']],
                      'failures':result['failures'],'time':result['full_analysis_time_comparison']}))
    return not result['failures']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.exit(0 if audit(a.root,a.output) else 1)
