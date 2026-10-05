#!/usr/bin/env python3
"""Train-only PCA/ridge readout of observed behavior return on reused development.

This tests retained information, not action gradients, optimal Q or robot success.
"""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi-rlt/rlt_online_rl/src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('fresh output required')
    args.output.mkdir(parents=True)
    import numpy as np
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.action_representation import ActionRepresentationAdapter
    from methods.openpi_rlt.experiments.target_attribution import reconstruct_observed_returns
    hist=args.root/'outputs/rlt/plug_v3_yyshadow/history'
    checkpoint=hist/'warmup_20260925_trials/experts120_5000/checkpoints/latest.pkl'
    paths=[checkpoint.parent.parent/'replay/replay_journal.pkl',hist/'warmup_20260924_v1/holdout/replay/replay_journal.pkl']
    payload=pickle.loads(checkpoint.read_bytes());config=dict(payload['rl_config'])
    config['action_norm_stats_path']=str(args.root/'models/rlt/plug_v3_yyshadow/warmup-5000/action_norm_stats.json')
    adapter=ActionRepresentationAdapter.from_config(RLTOnlineRLConfig(**config))
    pools=[];allintegrity=[]
    for path in paths:
        rows=[]
        with path.open('rb') as stream:
            while True:
                try:rows.append(pickle.load(stream))
                except EOFError:break
        data={k:np.stack([r[k] for r in rows]) for k in rows[0] if k!='collection_phase'}
        data['phase_online']=np.zeros(len(rows),bool);data['collection_phase_id']=np.ones(len(rows),np.uint8)
        target,integrity=reconstruct_observed_returns(data,config['gamma'])
        if not np.isfinite(target).all():raise ValueError('incomplete/conflicting observed return')
        batch=adapter.prepare_training_batch(data)
        pools.append((data,batch,target));allintegrity.append(integrity)
    tr,train,y=pools[0];dv,dev,ydev=pools[1]
    if set(tr['episode_id'])&set(dv['episode_id']):raise ValueError('Episode leakage')
    z=np.asarray(train['z_rl'],float);mean=z.mean(0);center=z-mean
    _,singular,axes=np.linalg.svd(center,full_matrices=False)
    n_components=min(64,len(singular));projection=axes[:n_components].T
    coords=[center@projection,(np.asarray(dev['z_rl'],float)-mean)@projection]
    lowdim=[np.concatenate([np.asarray(b['proprio']),np.asarray(b['action_chunk']).reshape(len(b['proprio']),-1)],-1)
            for b in [train,dev]]
    feature_sets={'state_and_action':lowdim,'token_only':coords,
                  'token_state_and_action':[np.concatenate([u,v],-1) for u,v in zip(coords,lowdim)]}
    weights=np.zeros(len(y))
    for ep in np.unique(tr['episode_id']):
        mask=tr['episode_id']==ep;weights[mask]=1/mask.sum()
    weights*=len(weights)/weights.sum()
    def summarize(values):
        rows=[]
        for ep in sorted(np.unique(dv['episode_id'])):
            ids=np.flatnonzero(dv['episode_id']==ep);steps=dv['step_id'][ids]
            first=ids[steps==steps.min()];hil=np.isin(dv['source_chunk'][ids],[2,3]).any()
            rows.append({'episode_id':int(ep),'windows':len(ids),'success':bool(dv['success'][ids].max()),'assisted':bool(hil),
                'mae':float(np.mean(np.abs(values[ids]-ydev[ids]))),'mse':float(np.mean((values[ids]-ydev[ids])**2)),
                'first_anchor_score':float(values[first].mean()),'observed_return_mean':float(ydev[ids].mean()),
                'predicted_mean':float(values[ids].mean()),'bias':float((values[ids]-ydev[ids]).mean())})
        return rows
    report={'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[checkpoint]},
            'gamma':config['gamma'],'train_episodes':len(np.unique(tr['episode_id'])),'development_episodes':len(np.unique(dv['episode_id'])),
            'split':'20 reused rollout development Episodes; no independent test.',
            'pca_components':n_components,'train_token_variance_explained':float(np.sum(singular[:n_components]**2)/np.sum(singular**2)),
            'target':'Observed discounted terminal behavior return, including human assistance; not optimal/autonomous Q.',
            'training_weights':'Each complete training Episode has equal total regression weight.',
            'ridge_alpha':1.0,'models':[],'integrity':allintegrity,
            'boundary':'Fresh readout with fixed settings. Information decoding does not prove useful Critic action gradients or increasing robot success.'}
    for name,(x,xd) in feature_sets.items():
        mu=np.average(x,axis=0,weights=weights)
        std=np.sqrt(np.average((x-mu)**2,axis=0,weights=weights));std=np.maximum(std,1e-8)
        x=np.column_stack([np.ones(len(x)),(x-mu)/std]);xd=np.column_stack([np.ones(len(xd)),(xd-mu)/std])
        regularizer=np.eye(x.shape[1]);regularizer[0,0]=0
        coefficients=np.linalg.solve(x.T@(weights[:,None]*x)+regularizer,x.T@(weights*y))
        pred=xd@coefficients;rows=summarize(pred)
        picks=np.random.default_rng(42).integers(len(rows),size=(20000,len(rows)))
        ep_mae=np.asarray([r['mae'] for r in rows]);scores=np.asarray([r['first_anchor_score'] for r in rows]);labels=np.asarray([r['success'] for r in rows])
        diff=scores[labels][:,None]-scores[~labels][None,:]
        report['models'].append({'name':name,'episodes':rows,'episode_mae_mean':float(ep_mae.mean()),
            'episode_mae_ci95':np.quantile(ep_mae[picks].mean(1),[.025,.975]).tolist(),
            'first_anchor_success_failure_auc':float(np.mean((diff>0)+.5*(diff==0)))})
        print(json.dumps({k:v for k,v in report['models'][-1].items() if k!='episodes'}),flush=True)
    base=np.asarray([r['mae'] for r in report['models'][0]['episodes']])
    for model in report['models'][1:]:
        delta=np.asarray([r['mae'] for r in model['episodes']])-base
        picks=np.random.default_rng(42).integers(len(delta),size=(20000,len(delta)))
        model['paired_mae_minus_state_action']={'mean':float(delta.mean()),'ci95':np.quantile(delta[picks].mean(1),[.025,.975]).tolist()}
    report['source_unchanged']=report['sources']=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in report['sources']}
    if not report['source_unchanged']:raise ValueError('source changed')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
