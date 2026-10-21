"""Read-only progress facts. Never infer sampled batch composition from Replay size."""
import hashlib,json,re,time,runpy
from pathlib import Path

def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def sampling_summary(o):
    native=o.get('async_mode')=='native_async_v1'
    batch=o.get('batch_size');utd=o.get('utd_ratio')
    valid=native and type(batch) is int and batch>0 and type(utd) is int and utd>0
    seed=o.get('seed_steps');total=o.get('replay_steps')
    return dict(critic_draws_per_group=batch*utd if valid else None,actor_batch_size=batch if valid else None,
        critic_batches=utd if valid else None,critic_sampling=o.get('critic_sampling'),
        actor_sampling='success_only' if native else None,replacement=True if native else None,
        seed_fraction=seed/total if isinstance(seed,(int,float)) and isinstance(total,(int,float)) and total>0 and 0<=seed<=total else None,
        actual_sample_composition=None)

def transfer_bytes(directory,version,manifest):
    directory=Path(directory)
    known=manifest.get('actor_version')==version and isinstance(manifest.get('records'),list)
    received=0;total=None;files=0;count=None
    if known:
        records=[x for x in manifest['records'] if not x.get('seed')]
        known=all(re.fullmatch(r'(actor|encoder|edit|critic)-[0-9]+\.gz',str(x.get('file',''))) and type(x.get('bytes')) is int and x['bytes']>=0 for x in records)
    if known:
        total=sum(x['bytes'] for x in records);count=len(records)
        for x in records:
            final=directory/x['file'];partial=final.with_suffix('.download')
            try:
                size=final.stat().st_size if final.exists() else partial.stat().st_size if partial.exists() else 0
            except OSError:size=0
            received+=min(size,x['bytes']);files+=int(final.exists() and size==x['bytes'])
    else:
        for p in directory.glob('*'):
            if re.fullmatch(r'(actor|encoder|edit|critic)-[0-9]+\.(gz|download)',p.name):
                try:received+=p.stat().st_size
                except OSError:pass
    return dict(actor_version=version,bytes_received=received,bytes_total=total,
        percent=100*received/total if total else None,files_received=files,files_total=count,
        verified_complete=read_json(directory/'manifest.json').get('actor_version')==version,
        source='requested_version_cache_not_wire_identity')

class ProgressProbe:
    def __init__(self,root,opener):
        self.root=Path(root);self.opener=opener;self.previous={};self.last_fetch={};self.inventory={}
    def enrich(self,status):
        status=dict(status);o=dict(status.get('online') or {});model=status.get('model') or {}
        if model.get('runtime_adapter')!='expo_online' or model.get('id')!=o.get('deployment_id'):return status
        o['sampling']=sampling_summary(o)
        raw=model.get('online_run_dir');run=Path(raw) if isinstance(raw,str) and raw.startswith('/') else None
        defaults=(model.get('expo_online_options') or {}).get('defaults')
        if run and defaults:
            try:
                configured=runpy.run_path(str(Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter/online_options.py'))['configured_run']
                run=Path(configured({'run_dir':str(run),**defaults},{'rtc_delay':o.get('rtc_delay'),'edit_scale':o.get('edit_scale')})['run_dir'])
            except (ValueError,KeyError,TypeError,OSError):status['online']=o;return status
        if not run or not run.is_dir():status['online']=o;return status
        key=hashlib.sha256(str(run).encode()).hexdigest()[:20];now=time.time()
        installed=read_json(run/'installed.json')
        if installed:o['installed_receipt']=installed
        download=o.get('download') or {};version=download.get('actor_version')
        if type(version) is int and version>=0:
            directory=run/'actors'/str(version);cache=self.root/'manifests'/(key+'-'+str(version)+'.json')
            manifest=read_json(directory/'manifest.json') or read_json(cache)
            if not manifest and download.get('state') in ('downloading','retrying') and now-self.last_fetch.get((key,version),0)>15:
                self.last_fetch[key,version]=now
                try:
                    with self.opener.open('http://127.0.0.1:18453/actor/manifest',timeout=1) as response:candidate=json.load(response)
                    if candidate.get('actor_version')==version:
                        manifest=candidate;cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(manifest))
                except (OSError,ValueError):pass
            transfer=transfer_bytes(directory,version,manifest)
            old=self.previous.get(key);speed=None
            if old and old[0]==version and now>old[1] and transfer['bytes_received']>=old[2]:speed=(transfer['bytes_received']-old[2])/(now-old[1])
            self.previous[key]=(version,now,transfer['bytes_received'])
            if download.get('state')!='downloading':speed=None
            transfer['bytes_per_second']=speed
            transfer['eta_seconds']=(transfer['bytes_total']-transfer['bytes_received'])/speed if speed and transfer['bytes_total'] else None
            o['transfer']=transfer
        # Local admitted receipts are a partial audit of the pool, never a sampled-batch claim.
        if key not in self.inventory or now-self.inventory[key][0]>10:
            counts={'success':0,'failure':0};policies={};episodes=0;excluded=0
            for receipt in (run/'upload_receipts').glob('*.json'):
                ack=read_json(receipt);meta=read_json(run/'episodes'/receipt.stem/'finished.json')
                eligible=ack.get('eligible',ack.get('training_eligible'))
                if eligible is False or meta.get('outcome')=='aborted':excluded+=1;continue
                if eligible is not True:continue
                outcome=meta.get('outcome');count=ack.get('count',meta.get('count'))
                if outcome not in counts or type(count) is not int:continue
                episodes+=1;counts[outcome]+=count
                versions=meta.get('policy_versions') or [meta.get('actor_version')]
                label=str(versions[0]) if len(versions)==1 and versions[0] is not None else 'mixed/unknown'
                policies[label]=policies.get(label,0)+count
            self.inventory[key]=(now,dict(known_online_samples=counts,known_online_episodes=episodes,excluded_episodes=excluded,policy_samples=policies,scope='local_admitted_receipts_not_complete_remote_pool'))
        o['replay_audit']=self.inventory[key][1];status['online']=o;return status
