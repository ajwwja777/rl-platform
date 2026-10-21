"""Read-only observer of published EXPO status; no learner imports/control calls.

Stores raw native update metrics in SQLite and publishes bounded browser history.
No credentials or W&B calls: export can be attached independently later.
"""
import argparse,hashlib,json,math,os,sqlite3,time,fcntl
from pathlib import Path
from urllib.request import build_opener,ProxyHandler

def scalar(v):
    return type(v) in (int,float) and math.isfinite(v)

def extract(status):
    model=status.get('model') or {}; online=status.get('online') or {}
    if model.get('runtime_adapter')!='expo_online' or online.get('deployment_id')!=model.get('id'):
        return None
    identity=[model['id'],model.get('online_run_dir'),online.get('pid'),online.get('rtc_delay'),online.get('edit_scale')]
    run=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
    progress={k:online.get(k) for k in ('critic_updates','actor_updates','learner_step','served_actor_version','published_actor_version','replay_steps','online_steps','completed_episodes','pending_episodes','pending_update_calls','rtc_delay','edit_scale','critic_prior','critic_sampling','compute_host','learner_phase','phase','updated_at','last_update_at','network_error','async_mode','batch_size','utd_ratio','seed_steps','seed_episodes','replay_online_episodes','pending_upload_episodes','pending_episode_ids','committed_learner_step','downloaded_actor_version','download','upload','transfer','sampling','replay_audit','publications','published_at','checkpoint_phase','ep_timeout_secs','installed_receipt','timings','install_error')}
    metrics={k:v for k,v in (online.get('metrics') or {}).items() if scalar(v)}
    step=online.get('critic_updates')
    point=None
    if metrics and type(step) is int and step>0:
        point=dict(critic_step=step,actor_step=online.get('actor_updates'),timestamp=online.get('updated_at'),metrics=metrics,
                   timings={k:v for k,v in (online.get('timings') or {}).items() if scalar(v)})
    return run,dict(run_id=run,model_id=model['id'],label=model.get('label'),options=dict(rtc_delay=online.get('rtc_delay'),edit_scale=online.get('edit_scale')),progress=progress),point

class Store:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, updated REAL, metadata TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS points (run TEXT, step INTEGER, data TEXT, PRIMARY KEY(run,step))')
        self.db.execute('CREATE TABLE IF NOT EXISTS version_events (run TEXT, stage TEXT, version INTEGER, observed REAL, PRIMARY KEY(run,stage,version))')
    def ingest(self,status):
        result=extract(status)
        if result is None:return
        run,meta,point=result
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?)',(run,time.time(),json.dumps(meta)))
            for stage,key in [('trained','learner_step'),('saved','committed_learner_step'),('published','published_actor_version'),('downloaded','downloaded_actor_version'),('served','served_actor_version')]:
                version=meta['progress'].get(key)
                if type(version) is int and version>=0:self.db.execute('INSERT OR IGNORE INTO version_events VALUES (?,?,?,?)',(run,stage,version,time.time()))
            if point:self.db.execute('INSERT OR IGNORE INTO points VALUES (?,?,?)',(run,point['critic_step'],json.dumps(point)))
    def payload(self):
        runs=[]
        for run,updated,meta in self.db.execute('SELECT id,updated,metadata FROM runs ORDER BY updated DESC LIMIT 12'):
            m=json.loads(meta);m['observed_at']=updated
            m['points']=[json.loads(p[0]) for p in self.db.execute('SELECT data FROM points WHERE run=? ORDER BY step DESC LIMIT 600',(run,))][::-1]
            m['version_events']=[dict(stage=s,version=v,observed_at=t) for s,v,t in self.db.execute('SELECT stage,version,observed FROM version_events WHERE run=? ORDER BY observed DESC LIMIT 200',(run,))][::-1]
            runs.append(m)
        return dict(schema=1,sampled_at=time.time(),runs=runs,transport='local_read_only_observer',source='Native learner update metrics, observed via read-only status; polling may miss updates; missing history is not reconstructed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:8015/api/deployment/status');a=p.parse_args()
    if not a.url.startswith('http://127.0.0.1:'):raise ValueError('Only local read-only status URL allowed')
    a.root.mkdir(parents=True,exist_ok=True);a.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(a.root/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    store=Store(a.root/'metrics.sqlite');opener=build_opener(ProxyHandler({}))
    (a.root/'observer.pid').write_text(str(os.getpid()))
    from expo_progress import ProgressProbe
    probe=ProgressProbe(a.root,opener)
    while True:
        error=None
        try:
            with opener.open(a.url,timeout=8) as response:store.ingest(probe.enrich(json.load(response)))
        except Exception as exc:error=type(exc).__name__
        payload=store.payload();payload['read_error']=error
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False));tmp.replace(a.output)
        time.sleep(2)
if __name__=='__main__':main()
