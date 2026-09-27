"""Isolated sampler ablation: disjoint 20% human / 80% policy, not upstream stratified."""
import json,sys,numpy as np
from pathlib import Path
from . import executed_command_train as runner
class QuotaSource(runner.PreciseSource):
 def __init__(self,*args):
  super().__init__(*args);self.rng=np.random.default_rng(42);h=np.asarray([r['source_chunk'][0]==2 for r in self.rows]);self.human=np.where(h)[0];self.policy=np.where(~h)[0]
  if not len(self.human) or not len(self.policy):raise ValueError('both source pools required')
 def sample_batch(self,n):
  nh=round(n*.2);at=np.r_[self.rng.choice(self.human,nh),self.rng.choice(self.policy,n-nh)];self.rng.shuffle(at)
  return {k:np.asarray([self.rows[i][k] for i in at]) for k in self.rows[0]}|{'step_id':at.astype(np.int32)}
def main():
 runner.PreciseSource=QuotaSource;runner.main();out=Path(sys.argv[sys.argv.index('--output')+1]);(out/'sampler_protocol.json').write_text(json.dumps({'human_per_128':26,'policy_per_128':102,'pools_disjoint':True,'not_upstream_sampler':True,'production_publish':False}))
if __name__=='__main__':main()
