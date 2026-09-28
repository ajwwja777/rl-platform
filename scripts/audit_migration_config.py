from pathlib import Path
import yaml,json
root=Path(__file__).resolve().parents[1]
original=root/'outputs/migrations/20260927-rlt/site-source/methods/openpi_rlt/plug_v3_yyshadow'
def flatten(value, prefix=''):
 if isinstance(value,dict):
  for key,v in value.items():yield from flatten(v,prefix+'.'+key)
 else:yield prefix,value
changed={}
for filename in ['online_rl.yaml','online_rl_frozen.yaml']:
 old=dict(flatten(yaml.safe_load((original/filename).read_text())))
 new=dict(flatten(yaml.safe_load((root/'configs/rlt/plug_v3_yyshadow'/filename).read_text())))
 assert old.keys()==new.keys()
 diffs={k:[old[k],new[k]] for k in old if old[k]!=new[k]}
 for key,(before,after) in diffs.items():
  assert isinstance(before,str) and before.startswith('../../../runs/'),(key,before,after)
  assert after.startswith(('../../../models/','../../../outputs/','/media/agilex/Getea1/jiaan/data/','/media/agilex/Getea1/jiaan/model/')), (key,before,after)
 changed[filename]=diffs
print(json.dumps({'algorithm_parameters_unchanged':True,'path_changes':changed},indent=2))
