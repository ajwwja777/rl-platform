"""Trusted optional candidate selection. No model import, motion or publication."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def row_for(model_id,root=ROOT):
    return next((r for r in json.loads((Path(root)/'configs/deployment_models.json').read_text())['models']if r['id']==model_id),{})

def hold_gripper(environ=None,root=ROOT):
    environ=os.environ if environ is None else environ
    value=environ.get('COBOT_RLT_HOLD_RIGHT_GRIPPER')
    if value is not None:
        if value not in ('0','1'):raise ValueError('Gripper hold must be 0 or 1')
        if row_for(environ.get('COBOT_DEPLOYMENT_MODEL_ID'),root).get('hold_right_gripper') and value!='1':raise ValueError('Selected candidate requires held gripper')
        return value=='1'
    return row_for(environ.get('COBOT_DEPLOYMENT_MODEL_ID'),root).get('hold_right_gripper',False) is True

def resolve(model_id,root=ROOT,models_root=None):
    row=row_for(model_id,root);relative=row.get('supported_runtime')
    if not relative:return None
    if row.get('kind')!='rlt' or row.get('mode')!='online':raise ValueError('Supported runtime requires online RLT')
    if models_root is None:
        from .paths import RLT_MODELS
        models_root=row.get('supported_model_root') or RLT_MODELS
    base=Path(models_root).resolve();run=(base/relative).resolve()
    if base not in run.parents or 'candidates'not in run.parts:raise ValueError('Candidate path must remain isolated')
    profile=run/'profile.json';config=run/'online.yaml'
    if not profile.is_file()or not config.is_file():raise ValueError('Prepare supported runtime before loading')
    return dict(profile=str(profile),config=str(config),run=str(run))

if __name__=='__main__':
    import sys
    value=resolve(sys.argv[1]if len(sys.argv)>1 else'')
    if value:
        for key in ['profile','config','run']:print(value[key])
