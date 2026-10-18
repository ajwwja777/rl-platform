import importlib.util,os
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('hpc_gateway',Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter/hpc_gateway.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def config():return dict(job_id=123,slurm_user='owner',port=18443,learner_port=18443,run_dir='/owned/run')
def output(**changes):
 fields=dict(JobId='123',UserId='owner(%d)'%os.getuid(),JobState='RUNNING',NumNodes='1',NodeList='ACD1-8');fields.update(changes)
 return ' '.join('%s=%s'%x for x in fields.items())
def test_gateway_uses_only_owned_single_node():
 g.validate(config());assert g.allocated_node(config(),output())=='ACD1-8'
@pytest.mark.parametrize('change',[dict(JobId='124'),dict(UserId='other(123)'),dict(JobState='PENDING'),dict(NodeList='node;id'),dict(NumNodes='2'),dict(NodeList='node[1-2]')])
def test_gateway_rejects_wrong_allocation(change):
 with pytest.raises(ValueError):g.allocated_node(config(),output(**change))
@pytest.mark.parametrize('change',[dict(job_id='123;id'),dict(slurm_user='owner@host'),dict(port=80),dict(learner_port=True),dict(run_dir='relative')])
def test_gateway_validates_fixed_profile(change):
 with pytest.raises(ValueError):g.validate({**config(),**change})


def test_hpc_learner_requires_own_allocation(tmp_path,monkeypatch):
 import json,sys
 sys.path.insert(0,str(Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter'))
 import hpc_learner
 profile=tmp_path/'profile.json';profile.write_text(json.dumps(dict(job_id=123)))
 monkeypatch.setattr(sys,'argv',['hpc_learner','--config',str(profile)])
 monkeypatch.delenv('SLURM_JOB_ID',raising=False)
 with pytest.raises(ValueError,match='configured Slurm'):hpc_learner.main()
 monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
 with pytest.raises(ValueError,match='four assigned'):hpc_learner.main()
