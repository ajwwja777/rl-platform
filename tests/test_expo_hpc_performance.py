"""Exercise the real HPC launcher up to its process-exec boundary, without GPUs."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
LAUNCHER=Path(__file__).parents[1]/'methods/expo_ft/cobot_adapter/hpc_learner.py'

def launch(tmp_path,**options):
    run=tmp_path/'run';run.mkdir();(run/'committed.json').write_text('{}')
    capture=tmp_path/'exec.json'
    interpreter=tmp_path/'capture-python'
    interpreter.write_text('#!'+sys.executable+'\nimport os,sys,json\nfrom pathlib import Path\nPath('+repr(str(capture))+').write_text(json.dumps(dict(argv=sys.argv[1:],env={k:v for k,v in os.environ.items() if k.startswith(("XLA_","OMP_","CUDA_","JAX_"))})))\n')
    interpreter.chmod(0o755)
    cfg=dict(job_id=123,run_dir=str(run),python=str(interpreter),project='/native',bundle='/bundle',seed_dir='/seed',cache_dir='/cache',model_id='test',port=18443,rtc_delay=5,episodes_per_update=5,critic_prior='demo50_success30_failure20',edit_scale=.01,**options)
    profile=tmp_path/'profile.json';profile.write_text(json.dumps(cfg))
    env=dict(os.environ,SLURM_JOB_ID='123',CUDA_VISIBLE_DEVICES='0,1,2,3',XLA_PYTHON_CLIENT_ALLOCATOR='platform')
    result=subprocess.run([sys.executable,str(LAUNCHER),'--config',str(profile)],env=env,text=True,capture_output=True)
    return result,json.loads(capture.read_text()) if capture.exists() else None

def test_hpc_default_keeps_existing_launcher_and_allocator(tmp_path):
    result,value=launch(tmp_path)
    assert result.returncode==0,result.stderr
    assert value['argv'][0].endswith('/observed_learner.py')
    assert value['env']['XLA_PYTHON_CLIENT_ALLOCATOR']=='platform'
    assert value['argv'][-2:]==['--num-devices','4']
    runtime=json.loads((tmp_path/'run/hpc_runtime.json').read_text())
    assert runtime['forward_microbatch']==4 and runtime['gpu_allocator']=='platform'

def test_hpc_pool_override_reaches_child_without_stale_allocator(tmp_path):
    result,value=launch(tmp_path,gpu_allocator='default',memory_fraction=.65)
    assert result.returncode==0,result.stderr
    assert 'XLA_PYTHON_CLIENT_ALLOCATOR' not in value['env']
    assert value['env']['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.65'
    assert value['env']['XLA_PYTHON_CLIENT_PREALLOCATE']=='false'
    assert value['env']['CUDA_VISIBLE_DEVICES']=='0,1,2,3'

@pytest.mark.parametrize('options',[{'gpu_allocator':'unknown'},{'memory_fraction':0},{'memory_fraction':1.1},{'memory_fraction':True},{'memory_fraction':float('nan')}])
def test_hpc_rejects_invalid_performance_options_before_child_exec(tmp_path,options):
    result,value=launch(tmp_path,**options)
    assert result.returncode!=0
    assert value is None
