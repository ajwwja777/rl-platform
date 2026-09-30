"""Exercise the evaluator's real make_loader with a synthetic Dataset, no JAX/models."""
import ast, pathlib, sys, types

def verify():
    import numpy as np
    import torch
    
    path=pathlib.Path(__file__).parents[1]/'methods/openpi_rlt/plug_v3_yyshadow/evaluate_stage1.py'
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_loader')
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,*a,**kw):self.episode_data_index={'from':[0,5],'to':[5,12]}
        def __len__(self):return 12
        def __getitem__(self,i):return np.asarray(i)
    mods={name:types.ModuleType(name) for name in ['lerobot','lerobot.common','lerobot.common.datasets','lerobot.common.datasets.lerobot_dataset','openpi','openpi.transforms','openpi.training','openpi.training.data_loader']}
    mods['lerobot.common.datasets.lerobot_dataset'].LeRobotDataset=Dataset
    mods['lerobot.common.datasets.lerobot_dataset'].LeRobotDatasetMetadata=lambda *a,**k:types.SimpleNamespace(fps=30,tasks=[])
    dl=mods['openpi.training.data_loader'];dl.transform_dataset=lambda ds,data:ds;dl._collate_fn=lambda xs:np.stack(xs);dl.DataLoaderImpl=lambda data,loader:loader
    mods['openpi'].transforms=mods['openpi.transforms'];mods['openpi.training'].data_loader=dl
    sys.modules.update(mods);ns={'torch':torch,'Path':pathlib.Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    data=types.SimpleNamespace(repo_id='synthetic',action_sequence_keys=['action'],prompt_from_task=False)
    cfg=types.SimpleNamespace(data=types.SimpleNamespace(create=lambda *a:data),assets_dirs=[],model=types.SimpleNamespace(action_horizon=50),batch_size=4)
    loader,size=ns['make_loader'](cfg,pathlib.Path('.'),[0,1],0)
    batches=list(loader);assert size==12;assert np.concatenate(batches).tolist()==list(range(12));assert len(batches)==3
    loader,size=ns['make_loader'](cfg,pathlib.Path('.'),[1],0)
    batches=list(loader);assert [len(b) for b in batches]==[4,3];assert np.concatenate(batches).tolist()==list(range(5,12))
    print('PASS: real evaluator make_loader: finite complete Episodes, no repeats, partial final batch retained; synthetic data only.')

def test_finite_evaluation_loader():
    import pytest
    pytest.importorskip("torch")
    import sys
    before=dict(sys.modules)
    try:
        verify()
    finally:
        for name in list(sys.modules):
            if name.startswith(("lerobot", "openpi")):
                if name in before: sys.modules[name]=before[name]
                else: sys.modules.pop(name,None)

if __name__=="__main__":
    verify()
