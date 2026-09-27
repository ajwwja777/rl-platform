import ast,types,threading,json
from pathlib import Path
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[4]

def method():
    t=ast.parse((ROOT/'methods/openpi_rlt/plug_v2/runtime.py').read_text());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Runtime');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='start_episode')
    ns={'json':json,'__package__':'methods.openpi_rlt.plug_v2'};exec(compile(ast.Module(body=[f],type_ignores=[]),'runtime-start-episode','exec'),ns);return ns['start_episode']

@pytest.mark.parametrize('mode,expected',[('warmup',[5000,5000]),('latest',[5000,5485])])
def test_version_only_selected_on_episode_boundary(mode,expected):
    obj=types.SimpleNamespace(frame_lock=threading.RLock(),rows=[],plans=[],deadline_recoveries=0,
        passive_commands=types.SimpleNamespace(reset=lambda:None),
        deadline_budget=types.SimpleNamespace(reset_episode=lambda:None),
        deadline_reanchor_pending=False,args=types.SimpleNamespace(actor=mode),
        v5=True,v5_release=None,app=types.SimpleNamespace(update_metrics=lambda **k:None))
    releases=iter([(Path('a'),{'name':'a','global_step':5000,'checkpoint_sha256':'a'}),(Path('b'),{'name':'b','global_step':5485,'checkpoint_sha256':'b'})])
    def select(p):return object(),5000 if str(p)=='a' else 5485
    with patch('methods.openpi_rlt.plug_v2.rtc_release.selected',side_effect=lambda:next(releases)),patch('methods.openpi_rlt.plug_v2.rtc_actor_client.select',side_effect=select):
        f=method();f(obj);first=obj.actor;assert obj.actor is first
        actual=[obj.actor_version];f(obj);actual.append(obj.actor_version)
    assert actual==expected
