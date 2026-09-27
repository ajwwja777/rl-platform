from types import SimpleNamespace
from methods.openpi_rlt.cobot_adapter.collection_gate import install_collection_gate

def test_collection_never_trains_or_latches_above_threshold(tmp_path):
    writes=[]
    class Learner:
        def train_once(self, **kwargs): raise AssertionError("training called")
        def _refresh_progress(self, stats): raise AssertionError("warmup latch called")
        def _write_status(self, progress): raise AssertionError("old status called")
    install_collection_gate(Learner, write_json=lambda p,v:writes.append(v))
    obj=Learner()
    obj._state=SimpleNamespace(global_step=0,actor_version=0)
    obj._rl_config=SimpleNamespace(warmup_post_collect_updates=200,grad_updates_per_cycle=1)
    obj._status_path=tmp_path/"status.json"
    obj._warmup_ready_adds_total=None
    obj._pending_update_budget=0
    for size in (98,128,1000):
        obj._replay_source=SimpleNamespace(stats=lambda:dict(size=size,adds_total=size))
        assert obj.train_once() is None
        assert writes[-1]["replay_size"]==size
        assert writes[-1]["training_frozen"] is True
        assert writes[-1]["ready_for_online"] is False
        assert writes[-1]["global_step"]==0
        assert obj._warmup_ready_adds_total is None
        assert obj._pending_update_budget==0
