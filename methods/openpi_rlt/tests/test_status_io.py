from types import SimpleNamespace
from methods.openpi_rlt.cobot_adapter.status_io import install_status_write_throttle

def test_idle_and_training_writes_are_bounded_but_readiness_is_immediate():
    writes=[];callbacks=[];now=[0.]
    class Learner:
        def _write_status(self, progress): writes.append(dict(progress))
    install_status_write_throttle(Learner, clock=lambda:now[0], register_exit=callbacks.append)
    obj=Learner()
    for step in range(100):
        obj._write_status(dict(global_step=step,actor_version=0,ready_for_online=False))
    assert len(writes)==1
    now[0]=1.
    obj._write_status(dict(global_step=100,actor_version=0,ready_for_online=False))
    assert writes[-1]["global_step"]==100
    now[0]=1.01
    obj._write_status(dict(global_step=101,actor_version=1,ready_for_online=True))
    assert len(writes)==3 and writes[-1]["ready_for_online"]
    obj._write_status(dict(global_step=102,actor_version=1,ready_for_online=True))
    assert len(writes)==3
    callbacks[0]()
    assert writes[-1]["global_step"]==102

def test_failed_status_write_is_not_cached_as_success():
    callbacks=[]
    class Learner:
        calls=0
        def _write_status(self, progress):
            self.calls+=1
            if self.calls==1: raise OSError("disk")
    install_status_write_throttle(Learner,clock=lambda:0.,register_exit=callbacks.append)
    obj=Learner()
    import pytest
    with pytest.raises(OSError):obj._write_status({})
    obj._write_status({})
    assert obj.calls==2
