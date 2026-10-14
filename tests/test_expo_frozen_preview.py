import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

ADAPTER=Path(__file__).resolve().parents[1]/'methods/expo_ft/cobot_adapter'
sys.path.insert(0,str(ADAPTER))
from frozen_preview import PreviewRuntime

class FrozenPreviewTests(unittest.TestCase):
    def runtime(self):
        value=object.__new__(PreviewRuntime)
        value.args=SimpleNamespace(model_id='preview')
        value.episode='test'
        return value
    def test_rejects_online_or_wrong_scale_identity(self):
        runtime=self.runtime()
        status=dict(deployment_id='preview',actor_version=0,rtc_delay=5,edit_scale=.01,training_enabled=False,frozen_preview=True)
        runtime.validate_server_status(status)
        for key,value in [('training_enabled',True),('actor_version',1),('edit_scale',.1),('rtc_delay',2),('frozen_preview',False)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):runtime.validate_server_status({**status,key:value})
    def test_selection_uses_fresh_observation(self):
        runtime=self.runtime();calls=[]
        runtime.observation=lambda:{'state':np.full(7,9.)}
        def rpc(path,**values):
            calls.append((path,values));return {'actions':np.zeros((8,7)).tolist()}
        runtime.array_rpc=rpc
        result=runtime.resolve_inference(dict(precached=np.zeros((32,16,7)),delay=5,actor_version=0))
        self.assertEqual(result.shape,(8,7));self.assertEqual(calls[0][0],'/select')
        np.testing.assert_array_equal(calls[0][1]['state'],np.full(7,9.))
    def test_slow_phase_preserves_physical_prefix(self):
        runtime=self.runtime();calls=[]
        runtime.array_rpc=lambda path,**values:calls.append((path,values)) or {'ok':True}
        prefix=np.arange(35,dtype=np.float32).reshape(5,7)
        runtime.infer({'state':np.zeros(7)},prefix)
        self.assertEqual(calls[0][0],'/precache');np.testing.assert_array_equal(calls[0][1]['prefix'],prefix)
    def test_preview_has_no_online_status(self):
        status=self.runtime().status_extra()
        self.assertFalse(status['training_enabled']);self.assertNotIn('online',status)

if __name__=='__main__':unittest.main()
