import sys
from pathlib import Path
import zlib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"methods/expo_ft/cobot_adapter"))
from wire import encode,decode

@pytest.mark.parametrize("compressed",[False,True])
def test_image_actions_and_episode_are_lossless(compressed):
    rng=np.random.default_rng(81)
    values={"left_wrist_image":rng.integers(0,256,(180,320,3),dtype=np.uint8),
            "state":rng.normal(size=7).astype(np.float32),"prefix":rng.normal(size=(5,7)).astype(np.float32),
            "episode_uuid":np.asarray("ef44e1fa-92e9-40ab-8127-71437a77350e")}
    body,encoding=encode(values,compressed)
    actual=decode(body,encoding)
    for key in values:
        assert actual[key].dtype==values[key].dtype
        np.testing.assert_array_equal(actual[key],values[key])

def test_rejects_truncation_trailing_and_expansion():
    body,_=encode({"state":np.arange(7)},True)
    for invalid in (body[:-1],body+b"trailing",zlib.compress(bytes(3*1024*1024))):
        with pytest.raises(ValueError):decode(invalid,"deflate")
