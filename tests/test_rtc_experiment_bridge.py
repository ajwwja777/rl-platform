import numpy as np
import pytest
from methods.openpi_rlt.experiments.rtc import encode_prefix

def observation():
    return {"state":np.zeros(7),"images":{}}

def transform(values):
    return {"actions":np.pad(values["actions"]-values["state"],((0,0),(0,25)))}

def test_seven_dimensional_prefix_uses_model_transform():
    request=dict(previous_actions=np.ones((10,7)),delay_steps=2,execution_horizon=5)
    encoded,delay,horizon=encode_prefix(observation(),request,transform)
    assert encoded.shape==(50,32) and delay==2 and horizon==5
    assert (encoded[:10,:7]==1).all() and not encoded[10:].any()
    assert not observation()["state"].any()

@pytest.mark.parametrize("change",[
    {"delay_steps":5,"execution_horizon":5},{"previous_actions":np.zeros((10,14))},
    {"execution_horizon":11},{"previous_actions":np.full((10,7),np.nan)},
    {"delay_steps":True},{"extra":1}])
def test_invalid_prefix_rejected(change):
    request=dict(previous_actions=np.ones((10,7)),delay_steps=2,execution_horizon=5)
    request.update(change)
    with pytest.raises(ValueError):encode_prefix(observation(),request,transform)

def test_wrong_transform_rejected():
    with pytest.raises(ValueError):
        encode_prefix(observation(),dict(previous_actions=np.zeros((10,7)),delay_steps=1,execution_horizon=5),lambda x:x)
