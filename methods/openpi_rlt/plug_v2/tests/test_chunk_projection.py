import numpy as np
import pytest
from methods.openpi_rlt.plug_v2.chunk_projection import project_correction

def test_identity_and_passive_gripper():
    r=np.arange(70,dtype=np.float32).reshape(10,7)/100
    np.testing.assert_array_equal(project_correction(r,r),r)
    a=r.copy();a[:,6]+=20
    np.testing.assert_array_equal(project_correction(a,r),r)

def test_linear_correction_is_preserved():
    r=np.zeros((2,10,7),np.float32);a=r.copy()
    a[:,:,:6]=np.arange(10)[None,:,None]*.003+.04
    np.testing.assert_allclose(project_correction(a,r),a,atol=1e-7)

def test_alternating_noise_attenuated():
    r=np.zeros((10,7),np.float32);a=r.copy();a[:,:6]=((-1.)**np.arange(10))[:,None]*.02
    result=project_correction(a,r)
    assert np.linalg.norm(np.diff(result[:,:6],n=2,axis=0))<1e-6
    assert np.linalg.norm(result)<np.linalg.norm(a)

def test_invalid_input_rejected():
    r=np.zeros((10,7),np.float32)
    with pytest.raises(ValueError):project_correction(r+np.nan,r)
    with pytest.raises(ValueError):project_correction(r[:9],r)
    with pytest.raises(ValueError):project_correction(r,r,2)
