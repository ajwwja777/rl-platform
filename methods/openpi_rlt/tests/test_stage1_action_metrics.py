import numpy as np


def test_service_image_recovery_preserves_all_uint8_values():
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import recover_uint8_image
    pixels=np.arange(256,dtype=np.uint8).reshape(16,16)
    normalized=pixels.astype(np.float32)/255.0*2.0-1.0
    np.testing.assert_array_equal(recover_uint8_image(normalized),pixels)


def test_service_image_recovery_rejects_nonimage_values():
    import pytest
    from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import recover_uint8_image
    for invalid in [np.asarray([np.nan]),np.asarray([2.]),np.asarray([0.])]:
        with pytest.raises(ValueError):recover_uint8_image(invalid)
import pytest
from methods.openpi_rlt.plug_v3_yyshadow.stage1_action_metrics import interpolate_chunk, time_targets, action_summary


def test_ramp_physical_time_resampling():
    a=np.arange(50, dtype=float)[:, None]*np.ones((1,7))
    resampled=interpolate_chunk(a,30,20,10)
    np.testing.assert_allclose(resampled[:,0],np.arange(10)*1.5)
    truth,valid=time_targets(a,30,20,10)
    np.testing.assert_allclose(truth[0],resampled)
    assert valid[0].all() and valid[-1].sum()==1
    assert not valid[-2,1]


def test_tail_excluded_instead_of_fabricated_independence():
    a=np.zeros((2,7));truth,valid=time_targets(a,30,30,3)
    pred=truth.copy();pred[~valid]=999
    report=action_summary(pred,truth,valid)
    assert report['mae_per_dim']==[0.]*7 and report['valid_action_slots']==3
    assert report['padded_slots_excluded']==3


def test_horizon_refuses_extrapolation():
    with pytest.raises(ValueError):interpolate_chunk(np.zeros((10,7)),30,20,10)


def test_invalid_shapes_and_nan_fail():
    with pytest.raises(ValueError):action_summary(np.zeros((2,10,6)),np.zeros((2,10,6)),np.ones((2,10)))
    a=np.zeros((2,10,7));a[0,0,0]=np.nan
    with pytest.raises(ValueError):action_summary(a,np.zeros_like(a),np.ones((2,10)))


def test_serving_default_is_exact_and_retiming_uses_full_horizon():
    from methods.openpi_rlt.plug_v3_yyshadow.serve_stage1 import sample_reference, load
    a=np.arange(350,dtype=np.float32).reshape(50,7)
    np.testing.assert_array_equal(sample_reference(a),a[:10])
    np.testing.assert_array_equal(sample_reference(a,20),interpolate_chunk(a,30,20,10))
    with pytest.raises(ValueError):sample_reference(a,40)
    with pytest.raises(ValueError,match='RTC'):load(None,None,reference_hz=20,rtc_overlay='unused')
