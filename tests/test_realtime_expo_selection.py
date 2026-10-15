import copy
import pytest
from methods.openpi_rlt.realtime_expo.selection import select_checkpoint

def report(tag,joint=.01,gripper=.002,p95=.02):
    cohort=lambda uuid:dict(per_episode=[dict(uuid=uuid,frames=10,evaluated_frames=10)],summary=dict(episode_equal_mean=dict(c10_joint_mae_rad=joint,c10_gripper_mae_m=gripper,c10_p95_per_dim=[p95]*6+[gripper])))
    return dict(checkpoint=f'/candidate/{tag}',effective_batch_size=1,dataset_manifest_sha256='same',
        metadata=dict(arm='left',camera_keys=['left','right'],model_action_sampling_hz=10,denoising_steps=10,compute_restore_dtype='same',norm_stats_sha256='same'),
        cohorts=dict(train=cohort('train'),dev=cohort('dev')))

def test_gripper_conflict_prevents_automatic_selection():
    result=select_checkpoint([report(1000,.02,.002),report(2000,.01,.004)])
    assert result['selected_checkpoint'] is None and result['conflicts']==['gripper_mae_m']

def test_near_tie_prefers_earlier_checkpoint():
    result=select_checkpoint([report(1000,.01005),report(2000,.01)])
    assert result['selected_checkpoint']=='/candidate/1000'

@pytest.mark.parametrize('defect',['partial','leak','manifest','batch','norm'])
def test_invalid_comparisons_rejected(defect):
    a,b=report(1000),report(2000)
    if defect=='partial':b['cohorts']['dev']['per_episode'][0]['evaluated_frames']=9
    if defect=='leak':b['cohorts']['dev']['per_episode'][0]['uuid']='train'
    if defect=='manifest':b['dataset_manifest_sha256']='other'
    if defect=='batch':b['effective_batch_size']=8
    if defect=='norm':b['metadata']['norm_stats_sha256']='other'
    with pytest.raises(ValueError):select_checkpoint([a,b])
