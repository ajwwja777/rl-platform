import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from methods.openpi_rlt.realtime_expo.dataset import ExpertDataset

def fixture(root):
 records=[]
 for native in [1,5]:
  path=root/f'episode_{native:06}.npz';state=np.arange(42,dtype=np.float32).reshape(6,7);image=np.zeros((6,224,224,3),np.uint8)
  np.savez_compressed(path,state=state,action=state+2,left=image,right=image,valid=[True,True,False,True,True,True])
  records.append(dict(file=path.with_suffix('.hdf5').name,native_id=native,uuid=str(native),accepted_frames=5,segments=[[0,2],[3,6]],payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 (root/'manifest.json').write_text(json.dumps(dict(records=records)))
 return records

def test_train_dev_and_segment_padding(tmp_path):
 fixture(tmp_path);train=ExpertDataset(tmp_path,'train',4);dev=ExpertDataset(tmp_path,'dev',4)
 assert len(train)==len(dev)==5
 assert [r['native_id'] for r in train.identities]==[1]
 assert [r['native_id'] for r in dev.identities]==[5]
 np.testing.assert_array_equal(train[0]['actions'][:,0],[2,9,9,9])
 np.testing.assert_array_equal(train[2]['actions'][:,0],[23,30,37,37])
 assert train[-1]['actions'].dtype==np.float32

def test_reject_payload_change(tmp_path):
 fixture(tmp_path);f=tmp_path/'episode_000001.npz';f.write_bytes(f.read_bytes()+b'changed')
 with pytest.raises(ValueError,match='SHA'):ExpertDataset(tmp_path,'train')

def test_reject_duplicate_uuid(tmp_path):
 rows=fixture(tmp_path);rows[1]['uuid']=rows[0]['uuid'];(tmp_path/'manifest.json').write_text(json.dumps(dict(records=rows)))
 with pytest.raises(ValueError,match='duplicate'):ExpertDataset(tmp_path,'train')

def test_invalid_segment_never_bridged(tmp_path):
 rows=fixture(tmp_path);rows[0]['segments']=[[0,6]];(tmp_path/'manifest.json').write_text(json.dumps(dict(records=rows)))
 with pytest.raises(ValueError,match='invalid segment'):ExpertDataset(tmp_path,'train')


def test_wrist_transform_preserves_input_and_absolute_gripper():
 from methods.openpi_rlt.realtime_expo.configuration import WristInputs
 from openpi import transforms
 image=np.zeros((224,224,3),np.uint8);state=np.ones(7,np.float32);actions=np.full((50,7),2,np.float32)
 raw=dict(state=state,actions=actions,images={'left_wrist_0_rgb':image,'right_wrist_0_rgb':image},prompt='plug')
 converted=transforms.DeltaActions(transforms.make_bool_mask(6,-1))(WristInputs()(raw))
 np.testing.assert_array_equal(raw['actions'],np.full((50,7),2))
 np.testing.assert_array_equal(converted['actions'][:,:6],np.ones((50,6)))
 np.testing.assert_array_equal(converted['actions'][:,6],np.full(50,2))
 assert not converted['image_mask']['base_0_rgb']
 del raw['actions'];assert 'actions' not in WristInputs()(raw)
