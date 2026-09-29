import pickle
import numpy as np
from methods.openpi_rlt.experiments.runtime import CreditIndex

def test_incremental_terminal_labels_and_partial_pickle(tmp_path):
    path=tmp_path/"journal.pkl"
    rows=[dict(collection_phase_id=2,episode_id=1,step_id=i,rewards=np.array([float(i==2)]),done=i==2,success=int(i==2)) for i in range(3)]
    path.write_bytes(b"".join(pickle.dumps(r) for r in rows[:2]))
    index=CreditIndex(path,.9)
    batch={k:np.stack([r[k] for r in rows[:2]]) for k in rows[0]}
    result=index.attach(batch)
    assert not result["mc_valid"].any()
    last=pickle.dumps(rows[-1])
    with path.open("ab") as f:f.write(last[:10])
    assert not index.attach(batch)["mc_valid"].any()
    with path.open("ab") as f:f.write(last[10:])
    result=index.attach(batch)
    assert result["mc_valid"].all()
    np.testing.assert_allclose(result["mc_return"],[.81,.9])
    assert not batch["success"].any()

def test_missing_identity_uses_native_td(tmp_path):
    path=tmp_path/"empty.pkl";path.touch()
    batch=dict(collection_phase_id=np.array([2]),episode_id=np.array([99]),step_id=np.array([0]))
    assert not CreditIndex(path,.99).attach(batch)["mc_valid"].any()
