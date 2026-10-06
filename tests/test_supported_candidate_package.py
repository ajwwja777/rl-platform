import numpy as np
from scripts.prepare_supported_candidate import canonicalize_training_row


def test_legacy_expert_phase_and_id_match_training_sampler_contract():
    original=dict(phase_online=False,collection_phase_id=np.uint8(0),episode_id=100003,source=0)
    row=canonicalize_training_row(original)
    assert row['collection_phase_id']==1 and row['collection_phase']=='warmup'
    assert row['episode_id']==-4 and 'phase_online' not in row
    assert original['episode_id']==100003 and original['collection_phase_id']==0


def test_regular_and_online_ids_are_preserved():
    for online,episode in [(False,36),(True,239)]:
        row=canonicalize_training_row(dict(phase_online=online,collection_phase_id=0,episode_id=episode))
        assert row['episode_id']==episode
        assert row['collection_phase_id']==(2 if online else 1)
