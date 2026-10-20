"""Probe control-flow checks with immutable CPU doubles, never a GPU qualification."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import os
import numpy as np
from test_expo_continuous_actor import actor, module


def test_probe_collects_true_stage_count_and_writes_isolated_receipts(actor, monkeypatch, tmp_path):
    script = runpy.run_path(str(Path(__file__).parents[1]/'scripts/verify_expo_continuous_actor.py'))
    def native(self, route, data):
        self.agent = self.agent.replace(rng=self.agent.rng + (1 if route == '/precache' else 10))
        if route == '/precache':
            return dict(precached=np.zeros((8, 16, 7)).tolist(), delay=len(data['prefix']), actor_version=self.meta['updates'])
        return dict(actions=np.full((8, 7), self.agent.version).tolist(), actor_version=self.meta['updates'])
    monkeypatch.setattr(module.Actor, 'infer', native)
    monkeypatch.setattr(module, 'restore', lambda path, template, identity: (template.replace(version=2), {'actor_version': 2}))
    snapshot = actor.args.snapshot_root/'2'
    snapshot.mkdir()
    (snapshot/'manifest.json').write_text(json.dumps({'actor_version': 2}))
    common = tmp_path/'assets/common'
    common.mkdir(parents=True)
    np.savez(common/'dev_observation.npz', state=np.zeros(7), left_wrist_image=np.zeros((2, 2, 3)), right_wrist_image=np.zeros((2, 2, 3)))
    args = SimpleNamespace(project=tmp_path, bundle=tmp_path/'assets/bundle', snapshots=actor.args.snapshot_root,
        initial=0, updated=2, samples=20, stage_timeout=5, output=tmp_path/'probe/result.json')
    result = dict(pid=os.getpid(), probe_uuid='cpu-double', samples=[])
    script['qualify'](args, result)
    assert result['passed'] and result['receipt']['policy_versions'] == [0, 2]
    assert result['canceled_precache_preserves_rng'] and result['persistent_episode_rng']
    assert result['stage_warmup_calls'] == 0
    assert len(result['samples']) >= 61
    assert not (tmp_path/'actor_receipts').exists()
    assert Path(result['receipt_directory']).parent == args.output.parent
    # Summary fields must remain ordinary JSON numbers, including numpy comparisons.
    value = script['summary']([dict(total_seconds=.2), dict(total_seconds=.7)], .5)
    assert value['above_d5_reference'] == 1
    json.dumps(value, allow_nan=False)
