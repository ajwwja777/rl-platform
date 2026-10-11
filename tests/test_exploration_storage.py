"""No robot or recorder: web directory selection must survive Episode latching."""
import json
from dataclasses import dataclass
from pathlib import Path
import pytest
from integrations.cobot_runtime import exploration_collection as explore, profile_storage as storage
from integrations.cobot_runtime.shared_model_env import SharedEpisodeLifecycle


@pytest.fixture
def roots(monkeypatch,tmp_path):
    settings=tmp_path/'settings.json'
    monkeypatch.setattr(storage,'SETTINGS',settings)
    monkeypatch.setattr(storage,'LEGACY_SETTINGS',tmp_path/'absent.json')
    monkeypatch.setattr(storage,'ALLOWED',tmp_path)
    monkeypatch.setattr(storage,'LEGACY_BASE',tmp_path/'legacy')
    monkeypatch.setattr(explore,'locations',lambda:{'recording':tmp_path/'private'})
    return settings,tmp_path


def test_web_selected_directory_overrides_private_default(roots):
    settings,root=roots
    settings.write_text(json.dumps({'selected':str(root/'test'),'current':{'online':str(root/'old')}}))
    assert explore.recording_root('online')==root/'test'


def test_phase_selection_and_unselected_fallback(roots):
    settings,root=roots
    assert explore.recording_root('online')==root/'private'
    settings.write_text(json.dumps({'current':{'online':str(root/'phase')}}))
    assert explore.recording_root('online')==root/'phase'


def test_invalid_user_selection_is_not_silently_ignored(roots):
    settings,root=roots
    settings.write_text(json.dumps({'selected':'/outside-allowed-data'}))
    with pytest.raises(ValueError): explore.recording_root('online')


def test_directory_latched_per_episode_and_refreshed_next_episode(roots):
    settings,root=roots
    @dataclass
    class Identity:
        data_root:str
    class Recorder:
        def start_episode(self,identity):return identity
        def finish_episode(self,reference,outcome):return reference
    lifecycle=SharedEpisodeLifecycle(Recorder(),read_use=lambda:'collection',root_for_phase=explore.recording_root)
    settings.write_text(json.dumps({'selected':str(root/'first')}))
    first=lifecycle.start_episode(Identity('/ignored'))
    settings.write_text(json.dumps({'selected':str(root/'second')}))
    assert lifecycle.identity.data_root==str(root/'first')
    lifecycle.finish_episode(first,'failure')
    second=lifecycle.start_episode(Identity('/ignored'))
    assert second.data_root==str(root/'second')
    assert explore.locations()=={'recording':root/'private'}
