import json
from pathlib import Path
import pytest
from methods.openpi_rlt.cobot_adapter.online_cycle import read_optional_json

def test_missing_request_is_idle(tmp_path):
    assert read_optional_json(tmp_path/"request.json")=={}

def test_disappears_on_open_then_new_request_is_seen(tmp_path, monkeypatch):
    path=tmp_path/"request.json";path.write_text('{"request_id":"old"}')
    original=Path.read_text
    calls=0
    def racing_read(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:
            self.unlink()
            raise FileNotFoundError(str(self))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",racing_read)
    assert read_optional_json(path)=={}
    path.write_text('{"request_id":"new"}')
    assert read_optional_json(path)=={"request_id":"new"}

def test_no_exists_check(tmp_path,monkeypatch):
    path=tmp_path/"request.json";path.write_text('{"request_id":"new"}')
    def forbidden(*args):raise AssertionError("TOCTOU exists check")
    monkeypatch.setattr(Path,"exists",forbidden)
    assert read_optional_json(path)["request_id"]=="new"

def test_corrupt_json_still_fails(tmp_path):
    path=tmp_path/"request.json";path.write_text("{")
    with pytest.raises(json.JSONDecodeError):read_optional_json(path)

def test_permission_error_still_fails(tmp_path,monkeypatch):
    def denied(*args,**kwargs):raise PermissionError("denied")
    monkeypatch.setattr(Path,"read_text",denied)
    with pytest.raises(PermissionError):read_optional_json(tmp_path/"request.json")
