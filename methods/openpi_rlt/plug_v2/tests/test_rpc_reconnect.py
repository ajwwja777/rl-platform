import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('rpc_under_test', Path(__file__).parents[1] / 'rpc.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Closed(Exception):
    pass


class FakeSocket:
    def __init__(self, metadata, *, close_on_send=False, timeout_on_recv=False):
        self.metadata=metadata;self.close_on_send=close_on_send;self.timeout_on_recv=timeout_on_recv
        self.metadata_sent=False;self.closed=False
    def send(self, payload):
        if self.close_on_send:
            self.close_on_send=False
            raise Closed()
    def recv(self, timeout=None):
        if not self.metadata_sent:
            self.metadata_sent=True
            return self.metadata
        if self.timeout_on_recv:raise TimeoutError('late')
        return self.metadata
    def close(self):self.closed=True


def pack(value):return MODULE.msgpack_numpy.packb(value)


def test_closed_socket_reconnects_once(monkeypatch):
    sockets=[FakeSocket(pack({'id':1}),close_on_send=True),FakeSocket(pack({'id':1}))]
    monkeypatch.setattr(MODULE,'ConnectionClosed',Closed)
    monkeypatch.setattr(MODULE,'connect',lambda *args,**kwargs:sockets.pop(0))
    client=MODULE.ModelClient()
    assert client.infer({'x':1})=={'id':1}
    assert client.reconnect_count==1
    client.close();client.close()


def test_timeout_resets_socket_and_next_request_reconnects(monkeypatch):
    first=FakeSocket(pack({'id':1}),timeout_on_recv=True)
    sockets=[first,FakeSocket(pack({'id':1}))]
    monkeypatch.setattr(MODULE,'connect',lambda *args,**kwargs:sockets.pop(0))
    client=MODULE.ModelClient()
    with pytest.raises(TimeoutError):client.infer({'x':1})
    assert first.closed and client._ws is None
    assert client.infer({'x':2})=={'id':1}
