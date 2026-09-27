# Fixed installed openpi msgpack contract with bounded loopback I/O.
import os
os.environ['NO_PROXY']=','.join(filter(None,[os.environ.get('NO_PROXY',''),'127.0.0.1','localhost','::1']))
os.environ['no_proxy']=os.environ['NO_PROXY']
from openpi_client import msgpack_numpy
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

class ModelClient:
    def __init__(self,host='127.0.0.1',port=8020):
        if host not in ('127.0.0.1','localhost'):raise ValueError('model RPC must remain loopback')
        self._host=host;self._port=port
        self._packer=msgpack_numpy.Packer()
        self._ws=None;self._metadata=None;self.reconnect_count=0
        self._connect()
    def _connect(self):
        ws=connect(f'ws://{self._host}:{self._port}',compression=None,max_size=16*1024*1024,
                   open_timeout=5,close_timeout=1)
        try:metadata=msgpack_numpy.unpackb(ws.recv(timeout=5))
        except Exception:
            ws.close();raise
        if self._metadata is not None and metadata!=self._metadata:
            ws.close();raise RuntimeError('model RPC metadata changed across reconnect')
        self._metadata=metadata;self._ws=ws
    def _reset(self):
        ws,self._ws=self._ws,None
        if ws is not None:
            try:ws.close()
            except Exception:pass
    def get_server_metadata(self):return dict(self._metadata)
    def infer(self,obs):
        payload=self._packer.pack(obs)
        for attempt in range(2):
            if self._ws is None:self._connect()
            try:
                self._ws.send(payload)
                response=self._ws.recv(timeout=3)
                break
            except ConnectionClosed:
                self._reset()
                if attempt:raise
                self.reconnect_count+=1
            except Exception:
                # A timed-out response is stale for RTC. Drop this socket; the
                # next fresh request reconnects instead of reusing it.
                self._reset();raise
        if isinstance(response,str):raise RuntimeError('model RPC error: '+response)
        return msgpack_numpy.unpackb(response)
    def close(self):self._reset()
