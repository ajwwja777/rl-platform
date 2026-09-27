# Explicit flat-series adaptation; no nested Task/Model/Round directories.
from pathlib import PurePosixPath
from urllib.parse import urlsplit,parse_qs,urlencode,urlunsplit
from methods.openpi_rlt.cobot_adapter.task5_client import Task5Client

class FlatRecorderClient(Task5Client):
    def _request(self, method, path, body=None):
        if method=='POST' and path=='/api/episodes/stop':
            # Stop acquisition promptly but allow queued image frames to drain.
            return Task5Client(self._base_url,timeout_sec=120)._request(method,path,body)
        if method=='POST' and path in ('/api/storage/prepare','/api/episodes/start'):
            body={**body,'storage_layout':'flat'}
        parts=urlsplit(path)
        query=parse_qs(parts.query)
        if 'episode_relative_path' in query:
            previous=query['episode_relative_path']
            if len(previous)!=1:raise ValueError('ambiguous episode path')
            basename=PurePosixPath(previous[0]).name
            if not basename.startswith('episode_') or not basename.endswith('.hdf5'):
                raise ValueError('invalid flat episode path')
            query['episode_relative_path']=[basename]
            path=urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(query,doseq=True),parts.fragment))
        return super()._request(method,path,body)
