"""Async transport never waits for the optimizer lock; committed snapshots only."""
import json
import re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from actor_http import serve as serve_file


def make_server(learner,port):
    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm=True
        def log_message(self,*args):pass
        def send(self,status,value):
            body=json.dumps(value,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                if self.path in ('/ready','/status'):
                    return self.send(200,dict(learner.snapshot(),step=2000))
                if self.path=='/actor/manifest':
                    version=learner.published_version
                    if version<0:raise ValueError('No committed Actor snapshot')
                    return self.send(200,json.loads((learner.root/'actors'/str(version)/'manifest.json').read_text()))
                match=re.fullmatch(r'/actor/([0-9]+)/((?:actor|encoder|edit|critic)-[0-9]+\.gz)',self.path)
                if match and int(match[1]) in learner.known_versions():
                    directory=learner.root/'actors'/match[1]
                    manifest=json.loads((directory/'manifest.json').read_text())
                    record=next(x for x in manifest['records'] if x.get('file')==match[2])
                    return serve_file(self,directory/match[2],record['bytes'])
                return self.send(404,{'error':'not_found'})
            except (OSError,ValueError,StopIteration) as exc:self.send(409,{'error':str(exc)})
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=2000000:raise ValueError('Invalid async payload size')
                body=self.rfile.read(length)
                match=re.fullmatch(r'/async/journal/([^/]+)/([^/]+)',self.path)
                if match:
                    result=learner.inbox.receive(match[1],match[2],body,self.headers.get('X-Content-SHA256',''))
                elif self.path in ('/async/episode/end','/async/episode/receipt'):
                    identity=json.loads(body)['episode_uuid']
                    result=learner.inbox.admit(identity) if self.path.endswith('/end') else learner.inbox.receipt(identity)
                    if self.path.endswith('/end'):learner.wake.set()
                else:return self.send(404,{'error':'not_found'})
                self.send(200,result)
            except (OSError,ValueError,KeyError,TypeError) as exc:self.send(409,{'error':str(exc)})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def serve(learner,port):
    server=make_server(learner,port)
    print('EXPO bounded async Learner ready; admission and optimizer independent',flush=True)
    try:server.serve_forever()
    finally:learner.stop.set();server.server_close()
