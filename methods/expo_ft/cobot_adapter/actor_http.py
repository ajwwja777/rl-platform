"""Resumable reads of immutable versioned Actor payloads."""
import re


def serve(handler,path,size):
    start=0
    requested=handler.headers.get('Range')
    if requested:
        match=re.fullmatch(r'bytes=([0-9]+)-',requested)
        if not match or int(match[1])>=size:
            handler.send_response(416);handler.send_header('Content-Length','0');handler.end_headers();return
        start=int(match[1])
    handler.send_response(206 if requested else 200)
    handler.send_header('Content-Length',str(size-start))
    handler.send_header('Accept-Ranges','bytes')
    if requested:handler.send_header('Content-Range','bytes %d-%d/%d'%(start,size-1,size))
    handler.end_headers()
    with path.open('rb') as stream:
        stream.seek(start)
        remaining=size-start
        while remaining:
            chunk=stream.read(min(1024*1024,remaining))
            if not chunk:raise IOError('Actor payload changed during transfer')
            handler.wfile.write(chunk);remaining-=len(chunk)
