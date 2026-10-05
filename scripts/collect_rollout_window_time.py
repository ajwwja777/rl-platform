"""Read-only Cobot trace time collector, expected signatures supplied in stdin.

No model imports, services, writes or interpolation of missing observations.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np

def signature(actions,sources,state,next_state):
    return hashlib.sha256(np.asarray(actions,dtype=np.float16).tobytes()+np.asarray(sources,dtype=np.uint8).tobytes()+np.asarray(state,dtype=np.float32).tobytes()+np.asarray(next_state,dtype=np.float32).tobytes()).hexdigest()

def main():
    expected=json.load(sys.stdin)
    root=Path('/media/agilex/Getea1/jiaan/data/datasets/plug_insertion/derived/rl-platform/rlt/traces')
    output={'matches':[],'files':[],'expected_records':sum(len(v) for v in expected.values()),
        'boundary':'Recorded completed-step timestamps, not hardware command publication stamps. Exact full C10 action/source and FP32 start/next state join. No image-anchor identity established.'}
    for path in sorted(root.rglob('*.jsonl')):
        content=path.read_bytes();rows=[json.loads(line) for line in content.splitlines() if line.strip()]
        rows=[r for r in rows if np.asarray(r.get('action',[])).shape==(7,) and 'observation' in r and 'next_observation' in r]
        matched=0
        for start in range(max(0,len(rows)-9)):
            window=rows[start:start+10]
            if any(r.get('done',False) for r in window[:-1]):continue
            key=signature([r['action'] for r in window],[r['source'] for r in window],window[0]['observation']['state'],window[-1]['next_observation']['state'])
            if key not in expected:continue
            times=np.asarray([r['timestamp'] for r in window],float);gaps=np.diff(times)
            for identity in expected[key]:
                output['matches'].append(dict(identity,trace=str(path),trace_start_row=start,prompt=window[0]['observation'].get('prompt'),
                    signature_sha256=key,first_timestamp=float(times[0]),last_timestamp=float(times[-1]),
                    recorded_first_to_last_seconds=float(times[-1]-times[0]),recorded_gap_seconds=gaps.tolist(),
                    trace_sources=[r['source'] for r in window],nonpositive_gaps=int((gaps<=0).sum())))
                matched+=1
        if matched:output['files'].append({'path':str(path),'sha256':hashlib.sha256(content).hexdigest(),'matches':matched,'steps':len(rows)})
    print(json.dumps(output,allow_nan=False))

if __name__=='__main__':main()
