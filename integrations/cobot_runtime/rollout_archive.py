"""Durable per-run Q telemetry, independent of model/robot lifecycle."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time


def connect(root):
    directory = Path(root)/'telemetry'
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(directory/'rollout-history.sqlite3'), timeout=2)
    db.execute('CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, started REAL, actor INTEGER, critic INTEGER, count INTEGER, payload TEXT NOT NULL)')
    return db


def save_snapshot(root, value):
    uid = value.get('episode_uuid')
    points = value.get('points') or []
    if not isinstance(uid, str) or not uid or not points:
        return
    with closing(connect(root)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT payload FROM episodes WHERE id=?', (uid,)).fetchone()
        previous = json.loads(old[0]) if old else {}
        merged = {(p.get('timestamp'), p.get('seconds')): p for p in previous.get('points', [])}
        merged.update({(p.get('timestamp'), p.get('seconds')): p for p in points})
        points = sorted(merged.values(), key=lambda p:p.get('seconds', 0))
        first = points[0]
        started = value.get('started_at') or previous.get('started_at') or (first.get('timestamp', time.time())-first.get('seconds', 0))
        payload = {**previous, **value, 'started_at': started, 'points': points}
        db.execute('INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?)',
                   (uid, started, payload.get('actor_version'), payload.get('critic_version'), len(points), json.dumps(payload, allow_nan=False)))


def capture_current(root):
    try:
        value = json.loads((Path(root)/'telemetry/rollout.json').read_text())
        save_snapshot(root, value)
    except (OSError, ValueError):
        pass


def history(root, episode=None):
    path = Path(root)/'telemetry/rollout-history.sqlite3'
    if not path.exists():
        return [], None
    db = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True, timeout=2)
    try:
        rows = db.execute('SELECT id,started,actor,critic,count FROM episodes ORDER BY started DESC,id').fetchall()
        listing = [dict(episode_uuid=r[0],started_at=r[1],actor_version=r[2],critic_version=r[3],count=r[4]) for r in rows]
        row = db.execute('SELECT payload FROM episodes WHERE id=?', (episode,)).fetchone() if episode else None
        return listing, json.loads(row[0]) if row else None
    finally:
        db.close()


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--follow',action='store_true')
    args=parser.parse_args();stamp=None
    while True:
        try:
            current=(Path(args.root)/'telemetry/rollout.json').stat().st_mtime_ns
            if current!=stamp:
                capture_current(args.root);stamp=current
        except (OSError,sqlite3.Error):pass
        if not args.follow:break
        time.sleep(.2)
