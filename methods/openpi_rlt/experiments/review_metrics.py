"""Episode identities and actual-step returns for repeated Replay diagnostics."""
from collections import defaultdict
import numpy as np
from .guarded_credit import episode_credit

def episode_groups(rows):
    groups = defaultdict(list)
    seen = set()
    for i, r in enumerate(rows):
        ep = (int(r["collection_phase_id"]), int(r["episode_id"]))
        key = (*ep, int(r["step_id"]))
        if key in seen:
            raise ValueError("Duplicate source/episode/step identity; import independent sources first")
        seen.add(key)
        groups[ep].append(i)
    return {ep: np.array(sorted(ix, key=lambda i: int(rows[i]["step_id"]))) for ep, ix in groups.items()}

def review_metadata(rows, gamma):
    groups = episode_groups(rows)
    canonical = [dict(r, collection_phase=int(r["collection_phase_id"])) for r in rows]
    observed, valid = episode_credit(canonical, gamma)
    human = np.stack([np.isin(r["source_chunk"], [2, 3]) for r in rows])
    outcome = np.empty(len(rows), object)
    kind = np.empty(len(rows), object)
    for ep, ix in groups.items():
        if not valid[ix].all():
            outcome[ix] = "unknown"
        else:
            success = any(bool(rows[i]["success"]) for i in ix)
            outcome[ix] = ("assisted_success" if human[ix].any() else "autonomous_success") if success else "failure"
        if ep[1] < 0:
            kind[ix] = "expert"
        elif not human[ix].any():
            kind[ix] = outcome[ix]
        else:
            first = min(int(rows[i]["step_id"]) + int(np.flatnonzero(human[i])[0]) for i in ix if human[i].any())
            for i in ix:
                kind[i] = "human_window" if human[i].any() else ("pre_hil" if int(rows[i]["step_id"]) < first else "post_hil")
    return groups, observed, valid, human, outcome, kind

def episode_mean(values, mask, groups):
    values, mask = np.asarray(values), np.asarray(mask, bool)
    selected = [values[ix[mask[ix]]].mean(axis=0) for ix in groups.values() if mask[ix].any()]
    return np.mean(selected, axis=0).tolist() if selected else None

def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    p, n = scores[labels == 1], scores[labels == 0]
    if not len(p) or not len(n):
        return None
    return float(np.mean((p[:, None] > n) + .5 * (p[:, None] == n)))
