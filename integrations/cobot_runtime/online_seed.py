"""Fork a registered full RLT training checkpoint without modifying its seed."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import pickle
from uuid import uuid4
import hashlib

import numpy as np

import yaml


def _same_tree(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right))
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def _journal_boundary(path):
    """Count a stable trusted Replay journal without constructing a buffer.

    Preparing a branch during an append is refused, rather than guessing an
    adds_total anchor. This does not alter or copy the shared Replay.
    """
    path = Path(path)
    before = path.stat()
    rows = 0
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while stream.tell() < before.st_size:
            try:
                row = pickle.load(stream)
            except (EOFError, pickle.UnpicklingError) as error:
                raise ValueError('online_seed_replay_incomplete') from error
            if not isinstance(row, dict) or 'episode_id' not in row:
                raise ValueError('online_seed_replay_invalid')
            rows += 1
        if stream.tell() != before.st_size:
            raise ValueError('online_seed_replay_changed')
        stream.seek(0)
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('online_seed_replay_changed')
    return dict(adds_total=rows, bytes=before.st_size, sha256=digest.hexdigest())


def prepare(seed, destination, config_source, config_target, run_root, *, replay_budget_policy='new_arrivals',
            publication_policy='automatic'):
    seed, destination = Path(seed).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('online_seed_destination_already_exists')
    sources = {
        'actor_snapshot/actor_snapshot.pkl': seed/'actor_snapshot/actor_snapshot.pkl',
        'checkpoints/latest.pkl': seed/'checkpoints/latest.pkl',
        'action_norm_stats.json': seed/'action_norm_stats.json',
    }
    if any(not path.is_file() or path.is_symlink() for path in sources.values()):
        raise ValueError('online_seed_requires_full_training_checkpoint')
    # This is the registered local seed, also deserialized by the upstream
    # learner. Validate resume identity before creating any new branch assets.
    with sources['actor_snapshot/actor_snapshot.pkl'].open('rb') as stream:
        actor = pickle.load(stream)
    with sources['checkpoints/latest.pkl'].open('rb') as stream:
        checkpoint = pickle.load(stream)
    state = checkpoint['state']
    required = {'actor_params', 'target_actor_params', 'critic_params',
                'target_critic_params', 'actor_opt_state', 'critic_opt_state',
                'rng', 'global_step', 'actor_version'}
    if (not required.issubset(state) or int(state['global_step']) != 5000
            or int(actor.get('global_step', state['global_step'])) != 5000
            or int(state['actor_version']) != int(actor.get('version', -1))
            or not _same_tree(state['actor_params'], actor.get('actor_params'))
            or checkpoint.get('progress', {}).get('warmup_ready_adds_total') is None):
        raise ValueError('online_seed_training_identity_mismatch')
    config = yaml.safe_load(Path(config_source).read_text())
    if replay_budget_policy not in ('new_arrivals', 'inherit'):
        raise ValueError('online_seed_replay_budget_policy_invalid')
    if publication_policy not in ('automatic', 'staged'):
        raise ValueError('online_seed_publication_policy_invalid')
    runtime = config['runtime']
    config['experiment']['rl']['action_norm_stats_path'] = str(destination/'action_norm_stats.json')
    config['experiment']['rl']['freeze_after_warmup'] = False
    snapshot = str(destination/'actor_snapshot/actor_snapshot.pkl')
    runtime['actor_service']['snapshot_path'] = snapshot
    runtime['learner_service']['actor_snapshot_path'] = (
        str(destination/'pending_actor/actor_snapshot.pkl') if publication_policy == 'staged' else snapshot)
    runtime['learner_service']['checkpoint_dir'] = str(destination/'checkpoints')
    # The existing Replay is referenced, never rewritten or copied here.
    journal = Path(runtime['replay']['journal_path'])
    if not journal.is_absolute():
        journal = (Path(config_source).parent/journal).resolve()
    runtime['replay']['journal_path'] = str(journal)
    boundary = _journal_boundary(journal)
    source_anchor = int(checkpoint['progress']['warmup_ready_adds_total'])
    if replay_budget_policy == 'new_arrivals':
        # The branch starts at 5000. Existing shared data remain sampleable but
        # earn no new update budget merely because the seed was created earlier.
        if config['experiment']['rl'].get('warmup_post_collect_updates') != 5000:
            raise ValueError('online_seed_new_arrivals_requires_warmup5000')
        checkpoint['progress'] = dict(checkpoint['progress'], warmup_ready_adds_total=boundary['adds_total'])
    branch_anchor = int(checkpoint['progress']['warmup_ready_adds_total'])
    runtime['monitoring']['wandb_dir'] = str(Path(run_root)/'online/wandb')
    runtime['env_driver']['actor_deterministic'] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.seed-', dir=str(destination.parent)))
    try:
        for relative, source in sources.items():
            target = staging/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        # Only branch budget metadata changes; every training-state leaf stays
        # identical to the seed, which is never a write target.
        with (staging/'checkpoints/latest.pkl').open('wb') as stream:
            pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
        (staging/'seed.json').write_text(json.dumps({
            'model_id': 'plug-v3-warmup-5k', 'source': str(seed),
            'source_step': 5000, 'training_method': 'original',
            'run_root': str(Path(run_root).resolve()),
            'config_target': str(Path(config_target).resolve()),
            'replay_budget_policy': replay_budget_policy,
            'publication_policy': publication_policy,
            'served_actor_snapshot': snapshot,
            'candidate_actor_snapshot': runtime['learner_service']['actor_snapshot_path'],
            'seed_warmup_ready_adds_total': source_anchor,
            'branch_warmup_ready_adds_total': branch_anchor,
            'replay_boundary': boundary,
            'state_unchanged_from_seed': True,
            'actor_step_source': 'snapshot_metadata' if 'global_step' in actor else 'parameter_matched_full_checkpoint',
        }, indent=2)+'\n')
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    target = Path(config_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    os.replace(temporary, target)
    return destination


def latest_branch(model_root):
    """Only the newest user-created live branch is offered; older assets remain."""
    root = Path(model_root)/'online_from_5000'
    candidates = []
    for path in root.glob('*/seed.json'):
        try:
            if path.is_symlink() or path.parent.is_symlink():
                continue
            metadata = json.loads(path.read_text())
            if metadata.get('model_id') != 'plug-v3-warmup-5k' or metadata.get('source_step') != 5000:
                continue
            branch = path.parent
            if not (branch/'actor_snapshot/actor_snapshot.pkl').is_file():
                continue
            candidates.append((path.stat().st_mtime_ns, branch, metadata))
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        return None
    _, branch, metadata = max(candidates, key=lambda item: item[0])
    return dict(metadata, checkpoint=str(branch/'actor_snapshot/actor_snapshot.pkl'), branch_id=branch.name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    for name in ('seed', 'destination', 'config_source', 'config_target', 'run_root'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--replay-budget-policy', choices=('new_arrivals', 'inherit'), default='new_arrivals')
    parser.add_argument('--publication-policy', choices=('automatic', 'staged'), default='automatic')
    args = parser.parse_args()
    prepare(args.seed, args.destination, args.config_source, args.config_target, args.run_root,
            replay_budget_policy=args.replay_budget_policy, publication_policy=args.publication_policy)
