"""Fork a registered full RLT training checkpoint without modifying its seed."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import pickle
from uuid import uuid4

import yaml


def prepare(seed, destination, config_source, config_target, run_root):
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
            or int(actor['global_step']) != 5000
            or int(state['actor_version']) != int(actor['version'])
            or checkpoint.get('progress', {}).get('warmup_ready_adds_total') is None):
        raise ValueError('online_seed_training_identity_mismatch')
    del actor, checkpoint, state
    config = yaml.safe_load(Path(config_source).read_text())
    runtime = config['runtime']
    config['experiment']['rl']['action_norm_stats_path'] = str(destination/'action_norm_stats.json')
    config['experiment']['rl']['freeze_after_warmup'] = False
    snapshot = str(destination/'actor_snapshot/actor_snapshot.pkl')
    runtime['actor_service']['snapshot_path'] = snapshot
    runtime['learner_service']['actor_snapshot_path'] = snapshot
    runtime['learner_service']['checkpoint_dir'] = str(destination/'checkpoints')
    # The existing Replay is referenced, never rewritten or copied here.
    journal = Path(runtime['replay']['journal_path'])
    if not journal.is_absolute():
        journal = (Path(config_source).parent/journal).resolve()
    runtime['replay']['journal_path'] = str(journal)
    runtime['monitoring']['wandb_dir'] = str(Path(run_root)/'online/wandb')
    runtime['env_driver']['actor_deterministic'] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.seed-', dir=str(destination.parent)))
    try:
        for relative, source in sources.items():
            target = staging/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (staging/'seed.json').write_text(json.dumps({
            'model_id': 'plug-v3-warmup-5k', 'source': str(seed),
            'source_step': 5000, 'training_method': 'original',
            'run_root': str(Path(run_root).resolve()),
            'config_target': str(Path(config_target).resolve()),
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
    args = parser.parse_args()
    prepare(args.seed, args.destination, args.config_source, args.config_target, args.run_root)
