"""Episode-aware summaries of command/feedback evidence, without invented labels."""
import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--episode-coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text())
    coverage = json.loads(args.episode_coverage.read_text())
    complete = {(x['split'], x['episode']) for x in coverage['episodes']
                if x['fully_covered_complete_episode']}
    groups = collections.defaultdict(list)
    for row in data['steps']:
        groups[row['split'], row['episode']].append(row)
    episodes = []
    for (split, episode), rows in sorted(groups.items()):
        deltas = []
        for row in rows:
            candidates = [x for x in row['compatible_snapshots']
                          if x['feedback_matches_recorded_action'] and x['feedback_matches_next_state']]
            if candidates and row['all_exact_candidate_commands_equal']:
                deltas.append(np.abs(candidates[0]['command_minus_trace_action']))
        episodes.append({
            'split': split, 'episode': episode, 'audited_hil_steps': len(rows),
            'complete_episode_trace_coverage': (split, episode) in complete,
            'action_equals_next_feedback': sum(x['action_equals_next_state'] for x in rows),
            'same_command_snapshot_candidates': len(deltas),
            'ambiguous_commands': sum(x['exact_action_feedback_compatible_snapshots'] > 0
                and not x['all_exact_candidate_commands_equal'] for x in rows),
            'compatible_snapshot_mean_abs_command_minus_feedback_per_dim':
                np.mean(deltas, axis=0).tolist() if deltas else None,
            'exact_historical_command_actions_recovered': 0,
        })
    result = {
        'status': 'verified_feedback_action_semantics_command_identity_insufficient',
        'total_hil_steps': len(data['steps']), 'episodes': episodes,
        'action_equals_next_feedback': sum(x['action_equals_next_state'] for x in data['steps']),
        'steps_with_exact_action_feedback_compatible_snapshot': sum(
            x['exact_action_feedback_compatible_snapshots'] > 0 for x in data['steps']),
        'steps_with_same_command_snapshot_candidates': sum(x['same_command_snapshot_candidates'] for x in episodes),
        'historical_command_actions_recovered': 0,
        'training_targets_modified': False,
        'boundary': 'Command differences concern partial time-compatible raw snapshots, '
                    'not verified executed actions or optimal counterfactual targets. '
                    'No confidence interval is provided for missing command coverage. '
                    'Training and repeatedly used development Episodes are separate; no independent test.',
    }
    (args.output/'hil_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, split, title, color in zip(axes, ['train', 'development'],
            ['Warmup training', 'Reused development'], ['#276fbf', '#d78324']):
        selected = [e for e in episodes if e['split'] == split]
        x = np.arange(len(selected))
        totals = np.array([e['audited_hil_steps'] for e in selected])
        matched = np.array([e['same_command_snapshot_candidates'] for e in selected])
        ax.bar(x, totals, color='#dddddd', label='Audited HIL steps')
        ax.bar(x, matched, color=color, label='Same-command compatible snapshots')
        ax.set_xticks(x); ax.set_xticklabels([str(e['episode']) for e in selected], rotation=90)
        ax.set_ylabel('Step count (not independent samples)'); ax.set_xlabel('Replay Episode ID')
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle('Historical command identity remains unverified; independent test absent')
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(args.output/'hil_command_coverage.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for split, color, label in [('train', '#276fbf', 'Training'), ('development', '#d78324', 'Reused development')]:
        selected = [e for e in episodes if e['split'] == split and
                    e['compatible_snapshot_mean_abs_command_minus_feedback_per_dim'] is not None]
        for e in selected:
            values = e['compatible_snapshot_mean_abs_command_minus_feedback_per_dim']
            axes[0].plot(np.arange(1, 7), values[:6], 'o-', alpha=.35, color=color)
            axes[1].scatter(0 if split == 'train' else 1, values[6]*1000,
                            color=color, alpha=.6)
        axes[0].plot([], [], 'o-', color=color, label=label)
    axes[0].set_xlabel('Joint'); axes[0].set_ylabel('Mean absolute command-feedback difference (rad)')
    axes[0].legend(fontsize=9)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Training', 'Reused development'])
    axes[1].set_ylabel('Gripper command-feedback difference (mm)')
    fig.suptitle('Each line/dot is one Episode, using its available compatible snapshots')
    fig.text(.5, .01, 'Partial snapshot coverage only; no recovered command identity, optimal-action or autonomy claim.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .92)); fig.savefig(args.output/'hil_command_feedback_gap.png', dpi=150); plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
