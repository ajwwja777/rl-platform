#!/usr/bin/env python3
"""Export native JSONL metrics as a standalone HTML/SVG diagnostic panel.

No server, external JS, model loading, training or success-rate inference.
Missing historical metrics stay empty; no interpolation or smoothing.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

PANELS = [
    ("Replay transitions", "replay_size", False),
    ("Actor version", "actor_version", False),
    ("Q1 mean (recorded actions)", "q1_mean", False),
    ("Q1 batch maximum", "q1_max", False),
    ("Q1 positive-reward chunks", "q1_rewarded_chunk_mean", False),
    ("Q1 other chunks (NOT failed Episodes)", "q1_unrewarded_chunk_mean", False),
    ("Q2 mean (recorded actions)", "q2_mean", False),
    ("Q2 batch maximum", "q2_max", False),
    ("TD target mean (min-Q bootstrap)", "target_q_mean", False),
    ("TD target batch maximum", "td_target_max", False),
    ("Critic loss (sum of two MSEs)", "critic_loss", False),
    ("Actor Q1 (update steps only)", "actor_q", True),
    ("Weighted BC", "weighted_bc", True),
    ("Weighted Q1 reward", "weighted_q", True),
    ("Weighted delta penalty", "weighted_delta", True),
    ("Actual sampled HIL ratio", "sample_human_intervention_ratio", False),
]


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_rows(path, *, data=None):
    rows, incomplete = [], 0
    lines = (path.read_bytes() if data is None else data).splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if i != len(lines) - 1:
                raise ValueError(f"Malformed completed metric row {i + 1}")
            incomplete += 1  # A concurrently appended final row can be retried.
            continue
        if not finite_number(row.get("global_step")):
            raise ValueError(f"Missing finite global_step on row {i + 1}")
        if rows and row["global_step"] <= rows[-1]["global_step"]:
            raise ValueError("Non-increasing steps: separate resumed/other-run logs first")
        rows.append(row)
    return rows, incomplete


def series(rows, key, actor_only):
    return [(row["global_step"], row[key]) for row in rows
            if (not actor_only or row.get("did_actor_update") == 1)
            and finite_number(row.get(key))]


def svg(points, *, segments=None, missing_rows=0):
    if not points:
        return '<p class="missing">Not recorded / no eligible rows. No values inferred.</p>'
    xmin, xmax = points[0][0], points[-1][0]
    ymin, ymax = min(v for _, v in points), max(v for _, v in points)
    xd, yd = max(xmax - xmin, 1), max(ymax - ymin, 1e-12)
    # Keep bin extrema, not averages: short excursions remain visible.
    lines, displayed = [], 0
    for segment in segments if segments is not None else [points]:
        shown = segment
        if len(segment) > 800:
            shown = []
            stride = math.ceil(len(segment) / 200)
            for offset in range(0, len(segment), stride):
                block = segment[offset:offset + stride]
                chosen = {0, len(block)-1,
                          min(range(len(block)), key=lambda i: block[i][1]),
                          max(range(len(block)), key=lambda i: block[i][1])}
                shown.extend(block[i] for i in sorted(chosen))
        displayed += len(shown)
        coords = [(35 + (x-xmin)/xd*400, 160-(v-ymin)/yd*135) for x, v in shown]
        line = ' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)
        if len(coords) == 1:
            x,y = coords[0]
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#54b7dc"/>')
        else:
            lines.append(f'<polyline points="{line}" fill="none" stroke="#54b7dc" stroke-width="1.5"/>')
    return (f'<svg viewBox="0 0 460 200"><path d="M35 20V160H440" fill="none" stroke="#75899b"/>'
            + ''.join(lines) +
            f'<text x="2" y="25">{ymax:.4g}</text><text x="2" y="160">{ymin:.4g}</text>'
            f'<text x="35" y="183">{xmin:g}</text><text x="390" y="183">{xmax:g}</text>'
            f'<text x="180" y="198">Learner updates</text></svg>'
            f'<small>last={points[-1][1]:.6g}; recorded rows={len(points)}; missing eligible rows={missing_rows}; display points={displayed}</small>')


def render(rows, *, label, role, source_sha, incomplete=0):
    panels = []
    missing = []
    for title, key, actor_only in PANELS:
        points = series(rows, key, actor_only)
        segments, segment, missing_rows = [], [], 0
        for row in rows:
            if actor_only and row.get('did_actor_update') != 1:
                continue
            if finite_number(row.get(key)):
                segment.append((row['global_step'], row[key]))
            else:
                missing_rows += 1
                if segment:
                    segments.append(segment); segment = []
        if segment:
            segments.append(segment)
        if not points:
            missing.append(key)
        panels.append(f'<article><h3>{html.escape(title)}</h3>{svg(points, segments=segments, missing_rows=missing_rows)}</article>')
    page = ('<!doctype html><meta charset="utf-8"><title>RLT training diagnostics</title>'
            '<style>body{font:15px sans-serif;background:#0e1727;color:#dfebf6;margin:24px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}'
            'article{background:#18263b;padding:12px;border-radius:8px}svg{width:100%;fill:#b9c9d9;font:11px sans-serif}'
            'h3{font-size:14px}.missing{color:#e5b574}small{color:#aabbcf}</style>'
            f'<h1>{html.escape(label)}</h1><p>Data role: {html.escape(role)}. SHA256: {source_sha}</p>'
            '<p>Training diagnostics only. No autonomous success / assisted success / independent-test AUC inferred.'
            ' Rewarded chunks are NOT successful Episodes. No smoothing; extrema-preserving display reduction.'
            ' Actor values use actual update steps. Q1 drives Actor; min-Q bootstraps TD.</p>'
            f'<p>Complete metric rows: {len(rows)}; incomplete tail ignored: {incomplete}.</p>'
            '<div class="grid">' + ''.join(panels) + '</div>')
    return page, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--data-role', required=True, choices=['training', 'reused_development', 'independent_test'])
    args = parser.parse_args()
    data = args.input.read_bytes()
    rows, incomplete = load_rows(args.input, data=data)
    page, missing = render(rows, label=args.label, role=args.data_role,
                           source_sha=hashlib.sha256(data).hexdigest(), incomplete=incomplete)
    # The role names the log's cohort; it is not an acceptance claim.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps({
        'source': str(args.input), 'sha256': hashlib.sha256(data).hexdigest(),
        'rows': len(rows), 'incomplete_tail': incomplete, 'missing_metrics': missing,
        'data_role': args.data_role, 'success_rate_learning_auc': None,
        'boundary': 'Training-log metrics cannot establish autonomous task improvement.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
