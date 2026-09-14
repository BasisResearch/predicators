"""Capture verified continual cohorts and render the paper's Figure 4.

Capture once with --capture --paper-root PATH; replot the archived
--snapshot without reading live experiment directories. Costs use
successful runs only.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use('Agg')
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

# pylint: enable=wrong-import-position

ROOT = Path(__file__).resolve().parent.parent.parent
DOMAINS = ['Boil', 'Domino', 'Fan', 'Bridge', 'Balloons']
ARMS = [
    'MB', 'MF', 'agent_continual_program_world_model',
    'agent_continual_oracle_dynamics', 'agent_continual_oracle_scene',
    'agent_continual_zero_shot', 'agent_continual_no_fitting',
    'agent_continual_no_uncertainty'
]
LABELS = [
    'EMPIRIC', 'Direct agent', 'Standalone sim.', 'Oracle dynamics',
    'Oracle scene', 'Zero-shot model', 'No harness fitting',
    'No explicit uncert.'
]
COLORS = [
    '#087f8c', '#bd5929', '#7467a6', '#397957', '#6b9483', '#5588ad',
    '#b19658', '#88929d'
]


def capture(paper: Path, target: Path) -> None:
    """Verify every selected final scorecard before freezing plot inputs."""
    sys.path.insert(0, str(paper / 'scripts'))
    spec = importlib.util.spec_from_file_location(
        'paper_artifacts', paper / 'scripts/build_artifacts.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, _ = module.verified_reported_rows()
    reports = [
        ROOT / 'docs/comparisons/continual-results.json',
        ROOT / 'docs/comparisons/bridge-three-span-results.json'
    ]
    sources: List[Dict[str, str]] = []
    for index, report in enumerate(reports):
        document = json.loads(report.read_text())
        for row in document['rows']:
            if (row['domain'] == 'Bridge') != (index == 1):
                continue
            assert row['finished'] and row['approach'] in ARMS
            path = Path(row['scorecard'])
            raw = path.read_bytes()
            card = json.loads(raw)
            totals = card['totals']
            assert card['finished_at'] and card['end_reason'] in {
                'all_levels_won', 'level_lost', 'level_not_won', 'agent_ended',
                'step_cap', 'wall_clock_cap'
            }
            assert card['seed'] == row['seed'] and card['arm'] == row[
                'approach']
            assert document['source_commit'].startswith(card['git_sha'])
            assert totals['total_steps'] == row['steps'] == sum(
                l['steps'] for l in card['levels'])
            assert totals['total_resets'] == row['resets'] == sum(
                l['resets'] for l in card['levels'])
            assert totals['levels_completed'] == row['wins'] == sum(
                l['won'] for l in card['levels'])
            assert totals['levels_total'] == row['levels']
            rows.append(
                dict(domain=row['domain'],
                     arm=row['approach'],
                     seed=row['seed'],
                     won=row['wins'],
                     levels=row['levels'],
                     steps=row['steps'],
                     resets=row['resets'],
                     source=str(path),
                     sha256=hashlib.sha256(raw).hexdigest(),
                     git_sha=card['git_sha']))
        sources.append({
            'path':
            str(report),
            'sha256':
            hashlib.sha256(report.read_bytes()).hexdigest()
        })
    assert len(rows) == 115
    assert len({(r['domain'], r['arm'], r['seed']) for r in rows}) == 115
    for domain in DOMAINS:
        for arm in ARMS:
            assert sum(r['domain'] == domain and r['arm'] == arm
                       for r in rows) == (2 if arm == 'MF' else 3)
    payload = {
        'generated_by':
        'scripts/plotting/plot_continual_comparisons.py',
        'policy':
        ('Preserve paper MB (3) and historical MF (2); add six comparison '
         'arms (3 each). Bridge uses three-span integrity-fixed cohort. '
         'Steps: whole-run successes only; solve: all levels; resets: all '
         'finished runs.'),
        'sources':
        sources,
        'records':
        rows,
        'caveats':
        [('Cohorts differ in observation handling and agent runtime; not '
          'a matched causal ablation.'),
         ('Only new Bridge standalone permits engine imports; other '
          'domains retain stricter historical prompt.'),
         ('Some historical standalone agents did not use a model; '
          'uncertainty arms include known custom uncertainty checks.'),
         ('Original no-fitting Bridge seed 0 retained; post-completion '
          'scheduler repeat excluded.'),
         'Balloons is original non-hatch with historical instantaneous goal.']
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + '\n')


def render(snapshot: Path, output: Path) -> None:
    """Means plus individual seed values, without small-sample CI claims."""
    data = json.loads(snapshot.read_text())
    rows = data['records']
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'svg.hashsalt': 'continual-comparisons'
    })
    fig, axes = plt.subplots(3, 5, figsize=(12.8, 8.7), sharey=True)
    summary: List[Dict[str, Any]] = []
    for col, domain in enumerate(DOMAINS):
        for metric, field in enumerate(['solve', 'steps', 'resets']):
            ax = axes[metric, col]
            for i, arm in enumerate(ARMS):
                group = [
                    r for r in rows
                    if r['domain'] == domain and r['arm'] == arm
                ]
                eligible = [r for r in group if r['won'] == r['levels']
                            ] if field == 'steps' else group
                vals = [
                    100 * r['won'] /
                    r['levels'] if field == 'solve' else r[field]
                    for r in eligible
                ]
                avg = float(np.mean(vals)) if vals else None
                summary.append(
                    dict(domain=domain,
                         arm=arm,
                         metric=field,
                         mean=avg,
                         n=len(vals),
                         values=vals))
                if i % 2 == 0:
                    ax.axhspan(i - .48, i + .48, color='#f3f6f7', zorder=0)
                if vals:
                    ax.barh(i, avg, height=.56, color=COLORS[i], zorder=2)
                    jitter = np.linspace(-.15, .15,
                                         len(vals)) if len(vals) > 1 else [0]
                    ax.scatter(vals,
                               i + np.asarray(jitter),
                               s=12,
                               facecolors='white',
                               edgecolors='#263c46',
                               linewidths=.6,
                               zorder=3,
                               clip_on=False)
                if field == 'steps':
                    ax.text(.98,
                            i,
                            f'n={len(vals)}' if vals else 'no success',
                            transform=ax.get_yaxis_transform(),
                            ha='right',
                            va='center',
                            fontsize=7,
                            color='#344a55',
                            bbox=dict(facecolor='white',
                                      edgecolor='none',
                                      pad=.6))
            ax.set_ylim(7.6, -.7)
            ax.set_yticks(range(8), LABELS, fontsize=10)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color('#aebec5')
            ax.tick_params(length=0, pad=4, labelsize=10)
            ax.set_axisbelow(True)
            ax.grid(axis='x', color='#dce4e7', lw=.6)
            if field == 'solve':
                ax.set_xlim(-3, 108)
                ax.set_xticks([0, 50, 100])
                ax.set_title(domain,
                             fontsize=12,
                             fontweight='bold',
                             color='#203744',
                             pad=12)
                ax.set_xlabel('Levels solved (%)', fontsize=11)
            elif field == 'steps':
                maximum = max(
                    (r['steps'] for r in rows
                     if r['domain'] == domain and r['won'] == r['levels']),
                    default=1)
                ax.set_xlim(0, maximum * 1.30)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f'{x/1000:g}k'
                                  if x >= 1000 else f'{x:g}'))
                ax.set_xlabel('Successful-run steps', fontsize=11)
            else:
                ax.set_xlim(left=-.05, right=max(1, ax.get_xlim()[1]))
                ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
                ax.set_xlabel('Resets (all runs)', fontsize=11)
    fig.subplots_adjust(left=.095,
                        right=.99,
                        top=.94,
                        bottom=.145,
                        wspace=.23,
                        hspace=.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(output.with_suffix('.' + extension),
                    dpi=180,
                    bbox_inches='tight',
                    pad_inches=.06)
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip()
                             for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    output.with_name(output.name + '-summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


def main() -> None:
    """Optionally capture a snapshot, then render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root',
                        type=Path,
                        default=ROOT.parent / 'sim-predicator-paper')
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if args.capture:
        capture(args.paper_root, args.snapshot)
    render(args.snapshot, args.output)


if __name__ == '__main__':
    main()
