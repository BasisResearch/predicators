# Analysis outputs

[Results index](INDEX.md) | [Documentation index](../README.md)

This catalog covers the folders present under `logs/analysis/` when inspected on September 11, 2026.
Those folders are local, ignored experiment and rendering artifacts; this catalog is kept with the versioned docs.
Their paths and contents are preserved so manifests, reports, and external references still identify the same files.

## Experiment comparison

[mb_vs_mf_five_domains_20260908](../../logs/analysis/mb_vs_mf_five_domains_20260908/) contains the completed noiseless five-domain comparison.
Its [status](../../logs/analysis/mb_vs_mf_five_domains_20260908/status.md) records all 30 selected runs finished on September 8, 2026.
The report documents source-version differences and uses all-run step totals, including unsuccessful runs.

| Artifact | Purpose |
|---|---|
| [README](../../logs/analysis/mb_vs_mf_five_domains_20260908/README.md) | Metric definitions, cohort decisions, and original watcher command. |
| [PDF](../../logs/analysis/mb_vs_mf_five_domains_20260908/mb_vs_mf.pdf), [PNG](../../logs/analysis/mb_vs_mf_five_domains_20260908/mb_vs_mf.png), [SVG](../../logs/analysis/mb_vs_mf_five_domains_20260908/mb_vs_mf.svg) | Final comparison plots. |
| [Per-seed CSV](../../logs/analysis/mb_vs_mf_five_domains_20260908/per_seed.csv) | Run metrics, source paths, and hashes. |
| [Summary CSV](../../logs/analysis/mb_vs_mf_five_domains_20260908/summary.csv) | Aggregates and sample standard deviations. |
| [Manifest](../../logs/analysis/mb_vs_mf_five_domains_20260908/manifest.json) | Selected run sources. |
| [Scorecard snapshots](../../logs/analysis/mb_vs_mf_five_domains_20260908/scorecard_snapshots.json) | Saved inputs to the comparison. |
| [Preview](../../logs/analysis/mb_vs_mf_five_domains_20260908/preview.png) | Earlier layout check; use the final plots for results. |

The original regeneration command depends on a local monitor script outside the repository.
The saved data and plots remain available here, but the command is not a self-contained reproduction recipe for a fresh checkout.
For the newer noisy sweep, use the [maintained report](noisy-sweep-table.md).

## Slide rendering checks

| Folder | Contents and context |
|---|---|
| [balloons_slides_20260908](../../logs/analysis/balloons_slides_20260908/) | Twelve page previews and render logs for the earlier [Balloons slide snapshot](../envs/balloons/sweep_20260907/README.md). |
| [balloons_slides_20260909](../../logs/analysis/balloons_slides_20260909/) | Seventeen page previews and browser checks for the later Balloons deck. |
| [September 9 browser checks](../../logs/analysis/balloons_slides_20260909/browser/) | Slide screenshots, video-slide checks at two viewport widths, and a [report](../../logs/analysis/balloons_slides_20260909/browser/report.json). |

These are rendering checks, not additional benchmark trials or statistical seeds.
Use the [slides index](../slides/README.md) for presentation exports and source scripts.
