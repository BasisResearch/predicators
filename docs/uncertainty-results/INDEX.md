# Experiment results

[Documentation index](../README.md)

Start with the [five-domain noisy sweep](noisy-sweep-table.md) for the maintained comparison across balloons, boil, bridge, domino, and fan.
Read its update timestamp, finished-seed counts, and [experiment plan](noisy-sweep-plan.md) before quoting results.
The older [README](README.md) in this directory is a generated Domino/Boil report, not the index for the entire collection.

## Main comparisons

| Collection | Scope | Data and reproduction |
|---|---|---|
| [Five-domain noisy sweep](noisy-sweep-table.md) | Three planned seeds per domain and arm, with explicit completion counts and reused Balloons MB seeds. | [Plan](noisy-sweep-plan.md), [TSV](noisy-sweep-summary.tsv), [snapshot](noisy-sweep-snapshot.json), [generator](make_noisy_sweep_table.py). |
| [Completed five-domain noise comparison](five-domain-latest/README.md) | September 10 snapshot: three noiseless and two noisy seeds per domain and arm; historical cohorts with documented version and task differences. | [Plot](five-domain-latest/mb_vs_mf.pdf), [successful-run cost](five-domain-latest/successful_steps.pdf), [TSV](five-domain-latest/summary.tsv), [selection](five-domain-latest/selection.json), [snapshot](five-domain-latest/snapshot.json), [generator](five-domain-latest/make_plot.py). |
| [Completed noiseless five-domain comparison](../../logs/analysis/mb_vs_mf_five_domains_20260908/README.md) | September 8 archive: 30 selected runs, three seeds per domain and arm; source versions differ across some arms. | [Analysis catalog](analysis.md) with plot, per-seed data, manifest, and scorecard snapshots. |
| [Domino/Boil noise study](README.md) | Earlier noise levels and the combined uncertainty-feature configuration. | [TSV](summary.tsv), [snapshot](snapshot.json), [generator](make_figures.py). |
| [Bridge/Fan/Balloons validation](crossdomain-table.md) | Earlier cross-domain validation. | [Plan](crossdomain-plan.md), [TSV](crossdomain-summary.tsv), [snapshot](crossdomain-snapshot.json), [generator](make_crossdomain_table.py). |

The directory name `five-domain-latest` is historical: its saved selection is not automatically the newest three-seed noisy sweep.
Consult each report's timestamp and manifest rather than choosing a plot by filename.
Local analysis logs are not tracked in Git and may be unavailable in another checkout.

## Balloons and model-repair studies

| Study | Reports and supporting records |
|---|---|
| Original-task model repair | [Plan](model-repair-plan.md), [table](model-repair-table.md), [TSV](model-repair-summary.tsv), [snapshot](model-repair-snapshot.json). |
| Corrected task generation | [Plan and cancelled comparison](balloons-v2-plan.md), [table](balloons-v2-table.md), [TSV](balloons-v2-summary.tsv), [snapshot](balloons-v2-snapshot.json). |
| Original-task subclass pilot | [Pilot](balloons-original-subclass-pilot.md) and [subclass follow-up](balloons-subclass-followup.md). |
| Hatch prototype | [V4 pilot](balloons-hatch-v4-pilot.md), [oracle replay](hatch-oracle-replay.md), and [seed 5 diagnosis](balloons-hatch-seed5-diagnosis.md). |
| Original noisy failure | [Failure analysis](balloons-failure-analysis.md). |

The original chute tasks, corrected task generation, and hatch prototypes are separate experiment settings.
Do not pool their scores without an explicit selection and justification.
The [model-repair table generator](make_model_repair_table.py) supports the model-repair and corrected-task reports.
The [hatch diagnostic generator](plot_hatch_seed5_diagnostic.py) produces the figures in the seed 5 diagnosis.

## Reading and maintaining results

A level solve rate and a whole-run solve rate measure different outcomes.
Step means may include all runs or only runs that won every level; the report must specify which and give the contributing seed count.
An unfinished run is not a completed agent failure, and a repeated execution of one seed is not an additional seed.
Use the report's selection and duplicate-execution audit rather than selecting favorable runs.
Comparisons across code versions or task distributions do not isolate the effect of one uncertainty feature.

Keep generated tables, snapshots, plots, and their scripts together at their existing paths.
The sweep watcher writes to these paths, so moving its outputs would disrupt maintenance.
Edit generators rather than generated reports, and use each report's documented command when regeneration is needed.
This index does not refresh or change any result selection.

For browser screenshots and slide-rendering checks, see the [analysis catalog](analysis.md).
For the implementation behind the uncertainty features, see the [uncertainty explanation](../uncertainty/explained.md).
