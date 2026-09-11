# Slides

[Documentation index](../README.md)

These decks describe the code, domains, and experiments at the time they were created.
Use the [results index](../uncertainty-results/INDEX.md) for result selection and metric definitions.

## Domains

| Deck | Sources and context |
|---|---|
| [Balloons](balloons_domain_slides.html) ([PDF](balloons_domain_slides.pdf)) | [Generator](make_balloons_domain_slides.py), [domain notes](../envs/balloons/README.md), and [visual variants](../envs/balloons/visuals/README.md). |
| [Bridge](bridge_simple_task_slides.html) | [Source HTML](assets/bridge/bridge_simple_task_slides.src.html) and [domain assets](../envs/bridge/README.md). |
| [Busyboard](busyboard_domain_slides.html) | [Domain and rendering notes](../envs/busyboard/README.md). |
| [Domino task generation](domino_min_block_task_gen_slides.html) | [Deck builder](../envs/domino_min_block/build_deck.py) and companion figure scripts. |

## Methods and results

| Deck | Context |
|---|---|
| [Uncertainty results](uncertainty_results_slides.html) ([PDF](uncertainty_results_slides.pdf), [notes](uncertainty_results_slides.md)) | Earlier Domino/Boil noise comparison and six-feature description; [generator](make_uncertainty_results_slides.py). |
| [System identification](sysid_pipeline_slides.html) | [Figure sources](../sysid/README.md). |
| [Agent planning and learning](agent_planning_learning_slides.html) | Earlier agent workflow and run illustrations. |
| [Fan model-learning postmortem](fan_model_learning_postmortem_slides.html) | Historical failure analysis. |

## Weekly updates

- [July 30, 2026](weekly_sync_20260730_slides.html)
- [August 13, 2026](weekly_sync_20260813_slides.html)
- [August 20, 2026](weekly_sync_20260820_slides.html)
- [August 27, 2026](weekly_sync_20260827_slides.html)

Use the source script or source HTML when an export is generated.
[make_standalone.py](make_standalone.py) provides the standalone-export utility.
Keep existing export paths stable because decks, media, and external notes refer to them.
Historical path references can be looked up in the [document relocation table](../README.md#moved-documents).
Slide rendering checks are listed in the [analysis catalog](../uncertainty-results/analysis.md).
