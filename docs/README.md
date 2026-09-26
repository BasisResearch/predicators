# Documentation

Start with the [continual protocol overview](protocol/overview.md), the [simulator subclass interface](models/subclass-simulator.md), and the [current uncertainty implementation](uncertainty/explained.md).
For experiment results, use the [results index](uncertainty-results/INDEX.md).

## Topics

| Topic | Contents |
|---|---|
| [Protocol and interaction](protocol/README.md) | Evaluation rules, agent interaction, and the original protocol design. |
| [Learned models](models/README.md) | Simulator subclasses and system identification. |
| [Simulator probe API](models/sim-api.md) | `sim.run` modes, uncertainty checks, evaluator acceptance, and model diagnostics. |
| [Uncertainty](uncertainty/README.md) | Implementation audit, observation noise, and the proposed simplification. |
| [Environments](envs/README.md) | Domain descriptions, illustrations, videos, and the PyBullet development guide. |
| [Results](uncertainty-results/INDEX.md) | Noisy sweep, completed comparisons, diagnostics, and analysis outputs. |
| [Slides](slides/README.md) | Domain presentations, method illustrations, and dated research updates. |
| [Prompt review](prompt-review/README.md) | September 9, 2026 review copies and validation notes. |
| [Archived notes](archive/README.md) | Earlier proposals and experiment designs. |

## Reading the status of a document

Interface references describe the implementation at their recorded revision.
Design notes can contain both implemented and proposed work; read their status sections before treating them as current behavior.
The uncertainty simplification is a proposal, and structural uncertainty is an archived proposal rather than a capability established by the current experiments.
Dated result reports describe their selected runs, not necessarily the code in this checkout.

## Keeping the docs organized

Put authored guides beside their topic and link them from its index.
Keep each generated report with its generator, input snapshot, selection manifest, and exported figures.
Edit the generator to change a generated report.
Preserve source scorecards and dated analysis folders so existing result selections remain reproducible.
Use the results index to distinguish completed snapshots, actively maintained reports, and visual checks.
Keep slide exports with their assets and source scripts.
Local logs may be absent from a fresh checkout; tracked report snapshots are the portable record.

## Moved documents

The September 11, 2026 cleanup grouped the formerly top-level guides by topic.
Historical slide exports and external notes may still quote the old paths.
This table maps those paths to their current locations.

| Previous path within `docs/` | Current document |
|---|---|
| `continual-protocol-overview.md` | [protocol/overview.md](protocol/overview.md) |
| `continual-protocol.md` | [protocol/design.md](protocol/design.md) |
| `interaction-driver.md` | [protocol/interaction-driver.md](protocol/interaction-driver.md) |
| `subclass-simulator.md` | [models/subclass-simulator.md](models/subclass-simulator.md) |
| `continual-uncertainty.md` | [uncertainty/design.md](uncertainty/design.md) |
| `uncertainty-explained.md` | [uncertainty/explained.md](uncertainty/explained.md) |
| `uncertainty-simplification-proposal.md` | [uncertainty/simplification-proposal.md](uncertainty/simplification-proposal.md) |
| `structural-uncertainty.md` | [archive/structural-uncertainty.md](archive/structural-uncertainty.md) |
| `continual-minimal-knowledge.md` | [archive/continual-minimal-knowledge.md](archive/continual-minimal-knowledge.md) |
| `domino-openloop-continuous-perception.md` | [envs/domino/continuous-perception.md](envs/domino/continuous-perception.md) |
| `pybullet_env_guide.md` | [envs/pybullet-guide.md](envs/pybullet-guide.md) |
