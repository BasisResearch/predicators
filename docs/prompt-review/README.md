# Continual prompt review, 2026-09-09

The reviewed run is `balloons-agent_continual_hatch_v4_mb/seed4/run_20260909_145816`.
Its recorded system prompt mixes workflow, protocol rules, API details, and repeated warnings.
The cleanup was developed against its simulator-subclass worktree at `86391e7ac` and integrated into `/home/ycliang/predicators` on `bridge-learning`.
The existing native simulator implementation, shared subclass contract, and visible-base fix were brought into the primary checkout as prerequisites (`d4d4149f3`, `bab3f0819`, `675480efb`).
The integration preserves the newer level-player and conversation-round names.
Running experiment worktrees and historical logs were not edited.

## Review copies

- [Revised system prompt](balloons-system-after.md), rendered with the run's noise settings and supplied physical-parameter menu.
- [Revised first round](balloons-first-round-after.md), rendered with the recorded task, observation, skill signatures, and object vocabulary.
- Reproducible domain-neutral variants live in `tests/agent_sdk/prompt_goldens/continual_*.md`.

The review copies are reconstructed previews, not prompts delivered to the running experiment.
The system preview includes the original sandbox tool appendix for an equivalent word-count comparison.
The first-round preview retains the full scene and belief data.

| Text | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| System, including sandbox appendix | 5,000 words | 3,177 words | 36.5% |
| First round | 817 words | 591 words | 27.7% |

## Organization

The system prompt presents run rules, observation semantics, and a short decision workflow before tools and implementation details.
The workbench API is grouped by purpose, followed by the simulator and predicate contracts.
The query carries current task data, model status, prior records, and the available vocabulary.
The sandbox document owns filesystem and Python mechanics.
Each full query shows its goal and budget once; continuations retain the goal in the observation.
Ordinary tool results still include their goal and budget fields.

## Corrections beyond shortening

- The ordinary simulator contract consistently uses `RESIDUAL_ENV` and `AGENT_PARAM_SPECS`.
- Supplied environment predicates are distinguished from invented predicates without claiming either that every environment atom is visible or that no supplied predicates can exist.
- Hidden-memory classifiers are described as model-dependent estimates, removing the blanket statement that they are always false on real observations.
- Simulator successes and failures are conditional predictions, removing the claim that an unfitted simulator can conclusively veto a real plan.
- A disagreement in magnitude no longer proves that a parameter is the only possible cause; missing mechanisms remain possible.
- Rejected fits call for diagnosis rather than proving model error or forbidding all actions.
- Full-plan rehearsal, action labels, evaluator notes, contacts, uncertainty checks, explicit fitting, and validation against rejected recordings are retained.
- Resume instructions direct the agent to recorded progress rather than claiming an interrupted tool call necessarily consumed no steps.
- Sandbox data, log naming, and image instructions now cover continual play rather than only the old phased pipeline.
- Model-status text reports artifacts, fit status, and data counts; the system prompt owns the instructions for acting on that status.

The prompt cleanup itself leaves tools, evaluators, budgets, model fitting, and action execution unchanged; the prerequisite commits supply the runtime already used by the reviewed experiment.
These are prompt and presentation changes; their effect on agent performance has not been measured in a new sweep.

## Validation in the isolated worktree

Focused prompt and continual integration tests passed across the corrected reruns, including scripted MB/MF interaction, all four round kinds, noisy observations, model-repair gating, and minimal agents.
The checks preserve tool access and verify that full and continuation queries carry one ledger and context block.
The rendered prompt tests passed after regeneration and review.
Mypy passed on all 886 source files; lint passed on the nine changed Python files; pinned formatter checks and `git diff --check` passed.
Logs are under `/home/ycliang/predicators/logs/prompt_streamline_checks-*.out`, with final functional/type results in `22410650` and the final prompt-test lint correction in `22411050`.

## Main-worktree integration validation

The integrated code passed 95 focused tests, including native subclass fitting, model-state tracking, prompt rendering, and scripted continual agents.
Mypy passed on all 885 source files in the primary checkout.
Yapf, docformatter, and pinned isort checks passed for all 28 Python files touched by the prerequisites and cleanup.
Lint passed on the ten checked integration files after shortening an overlong module docstring from the earlier naming change.
The main integration log is `/home/ycliang/predicators/logs/prompt_main_integration_checks-22411364.out`; the final docstring lint recheck is `prompt_main_integration_checks-22411869.out` in the same directory.
The cleanup is committed as `a2d4739d1` on `bridge-learning`.
