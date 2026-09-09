# Hatch task certification and public Release

The seed 4 hatch oracle failed because the task generator certified a release controller with motion planning disabled, while the continual oracle used motion planning.
The resulting timing difference trapped the payload below the hatch after gold then blue.
The saved 279-action replay reproduces this failure; replaying those same actions without payload-panel collisions wins at step 81.
These are diagnostics of the earlier oracle execution, not additional agent seeds.

The regression reconstructs the failed task, including its realized initial robot pose, and executes the public Release options through `env.step()`.
Before the fix, the public controller failed while the validator reported a win in 66 actions.
The opposite order, blue then gold, is executable on this task.
The test covers motion planning both enabled and disabled.

Task certification now constructs Release through the same factory as the public option set.
It no longer retains a process-global option that can outlive a configuration change.
Candidate caching distinguishes motion-planning mode, planner seed and release parameters.
The generator still requires every tested immediate order of its reference subset to win, so the process oracle can choose any of those orders.
The physics, goal evaluator and release parameters are unchanged.
This certifies the configured immediate sequences, not arbitrary additional waits or custom skill parameters.

Because the correction can reject previously accepted tasks, hatch task-generation metadata advances to version 4 and chute metadata to version 3.
The metadata also records whether motion planning was enabled during validation.
Any regenerated oracle results must remain separate from the earlier hatch version 3 results.
The original failed task and scorecard remain unchanged; resampling a corrected task does not retroactively solve that task.

The original-balloons offline model synthesis is independent of this fix.
Its frozen source remains untouched, and it uses only 422 previously recorded training actions.
Its conversation log is `logs/balloons_followup_20260909/offline-learning-v2/agent/001_learn_20260909_114132.md` in the main checkout.
Detailed fitting progress appears in `logs/balloons_followup_20260909/offline-learning-22389016.out`.
This diagnostic has no continual scorecard and therefore does not appear in the continual viewer's run list.

Validation runs and their outcomes are recorded in the main checkout's `docs/uncertainty-results/hatch-oracle-replay.md`.
