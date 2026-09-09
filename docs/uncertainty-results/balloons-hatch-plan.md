# Noisy balloons hatch prototype

This separate prototype tests whether a learned simulator helps choose balloon releases when success depends on the payload's orientation and contacts during ascent.
The original balloons results and the cancelled corrected-generator comparison remain separate.

## Task and observations

A 20 x 7 x 3.6 cm rectangular payload starts below a horizontal hatch at 0.57 m.
The opening is 17 cm wide and its centre is displaced 1.2 cm from the payload's initial centre.
The robot uses the existing Release and Wait skills.
The payload must reach the target band above the hatch without a balloon bursting at the ceiling.
Lift laws, mass classes, drag and the evaluator remain those of balloons.
The scene is selected with `balloons_scene=hatch`; the default chute scene is unchanged.

Both arms receive the same task geometry description and the same noisy observation schema.
Payload and balloon roll, pitch and yaw are included because orientation is necessary for reconstructing the task dynamics.
Position noise is 0.01 m, orientation noise is 0.02 rad, and scalar noise is zero.
No noiseless agent controls are planned.

## Validation before agent launch

Generated training levels need an executable winning reference.
Test levels additionally need a verified contact-blocked sequence: sustained off-goal rest with vertical obstacle load, plus a winning replay of the identical release order with obstacle collisions disabled.
A timeout remains unresolved and never establishes a jam.
For example, some red/gold sequences drift horizontally below the hatch even after 2,000 extra steps; those are unresolved, although their wall-free replays win.
The generator requires a witnessed equal-count release sequence that causes a verified jam.
Reversing that sequence may win; the generator does not classify the whole subset as losing.
This excludes the simple fewest-balloons distinction seen in the first, lower-hatch development variant.
A unique solution is not required or claimed.

The intended subclass simulator interface must reproduce the environment under actual Release actions at ground-truth parameters before an agent experiment is launched.
This is an expressibility and integration check, not evidence that an agent can learn those parameters.
The normal MB subclass contract has been consolidated in the separate `simulator-subclass-unification` branch.
The hatch pilot must include that contract and the cleaned balloons visible base before launch.
The legacy rule-interface discrepancy below is tracked separately and is not a prerequisite for a subclass-based pilot.
The lower-hatch development variant passed its corrected parity test in compute job 22378625.
The final 0.57 m configuration failed its successful-passage replay assertion in job 22379274: maximum box-position error was 0.0736 m against the 0.005 m threshold.
The blocked-sequence check passed, but this does not establish simulator fidelity.
The diagnostic in job 22379562 also found a 0.353 m maximum transient discrepancy when replaying the primitive actions in a fresh real environment; the cause remains under investigation.
Copying the action arrays did not remove the discrepancy (job 22379758), and tracing state resets found no unexpected reset during the original trajectory (job 22379768).
Diagnostic job 22379858 enabled deterministic contact ordering: fresh real replay then matched exactly, and the subclass model differed by at most 0.00000407 m.
The rule model still differed by 0.349 m, so deterministic contact ordering alone does not resolve its integration failure.
Deterministic contact ordering is now enabled in the opt-in hatch scene.
The final replay test compares original continuous Release trajectories against fresh real and native subclass simulators at configured ground-truth constants.
Job `22388885` passed all five hatch tests, changed-file lint and whole-repository mypy on 882 source files.
The old fixed witness becomes blocked under the deterministic setting; its former success claim does not carry over.
A bounded audit of all 24 four-color permutations found several valid order-dependent contact challenges, but none with a winning reference and a decoy that jams in both orders.
The final fixture uses colors red, blue, gold and green: blue/green succeeds in either order, red then gold jams, and gold then red succeeds.
The jam is verified by a winning wall-free replay of the identical red-then-gold sequence.
Generated-task audits for seeds 0, 4 and 5 are job `22389017`, separate from agent experiments.
The first parity test failed because its setup left a plain state without robot joint data; correcting the test setup resolved that failure.

The development seed-0 generator check passed two training levels and one test level in compute job 22377725.
Full mypy on the four implementation files' tree passed for 881 source files in job 22377944.
Further task audits, regression tests and the noisy continual oracle are required before freezing the experiment source.

## Proposed pilot

The user requested cancelling held experiments until the domains and agent are finalized.
All 11 held agent seeds and their two collectors were cancelled; their configurations remain available.
The pilot below has not been launched.

Configuration: `scripts/configs/predicatorv3/protocol_continual_balloons_hatch_v3.yaml`.
Use paired seeds 4 and 5 with two training levels and one test level.
Compare MB with all six uncertainty features and the advisory model-repair workflow against the unchanged MF baseline.
Prioritize MB on `mit_preemptable` and preserve the frozen runtime once jobs are launched.
The task audit may examine oracle outcomes on the paired seeds, but agent outcomes must not be used to select the task distribution.

Report per-seed levels won, real steps and resets.
Average steps only over whole-run successful seeds, with the qualifying count.
Keep simulator computation separate from real environment steps.
Interrupted and unstarted jobs are excluded from agent success and failure counts.
No MB advantage is established by mechanical or oracle validation.

## Agent failure tracked separately

The original repaired MB seed 0 retained the same no-op residual in all three saved simulator versions and never explicitly invoked the canonical fit before its level-end fallback.
Its test actions were green, blue, red and gold; the last release caused a ceiling burst.
An action replay reproduced that loss within 0.011 mm maximum box-height error in job 22377943.
Changing the task generator does not resolve this failure to construct a dynamics model.
The user is open to a domain-neutral agent extension, but none is included in this prototype yet.
