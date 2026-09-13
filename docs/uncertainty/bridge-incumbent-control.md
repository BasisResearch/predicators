# Revised Bridge incumbent comparison control

This control uses the [verified revised subclass](bridge-subclass-inference.md) to prepare the incumbent arm of the Bridge estimator comparison.
It keeps the full incumbent fitter, including its objective, parameter-selection rules, interval handling and fit-evidence configuration.
It does not substitute the exact-rate component for the incumbent's fitting behavior.
The new probability-model arm remains unfinished and is not launched by this control.

## Frozen development split

The fitting data are actions 1-600 and observations 0-600 from the same 1,186-action Bridge training recording used in the program diagnostics.
The remaining 586 actions are reserved for causal forecast assessment.
This extends the earlier 124-action geometry diagnostic prefix so that fitting includes the first recorded bond at action 583, while the second bond at action 835 lies in the later segment.
The changed split is explicit; results from the two prefixes must not be treated as interchangeable.
The recording and candidate program have already been inspected during development, so the later segment is not an untouched final evaluation set.

The source program, subclass implementation, recording files, prefix length and preflight driver are frozen in the control plan.
The fitting data identity hashes the serialized prefix episode separately from the full recording identity.
The model's memory requires the incumbent preparation to retain the full 600-action prefix as one trajectory.
The driver checks every prepared observation and action against that prefix.
No rest-point reset of glue levels, dwell counters or bonds is introduced.

Both complete forecast repetitions start from the same original public noisy initial state and run all 1,186 recorded actions.
They receive no later observation corrections.
Assessment independently reconstructs the selected forecast through the manual native reset/pin/zero-velocity/step lifecycle, then computes feature errors and exact glue-reading mismatches only on actions 601-1186.
Angular feature errors use circular differences; these metrics are distinct from a probability-model likelihood.
Selected-point predictions are not the incumbent's full interval or planning ensemble.

## Compute pipeline

| Job | Purpose | Verified state on September 13 |
| --- | --- | --- |
| `22696941` | Check full-prefix preparation and helper/manual native parity at default dwell, dwell 20 and dwell 40. | Completed; all three pairs exact, 7,116 native actions. |
| `22697046` | Run the complete incumbent fitter, then two complete causal forecasts. | Running. |
| `22697048` | Independently reproduce the selected forecast and assess the 586-action suffix. | Depends on successful fitting. |

The preflight compares three pairs of full trajectories and checks that changing bond dwell changes predictions.
The fitting driver requires that certificate, preserves all six declared parameters, and records fitted versus actually applied values separately.
If the incumbent rejects every segment, it retains its declared default parameter behavior rather than treating the fit as successful evidence.
Infrastructure failures remain separate from model prediction failures.

All jobs use `mit_preemptable` on the previously audited `node1412`.
The incumbent fitting implementation is explicitly serial, so its allocation requests one CPU, 20 GB and an eight-hour limit.
The preflight initially waited because all 64 CPUs on that node were allocated; it subsequently completed and released the dependent fit.
These scheduler states are dated observations, not permanent status claims.
The frozen bundle is `logs/uncertainty_bridge_incumbent_prefix600_20260913/`.

## Remaining comparison requirements

The revised subclass's geometry, attachments and other exact observations still need a complete joint probability target before the replacement arm can be fitted.
The [new initial-scene construction](bridge-initial-scene.md) supplies a declared reset law and verified 600-action native continuations, including moving bodies and robot joints.
Those trajectories repeat exactly but still contradict exact recorded outputs, so replay support does not establish a usable conditional target.
The exact-rate reference supplies only one component of that target.
The physical initial-state inventory, supported geometry and remaining continuous exact-output representations must stay explicit.
Use the same fixed program and 600-action prefix for both estimator arms, and separately identify any new transition-discrepancy assumptions or state-inference approximations.
There is no estimator advantage, posterior adequacy or agent-performance result to report from a queued incumbent control.
The incumbent remains the production default while Stage A/B validation continues.
