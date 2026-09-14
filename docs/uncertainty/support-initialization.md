# Complete-support initialization for offline inference

September 13, 2026.
The [Bridge joint-target screens](bridge-joint-inference.md) found zero complete-support candidates under the broad proposal and only three and twelve under the local mixture, with early weight concentration in both populations.
The offline sampler now has an optional `initialize_on_support` setting, defaulting to false.
The acting agent remains on the incumbent estimator.

## Joint rejection and retained target

Let `u` denote the complete proposal vector, `h(u)` its reduced unnormalized target density, and `S` its finite positive support.
Initialization draws complete vectors independently from the declared uniform proposal and rejects those outside `S` until the requested population is filled or the evaluation budget is exhausted.
The accepted proposal is the original proposal conditioned on `S`.
Its unknown acceptance probability is a single constant across the whole joint vector and cancels from normalized weights.
The existing target factors therefore remain valid for posterior sampling up to that common constant; this does not support absolute-evidence estimates.

Rejecting only a scene at fixed parameters would instead introduce a parameter-dependent acceptance probability and generally change the target.
This implementation redraws the entire parameter, scene and auxiliary vector together.
Every evaluated rejection consumes the original budget, including partial final batches.
An incomplete initialization emits neither a checkpoint nor a usable posterior.
Impossible support is reported as budget exhaustion for this optional search, not as proof of model inconsistency.

The default initialization path and its historical checkpoint signatures remain unchanged.
Supported initialization has a distinct checkpoint identity and validates complete finite support on recovery.

## Separate Bridge tempering experiment

The native Bridge fixture also changes the intermediate distributions, separately from the initialization method.
For each guided proposal, let `b` be its complete original base log factor, `l` its remaining conditional log factor and `c` its local-mixture log correction.
The endpoint remains `b + l + c`.
The experimental path has zero log base on complete support, negative infinity outside it, and tempered finite score `beta * (b + l + c)`.
At zero temperature its reference is the proposal conditioned on complete support; it is deliberately not the original physical prior.
At temperature one it recovers the same complete reduced target.
Exact constraints remain hard at every temperature, and no event is assigned artificial sensor noise.

This differs from the earlier [guided tempering](guided-tempering.md) experiment, which moved only a proposal correction into the tempered factor.
It addresses concentration already present in Bridge's remaining base terms, without claiming that a balanced initial population guarantees later exploration.

## Validation and pending physical evidence

The analytical reference has density proportional to `x` on `0 < y < x < 1`, with an independent uninformed coordinate.
Joint rejection alone has means `E[x] = 2/3` and `E[y] = 1/3`; the final target has means `3/4` and `3/8`.
Tests exercise both the original finite-factor split and tempering all finite factors, verify moments of the uninformed coordinate, and check exact checkpoint recovery.
Additional cases exhaust the budget with possible or impossible support, using scalar and batched evaluation.

The first compute check passed 35 functional tests and 32 exact default-path/checkpoint comparisons, then failed type checking in a deliberately invalid-input test.
That test's typing is corrected; replacement check `22706421` completed in 2:09 with all 35 functional tests, 32 exact default-path/checkpoint comparisons, type checking, lint and pinned format checks passing.
The dependent native allocations were cancelled before starting and produced no native results.

Native fixture `22706430` completed for numerical seeds 810 and 811, collecting 32 accepted candidates each in 209 and 123 evaluations, respectively, within the 2,048-evaluation limit.
The allocations took 9:12 and 8:12 and executed 35,332 and 28,389 native actions.
It stops at the completed initialization checkpoint, retaining all rejected evaluations and reporting the weight concentration implied by several temperatures.
Independent readers `22706433` and `22706435` both completed in 1:02, verifying the rejection ledger, proposal inverses and mixture corrections, first-32-candidate parity with the original screen, and a fresh native target repetition per population.
Both readers reject all four corrupted-ledger controls, with maximum inverse-coordinate discrepancy 1.222e-15.
At temperature 0.0001, effective sample sizes are 30.06 and 31.03; at 0.001 they fall to 8.37 and 6.91, and direct weighting to temperature one concentrates on one candidate in each population.
The artificial starting distributions now contain supported, balanced populations, but gradual weighting and rejuvenation are still necessary.
These are completed initialization checks, not posterior fits, predictive acceptance or agent results.
Paired full fitting pilots `22706932` are now submitted with the unchanged complete target, 64 geometrically spaced temperatures, eight blocked moves per stage and a 20,000-evaluation budget per run.

The frozen native inputs and outputs are in `logs/uncertainty_bridge_supported_initialization_20260913/`.
The separate reader is in `logs/uncertainty_bridge_supported_verification_20260913/`.
