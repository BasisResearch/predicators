# Boil joint inference on a held-out-prefix split

This follows the [verified full-scene support audit](boil-full-scene-prior.md).
The candidate retains the complete declared unheld-scene prior and the historical program's eight independent original-uniform parameter priors.
It is an offline Stage B experiment; the acting agent still uses the incumbent estimator.

## Fitting and prediction boundary

Only actions 0 through 131 and observations 0 through 132 enter the fitting target.
The remaining 132 recorded actions and observations are reserved for a later causal forecast comparison.
The target context retains only the fitting prefix, including its own immutable data identity.
Fixture proposal guidance uses observations 0 through 64, wholly inside that prefix.
Jug pose and water guidance use observation zero.
No future reading conditions the fitting trajectory, initializes memory or changes the proposal guide.

The physical scene uses 76 unit inputs, including mixture selectors and unused case auxiliaries; eight more unit inputs select the original physical parameter values.
The joint sample retains the eight named parameter values together with the 76 scene auxiliaries, preserving parameter/state dependence.
This dimension describes the computational representation, not 84 independent physical degrees of freedom.

The conditioned base contains the scene prior/proposal correction, geometric support indicator, initial exact-joint conditioning densities and the complete initial output factor exactly once.
The tempered likelihood contains subsequent joint-transition densities and observations 1 through 132.
The unknown geometric normalizer remains common to the parameters for this fixed feature-only program and is not used to claim absolute model evidence.

## Exact physical-history reuse

The frozen program changes only `jug0.water_volume`, `jug0.bubbling_level` and `faucet.spilled_level` in the public predictions.
All three use their original scalar sensor likelihoods; none belongs to a coupled Euler, temporal discrepancy or derived-readout factor.
Every other physical prediction and output factor is independent of the eight program parameters under the previously verified command-free boundary.

A worker caches the native 132-action history by the complete 76-coordinate scene input.
Changing a physical scene coordinate requires a new native replay.
Changing only program parameters reuses that history, restarts the literal program memory, and recomputes all three variable observation channels.
The code checks every unchanged channel and refuses physical commands rather than assuming that every learned program permits this optimization.
The cache is bounded to 32 scenes per worker and does not modify target values or the sampler's random stream.

The target sums the cached physical output/transition factors and newly evaluated scalar factors.
It never subtracts two very large full-history scores to obtain a remaining likelihood.
The initial scalar factor is included in the base, with only later scalar observations included in the parameter-dependent term.
Zero geometric support and exact-event contradictions remain explicit zero density.

## Validation and numerical plan

The preflight reconstructs all sixteen previously audited physical scenes and compares every native fitting-prefix frame with the saved 264-action histories.
For every geometrically feasible scene it evaluates both historical default and independently sampled parameter settings through the literal rules.
It compares the factored target against a direct complete-prefix output likelihood plus all original scene and transition factors.
It also checks unchanged targets across repeated calls, reuse after parameter changes, and serial/parallel agreement.
This is a target and cache validation, not a posterior assessment.

The first preflight, `22689476`, stopped during import before simulation because the historical snapshot lacked the new evaluation module.
The second, `22689537`, exposed the missing causal future-likelihood method in the historical observation module.
Both failures remain recorded; their 13-second and 36-second allocations are setup costs, not failed inference or agent seeds.
The corrected snapshot explicitly pins the evaluation, checkpoint, sampler and observation modules alongside the scene-prior components.

The planned pilots use numerical seeds 410 and 411, 32 particles, 32 cubic-spaced temperatures, eight moves per temperature and a cap of 8,224 target evaluations each.
Thirteen disjoint proposal blocks cover five scene groups and eight individual parameters.
Each block uses the existing 50/50 local/full-range proposal mixture with local scale 0.05.
Complete-stage checkpoints retain the original prior, data, random state, weights, ancestry and cumulative evaluation count on continuation.
Per-allocation counters report native actions, physical replays, cache hits, rule evaluations and worker time separately from sampler evaluations.
Each run requests four compute CPUs and at most eight hours in `mit_preemptable`.

The experiment bundle is `logs/uncertainty_boil_joint_inference_20260913/`.
A passing preflight is required before either posterior pilot starts.
Neither a completed sampler nor agreement on this one recording establishes calibrated uncertainty, acceptable latency or live-agent non-regression.
Both posterior replication and the reserved-action prediction comparison remain required before advancing.

## Initialization-order obstruction and correction

The first native-prefix preflight failed after the target stopped reproducing intermediate initialization operations from the older audit.
Jobs `22689582` and `22689612` record that failure and its detailed reproduction.
Final initial observations and the entire saved scene geometry still match exactly, but contact-rich continuations differ.
Scene 3 first differs in jug rotation at action 26, scene 7 in robot/contact motion at action 61, and scene 13 in switch state and robot motion at action 23.
The two repeatedly supported scenes, 2 and 12, initially match under both tested orders.

Controlled diagnostic `22689667` compares both initialization orders on all five selected scenes and repeats each fresh trajectory.
Restoring the archived intermediate fixture-position resets and observation queries restores exact agreement on every scene.
Both orders repeat exactly independently, so this is deterministic dependence on initialization operations, not evidence of random sensor error.
The experiment identifies the responsible operation sequence; it does not identify a particular internal engine cache.
Final body poses, joints and collision witnesses alone are therefore insufficient to identify this simulator's initial runtime state.

Simply retaining observation-derived intermediate poses would leave an unaccounted path from noisy readings into engine initialization.
The corrected candidate uses a fixed numeric template before applying the sampled scene.
The template retains only conditioned object identities/types/colors and the exact fixed base pose, if supplied.
Controlled joint positions are initially zero, and non-position numeric feature values are zero.
Jug, faucet, faucet-switch, burner and burner-switch template positions are explicitly recorded constants in the new plan.
The complete uncertain scene, including conditioned actual robot joints and all sampled poses/motion, is then applied through the unchanged component map.
No noisy numeric reading selects an intermediate reset pose.

An attempt to omit task-state setup entirely, `22689702`, failed because the native world had not activated its jug state.
The fixed-template approach preserves that required setup while removing the observation-derived numeric anchor.
Diagnostic `22689937` repeats all five selected fixed-template histories exactly, with identical initial observations and scene geometry to the older audit.
Both previously supported scenes retain finite fitting-prefix factors; some subsequent histories differ, as expected from the changed initialization rule.
This is a separately identified generative runtime, not an assertion that the older and corrected likelihoods are equivalent.

The independent archived-history factorization reference, `22689973`, completes all sixteen roots and both parameter settings for each feasible root.
Its maximum finite discrepancy between direct and factored log targets is 1.456e-11.
Three archived scenes have finite 132-action prefix targets; failed exact-event cases retain zero density.
That reference validates arithmetic and literal-rule reuse only, and the fit launcher explicitly refuses to accept it as a native preflight certificate.

Full canonical preflight `22690049` completed in 1:52 on four CPUs.
It passes all sixteen scene checks, exact fresh repeats, unchanged source initial-state/geometry components, direct likelihood factorization, a retained historical trajectory control and serial/parallel target equality.
It also changes every non-conditioned numeric initial reading and checks that the fixed template is unchanged.
The canonical target and planned pilots are isolated in `logs/uncertainty_boil_joint_inference_canonical_20260913/`.
The original experimental bundle and its failed traces remain available as controls.

Three of the sixteen canonical candidates have finite prefix targets, with maximum finite factored-versus-direct log-target error 1.456e-11.
Eight of thirteen geometrically feasible candidates also retain the historical prefix trajectory exactly; the other five retain their explicitly changed canonical-runtime histories.
The corrected initialization remains separate from the old intermediate-reset control even where their predictions agree.

Array `22690118` launches the two gated canonical posterior pilots, numerical seeds 410 and 411.
The launcher verifies the native preflight kind, exact script set and hashes, plan identity, fitting-data identity and output-model identity before sampling.
An archived-history arithmetic certificate cannot satisfy that gate.
The canonical bundle's `verified-inputs.json` records the completed preflight and submitted jobs.
Posterior numerical adequacy and the 132-action reserved suffix remain unevaluated.

Both pilots are running and have written their initialization checkpoints.
Of 32 initial particles, seed 410 has eleven finite targets and seed 411 has seven.
These initialization counts establish that sampling started with support; they do not establish adequate posterior exploration.

The [canonical forecast continuation](boil-canonical-forecasts.md) now verifies fresh whole-history generation, separate future-conditioned density evaluation and literal memory across the fitting boundary.
Its completed-population adapter preserves original checkpoint weights and has passed native and mixture checks.
Full forecast/verifier jobs are queued behind the two fits, with a dependent paired stability report.
No completed-population prediction result is available yet.
